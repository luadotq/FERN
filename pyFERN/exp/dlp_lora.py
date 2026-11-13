import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from ..network import FERNModel
from ..lora import LoRALinear


class DLPLoRATrainer:
    def __init__(self, model: FERNModel, lr: float = 0.05, beta: float = 0.85):
        self.m = model
        self.lr = lr
        self.beta = beta
        self.is_pla = getattr(model.config, "attn_type", "vla") == "pla"
        
        cfg = model.config
        v, d, nl = cfg.vocab_size, cfg.d_model, cfg.num_layers
        dev = next(model.parameters()).device
        
        # Frozen projection head matching model.decoder.weight
        if hasattr(model, "decoder") and hasattr(model.decoder, "weight"):
            self.head = model.decoder.weight.detach()
        else:
            self.head = torch.randn(v, d, device=dev) * 0.02
        
        # Momentum buffers for LoRA adapters & forward hooks for exact runtime inputs
        self.moms: Dict[str, torch.Tensor] = {}
        self.inps: Dict[str, torch.Tensor] = {}
        for name, mod in model.named_modules():
            if isinstance(mod, LoRALinear):
                self.moms[f"{name}_a"] = torch.zeros_like(mod.lora_a)
                self.moms[f"{name}_b"] = torch.zeros_like(mod.lora_b)
                mod.register_forward_hook(lambda m, inp, out, k=name: self.inps.__setitem__(k, inp[0].detach()))

    def _upd_lora(self, layer: LoRALinear, inp: torch.Tensor, delta: torch.Tensor, key: str, lr: float):
        bt = inp.shape[0]
        sc = layer.scaling
        
        ni = F.rms_norm(inp, (layer.base.in_features,), eps=1e-5)
        h = ni @ layer.lora_a.t()
        nh = F.rms_norm(h, (layer.r,), eps=1e-5)
        nd = F.rms_norm(delta, (layer.base.out_features,), eps=1e-5)
        
        # B update
        gb = (nd.t() @ nh) / bt * sc
        ngb = gb / (torch.norm(gb) + 1e-6)
        mb = self.moms[f"{key}_b"]
        mb.copy_(self.beta * mb + (1.0 - self.beta) * ngb)
        layer.lora_b.data.add_(lr * mb * (layer.lora_b.numel()**0.5) * 0.02)
        
        # A update
        dh = (nd @ layer.lora_b) * sc
        ndh = F.rms_norm(dh, (layer.r,), eps=1e-5)
        ga = (ndh.t() @ ni) / bt
        nga = ga / (torch.norm(ga) + 1e-6)
        ma = self.moms[f"{key}_a"]
        ma.copy_(self.beta * ma + (1.0 - self.beta) * nga)
        layer.lora_a.data.add_(lr * ma * (layer.lora_a.numel()**0.5) * 0.02)

    def train_step(self, x: torch.Tensor, y: torch.Tensor, win_size: Optional[int] = None) -> Tuple[float, float, float]:
        if x.shape[1] < 2:
            return 0.0, 0.0, 0.0
        x, y = x[:, :-1], y[:, 1:]

        if win_size and x.shape[1] > win_size:
            chunks = []
            for start in range(0, x.shape[1], win_size):
                end = min(x.shape[1], start + win_size)
                xw, yw = x[:, start:end], y[:, start:end]
                nv = int((yw != -100).sum().item())
                if nv > 0:
                    chunks.append((xw, yw, nv))
            if not chunks:
                return 0.0, 0.0, 0.0

            lr_scale = 1.0 / max(1.0, len(chunks) ** 0.5)
            tot_l, tot_ce, tot_fe, n_toks = 0.0, 0.0, 0.0, 0
            for xw, yw, nv in chunks:
                l, ce, fe = self._train_chunk(xw, yw, lr_scale=lr_scale)
                tot_l += l * nv
                tot_ce += ce * nv
                tot_fe += fe * nv
                n_toks += nv
            return tot_l / n_toks, tot_ce / n_toks, tot_fe / n_toks
        return self._train_chunk(x, y)

    def _train_chunk(self, x: torch.Tensor, y: torch.Tensor, lr_scale: float = 1.0) -> Tuple[float, float, float]:
        bt = x.numel()
        v = self.m.config.vocab_size
        d = self.m.config.d_model
        
        if self.is_pla:
            return self._step_pla(x, y, bt, v, d, lr_scale=lr_scale)
        return self._step_vla(x, y, bt, v, d, lr_scale=lr_scale)

    def _step_vla(self, x: torch.Tensor, y: torch.Tensor, bt: int, v: int, d: int, lr_scale: float = 1.0) -> Tuple[float, float, float]:
        last_l = 0.0
        tot_fe = 0.0
        lr = self.lr * lr_scale
        y_flat = y.reshape(-1)
        mask = (y_flat != -100)
        n_val = int(mask.sum().item())
        if n_val == 0:
            return 0.0, 0.0, 0.0
        val_idx = torch.nonzero(mask).squeeze(-1)
        y_val = y_flat[val_idx]

        with torch.no_grad():
            h = self.m.encoder(x)
            for i, b in enumerate(self.m.blocks):
                norm_x = F.rms_norm(h, (d,), eps=1e-5)
                vla_out, fe_l = b.vla.forward_parallel(norm_x)
                tot_fe += float(fe_l)
                h_mid = h + vla_out
                norm_h = F.rms_norm(h_mid, (d,), eps=1e-5)
                
                mlp_in = b.w_up(norm_h)
                mlp_act = b.act(mlp_in)
                mlp_out = b.w_down(mlp_act)
                h_next = h_mid + mlp_out
                
                h_flat = h_next.reshape(bt, d)
                logits = h_flat @ self.head.t()
                
                if i == len(self.m.blocks) - 1:
                    last_l = F.cross_entropy(logits[val_idx], y_val).item()
                
                prob = F.softmax(logits, dim=-1)
                prob_val = prob[val_idx]
                h_val = h_flat[val_idx]
                
                prob_h = prob_val @ self.head
                delta = torch.zeros_like(h_flat)
                delta[val_idx] = self.head[y_val] - prob_h
                delta = F.rms_norm(delta, (d,), eps=1e-5)
                
                k_out = f"blocks.{i}.vla.w_out"
                if k_out in self.inps and isinstance(b.vla.w_out, LoRALinear):
                    self._upd_lora(b.vla.w_out, self.inps[k_out].reshape(bt, -1), delta, k_out, lr)
                    
                k_down = f"blocks.{i}.w_down"
                if k_down in self.inps and isinstance(b.w_down, LoRALinear):
                    self._upd_lora(b.w_down, self.inps[k_down].reshape(bt, -1), delta, k_down, lr)
                
                h = h_next
        return last_l, last_l, tot_fe / len(self.m.blocks)

    def _step_pla(self, x: torch.Tensor, y: torch.Tensor, bt: int, v: int, d: int, lr_scale: float = 1.0) -> Tuple[float, float, float]:
        last_l = 0.0
        tot_fe = 0.0
        lr = self.lr * lr_scale
        y_flat = y.reshape(-1)
        mask = (y_flat != -100)
        n_val = int(mask.sum().item())
        if n_val == 0:
            return 0.0, 0.0, 0.0
        val_idx = torch.nonzero(mask).squeeze(-1)
        y_val = y_flat[val_idx]

        with torch.no_grad():
            h = self.m.encoder(x)
            vf = None
            for i, b in enumerate(self.m.blocks):
                res = b.forward_parallel(h, vf)
                h_next, fe_l, vf = res[0], res[1], res[2]
                tot_fe += float(fe_l)
                
                h_flat = F.rms_norm(h_next, (d,), eps=1e-5).reshape(bt, d)
                logits = h_flat @ self.head.t()
                
                if i == len(self.m.blocks) - 1:
                    last_l = F.cross_entropy(logits[val_idx], y_val).item()
                    
                prob = F.softmax(logits, dim=-1)
                prob_val = prob[val_idx]
                h_val = h_flat[val_idx]
                
                # O(1) error projection without dense one-hot
                prob_h = prob_val @ self.head
                delta = torch.zeros_like(h_flat)
                delta[val_idx] = self.head[y_val] - prob_h
                delta = F.rms_norm(delta, (d,), eps=1e-5)
                
                # Direct projection updates with exact runtime activations
                k_out = f"blocks.{i}.vla.w_out"
                if k_out in self.inps and isinstance(b.vla.w_out, LoRALinear):
                    self._upd_lora(b.vla.w_out, self.inps[k_out].reshape(bt, -1), delta, k_out, lr)
                    
                k_down = f"blocks.{i}.w_down"
                if k_down in self.inps and isinstance(b.w_down, LoRALinear):
                    self._upd_lora(b.w_down, self.inps[k_down].reshape(bt, -1), delta, k_down, lr * 0.1)
                    
                h = h_next
        return last_l, last_l, tot_fe / len(self.m.blocks)
