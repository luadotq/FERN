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

    def train_step(self, x: torch.Tensor, y: torch.Tensor, win_size: Optional[int] = None, err_scale: float = 1.0) -> Tuple[float, float, float]:
        if x.shape[1] < 2:
            return 0.0, 0.0, 0.0
        x, y = x[:, :-1], y[:, 1:]

        if win_size and x.shape[1] > win_size:
            chunks = []
            for start in range(0, x.shape[1], win_size):
                end = min(x.shape[1], start + win_size)
                xw, yw = x[:, start:end], y[:, start:end]
                nv = int((yw != -100).sum().item())
                chunks.append((xw, yw, nv))
            tot_nv = sum(c[2] for c in chunks)
            if tot_nv == 0:
                return 0.0, 0.0, 0.0

            lr_scale = 1.0 / max(1.0, sum(1 for c in chunks if c[2] > 0) ** 0.5)
            states = [None] * len(self.m.blocks) if self.is_pla else None
            tot_l, tot_ce, tot_fe, n_toks = 0.0, 0.0, 0.0, 0
            for xw, yw, nv in chunks:
                l, ce, fe, states = self._train_chunk(xw, yw, lr_scale=lr_scale, err_scale=err_scale, states=states)
                if nv > 0:
                    tot_l += l * nv
                    tot_ce += ce * nv
                    tot_fe += fe * nv
                    n_toks += nv
            return tot_l / n_toks, tot_ce / n_toks, tot_fe / n_toks
        l, ce, fe, _ = self._train_chunk(x, y, err_scale=err_scale)
        return l, ce, fe

    def _train_chunk(self, x: torch.Tensor, y: torch.Tensor, lr_scale: float = 1.0, err_scale: float = 1.0, states=None):
        bt = x.numel()
        v = self.m.config.vocab_size
        d = self.m.config.d_model

        if self.is_pla:
            return self._step_pla(x, y, bt, v, d, lr_scale=lr_scale, err_scale=err_scale, states=states)
        l, ce, fe = self._step_vla(x, y, bt, v, d, lr_scale=lr_scale, err_scale=err_scale)
        return l, ce, fe, None

    def _step_vla(self, x: torch.Tensor, y: torch.Tensor, bt: int, v: int, d: int, lr_scale: float = 1.0, err_scale: float = 1.0) -> Tuple[float, float, float]:
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

                prob_h = prob_val @ self.head
                delta = torch.zeros_like(h_flat)
                delta[val_idx] = (self.head[y_val] - prob_h) * err_scale
                delta = F.rms_norm(delta, (d,), eps=1e-5)

                k_out = f"blocks.{i}.vla.w_out"
                if k_out in self.inps and isinstance(b.vla.w_out, LoRALinear):
                    self._upd_lora(b.vla.w_out, self.inps[k_out].reshape(bt, -1), delta, k_out, lr)

                k_down = f"blocks.{i}.w_down"
                if k_down in self.inps and isinstance(b.w_down, LoRALinear):
                    self._upd_lora(b.w_down, self.inps[k_down].reshape(bt, -1), delta, k_down, lr)

                h = h_next
        return last_l, last_l, tot_fe / len(self.m.blocks)

    def _step_pla(self, x: torch.Tensor, y: torch.Tensor, bt: int, v: int, d: int, lr_scale: float = 1.0, err_scale: float = 1.0, states=None):
        last_l = 0.0
        tot_fe = 0.0
        lr = self.lr * lr_scale
        y_flat = y.reshape(-1)
        mask = (y_flat != -100)
        n_val = int(mask.sum().item())
        val_idx = torch.nonzero(mask).squeeze(-1) if n_val > 0 else None
        y_val = y_flat[val_idx] if n_val > 0 else None
        new_states = [] if states is not None else None

        with torch.no_grad():
            h = self.m.encoder(x)
            vf = None
            for i, b in enumerate(self.m.blocks):
                st_i = states[i] if states is not None else None
                if states is not None:
                    h_next, fe_l, vf, st_out = b.forward_parallel(h, vf, state=st_i, return_state=True)
                    new_states.append(st_out)
                else:
                    res = b.forward_parallel(h, vf)
                    h_next, fe_l, vf = res[0], res[1], res[2]
                tot_fe += float(fe_l)

                if n_val > 0:
                    h_norm = self.m.ln_out(h_next) if hasattr(self.m, "ln_out") else F.rms_norm(h_next, (d,), eps=1e-5)
                    h_flat = h_norm.reshape(bt, d)
                    logits = h_flat @ self.head.t()

                    if i == len(self.m.blocks) - 1:
                        last_l = F.cross_entropy(logits[val_idx], y_val).item()

                    prob = F.softmax(logits, dim=-1)
                    prob_val = prob[val_idx]

                    # O(1) error projection without dense one-hot
                    prob_h = prob_val @ self.head
                    delta = torch.zeros_like(h_flat)
                    delta[val_idx] = (self.head[y_val] - prob_h) * err_scale
                    delta = F.rms_norm(delta, (d,), eps=1e-5)

                    # Direct projection updates with exact runtime activations
                    k_out = f"blocks.{i}.vla.w_out"
                    if k_out in self.inps and isinstance(b.vla.w_out, LoRALinear):
                        self._upd_lora(b.vla.w_out, self.inps[k_out].reshape(bt, -1), delta, k_out, lr)

                    k_down = f"blocks.{i}.w_down"
                    if k_down in self.inps and isinstance(b.w_down, LoRALinear):
                        self._upd_lora(b.w_down, self.inps[k_down].reshape(bt, -1), delta, k_down, lr * 0.1)

                h = h_next
        return last_l, last_l, tot_fe / len(self.m.blocks), new_states

    def logp_seq(self, x: torch.Tensor, y: torch.Tensor, win_size: int = 512) -> Tuple[float, int]:
        if x.shape[1] < 2:
            return 0.0, 0
        x, y = x[:, :-1], y[:, 1:]
        d = self.m.config.d_model
        states = [None] * len(self.m.blocks) if self.is_pla else None
        tot_lp, tot_n = 0.0, 0

        with torch.no_grad():
            for start in range(0, x.shape[1], win_size):
                end = min(x.shape[1], start + win_size)
                xw, yw = x[:, start:end], y[:, start:end]
                h = self.m.encoder(xw)
                vf = None
                for i, b in enumerate(self.m.blocks):
                    if self.is_pla:
                        h, _, vf, st_out = b.forward_parallel(h, vf, state=states[i], return_state=True)
                        states[i] = st_out
                    else:
                        h, _ = b.forward_parallel(h)
                y_flat = yw.reshape(-1)
                mask = (y_flat != -100)
                nv = int(mask.sum().item())
                if nv > 0:
                    val_idx = torch.nonzero(mask).squeeze(-1)
                    h_norm = self.m.ln_out(h) if hasattr(self.m, "ln_out") else F.rms_norm(h, (d,), eps=1e-5)
                    logits = h_norm.reshape(-1, d)[val_idx] @ self.head.t()
                    tot_lp -= F.cross_entropy(logits, y_flat[val_idx], reduction="sum").item()
                    tot_n += nv
        return tot_lp, tot_n

    def train_dpo_step(
        self,
        x_w: torch.Tensor,
        y_w: torch.Tensor,
        x_l: torch.Tensor,
        y_l: torch.Tensor,
        ref_w: Optional[float] = None,
        ref_l: Optional[float] = None,
        beta: float = 0.1,
        gamma: float = 0.5,
        sft_w: float = 0.1,
        win_size: int = 512,
    ) -> Dict[str, float]:
        lp_w, nw = self.logp_seq(x_w, y_w, win_size=win_size)
        lp_l, nl = self.logp_seq(x_l, y_l, win_size=win_size)
        if nw == 0 or nl == 0:
            return {"loss": 0.0, "w_dpo": 0.0, "margin": 0.0}

        if ref_w is not None and ref_l is not None:
            u = beta * ((lp_w - ref_w) - (lp_l - ref_l))
        else:
            # SimPO (reference-free, length-normalized)
            u = beta * ((lp_w / nw) - (lp_l / nl)) - gamma

        w_dpo = float(torch.sigmoid(torch.tensor(-u)).item())
        dpo_loss = float(F.softplus(torch.tensor(-u)).item())

        self.train_step(x_w, y_w, win_size=win_size, err_scale=w_dpo + sft_w)
        self.train_step(x_l, y_l, win_size=win_size, err_scale=-w_dpo)
        return {"loss": dpo_loss, "w_dpo": w_dpo, "margin": float(u)}
