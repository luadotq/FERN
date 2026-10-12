import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, List, Optional
from ..network import FERNModel


@dataclass
class DLPConfig:
    lr: float = 0.05
    beta: float = 0.85
    lr_pla: float = 0.08


class DLPTrainer:
    def __init__(self, model: FERNModel, lr: float = 0.05, beta: float = 0.85):
        self.m = model
        self.lr = lr
        self.beta = beta
        self.is_pla = getattr(model.config, "attn_type", "vla") == "pla"
        
        cfg = model.config
        v = cfg.vocab_size
        d = cfg.d_model
        nl = cfg.num_layers
        
        dev = next(model.parameters()).device
        self.heads = [torch.randn(v, d, device=dev) * 0.02 for _ in range(nl)]
        
        # Momentum buffers for PLA
        if self.is_pla:
            self.mom_d = [torch.zeros_like(b.w_down.weight) for b in model.blocks]
            self.mom_u = [torch.zeros_like(b.w_up.weight) for b in model.blocks]
            self.mom_o = [torch.zeros_like(b.vla.w_out.weight) for b in model.blocks]
            self.mom_v = [torch.zeros_like(b.vla.w_v.weight) for b in model.blocks]
            self.mom_h = [torch.zeros_like(hd) for hd in self.heads]

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
        bt = x.numel()
        v = self.m.config.vocab_size
        d = self.m.config.d_model
        
        if self.is_pla:
            return self._step_pla(x, y, bt, v, d)
        return self._step_vla(x, y, bt, v, d)

    def _step_vla(self, x: torch.Tensor, y: torch.Tensor, bt: int, v: int, d: int) -> Tuple[float, float, float]:
        last_l = 0.0
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
                vla_out, _ = b.vla.forward_parallel(norm_x)
                h_mid = h + vla_out
                norm_h = F.rms_norm(h_mid, (d,), eps=1e-5)
                
                mlp_in = b.w_up(norm_h)
                mlp_act = b.act(mlp_in)
                mlp_out = b.w_down(mlp_act)
                h_next = h_mid + mlp_out

                h_flat = h_next.reshape(bt, d)
                logits = h_flat @ self.heads[i].t()
                
                if i == len(self.m.blocks) - 1:
                    last_l = F.cross_entropy(logits[val_idx], y_val).item()

                prob = F.softmax(logits, dim=-1)
                prob_val = prob[val_idx]
                h_val = h_flat[val_idx]

                prob_h = prob_val @ self.heads[i]
                delta = torch.zeros_like(h_flat)
                delta[val_idx] = self.heads[i][y_val] - prob_h
                delta = F.rms_norm(delta, (d,), eps=1e-5)

                gh = - (prob_val.t() @ h_val)
                gh.index_add_(0, y_val, h_val)
                gh = gh / n_val
                self.heads[i] += self.lr * gh / (torch.norm(gh) + 1e-6) * (v * d)**0.5 * 0.05

                gd = (delta.t() @ mlp_act.reshape(bt, -1)) / bt
                b.w_down.weight += self.lr * gd / (torch.norm(gd) + 1e-6) * b.w_down.weight.numel()**0.5 * 0.01

                delta_in = (delta @ b.w_down.weight) * torch.sigmoid(1.702 * mlp_in.reshape(bt, -1))
                gu = (delta_in.t() @ norm_h.reshape(bt, -1)) / bt
                b.w_up.weight += self.lr * gu / (torch.norm(gu) + 1e-6) * b.w_up.weight.numel()**0.5 * 0.01

                go = (delta.t() @ vla_out.reshape(bt, -1)) / bt
                b.vla.w_out.weight += self.lr * go / (torch.norm(go) + 1e-6) * b.vla.w_out.weight.numel()**0.5 * 0.01

                delta_v = delta @ b.vla.w_out.weight
                gv = (delta_v.t() @ norm_x.reshape(bt, -1)) / bt
                b.vla.w_v.weight += self.lr * gv / (torch.norm(gv) + 1e-6) * b.vla.w_v.weight.numel()**0.5 * 0.01

                h = h_next

        return last_l, last_l, 0.0

    def _step_pla(self, x: torch.Tensor, y: torch.Tensor, bt: int, v: int, d: int) -> Tuple[float, float, float]:
        lr = 0.08
        beta = self.beta
        last_l = 0.0
        
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
                h_next, vf = res[0], res[2]
                
                h_norm = F.rms_norm(h_next, (d,), eps=1e-5)
                h_flat = h_norm.reshape(bt, d)
                logits = h_flat @ self.heads[i].t()
                
                if i == len(self.m.blocks) - 1:
                    last_l = F.cross_entropy(logits[val_idx], y_val).item()

                prob = F.softmax(logits, dim=-1)
                prob_val = prob[val_idx]
                h_val = h_flat[val_idx]

                prob_h = prob_val @ self.heads[i]
                delta = torch.zeros_like(h_flat)
                delta[val_idx] = self.heads[i][y_val] - prob_h
                delta = F.rms_norm(delta, (d,), eps=1e-5)

                gh = - (prob_val.t() @ h_val)
                gh.index_add_(0, y_val, h_val)
                gh = gh / n_val
                self.mom_h[i] = beta * self.mom_h[i] + (1 - beta) * (gh / (torch.norm(gh) + 1e-6))
                self.heads[i] += lr * self.mom_h[i] * (v * d)**0.5 * 0.08

                xx2 = b.ln2(h)
                dx = torch.cat([torch.zeros_like(xx2[:, :1]), xx2[:, :-1] - xx2[:, 1:]], dim=1)
                k = xx2 + dx * b.x_k
                norm_k = F.rms_norm(k, (d,), eps=1e-5)
                
                mlp_in = b.w_up(norm_k)
                mlp_act = torch.relu(mlp_in) ** 2
                # Clamped derivative avoids quadratic explosion
                d_act = torch.clamp(2.0 * torch.relu(mlp_in.reshape(bt, -1)), max=2.0)

                gd = (delta.t() @ mlp_act.reshape(bt, -1)) / bt
                norm_gd = gd / (torch.norm(gd) + 1e-6)
                self.mom_d[i] = beta * self.mom_d[i] + (1 - beta) * norm_gd
                b.w_down.weight += lr * self.mom_d[i] * b.w_down.weight.numel()**0.5 * 0.02

                delta_in = F.rms_norm((delta @ b.w_down.weight) * d_act, (b.w_up.weight.shape[0],), eps=1e-5)
                gu = (delta_in.t() @ norm_k.reshape(bt, -1)) / bt
                norm_gu = gu / (torch.norm(gu) + 1e-6)
                self.mom_u[i] = beta * self.mom_u[i] + (1 - beta) * norm_gu
                b.w_up.weight += lr * self.mom_u[i] * b.w_up.weight.numel()**0.5 * 0.02

                norm_xx1 = F.rms_norm(b.ln1(h), (d,), eps=1e-5)
                go = (delta.t() @ norm_xx1.reshape(bt, -1)) / bt
                norm_go = go / (torch.norm(go) + 1e-6)
                self.mom_o[i] = beta * self.mom_o[i] + (1 - beta) * norm_go
                b.vla.w_out.weight += lr * self.mom_o[i] * b.vla.w_out.weight.numel()**0.5 * 0.02

                delta_att = F.rms_norm(delta @ b.vla.w_out.weight, (d,), eps=1e-5)
                gv = (delta_att.t() @ norm_xx1.reshape(bt, -1)) / bt
                norm_gv = gv / (torch.norm(gv) + 1e-6)
                self.mom_v[i] = beta * self.mom_v[i] + (1 - beta) * norm_gv
                b.vla.w_v.weight += lr * self.mom_v[i] * b.vla.w_v.weight.numel()**0.5 * 0.02

                h = h_next

        return last_l, last_l, 0.0
