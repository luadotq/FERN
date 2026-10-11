from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kernels.pla_kernel import pla_step, chunkwise_pla


class PredictiveLinearAttention(nn.Module):
    def __init__(self, layer_id: int, d_model: int, num_heads: int, head_dim: int):
        super().__init__()
        self.layer_id = layer_id
        self.h = num_heads
        self.n = head_dim
        self.x_r = nn.Parameter(torch.zeros(1, 1, d_model))
        self.x_w = nn.Parameter(torch.zeros(1, 1, d_model))
        self.x_k = nn.Parameter(torch.zeros(1, 1, d_model))
        self.x_v = nn.Parameter(torch.zeros(1, 1, d_model))
        self.x_a = nn.Parameter(torch.zeros(1, 1, d_model))
        self.x_g = nn.Parameter(torch.zeros(1, 1, d_model))
        self.w0 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.w1 = nn.Parameter(torch.randn(d_model, 64) * 0.02)
        self.w2 = nn.Parameter(torch.zeros(64, d_model))
        self.a0 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.a1 = nn.Parameter(torch.randn(d_model, 64) * 0.02)
        self.a2 = nn.Parameter(torch.zeros(64, d_model))
        self.v0 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.v1 = nn.Parameter(torch.randn(d_model, 32) * 0.02)
        self.v2 = nn.Parameter(torch.zeros(32, d_model))
        self.g1 = nn.Parameter(torch.randn(d_model, 128) * 0.02)
        self.g2 = nn.Parameter(torch.zeros(128, d_model))
        self.k_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.k_a = nn.Parameter(torch.ones(1, 1, d_model))
        self.r_k = nn.Parameter(torch.zeros(num_heads, head_dim))

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
        self.ln_x = nn.LayerNorm(d_model)

    def forward_step(self, x: torch.Tensor, x_prev: torch.Tensor, v_first: torch.Tensor, state: torch.Tensor):
        h, n = self.h, self.n
        dx = x_prev - x
        xr = x + dx * self.x_r.squeeze()
        xw = x + dx * self.x_w.squeeze()
        xk = x + dx * self.x_k.squeeze()
        xv = x + dx * self.x_v.squeeze()
        xa = x + dx * self.x_a.squeeze()
        xg = x + dx * self.x_g.squeeze()

        r = self.w_q(xr).view(h, n)
        w = torch.tanh(xw @ self.w1) @ self.w2
        k = self.w_k(xk)
        v = self.w_v(xv)
        a = torch.sigmoid(self.a0.squeeze() + (xa @ self.a1) @ self.a2).view(h, n)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k.squeeze()
        kk = F.normalize(kk.view(h, n), dim=-1, p=2.0)
        k = (k * (1 + (a.view(-1) - 1) * self.k_a.squeeze())).view(h, n)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0.squeeze() + (xv @ self.v1) @ self.v2)
        v = v.view(h, n)

        w = self.w0.squeeze().float() + w.float()
        w = torch.exp(-0.606531 * torch.sigmoid(w)).to(dtype=r.dtype).view(h, n)

        out, state = pla_step(r, w, k, v, a, kk, state)
        normed = F.group_norm(out.view(1, h * n), num_groups=h, weight=self.ln_x.weight.to(dtype=out.dtype), bias=self.ln_x.bias.to(dtype=out.dtype) if self.ln_x.bias is not None else None, eps=64e-5).view(h * n)
        bonus = ((r * k * self.r_k).sum(dim=-1, keepdim=True) * v).view(h * n)
        fe_pla = ((v.view(h * n) - normed) ** 2).mean()
        return self.w_out((normed + bonus) * g), x, state, v_first, fe_pla

    def forward_parallel(self, x: torch.Tensor, v_first: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
        b, t, d = x.shape
        h, n = self.h, self.n
        dx = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1] - x[:, 1:]], dim=1)
        xr = x + dx * self.x_r
        xw = x + dx * self.x_w
        xk = x + dx * self.x_k
        xv = x + dx * self.x_v
        xa = x + dx * self.x_a
        xg = x + dx * self.x_g

        r = self.w_q(xr).view(b, t, h, n)
        w = torch.tanh(xw @ self.w1) @ self.w2
        w = self.w0.float() + w.float()
        w = torch.exp(-0.606531 * torch.sigmoid(w)).to(dtype=r.dtype).view(b, t, h, n)

        k = self.w_k(xk)
        v = self.w_v(xv)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2).view(b, t, h, n)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(b, t, h, n), dim=-1, p=2.0)
        k = (k * (1 + (a.view(b, t, -1) - 1) * self.k_a)).view(b, t, h, n)

        if self.layer_id == 0:
            v_first = v
        elif v_first is not None:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        v = v.view(b, t, h, n)

        raw, _ = chunkwise_pla(r, w, k, v, a, kk, chunk_size=64, with_bonus=False)
        normed = F.group_norm(raw.view(b * t, h * n), num_groups=h, weight=self.ln_x.weight.to(dtype=raw.dtype), bias=self.ln_x.bias.to(dtype=raw.dtype) if self.ln_x.bias is not None else None, eps=64e-5).view(b, t, h * n)
        bonus = ((r * k * self.r_k.view(h, n)).sum(dim=-1, keepdim=True) * v).view(b, t, h * n)
        out = self.w_out((normed + bonus) * g)
        fe_pla = ((v.view(b, t, h * n) - normed) ** 2).mean()
        return out, fe_pla, v_first
