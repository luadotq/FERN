from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kernels.pla_kernel import pla_step


class PredictiveLinearAttention(nn.Module):
    def __init__(self, layer_id: int, d_model: int, num_heads: int, head_dim: int):
        super().__init__()
        self.layer_id = layer_id
        self.h = num_heads
        self.n = head_dim
        self.x_r = nn.Parameter(torch.empty(1, 1, d_model))
        self.x_w = nn.Parameter(torch.empty(1, 1, d_model))
        self.x_k = nn.Parameter(torch.empty(1, 1, d_model))
        self.x_v = nn.Parameter(torch.empty(1, 1, d_model))
        self.x_a = nn.Parameter(torch.empty(1, 1, d_model))
        self.x_g = nn.Parameter(torch.empty(1, 1, d_model))
        self.w0 = nn.Parameter(torch.empty(1, 1, d_model))
        self.w1 = nn.Parameter(torch.empty(d_model, 64))
        self.w2 = nn.Parameter(torch.empty(64, d_model))
        self.a0 = nn.Parameter(torch.empty(1, 1, d_model))
        self.a1 = nn.Parameter(torch.empty(d_model, 64))
        self.a2 = nn.Parameter(torch.empty(64, d_model))
        self.v0 = nn.Parameter(torch.empty(1, 1, d_model))
        self.v1 = nn.Parameter(torch.empty(d_model, 32))
        self.v2 = nn.Parameter(torch.empty(32, d_model))
        self.g1 = nn.Parameter(torch.empty(d_model, 128))
        self.g2 = nn.Parameter(torch.empty(128, d_model))
        self.k_k = nn.Parameter(torch.empty(1, 1, d_model))
        self.k_a = nn.Parameter(torch.empty(1, 1, d_model))
        self.r_k = nn.Parameter(torch.empty(num_heads, head_dim))

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
        w = torch.exp(-0.606531 * torch.sigmoid(w)).view(h, n)

        out, state = pla_step(r, w, k, v, a, kk, state)
        out = F.group_norm(out.view(1, h * n), num_groups=h, weight=self.ln_x.weight, bias=self.ln_x.bias, eps=64e-5).view(h * n)
        out = out + ((r * k * self.r_k).sum(dim=-1, keepdim=True) * v).view(h * n)
        return self.w_out(out * g), x, state, v_first, 0.0
