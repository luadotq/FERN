import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
from .layers import rms_norm
from .kernels import triton_vla_forward, chunkwise_vla

class VectorLinearAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mem_dim: int,
        num_heads: int = 4,
        rms_eps: float = 1e-5,
        gamma_min: float = 0.90,
        gamma_max: float = 0.98,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.num_heads = max(num_heads, 1)
        self.head_dim = mem_dim // self.num_heads
        self.mem_dim = mem_dim
        self.rms_eps = rms_eps
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.chunk_size = chunk_size

        self.w_q = nn.Linear(in_dim, mem_dim)
        self.w_k = nn.Linear(in_dim, mem_dim)
        self.w_v = nn.Linear(in_dim, mem_dim)
        self.w_gate = nn.Linear(in_dim, mem_dim)
        self.w_out = nn.Linear(mem_dim, mem_dim)

    def get_head_gammas(self, device, dtype=torch.float32) -> torch.Tensor:
        h = self.num_heads
        if h <= 1:
            return torch.tensor([self.gamma_min], device=device, dtype=dtype)
        return torch.linspace(self.gamma_min, self.gamma_max, steps=h, device=device, dtype=dtype)

    def forward_parallel(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, _ = c.shape
        device = c.device

        q_raw = self.w_q(c)
        k_raw = self.w_k(c)
        v_raw = self.w_v(c)

        q = rms_norm(q_raw, self.rms_eps)
        k = rms_norm(k_raw, self.rms_eps)

        h, d = self.num_heads, self.head_dim
        q = q.view(b, s, h, d).transpose(1, 2)
        k = k.view(b, s, h, d).transpose(1, 2)
        v = v_raw.view(b, s, h, d).transpose(1, 2)

        gammas = self.get_head_gammas(device=device)
        scale = 1.0 / math.sqrt(d)

        # Chunkwise decayed attention
        o_heads = triton_vla_forward(q, k, v, gammas, scale=scale, chunk_size=self.chunk_size)
        o_flat = o_heads.transpose(1, 2).contiguous().view(b, s, h * d)
        o_norm = rms_norm(o_flat, self.rms_eps)
        o_out = self.w_out(o_norm)

        v_norm = rms_norm(v_raw, self.rms_eps)
        v_pred = rms_norm(o_flat, self.rms_eps)
        e_vla = v_norm - v_pred
        fe_vla = (e_vla ** 2).mean()

        return o_out, fe_vla

    def forward_step(self, c_t: torch.Tensor, s_prev: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = c_t.shape[0]
        if s_prev is None:
            s_prev = torch.zeros(b, self.num_heads, self.head_dim, self.head_dim, device=c_t.device, dtype=c_t.dtype)
        q_raw = self.w_q(c_t)
        k_raw = self.w_k(c_t)
        v_raw = self.w_v(c_t)
        g_raw = torch.sigmoid(self.w_gate(c_t))

        q = rms_norm(q_raw, self.rms_eps)
        k = rms_norm(k_raw, self.rms_eps)

        h, d = self.num_heads, self.head_dim
        scale = 1.0 / math.sqrt(d)
        gammas = self.get_head_gammas(c_t.device).view(1, h, 1, 1)

        q_h = q.view(b, h, 1, d)
        k_h = k.view(b, h, d, 1)
        v_h = v_raw.view(b, h, 1, d)

        s_decayed = s_prev * gammas
        kv_outer = torch.matmul(k_h, v_h)
        s_new = s_decayed + kv_outer

        o_h = torch.matmul(q_h, s_new).squeeze(-2) * scale
        o_flat = o_h.view(b, h * d)
        o_norm = rms_norm(o_flat, self.rms_eps)
        o_out = self.w_out(o_norm)

        v_pred = rms_norm(torch.matmul(q_h, s_prev).squeeze(-2).view(b, h * d) * scale, self.rms_eps)
        e_vla = rms_norm(v_raw, self.rms_eps) - v_pred
        fe_vla = (e_vla ** 2).mean()

        return o_out, s_new, fe_vla
