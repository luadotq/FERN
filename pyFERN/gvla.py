import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import rms_norm
from .kernels.gvla_kernel import gvla_step, triton_gvla_forward


class GammaVectorLinearAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mem_dim: int,
        num_heads: int = 16,
        rms_eps: float = 1e-5,
        chunk_size: int = 64,
        conv_kernel: int = 4,
        d_state: Optional[int] = None,
        ngroups: int = 1,
        qk_norm: bool = False,
        scale: Optional[float] = 1.0,
    ):
        super().__init__()
        self.num_heads = max(num_heads, 1)
        self.head_dim = mem_dim // self.num_heads
        self.mem_dim = mem_dim
        self.d_state = d_state if d_state is not None else self.head_dim
        self.ngroups = ngroups if d_state is not None else self.num_heads
        self.qk_dim = self.ngroups * self.d_state
        self.heads_per_group = self.num_heads // self.ngroups
        self.rms_eps = rms_eps
        self.chunk_size = chunk_size
        self.conv_kernel = conv_kernel
        self.qk_norm = qk_norm
        self.scale = scale if scale is not None else (1.0 / math.sqrt(self.d_state))

        self.w_q = nn.Linear(in_dim, self.qk_dim, bias=False)
        self.w_k = nn.Linear(in_dim, self.qk_dim, bias=False)
        self.w_v = nn.Linear(in_dim, mem_dim, bias=False)
        self.w_gate = nn.Linear(in_dim, mem_dim, bias=False)
        self.w_dt = nn.Linear(in_dim, self.num_heads, bias=True)
        self.w_out = nn.Linear(mem_dim, in_dim, bias=False)

        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.ln_x = nn.Parameter(torch.ones(mem_dim))

        if conv_kernel > 1:
            c_tot = mem_dim + 2 * self.qk_dim
            self.conv1d = nn.Conv1d(
                c_tot, c_tot, kernel_size=conv_kernel, groups=c_tot, padding=0, bias=True
            )
        else:
            self.conv1d = None

    def _conv_parallel(self, xbc: torch.Tensor, conv_st: Optional[torch.Tensor]):
        k_sz = self.conv_kernel
        b, c, t = xbc.shape
        if conv_st is None:
            conv_st = torch.zeros((b, c, k_sz - 1), device=xbc.device, dtype=xbc.dtype)
        inp = torch.cat([conv_st, xbc], dim=-1)
        new_conv = inp[:, :, -(k_sz - 1):].detach()
        return F.silu(self.conv1d(inp)), new_conv

    def _conv_step(self, xbc_t: torch.Tensor, conv_st: Optional[torch.Tensor]):
        k_sz = self.conv_kernel
        b, c = xbc_t.shape
        if conv_st is None:
            conv_st = torch.zeros((b, c, k_sz - 1), device=xbc_t.device, dtype=xbc_t.dtype)
        inp = torch.cat([conv_st, xbc_t.unsqueeze(-1)], dim=-1)
        new_conv = inp[:, :, 1:].detach()
        out = (inp * self.conv1d.weight.squeeze(1).unsqueeze(0)).sum(dim=-1)
        if self.conv1d.bias is not None:
            out = out + self.conv1d.bias
        return F.silu(out), new_conv

    def _expand_qk(self, t_qk: torch.Tensor) -> torch.Tensor:
        # [B, ..., ngroups * d_state] -> [B, ..., num_heads, d_state]
        shape = t_qk.shape[:-1]
        t_qk = t_qk.view(*shape, self.ngroups, self.d_state)
        if self.heads_per_group > 1:
            t_qk = t_qk.repeat_interleave(self.heads_per_group, dim=-2)
        return t_qk

    def forward_parallel(
        self,
        c: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
        return_state: bool = False,
    ):
        b, t, _ = c.shape
        h, d = self.num_heads, self.head_dim
        s_att, conv_st = state if state is not None else (None, None)

        q_raw = self.w_q(c)
        k_raw = self.w_k(c)
        v_raw = self.w_v(c)

        new_conv = None
        if self.conv1d is not None:
            xbc = torch.cat([v_raw, k_raw, q_raw], dim=-1).transpose(1, 2)
            xbc, new_conv = self._conv_parallel(xbc, conv_st)
            v_raw, k_raw, q_raw = xbc.transpose(1, 2).split([self.mem_dim, self.qk_dim, self.qk_dim], dim=-1)

        q = rms_norm(q_raw, self.rms_eps) if self.qk_norm else q_raw
        k = rms_norm(k_raw, self.rms_eps) if self.qk_norm else k_raw

        dt = F.softplus(self.w_dt(c))
        log_g = (-torch.exp(self.A_log.float()) * dt.float()).transpose(1, 2)

        q = self._expand_qk(q).transpose(1, 2)
        k = self._expand_qk(k).transpose(1, 2)
        v = v_raw.view(b, t, h, d).transpose(1, 2)
        v_scaled = v * dt.transpose(1, 2).to(v.dtype).unsqueeze(-1)

        o_heads, new_s = triton_gvla_forward(
            q, k, v_scaled, log_g, s=s_att, chunk_size=self.chunk_size, scale=self.scale
        )
        o_heads = o_heads + self.D.to(o_heads.dtype).view(1, h, 1, 1) * v
        o_flat = o_heads.transpose(1, 2).contiguous().view(b, t, h * d)

        gate = F.silu(self.w_gate(c))
        o_norm = rms_norm(o_flat * gate, self.rms_eps) * self.ln_x.to(o_flat.dtype)
        o_out = self.w_out(o_norm)
        fe = ((rms_norm(v_raw, self.rms_eps) - rms_norm(o_flat, self.rms_eps)) ** 2).mean()

        if return_state:
            return o_out, fe, (new_s.detach(), new_conv)
        return o_out, fe

    def forward_step(
        self,
        c_t: torch.Tensor,
        state: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    ):
        b = c_t.shape[0]
        h, d = self.num_heads, self.head_dim
        s_prev, conv_st = state if state is not None else (None, None)
        if s_prev is None:
            s_prev = torch.zeros((b, h, self.d_state, d), device=c_t.device, dtype=c_t.dtype)

        q_raw = self.w_q(c_t)
        k_raw = self.w_k(c_t)
        v_raw = self.w_v(c_t)

        new_conv = None
        if self.conv1d is not None:
            xbc_t = torch.cat([v_raw, k_raw, q_raw], dim=-1)
            xbc_t, new_conv = self._conv_step(xbc_t, conv_st)
            v_raw, k_raw, q_raw = xbc_t.split([self.mem_dim, self.qk_dim, self.qk_dim], dim=-1)

        q = rms_norm(q_raw, self.rms_eps) if self.qk_norm else q_raw
        k = rms_norm(k_raw, self.rms_eps) if self.qk_norm else k_raw

        dt = F.softplus(self.w_dt(c_t))
        log_g = -torch.exp(self.A_log.float()) * dt.float()

        q_h = self._expand_qk(q)
        k_h = self._expand_qk(k)
        v_h = v_raw.view(b, h, d)
        v_scaled = v_h * dt.to(v_h.dtype).unsqueeze(-1)

        o_h, new_s = gvla_step(q_h, k_h, v_scaled, log_g, s_prev, scale=self.scale)
        o_h = o_h + self.D.to(o_h.dtype).view(1, h, 1) * v_h
        o_flat = o_h.view(b, h * d)

        gate = F.silu(self.w_gate(c_t))
        o_norm = rms_norm(o_flat * gate, self.rms_eps) * self.ln_x.to(o_flat.dtype)
        o_out = self.w_out(o_norm)
        fe = ((rms_norm(v_raw, self.rms_eps) - rms_norm(o_flat, self.rms_eps)) ** 2).mean()
        return o_out, (new_s, new_conv), fe
