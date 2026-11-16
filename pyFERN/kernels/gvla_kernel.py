import math
from typing import Tuple, Optional
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def gvla_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_g: torch.Tensor,
    s: torch.Tensor,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, H, Dk], v: [B, H, Dv], log_g: [B, H], s: [B, H, Dk, Dv]
    dk = q.shape[-1]
    sc = scale if scale is not None else (1.0 / math.sqrt(dk))
    dt = q.dtype
    g = torch.exp(log_g.float()).to(dt).unsqueeze(-1).unsqueeze(-1)
    s_new = s.to(dt) * g + k.unsqueeze(-1) * v.unsqueeze(-2)
    out = torch.matmul((q * sc).unsqueeze(-2), s_new).squeeze(-2)
    return out, s_new


def gvla_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_g: torch.Tensor,
    s: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, H, T, Dk], v: [B, H, T, Dv], log_g: [B, H, T]
    b, h, t, dk = q.shape
    dv = v.shape[-1]
    sc = scale if scale is not None else (1.0 / math.sqrt(dk))
    dt = q.dtype
    state = torch.zeros((b, h, dk, dv), device=q.device, dtype=dt) if s is None else s.to(dt).clone()
    outs = []
    for i in range(t):
        out_i, state = gvla_step(q[:, :, i], k[:, :, i], v[:, :, i], log_g[:, :, i], state, scale=sc)
        outs.append(out_i)
    return torch.stack(outs, dim=2), state


def chunkwise_gvla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_g: torch.Tensor,
    s: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, H, T, Dk], v: [B, H, T, Dv], log_g: [B, H, T] (<= 0)
    b, h, t, dk = q.shape
    dv = v.shape[-1]
    sc = scale if scale is not None else (1.0 / math.sqrt(dk))
    dt = q.dtype
    dev = q.device

    if t <= chunk_size and s is None:
        return gvla_recurrent(q, k, v, log_g, s=s, scale=sc)

    c = chunk_size
    pad = (c - (t % c)) % c
    if pad > 0:
        q_p = torch.nn.functional.pad(q, (0, 0, 0, pad))
        k_p = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v_p = torch.nn.functional.pad(v, (0, 0, 0, pad))
        lg_p = torch.nn.functional.pad(log_g, (0, pad))
    else:
        q_p, k_p, v_p, lg_p = q, k, v, log_g

    tp = t + pad
    nc = tp // c

    qc = q_p.view(b, h, nc, c, dk)
    kc = k_p.view(b, h, nc, c, dk)
    vc = v_p.view(b, h, nc, c, dv)
    lgc = lg_p.view(b, h, nc, c).float()

    cs = torch.cumsum(lgc, dim=-1)
    causal = torch.tril(torch.ones(c, c, device=dev, dtype=torch.bool))
    diff = (cs.unsqueeze(-1) - cs.unsqueeze(-2)).masked_fill(~causal, -1e9)
    decay_m = torch.exp(diff).to(dt)

    scores = torch.matmul(qc, kc.transpose(-2, -1)) * sc
    o_intra = torch.matmul(scores * decay_m, vc)

    # Decay from token j to chunk end
    k_dec = kc * torch.exp(cs[..., -1:] - cs).to(dt).unsqueeze(-1)
    kv_chunk = torch.matmul(k_dec.transpose(-2, -1), vc)

    g_chunk = torch.exp(cs[..., -1]).to(dt)
    q_dec = qc * torch.exp(cs).to(dt).unsqueeze(-1) * sc

    curr = torch.zeros((b, h, dk, dv), device=dev, dtype=dt) if s is None else s.to(dt)
    states = []
    for i in range(nc):
        states.append(curr)
        curr = curr * g_chunk[:, :, i].unsqueeze(-1).unsqueeze(-1) + kv_chunk[:, :, i]

    states_t = torch.stack(states, dim=2)
    o_inter = torch.matmul(q_dec, states_t)
    out = (o_intra + o_inter).view(b, h, tp, dv)
    if pad > 0:
        out = out[:, :, :t, :]
    return out, curr


if HAS_TRITON:
    @triton.jit
    def _gvla_intra_kernel(
        Q, K, V, CS, Out,
        sq_b, sq_h, sq_c, sq_t, sq_d,
        so_b, so_h, so_c, so_t, so_d,
        scs_b, scs_h, scs_c, scs_t,
        scale,
        H: tl.constexpr,
        C: tl.constexpr,
        D: tl.constexpr,
    ):
        bh = tl.program_id(0)
        cid = tl.program_id(1)
        b = bh // H
        h = bh % H

        offs_t = tl.arange(0, C)
        offs_d = tl.arange(0, D)

        q_ptr = Q + b * sq_b + h * sq_h + cid * sq_c + offs_t[:, None] * sq_t + offs_d[None, :] * sq_d
        k_ptr = K + b * sq_b + h * sq_h + cid * sq_c + offs_t[:, None] * sq_t + offs_d[None, :] * sq_d
        v_ptr = V + b * sq_b + h * sq_h + cid * sq_c + offs_t[:, None] * sq_t + offs_d[None, :] * sq_d
        cs_ptr = CS + b * scs_b + h * scs_h + cid * scs_c + offs_t * scs_t

        q = tl.load(q_ptr)
        k = tl.load(k_ptr)
        v = tl.load(v_ptr)
        cs = tl.load(cs_ptr).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        mask = offs_t[:, None] >= offs_t[None, :]
        decay = tl.exp(tl.where(mask, cs[:, None] - cs[None, :], -1e9))
        att = (qk * tl.where(mask, decay, 0.0)).to(v.dtype)
        o = tl.dot(att, v)

        o_ptr = Out + b * so_b + h * so_h + cid * so_c + offs_t[:, None] * so_t + offs_d[None, :] * so_d
        tl.store(o_ptr, o)


def triton_gvla_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_g: torch.Tensor,
    s: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, h, t, d = q.shape
    dv = v.shape[-1]
    sc = scale if scale is not None else (1.0 / math.sqrt(d))
    if (
        HAS_TRITON
        and q.is_cuda
        and not (q.requires_grad or k.requires_grad or v.requires_grad or log_g.requires_grad)
        and d == dv
        and d in (32, 64, 128)
        and chunk_size in (32, 64)
        and t >= chunk_size
    ):
        dt = q.dtype
        c = chunk_size
        pad = (c - (t % c)) % c
        qp = torch.nn.functional.pad(q, (0, 0, 0, pad)) if pad else q
        kp = torch.nn.functional.pad(k, (0, 0, 0, pad)) if pad else k
        vp = torch.nn.functional.pad(v, (0, 0, 0, pad)) if pad else v
        lgp = torch.nn.functional.pad(log_g, (0, pad)) if pad else log_g
        tp = t + pad
        nc = tp // c

        qc = qp.view(b, h, nc, c, d).contiguous()
        kc = kp.view(b, h, nc, c, d).contiguous()
        vc = vp.view(b, h, nc, c, d).contiguous()
        cs = torch.cumsum(lgp.view(b, h, nc, c).float(), dim=-1).contiguous()
        o_intra = torch.empty_like(qc)

        _gvla_intra_kernel[(b * h, nc)](
            qc, kc, vc, cs, o_intra,
            qc.stride(0), qc.stride(1), qc.stride(2), qc.stride(3), qc.stride(4),
            o_intra.stride(0), o_intra.stride(1), o_intra.stride(2), o_intra.stride(3), o_intra.stride(4),
            cs.stride(0), cs.stride(1), cs.stride(2), cs.stride(3),
            sc, H=h, C=c, D=d,
        )

        k_dec = kc * torch.exp(cs[..., -1:] - cs).to(dt).unsqueeze(-1)
        kv_chunk = torch.matmul(k_dec.transpose(-2, -1), vc)
        g_chunk = torch.exp(cs[..., -1]).to(dt)
        q_dec = qc * torch.exp(cs).to(dt).unsqueeze(-1) * sc

        curr = torch.zeros((b, h, d, d), device=q.device, dtype=dt) if s is None else s.to(dt)
        states = []
        for i in range(nc):
            states.append(curr)
            curr = curr * g_chunk[:, :, i].unsqueeze(-1).unsqueeze(-1) + kv_chunk[:, :, i]

        o_inter = torch.matmul(q_dec, torch.stack(states, dim=2))
        out = (o_intra + o_inter).view(b, h, tp, d)
        return (out[:, :, :t, :] if pad else out), curr

    return chunkwise_gvla(q, k, v, log_g, s=s, chunk_size=chunk_size, scale=sc)
