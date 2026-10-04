import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_vla_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr, Gamma_ptr,
        b_stride, h_stride, s_stride, d_stride,
        out_b_stride, out_h_stride, out_s_stride, out_d_stride,
        scale,
        seq_len,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        block_m_id = tl.program_id(2)

        gamma = tl.load(Gamma_ptr + head_id)

        m_offs = block_m_id * BLOCK_M + tl.arange(0, BLOCK_M)
        d_offs = tl.arange(0, HEAD_DIM)

        # Base pointers for batch and head
        q_base = Q_ptr + batch_id * b_stride + head_id * h_stride
        k_base = K_ptr + batch_id * b_stride + head_id * h_stride
        v_base = V_ptr + batch_id * b_stride + head_id * h_stride
        out_base = Out_ptr + batch_id * out_b_stride + head_id * out_h_stride

        # Load Q block [BLOCK_M, HEAD_DIM]
        q_ptrs = q_base + (m_offs[:, None] * s_stride + d_offs[None, :] * d_stride)
        q_mask = m_offs[:, None] < seq_len
        q = tl.load(q_ptrs, mask=q_mask, other=0.0) * scale

        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        max_n = tl.minimum((block_m_id + 1) * BLOCK_M, seq_len)
        
        # Iterate over key-value blocks
        for block_n_start in range(0, max_n, BLOCK_N):
            n_offs = block_n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs[None, :] < seq_len

            k_ptrs = k_base + (n_offs[None, :] * s_stride + d_offs[:, None] * d_stride)
            k = tl.load(k_ptrs, mask=n_mask, other=0.0)

            v_ptrs = v_base + (n_offs[:, None] * s_stride + d_offs[None, :] * d_stride)
            v = tl.load(v_ptrs, mask=n_offs[:, None] < seq_len, other=0.0)

            # [BLOCK_M, BLOCK_N]
            scores = tl.dot(q, k)

            # Causal and exponential decay
            diff = m_offs[:, None] - n_offs[None, :]
            causal_mask = (diff >= 0) & (m_offs[:, None] < seq_len) & (n_offs[None, :] < seq_len)
            
            # Decay calculation: gamma ** diff = exp(diff * log(gamma))
            decay = tl.exp(diff * tl.log(gamma))
            scores = tl.where(causal_mask, scores * decay, 0.0)

            acc += tl.dot(scores.to(v.dtype), v)

        out_ptrs = out_base + (m_offs[:, None] * out_s_stride + d_offs[None, :] * out_d_stride)
        tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=q_mask)


def chunkwise_vla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gammas: torch.Tensor,
    scale: Optional[float] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    b, h, s, d = q.shape
    scale = scale or (1.0 / math.sqrt(d))
    device, dtype = q.device, q.dtype

    if s <= chunk_size:
        return pytorch_vla_reference(q, k, v, gammas, scale)

    C = chunk_size
    pad_len = (C - (s % C)) % C
    if pad_len > 0:
        q_p = F.pad(q, (0, 0, 0, pad_len))
        k_p = F.pad(k, (0, 0, 0, pad_len))
        v_p = F.pad(v, (0, 0, 0, pad_len))
    else:
        q_p, k_p, v_p = q, k, v

    s_p = s + pad_len
    nc = s_p // C

    q_c = q_p.view(b, h, nc, C, d)
    k_c = k_p.view(b, h, nc, C, d)
    v_c = v_p.view(b, h, nc, C, d)

    steps = torch.arange(C, device=device, dtype=torch.float32)
    diff = steps.unsqueeze(1) - steps.unsqueeze(0)
    causal = diff >= 0
    intra_decay = torch.where(
        causal.view(1, 1, 1, C, C),
        gammas.view(1, h, 1, 1, 1) ** diff.view(1, 1, 1, C, C),
        torch.zeros(1, 1, 1, C, C, device=device, dtype=torch.float32)
    ).to(dtype)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) * scale
    o_intra = torch.matmul(scores * intra_decay, v_c)

    decay_k = (gammas.view(1, h, 1, 1, 1) ** (C - 1 - steps).view(1, 1, 1, C, 1)).to(dtype)
    k_decayed = k_c * decay_k
    kv_chunk = torch.matmul(k_decayed.transpose(-2, -1), v_c)

    decay_chunk = (gammas.view(1, h, 1, 1, 1) ** C).to(dtype)
    decay_q = ((gammas.view(1, h, 1, 1, 1) ** (steps + 1).view(1, 1, 1, C, 1)) * scale).to(dtype)
    q_decayed = q_c * decay_q

    states = []
    curr_state = torch.zeros(b, h, 1, d, d, device=device, dtype=dtype)
    for i in range(nc):
        states.append(curr_state)
        curr_state = curr_state * decay_chunk + kv_chunk[:, :, i : i + 1]
    states = torch.cat(states, dim=2)

    o_inter = torch.matmul(q_decayed, states)
    out = (o_intra + o_inter).view(b, h, s_p, d)
    if pad_len > 0:
        out = out[:, :, :s, :]
    return out


def triton_vla_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gammas: torch.Tensor,
    scale: Optional[float] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    b, h, s, d = q.shape
    scale = scale or (1.0 / math.sqrt(d))

    if HAS_TRITON and q.is_cuda and k.is_cuda and v.is_cuda and not (q.requires_grad or k.requires_grad or v.requires_grad):
        out = torch.empty_like(v)
        BLOCK_M = 32 if s >= 32 else 16
        BLOCK_N = 32 if s >= 32 else 16
        grid = (b, h, triton.cdiv(s, BLOCK_M))

        _fused_vla_fwd_kernel[grid](
            q, k, v, out, gammas.contiguous(),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scale,
            s,
            HEAD_DIM=d,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return out
    else:
        return chunkwise_vla(q, k, v, gammas, scale=scale, chunk_size=chunk_size)


def pytorch_vla_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gammas: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    b, h, s, d = q.shape
    scale = scale or (1.0 / math.sqrt(d))
    device, dtype = q.device, q.dtype

    steps = torch.arange(s, device=device, dtype=torch.float32)
    diff = steps.unsqueeze(1) - steps.unsqueeze(0)
    causal = diff >= 0

    gamma_exp = gammas.view(1, h, 1, 1) ** diff.view(1, 1, s, s)
    decay_mask = torch.where(causal.view(1, 1, s, s), gamma_exp, torch.zeros((), device=device, dtype=torch.float32)).to(dtype)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores_decayed = scores * decay_mask
    return torch.matmul(scores_decayed, v)
