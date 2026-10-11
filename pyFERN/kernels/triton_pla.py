from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from .pla_kernel import chunkwise_pla, pla_recurrent

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_pla_fwd_kernel(
        R_ptr, W_ptr, K_ptr, V_ptr, A_ptr, KK_ptr, RK_ptr, Out_ptr,
        b_stride, t_stride, h_stride, d_stride,
        out_b_stride, out_t_stride, out_d_stride,
        seq_len,
        HEAD_DIM: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)

        rk_offs = tl.arange(0, HEAD_DIM)
        r_k = tl.load(RK_ptr + head_id * HEAD_DIM + rk_offs)

        s = tl.zeros([HEAD_DIM, HEAD_DIM], dtype=tl.float32)
        offs_d = tl.arange(0, HEAD_DIM)

        for t in range(0, seq_len):
            base_offset = batch_id * b_stride + t * t_stride + head_id * h_stride

            r = tl.load(R_ptr + base_offset + offs_d * d_stride)
            w = tl.load(W_ptr + base_offset + offs_d * d_stride)
            k = tl.load(K_ptr + base_offset + offs_d * d_stride)
            v = tl.load(V_ptr + base_offset + offs_d * d_stride)
            a = tl.load(A_ptr + base_offset + offs_d * d_stride)
            kk = tl.load(KK_ptr + base_offset + offs_d * d_stride)

            sa = tl.sum(s * (-kk)[None, :], axis=1)
            pred = sa[:, None] * (kk * a)[None, :]
            kv = v[:, None] * k[None, :]

            s = s * w[None, :] + kv + pred
            out_i = tl.sum(s * r[None, :], axis=1)

            bonus_val = tl.sum(r * k * r_k)
            out_final = out_i + bonus_val * v

            out_offset = batch_id * out_b_stride + t * out_t_stride + (head_id * HEAD_DIM + offs_d) * out_d_stride
            tl.store(Out_ptr + out_offset, out_final.to(Out_ptr.dtype.element_ty))


def triton_pla_forward(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    kk: torch.Tensor,
    r_k: torch.Tensor,
    s: Optional[torch.Tensor] = None,
    chunk_size: int = 32,
    with_bonus: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    b, t, h, n = r.shape
    if (
        HAS_TRITON
        and r.is_cuda
        and n <= 64
        and s is None
        and with_bonus
        and not (r.requires_grad or w.requires_grad or k.requires_grad or v.requires_grad)
    ):
        out = torch.empty((b, t, h * n), device=r.device, dtype=r.dtype)
        grid = (b, h)
        _fused_pla_fwd_kernel[grid](
            r, w, k, v, a, kk, r_k.contiguous(), out,
            r.stride(0), r.stride(1), r.stride(2), r.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            t,
            HEAD_DIM=n,
        )
        return out, None

    return chunkwise_pla(
        r, w, k, v, a, kk, r_k=r_k, s=s, chunk_size=chunk_size, with_bonus=with_bonus
    )
