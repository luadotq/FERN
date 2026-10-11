import torch
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _fused_pla_fwd_kernel(
        r_ptr,
        w_ptr,
        k_ptr,
        v_ptr,
        u_ptr,
        z_ptr,
        out_ptr,
        s_ptr,
        stride_b,
        stride_t,
        stride_h,
        stride_n,
        stride_sb,
        stride_st,
        stride_sh,
        stride_sn1,
        stride_sn2,
        T,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        offset_base = pid_b * stride_b + pid_h * stride_h
        cols = tl.arange(0, BLOCK_N)
        rows = tl.arange(0, BLOCK_N)
        mask = cols < N
        mask_2d = (rows[:, None] < N) & (cols[None, :] < N)

        p_r = r_ptr + offset_base + cols
        p_w = w_ptr + offset_base + cols
        p_k = k_ptr + offset_base + cols
        p_v = v_ptr + offset_base + cols
        p_u = u_ptr + offset_base + cols
        p_z = z_ptr + offset_base + cols
        p_out = out_ptr + offset_base + cols

        offset_s_base = pid_b * stride_sb + pid_h * stride_sh + rows[:, None] * stride_sn1 + cols[None, :] * stride_sn2
        p_s = s_ptr + offset_s_base

        b_h = tl.zeros([BLOCK_N, BLOCK_N], dtype=tl.float32)

        for _ in range(0, T):
            b_r = tl.load(p_r, mask=mask, other=0.0).to(tl.float32)
            b_w = tl.load(p_w, mask=mask, other=0.0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask, other=0.0).to(tl.float32)
            b_u = tl.load(p_u, mask=mask, other=0.0).to(tl.float32)
            b_z = tl.load(p_z, mask=mask, other=0.0).to(tl.float32)

            # sa = S @ u
            b_sa = tl.sum(b_h * b_u[None, :], axis=1)

            # S = S * w + v k^T + sa z^T
            b_h = b_h * b_w[None, :] + b_v[:, None] * b_k[None, :] + b_sa[:, None] * b_z[None, :]

            # out = S @ r
            b_out = tl.sum(b_h * b_r[None, :], axis=1)
            tl.store(p_out, b_out.to(out_ptr.dtype.element_ty), mask=mask)

            # Save state for backward
            tl.store(p_s, b_h.to(s_ptr.dtype.element_ty), mask=mask_2d)

            p_r += stride_t
            p_w += stride_t
            p_k += stride_t
            p_v += stride_t
            p_u += stride_t
            p_z += stride_t
            p_out += stride_t
            p_s += stride_st

    @triton.jit
    def _fused_pla_bwd_kernel(
        r_ptr,
        w_ptr,
        k_ptr,
        v_ptr,
        u_ptr,
        z_ptr,
        dout_ptr,
        s_ptr,
        dr_ptr,
        dw_ptr,
        dk_ptr,
        dv_ptr,
        du_ptr,
        dz_ptr,
        stride_b,
        stride_t,
        stride_h,
        stride_n,
        stride_sb,
        stride_st,
        stride_sh,
        stride_sn1,
        stride_sn2,
        T,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        offset_base = pid_b * stride_b + pid_h * stride_h
        cols = tl.arange(0, BLOCK_N)
        rows = tl.arange(0, BLOCK_N)
        mask = cols < N
        mask_2d = (rows[:, None] < N) & (cols[None, :] < N)

        t_last = T - 1
        offset_last = offset_base + t_last * stride_t

        p_r = r_ptr + offset_last + cols
        p_w = w_ptr + offset_last + cols
        p_k = k_ptr + offset_last + cols
        p_v = v_ptr + offset_last + cols
        p_u = u_ptr + offset_last + cols
        p_z = z_ptr + offset_last + cols
        p_dout = dout_ptr + offset_last + cols

        p_dr = dr_ptr + offset_last + cols
        p_dw = dw_ptr + offset_last + cols
        p_dk = dk_ptr + offset_last + cols
        p_dv = dv_ptr + offset_last + cols
        p_du = du_ptr + offset_last + cols
        p_dz = dz_ptr + offset_last + cols

        offset_s_base = pid_b * stride_sb + pid_h * stride_sh + rows[:, None] * stride_sn1 + cols[None, :] * stride_sn2
        p_s = s_ptr + offset_s_base + t_last * stride_st

        b_ds = tl.zeros([BLOCK_N, BLOCK_N], dtype=tl.float32)

        for step_idx in range(T - 1, -1, -1):
            b_dout = tl.load(p_dout, mask=mask, other=0.0).to(tl.float32)
            b_r = tl.load(p_r, mask=mask, other=0.0).to(tl.float32)
            b_w = tl.load(p_w, mask=mask, other=0.0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask, other=0.0).to(tl.float32)
            b_u = tl.load(p_u, mask=mask, other=0.0).to(tl.float32)
            b_z = tl.load(p_z, mask=mask, other=0.0).to(tl.float32)

            b_s_curr = tl.load(p_s, mask=mask_2d, other=0.0).to(tl.float32)

            # 1. dr = S^T @ dout
            b_dr = tl.sum(b_s_curr * b_dout[:, None], axis=0)
            tl.store(p_dr, b_dr.to(dr_ptr.dtype.element_ty), mask=mask)

            # 2. ds += dout @ r^T
            b_ds += b_dout[:, None] * b_r[None, :]

            # 3. dv = ds @ k
            b_dv = tl.sum(b_ds * b_k[None, :], axis=1)
            tl.store(p_dv, b_dv.to(dv_ptr.dtype.element_ty), mask=mask)

            # 4. dk = ds^T @ v
            b_dk = tl.sum(b_ds * b_v[:, None], axis=0)
            tl.store(p_dk, b_dk.to(dk_ptr.dtype.element_ty), mask=mask)

            # 5. S_prev
            b_s_prev = tl.load(p_s - stride_st, mask=mask_2d & (step_idx > 0), other=0.0).to(tl.float32)

            # sa = S_prev @ u
            b_sa = tl.sum(b_s_prev * b_u[None, :], axis=1)

            # 6. dz = ds^T @ sa
            b_dz = tl.sum(b_ds * b_sa[:, None], axis=0)
            tl.store(p_dz, b_dz.to(dz_ptr.dtype.element_ty), mask=mask)

            # 7. d_sa = ds @ z
            b_dsa = tl.sum(b_ds * b_z[None, :], axis=1)

            # 8. du = S_prev^T @ d_sa
            b_du = tl.sum(b_s_prev * b_dsa[:, None], axis=0)
            tl.store(p_du, b_du.to(du_ptr.dtype.element_ty), mask=mask)

            # 9. dw = sum_rows(ds * S_prev)
            b_dw = tl.sum(b_ds * b_s_prev, axis=0)
            tl.store(p_dw, b_dw.to(dw_ptr.dtype.element_ty), mask=mask)

            # 10. ds = ds * w + d_sa @ u^T
            b_ds = b_ds * b_w[None, :] + b_dsa[:, None] * b_u[None, :]

            p_r -= stride_t
            p_w -= stride_t
            p_k -= stride_t
            p_v -= stride_t
            p_u -= stride_t
            p_z -= stride_t
            p_dout -= stride_t

            p_dr -= stride_t
            p_dw -= stride_t
            p_dk -= stride_t
            p_dv -= stride_t
            p_du -= stride_t
            p_dz -= stride_t

            p_s -= stride_st


    class TritonPLAFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, w, k, v, a, kk):
            b, t, h, n = r.shape
            r, w, k, v, a, kk = [x.contiguous() for x in (r, w, k, v, a, kk)]
            u = (-kk).contiguous()
            z = (kk * a).contiguous()

            out = torch.empty_like(r)
            s = torch.empty((b, t, h, n, n), device=r.device, dtype=r.dtype)

            block_n = triton.next_power_of_2(n)
            grid = (b, h)

            _fused_pla_fwd_kernel[grid](
                r, w, k, v, u, z, out, s,
                r.stride(0), r.stride(1), r.stride(2), r.stride(3),
                s.stride(0), s.stride(1), s.stride(2), s.stride(3), s.stride(4),
                t, N=n, BLOCK_N=block_n,
            )

            ctx.save_for_backward(r, w, k, v, a, kk, s)
            final_s = s[:, -1].to(r.dtype)
            return out.view(b, t, h * n), final_s

        @staticmethod
        def backward(ctx, dout, ds_final):
            r, w, k, v, a, kk, s = ctx.saved_tensors
            b, t, h, n = r.shape
            dout = dout.contiguous().view(b, t, h, n)
            u = (-kk).contiguous()
            z = (kk * a).contiguous()

            dr = torch.empty_like(r)
            dw = torch.empty_like(w)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            du = torch.empty_like(u)
            dz = torch.empty_like(z)

            block_n = triton.next_power_of_2(n)
            grid = (b, h)

            _fused_pla_bwd_kernel[grid](
                r, w, k, v, u, z, dout, s,
                dr, dw, dk, dv, du, dz,
                r.stride(0), r.stride(1), r.stride(2), r.stride(3),
                s.stride(0), s.stride(1), s.stride(2), s.stride(3), s.stride(4),
                t, N=n, BLOCK_N=block_n,
            )

            dkk = -du + dz * a
            da = dz * kk
            return dr, dw, dk, dv, da, dkk
else:
    TritonPLAFunction = None


def triton_pla_forward(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    kk: torch.Tensor,
    r_k: Optional[torch.Tensor] = None,
    s: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    with_bonus: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if TRITON_AVAILABLE and r.is_cuda and s is None:
        out, final_s = TritonPLAFunction.apply(r, w, k, v, a, kk)
        if with_bonus and r_k is not None:
            bonus = (r * k * r_k.to(r.dtype).view(1, 1, r.shape[2], r.shape[3])).sum(dim=-1, keepdim=True) * v
            out = out + bonus.view(r.shape[0], r.shape[1], r.shape[2] * r.shape[3])
        return out, final_s
    from .pla_kernel import chunkwise_pla
    return chunkwise_pla(r, w, k, v, a, kk, r_k=r_k, s=s, chunk_size=chunk_size, with_bonus=with_bonus)

