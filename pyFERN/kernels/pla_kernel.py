from typing import Tuple, Optional
import torch


def pla_step(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    kk: torch.Tensor,
    s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dt = r.dtype
    w, k, v, a, kk, s = w.to(dt), k.to(dt), v.to(dt), a.to(dt), kk.to(dt), s.to(dt)
    h, n = r.shape[0], r.shape[1]
    sa = s @ (-kk).view(h, n, 1)
    e = v.view(h, n, 1) @ k.view(h, 1, n) + sa @ (kk * a).view(h, 1, n)
    s_new = s * w.view(h, 1, n) + e
    out = (s_new @ r.view(h, n, 1)).view(h * n)
    return out, s_new


def pla_recurrent(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    kk: torch.Tensor,
    r_k: Optional[torch.Tensor] = None,
    s: Optional[torch.Tensor] = None,
    with_bonus: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, t, h, n = r.shape
    dt = r.dtype
    w, k, v, a, kk = w.to(dt), k.to(dt), v.to(dt), a.to(dt), kk.to(dt)
    if s is None:
        s = torch.zeros((b, h, n, n), device=r.device, dtype=dt)
    else:
        s = s.to(dt)

    outs = []
    r_k_flat = r_k.to(dt).view(h, n) if (r_k is not None and with_bonus) else None
    for i in range(t):
        r_i = r[:, i]
        w_i = w[:, i]
        k_i = k[:, i]
        v_i = v[:, i]
        a_i = a[:, i]
        kk_i = kk[:, i]

        sa = torch.matmul(s, (-kk_i).unsqueeze(-1))
        kv = torch.matmul(v_i.unsqueeze(-1), k_i.unsqueeze(-2))
        pred = torch.matmul(sa, (kk_i * a_i).unsqueeze(-2))
        s = s * w_i.unsqueeze(-2) + kv + pred

        out_i = torch.matmul(s, r_i.unsqueeze(-1)).squeeze(-1)
        if r_k_flat is not None:
            bonus_i = (r_i * k_i * r_k_flat).sum(dim=-1, keepdim=True) * v_i
            out_i = out_i + bonus_i
        outs.append(out_i.view(b, h * n))

    return torch.stack(outs, dim=1), s


class FastPLAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, kk, s_init, chunk_size=64):
        b, t, h, n = r.shape
        dt = r.dtype
        dev = r.device
        u = (-kk).float()
        z = (kk * a).float()
        rf, wf, kf, vf = r.float(), w.float(), k.float(), v.float()

        s = s_init.float().clone() if s_init is not None else torch.zeros(b, h, n, n, device=dev, dtype=torch.float32)
        outs = []
        s_ckpts = [s]

        for i in range(t):
            r_i, w_i, k_i, v_i, u_i, z_i = rf[:, i], wf[:, i], kf[:, i], vf[:, i], u[:, i], z[:, i]
            sa = torch.matmul(s, u_i.unsqueeze(-1))
            kv = torch.matmul(v_i.unsqueeze(-1), k_i.unsqueeze(-2))
            pred = torch.matmul(sa, z_i.unsqueeze(-2))
            s = s * w_i.unsqueeze(-2) + kv + pred
            out_i = torch.matmul(s, r_i.unsqueeze(-1)).squeeze(-1)
            outs.append(out_i.to(dt).view(b, h * n))
            if (i + 1) % chunk_size == 0 and (i + 1) < t:
                s_ckpts.append(s)

        ctx.save_for_backward(r, w, k, v, a, kk)
        ctx.s_ckpts = s_ckpts
        ctx.chunk_size = chunk_size
        return torch.stack(outs, dim=1), s.to(dt)

    @staticmethod
    def backward(ctx, dout, ds_final):
        r, w, k, v, a, kk = ctx.saved_tensors
        s_ckpts = ctx.s_ckpts
        C = ctx.chunk_size
        b, t, h, n = r.shape
        dt = r.dtype
        dev = r.device

        u = (-kk).float()
        z = (kk * a).float()
        rf, wf, kf, vf = r.float(), w.float(), k.float(), v.float()
        dout_f = dout.float()

        dr = torch.zeros_like(rf)
        dw = torch.zeros_like(wf)
        dk = torch.zeros_like(kf)
        dv = torch.zeros_like(vf)
        du = torch.zeros_like(u)
        dz = torch.zeros_like(z)

        num_chunks = (t + C - 1) // C
        ds = ds_final.float().clone() if ds_final is not None else torch.zeros(b, h, n, n, device=dev, dtype=torch.float32)

        for c in range(num_chunks - 1, -1, -1):
            c_start = c * C
            c_end = min(t, (c + 1) * C)
            c_len = c_end - c_start

            s_local = [s_ckpts[c]]
            for i in range(c_start, c_end):
                w_i, k_i, v_i, u_i, z_i = wf[:, i], kf[:, i], vf[:, i], u[:, i], z[:, i]
                s_prev = s_local[-1]
                sa = torch.matmul(s_prev, u_i.unsqueeze(-1))
                kv = torch.matmul(v_i.unsqueeze(-1), k_i.unsqueeze(-2))
                pred = torch.matmul(sa, z_i.unsqueeze(-2))
                s_curr = s_prev * w_i.unsqueeze(-2) + kv + pred
                s_local.append(s_curr)

            for idx in range(c_len - 1, -1, -1):
                i = c_start + idx
                r_i, w_i, k_i, v_i, u_i, z_i = rf[:, i], wf[:, i], kf[:, i], vf[:, i], u[:, i], z[:, i]
                s_prev = s_local[idx]
                s_curr = s_local[idx + 1]
                dout_i = dout_f[:, i].view(b, h, n)

                dr[:, i] = torch.matmul(s_curr.transpose(-1, -2), dout_i.unsqueeze(-1)).squeeze(-1)
                ds += torch.matmul(dout_i.unsqueeze(-1), r_i.unsqueeze(-2))

                dv[:, i] = torch.matmul(ds, k_i.unsqueeze(-1)).squeeze(-1)
                dk[:, i] = torch.matmul(ds.transpose(-1, -2), v_i.unsqueeze(-1)).squeeze(-1)

                sa = torch.matmul(s_prev, u_i.unsqueeze(-1))
                dz[:, i] = torch.matmul(ds.transpose(-1, -2), sa).squeeze(-1)
                d_sa = torch.matmul(ds, z_i.unsqueeze(-1))
                du[:, i] = torch.matmul(s_prev.transpose(-1, -2), d_sa).squeeze(-1)

                dw[:, i] = (ds * s_prev).sum(dim=-2)
                ds = ds * w_i.unsqueeze(-2) + torch.matmul(d_sa, u_i.unsqueeze(-2))

        dkk = (-du + dz * a.float()).to(dt)
        da = (dz * kk.float()).to(dt)
        ds_ret = ds.to(dt) if ctx.needs_input_grad[6] else None
        return dr.to(dt), dw.to(dt), dk.to(dt), dv.to(dt), da, dkk, ds_ret, None


def chunkwise_pla(
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, t, h, n = r.shape
    dt = r.dtype
    w, k, v, a, kk = w.to(dt), k.to(dt), v.to(dt), a.to(dt), kk.to(dt)
    out, final_s = FastPLAFunction.apply(r, w, k, v, a, kk, s, chunk_size)
    if with_bonus and r_k is not None:
        bonus = (r * k * r_k.to(dt).view(1, 1, h, n)).sum(dim=-1, keepdim=True) * v
        out = out + bonus.view(b, t, h * n)
    return out, final_s
