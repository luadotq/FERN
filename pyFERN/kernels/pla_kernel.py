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
    h, n = r.shape[0], r.shape[1]
    sa = s @ (-kk).view(h, n, 1)
    e = v.view(h, n, 1) @ k.view(h, 1, n) + sa @ (kk * a).view(h, 1, n)
    s_new = s * w.view(h, 1, n) + e
    out = (s_new.to(dtype=r.dtype) @ r.view(h, n, 1)).view(h * n)
    return out, s_new


def pla_recurrent(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    kk: torch.Tensor,
    r_k: torch.Tensor,
    s: Optional[torch.Tensor] = None,
    with_bonus: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, t, h, n = r.shape
    if s is None:
        s = torch.zeros((b, h, n, n), device=r.device, dtype=torch.float32)

    outs = []
    r_k_flat = r_k.view(h, n) if (r_k is not None and with_bonus) else None
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

        out_i = torch.matmul(s.to(dtype=r.dtype), r_i.unsqueeze(-1)).squeeze(-1)
        if r_k_flat is not None:
            bonus_i = (r_i * k_i * r_k_flat).sum(dim=-1, keepdim=True) * v_i
            out_i = out_i + bonus_i
        outs.append(out_i.view(b, h * n))

    return torch.stack(outs, dim=1), s


def chunkwise_pla(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    kk: torch.Tensor,
    r_k: Optional[torch.Tensor] = None,
    s: Optional[torch.Tensor] = None,
    chunk_size: int = 32,
    with_bonus: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, t, h, n = r.shape
    c_size = chunk_size
    if t <= c_size:
        return pla_recurrent(r, w, k, v, a, kk, r_k, s=s, with_bonus=with_bonus)

    nc = t // c_size
    rem = t % c_size

    r_m = r[:, : nc * c_size]
    w_m = w[:, : nc * c_size]
    k_m = k[:, : nc * c_size]
    v_m = v[:, : nc * c_size]
    a_m = a[:, : nc * c_size]
    kk_m = kk[:, : nc * c_size]

    r_c = r_m.view(b, nc, c_size, h, n).permute(0, 3, 1, 2, 4)
    w_c = w_m.view(b, nc, c_size, h, n).permute(0, 3, 1, 2, 4)
    k_c = k_m.view(b, nc, c_size, h, n).permute(0, 3, 1, 2, 4)
    v_c = v_m.view(b, nc, c_size, h, n).permute(0, 3, 1, 2, 4)
    a_c = a_m.view(b, nc, c_size, h, n).permute(0, 3, 1, 2, 4)
    kk_c = kk_m.view(b, nc, c_size, h, n).permute(0, 3, 1, 2, 4)

    u_c = -kk_c
    z_c = kk_c * a_c

    A = torch.zeros(b, h, nc, c_size, c_size, device=r.device, dtype=r.dtype)
    r_tilde = torch.zeros(b, h, nc, c_size, n, device=r.device, dtype=r.dtype)
    k_tilde = torch.zeros(b, h, nc, c_size, n, device=r.device, dtype=r.dtype)

    for tau in range(c_size):
        A[:, :, :, tau, tau] = (k_c[:, :, :, tau] * r_c[:, :, :, tau]).sum(dim=-1)
        curr = r_c[:, :, :, tau]
        for j in range(tau, -1, -1):
            dot = (z_c[:, :, :, j] * curr).sum(dim=-1, keepdim=True)
            curr = w_c[:, :, :, j] * curr + u_c[:, :, :, j] * dot
            if j > 0:
                A[:, :, :, j - 1, tau] = (k_c[:, :, :, j - 1] * curr).sum(dim=-1)
        r_tilde[:, :, :, tau] = curr

    for j in range(c_size):
        curr = k_c[:, :, :, j]
        for m in range(j + 1, c_size):
            dot = (curr * u_c[:, :, :, m]).sum(dim=-1, keepdim=True)
            curr = curr * w_c[:, :, :, m] + dot * z_c[:, :, :, m]
        k_tilde[:, :, :, j] = curr

    delta_s = torch.matmul(v_c.transpose(-2, -1), k_tilde)
    intra_out = torch.matmul(A.transpose(-2, -1), v_c)

    curr_s = s if s is not None else torch.zeros(b, h, n, n, device=r.device, dtype=torch.float32)
    outs = []

    for c in range(nc):
        inter_out = torch.matmul(curr_s.unsqueeze(2), r_tilde[:, :, c].unsqueeze(-1)).squeeze(-1)
        out_c = inter_out + intra_out[:, :, c]
        outs.append(out_c)

        s_trans = curr_s
        for m in range(c_size):
            dot = torch.matmul(s_trans, u_c[:, :, c, m].unsqueeze(-1)).squeeze(-1)
            s_trans = s_trans * w_c[:, :, c, m].unsqueeze(2) + dot.unsqueeze(-1) * z_c[:, :, c, m].unsqueeze(2)
        curr_s = s_trans + delta_s[:, :, c]

    main_outs = torch.cat(outs, dim=2)
    if with_bonus and r_k is not None:
        r_h = r_m.transpose(1, 2)
        k_h = k_m.transpose(1, 2)
        v_h = v_m.transpose(1, 2)
        bonus = (r_h * k_h * r_k.view(1, h, 1, n)).sum(dim=-1, keepdim=True) * v_h
        main_outs = main_outs + bonus

    final_main = main_outs.permute(0, 2, 1, 3).contiguous().view(b, nc * c_size, h * n)

    if rem > 0:
        rem_out, curr_s = chunkwise_pla(
            r[:, nc * c_size :],
            w[:, nc * c_size :],
            k[:, nc * c_size :],
            v[:, nc * c_size :],
            a[:, nc * c_size :],
            kk[:, nc * c_size :],
            r_k=r_k,
            s=curr_s,
            chunk_size=c_size,
            with_bonus=with_bonus,
        )
        return torch.cat([final_main, rem_out], dim=1), curr_s

    return final_main, curr_s
