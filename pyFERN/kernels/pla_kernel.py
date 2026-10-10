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
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, t, h, n = r.shape
    if s is None:
        s = torch.zeros((b, h, n, n), device=r.device, dtype=torch.float32)

    outs = []
    r_k_flat = r_k.view(h, n)
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
        bonus_i = (r_i * k_i * r_k_flat).sum(dim=-1, keepdim=True) * v_i
        outs.append((out_i + bonus_i).view(b, h * n))

    return torch.stack(outs, dim=1), s
