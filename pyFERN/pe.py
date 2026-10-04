import math
import torch

def sinusoidal_pe(pos: int, d: int, device=None, dtype=torch.float32) -> torch.Tensor:
    pe = torch.zeros((1, d), device=device, dtype=dtype)
    half_d = d // 2
    for i in range(half_d):
        div = 10000.0 ** (2.0 * i / d)
        val = pos / div
        pe[0, 2 * i] = math.sin(val)
        pe[0, 2 * i + 1] = math.cos(val)
    return pe

def apply_rope(x: torch.Tensor, pos: int) -> torch.Tensor:
    if pos == 0 or x.numel() == 0:
        return x
    d = x.shape[-1]
    half_d = d // 2
    if half_d == 0:
        return x

    device, dtype = x.device, x.dtype
    cos = torch.ones(d, device=device, dtype=dtype)
    sin = torch.zeros(d, device=device, dtype=dtype)

    for i in range(half_d):
        div = 10000.0 ** (2.0 * i / d)
        theta = pos / div
        c, s = math.cos(theta), math.sin(theta)
        cos[i] = c
        cos[i + half_d] = c
        sin[i] = s
        sin[i + half_d] = s

    x1 = x[..., :half_d]
    x2 = x[..., half_d:2 * half_d]
    x_rot = torch.cat([-x2, x1], dim=-1)
    if 2 * half_d < d:
        tail = torch.zeros_like(x[..., 2 * half_d:])
        x_rot = torch.cat([x_rot, tail], dim=-1)

    return x * cos + x_rot * sin
