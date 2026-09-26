import torch
import torch.nn as nn
import torch.nn.functional as F

def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class LinearNoBias(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        mid_dim = max(in_dim + out_dim, 32)
        self.linear1 = nn.Linear(in_dim, mid_dim)
        self.linear2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))

class HierarchicalLayer(nn.Module):
    def __init__(self, l: int, d_sensory: int, d_prev: int, d_curr: int):
        super().__init__()
        self.l = l
        self.f_pred = MLP(d_curr, d_prev)
        self.W_up = nn.Linear(d_prev, d_curr)
        self.W_gate = nn.Linear(2 * d_curr, d_curr)
        self.W_rec = LinearNoBias(d_curr, d_curr)
        self.q_phi = MLP(d_sensory + d_curr, d_curr)
