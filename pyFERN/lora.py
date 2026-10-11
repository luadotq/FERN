import json
import os
import math
from typing import List, Dict, Optional, Tuple, Set
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int = 64,
        alpha: float = 128.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_f = linear.in_features
        out_f = linear.out_features
        dev = linear.weight.device
        dtype = linear.weight.dtype

        self.lora_a = nn.Parameter(torch.empty(r, in_f, device=dev, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=dtype))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Kaiming uniform init for A, zero init for B
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    @property
    def base(self) -> nn.Linear:
        return self.linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.linear(x)
        if not self.merged:
            lora_out = (self.dropout(x) @ self.lora_a.T @ self.lora_b.T) * self.scaling
            res = res + lora_out
        return res

    def merge(self):
        if not self.merged:
            delta = (self.lora_b @ self.lora_a) * self.scaling
            self.linear.weight.data.add_(delta.to(self.linear.weight.dtype))
            self.merged = True

    def unmerge(self):
        if self.merged:
            delta = (self.lora_b @ self.lora_a) * self.scaling
            self.linear.weight.data.sub_(delta.to(self.linear.weight.dtype))
            self.merged = False


def inject_lora(
    model: nn.Module,
    r: int = 64,
    alpha: float = 128.0,
    dropout: float = 0.0,
    targets: Tuple[str, ...] = ("w_q", "w_k", "w_v", "w_out", "w_up", "w_down"),
) -> Tuple[int, int]:
    # Freeze all existing parameters
    for p in model.parameters():
        p.requires_grad = False

    target_set = set(targets)
    injected = 0

    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if child_name in target_set and isinstance(child, nn.Linear):
                lora_layer = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_layer)
                injected += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_adapter(model: nn.Module, save_dir: str, r: int = 64, alpha: float = 128.0):
    os.makedirs(save_dir, exist_ok=True)
    tensors = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            tensors[f"{name}.lora_a"] = module.lora_a.data.contiguous()
            tensors[f"{name}.lora_b"] = module.lora_b.data.contiguous()

    save_file(tensors, os.path.join(save_dir, "adapter_model.safetensors"))
    cfg = {"r": r, "alpha": alpha}
    with open(os.path.join(save_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_adapter(model: nn.Module, save_dir: str):
    weights = load_file(os.path.join(save_dir, "adapter_model.safetensors"))
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_a"
            b_key = f"{name}.lora_b"
            if a_key in weights and b_key in weights:
                module.lora_a.data.copy_(weights[a_key])
                module.lora_b.data.copy_(weights[b_key])


def merge_and_save(model: nn.Module, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    # clone tensors to avoid shared memory error in safetensors
    state = {k: v.clone() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(save_dir, "model.safetensors"))
