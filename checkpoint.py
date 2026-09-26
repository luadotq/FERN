import os
from typing import Optional, Union
import torch
from safetensors.torch import load, save_file

from .config import ModelConfig
from .network import FERNModel

def from_pretrained(
    checkpoint_path: str,
    config: Optional[Union[str, ModelConfig]] = None,
    device: str = "cpu"
) -> FERNModel:
    if os.path.isdir(checkpoint_path):
        weights_file = os.path.join(checkpoint_path, "model.safetensors")
        default_cfg_file = os.path.join(checkpoint_path, "config.json")
    else:
        base = checkpoint_path
        for suffix in [".safetensors", ".json", ".tokenizer.json"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break

        weights_file = checkpoint_path if checkpoint_path.endswith(".safetensors") else f"{base}.safetensors"
        default_cfg_file = f"{base}.json"
        if not os.path.exists(default_cfg_file):
            alt_cfg = os.path.join(os.path.dirname(checkpoint_path), "config.json")
            if os.path.exists(alt_cfg):
                default_cfg_file = alt_cfg

    if isinstance(config, ModelConfig):
        cfg = config
    elif isinstance(config, str):
        if not os.path.exists(config):
            raise FileNotFoundError(f"Config file not found: {config}")
        cfg = ModelConfig.from_json(config)
    else:
        if not os.path.exists(default_cfg_file):
            raise FileNotFoundError(
                f"Config file not found for checkpoint '{checkpoint_path}'. "
                f"Expected config at '{default_cfg_file}'. "
                f"Please provide config explicitly via `config` parameter."
            )
        cfg = ModelConfig.from_json(default_cfg_file)

    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    model = FERNModel(cfg)
    with open(weights_file, "rb") as f:
        state_dict = load(f.read())
    model.load_state_dict(state_dict, strict=False)
    if model.is_deep:
        model.decoder.weight = model.encoder.weight
    model.to(device)
    return model


def save_pretrained(model: FERNModel, save_path: str):
    if os.path.isdir(save_path):
        cfg_file = os.path.join(save_path, "config.json")
        weights_file = os.path.join(save_path, "model.safetensors")
    elif save_path.endswith(".safetensors"):
        base = save_path[:-len(".safetensors")]
        cfg_file = f"{base}.json"
        weights_file = save_path
    else:
        base = save_path
        for suffix in [".json", ".tokenizer.json"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        cfg_file = f"{base}.json"
        weights_file = f"{base}.safetensors"

    os.makedirs(os.path.dirname(os.path.abspath(weights_file)), exist_ok=True)
    model.config.to_json(cfg_file)
    save_file(model.state_dict(), weights_file)
