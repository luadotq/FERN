import os
import re
import json
import shutil
from typing import Optional, Union, Dict, Any, Tuple
import torch
from safetensors.torch import load, save_file

from .config import ModelConfig, TrainingConfig
from .network import FERNModel

def is_hf_repo(path: str) -> bool:
    if os.path.exists(path):
        return False
    return bool(re.match(r"^[\w\-]+/[\w\-\.]+$", path)) or "/" in path

def resolve_checkpoint_dir(checkpoint_path: str, token: Optional[str] = None, revision: Optional[str] = None) -> str:
    if os.path.isdir(checkpoint_path):
        return checkpoint_path
    if os.path.isfile(checkpoint_path):
        return os.path.dirname(os.path.abspath(checkpoint_path))
    if is_hf_repo(checkpoint_path):
        from huggingface_hub import snapshot_download
        return snapshot_download(
            repo_id=checkpoint_path,
            token=token,
            revision=revision,
            allow_patterns=["*.safetensors", "*.json", "*.bin", "*.pt", "*.flax"],
        )
    raise FileNotFoundError(f"Checkpoint not found locally or on Hugging Face: {checkpoint_path}")

def check_hf_hub_access(repo_id: str, token: Optional[str] = None, private: bool = True) -> Tuple[bool, Optional[str]]:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True, token=token)
        return True, None
    except Exception as e:
        return False, str(e)

def push_to_hub(
    folder_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = True,
    commit_message: Optional[str] = None,
) -> Optional[str]:
    try:
        from huggingface_hub import HfApi, upload_folder
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True, token=token)
        return upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message or "Upload pyFERN checkpoint",
        )
    except Exception as e:
        print(f"[Warning] Failed to push to Hugging Face Hub ({repo_id}): {e}")
        return None

def from_pretrained(
    checkpoint_path: str,
    config: Optional[Union[str, ModelConfig]] = None,
    device: str = "cpu",
    token: Optional[str] = None,
) -> FERNModel:
    ckpt_dir = resolve_checkpoint_dir(checkpoint_path, token=token)
    weights_file = os.path.join(ckpt_dir, "model.safetensors")
    default_cfg = os.path.join(ckpt_dir, "config.json")

    if isinstance(config, ModelConfig):
        cfg = config
    elif isinstance(config, str):
        if not os.path.exists(config):
            raise FileNotFoundError(f"Config file not found: {config}")
        cfg = ModelConfig.from_json(config)
    else:
        if not os.path.exists(default_cfg):
            raise FileNotFoundError(f"config.json not found in '{ckpt_dir}'")
        cfg = ModelConfig.from_json(default_cfg)

    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"model.safetensors not found in '{ckpt_dir}'")

    model = FERNModel(cfg)
    with open(weights_file, "rb") as f:
        state_dict = load(f.read())
    model.load_state_dict(state_dict, strict=False)
    # Tied embeddings decoder weight
    if model.is_deep and hasattr(model, "decoder") and hasattr(model, "encoder"):
        model.decoder.weight = model.encoder.weight
    model.to(device)
    return model

def save_pretrained(
    model: FERNModel,
    save_path: str,
    hf_repo_id: Optional[str] = None,
    hf_private: bool = True,
    hf_token: Optional[str] = None,
):
    save_dir = save_path if os.path.isdir(save_path) or not save_path.endswith(".safetensors") else os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    cfg_file = os.path.join(save_dir, "config.json")
    weights_file = os.path.join(save_dir, "model.safetensors")

    model.config.to_json(cfg_file)
    try:
        from safetensors.torch import save_model
        save_model(model, weights_file)
    except Exception:
        state = {k: v.clone() for k, v in model.state_dict().items()}
        save_file(state, weights_file)

    if hf_repo_id:
        push_to_hub(save_dir, hf_repo_id, token=hf_token, private=hf_private)

def prune_checkpoints(checkpoint_dir: str, max_to_keep: int):
    if max_to_keep <= 0 or not os.path.isdir(checkpoint_dir):
        return
    entries = []
    for d in os.listdir(checkpoint_dir):
        if d.startswith("step_"):
            try:
                step_val = int(d.split("_")[1])
                entries.append((step_val, os.path.join(checkpoint_dir, d)))
            except ValueError:
                continue
    entries.sort(key=lambda x: x[0])
    while len(entries) > max_to_keep:
        _, old_dir = entries.pop(0)
        shutil.rmtree(old_dir, ignore_errors=True)

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    if not os.path.isdir(checkpoint_dir):
        return None
    latest_meta = os.path.join(checkpoint_dir, "latest_checkpoint.json")
    if os.path.exists(latest_meta):
        try:
            with open(latest_meta, "r", encoding="utf-8") as f:
                data = json.load(f)
            cand = os.path.join(checkpoint_dir, data.get("latest", ""))
            if os.path.isdir(cand):
                return cand
        except Exception:
            pass

    entries = []
    for d in os.listdir(checkpoint_dir):
        if d.startswith("step_"):
            try:
                step_val = int(d.split("_")[1])
                entries.append((step_val, os.path.join(checkpoint_dir, d)))
            except ValueError:
                continue
    if not entries:
        return None
    entries.sort(key=lambda x: x[0])
    return entries[-1][1]

def save_training_checkpoint(
    checkpoint_dir: str,
    step: int,
    model: FERNModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    training_state: Optional[Dict[str, Any]] = None,
    training_config: Optional[TrainingConfig] = None,
    max_to_keep: int = 3,
    hf_repo_id: Optional[str] = None,
    hf_private: bool = True,
    hf_token: Optional[str] = None,
) -> str:
    step_dir = os.path.join(checkpoint_dir, f"step_{step:07d}")
    os.makedirs(step_dir, exist_ok=True)
    save_pretrained(model, step_dir)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(step_dir, "optimizer.pt"))
    if training_config is not None:
        training_config.to_json(os.path.join(step_dir, "training_config.json"))
    if training_state is not None:
        with open(os.path.join(step_dir, "training_state.json"), "w", encoding="utf-8") as f:
            json.dump(training_state, f, indent=2)

    with open(os.path.join(checkpoint_dir, "latest_checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump({"latest": f"step_{step:07d}", "step": step}, f, indent=2)

    prune_checkpoints(checkpoint_dir, max_to_keep)

    if hf_repo_id:
        p_res = push_to_hub(step_dir, hf_repo_id, token=hf_token, private=hf_private)
        if p_res and os.path.exists(step_dir):
            shutil.rmtree(step_dir, ignore_errors=True)
            print(f"[Checkpoint] Removed local '{step_dir}' after HF upload")

    return step_dir

def load_training_checkpoint(
    checkpoint_path: str,
    model: FERNModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    if checkpoint_path == "latest":
        raise ValueError("Specify directory containing checkpoints to resolve 'latest'")
    ckpt_dir = resolve_checkpoint_dir(checkpoint_path, token=token)

    weights_file = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(weights_file):
        with open(weights_file, "rb") as f:
            state_dict = load(f.read())
        model.load_state_dict(state_dict, strict=False)
        if model.is_deep and hasattr(model, "decoder") and hasattr(model, "encoder"):
            model.decoder.weight = model.encoder.weight
        model.to(device)

    opt_file = os.path.join(ckpt_dir, "optimizer.pt")
    if optimizer is not None and os.path.exists(opt_file):
        opt_state = torch.load(opt_file, map_location=device)
        optimizer.load_state_dict(opt_state)

    state_file = os.path.join(ckpt_dir, "training_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"step": 0}
