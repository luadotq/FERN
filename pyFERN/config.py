import json
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any

@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_layers: List[int] = field(default_factory=lambda: [64, 128, 128])
    d_model: Optional[int] = None
    num_layers: Optional[int] = None
    d_mem: int = 128
    num_heads: int = 4
    mlp_dim: Optional[int] = None
    head_dim: Optional[int] = None
    kappa: float = 0.3
    alpha: float = 0.9
    epsilon_min: float = 1e-4
    max_drive: float = 5.0
    d_protected: int = 16
    pi_max: float = 20.0
    adaptive_k: bool = False
    legacy_ampc: bool = False
    
    # Mathematical & Normalization constants
    rms_eps: float = 1e-5
    gamma_min: float = 0.90
    gamma_max: float = 0.98
    pe_scale: float = 0.1
    init_range: float = 0.02
    fe_weight: float = 0.1

    # Chunkwise Attention & Scaling
    chunk_size: int = 64
    gradient_checkpointing: bool = False

    # Special token IDs
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    min_lr: float = 5e-5
    warmup_steps: int = 10
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    fe_weight: float = 0.1

    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8

    batch_size: int = 4
    seq_len: int = 64
    total_steps: int = 200
    log_interval: int = 20
    seed: int = 42

    device: Optional[str] = None
    mixed_precision: str = "no"
    use_deepspeed: bool = False
    zero_stage: int = 2
    deepspeed_config: Optional[Union[str, Dict[str, Any]]] = None
    distributed_backend: str = "nccl"

    # Checkpointing (steps / time)
    save_interval_steps: Optional[int] = None
    save_interval_seconds: Optional[int] = None
    checkpoint_dir: str = "checkpoints"
    max_checkpoints_to_keep: int = 3
    resume_from_checkpoint: Optional[str] = None

    # Hugging Face Hub
    hf_repo_id: Optional[str] = None
    hf_private: bool = True
    hf_token: Optional[str] = None
    hf_push_on_save: bool = False

    # Telegram & Monitoring
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_interval: int = 100

    # Validation & Metrics
    val_interval_steps: Optional[int] = None
    val_steps: int = 50
    bytes_per_token: float = 3.8

    # Dynamic Streaming
    sliding_window_shards: int = 2

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)
