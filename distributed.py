import os
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
from .config import TrainingConfig

def init_distributed(backend: Optional[str] = None) -> Tuple[int, int, int]:
    if not torch.distributed.is_available():
        return 0, 1, 0

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if not torch.distributed.is_initialized():
            if backend is None:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def get_deepspeed_config(config: TrainingConfig, world_size: int = 1) -> Dict[str, Any]:
    if isinstance(config.deepspeed_config, dict):
        return config.deepspeed_config

    stage = config.zero_stage
    fp16_enabled = config.mixed_precision == "fp16"
    bf16_enabled = config.mixed_precision == "bf16"

    ds_config = {
        "train_micro_batch_size_per_gpu": config.batch_size,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": config.grad_clip,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.lr,
                "betas": [config.adam_b1, config.adam_b2],
                "eps": config.adam_eps,
                "weight_decay": config.weight_decay,
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": config.total_steps,
                "warmup_min_lr": config.min_lr,
                "warmup_max_lr": config.lr,
                "warmup_num_steps": config.warmup_steps,
            }
        },
        "fp16": {"enabled": fp16_enabled},
        "bf16": {"enabled": bf16_enabled},
        "zero_optimization": {
            "stage": stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }

    if stage == 3:
        ds_config["zero_optimization"].update({
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "sub_group_size": 1e9,
        })

    return ds_config


def setup_distributed_engine(
    model: nn.Module,
    config: TrainingConfig,
) -> Tuple[nn.Module, Optional[Any], Optional[Any]]:
    rank, world_size, local_rank = init_distributed(config.distributed_backend)

    if config.use_deepspeed:
        try:
            import deepspeed
        except ImportError:
            raise ImportError("DeepSpeed is not installed. Install it via `pip install deepspeed`.")

        ds_config = get_deepspeed_config(config, world_size)
        engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
        return engine, optimizer, lr_scheduler

    if world_size > 1:
        device_ids = [local_rank] if torch.cuda.is_available() else None
        model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    return model, None, None
