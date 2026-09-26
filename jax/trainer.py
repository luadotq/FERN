import time
from functools import partial
from typing import Optional, List, Union, Tuple, Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import optax
from flax.training.train_state import TrainState

from ..config import ModelConfig, TrainingConfig
from .model import JAXFERNModel

def create_tpu_mesh() -> Optional[Mesh]:
    devices = jax.devices()
    if len(devices) > 1:
        return Mesh(np.array(devices), axis_names=('data',))
    return None

def get_data_sharding(mesh: Optional[Mesh]) -> Optional[NamedSharding]:
    if mesh is not None:
        return NamedSharding(mesh, P('data', None))
    return None

def get_replicated_sharding(mesh: Optional[Mesh]) -> Optional[NamedSharding]:
    if mesh is not None:
        return NamedSharding(mesh, P())
    return None


class JAXFERNTrainer:
    def __init__(
        self,
        model: JAXFERNModel,
        params: Dict[str, Any],
        config: Optional[TrainingConfig] = None,
        **kwargs,
    ):
        self.model = model
        self.config = config or TrainingConfig(**kwargs)
        self.mesh = create_tpu_mesh()
        self.data_sharding = get_data_sharding(self.mesh)
        self.replicated_sharding = get_replicated_sharding(self.mesh)

        # LR schedule with warmup and cosine decay
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=self.config.min_lr,
            peak_value=self.config.lr,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.config.total_steps,
            end_value=self.config.min_lr,
        )

        # Optimizer chain: clip -> adamw
        tx = optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.adamw(
                learning_rate=schedule,
                b1=self.config.adam_b1,
                b2=self.config.adam_b2,
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay,
            )
        )

        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params["params"] if "params" in params else params,
            tx=tx,
        )

        # Static JIT compilation for maximum efficiency with buffer donation
        self._compiled_step = jax.jit(
            partial(self._train_step_fn, model=self.model, fe_weight=self.config.fe_weight, vocab_size=self.model.config.vocab_size),
            donate_argnums=(0,),
        )

    @staticmethod
    def _train_step_fn(state: TrainState, inputs: jnp.ndarray, targets: jnp.ndarray, model: JAXFERNModel, fe_weight: float, vocab_size: int):
        def loss_fn(params):
            logits, avg_fe = model.apply({"params": params}, inputs, method=model.forward_parallel)
            one_hot = jax.nn.one_hot(targets, vocab_size)
            log_probs = jax.nn.log_softmax(logits)
            ce_loss = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
            total_loss = ce_loss + fe_weight * avg_fe
            return total_loss, (ce_loss, avg_fe)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (ce_loss, avg_fe)), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, total_loss, ce_loss, avg_fe

    def train_step(self, inputs: jnp.ndarray, targets: jnp.ndarray) -> Tuple[float, float, float]:
        if self.data_sharding is not None:
            inputs = jax.device_put(inputs, self.data_sharding)
            targets = jax.device_put(targets, self.data_sharding)

        self.state, total_loss, ce_loss, avg_fe = self._compiled_step(self.state, inputs, targets)
        return float(total_loss), float(ce_loss), float(avg_fe)

    def train(
        self,
        tokens: Union[List[int], np.ndarray],
        log_interval: Optional[int] = None,
    ):
        cfg = self.config
        total_steps = cfg.total_steps
        bs, sl = cfg.batch_size, cfg.seq_len
        log_int = log_interval or cfg.log_interval

        if isinstance(tokens, list):
            tokens = np.array(tokens, dtype=np.int32)

        start_time = time.time()
        print(f"[JAX TPU] Training on {total_steps * bs * sl:,} tokens ({total_steps} steps, devices: {jax.device_count()})")

        for step in range(1, total_steps + 1):
            offset = (step - 1) * (bs * sl)
            in_batch = tokens[offset : offset + bs * sl].reshape((bs, sl))
            tgt_batch = tokens[offset + 1 : offset + bs * sl + 1].reshape((bs, sl))

            in_jnp = jnp.array(in_batch)
            tgt_jnp = jnp.array(tgt_batch)

            tot, ce, fe = self.train_step(in_jnp, tgt_jnp)

            if step % log_int == 0 or step == 1 or step == total_steps:
                elapsed = time.time() - start_time
                cum_tok = step * bs * sl
                speed = cum_tok / elapsed if elapsed > 0 else 0
                print(f"Step {step:>3}/{total_steps} ({cum_tok:>5} tok) | CE: {ce:.4f} | FE: {fe:.4f} | Loss: {tot:.4f} | Speed: {speed:.0f} tok/s")
