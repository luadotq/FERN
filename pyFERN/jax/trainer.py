import os
import time
import json
from functools import partial
from typing import Optional, List, Union, Tuple, Dict, Any, Iterator
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import optax
import flax.serialization as fser
from flax.training.train_state import TrainState
from safetensors.numpy import save_file

from ..config import ModelConfig, TrainingConfig
from ..data import PretokenizedDataset, JAXDataIterator
from ..checkpoint import (
    find_latest_checkpoint,
    prune_checkpoints,
    push_to_hub,
    resolve_checkpoint_dir,
    check_hf_hub_access,
)
from .model import JAXFERNModel, load_safetensors_to_jax

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

        warmup = min(self.config.warmup_steps, max(1, self.config.total_steps // 10))
        decay_steps = max(warmup + 1, self.config.total_steps)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=self.config.min_lr,
            peak_value=self.config.lr,
            warmup_steps=warmup,
            decay_steps=decay_steps,
            end_value=self.config.min_lr,
        )

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

        raw_params = params["params"] if "params" in params else params
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=raw_params,
            tx=tx,
        )

        self._compiled_step = jax.jit(
            partial(
                self._train_step_fn,
                model=self.model,
                fe_weight=self.config.fe_weight,
                vocab_size=self.model.config.vocab_size,
            ),
            donate_argnums=(0,),
        )

        self._compiled_eval_step = jax.jit(
            partial(
                self._eval_step_fn,
                model=self.model,
                fe_weight=self.config.fe_weight,
                vocab_size=self.model.config.vocab_size,
            )
        )

    @staticmethod
    def _train_step_fn(
        state: TrainState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        model: JAXFERNModel,
        fe_weight: float,
        vocab_size: int,
    ):
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

    @staticmethod
    def _eval_step_fn(
        params: Any,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        model: JAXFERNModel,
        fe_weight: float,
        vocab_size: int,
    ):
        logits, avg_fe = model.apply({"params": params}, inputs, method=model.forward_parallel)
        one_hot = jax.nn.one_hot(targets, vocab_size)
        log_probs = jax.nn.log_softmax(logits)
        ce_loss = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
        total_loss = ce_loss + fe_weight * avg_fe
        return total_loss, ce_loss, avg_fe

    def train_step(self, inputs: jnp.ndarray, targets: jnp.ndarray) -> Tuple[float, float, float]:
        if self.data_sharding is not None:
            inputs = jax.device_put(inputs, self.data_sharding)
            targets = jax.device_put(targets, self.data_sharding)

        self.state, total_loss, ce_loss, avg_fe = self._compiled_step(self.state, inputs, targets)
        return float(total_loss), float(ce_loss), float(avg_fe)

    def evaluate(self, val_dataset, val_steps: int = 50) -> Tuple[float, float, float, float, float]:
        if val_dataset is None or len(val_dataset) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        val_iter = JAXDataIterator(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        total_loss, total_ce, total_fe = 0.0, 0.0, 0.0
        steps = min(val_steps, max(1, len(val_dataset) // self.config.batch_size))

        for _ in range(steps):
            in_batch, tgt_batch = next(val_iter)
            in_jnp = jnp.array(in_batch)
            tgt_jnp = jnp.array(tgt_batch)
            if self.data_sharding is not None:
                in_jnp = jax.device_put(in_jnp, self.data_sharding)
                tgt_jnp = jax.device_put(tgt_jnp, self.data_sharding)
            l, ce, fe = self._compiled_eval_step(self.state.params, in_jnp, tgt_jnp)
            total_loss += float(l)
            total_ce += float(ce)
            total_fe += float(fe)

        avg_loss = total_loss / steps
        avg_ce = total_ce / steps
        avg_fe = total_fe / steps
        ppl = float(np.exp(min(20.0, avg_ce)))
        bpb = float(avg_ce / (np.log(2) * max(0.1, self.config.bytes_per_token)))
        return avg_loss, avg_ce, avg_fe, ppl, bpb

    def train(
        self,
        dataset_or_tokens: Union[PretokenizedDataset, JAXDataIterator, List[int], np.ndarray, str, Any],
        save_path: Optional[str] = None,
        log_interval: Optional[int] = None,
    ):
        cfg = self.config
        total_steps = cfg.total_steps
        bs, sl = cfg.batch_size, cfg.seq_len
        log_int = log_interval or cfg.log_interval
        n_tok = bs * sl

        from ..telemetry import TelegramMonitor, MetricsLogger, TrainingControl, setup_emergency_handler
        import shutil

        val_dataset = None
        if hasattr(dataset_or_tokens, "val_dataset"):
            data_iter = dataset_or_tokens
            val_dataset = dataset_or_tokens.val_dataset
        elif isinstance(dataset_or_tokens, JAXDataIterator):
            data_iter = dataset_or_tokens
        elif isinstance(dataset_or_tokens, PretokenizedDataset):
            data_iter = JAXDataIterator(dataset_or_tokens, batch_size=bs, shuffle=True)
        elif isinstance(dataset_or_tokens, str) and not os.path.exists(dataset_or_tokens) and ("/" in dataset_or_tokens):
            from ..streaming import SlidingShardLoader
            loader = SlidingShardLoader(dataset_or_tokens, batch_size=bs, seq_len=sl, hf_token=cfg.hf_token, window_size=cfg.sliding_window_shards)
            data_iter = loader
            val_dataset = loader.val_dataset
        elif isinstance(dataset_or_tokens, str) and (os.path.isdir(dataset_or_tokens) or dataset_or_tokens.endswith((".bin", ".npy"))):
            ds = PretokenizedDataset(dataset_or_tokens, seq_len=sl)
            data_iter = JAXDataIterator(ds, batch_size=bs, shuffle=True)
        else:
            raw_tokens = np.array(dataset_or_tokens, dtype=np.int32)
            max_start = max(1, len(raw_tokens) - n_tok - 1)

            def memory_generator():
                step = 0
                while True:
                    offset = (step * n_tok) % max_start
                    in_batch = raw_tokens[offset : offset + n_tok].reshape((bs, sl))
                    tgt_batch = raw_tokens[offset + 1 : offset + n_tok + 1].reshape((bs, sl))
                    yield in_batch, tgt_batch
                    step += 1

            data_iter = memory_generator()

        start_step = 1
        cum_tok = 0

        # Resume from checkpoint if configured
        if cfg.resume_from_checkpoint:
            resume_path = cfg.resume_from_checkpoint
            if resume_path == "latest":
                resume_path = find_latest_checkpoint(cfg.checkpoint_dir)
                if not resume_path:
                    raise FileNotFoundError(f"No checkpoint found in '{cfg.checkpoint_dir}' to resume from")

            print(f"[JAX Resume] Loading state from '{resume_path}'...")
            _, loaded = load_safetensors_to_jax(resume_path, config=self.model.config, token=cfg.hf_token)
            raw_p = loaded["params"] if "params" in loaded else loaded
            self.state = self.state.replace(params=raw_p)

            # Restore optax state if saved
            ckpt_dir = resolve_checkpoint_dir(resume_path, token=cfg.hf_token)
            opt_file = os.path.join(ckpt_dir, "opt_state.flax")
            if os.path.exists(opt_file):
                with open(opt_file, "rb") as f:
                    restored_opt = fser.from_bytes(self.state.opt_state, f.read())
                    self.state = self.state.replace(opt_state=restored_opt)

            state_file = os.path.join(ckpt_dir, "training_state.json")
            if os.path.exists(state_file):
                with open(state_file, "r", encoding="utf-8") as f:
                    t_state = json.load(f)
                    saved_step = t_state.get("step", 0)
                    start_step = saved_step + 1
                    cum_tok = t_state.get("tokens_seen", saved_step * n_tok)
            print(f"[JAX Resume] Resumed from step {start_step - 1} ({cum_tok:,} tokens seen)")

        if cfg.hf_repo_id:
            ok, err = check_hf_hub_access(cfg.hf_repo_id, token=cfg.hf_token, private=cfg.hf_private)
            if not ok:
                print(f"[Warning] Hugging Face Hub validation failed for '{cfg.hf_repo_id}': {err}")
                print("[Warning] Disabling HF uploads; continuing training with local checkpoints only.")
                cfg.hf_repo_id = None

        control = TrainingControl()
        logger = MetricsLogger(log_dir=cfg.checkpoint_dir, csv_name="metrics.csv")

        start_time = time.time()
        last_save_time = start_time
        step = start_step
        tot, ce, fe = 0.0, 0.0, 0.0

        def get_status():
            now = time.time()
            elapsed = now - start_time
            speed = cum_tok / elapsed if elapsed > 0 else 0
            rem_tok = max(0, total_toks - (step - start_step + 1) * n_tok)
            eta_sec = rem_tok / speed if speed > 0 else 0
            free_gb = shutil.disk_usage(cfg.checkpoint_dir)[2] / 1e9 if os.path.exists(cfg.checkpoint_dir) else 0.0
            latest = logger.get_latest()
            return {
                "step": step,
                "total_steps": total_steps,
                "pct": (step / total_steps) * 100,
                "tokens_seen": cum_tok,
                "loss": tot,
                "ce": ce,
                "fe": fe,
                "val_loss": latest.get("val_loss", "-"),
                "val_ppl": latest.get("val_ppl", "-"),
                "val_bpb": latest.get("val_bpb", "-"),
                "lr": cfg.lr,
                "speed": speed,
                "elapsed_str": f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m",
                "eta_str": f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m",
                "free_disk_gb": free_gb,
            }

        bot = TelegramMonitor(
            token=cfg.telegram_token,
            chat_id=cfg.telegram_chat_id,
            control=control,
            logger=logger,
            get_status_fn=get_status,
        )

        def emergency_save():
            save_dir = os.path.join(cfg.checkpoint_dir, f"step_emergency_{step:07d}")
            self.save_checkpoint(save_dir, training_state={"step": step, "tokens_seen": cum_tok, "emergency": True})
            if cfg.hf_repo_id:
                push_to_hub(save_dir, cfg.hf_repo_id, token=cfg.hf_token, private=cfg.hf_private)
                try:
                    from huggingface_hub import HfApi
                    HfApi(token=cfg.hf_token).upload_file(
                        path_or_fileobj=logger.csv_path,
                        path_in_repo="metrics.csv",
                        repo_id=cfg.hf_repo_id,
                        repo_type="model",
                        token=cfg.hf_token,
                    )
                except Exception:
                    pass
            if bot:
                bot.send_message(f"emergency: jax checkpoint step_{step:07d} saved and uploaded.")
        setup_emergency_handler(emergency_save)

        total_toks = (total_steps - start_step + 1) * n_tok
        print(f"[JAX TPU] Training: steps {start_step}->{total_steps} ({total_toks:,} tokens, devices: {jax.device_count()})")
        if bot:
            bot.send_message(f"jax tpu training started: steps {start_step}->{total_steps} ({total_toks:,} tokens, {jax.device_count()} devices)")

        for step in range(start_step, total_steps + 1):
            if control.emergency_exit or control.request_stop:
                print(f"[Control] Stopping JAX training at step {step}.")
                break

            in_batch, tgt_batch = next(data_iter)
            in_jnp = jnp.array(in_batch)
            tgt_jnp = jnp.array(tgt_batch)

            tot, ce, fe = self.train_step(in_jnp, tgt_jnp)
            cum_tok += n_tok

            # Periodic validation
            trigger_val = (val_dataset is not None) and (control.request_val or (cfg.val_interval_steps and step % cfg.val_interval_steps == 0))
            if trigger_val:
                control.request_val = False
                val_loss, val_ce, val_fe, val_ppl, val_bpb = self.evaluate(val_dataset, val_steps=cfg.val_steps)
                print(f"[JAX Val] Step {step} | Loss: {val_loss:.4f} | PPL: {val_ppl:.2f} | BPB: {val_bpb:.3f}")
                logger.log({
                    "step": step, "tokens": cum_tok,
                    "train_loss": tot, "ce_loss": ce, "fe_loss": fe,
                    "val_loss": val_loss, "val_ce": val_ce, "val_fe": val_fe,
                    "val_ppl": val_ppl, "val_bpb": val_bpb,
                    "lr": cfg.lr, "speed": cum_tok / max(0.1, time.time() - start_time), "elapsed": time.time() - start_time,
                })
                if bot:
                    bot.send_message(f"val step {step}: loss={val_loss:.4f} ppl={val_ppl:.2f} bpb={val_bpb:.3f}")

            # Checkpoint triggers: step, time, or manual /save
            now = time.time()
            trigger_step = cfg.save_interval_steps and (step % cfg.save_interval_steps == 0)
            trigger_time = cfg.save_interval_seconds and (now - last_save_time >= cfg.save_interval_seconds)
            trigger_save = control.request_save or trigger_step or trigger_time

            if trigger_save:
                control.request_save = False
                last_save_time = now
                state_dict = {
                    "step": step,
                    "tokens_seen": cum_tok,
                    "total_loss": tot,
                    "ce_loss": ce,
                    "fe_loss": fe,
                    "timestamp": now,
                }
                saved_dir = self.save_checkpoint(
                    path_or_dir=os.path.join(cfg.checkpoint_dir, f"step_{step:07d}"),
                    training_state=state_dict,
                )
                prune_checkpoints(cfg.checkpoint_dir, cfg.max_checkpoints_to_keep)
                if cfg.hf_repo_id and cfg.hf_push_on_save:
                    push_to_hub(saved_dir, cfg.hf_repo_id, token=cfg.hf_token, private=cfg.hf_private)
                    try:
                        from huggingface_hub import HfApi
                        HfApi(token=cfg.hf_token).upload_file(
                            path_or_fileobj=logger.csv_path,
                            path_in_repo="metrics.csv",
                            repo_id=cfg.hf_repo_id,
                            repo_type="model",
                            token=cfg.hf_token,
                        )
                    except Exception:
                        pass
                print(f"[JAX Checkpoint] Saved step {step} -> {saved_dir}")
                if bot:
                    bot.send_message(f"checkpoint saved: step_{step:07d}")

            if step % log_int == 0 or step == start_step or step == total_steps:
                elapsed = time.time() - start_time
                speed = cum_tok / elapsed if elapsed > 0 else 0
                print(f"Step {step:>4}/{total_steps} ({cum_tok:>6} tok) | CE: {ce:.4f} | FE: {fe:.4f} | Loss: {tot:.4f} | Speed: {speed:.0f} tok/s")
                logger.log({
                    "step": step, "tokens": cum_tok,
                    "train_loss": tot, "ce_loss": ce, "fe_loss": fe,
                    "lr": cfg.lr, "speed": speed, "elapsed": elapsed,
                })

        final_dir = save_path or os.path.join(cfg.checkpoint_dir, f"step_{step:07d}")
        self.save_checkpoint(
            path_or_dir=final_dir,
            training_state={
                "step": step,
                "tokens_seen": cum_tok,
                "total_loss": tot,
                "ce_loss": ce,
                "fe_loss": fe,
                "timestamp": time.time(),
            },
        )
        if cfg.checkpoint_dir:
            prune_checkpoints(cfg.checkpoint_dir, cfg.max_checkpoints_to_keep)
        if cfg.hf_repo_id:
            push_to_hub(final_dir, cfg.hf_repo_id, token=cfg.hf_token, private=cfg.hf_private)
            try:
                from huggingface_hub import HfApi
                HfApi(token=cfg.hf_token).upload_file(
                    path_or_fileobj=logger.csv_path,
                    path_in_repo="metrics.csv",
                    repo_id=cfg.hf_repo_id,
                    repo_type="model",
                    token=cfg.hf_token,
                )
            except Exception:
                pass
        print(f"[OK] Saved final JAX checkpoint to '{final_dir}'")
        if bot:
            bot.send_message(f"training completed: step_{step:07d}")
            bot.close()

    def save_checkpoint(self, path_or_dir: str, training_state: Optional[Dict[str, Any]] = None) -> str:
        if path_or_dir.endswith(".safetensors"):
            save_dir = os.path.dirname(os.path.abspath(path_or_dir))
            weights_file = path_or_dir
        else:
            save_dir = path_or_dir
            weights_file = os.path.join(save_dir, "model.safetensors")

        os.makedirs(save_dir, exist_ok=True)
        flat_dict: Dict[str, np.ndarray] = {}

        def _flatten(d: dict, prefix: str = ""):
            for k, v in d.items():
                name = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(v, name)
                elif hasattr(v, "shape"):
                    arr = np.array(v)
                    # JAX [in, out] -> PyTorch [out, in]
                    if name.endswith(".kernel"):
                        name_pt = name[:-7] + ".weight"
                        arr = arr.T
                    elif name.endswith(".embedding"):
                        name_pt = name[:-10] + ".weight"
                    else:
                        name_pt = name

                    parts = name_pt.split(".")
                    for i, p in enumerate(parts):
                        if p.startswith("block_") and p[6:].isdigit():
                            parts[i] = f"blocks.{p[6:]}"
                    final_name = ".".join(parts)
                    flat_dict[final_name] = np.ascontiguousarray(arr)

        _flatten(self.state.params)
        save_file(flat_dict, weights_file)
        self.model.config.to_json(os.path.join(save_dir, "config.json"))
        self.config.to_json(os.path.join(save_dir, "training_config.json"))

        # Save optax optimizer state
        opt_bytes = fser.to_bytes(self.state.opt_state)
        with open(os.path.join(save_dir, "opt_state.flax"), "wb") as f:
            f.write(opt_bytes)

        if training_state:
            with open(os.path.join(save_dir, "training_state.json"), "w", encoding="utf-8") as f:
                json.dump(training_state, f, indent=2)

        parent = os.path.dirname(save_dir)
        if os.path.isdir(parent):
            with open(os.path.join(parent, "latest_checkpoint.json"), "w", encoding="utf-8") as f:
                json.dump({"latest": os.path.basename(save_dir)}, f, indent=2)

        return save_dir
