import os
import math
import time
import shutil
from typing import Optional, List, Union, Tuple, Any
import torch
import torch.nn as nn

from .config import TrainingConfig
from .network import FERNModel
from .checkpoint import (
    save_pretrained,
    save_training_checkpoint,
    load_training_checkpoint,
    find_latest_checkpoint,
    push_to_hub,
    check_hf_hub_access,
)
from .distributed import setup_distributed_engine, is_main_process

class FERNTrainer:
    def __init__(
        self,
        model: FERNModel,
        config: Optional[TrainingConfig] = None,
        **kwargs,
    ):
        self.config = config or TrainingConfig(**kwargs)
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.engine, self.ds_optimizer, self.ds_scheduler = setup_distributed_engine(self.model, self.config)

        if not self.config.use_deepspeed:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                betas=(self.config.adam_b1, self.config.adam_b2),
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = self.ds_optimizer

    def get_lr(self, step: int, total_steps: int) -> float:
        warmup = self.config.warmup_steps
        min_lr, lr = self.config.min_lr, self.config.lr
        if step < warmup:
            return min_lr + (lr - min_lr) * (step / max(1, warmup))
        prog = (step - warmup) / max(1, total_steps - warmup)
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * prog))

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float]:
        inp = inputs.to(self.device)
        tgt = targets.to(self.device)

        if self.config.use_deepspeed:
            self.engine.train()
            _, total_loss, ce_loss, avg_fe = self.engine(inp, targets=tgt, fe_weight=self.config.fe_weight)
            self.engine.backward(total_loss)
            self.engine.step()
            return total_loss.item(), ce_loss.item(), avg_fe.item()

        self.model.train()
        self.optimizer.zero_grad()
        _, total_loss, ce_loss, avg_fe = self.model(inp, targets=tgt, fe_weight=self.config.fe_weight)
        total_loss.backward()

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        self.optimizer.step()
        return total_loss.item(), ce_loss.item(), avg_fe.item()

    def evaluate(self, val_dataset, val_steps: int = 50) -> Tuple[float, float, float, float, float]:
        if val_dataset is None or len(val_dataset) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        from .data import JAXDataIterator
        val_iter = JAXDataIterator(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        total_loss, total_ce, total_fe = 0.0, 0.0, 0.0
        steps = min(val_steps, max(1, len(val_dataset) // self.config.batch_size))

        self.model.eval()
        with torch.no_grad():
            for _ in range(steps):
                in_batch, tgt_batch = next(val_iter)
                inp = torch.from_numpy(in_batch).long().to(self.device)
                tgt = torch.from_numpy(tgt_batch).long().to(self.device)
                _, l, ce, fe = self.model(inp, targets=tgt, fe_weight=self.config.fe_weight)
                total_loss += l.item()
                total_ce += ce.item()
                total_fe += fe.item()

        avg_loss = total_loss / steps
        avg_ce = total_ce / steps
        avg_fe = total_fe / steps
        ppl = math.exp(min(20.0, avg_ce))
        bpb = avg_ce / (math.log(2) * max(0.1, self.config.bytes_per_token))
        return avg_loss, avg_ce, avg_fe, ppl, bpb

    def train(
        self,
        tokens: Union[List[int], torch.Tensor, Any],
        save_path: Optional[str] = None,
        eval_prompt: Optional[str] = None,
        tokenizer = None,
    ):
        cfg = self.config
        total_steps = cfg.total_steps
        bs, sl = cfg.batch_size, cfg.seq_len
        n_tok = bs * sl

        from .data import PretokenizedDataset, JAXDataIterator
        from .telemetry import TelegramMonitor, MetricsLogger, TrainingControl, setup_emergency_handler

        val_dataset = None
        use_dataset_iter = False
        if hasattr(tokens, "val_dataset"):
            data_iter = tokens
            val_dataset = tokens.val_dataset
            use_dataset_iter = True
        elif isinstance(tokens, PretokenizedDataset):
            data_iter = JAXDataIterator(tokens, batch_size=bs, shuffle=True)
            use_dataset_iter = True
        elif isinstance(tokens, str) and not os.path.exists(tokens) and ("/" in tokens):
            from .streaming import SlidingShardLoader
            loader = SlidingShardLoader(tokens, batch_size=bs, seq_len=sl, hf_token=cfg.hf_token, window_size=cfg.sliding_window_shards)
            data_iter = loader
            val_dataset = loader.val_dataset
            use_dataset_iter = True
        elif isinstance(tokens, str) and (os.path.isdir(tokens) or tokens.endswith((".bin", ".npy"))):
            ds = PretokenizedDataset(tokens, seq_len=sl)
            data_iter = JAXDataIterator(ds, batch_size=bs, shuffle=True)
            use_dataset_iter = True
        elif isinstance(tokens, str):
            import json
            with open(tokens, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            tokens = raw_data.get("tokens", raw_data) if isinstance(raw_data, dict) else raw_data
        elif isinstance(tokens, dict):
            tokens = tokens.get("tokens", [])

        if not use_dataset_iter:
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, dtype=torch.long)
            elif not isinstance(tokens, torch.Tensor):
                import numpy as np
                tokens = torch.from_numpy(np.asarray(tokens)).long()
            tok_len = len(tokens)
            max_start = max(1, tok_len - n_tok - 1)

        start_step = 1
        cum_tok = 0

        # Resume from checkpoint if requested
        if cfg.resume_from_checkpoint:
            resume_path = cfg.resume_from_checkpoint
            if resume_path == "latest":
                resume_path = find_latest_checkpoint(cfg.checkpoint_dir)
                if not resume_path:
                    raise FileNotFoundError(f"No checkpoint found in '{cfg.checkpoint_dir}' to resume from")

            if is_main_process():
                print(f"[Resume] Loading state from '{resume_path}'...")
            t_state = load_training_checkpoint(
                resume_path,
                model=self.model,
                optimizer=self.optimizer if not cfg.use_deepspeed else None,
                device=self.device,
                token=cfg.hf_token,
            )
            saved_step = t_state.get("step", 0)
            start_step = saved_step + 1
            cum_tok = t_state.get("tokens_seen", saved_step * n_tok)
            if "dataset_state" in t_state and hasattr(data_iter, "set_state"):
                data_iter.set_state(t_state["dataset_state"])
                ds_st = t_state["dataset_state"]
                if is_main_process():
                    print(f"[Resume] Dataset restored: {ds_st.get('shard_name')} (shard {ds_st.get('shard_idx')}) | offset {ds_st.get('token_offset', 0):,} tok")
            if is_main_process():
                print(f"[Resume] Resumed from step {saved_step} ({cum_tok:,} tokens seen)")

        if cfg.hf_repo_id and is_main_process():
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
        cur_lr = cfg.lr

        def get_status():
            now = time.time()
            elapsed = now - start_time
            speed = cum_tok / elapsed if elapsed > 0 else 0
            rem_tok = max(0, total_toks - (step - start_step + 1) * n_tok)
            eta_sec = rem_tok / speed if speed > 0 else 0
            free_gb = shutil.disk_usage(cfg.checkpoint_dir)[2] / 1e9 if os.path.exists(cfg.checkpoint_dir) else 0.0
            latest = logger.get_latest()
            ds_st = data_iter.get_state() if hasattr(data_iter, "get_state") else {}
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
                "lr": cur_lr,
                "speed": speed,
                "elapsed_str": f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m",
                "eta_str": f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m",
                "free_disk_gb": free_gb,
                "shard_name": ds_st.get("shard_name", "-"),
                "shard_idx": ds_st.get("shard_idx", 0),
                "shard_offset": ds_st.get("token_offset", 0),
            }

        bot = TelegramMonitor(
            token=cfg.telegram_token,
            chat_id=cfg.telegram_chat_id,
            control=control,
            logger=logger,
            get_status_fn=get_status,
        )

        def emergency_save():
            if is_main_process():
                t_data = {"step": step, "tokens_seen": cum_tok, "emergency": True}
                if hasattr(data_iter, "get_state"):
                    t_data["dataset_state"] = data_iter.get_state()
                save_training_checkpoint(
                    checkpoint_dir=cfg.checkpoint_dir,
                    step=step,
                    model=self.model,
                    optimizer=self.optimizer if not cfg.use_deepspeed else None,
                    training_state=t_data,
                    training_config=cfg,
                    hf_repo_id=cfg.hf_repo_id,
                    hf_private=cfg.hf_private,
                    hf_token=cfg.hf_token,
                )
                if bot:
                    bot.send_message(f"emergency: checkpoint step_{step:07d} saved and uploaded.")
        setup_emergency_handler(emergency_save)

        total_toks = (total_steps - start_step + 1) * n_tok
        if is_main_process():
            print(f"Training pyFERN: steps {start_step}->{total_steps} ({total_toks:,} tokens)")
            if bot:
                bot.send_message(f"training started: steps {start_step}->{total_steps} ({total_toks:,} tokens)")

        for step in range(start_step, total_steps + 1):
            if control.emergency_exit or control.request_stop:
                print(f"[Control] Stopping training at step {step}.")
                break

            if control.override_lr is not None:
                cur_lr = control.override_lr
                control.override_lr = None

            if not cfg.use_deepspeed:
                cur_lr = self.get_lr(step, total_steps)
                for g in self.optimizer.param_groups:
                    g["lr"] = cur_lr
            else:
                cur_lr = self.optimizer.param_groups[0]["lr"]

            if use_dataset_iter:
                in_batch, tgt_batch = next(data_iter)
                inp = torch.from_numpy(in_batch).long()
                tgt = torch.from_numpy(tgt_batch).long()
            else:
                offset = ((step - 1) * n_tok) % max_start
                inp = tokens[offset : offset + n_tok].view(bs, sl)
                tgt = tokens[offset + 1 : offset + n_tok + 1].view(bs, sl)

            tot, ce, fe = self.train_step(inp, tgt)
            cum_tok += n_tok

            # Periodic validation
            trigger_val = (val_dataset is not None) and (control.request_val or (cfg.val_interval_steps and step % cfg.val_interval_steps == 0))
            if trigger_val:
                control.request_val = False
                val_loss, val_ce, val_fe, val_ppl, val_bpb = self.evaluate(val_dataset, val_steps=cfg.val_steps)
                if is_main_process():
                    print(f"Val Step {step} | Loss: {val_loss:.4f} | PPL: {val_ppl:.2f} | BPB: {val_bpb:.3f}")
                    logger.log({
                        "step": step, "tokens": cum_tok,
                        "train_loss": tot, "ce_loss": ce, "fe_loss": fe,
                        "val_loss": val_loss, "val_ce": val_ce, "val_fe": val_fe,
                        "val_ppl": val_ppl, "val_bpb": val_bpb,
                        "lr": cur_lr, "speed": cum_tok / max(0.1, time.time() - start_time), "elapsed": time.time() - start_time,
                    })
                    if bot:
                        bot.send_message(f"val step {step}: loss={val_loss:.4f} ppl={val_ppl:.2f} bpb={val_bpb:.3f}")

            # Checkpoint trigger: steps, time, or manual /save
            now = time.time()
            trigger_step = cfg.save_interval_steps and (step % cfg.save_interval_steps == 0)
            trigger_time = cfg.save_interval_seconds and (now - last_save_time >= cfg.save_interval_seconds)
            trigger_save = control.request_save or trigger_step or trigger_time

            if trigger_save and is_main_process():
                control.request_save = False
                last_save_time = now
                state_dict = {
                    "step": step,
                    "tokens_seen": cum_tok,
                    "total_loss": tot,
                    "ce_loss": ce,
                    "fe_loss": fe,
                    "lr": cur_lr,
                    "timestamp": now,
                }
                if hasattr(data_iter, "get_state"):
                    state_dict["dataset_state"] = data_iter.get_state()
                saved_dir = save_training_checkpoint(
                    checkpoint_dir=cfg.checkpoint_dir,
                    step=step,
                    model=self.model,
                    optimizer=self.optimizer if not cfg.use_deepspeed else None,
                    training_state=state_dict,
                    training_config=cfg,
                    max_to_keep=cfg.max_checkpoints_to_keep,
                    hf_repo_id=cfg.hf_repo_id if cfg.hf_push_on_save else None,
                    hf_private=cfg.hf_private,
                    hf_token=cfg.hf_token,
                )
                print(f"[Checkpoint] Saved step {step} -> {saved_dir}")
                if cfg.hf_repo_id:
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
                    bot.send_message(f"checkpoint saved: step_{step:07d}")

            if is_main_process() and (step % cfg.log_interval == 0 or step == start_step or step == total_steps):
                elapsed = time.time() - start_time
                speed = cum_tok / elapsed if elapsed > 0 else 0
                print(f"Step {step:>4}/{total_steps} ({cum_tok:>6} tok) | CE: {ce:.4f} | FE: {fe:.4f} | Loss: {tot:.4f} | LR: {cur_lr:.2e} | Speed: {speed:.0f} tok/s")
                logger.log({
                    "step": step, "tokens": cum_tok,
                    "train_loss": tot, "ce_loss": ce, "fe_loss": fe,
                    "lr": cur_lr, "speed": speed, "elapsed": elapsed,
                })

        if is_main_process():
            final_dir = save_path or os.path.join(cfg.checkpoint_dir, f"step_{step:07d}")
            final_state = {
                "step": step,
                "tokens_seen": cum_tok,
                "total_loss": tot,
                "ce_loss": ce,
                "fe_loss": fe,
                "lr": cur_lr,
                "timestamp": time.time(),
            }
            if hasattr(data_iter, "get_state"):
                final_state["dataset_state"] = data_iter.get_state()
            save_training_checkpoint(
                checkpoint_dir=cfg.checkpoint_dir,
                step=step,
                model=self.model,
                optimizer=self.optimizer if not cfg.use_deepspeed else None,
                training_state=final_state,
                training_config=cfg,
                max_to_keep=cfg.max_checkpoints_to_keep,
                hf_repo_id=cfg.hf_repo_id,
                hf_private=cfg.hf_private,
                hf_token=cfg.hf_token,
            )
            if save_path:
                save_pretrained(self.model, save_path)
                print(f"[OK] Saved final checkpoint to '{save_path}'")
            if cfg.hf_repo_id:
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
                bot.send_message(f"training completed: step_{step:07d}")
                bot.close()

        if eval_prompt and tokenizer and is_main_process():
            prompt_ids = tokenizer.encode(eval_prompt).ids
            gen_ids = self.model.generate(prompt_ids, max_tokens=25, temperature=0.7)
            print(f"[Generation] \"{tokenizer.decode(gen_ids)}\"")

