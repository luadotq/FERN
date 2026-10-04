import os
import math
import time
from typing import Optional, List, Union, Tuple
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

    def train(
        self,
        tokens: Union[List[int], torch.Tensor],
        save_path: Optional[str] = None,
        eval_prompt: Optional[str] = None,
        tokenizer = None,
    ):
        cfg = self.config
        total_steps = cfg.total_steps
        bs, sl = cfg.batch_size, cfg.seq_len
        n_tok = bs * sl

        from .data import PretokenizedDataset, JAXDataIterator

        use_dataset_iter = False
        if isinstance(tokens, PretokenizedDataset):
            data_iter = JAXDataIterator(tokens, batch_size=bs, shuffle=True)
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
            if is_main_process():
                print(f"[Resume] Resumed from step {saved_step} ({cum_tok:,} tokens seen)")

        if cfg.hf_repo_id and is_main_process():
            ok, err = check_hf_hub_access(cfg.hf_repo_id, token=cfg.hf_token, private=cfg.hf_private)
            if not ok:
                print(f"[Warning] Hugging Face Hub validation failed for '{cfg.hf_repo_id}': {err}")
                print("[Warning] Disabling HF uploads; continuing training with local checkpoints only.")
                cfg.hf_repo_id = None

        start_time = time.time()
        last_save_time = start_time

        if is_main_process():
            total_toks = (total_steps - start_step + 1) * n_tok
            print(f"Training pyFERN: steps {start_step}->{total_steps} ({total_toks:,} tokens)")

        tot, ce, fe = 0.0, 0.0, 0.0
        cur_lr = cfg.lr

        for step in range(start_step, total_steps + 1):
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

            # Checkpoint trigger: steps or time
            now = time.time()
            trigger_step = cfg.save_interval_steps and (step % cfg.save_interval_steps == 0)
            trigger_time = cfg.save_interval_seconds and (now - last_save_time >= cfg.save_interval_seconds)

            if (trigger_step or trigger_time) and is_main_process():
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

            if is_main_process() and (step % cfg.log_interval == 0 or step == start_step or step == total_steps):
                elapsed = time.time() - start_time
                speed = cum_tok / elapsed if elapsed > 0 else 0
                print(f"Step {step:>4}/{total_steps} ({cum_tok:>6} tok) | CE: {ce:.4f} | FE: {fe:.4f} | Loss: {tot:.4f} | LR: {cur_lr:.2e} | Speed: {speed:.0f} tok/s")

        if is_main_process():
            final_dir = save_path or os.path.join(cfg.checkpoint_dir, f"step_{total_steps:07d}")
            save_training_checkpoint(
                checkpoint_dir=cfg.checkpoint_dir,
                step=total_steps,
                model=self.model,
                optimizer=self.optimizer if not cfg.use_deepspeed else None,
                training_state={
                    "step": total_steps,
                    "tokens_seen": cum_tok,
                    "total_loss": tot,
                    "ce_loss": ce,
                    "fe_loss": fe,
                    "lr": cur_lr,
                    "timestamp": time.time(),
                },
                training_config=cfg,
                max_to_keep=cfg.max_checkpoints_to_keep,
                hf_repo_id=cfg.hf_repo_id,
                hf_private=cfg.hf_private,
                hf_token=cfg.hf_token,
            )
            if save_path:
                save_pretrained(self.model, save_path)
                print(f"[OK] Saved final checkpoint to '{save_path}'")

        if eval_prompt and tokenizer and is_main_process():
            prompt_ids = tokenizer.encode(eval_prompt).ids
            gen_ids = self.model.generate(prompt_ids, max_tokens=25, temperature=0.7)
            print(f"[Generation] \"{tokenizer.decode(gen_ids)}\"")
