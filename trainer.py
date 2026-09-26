import math
import time
from typing import Optional, List, Union, Tuple
import torch
import torch.nn as nn

from .config import TrainingConfig
from .network import FERNModel
from .checkpoint import save_pretrained
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

        if isinstance(tokens, str):
            import json
            with open(tokens, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            tokens = raw_data.get("tokens", raw_data) if isinstance(raw_data, dict) else raw_data
        elif isinstance(tokens, dict):
            tokens = tokens.get("tokens", [])

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        elif not isinstance(tokens, torch.Tensor):
            import numpy as np
            tokens = torch.from_numpy(np.asarray(tokens)).long()

        start_time = time.time()
        if is_main_process():
            print(f"Training pyFERN on {total_steps * bs * sl:,} tokens ({total_steps} steps)")

        n_tok = bs * sl
        tok_len = len(tokens)
        max_start = max(1, tok_len - n_tok - 1)

        for step in range(1, total_steps + 1):
            if not cfg.use_deepspeed:
                cur_lr = self.get_lr(step, total_steps)
                for g in self.optimizer.param_groups:
                    g["lr"] = cur_lr
            else:
                cur_lr = self.optimizer.param_groups[0]["lr"]

            offset = ((step - 1) * n_tok) % max_start
            inp = tokens[offset : offset + n_tok].view(bs, sl)
            tgt = tokens[offset + 1 : offset + n_tok + 1].view(bs, sl)

            tot, ce, fe = self.train_step(inp, tgt)

            if is_main_process() and (step % cfg.log_interval == 0 or step == 1 or step == total_steps):
                elapsed = time.time() - start_time
                cum_tok = step * bs * sl
                speed = cum_tok / elapsed if elapsed > 0 else 0
                print(f"Step {step:>3}/{total_steps} ({cum_tok:>5} tok) | CE: {ce:.4f} | FE: {fe:.4f} | Loss: {tot:.4f} | LR: {cur_lr:.2e} | Speed: {speed:.0f} tok/s")

        if save_path and is_main_process():
            save_pretrained(self.model, save_path)
            print(f"[OK] Saved checkpoint to '{save_path}'")

        if eval_prompt and tokenizer and is_main_process():
            prompt_ids = tokenizer.encode(eval_prompt).ids
            gen_ids = self.model.generate(prompt_ids, max_tokens=25, temperature=0.7)
            print(f"[Generation] \"{tokenizer.decode(gen_ids)}\"")
