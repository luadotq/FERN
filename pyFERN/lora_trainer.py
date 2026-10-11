import os
import math
import time
import shutil
from typing import Optional, List, Union, Tuple, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .network import FERNModel
from .data import SFTDataset, sft_collate_fn
from .lora import inject_lora, save_adapter, merge_and_save
from .checkpoint import push_to_hub
from .telemetry import TelegramMonitor, MetricsLogger, TrainingControl, setup_emergency_handler
from .distributed import setup_distributed_engine, is_main_process


class FERNLoRATrainer:
    def __init__(
        self,
        model: FERNModel,
        config: Optional[TrainingConfig] = None,
        tokenizer: Optional[Any] = None,
        **kwargs,
    ):
        self.config = config or TrainingConfig(**kwargs)
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer

        # inject LoRA adapter into model
        self.trainable_params, self.total_params = inject_lora(
            model,
            r=self.config.lora_r,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )
        self.model = model.to(self.device)

        if getattr(self.config, "gradient_checkpointing", True):
            self.model.config.gradient_checkpointing = True

        if self.config.use_deepspeed:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self.engine, self.optimizer, self.ds_scheduler = setup_distributed_engine(
                self.model, self.config, model_parameters=trainable
            )
            self.scaler = None
        else:
            self.engine = None
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable,
                lr=self.config.lr,
                betas=(self.config.adam_b1, self.config.adam_b2),
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay,
            )
            use_fp16 = self.config.mixed_precision in ("fp16", "bf16") and ("cuda" in self.device)
            self.scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

        self.crit = nn.CrossEntropyLoss(ignore_index=-100)

        if is_main_process():
            pct = 100 * self.trainable_params / max(1, self.total_params)
            print(f"LoRA Initialized: {self.trainable_params:,} / {self.total_params:,} trainable params ({pct:.2f}%)")

    def get_lr(self, step: int, total_steps: int) -> float:
        w = self.config.warmup_steps
        min_lr, lr = self.config.min_lr, self.config.lr
        if step < w:
            return min_lr + (lr - min_lr) * (step / max(1, w))
        prog = (step - w) / max(1, total_steps - w)
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * prog))

    def train_step(self, x: torch.Tensor, y: torch.Tensor, accum: int = 1) -> Tuple[float, float, float]:
        inp = x.to(self.device)
        tgt = y.to(self.device)

        if self.config.use_deepspeed:
            self.engine.train()
            active_m = self.engine.module if hasattr(self.engine, "module") else self.engine
            logits, fe = active_m.forward_parallel(inp)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tgt[:, 1:].contiguous()
            ce = self.crit(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
            loss = ce + self.config.fe_weight * fe
            self.engine.backward(loss)
            self.engine.step()
            return loss.item(), ce.item(), fe.item()

        use_fp16 = self.config.mixed_precision in ("fp16", "bf16") and ("cuda" in self.device)
        with torch.amp.autocast('cuda', enabled=use_fp16):
            logits, fe = self.model.forward_parallel(inp)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tgt[:, 1:].contiguous()
            ce = self.crit(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
            loss = (ce + self.config.fe_weight * fe) / accum

        self.scaler.scale(loss).backward()
        return loss.item() * accum, ce.item(), fe.item()

    def evaluate(self, val_loader: DataLoader, val_steps: int = 50) -> Tuple[float, float, float, float, float]:
        active_m = self.engine.module if (self.engine and hasattr(self.engine, "module")) else self.model
        active_m.eval()
        tot_loss, tot_ce, tot_fe = 0.0, 0.0, 0.0
        steps = min(val_steps, len(val_loader))
        use_fp16 = self.config.mixed_precision in ("fp16", "bf16") and ("cuda" in self.device) and (not self.config.use_deepspeed)

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= steps:
                    break
                inp, tgt = x.to(self.device), y.to(self.device)
                with torch.amp.autocast('cuda', enabled=use_fp16):
                    logits, fe = active_m.forward_parallel(inp)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = tgt[:, 1:].contiguous()
                    ce = self.crit(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
                    l = ce + self.config.fe_weight * fe
                tot_loss += l.item()
                tot_ce += ce.item()
                tot_fe += fe.item()

        steps = max(1, steps)
        avg_loss = tot_loss / steps
        avg_ce = tot_ce / steps
        avg_fe = tot_fe / steps
        ppl = math.exp(min(20.0, avg_ce))
        bpb = avg_ce / (math.log(2) * max(0.1, self.config.bytes_per_token))
        return avg_loss, avg_ce, avg_fe, ppl, bpb

    def save_adapter_checkpoint(self, step: int, is_final: bool = False) -> str:
        tag = f"adapter_step_{step:07d}" if not is_final else "final_adapter"
        save_dir = os.path.join(self.config.checkpoint_dir, tag)
        save_adapter(self.model, save_dir, r=self.config.lora_r, alpha=self.config.lora_alpha)

        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(save_dir)

        # rotate checkpoints
        if not is_final and self.config.max_checkpoints_to_keep > 0:
            ckpts = sorted(
                [d for d in os.listdir(self.config.checkpoint_dir) if d.startswith("adapter_step_")],
                key=lambda x: int(x.split("_")[-1]),
            )
            while len(ckpts) > self.config.max_checkpoints_to_keep:
                shutil.rmtree(os.path.join(self.config.checkpoint_dir, ckpts.pop(0)), ignore_errors=True)

        if self.config.hf_repo_id and (self.config.hf_push_on_save or is_final):
            push_to_hub(
                folder_path=save_dir,
                repo_id=self.config.hf_repo_id,
                token=self.config.hf_token,
                private=self.config.hf_private,
                commit_message=f"Upload {tag}",
            )
        return save_dir

    def merge_and_export(self, out_dir: Optional[str] = None) -> str:
        merged_path = out_dir or os.path.join(self.config.checkpoint_dir, "merged_model")
        merge_and_save(self.model, merged_path)
        self.model.config.to_json(os.path.join(merged_path, "config.json"))

        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(merged_path)

        if is_main_process():
            print(f"[LoRA] Model merged and saved to '{merged_path}'")

        if self.config.hf_repo_id:
            hub_repo = f"{self.config.hf_repo_id}-merged"
            push_to_hub(
                folder_path=merged_path,
                repo_id=hub_repo,
                token=self.config.hf_token,
                private=self.config.hf_private,
                commit_message="Upload merged pyFERN model",
            )
        return merged_path

    def train(
        self,
        dataset: Union[str, Any],
        epochs: int = 1,
        val_dataset: Optional[Union[str, Any]] = None,
    ):
        cfg = self.config
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        if isinstance(dataset, str):
            ds = SFTDataset(dataset, tok=self.tokenizer, max_len=cfg.max_seq_len)
        else:
            ds = dataset

        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=sft_collate_fn,
        )

        val_loader = None
        if val_dataset is not None:
            if isinstance(val_dataset, str):
                v_ds = SFTDataset(val_dataset, tok=self.tokenizer, max_len=cfg.max_seq_len)
            else:
                v_ds = val_dataset
            val_loader = DataLoader(v_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=sft_collate_fn)

        accum = max(1, cfg.grad_accum_steps)
        total_steps = (len(loader) * epochs) // accum
        control = TrainingControl()
        logger = MetricsLogger(cfg.checkpoint_dir)

        start_time = time.time()
        last_save_time = start_time
        cum_tok = 0
        step = 0
        cur_lr = cfg.lr
        tot, ce, fe = 0.0, 0.0, 0.0

        def get_status():
            now = time.time()
            elapsed = now - start_time
            speed = cum_tok / max(0.1, elapsed)
            rem_steps = max(0, total_steps - step)
            eta_sec = (rem_steps / max(1, step)) * elapsed if step > 0 else 0
            free_gb = shutil.disk_usage(cfg.checkpoint_dir)[2] / 1e9 if os.path.exists(cfg.checkpoint_dir) else 0.0
            latest = logger.get_latest()
            return {
                "step": step,
                "total_steps": total_steps,
                "pct": (step / max(1, total_steps)) * 100,
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
                save_dir = self.save_adapter_checkpoint(step=step, is_final=False)
                if bot:
                    bot.send_message(f"emergency: LoRA checkpoint saved to {save_dir}")
        setup_emergency_handler(emergency_save)

        if is_main_process():
            print(f"Starting LoRA training: {epochs} epoch(s), {total_steps} steps")
            if bot:
                bot.send_message(f"LoRA training started: {epochs} epochs, {total_steps} steps, r={cfg.lora_r}")

        self.model.train()
        for ep in range(epochs):
            for it, (x, y) in enumerate(loader):
                if control.emergency_exit or control.request_stop:
                    print(f"[Control] Stopping training at step {step}.")
                    break

                if control.override_lr is not None:
                    cur_lr = control.override_lr
                    control.override_lr = None

                tot, ce, fe = self.train_step(x, y, accum=accum)
                cum_tok += x.numel()

                if not cfg.use_deepspeed:
                    if (it + 1) % accum == 0 or (it + 1) == len(loader):
                        self.scaler.unscale_(self.optimizer)
                        if cfg.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        step += 1

                        cur_lr = self.get_lr(step, total_steps)
                        for g in self.optimizer.param_groups:
                            g["lr"] = cur_lr
                else:
                    step += 1
                    cur_lr = self.engine.get_lr()[0] if hasattr(self.engine, "get_lr") else cur_lr

                if step % cfg.log_interval == 0 and is_main_process():
                    spd = cum_tok / max(0.1, time.time() - start_time)
                    print(f"Epoch {ep+1}/{epochs} | Step {step}/{total_steps} | CE: {ce:.4f} | FE: {fe:.4f} | LR: {cur_lr:.2e} | Speed: {spd:.0f} tok/s")
                    logger.log({
                        "step": step, "tokens": cum_tok, "train_loss": tot, "ce_loss": ce, "fe_loss": fe,
                        "lr": cur_lr, "speed": spd, "elapsed": time.time() - start_time,
                    })

                    # Periodic validation
                    trigger_val = (val_loader is not None) and (control.request_val or (cfg.val_interval_steps and step % cfg.val_interval_steps == 0))
                    if trigger_val:
                        control.request_val = False
                        v_loss, v_ce, v_fe, v_ppl, v_bpb = self.evaluate(val_loader, val_steps=cfg.val_steps)
                        self.model.train()
                        if is_main_process():
                            print(f"[Val] Step {step} | Loss: {v_loss:.4f} | PPL: {v_ppl:.2f} | BPB: {v_bpb:.3f}")
                            logger.log({
                                "step": step, "tokens": cum_tok, "val_loss": v_loss, "val_ce": v_ce,
                                "val_fe": v_fe, "val_ppl": v_ppl, "val_bpb": v_bpb,
                            })
                            if bot:
                                bot.send_message(f"val step {step}: loss={v_loss:.4f} ppl={v_ppl:.2f} bpb={v_bpb:.3f}")

                    # Checkpoint triggers
                    now = time.time()
                    trigger_step = cfg.save_interval_steps and (step % cfg.save_interval_steps == 0)
                    trigger_time = cfg.save_interval_seconds and (now - last_save_time >= cfg.save_interval_seconds)
                    trigger_save = control.request_save or trigger_step or trigger_time

                    if trigger_save and is_main_process():
                        control.request_save = False
                        last_save_time = now
                        ckpt_dir = self.save_adapter_checkpoint(step=step)
                        print(f"[Checkpoint] Saved adapter step {step} -> '{ckpt_dir}'")
                        if bot:
                            bot.send_message(f"adapter step_{step:07d} saved and uploaded.")

            if control.emergency_exit or control.request_stop:
                break

        # Final save and optional merge
        if is_main_process():
            final_adapter = self.save_adapter_checkpoint(step=step, is_final=True)
            print(f"[LoRA] Final adapter saved to '{final_adapter}'")
            if cfg.lora_merge_on_end:
                merged_path = self.merge_and_export()
                if bot:
                    bot.send_message(f"training complete: merged model exported to {merged_path}")
            elif bot:
                bot.send_message(f"training complete: adapter saved to {final_adapter}")
