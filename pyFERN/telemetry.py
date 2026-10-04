import os
import sys
import time
import math
import json
import csv
import signal
import uuid
import shutil
import threading
import urllib.request
import urllib.parse
from typing import Optional, Dict, Any, List

class TrainingControl:
    def __init__(self):
        self.request_val: bool = False
        self.request_save: bool = False
        self.request_stop: bool = False
        self.override_lr: Optional[float] = None
        self.emergency_exit: bool = False

class MetricsLogger:
    def __init__(self, log_dir: str = "checkpoints", csv_name: str = "metrics.csv"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, csv_name)
        self.jsonl_path = os.path.join(log_dir, "metrics.jsonl")

        self.history: List[Dict[str, Any]] = []
        self.fieldnames = [
            "step", "tokens", "train_loss", "ce_loss", "fe_loss",
            "val_loss", "val_ce", "val_fe", "val_ppl", "val_bpb",
            "lr", "speed", "elapsed", "timestamp"
        ]

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, record: Dict[str, Any]):
        record.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.history.append(record)

        clean_rec = {k: record.get(k, "") for k in self.fieldnames}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(clean_rec)

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get_latest(self) -> Dict[str, Any]:
        return self.history[-1] if self.history else {}

    def generate_plots(self, output_path: str = "/tmp/fern_metrics.png") -> Optional[str]:
        if not self.history:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        steps = [r["step"] for r in self.history if "step" in r]
        if not steps:
            return None

        train_losses = [r.get("train_loss") for r in self.history]
        val_steps = [r["step"] for r in self.history if r.get("val_loss")]
        val_losses = [r["val_loss"] for r in self.history if r.get("val_loss")]
        val_ppls = [r["val_ppl"] for r in self.history if r.get("val_ppl")]
        val_bpbs = [r["val_bpb"] for r in self.history if r.get("val_bpb")]
        fe_losses = [r.get("fe_loss", 0.0) for r in self.history]
        lrs = [r.get("lr", 0.0) for r in self.history]

        fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
        plt.subplots_adjust(hspace=0.25, wspace=0.25)

        # Plot 1: Train & Val Loss
        axs[0, 0].plot(steps, train_losses, label="train", color="#2563eb", lw=1.5)
        if val_steps and val_losses:
            axs[0, 0].plot(val_steps, val_losses, "o-", label="val", color="#dc2626", lw=1.5, ms=4)
        axs[0, 0].set_title("Cross-Entropy Loss", fontsize=11)
        axs[0, 0].set_xlabel("Step", fontsize=9)
        axs[0, 0].legend(frameon=True, fontsize=8)
        axs[0, 0].grid(True, linestyle="--", alpha=0.5)

        # Plot 2: Val PPL and BPB
        if val_steps and val_ppls:
            ax2_1 = axs[0, 1]
            ax2_2 = ax2_1.twinx()
            ax2_1.plot(val_steps, val_ppls, "s-", color="#7c3aed", lw=1.5, ms=4, label="PPL")
            if val_bpbs:
                ax2_2.plot(val_steps, val_bpbs, "^-", color="#059669", lw=1.5, ms=4, label="BPB")
            ax2_1.set_title("Validation Quality", fontsize=11)
            ax2_1.set_xlabel("Step", fontsize=9)
            ax2_1.set_ylabel("PPL", color="#7c3aed", fontsize=9)
            ax2_2.set_ylabel("BPB", color="#059669", fontsize=9)
            ax2_1.grid(True, linestyle="--", alpha=0.5)
        else:
            axs[0, 1].text(0.5, 0.5, "no validation data yet", ha="center", va="center", color="#6b7280")
            axs[0, 1].set_title("Validation Quality", fontsize=11)

        # Plot 3: Predictive Coding Free Energy
        axs[1, 0].plot(steps, fe_losses, color="#ea580c", lw=1.5)
        axs[1, 0].set_title("Free Energy Loss", fontsize=11)
        axs[1, 0].set_xlabel("Step", fontsize=9)
        axs[1, 0].grid(True, linestyle="--", alpha=0.5)

        # Plot 4: Learning Rate
        axs[1, 1].plot(steps, lrs, color="#0284c7", lw=1.5)
        axs[1, 1].set_title("Learning Rate", fontsize=11)
        axs[1, 1].set_xlabel("Step", fontsize=9)
        axs[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[1, 1].grid(True, linestyle="--", alpha=0.5)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

class TelegramMonitor:
    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        control: Optional[TrainingControl] = None,
        logger: Optional[MetricsLogger] = None,
        get_status_fn = None,
    ):
        self.token = token
        self.chat_id = str(chat_id).strip() if chat_id else None
        self.control = control or TrainingControl()
        self.logger = logger
        self.get_status_fn = get_status_fn
        self.stopped = threading.Event()
        self.thread = None

        if self.token and self.chat_id:
            self.thread = threading.Thread(target=self._poll_loop, daemon=True)
            self.thread.start()

    def _api_call(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[dict]:
        if not self.token:
            return None
        url = f"https://api.telegram.org/bot{self.token}/{endpoint}"
        try:
            if params:
                data = urllib.parse.urlencode(params).encode("utf-8")
                req = urllib.request.Request(url, data=data)
            else:
                req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def send_message(self, text: str) -> bool:
        if not self.token or not self.chat_id:
            return False
        res = self._api_call("sendMessage", {"chat_id": self.chat_id, "text": text})
        return bool(res and res.get("ok"))

    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        if not self.token or not self.chat_id or not os.path.exists(photo_path):
            return False
        try:
            boundary = uuid.uuid4().hex
            url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
            parts = [
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{self.chat_id}\r\n".encode("utf-8")
            ]
            if caption:
                parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode("utf-8"))
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"plot.png\"\r\nContent-Type: image/png\r\n\r\n".encode("utf-8"))
            with open(photo_path, "rb") as f:
                parts.append(f.read())
            parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))

            body = b"".join(parts)
            req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return bool(res and res.get("ok"))
        except Exception:
            return False

    def _poll_loop(self):
        offset = 0
        while not self.stopped.is_set():
            res = self._api_call("getUpdates", {"offset": offset, "timeout": 10})
            if not res or not res.get("ok"):
                time.sleep(3.0)
                continue

            updates = res.get("result", [])
            for u in updates:
                offset = max(offset, u["update_id"] + 1)
                msg = u.get("message", {})
                chat = str(msg.get("chat", {}).get("id", ""))
                text = msg.get("text", "").strip()

                if chat != self.chat_id or not text:
                    continue

                self._handle_command(text)

    def _handle_command(self, text: str):
        cmd = text.split()[0].lower()
        if cmd == "/status":
            st = self.get_status_fn() if self.get_status_fn else {}
            shard_info = ""
            if st.get("shard_name") and st.get("shard_name") != "-":
                shard_info = f"shard: {st.get('shard_name')} (idx {st.get('shard_idx', 0)}) | offset: {st.get('shard_offset', 0):,} tok\n"
            msg = (
                f"step: {st.get('step', 0)}/{st.get('total_steps', 0)} ({st.get('pct', 0.0):.1f}%)\n"
                f"tokens: {st.get('tokens_seen', 0):,}\n"
                f"{shard_info}"
                f"train_loss: {st.get('loss', 0.0):.4f} (ce: {st.get('ce', 0.0):.4f}, fe: {st.get('fe', 0.0):.4f})\n"
                f"val_loss: {st.get('val_loss', '-')} | ppl: {st.get('val_ppl', '-')} | bpb: {st.get('val_bpb', '-')}\n"
                f"lr: {st.get('lr', 0.0):.2e} | speed: {st.get('speed', 0.0):,.0f} tok/s\n"
                f"elapsed: {st.get('elapsed_str', '-')} | eta: {st.get('eta_str', '-')}\n"
                f"free_disk: {st.get('free_disk_gb', 0.0):.1f} gb"
            )
            self.send_message(msg)

        elif cmd == "/plot":
            if self.logger:
                plot_file = self.logger.generate_plots("/tmp/fern_plot.png")
                if plot_file:
                    self.send_photo(plot_file, caption="training curves")
                else:
                    self.send_message("plot generation unavailable (no data or matplotlib missing)")
            else:
                self.send_message("metrics logger not attached")

        elif cmd == "/val":
            self.control.request_val = True
            self.send_message("validation requested for next step.")

        elif cmd == "/save":
            self.control.request_save = True
            self.send_message("checkpoint save and upload requested.")

        elif cmd == "/stop":
            self.control.request_stop = True
            self.send_message("stopping training. saving final checkpoint.")

        elif cmd == "/lr":
            parts = text.split()
            if len(parts) > 1:
                try:
                    val = float(parts[1])
                    self.control.override_lr = val
                    self.send_message(f"learning rate updated to {val:.2e}")
                except ValueError:
                    self.send_message("invalid lr format. use: /lr 1e-4")
            else:
                self.send_message("specify lr. usage: /lr 1e-4")

        elif cmd == "/help":
            help_text = (
                "available commands:\n"
                "/status - current step, loss, speed, eta and disk\n"
                "/plot   - loss, ppl, bpb and lr curves\n"
                "/val    - trigger validation step\n"
                "/save   - save and upload checkpoint\n"
                "/stop   - stop training and save\n"
                "/lr <x> - set learning rate"
            )
            self.send_message(help_text)

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=1.0)

def setup_emergency_handler(callback):
    def _handler(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[Signal] Intercepted {sig_name}. Triggering emergency checkpoint save...")
        try:
            callback()
        except Exception as e:
            print(f"[Signal] Error during emergency save: {e}")
        finally:
            sys.exit(0)

    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass
