import argparse
import sys
import os
import torch

from .config import ModelConfig, TrainingConfig
from .network import FERNModel
from .checkpoint import from_pretrained, save_pretrained
from .tokenizer import load_stock_tokenizer
from .trainer import FERNTrainer
from .kernels import HAS_TRITON

def cli_info():
    print("pyFERN Environment & Accelerator:")
    print(f"  PyTorch:       {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"  CUDA Device:   {torch.cuda.get_device_name(0)}")
    print(f"  Triton Kernel: {'Available' if HAS_TRITON else 'Not installed / CPU fallback active'}")

    try:
        import jax
        print(f"  JAX / TPU:     {jax.__version__} (Devices: {jax.devices()})")
    except ImportError:
        print("  JAX / TPU:     Not installed")

    try:
        import deepspeed
        print(f"  DeepSpeed:     {deepspeed.__version__}")
    except ImportError:
        print("  DeepSpeed:     Not installed")


def cli_generate(args):
    tok_path = args.tokenizer
    if tok_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint) if os.path.isfile(args.checkpoint) else args.checkpoint
        cand = os.path.join(ckpt_dir, "tokenizer.json")
        if os.path.exists(cand):
            tok_path = cand
        elif os.path.exists("tokenizer.json"):
            tok_path = "tokenizer.json"
        else:
            print("Error: No tokenizer specified and 'tokenizer.json' not found. Please provide --tokenizer <path>.")
            sys.exit(1)

    if not os.path.exists(tok_path):
        print(f"Error: Tokenizer file not found: {tok_path}")
        sys.exit(1)

    tokenizer = load_stock_tokenizer(tok_path)
    model = from_pretrained(args.checkpoint, device=args.device)

    prompt_ids = tokenizer.encode(args.prompt).ids
    gen_ids = model.generate(
        prompt_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.rep_penalty,
    )
    generated_text = tokenizer.decode(gen_ids)
    print(f"\n[Prompt]:    {args.prompt}")
    print(f"[Generated]: {generated_text}\n")


def cli_train(args):
    import json
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    with open(args.data, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tokens = raw if isinstance(raw, list) else raw.get("tokens", [])

    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        cfg = ModelConfig.from_json(args.config)
    else:
        cfg = ModelConfig()

    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        total_steps=args.steps,
        lr=args.lr,
        log_interval=args.log_interval,
        device=args.device,
    )

    model = FERNModel(cfg)
    trainer = FERNTrainer(model, config=train_cfg)
    trainer.train(tokens, save_path=args.save_path)


def main():
    parser = argparse.ArgumentParser(prog="pyfern", description="pyFERN")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # info
    subparsers.add_parser("info", help="Check hardware accelerators and library status")

    # generate
    gen_p = subparsers.add_parser("generate", help="Generate text from a trained checkpoint")
    gen_p.add_argument("--checkpoint", required=True, help="Path to checkpoint (.safetensors file or directory)")
    gen_p.add_argument("--prompt", default="Once upon a time", help="Text prompt")
    gen_p.add_argument("--tokenizer", default=None, help="Path to tokenizer.json (default: looks alongside checkpoint or tokenizer.json)")
    gen_p.add_argument("--max-tokens", type=int, default=32, help="Number of tokens to generate")
    gen_p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_p.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff")
    gen_p.add_argument("--rep-penalty", type=float, default=1.1, help="Repetition penalty")
    gen_p.add_argument("--device", default=None, help="Device to run inference on (cuda/cpu)")

    # train
    train_p = subparsers.add_parser("train", help="Train a model on tokenized dataset")
    train_p.add_argument("--data", required=True, help="Path to tokenized JSON file")
    train_p.add_argument("--config", default=None, help="Path to model config JSON file")
    train_p.add_argument("--save-path", default=None, help="Path to save checkpoint (e.g. checkpoints/model.safetensors)")
    train_p.add_argument("--steps", type=int, default=200, help="Total training steps")
    train_p.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_p.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    train_p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_p.add_argument("--log-interval", type=int, default=20, help="Logging step interval")
    train_p.add_argument("--device", default=None, help="Device to run on")

    args = parser.parse_args()
    if args.command == "info":
        cli_info()
    elif args.command == "generate":
        cli_generate(args)
    elif args.command == "train":
        cli_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
