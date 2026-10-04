import argparse
import sys
import os
import torch

from .config import ModelConfig, TrainingConfig
from .network import FERNModel
from .checkpoint import from_pretrained, save_pretrained, resolve_checkpoint_dir
from .tokenizer import load_tokenizer
from .trainer import FERNTrainer
from .data import PretokenizedDataset, prepare_pretokenized_dataset
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
        import transformers
        print(f"  Transformers:  {transformers.__version__}")
    except ImportError:
        print("  Transformers:  Not installed")

    try:
        import huggingface_hub
        print(f"  HF Hub:        {huggingface_hub.__version__}")
    except ImportError:
        print("  HF Hub:        Not installed")


def cli_generate(args):
    tok_path = args.tokenizer
    if tok_path is None:
        try:
            ckpt_dir = resolve_checkpoint_dir(args.checkpoint, token=args.token)
            cand = os.path.join(ckpt_dir, "tokenizer.json")
            if os.path.exists(cand):
                tok_path = cand
        except Exception:
            pass

    if tok_path is None:
        if os.path.exists("tokenizer.json"):
            tok_path = "tokenizer.json"
        else:
            print("Error: No tokenizer specified and 'tokenizer.json' not found. Please provide --tokenizer <path_or_hf_id>.")
            sys.exit(1)

    tokenizer = load_tokenizer(tok_path)
    model = from_pretrained(args.checkpoint, device=args.device, token=args.token)

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


def cli_tokenize(args):
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Loaded tokenizer '{args.tokenizer}' (Vocab size: {tokenizer.vocab_size:,})")
    print(f"Tokenizing data from '{args.input}' -> '{args.output_dir}' (dtype: {args.dtype})...")

    meta = prepare_pretokenized_dataset(
        input_files=args.input,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        dtype=args.dtype,
    )
    print(f"[OK] Done! Prepared {meta['total_tokens']:,} tokens across {meta['shards']} shard(s).")
    print(f"Metadata saved to '{os.path.join(args.output_dir, 'metadata.json')}'.")


def cli_train(args):
    if not os.path.exists(args.data):
        print(f"Error: Data path not found: {args.data}")
        sys.exit(1)

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
        save_interval_steps=args.save_interval_steps,
        save_interval_seconds=args.save_interval_seconds,
        checkpoint_dir=args.checkpoint_dir,
        max_checkpoints_to_keep=args.max_checkpoints,
        resume_from_checkpoint=args.resume,
        hf_repo_id=args.hf_repo_id,
        hf_private=args.hf_private,
        hf_token=args.hf_token,
        hf_push_on_save=args.hf_push_on_save,
    )

    model = FERNModel(cfg)
    trainer = FERNTrainer(model, config=train_cfg)
    trainer.train(args.data, save_path=args.save_path)


def cli_train_jax(args):
    try:
        import jax
        import jax.numpy as jnp
        from .jax.model import JAXFERNModel
        from .jax.trainer import JAXFERNTrainer
    except ImportError:
        print("Error: JAX is not installed. Please install JAX and Flax for TPU training.")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"Error: Data path not found: {args.data}")
        sys.exit(1)

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
        save_interval_steps=args.save_interval_steps,
        save_interval_seconds=args.save_interval_seconds,
        checkpoint_dir=args.checkpoint_dir,
        max_checkpoints_to_keep=args.max_checkpoints,
        resume_from_checkpoint=args.resume,
        hf_repo_id=args.hf_repo_id,
        hf_private=args.hf_private,
        hf_token=args.hf_token,
        hf_push_on_save=args.hf_push_on_save,
    )

    model = JAXFERNModel(config=cfg)
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros((args.batch_size, args.seq_len), dtype=jnp.int32)
    params = model.init(key, dummy_input, method=model.forward_parallel)

    trainer = JAXFERNTrainer(model=model, params=params, config=train_cfg)
    trainer.train(args.data, save_path=args.save_path)


def add_training_checkpoint_args(p):
    p.add_argument("--save-interval-steps", type=int, default=None, help="Save checkpoint every N steps")
    p.add_argument("--save-interval-seconds", type=int, default=None, help="Save checkpoint every N seconds")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for periodic checkpoints")
    p.add_argument("--max-checkpoints", type=int, default=3, help="Max recent checkpoints to keep")
    p.add_argument("--resume", default=None, help="Resume training from path or 'latest'")
    p.add_argument("--hf-repo-id", default=None, help="Hugging Face repo ID (e.g. user/repo)")
    p.add_argument("--hf-private", dest="hf_private", action="store_true", default=True, help="Set HF repo to private")
    p.add_argument("--hf-public", dest="hf_private", action="store_false", help="Set HF repo to public")
    p.add_argument("--hf-token", default=None, help="Hugging Face API token")
    p.add_argument("--hf-push-on-save", action="store_true", help="Push each periodic checkpoint to HF Hub")


def main():
    parser = argparse.ArgumentParser(prog="pyfern", description="pyFERN CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    subparsers.add_parser("info", help="Check hardware accelerators and library status")

    # tokenize
    tok_p = subparsers.add_parser("tokenize", aliases=["prepare-data"], help="Pre-tokenize text/JSON into binary memory-mapped shards")
    tok_p.add_argument("--tokenizer", required=True, help="HF Hub model ID, directory, or tokenizer.json")
    tok_p.add_argument("--input", required=True, help="Path to text file, JSON/JSONL, or directory")
    tok_p.add_argument("--output-dir", required=True, help="Directory to save .bin files and metadata.json")
    tok_p.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16", help="Token integer dtype")
    tok_p.add_argument("--seq-len", type=int, default=2048, help="Sequence length for sample boundaries")

    # generate
    gen_p = subparsers.add_parser("generate", help="Generate text from a trained checkpoint")
    gen_p.add_argument("--checkpoint", required=True, help="Local path or HF repo ID (e.g. username/fern-250m)")
    gen_p.add_argument("--prompt", default="Once upon a time", help="Text prompt")
    gen_p.add_argument("--tokenizer", default=None, help="HF model ID or path to tokenizer.json")
    gen_p.add_argument("--token", default=None, help="Hugging Face token for private repos")
    gen_p.add_argument("--max-tokens", type=int, default=32, help="Number of tokens to generate")
    gen_p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_p.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff")
    gen_p.add_argument("--rep-penalty", type=float, default=1.1, help="Repetition penalty")
    gen_p.add_argument("--device", default=None, help="Device to run inference on (cuda/cpu)")

    # train (PyTorch)
    train_p = subparsers.add_parser("train", help="Train a model with PyTorch")
    train_p.add_argument("--data", required=True, help="Path to pre-tokenized .bin file, directory, or JSON")
    train_p.add_argument("--config", default=None, help="Path to model config JSON file")
    train_p.add_argument("--save-path", default=None, help="Final checkpoint save path")
    train_p.add_argument("--steps", type=int, default=200, help="Total training steps")
    train_p.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_p.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    train_p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_p.add_argument("--log-interval", type=int, default=20, help="Logging step interval")
    train_p.add_argument("--device", default=None, help="Device to run on")
    add_training_checkpoint_args(train_p)

    # train-jax (JAX / TPU)
    jax_p = subparsers.add_parser("train-jax", help="Train a model with JAX / TPU")
    jax_p.add_argument("--data", required=True, help="Path to pre-tokenized .bin file or directory of shards")
    jax_p.add_argument("--config", default=None, help="Path to model config JSON file")
    jax_p.add_argument("--save-path", default=None, help="Final checkpoint save path")
    jax_p.add_argument("--steps", type=int, default=200, help="Total training steps")
    jax_p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    jax_p.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    jax_p.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    jax_p.add_argument("--log-interval", type=int, default=20, help="Logging step interval")
    add_training_checkpoint_args(jax_p)

    args = parser.parse_args()
    if args.command == "info":
        cli_info()
    elif args.command in ("tokenize", "prepare-data"):
        cli_tokenize(args)
    elif args.command == "generate":
        cli_generate(args)
    elif args.command == "train":
        cli_train(args)
    elif args.command == "train-jax":
        cli_train_jax(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
