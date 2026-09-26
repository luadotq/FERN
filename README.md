# pyFERN

`pyFERN` is a lightweight Python/PyTorch library implementing **FERN 1.0 (Evergreen)**: an autoregressive language model architecture combining hierarchical predictive coding with Multi-Head Vector Linear Attention (VLA).

Instead of an expanding Key-Value (KV) cache, the model maintains context in fixed-size recurrent state matrices. Memory consumption remains strictly constant at $\mathcal{O}(1)$ per step regardless of context length, completely eliminating Out-Of-Memory (OOM) failures during long conversations.

## Installation

```bash
# Core package (PyTorch CPU / CUDA)
pip install .

# Optional hardware backends
pip install .[triton]     # Fused GPU linear attention kernels
pip install .[jax]        # Google TPU / JAX functional backend
pip install .[deepspeed]  # Multi-GPU ZeRO distributed training
pip install .[all]        # All extras
```

For editable local development:
```bash
pip install -e .
```

## Quickstart

### 1. Generating Text (Inference)

```python
from pyFERN import from_pretrained, load_tokenizer

# Load tokenizer and model weights
tokenizer = load_tokenizer("tokenizer.json")
model = from_pretrained("checkpoints/model.safetensors")

# Autoregressive generation with constant O(1) memory
prompt_ids = tokenizer.encode("Once upon a time").ids
gen_ids = model.generate(
    prompt_ids,
    max_tokens=64,
    temperature=0.7,
    top_k=50,
)
print(tokenizer.decode(gen_ids))
```

### 2. Training

```python
from pyFERN import ModelConfig, TrainingConfig, FERNModel, FERNTrainer

# Initialize model and trainer
model = FERNModel(ModelConfig(vocab_size=1000))
trainer = FERNTrainer(model, TrainingConfig(lr=1e-3, batch_size=4, seq_len=64))

# Train over streaming token data
trainer.train(
    tokens="data/dataset.json",
    save_path="checkpoints/my_model.safetensors",
)
```

---

## Command Line Interface (CLI)

`pyFERN` includes the `pyfern` command line utility:

```bash
# Verify system accelerators and environment status
pyfern info

# Generate text from a trained checkpoint
pyfern generate \
  --checkpoint checkpoints/model.safetensors \
  --tokenizer tokenizer.json \
  --prompt "Once upon a time" \
  --max-tokens 32 \
  --temperature 0.7

# Train on a tokenized dataset
pyfern train \
  --data data/dataset.json \
  --config config.json \
  --steps 200 \
  --batch-size 4 \
  --seq-len 64 \
  --save-path checkpoints/model.safetensors
```

## Documentation

- **[Full API Reference](API.md)**: Detailed class, method, and parameter documentation for all `pyFERN` modules.

---

## License

MIT License.
