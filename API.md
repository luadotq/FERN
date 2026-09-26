# pyFERN API Reference

## 1. Configuration (`pyFERN.config`)

### `ModelConfig`

```python
from pyFERN import ModelConfig
```

Dataclass specifying the model architecture and hyperparameters.

#### Constructor Arguments

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `vocab_size` | `int` | `256` | Vocabulary size of the embedding and LM head. |
| `d_layers` | `List[int]` | `[64, 128, 128]` | Dimensions of hierarchical predictive coding layers ($d_0, d_1, \dots$). |
| `d_model` | `Optional[int]` | `None` | Hidden dimension for deep multi-layer transformer-style stacked mode. |
| `num_layers` | `Optional[int]` | `None` | Number of stacked layers (if `d_model` is specified). |
| `d_mem` | `int` | `128` | Total hidden dimension for VLA memory states. |
| `num_heads` | `int` | `4` | Number of attention heads ($d_{head} = d_{mem} / \text{num\_heads}$). |
| `rms_eps` | `float` | `1e-5` | Numerical epsilon for Root Mean Square Normalization. |
| `gamma_min` | `float` | `0.90` | Minimum linear attention decay factor across heads. |
| `gamma_max` | `float` | `0.98` | Maximum linear attention decay factor across heads. |
| `pe_scale` | `float` | `0.1` | Scaling factor for sinusoidal position embeddings. |
| `init_range` | `float` | `0.02` | Standard deviation for Gaussian weight initialization. |
| `fe_weight` | `float` | `0.1` | Default weighting coefficient $\lambda_{FE}$ for Free Energy loss. |
| `pad_token_id`| `int` | `0` | Token ID for padding. |
| `bos_token_id`| `int` | `2` | Token ID for beginning-of-sequence. |
| `eos_token_id`| `int` | `3` | Token ID for end-of-sequence. |

#### Methods

- **`from_json(path: str) -> ModelConfig`**  
  Loads and parses configuration from a JSON file. Filters out unsupported keys automatically.

- **`to_json(path: str) -> None`**  
  Serializes the configuration dataclass to an indented JSON file.

---

### `TrainingConfig`

```python
from pyFERN import TrainingConfig
```

Dataclass specifying training hyperparameters, optimization settings, and hardware options.

#### Constructor Arguments

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `lr` | `float` | `1e-3` | Peak learning rate for AdamW optimizer. |
| `min_lr` | `float` | `5e-5` | Minimum learning rate for cosine schedule. |
| `warmup_steps` | `int` | `10` | Linear warmup steps before cosine decay. |
| `weight_decay` | `float` | `0.01` | Decoupled weight decay coefficient. |
| `grad_clip` | `float` | `1.0` | Maximum gradient $L_2$ norm for clipping (0 to disable). |
| `fe_weight` | `float` | `0.1` | Multiplier $\lambda_{FE}$ for Free Energy loss penalty. |
| `adam_b1` | `float` | `0.9` | Adam $\beta_1$ moment factor. |
| `adam_b2` | `float` | `0.999` | Adam $\beta_2$ moment factor. |
| `adam_eps` | `float` | `1e-8` | Adam denominator epsilon. |
| `batch_size` | `int` | `4` | Number of sequences per batch. |
| `seq_len` | `int` | `64` | Sequence length (chunk size) in tokens. |
| `total_steps` | `int` | `200` | Total optimization steps. |
| `log_interval` | `int` | `20` | Interval in steps for printing progress and loss metrics. |
| `seed` | `int` | `42` | Random seed for reproducibility. |
| `device` | `Optional[str]` | `None` | Target device (`"cpu"`, `"cuda"`, `"cuda:0"`, etc.). Auto-detected if `None`. |
| `mixed_precision` | `str` | `"no"` | Precision mode: `"no"`, `"fp16"`, or `"bf16"`. |
| `use_deepspeed` | `bool` | `False` | Enable DeepSpeed ZeRO distributed engine. |
| `zero_stage` | `int` | `2` | DeepSpeed ZeRO stage (`1`, `2`, or `3`). |

#### Methods

- **`from_json(path: str) -> TrainingConfig`**
- **`to_json(path: str) -> None`**

---

## 2. Model Core (`pyFERN.network`)

### `FERNModel`

```python
from pyFERN import FERNModel
```

#### Methods

#### `__init__(config: Optional[ModelConfig] = None)`
Initializes model weights, predictive coding hierarchy, VLA projection layers, and RMSNorm blocks.

#### `init_state(batch_size: int = 1, device: Optional[str] = None) -> NetworkState`
Allocates and zeros out initial hidden belief vectors and square VLA memory matrices.
- **Returns**: `NetworkState` with shape:
  - In hierarchical mode: $S_t^{(h)} \in \mathbb{R}^{B \times H \times d_{head} \times d_{head}}$.
  - In stacked mode: list of states across all layers.

#### `forward(tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, fe_weight: Optional[float] = None) -> Tuple[...]`
Executes forward pass.
- **Inputs**:
  - `tokens`: Tensor of shape `[batch_size, seq_len]`, dtype `torch.long`.
  - `targets`: Optional target token tensor of shape `[batch_size, seq_len]`.
  - `fe_weight`: Optional override for $\lambda_{FE}$.
- **Returns**:
  - If `targets` is `None`: returns `logits` of shape `[batch_size, seq_len, vocab_size]`.
  - If `targets` is provided: returns tuple `(logits, total_loss, ce_loss, avg_fe)`.

#### `forward_step(token_t: torch.Tensor, state: NetworkState) -> Tuple[torch.Tensor, NetworkState]`
Performs a single-token autoregressive recurrent step with $\mathcal{O}(1)$ time and space complexity.
- **Inputs**:
  - `token_t`: Tensor of shape `[batch_size, 1]` or `[batch_size]`.
  - `state`: Current `NetworkState`.
- **Returns**:
  - `logits`: `[batch_size, 1, vocab_size]` next-token prediction logits.
  - `new_state`: Updated `NetworkState` for step $t+1$.

#### `generate(prompt_tokens: Union[List[int], torch.Tensor], max_tokens: int = 32, temperature: float = 0.7, top_k: int = 50, repetition_penalty: float = 1.1) -> List[int]`
Autoregressively generates token sequence from an initial prompt using nucleus/top-k filtering and temperature scaling.
- **Returns**: `List[int]` of generated token IDs (excluding prompt).

#### `get_state_size_bytes(batch_size: int = 1) -> int`
Calculates the exact memory consumption in bytes of the recurrent state for the current configuration.

---

### `NetworkState`

```python
from pyFERN import NetworkState
```

Dataclass holding recurrent memory representations between steps:
- `mu`: List of hidden activity vectors across predictive coding layers.
- `sigma2`: List of variance estimates.
- `s_vla`: Square attention memory tensor ($B \times H \times d_{head} \times d_{head}$).
- `s_vla_layers`: List of attention memory tensors for multi-layer stacked configurations.

## 3. Attention & Layers (`pyFERN.vla`, `pyFERN.layers`)

### `VectorLinearAttention`

```python
from pyFERN import VectorLinearAttention
```

#### Methods
- **`forward_parallel(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`**  
  Computes parallel prefix-scan attention over a full sequence `[B, S, D]`.
- **`forward_step(x_t: torch.Tensor, s_vla: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`**  
  Updates the square state matrix for a single time step and returns `(output, updated_s_vla, fe_loss)`.

### `RMSNorm` & `rms_norm`

```python
from pyFERN import RMSNorm, rms_norm
```

## 4. Checkpoint Management (`pyFERN.checkpoint`)

### `from_pretrained`

```python
from pyFERN import from_pretrained
```

```python
def from_pretrained(
    checkpoint_path: str,
    config: Optional[Union[str, ModelConfig]] = None,
    device: str = "cpu"
) -> FERNModel:
```

Loads model weights and configuration from disk.

- **`checkpoint_path`**: Can be:
  1. A filename without extension (e.g. `"checkpoints/model"`).
  2. A `.safetensors` file path (e.g. `"checkpoints/model.safetensors"`).
  3. A directory containing `model.safetensors` and `config.json`.
- **`config`**: Optional explicit `ModelConfig` or path to a config JSON. If `None`, automatically resolves `f"{base}.json"` or `config.json` alongside the weights.
- **`device`**: Target device (default: `"cpu"`).

---

### `save_pretrained`

```python
from pyFERN import save_pretrained
```

```python
def save_pretrained(model: FERNModel, save_path: str) -> None:
```

Saves model weights in Safetensors format and model configuration as JSON.

- **`model`**: `FERNModel` instance.
- **`save_path`**: Destination path prefix, `.safetensors` file, or directory.

## 5. Tokenization (`pyFERN.tokenizer`)

### `load_tokenizer`

```python
from pyFERN import load_tokenizer
```

```python
def load_tokenizer(path: str = "tokenizer.json") -> Tokenizer:
```

Loads a Hugging Face `tokenizers.Tokenizer` from a file path or directory.

## 6. Training (`pyFERN.trainer`)

### `FERNTrainer`

```python
from pyFERN import FERNTrainer, TrainingConfig
```

#### Methods

- **`__init__(model: FERNModel, config: Optional[TrainingConfig] = None, **kwargs)`**
- **`train_step(inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float]`**  
  Executes a single forward-backward pass and returns `(total_loss, ce_loss, fe_loss)`.
- **`train(tokens: Union[str, List[int], torch.Tensor], save_path: Optional[str] = None, eval_prompt: Optional[str] = None, tokenizer = None)`**  
  Trains the model over continuous streaming token sequence. `tokens` can be a path to a JSON file, a Python list, a NumPy array, or a PyTorch tensor.

## 7. Hardware & Distributed Backends (`pyFERN.kernels`, `pyFERN.distributed`)

### `triton_vla_forward` & `HAS_TRITON`

```python
from pyFERN.kernels import triton_vla_forward, HAS_TRITON
```


### `setup_distributed_engine`

```python
from pyFERN.distributed import setup_distributed_engine, is_main_process
```


## 8. Command Line Interface (CLI)

The package provides the `pyfern` command:

```bash
# Check hardware acceleration and system capabilities
pyfern info

# Generate text from a checkpoint
pyfern generate \
  --checkpoint checkpoints/model.safetensors \
  --tokenizer tokenizer.json \
  --prompt "Once upon a time" \
  --max-tokens 32 \
  --temperature 0.7 \
  --device cpu

# Train on a tokenized dataset
pyfern train \
  --data data/data.json \
  --config config.json \
  --save-path checkpoints/trained_model.safetensors \
  --steps 200 \
  --batch-size 4 \
  --seq-len 64 \
  --lr 0.001
```
