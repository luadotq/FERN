# FERN v0.3 Autobenchmark Report

Report automatically generated after model training and evaluation.

## Model & Training Specifications

| Parameter | Value |
| --- | --- |
| **Vocab Size** | 20 |
| **d_layers** | [32, 64, 64, 64] |
| **Kappa (κ)** | 0.3 |
| **Alpha (α)** | 0.9 |
| **Max Drive** | 5 |
| **Epsilon Min** | 0.0001 |
| **PRISM LR (f_pred)** | 0.001 |
| **PRISM LR (w_up)** | 0.0003 |
| **PRISM LR (w_gate)** | 0.001 |
| **PRISM LR (I/O)** | 0.0003 |
| **Warmup Steps** | 100 |
| **Epochs** | 300 |
| **Batch Size** | 32 |

## Training Convergence

- **Epochs to reach CE < 0.5**: 98
- **Final Total Loss**: 1.4354
- **Final CE Loss**: 0.2212
- **Final FE Loss**: 2.4283

| Epoch | Total Loss | CE Loss | FE Loss | fe_w |
| --- | --- | --- | --- | --- |
| 1 | 3.7650 | 3.7650 | 6.2081 | 0.000 |
| 50 | 1.8994 | 1.0192 | 3.6347 | 0.242 |
| 100 | 2.2072 | 0.6450 | 3.1252 | 0.500 |
| 150 | 1.9858 | 0.4659 | 3.0399 | 0.500 |
| 200 | 1.6756 | 0.3589 | 2.6334 | 0.500 |
| 250 | 1.4963 | 0.2601 | 2.4725 | 0.500 |
| 300 | 1.4354 | 0.2212 | 2.4283 | 0.500 |

## Baseline Comparison

| Task | FERN (Full) | Random | Majority |
| --- | --- | --- | --- |
| **Pattern (2/2)** | 100.0% | 50.0% | 50.0% |
| **Copy (16/16, D3)** | 93.8% | 6.3% | 25.0% |
| **Secret (4/4)** | 100.0% | 50.0% | 50.0% |

## Ablation Analysis

| Component | Pattern | Copy (D3) | Secret |
| --- | --- | --- | --- |
| **Full FERN** | 100.0% | 93.8% | 100.0% |
| **No FE** | 100.0% | 100.0% | 100.0% |
| **No Precision** | 100.0% | 87.5% | 100.0% |

## Detail Evaluation Results

### 1. Pattern Completion (Accuracy: 100.0%)

| Prompt | Expected Output | Actual Output | Result |
| --- | --- | --- | --- |
| `the cat` | `sat on the mat <eos>` | `sat on the mat <eos>` | **PASS** |
| `the dog` | `ran in the park <eos>` | `ran in the park <eos>` | **PASS** |

### 2. Copy with Delay

- **Delay 1 Accuracy**: 100.0%
- **Delay 3 Accuracy**: 93.8%
- **Delay 5 Accuracy**: 93.8%

#### Detail Logs for Delay 3:
| Prompt | Expected Output | Actual Output | Result |
| --- | --- | --- | --- |
| `a a <pad> <pad> <pad> copy` | `a a` | `a a` | **PASS** |
| `a b <pad> <pad> <pad> copy` | `a b` | `a b` | **PASS** |
| `a c <pad> <pad> <pad> copy` | `a c` | `a c` | **PASS** |
| `a d <pad> <pad> <pad> copy` | `a d` | `a d` | **PASS** |
| `b a <pad> <pad> <pad> copy` | `b a` | `a a` | **FAIL** |
| `b b <pad> <pad> <pad> copy` | `b b` | `b b` | **PASS** |
| `b c <pad> <pad> <pad> copy` | `b c` | `b c` | **PASS** |
| `b d <pad> <pad> <pad> copy` | `b d` | `b d` | **PASS** |
| `c a <pad> <pad> <pad> copy` | `c a` | `c a` | **PASS** |
| `c b <pad> <pad> <pad> copy` | `c b` | `c b` | **PASS** |
| `c c <pad> <pad> <pad> copy` | `c c` | `c c` | **PASS** |
| `c d <pad> <pad> <pad> copy` | `c d` | `c d` | **PASS** |
| `d a <pad> <pad> <pad> copy` | `d a` | `d a` | **PASS** |
| `d b <pad> <pad> <pad> copy` | `d b` | `d b` | **PASS** |
| `d c <pad> <pad> <pad> copy` | `d c` | `d c` | **PASS** |
| `d d <pad> <pad> <pad> copy` | `d d` | `d d` | **PASS** |

### 3. Secret Recall (Accuracy: 100.0%)

| Query Sequence | Expected Secret | Actual Output | Result |
| --- | --- | --- | --- |
| `key_x secret_x key_y secret_y key_x` | `secret_x` | `secret_x` | **PASS** |
| `key_x secret_x key_y secret_y key_y` | `secret_y` | `secret_y` | **PASS** |
| `key_y secret_y key_x secret_x key_x` | `secret_x` | `secret_x` | **PASS** |
| `key_y secret_y key_x secret_x key_y` | `secret_y` | `secret_y` | **PASS** |

> [!NOTE]
> The Gated Euler dynamics in FERN allow the higher-level belief states (layers L1-L3) to behave as a continuous-time memory cell.
> When sensory input is removed (clamped to `<pad>` 0), the gate $g$ shuts down incoming drive, enabling the beliefs to persist
> unchanged over multiple time steps. Once the trigger token (`copy` or the recall query `key`) is presented,
> the gate reopens and reads out the stored values through the decoder projection.

