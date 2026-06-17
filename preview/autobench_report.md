# FERN v0.4 Autobenchmark Report

Report automatically generated after model training and evaluation.

## Model & Training Specifications

| Parameter | Value |
| --- | --- |
| **Vocab Size** | 27 |
| **d_layers** | [64, 128, 128, 128] |
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

- **Epochs to reach CE < 0.5**: 72
- **Final Total Loss**: 2.6572
- **Final CE Loss**: 1.1325
- **Final FE Loss**: 3.0493

| Epoch | Total Loss | CE Loss | FE Loss | fe_w |
| --- | --- | --- | --- | --- |
| 1 | 4.5397 | 4.5397 | 3.8382 | 0.000 |
| 50 | 1.5548 | 0.7312 | 3.4012 | 0.242 |
| 100 | 3.7124 | 1.4495 | 4.5271 | 0.500 |
| 150 | 1.2400 | 0.2226 | 2.0348 | 0.500 |
| 200 | 2.6398 | 1.1568 | 2.9660 | 0.500 |
| 250 | 1.2088 | 0.2819 | 1.8537 | 0.500 |
| 300 | 2.6572 | 1.1325 | 3.0493 | 0.500 |

## Baseline Comparison (Standard)

| Task | FERN (Full) | Random | Majority |
| --- | --- | --- | --- |
| **Pattern (2/2)** | 100.0% | 50.0% | 50.0% |
| **Copy (16/16, D3)** | 25.0% | 6.3% | 25.0% |
| **Secret (4/4)** | 100.0% | 50.0% | 50.0% |

## Baseline Comparison (Big Bench)

| Task | FERN (Full) | Random | Majority |
| --- | --- | --- | --- |
| **Big Pattern (2/2)** | 0.0% | 25.0% | 25.0% |
| **Big Copy (256/256, D10)** | 0.0% | 0.4% | 25.0% |
| **Big Secret (30/30)** | 90.0% | 3.3% | 25.0% |

## Ablation Analysis

| Component | Pattern | Copy (D3) | Secret | Big Pattern | Big Copy (D10) | Big Secret |
| --- | --- | --- | --- | --- | --- | --- |
| **Full FERN** | 100.0% | 25.0% | 100.0% | 0.0% | 0.0% | 90.0% |
| **No FE** | 100.0% | 6.2% | 100.0% | 0.0% | 0.0% | 56.7% |
| **No Precision** | 0.0% | 25.0% | 100.0% | 0.0% | 0.0% | 90.0% |

## Detail Evaluation Results (Standard)

### 1. Pattern Completion (Accuracy: 100.0%)

| Prompt | Expected Output | Actual Output | Result |
| --- | --- | --- | --- |
| `the cat` | `sat on the mat <eos>` | `sat on the mat <eos>` | **PASS** |
| `the dog` | `ran in the park <eos>` | `ran in the park <eos>` | **PASS** |

### 2. Copy with Delay

- **Delay 1 Accuracy**: 31.2%
- **Delay 3 Accuracy**: 25.0%
- **Delay 5 Accuracy**: 25.0%

#### Detail Logs for Delay 3:
| Prompt | Expected Output | Actual Output | Result |
| --- | --- | --- | --- |
| `a a <pad> <pad> <pad> copy` | `a a` | `a a` | **PASS** |
| `a b <pad> <pad> <pad> copy` | `a b` | `a a` | **FAIL** |
| `a c <pad> <pad> <pad> copy` | `a c` | `c a` | **FAIL** |
| `a d <pad> <pad> <pad> copy` | `a d` | `a a` | **FAIL** |
| `b a <pad> <pad> <pad> copy` | `b a` | `a a` | **FAIL** |
| `b b <pad> <pad> <pad> copy` | `b b` | `b b` | **PASS** |
| `b c <pad> <pad> <pad> copy` | `b c` | `c c` | **FAIL** |
| `b d <pad> <pad> <pad> copy` | `b d` | `b a` | **FAIL** |
| `c a <pad> <pad> <pad> copy` | `c a` | `a a` | **FAIL** |
| `c b <pad> <pad> <pad> copy` | `c b` | `c b` | **PASS** |
| `c c <pad> <pad> <pad> copy` | `c c` | `c c` | **PASS** |
| `c d <pad> <pad> <pad> copy` | `c d` | `c <pad>` | **FAIL** |
| `d a <pad> <pad> <pad> copy` | `d a` | `a a` | **FAIL** |
| `d b <pad> <pad> <pad> copy` | `d b` | `b a` | **FAIL** |
| `d c <pad> <pad> <pad> copy` | `d c` | `c a` | **FAIL** |
| `d d <pad> <pad> <pad> copy` | `d d` | `d b` | **FAIL** |

### 3. Secret Recall (Accuracy: 100.0%)

| Query Sequence | Expected Secret | Actual Output | Result |
| --- | --- | --- | --- |
| `key_x secret_x key_y secret_y key_x` | `secret_x` | `secret_x` | **PASS** |
| `key_x secret_x key_y secret_y key_y` | `secret_y` | `secret_y` | **PASS** |
| `key_y secret_y key_x secret_x key_x` | `secret_x` | `secret_x` | **PASS** |
| `key_y secret_y key_x secret_x key_y` | `secret_y` | `secret_y` | **PASS** |

## Detail Evaluation Results (Big Bench)

### 4. Big Pattern Completion (Accuracy: 0.0%)

| Prompt | Expected Output | Actual Output | Result |
| --- | --- | --- | --- |
| `the cat sat` | `on the mat dog ran in the park cat <eos>` | `on the mat <eos>` | **FAIL** |
| `the dog ran` | `in the park cat sat on the mat dog <eos>` | `in the park <eos>` | **FAIL** |

### 5. Big Copy with Delay

- **Delay 5 Accuracy**: 0.4%
- **Delay 10 Accuracy**: 0.0%
- **Delay 15 Accuracy**: 0.0%

#### Detail Logs for Delay 10 (First 16 tests):
| Prompt | Expected Output | Actual Output | Result |
| --- | --- | --- | --- |
| `a a a a <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a a a` | `a a <eos>` | **FAIL** |
| `a a a b <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a a b` | `<eos>` | **FAIL** |
| `a a a c <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a a c` | `<eos>` | **FAIL** |
| `a a a d <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a a d` | `a a a a` | **FAIL** |
| `a a b a <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a b a` | `<eos>` | **FAIL** |
| `a a b b <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a b b` | `<eos>` | **FAIL** |
| `a a b c <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a b c` | `<eos>` | **FAIL** |
| `a a b d <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a b d` | `<eos>` | **FAIL** |
| `a a c a <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a c a` | `copy <eos>` | **FAIL** |
| `a a c b <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a c b` | `<eos>` | **FAIL** |
| `a a c c <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a c c` | `<eos>` | **FAIL** |
| `a a c d <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a c d` | `a <eos>` | **FAIL** |
| `a a d a <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a d a` | `<eos>` | **FAIL** |
| `a a d b <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a d b` | `a b b b` | **FAIL** |
| `a a d c <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a d c` | `a d a <eos>` | **FAIL** |
| `a a d d <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> copy` | `a a d d` | `<eos>` | **FAIL** |

### 6. Big Secret Recall (Accuracy: 90.0%)

| Query Sequence | Expected Secret | Actual Output | Result |
| --- | --- | --- | --- |
| `3-Key: key_x secret_x key_y secret_y key_z secret_z key_x` | `secret_x` | `secret_x` | **PASS** |
| `3-Key: key_x secret_x key_y secret_y key_z secret_z key_y` | `secret_y` | `secret_y` | **PASS** |
| `3-Key: key_x secret_x key_y secret_y key_z secret_z key_z` | `secret_z` | `secret_z` | **PASS** |
| `3-Key: key_x secret_x key_z secret_z key_y secret_y key_x` | `secret_x` | `secret_x` | **PASS** |
| `3-Key: key_x secret_x key_z secret_z key_y secret_y key_y` | `secret_y` | `secret_y` | **PASS** |
| `3-Key: key_x secret_x key_z secret_z key_y secret_y key_z` | `secret_z` | `secret_z` | **PASS** |
| `3-Key: key_y secret_y key_x secret_x key_z secret_z key_x` | `secret_x` | `secret_x` | **PASS** |
| `3-Key: key_y secret_y key_x secret_x key_z secret_z key_y` | `secret_y` | `secret_y` | **PASS** |
| `3-Key: key_y secret_y key_x secret_x key_z secret_z key_z` | `secret_z` | `secret_z` | **PASS** |
| `3-Key: key_y secret_y key_z secret_z key_x secret_x key_x` | `secret_x` | `secret_x` | **PASS** |
| `3-Key: key_y secret_y key_z secret_z key_x secret_x key_y` | `secret_y` | `secret_y` | **PASS** |
| `3-Key: key_y secret_y key_z secret_z key_x secret_x key_z` | `secret_z` | `secret_z` | **PASS** |
| `3-Key: key_z secret_z key_x secret_x key_y secret_y key_x` | `secret_x` | `secret_x` | **PASS** |
| `3-Key: key_z secret_z key_x secret_x key_y secret_y key_y` | `secret_y` | `secret_y` | **PASS** |
| `3-Key: key_z secret_z key_x secret_x key_y secret_y key_z` | `secret_z` | `secret_z` | **PASS** |
| `3-Key: key_z secret_z key_y secret_y key_x secret_x key_x` | `secret_x` | `secret_x` | **PASS** |
| `3-Key: key_z secret_z key_y secret_y key_x secret_x key_y` | `secret_y` | `secret_y` | **PASS** |
| `3-Key: key_z secret_z key_y secret_y key_x secret_x key_z` | `secret_z` | `secret_z` | **PASS** |
| `4-Key: key_x secret_x key_y secret_y key_z secret_z key_w secret_w key_x` | `secret_x` | `secret_x` | **PASS** |
| `4-Key: key_x secret_x key_y secret_y key_z secret_z key_w secret_w key_y` | `secret_y` | `secret_y` | **PASS** |
| `4-Key: key_x secret_x key_y secret_y key_z secret_z key_w secret_w key_z` | `secret_z` | `secret_z` | **PASS** |
| `4-Key: key_x secret_x key_y secret_y key_z secret_z key_w secret_w key_w` | `secret_w` | `secret_x` | **FAIL** |
| `4-Key: key_y secret_y key_w secret_w key_x secret_x key_z secret_z key_x` | `secret_x` | `secret_x` | **PASS** |
| `4-Key: key_y secret_y key_w secret_w key_x secret_x key_z secret_z key_y` | `secret_y` | `secret_y` | **PASS** |
| `4-Key: key_y secret_y key_w secret_w key_x secret_x key_z secret_z key_z` | `secret_z` | `secret_z` | **PASS** |
| `4-Key: key_y secret_y key_w secret_w key_x secret_x key_z secret_z key_w` | `secret_w` | `secret_y` | **FAIL** |
| `4-Key: key_w secret_w key_z secret_z key_y secret_y key_x secret_x key_x` | `secret_x` | `secret_x` | **PASS** |
| `4-Key: key_w secret_w key_z secret_z key_y secret_y key_x secret_x key_y` | `secret_y` | `secret_y` | **PASS** |
| `4-Key: key_w secret_w key_z secret_z key_y secret_y key_x secret_x key_z` | `secret_z` | `secret_z` | **PASS** |
| `4-Key: key_w secret_w key_z secret_z key_y secret_y key_x secret_x key_w` | `secret_w` | `secret_y` | **FAIL** |

> [!NOTE]
> The Gated Euler dynamics in FERN allow the higher-level belief states (layers L1-L3) to behave as a continuous-time memory cell.
> When sensory input is removed (clamped to `<pad>` 0), the gate $g$ shuts down incoming drive, enabling the beliefs to persist
> unchanged over multiple time steps. Once the trigger token (`copy` or the recall query `key`) is presented,
> the gate reopens and reads out the stored values through the decoder projection.

