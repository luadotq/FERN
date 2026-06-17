# FERN Per-Task Model  Report

Report automatically generated after per-task model training and evaluation.

## Model & Training Specifications

| Parameter | Value |
| --- | --- |
| **Vocab Size** | 27 |
| **d_layers (Standard tasks)** | [32, 64, 64, 64] |
| **d_layers (Big tasks)** | [64, 128, 128, 128] |
| **PRISM LR (f_pred)** | 0.001 |
| **PRISM LR (w_up)** | 0.0003 |
| **PRISM LR (w_gate)** | 0.001 |
| **PRISM LR (I/O)** | 0.0003 |
| **Base Epochs** | 300 (scaled by 1.00x) |
| **Batch Size** | 32 |

## Task Training Convergence

| Task | Final Total Loss | Final CE Loss | Final FE Loss |
| --- | --- | --- | --- |
| **Standard Pattern Completion** | 0.4912 | 0.1176 | 1.8678 |
| **Big Pattern Completion** | 1.1272 | 0.1946 | 4.6627 |
| **Standard Copy with Delay** | 0.7913 | 0.4488 | 3.4251 |
| **Big Copy with Delay** | 5.3388 | 1.0882 | 42.5056 |
| **Standard Secret Recall** | 0.3930 | 0.2066 | 1.8649 |
| **Big Secret Recall** | 0.7635 | 0.4602 | 3.0326 |

## Per-Task vs Mixed-Task Comparison

| Task | Per-Task | Mixed | Upper Bound Gap |
| --- | --- | --- | --- |
| **Standard Pattern** | 100.0% | 100.0% | +0.0% |
| **Standard Copy D3** | 100.0% | 25.0% | +75.0% |
| **Standard Secret** | 100.0% | 100.0% | +0.0% |
| **Big Pattern** | 100.0% | 0.0% | +100.0% |
| **Big Copy D10** | 0.4% | 0.0% | +0.4% |
| **Big Secret** | 100.0% | 76.7% | +23.3% |

> [!NOTE]
> By training a separate model for each benchmark task, we eliminate task-interference (mixed learning gradient conflict).
> This allows the belief dynamics and temporal prediction weights ($W_{rec}$) to dedicate 100% of the network capacity to the specific
> task, leading to much higher convergence and final accuracy scores.

