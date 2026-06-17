# FERN vs RNN Baselines (LSTM / GRU) Report

Report automatically generated after model training and evaluation.

## Training Specifications

| Parameter | Value |
| --- | --- |
| **Epochs** | 500 |
| **Batch Size** | 32 |

## Baseline Comparison Table

| Model | Parameters | Standard Pattern | Standard Copy D3 | Standard Secret | Big Pattern | Big Copy D10 | Big Secret |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **FERN (Active Inference)** | 571K | 100.0% | 37.5% | 100.0% | 0.0% | 0.4% | 63.3% |
| **LSTM (Standard)** | 575K | 100.0% | 100.0% | 100.0% | 50.0% | 40.6% | 100.0% |
| **GRU (Standard)** | 571K | 100.0% | 100.0% | 100.0% | 50.0% | 61.3% | 100.0% |

> [!NOTE]
> This benchmark compares the Free Energy Recurrent Network (FERN) under hierarchical active inference
> against standard recurrent neural architectures (LSTM and GRU) optimized with AdamW.
> All models are compared under matched parameter budgets to ensure scientific fairness.

