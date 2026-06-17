# Training Procedure

**FERN Technical Specification — Document 05**

---

## 1. Loss Function

The total training loss combines cross-entropy (task objective) with variational free energy (world model objective):

$$\mathcal{L} = \mathcal{L}_{CE} + w_{FE}(t) \cdot \mathcal{L}_{FE}$$

where:
- $\mathcal{L}_{CE}$ = standard next-token cross-entropy loss
- $\mathcal{L}_{FE}$ = average variational free energy across sequence and inner steps
- $w_{FE}(t)$ = cosine-scheduled FE weight (see Doc 04, §5)

### 1.1 Cross-Entropy Loss

$$\mathcal{L}_{CE} = -\frac{1}{B \cdot S} \sum_{b=1}^{B} \sum_{t=1}^{S} \log p(x_{t+1} | \mu[1..L]_t)$$

where logits are produced by the decoder from concatenated beliefs.

### 1.2 Free Energy Loss

$$\mathcal{L}_{FE} = \frac{1}{S \cdot K} \sum_{t=1}^{S} \sum_{k=1}^{K} \frac{1}{B} \sum_{b=1}^{B} \sum_{l=1}^{L} F_l^{(b,t,k)}$$

where K = `inner_steps` and $F_l$ is the per-level variational free energy (see Doc 02).

## 2. Training Loop

```
ALGORITHM: FERN Training Step
──────────────────────────────
Input:  batch of sequences {(input_t, target_t)}

1. Initialize fresh state: μ[l] ← 0, σ²[l] ← 1  ∀l

2. For each token position t = 1..S:
   a. Embed: e_t ← encoder(input_t)
   b. Run inner inference (K steps) → update state, accumulate FE
   c. Decode: logits_t ← decoder(concat(μ[1..L]))

3. Compute CE loss from logits vs targets

4. Get scheduled FE weight: w ← PRISM.fe_weight()

5. Total loss: L ← CE + w · FE

6. Backward pass: grads ← L.backward()

7. Extract layer precisions: π̄[l] ← mean(1/σ²[l])  ∀l

8. PRISM optimizer step: step(grads, π̄)
```

## 3. Hyperparameter Reference

### 3.1 Model Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| vocab_size | 10 | Vocabulary size |
| d_layers | [32, 64, 64, 64] | Layer dimensions [L0, L1, L2, L3] |
| kappa | 0.1 | Euler step size |
| alpha | 0.9 | EMA decay for σ² |
| epsilon_min | 1e-4 | Precision floor |
| max_drive | 5.0 | Drive clamp bound |
| inner_steps | 3 | Active inference iterations per token |

### 3.2 PRISM Optimizer

| Parameter | Value | Description |
|-----------|-------|-------------|
| pred_lr | 1e-3 | Learning rate for f_pred |
| error_lr | 3e-4 | Learning rate for W_up |
| gate_lr | 1e-3 | Learning rate for W_gate |
| io_lr | 3e-4 | Learning rate for encoder/decoder |
| beta1 | 0.9 | Adam first moment decay |
| beta2 | 0.999 | Adam second moment decay |
| eps | 1e-8 | Adam epsilon |
| weight_decay | 1e-4 | Decoupled weight decay |
| grad_clip | 1.0 | Per-variable gradient norm clip |
| fe_weight_target | 1.0 | Target FE weight after warmup |
| warmup_steps | 50 | Cosine warmup duration |
| precision_scaling | true | Enable precision-scaled LR |

### 3.3 Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| epochs | 300 | Number of training steps |
| batch_size | 32 | Sequences per batch |
| seq_len | 12 | Tokens per sequence |

## 4. Expected Training Dynamics

### 4.1 Phase 1: Pure CE (epochs 1-10)

- FE weight ≈ 0, model trains purely on next-token prediction
- CE drops from ~2.3 (random) toward ~2.0
- FE fluctuates but is not optimized
- Beliefs begin forming useful representations for decoding

### 4.2 Phase 2: Warmup (epochs 10-50)

- FE weight ramps from ~0 to ~1.0 via cosine schedule
- CE may temporarily plateau or slightly increase as FE gradients enter
- FE begins decreasing as the generative model aligns with learned beliefs
- Total loss increases due to growing FE contribution, then stabilizes

### 4.3 Phase 3: Joint Optimization (epochs 50-300)

- FE weight = 1.0 (full variational objective)
- Both CE and FE should decrease together
- Precision estimates stabilize, enabling precision-scaled LR
- Beliefs become hierarchically organized: L1 captures lexical, L2 syntactic, L3 thematic patterns

### 4.4 Convergence Indicators

| Metric | Healthy | Unhealthy |
|--------|---------|-----------|
| CE | Monotonically decreasing | Oscillating or increasing |
| FE | Decreasing after warmup | Increasing or negative |
| FE sign | Always ≥ 0 | Negative → collapse |
| Total loss | Decreasing after warmup | Unbounded growth |
| Generated text | Matches input patterns | Random or repetitive |

## 5. Synthetic Dataset

The training dataset consists of two repeating patterns:

```
Pattern A: "the cat sat on the mat" → [1, 2, 3, 4, 1, 5]
Pattern B: "the dog ran in the park" → [1, 6, 7, 8, 1, 9]
```

Each batch randomly selects Pattern A or B with 50% probability. Sequences of length 12 wrap around the 6-token pattern twice, testing the model's ability to:

1. **Distinguish patterns** after seeing the discriminative token (position 2: "cat" vs "dog")
2. **Maintain context** across the cyclic boundary
3. **Predict deterministic continuations** (e.g., after "cat" always comes "sat")

### 5.1 Theoretical Minimum CE

For position-dependent analysis:
- Position 0 ("the"): always follows pattern → CE = 0
- Position 1 ("cat"/"dog"): 50% probability each → CE = ln(2) ≈ 0.69
- Positions 2-5: deterministic given pattern → CE = 0

Average CE per position = 0.69/6 ≈ 0.115

With perfect learning, CE should converge toward ~0.12. Values above 1.0 indicate significant room for improvement.
