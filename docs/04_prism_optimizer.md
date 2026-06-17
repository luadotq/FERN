# PRISM: PRecision-Informed Stochastic Momentum

**FERN Technical Specification — Document 04**

---

## 1. Motivation

Standard optimizers (Adam, AdamW) treat all parameters uniformly. In FERN, parameters serve fundamentally different roles:

| Role | Parameters | Function | Risk |
|------|-----------|----------|------|
| **Prediction** | f_pred weights | Top-down generative model | Destroys good predictions if updated too aggressively |
| **Error Path** | W_up weights | Bottom-up error projection | Amplifies noise if learning rate too high |
| **Gate** | W_gate weights | Temporal integration control | Gate collapse (all 0 or all 1) |
| **I/O** | encoder, decoder | Interface with tokens | Embedding drift |

PRISM addresses this by providing:
1. Per-group learning rates
2. Precision-scaled updates for generative parameters
3. Cosine warmup for the free energy objective
4. Per-variable gradient clipping

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PRISM Optimizer                       │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  For each parameter θ_i:                        │    │
│  │    ├─ Adam moment: m_i, v_i                     │    │
│  │    ├─ Role: {Prediction, ErrorPath, Gate, IO}   │    │
│  │    └─ Layer index: l (for precision lookup)     │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐     │
│  │ Gradient  │→ │ Per-group │→ │ Precision-scaled  │     │
│  │ Clipping  │  │ Adam     │  │ Learning Rate     │     │
│  └──────────┘  └──────────┘  └───────────────────┘     │
│                                                         │
│  ┌──────────────────────────────────────────────┐       │
│  │ Cosine Warmup Schedule for FE Weight         │       │
│  │ fe_w(t) = w_target · ½ · (1 - cos(π·t/T))   │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

## 3. Parameter Classification

Variables are automatically classified by their names in the VarMap:

```
Name pattern         → Role         → Base LR
─────────────────────────────────────────────
*f_pred*             → Prediction   → pred_lr  (1e-3)
*W_up*               → ErrorPath    → error_lr (3e-4)
*W_gate*             → Gate         → gate_lr  (1e-3)
encoder*, decoder*   → IO           → io_lr    (3e-4)
```

The layer index is extracted from the prefix `layer_N`:
```
"layer_2.f_pred.linear1.weight" → Role=Prediction, layer_idx=2
```

## 4. Update Algorithm

### 4.1 Per-Variable Gradient Clipping

For each parameter θ_i with gradient g_i:

$$\hat{g}_i = \begin{cases} g_i & \text{if } \|g_i\|_2 \leq C \\ C \cdot \frac{g_i}{\|g_i\|_2} & \text{otherwise} \end{cases}$$

where C = `grad_clip` (default: 1.0).

This prevents any single parameter from dominating the update, which is critical when FE and CE gradients conflict.

### 4.2 Adam Moments

Standard Adam moment updates:

$$m_i \leftarrow \beta_1 \cdot m_i + (1 - \beta_1) \cdot \hat{g}_i$$
$$v_i \leftarrow \beta_2 \cdot v_i + (1 - \beta_2) \cdot \hat{g}_i^2$$

Bias-corrected:

$$\hat{m}_i = m_i / (1 - \beta_1^t), \quad \hat{v}_i = v_i / (1 - \beta_2^t)$$

### 4.3 Precision-Scaled Learning Rate

For **Prediction** parameters at layer l:

$$\text{lr}_{eff} = \text{lr}_{pred} \cdot \frac{\bar{\pi}_l}{\bar{\pi}_l + 1}$$

where $\bar{\pi}_l = \text{mean}(\pi[l])$ is the average precision at level l.

**Behavior**:
- When $\bar{\pi}_l \gg 1$ (model is confident): $\text{lr}_{eff} \approx \text{lr}_{pred}$. The generative model updates at full speed to match the confident beliefs.
- When $\bar{\pi}_l \ll 1$ (model is uncertain): $\text{lr}_{eff} \approx 0$. Updates are suppressed to prevent destructive changes during high uncertainty.
- At $\bar{\pi}_l = 1$ (balanced): $\text{lr}_{eff} = \text{lr}_{pred} / 2$.

This creates a natural **explore-exploit** tradeoff: the model explores (slow updates) when uncertain and exploits (fast updates) when confident.

### 4.4 Decoupled Weight Decay

Following AdamW, weight decay is applied directly to parameters (decoupled from the adaptive learning rate):

$$\theta_i \leftarrow \theta_i \cdot (1 - \text{lr} \cdot \lambda) - \text{lr} \cdot \frac{\hat{m}_i}{\sqrt{\hat{v}_i} + \epsilon}$$

where λ = `weight_decay` (default: 1e-4).

### 4.5 Complete Algorithm

```
ALGORITHM: PRISM Step
─────────────────────
Input:  gradients G, layer precisions π̄[0..L]
State:  moments {m_i, v_i}, step counter t

t ← t + 1
bc1 ← 1 - β₁^t
bc2 ← 1 - β₂^t

for each parameter θ_i with gradient g_i:
    // 1. Clip
    g ← clip(g_i, grad_clip)

    // 2. Moments
    m_i ← β₁·m_i + (1-β₁)·g
    v_i ← β₂·v_i + (1-β₂)·g²

    // 3. Bias correction
    m̂ ← m_i / bc1
    v̂ ← v_i / bc2

    // 4. Learning rate
    lr ← base_lr(role_i)
    if role_i = Prediction and precision_scaling:
        lr ← lr · π̄[layer_i] / (π̄[layer_i] + 1)

    // 5. Update (AdamW style)
    θ_i ← θ_i · (1 - lr·λ) - lr · m̂ / (√v̂ + ε)
```

## 5. Cosine Warmup Schedule

### 5.1 The CE-FE Conflict

Without warmup, the total loss `L = CE + w·FE` creates conflicting gradients from the first step:
- CE wants beliefs that maximize next-token prediction accuracy
- FE wants beliefs that minimize surprise in the generative model

When the generative model is randomly initialized, FE gradients are essentially noise, corrupting the useful CE signal.

### 5.2 Solution: Cosine Warmup

$$w_{FE}(t) = w_{target} \cdot \frac{1}{2} \left(1 - \cos\left(\pi \cdot \frac{t}{T_{warmup}}\right)\right)$$

**Properties**:
- $w_{FE}(0) = 0$ — pure CE training at the start
- $w_{FE}(T/2) = w_{target}/2$ — half FE weight at midpoint
- $w_{FE}(T) = w_{target}$ — full variational objective
- Zero derivative at both endpoints: smooth transition, no sudden jumps
- Monotonically increasing: no oscillation in training dynamics

### 5.3 Warmup Duration

Default: T_warmup = 50 steps.

Rationale: At epoch 50, CE has typically converged to a reasonable baseline (≈2.0 for 10-token vocabulary). The generative model has had enough gradient signal through CE to produce meaningful predictions, making FE gradients informative rather than random.

## 6. Performance Considerations

1. **Zero-copy gradient clipping**: If gradient norm ≤ max_norm, the original tensor is returned without allocation.
2. **Fused affine operations**: Uses `tensor.affine(scale, bias)` instead of separate multiply + add, reducing kernel launches.
3. **Pre-computed bias correction**: bc1 and bc2 are computed once per step, shared across all parameters.
4. **In-place parameter update**: Uses `Var::set()` for O(1) pointer swap.
