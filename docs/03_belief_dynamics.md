# Belief Dynamics: Gated Euler Integration

**FERN Technical Specification — Document 03**

---

## 1. Continuous-Time Formulation

In the continuous limit, belief dynamics at each hierarchical level follow the gradient flow of the variational free energy:

$$\dot{\mu}[l] = -\frac{\partial F}{\partial \mu[l]} = -\pi[l] \odot \epsilon_{raw}[l] + W_{up,l} \cdot (\pi[l-1] \odot \epsilon_{raw}[l-1])$$

This can be decomposed into two opposing forces:

- **Self-correction**: $-\pi[l] \odot \epsilon_{raw}[l]$ — reduces the prediction error AT level l
- **Bottom-up drive**: $W_{up} \cdot \epsilon[l-1]$ — incorporates information from BELOW level l

## 2. Discrete Integration

### 2.1 Naive Euler (Unstable)

$$\mu[l]_{t+1} = \mu[l]_t + \kappa \cdot d_t$$

where $d_t = -\epsilon[l] + W_{up} \cdot \epsilon[l-1]$ is the drive and $\kappa$ is the step size.

**Problem**: With fixed $\kappa$, this can oscillate or diverge when the drive is large, especially during early training when precision estimates are unreliable.

### 2.2 Gated Euler (Stable — FERN's approach)

Inspired by Closed-form Continuous-time (CfC) networks, we introduce a learned gate:

$$g = \sigma(W_{gate} \cdot [\mu[l]; \epsilon[l]] + b_{gate})$$

$$\mu[l]_{t+1} = \mu[l]_t + g \odot (\kappa \cdot \text{clamp}(d_t, -M, M))$$

where:
- $g \in [0, 1]^{d_l}$ — per-dimension gate (sigmoid output)
- $\kappa > 0$ — global step size (default: 0.1)
- $M > 0$ — maximum drive magnitude (default: 5.0)
- $[\cdot; \cdot]$ denotes concatenation

### 2.3 Why Gating?

The gate $g$ provides **three stability mechanisms**:

1. **Selective update**: Dimensions where the network is uncertain (high ε) can be gated differently from stable dimensions.

2. **Adaptive time constant**: $g \approx 0$ slows the effective dynamics (belief changes slowly), while $g \approx 1$ allows rapid updates. This is analogous to a time constant $\tau = 1/g$ in a continuous-time ODE.

3. **Gradient highway**: Since $g$ depends on $\mu[l]$ and $\epsilon[l]$, it creates a shortcut for gradients to flow from the loss directly to the gate parameters, avoiding vanishing gradients through many inner steps.

## 3. Drive Clamping

### 3.1 Motivation

The drive $d_t = -\epsilon[l] + W_{up} \cdot \epsilon[l-1]$ is **unbounded** in general. While the gate $g \in [0,1]$ limits the multiplicative factor, $\kappa \cdot d_t$ itself can be arbitrarily large, leading to:

- Belief states flying to extreme values
- NaN propagation through subsequent computations
- Training instability

### 3.2 Hard Clamping

```
d_clamped = clamp(d, -max_drive, max_drive)
```

Using element-wise minimum and maximum operations:

```rust
let bound = Tensor::ones_like(&drive)?.affine(max_drive, 0.0)?;
let neg_bound = bound.neg()?;
let drive = drive.minimum(&bound)?.maximum(&neg_bound)?;
```

This preserves gradient flow for non-clamped elements (subgradient = 1) while preventing catastrophic updates (subgradient = 0 at the boundary).

### 3.3 Choosing max_drive

The effective maximum belief change per inner step is:

$$\Delta\mu_{max} = g_{max} \cdot \kappa \cdot M = 1.0 \cdot 0.1 \cdot 5.0 = 0.5$$

This means each dimension of μ can change by at most 0.5 per inner step. With `inner_steps = 3`, the total possible change is 1.5, which is bounded and reasonable for beliefs initialized at 0.

## 4. Algorithm

```
ALGORITHM: Gated Euler Belief Update
────────────────────────────────────
Input:  state = {μ[0..L], σ²[0..L]}, model, embedding e_t
Output: updated state, free energy F

for step = 1 to inner_steps:
    μ[0] ← e_t                           // clamp sensory

    for l = 0 to L:
        if l < L:
            p̂[l] ← f_pred_{l+1}(μ[l+1])  // top-down prediction
        else:
            p̂[l] ← 0                      // prior for top level

        ε_raw[l] ← μ[l] - p̂[l]           // raw prediction error
        σ²[l] ← α·σ²[l] + (1-α)·ε_raw²   // update variance (EMA)
        π[l] ← 1 / (σ²[l] + ε_min)        // precision
        ε[l] ← π[l] ⊙ ε_raw[l]            // weighted error

        // Free energy (skip L0; KL-only for L_max)
        if l > 0:
            acc ← ε_raw² · π[l]
            kl  ← σ² + μ² - 1 - log(σ² + ε_min)
            F_l ← (acc + kl) / d_l    if l < L
            F_l ← kl / d_l            if l = L

    for l = 1 to L:
        up  ← W_up_l · ε[l-1]             // bottom-up error signal
        d   ← up - ε[l]                    // drive
        d   ← clamp(d, -M, M)             // stabilize
        g   ← σ(W_gate · [μ[l]; ε[l]])    // CfC gate
        μ[l] ← μ[l] + g ⊙ (κ · d)        // Gated Euler step

return state, mean(F)
```

## 5. Properties

### 5.1 Convergence

Under mild conditions (bounded weights, positive precision), the Gated Euler iteration is a **contraction mapping** when $\kappa \cdot g_{max} < 2/\lambda_{max}$ where $\lambda_{max}$ is the largest eigenvalue of the Hessian of F with respect to μ. The default settings ($\kappa = 0.1$, $g \leq 1$) satisfy this for typical network configurations.

### 5.2 Computational Cost

Per token, per inner step:
- 3 matrix multiplications per layer (f_pred forward, W_up, W_gate)
- Element-wise operations: O(B × Σd_l)
- Total: O(B × L × d² × inner_steps)

Memory: O(B × Σd_l) — constant in sequence length.
