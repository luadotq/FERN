# Variational Free Energy

**FERN Technical Specification — Document 02**

---

## 1. Motivation

The variational free energy F provides a tractable upper bound on the negative log-evidence (surprisal) of observations:

$$F = -\log p(x) + \text{KL}[q(z) \| p(z|x)] \geq -\log p(x)$$

Minimizing F simultaneously:
1. Maximizes the accuracy of predictions (data fit)
2. Minimizes the complexity of the internal model (regularization)

This is the core objective of FERN: beliefs are updated to minimize free energy, and model parameters are trained to make free energy minimization easier.

## 2. Derivation

### 2.1 Factored Free Energy

Given the factored generative model (Doc 01, §2) and factored recognition model (Doc 01, §3), the total free energy decomposes across levels:

$$F = \sum_{l=1}^{L} F_l$$

where level 0 is excluded (clamped to observations) and each per-level free energy is:

$$F_l = \underbrace{\mathbb{E}_{q(z_l)}[-\log p(z_l | z_{l+1})]}_{\text{accuracy}} + \underbrace{\text{KL}[q(z_l) \| p(z_l)]}_{\text{complexity}}$$

### 2.2 Gaussian Case

With $q(z_l) = \mathcal{N}(\mu_l, \sigma^2_l I)$ and $p(z_l) = \mathcal{N}(0, I)$:

**Accuracy term** (negative log-likelihood under the generative model):

$$\text{accuracy}_l = \frac{1}{d_l} \sum_{j=1}^{d_l} \frac{(\mu_{l,j} - \hat{p}_{l,j})^2}{\sigma^2_{l,j} + \epsilon_{min}}$$

**Complexity term** (KL divergence to standard normal prior):

$$\text{KL}_l = \frac{1}{2d_l} \sum_{j=1}^{d_l} \left[ \sigma^2_{l,j} + \mu^2_{l,j} - 1 - \log(\sigma^2_{l,j} + \epsilon_{min}) \right]$$

**Total per-level free energy**:

$$F_l = \text{accuracy}_l + \text{KL}_l$$

### 2.3 Non-Negativity Proof

**Theorem**: $F_l \geq 0$ for all valid σ², μ, and ε_raw.

**Proof**: The KL divergence $\text{KL}[q \| p] \geq 0$ by Gibbs' inequality. The accuracy term is a sum of non-negative ratios (squared errors divided by positive denominators). Therefore $F_l \geq 0$.

More specifically, consider the function $f(\sigma^2) = \sigma^2 - 1 - \log(\sigma^2)$ for $\sigma^2 > 0$. Taking the derivative: $f'(\sigma^2) = 1 - 1/\sigma^2$, which equals zero at $\sigma^2 = 1$. The second derivative $f''(\sigma^2) = 1/\sigma^4 > 0$, confirming this is a minimum. At the minimum, $f(1) = 0$. Therefore $f(\sigma^2) \geq 0$.

Combined with $\mu^2 \geq 0$ and $\text{accuracy} \geq 0$, we have $F_l \geq 0$. ∎

**Corollary**: The free energy cannot collapse to $-\infty$, which was the critical failure mode in previous formulations where $F = \|ε\|^2/\pi + \log(\pi)$ allowed the model to exploit $\pi \to 0$.

## 3. Level-Specific Treatment

### 3.1 Level 0 (Sensory — Clamped)

Level 0 beliefs are clamped to the token embedding: $\mu[0] = \text{encoder}(x_t)$.

Since μ[0] is not a free variable, including it in the free energy would:
- Create parasitic gradients pulling encoder weights toward zero (via μ² term)
- Incorrectly penalize the input representation

**Implementation**: L0 is excluded from the FE summation. Prediction errors at L0 are still computed and propagated upward.

### 3.2 Intermediate Levels (L1 .. L_{max-1})

Full free energy: $F_l = \text{accuracy}_l + \text{KL}_l$

Both terms are active. The accuracy term ensures the generative model accurately predicts downward, while the KL term regularizes beliefs toward the prior.

### 3.3 Top Level (L_max)

The top level has no higher-level predictor, so $\hat{p}[L_{max}] = 0$ (zero prior).

If we include the accuracy term, we get:
$$\text{accuracy}_{L_{max}} = \frac{\mu_{L_{max}}^2}{\sigma^2_{L_{max}}}$$

Combined with the KL term (which also contains $\mu_{L_{max}}^2$), this creates a **double penalty** on top-level beliefs, suppressing them toward zero and preventing the formation of abstract representations.

**Implementation**: L_max uses only the KL complexity term (no accuracy term). The belief update at L_max is still driven by $W_{up} \cdot \epsilon[L_{max}-1]$, which provides sufficient learning signal.

## 4. Gradient Flow Analysis

### 4.1 Through Precision (BUG-1 Fix)

The precision π[l] must be computed from the **live** (non-detached) σ² tensor:

```
σ²_new = α · σ²_old + (1-α) · ε_raw²     ← ε_raw depends on f_pred params
π = 1 / (σ²_new + ε_min)                   ← gradient flows through here
F = ε_raw² · π + KL(σ²_new)                ← F connects to f_pred via π and ε_raw
```

After computing F, σ² is detached for state storage:
```
state.σ²[l] = σ²_new.detach()              ← prevents infinite computation graph
```

This ensures gradients flow: $F \xrightarrow{\partial} \pi \xrightarrow{\partial} \sigma^2_{new} \xrightarrow{\partial} \epsilon_{raw}^2 \xrightarrow{\partial} f_{pred}$

### 4.2 Dual Gradient Pathways

The model parameters receive gradients from two sources:

1. **Cross-Entropy path**: $\text{CE} \to \text{logits} \to \text{decoder} \to \mu[l] \to \text{inner steps} \to f_{pred}, W_{up}, W_{gate}$
2. **Free Energy path**: $\text{FE} \to \pi, \epsilon_{raw}, \mu^2, \sigma^2 \to f_{pred}, W_{up}, W_{gate}$

The cosine warmup (Doc 04) ensures these paths are introduced gradually, preventing early gradient conflicts.

## 5. Numerical Stability

| Concern | Mitigation |
|---------|-----------|
| Division by zero in π | ε_min = 1e-4 added to denominator |
| log(0) in KL | log(σ² + ε_min) used instead of log(σ²) |
| Infinite FE | KL-regularized formula guarantees F ≥ 0 |
| Large gradients through π | Gradient clipping in PRISM optimizer |
| σ² negativity | EMA of squared errors ensures σ² ≥ 0 |
