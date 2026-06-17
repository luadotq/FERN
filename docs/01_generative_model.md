# Hierarchical Generative Model

**FERN Technical Specification — Document 01**

---

## 1. Definitions

### 1.1 Notation

| Symbol | Domain | Description |
|--------|--------|-------------|
| L | ℕ | Number of hierarchical levels (L0..L_max) |
| d_l | ℕ | Dimensionality of level l |
| μ[l] ∈ ℝ^{B×d_l} | tensor | Belief state (posterior mean) at level l |
| σ²[l] ∈ ℝ₊^{B×d_l} | tensor | Variance estimate at level l |
| π[l] ∈ ℝ₊^{B×d_l} | tensor | Precision (inverse variance) at level l |
| p̂[l] ∈ ℝ^{B×d_l} | tensor | Top-down prediction for level l |
| ε[l] ∈ ℝ^{B×d_l} | tensor | Precision-weighted prediction error at level l |
| B | ℕ | Batch size |
| x_t | {0,..,V-1} | Input token at time t |

### 1.2 Model Components

Each hierarchical layer l ∈ {1, ..., L_max} contains:

- **f_pred_l**: MLP(d_l → d_{l-1}). Generates top-down prediction of level l-1 from beliefs at level l.
- **W_up_l**: Linear(d_{l-1} → d_l). Projects prediction errors from level l-1 up to level l.
- **W_gate_l**: Linear(2·d_l → d_l). Computes the CfC temporal gate from concatenated beliefs and errors.

Global components:
- **encoder**: Embedding(V → d_0). Maps discrete tokens to continuous sensory representations.
- **decoder**: Linear(Σd_{1..L} → V). Maps concatenated beliefs to next-token logits.

## 2. Generative Process

The hierarchical generative model defines a joint distribution over latent states:

$$p(z_0, z_1, ..., z_L) = p(z_L) \prod_{l=0}^{L-1} p(z_l | z_{l+1})$$

where:
- $p(z_L) = \mathcal{N}(0, I)$ — isotropic Gaussian prior at the top level
- $p(z_l | z_{l+1}) = \mathcal{N}(f_{pred}(z_{l+1}), \sigma^2_l I)$ — conditional Gaussian

### 2.1 Top-Down Prediction

For each level l ∈ {0, ..., L_max - 1}:

```
p̂[l] = f_pred_{l+1}(μ[l+1])
```

For the top level L_max, the prediction comes from the prior:

```
p̂[L_max] = 0   (standard normal prior)
```

### 2.2 MLP Architecture

Each f_pred is a two-layer MLP with ReLU activation:

```
f_pred(x) = W₂ · ReLU(W₁ · x + b₁) + b₂
```

where:
- W₁ ∈ ℝ^{mid × d_l}, b₁ ∈ ℝ^{mid}
- W₂ ∈ ℝ^{d_{l-1} × mid}, b₂ ∈ ℝ^{d_{l-1}}
- mid = max(d_l + d_{l-1}, 32)

The non-linearity allows f_pred to learn complex inter-level mappings beyond linear projections.

## 3. Inference (Recognition) Model

The approximate posterior factorizes as:

$$q(z_0, ..., z_L) = \prod_{l=0}^{L} q(z_l)$$

where $q(z_l) = \mathcal{N}(\mu[l], \sigma^2[l] \cdot I)$.

The parameters μ[l] (beliefs) are updated iteratively through active inference (see Document 03), while σ²[l] is estimated via exponential moving average of squared prediction errors.

### 3.1 Variance Estimation

```
σ²[l]_new = α · σ²[l]_old + (1 - α) · ε_raw[l]²
```

where:
- α ∈ (0, 1) is the EMA decay factor (default: 0.9)
- ε_raw[l] = μ[l] - p̂[l] is the raw (unweighted) prediction error

### 3.2 Precision

```
π[l] = 1 / (σ²[l] + ε_min)
```

where ε_min > 0 prevents division by zero (default: 1e-4).

**Critical implementation note**: π must be computed from the live (non-detached) σ² tensor to allow gradients to flow through the precision pathway into the generative model parameters.

## 4. Decoder

The decoder maps higher-level beliefs to next-token predictions:

```
logits = W_dec · concat(μ[1], μ[2], ..., μ[L_max]) + b_dec
```

Note that L0 (sensory) is excluded from decoding since it merely reflects the current input.

## 5. Parameter Counts

For the default configuration d_layers = [32, 64, 64, 64]:

| Component | Parameters | Description |
|-----------|-----------|-------------|
| encoder | 32 × V | Token embeddings |
| f_pred (×3) | ~12K | Generative predictions |
| W_up (×3) | ~12K | Error projections |
| W_gate (×3) | ~24K | Temporal gates |
| decoder | 192 × V | Belief-to-logit mapping |

Total (V=10): ~50K parameters.
