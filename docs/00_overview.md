# FERN: Free Energy Recurrent Network

**Technical Specification v0.2**

---

## Abstract

FERN (Free Energy Recurrent Network) is a recurrent neural architecture grounded in the **Free Energy Principle** and **predictive coding** theory from computational neuroscience. Unlike transformer-based models that rely on attention mechanisms with O(n²) complexity, FERN maintains O(1) memory per time step through a hierarchical generative model that continuously minimizes variational free energy.

The architecture processes sequential data by maintaining a hierarchy of belief states μ[l] that are iteratively refined through active inference — a process where top-down predictions are compared against bottom-up sensory signals, producing precision-weighted prediction errors that drive belief updates.

## Key Innovations

1. **Hierarchical Predictive Coding**: Multi-level generative model where each level predicts the activity of the level below. Prediction errors propagate upward, weighted by learned precision.

2. **Gated Euler Integration (CfC-inspired)**: Belief dynamics follow continuous-time ODEs discretized via gated Euler steps, inspired by Closed-form Continuous-time (CfC) neural networks.

3. **Variational Free Energy as Native Loss**: The model's loss function is derived directly from the variational free energy bound, combining prediction accuracy with KL complexity regularization.

4. **PRISM Optimizer**: Custom optimizer with precision-scaled learning rates, per-group gradient control, and cosine warmup scheduling for the free energy objective.

## Architecture Overview

```
                    ┌─────────────┐
                    │  L3 (top)   │  μ[3], σ²[3]     Abstract/thematic
                    │  d = 64     │
                    └──────┬──────┘
                     f_pred│↓  ↑w_up
                    ┌──────┴──────┐
                    │  L2         │  μ[2], σ²[2]     Syntactic
                    │  d = 64     │
                    └──────┬──────┘
                     f_pred│↓  ↑w_up
                    ┌──────┴──────┐
                    │  L1         │  μ[1], σ²[1]     Lexical
                    │  d = 64     │
                    └──────┬──────┘
                     f_pred│↓  ↑w_up
                    ┌──────┴──────┐
  token → encoder → │  L0         │  μ[0] = e(x_t)   Sensory (clamped)
                    │  d = 32     │
                    └─────────────┘
                           │
                     μ[1..3] → decoder → logits → next token
```

## Information Flow

For each input token x_t, the following cycle executes `inner_steps` times:

1. **Sensory clamping**: μ[0] ← encoder(x_t)
2. **Top-down prediction**: p̂[l-1] = f_pred_l(μ[l]) for each layer
3. **Prediction error**: ε[l] = π[l] ⊙ (μ[l] - p̂[l])
4. **Belief update**: μ[l] ← μ[l] + g ⊙ (κ · drive[l])
5. **Decoding**: logits = decoder(concat(μ[1], μ[2], μ[3]))

## Document Index

| Document | Description |
|----------|-------------|
| [01_generative_model.md](./01_generative_model.md) | Hierarchical generative model specification |
| [02_free_energy.md](./02_free_energy.md) | Variational free energy: derivation and bounds |
| [03_belief_dynamics.md](./03_belief_dynamics.md) | Gated Euler belief update dynamics |
| [04_prism_optimizer.md](./04_prism_optimizer.md) | PRISM optimizer specification |
| [05_training.md](./05_training.md) | Training procedure and hyperparameters |
| [06_architecture_design.md](./06_architecture_design.md) | System architecture design and Mermaid diagrams |

## References

- Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience.
- Rao, R. & Ballard, D. (1999). *Predictive coding in the visual cortex.* Nature Neuroscience.
- Hasani, R. et al. (2021). *Closed-form Continuous-time Neural Networks.* Nature Machine Intelligence.
- Kingma, D. & Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR.
