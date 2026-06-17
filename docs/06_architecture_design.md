# System Architecture & Information Flow

**FERN Technical Specification — Document 06**

---

This document outlines the detailed system architecture of FERN (v0.6), describing how the embedding, hierarchical layers, temporal prediction transition, active inference loops, working memory cell, and decoder connect dynamically across sequence time steps.

---

## 1. Architectural Components & State

For a hierarchical generative model with levels $l \in \{0, \dots, L\}$, vocabulary size $V$, embedding dimension $d_0$, hidden layer dimensions $d_1, \dots, d_L$, and memory dimension $d_{\text{mem}}$:

### 1.1 Static Layers (Weights)
*   **Encoder**: $E \in \mathbb{R}^{V \times d_0}$ (Embedding table).
*   **Hierarchical Layers** ($l \in \{1, \dots, L\}$):
    *   **$f_{\text{pred}, l}$ (MLP)**: Takes $\mu[l] \in \mathbb{R}^{d_l}$ and outputs top-down prediction $\hat{p}[l-1] \in \mathbb{R}^{d_{l-1}}$.
    *   **$W_{\text{up}, l}$ (Linear)**: Projects prediction errors up from $\epsilon[l-1] \in \mathbb{R}^{d_{l-1}}$ to $\mathbb{R}^{d_l}$.
    *   **$W_{\text{gate}, l}$ (Linear)**: Projects concatenated $[\mu[l]; \epsilon[l]] \in \mathbb{R}^{2 d_l}$ to temporal gate control $g[l] \in \mathbb{R}^{d_l}$.
    *   **$W_{\text{rec}, l}$ (Linear, No Bias)**: Recurrent transition matrix mapping $\mu[l]_{t-1} \to \mu[l]_{t}^{\text{prior}} \in \mathbb{R}^{d_l}$.
*   **GRU Memory Cell**: Standard GRU cell updating working memory $h_t \in \mathbb{R}^{d_{\text{mem}}}$ using concatenated beliefs $\mu[1 \dots L]$.
*   **Decoder**: $W_{\text{dec}} \in \mathbb{R}^{(\sum_{l=1}^L d_l + d_{\text{mem}}) \times V}$ mapping current beliefs + working memory to output logits.

### 1.2 Dynamic States
*   **Belief states** $\mu[l] \in \mathbb{R}^{d_l}$ for all $l \in \{0, \dots, L\}$.
*   **Variance estimates** $\sigma^2[l] \in \mathbb{R}^{d_l}$ for all $l \in \{0, \dots, L\}$.
*   **Precision weights** $\pi[l] \in \mathbb{R}^{d_l}$ for all $l \in \{0, \dots, L\}$.
*   **Precision-weighted errors** $\epsilon[l] \in \mathbb{R}^{d_l}$ for all $l \in \{0, \dots, L\}$.
*   **GRU working memory** $h \in \mathbb{R}^{d_{\text{mem}}}$.

---

## 2. Dynamic Processing Flow

For each sequence token $x_t$ presented at sequence step $t$:

```
               [Time Step t-1] State
                      │
            (1) Recurrent Transition
                      ▼
               [Temporal Prior]
                      │
            (2) Sensory Clamping L0
                      ▼
            (3) Active Inference Loop (inner_steps)
             ┌─────────────────────────┐
             │  * Top-down predictions │
             │  * Sensory masking      │
             │  * Precision & Errors   │
             │  * Gated Euler updates  │
             └────────────┬────────────┘
                          ▼
            (4) GRU Memory Update
                          ▼
            (5) Logits Decoding & Output
```

### 2.1 Step 1: Across-Time Recurrent Transition
Before processing the new sensory token, the belief state of all levels $l \ge 1$ transitions dynamically via $W_{\text{rec}, l}$ to form the temporal predictive prior:

$$\mu[l] \leftarrow W_{\text{rec}, l} \cdot \mu[l]$$

### 2.2 Step 2: Sensory Clamping
The input token $x_t$ is mapped to continuous embedding space and clamped directly to the L0 belief state:

$$\mu[0] \leftarrow \text{Encoder}(x_t)$$

### 2.3 Step 3: Active Inference Loop
The hierarchy runs $S$ steps of active inference. For each inner step:
1.  **Top-Down Prediction**: Compute $\hat{p}[l-1] = f_{\text{pred}, l}(\mu[l])$ for $l \in \{1, \dots, L\}$.
2.  **Sensory Masking**: Compute raw error $e_{\text{raw}}[l] = \mu[l] - \hat{p}[l]$. If $l = 0$ (sensory input) and $x_t = \langle\text{pad}\rangle$ (ID 0), multiply $e_{\text{raw}}[0]$ by $0.0$, silencing bottom-up noise.
3.  **Precision Scaling**: Update EMA variance $\sigma^2[l]$ and compute precision $\pi[l] = 1 / (\sigma^2[l] + \epsilon_{\text{min}})$.
4.  **Error Propagation**: Compute precision-weighted errors $\epsilon[l] = \pi[l] \odot e_{\text{raw}}[l]$.
5.  **Euler Drive**: Compute layer-wise drive $d[l] = -\epsilon[l] + W_{\text{up}, l} \cdot \epsilon[l-1]$ (clamped to $[-D_{\text{max}}, D_{\text{max}}]$).
6.  **Belief Update**: Compute temporal Euler gate $g[l] = \sigma(W_{\text{gate}, l} \cdot [\mu[l]; \epsilon[l]])$ and update beliefs:
    $$\mu[l] \leftarrow \mu[l] + g[l] \odot (\kappa \cdot d[l])$$

### 2.4 Step 4: Memory Cell Update
The beliefs of all active levels $l \ge 1$ are concatenated to form the context vector $C_t = [\mu[1]; \dots; \mu[L]]$. The GRU cell updates the working memory:

$$h_t \leftarrow \text{GRUMemory}(C_t, h_{t-1})$$

### 2.5 Step 5: Decoding
Logits are decoded from the combined representation of active beliefs and working memory:

$$\text{logits}_t = W_{\text{dec}} \cdot [C_t; h_t] + b_{\text{dec}}$$

---

## 3. Detailed Architecture Diagram

The Mermaid diagram below visualizes the complete state, tensor shapes, and feedforward/feedback projections during one active inference sequence step:

```mermaid
graph TB
    subgraph Legend
        input_token["Discrete Token"]
        state_node(["Belief State (μ)"])
        err_node(["Weighted Error (ε)"])
        matrix_weight{"Weights/MLP"}
        precision_gate(("Precision/Gate"))
    end

    subgraph Inputs
        X["Token Input x_t"] -->|ID Lookup| EMB["Encoder Matrix"]
        EMB -->|shape: [B, d_0]| L0_MU
        MASK["Sensory Mask [B, 1]"]
    end

    subgraph L3_Top_Layer ["Level L3 (Abstract Layer)"]
        L3_MU(["μ[3] State [B, d_3]"])
        L3_ERR(["ε[3] Error [B, d_3]"])
        L3_GATE{{"W_gate_3"}}
        L3_REC{{"W_rec_3"}}
    end

    subgraph L2_Middle_Layer ["Level L2 (Syntactic Layer)"]
        L2_MU(["μ[2] State [B, d_2]"])
        L2_ERR(["ε[2] Error [B, d_2]"])
        L2_PRED{{"f_pred_3 (MLP)"}}
        L2_UP{{"W_up_3"}}
        L2_GATE{{"W_gate_2"}}
        L2_REC{{"W_rec_2"}}
    end

    subgraph L1_Lexical_Layer ["Level L1 (Lexical Layer)"]
        L1_MU(["μ[1] State [B, d_1]"])
        L1_ERR(["ε[1] Error [B, d_1]"])
        L1_PRED{{"f_pred_2 (MLP)"}}
        L1_UP{{"W_up_2"}}
        L1_GATE{{"W_gate_1"}}
        L1_REC{{"W_rec_1"}}
    end

    subgraph L0_Sensory_Layer ["Level L0 (Sensory Layer)"]
        L0_MU(["μ[0] State [B, d_0]"])
        L0_ERR(["ε[0] Error [B, d_0]"])
        L0_PRED{{"f_pred_1 (MLP)"}}
        L0_UP{{"W_up_1"}}
        L0_MASK_MULT(("Mask Mult"))
    end

    subgraph Temporal_Recurrency ["Recurrent Transitions (Across Steps)"]
        L3_MU_prev(["μ[3] at t-1"]) --> L3_REC --> L3_MU
        L2_MU_prev(["μ[2] at t-1"]) --> L2_REC --> L2_MU
        L1_MU_prev(["μ[1] at t-1"]) --> L1_REC --> L1_MU
    end

    subgraph Inference_Loops ["Active Inference Error and Prediction Flow"]
        %% Predictions
        L3_MU --> L2_PRED -->|p̂[2]| L2_ERR
        L2_MU --> L1_PRED -->|p̂[1]| L1_ERR
        L1_MU --> L0_PRED -->|p̂[0]| L0_ERR

        %% Raw Errors
        L2_MU --> L2_ERR
        L1_MU --> L1_ERR
        L0_MU --> L0_ERR

        %% Masking
        MASK -.->|multiply| L0_MASK_MULT
        L0_ERR --> L0_MASK_MULT --> L0_ERR_masked["Masked L0 Error"]

        %% Upward Projection
        L0_ERR_masked --> L0_UP --> L1_ERR
        L1_ERR --> L1_UP --> L2_ERR
        L2_ERR --> L2_UP --> L3_ERR

        %% Gate Euler Updates
        L3_ERR --> L3_GATE -->|gate g3| L3_MU
        L2_ERR --> L2_GATE -->|gate g2| L2_MU
        L1_ERR --> L1_GATE -->|gate g1| L1_MU
    end

    subgraph Memory_and_Outputs ["Working Memory & Decoding"]
        L1_MU --> CONCAT["Concat μ[1..3] [B, d_1+d_2+d_3]"]
        L2_MU --> CONCAT
        L3_MU --> CONCAT

        CONCAT --> GRU["GRU Memory Cell"]
        H_prev["Memory h_t-1"] --> GRU --> H_curr["Memory h_t [B, d_mem]"]

        CONCAT --> DEC_CONCAT["Decoder Concat [B, Σd + d_mem]"]
        H_curr --> DEC_CONCAT

        DEC_CONCAT --> DEC_LIN{{"Decoder Linear"}} --> LOGITS["Logits [B, V]"]
    end
```
