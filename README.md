# FERN: Free Energy Recurrent Network

FERN is a recurrent neural network framework implemented in Rust. Unlike transformer-based models that rely on self-attention mechanisms with quadratic complexity, FERN maintains O(1) memory per time step through a hierarchical generative model that continuously minimizes variational free energy.

The framework processes sequential data by maintaining a hierarchy of belief states that are iteratively updated through active inference. This process compares top-down generative predictions with bottom-up sensory inputs to calculate precision-weighted prediction errors, which in turn drive state updates.

## System Architecture

### 1. Hierarchical Generative Model
The architecture consists of a hierarchy of latent belief layers. Each layer consists of:
* **Predictor Module**: A two-layer MLP that projects beliefs from the current layer down to predict the state of the layer below.
* **Error Projection Module**: A linear layer mapping raw prediction errors from the layer below up to the current layer.
* **Temporal Gating**: A gating block that processes concatenated belief and error states to modulate the integration step size (CfC-inspired).
* **Sensory Layer**: The bottom-most layer, which receives discrete input tokens mapped to continuous vectors via an embedding encoder.
* **Logit Decoder**: A linear projection layer mapping concatenated latent beliefs to next-token predictions.

### 2. State Management & Precision Tracking
The framework manages two main types of state during processing:
* **Belief States**: Dynamic tensors representing the active posterior mean estimates for each hierarchical level.
* **Precision Tensors**: Dynamic estimates representing inverse variance tracking. These are calculated via an exponential moving average (EMA) of squared raw prediction errors, ensuring that layers with higher predictability receive higher precision weights.

### 3. Computation Loop (Timestep Step)
For each input token in a sequence, the framework executes a multi-step inner optimization loop:
1. **Sensory Clamping**: Clamps the sensory representation layer to the embedding vector of the active token.
2. **Inner Inference Iterations**: Runs a configurable number of inner loops to minimize free energy by propagating top-down predictions, calculating bottom-up errors, and updating active beliefs.
3. **Temporal Gate Modulation**: Uses the gating parameters to update belief states over time steps, allowing working memory persistence.
4. **Next-Token Projection**: Passes the final belief state hierarchy to the logit decoder to produce predictions.

### 4. PRISM Optimizer
The Precision-scaled Learning Rate Optimizer (PRISM) controls parameter updates based on precision estimates:
* **Precision-Weighted Learning Rates**: Automatically scales gradients so that layers with higher precision receive larger parameter updates.
* **Multi-Group Parameters**: Separates learning rates and weight decays for predictors, temporal transition layers, gates, and standard linear layers.
* **Free Energy Weight Scheduling**: Implements a cosine warm-up schedule for the objective function weight.

---

## Technical Specifications

### File Structure
* `src/model.rs`: Core structures for HierarchicalLayer, LSTM/GRU baseline networks, and network state representations.
* `src/optimizer.rs`: PRISM Optimizer configuration and parameter update routines.
* `src/step.rs`: Implementation of inner inference optimization loops and sequence-level forward passes.
* `src/train.rs`: Gradient collection, training step controllers, and metric calculations.
* `src/main.rs`: CLI subcommand definition, synthetic data generators, and test evaluation matrices.

### Pre-trained Models and Reports
All pre-trained benchmark models (configuration JSON files and `.safetensors` weight checkpoints) and generated scientific reports are located in the preview directory.

---

## CLI Command Reference

Execute commands using `cargo run -- <command> [args]`.

### 1. Train
Train the framework on the synthetic multi-task mixture dataset (pattern completion, delayed copying, and secret recall).
```bash
cargo run --release -- train --epochs 500 --batch-size 32 --checkpoint checkpoint_path
```

### 2. Generate
Generate sequence extensions from a prompt string.
```bash
cargo run --release -- generate --checkpoint checkpoint_path --prompt "the cat" --max-tokens 15
```

### 3. Interactive REPL
Start an interactive console shell to query the model.
```bash
cargo run --release -- repl --checkpoint checkpoint_path --temperature 0.7
```

### 4. Evaluate (Bench)
Evaluate a saved checkpoint against the standard and big task suites.
```bash
cargo run --release -- bench --checkpoint checkpoint_path
```

### 5. Full Autobenchmark
Run the scientific training pipeline for Full FERN alongside ablation baselines (No Free Energy, No Precision Scaling) and write a report.
```bash
cargo run --release -- autobench --epochs 800 --batch-size 32 --report autobench_report.md
```

### 6. Per-Task Benchmarking
Train independent FERN models on each task to isolate performance and prevent task interference.
```bash
cargo run --release -- pertask-autobench --epochs 300 --batch-size 32 --report pertask_report.md
```

### 7. LSTM/GRU Baseline Comparison
Train and compare standard LSTM and GRU baselines against FERN under matched parameter budgets.
```bash
cargo run --release -- compare --epochs 300 --batch-size 32 --report compare_report.md
```
During execution, this command:
1. Trains the active FERN model and counts its parameters.
2. Dynamically searches for LSTM and GRU hidden dimensions that yield matching parameter counts.
3. Trains and evaluates the baselines under identical budgets.
4. Outputs the model parameter counts to stdout and generates a comprehensive markdown comparison report.

## License
This project is licensed under the MIT License.
