use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{embedding, Embedding, linear, Linear, Module, VarBuilder};

// ============================================================================
// MLP: Two-layer perceptron for top-down predictions f_pred
// ============================================================================

#[derive(Clone, Debug)]
pub struct MLP {
    pub linear1: Linear,
    pub linear2: Linear,
}

impl MLP {
    pub fn new(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Self> {
        let mid_dim = (in_dim + out_dim).max(32);
        let linear1 = linear(in_dim, mid_dim, vs.pp("linear1"))?;
        let linear2 = linear(mid_dim, out_dim, vs.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(x)?.relu()?;
        self.linear2.forward(&h)
    }
}

// ============================================================================
// LinearNoBias: Custom linear projection without bias
// ============================================================================

#[derive(Clone, Debug)]
pub struct LinearNoBias {
    pub weight: Tensor,
}

impl LinearNoBias {
    pub fn new(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Self> {
        let weight = vs.get_with_hints(
            (out_dim, in_dim),
            "weight",
            candle_nn::Init::Const(0.0),
        )?;
        Ok(Self { weight })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight.t()?)
    }
}

// ============================================================================
// HierarchicalLayer: One level of the generative hierarchy
// ============================================================================

#[derive(Clone, Debug)]
pub struct HierarchicalLayer {
    pub l: usize,
    /// Top-down prediction: f_pred(μ[l]) → p̂[l-1]
    pub f_pred: MLP,
    /// Upward error projection: w_up · ε[l-1] → Δμ[l]
    pub w_up: Linear,
    /// CfC gate: σ(w_gate · [μ[l]; ε[l]]) → g ∈ [0,1]
    pub w_gate: Linear,
    /// Temporal recurrent transition: w_rec · μ[l] → predicted state prior
    pub w_rec: LinearNoBias,
}

impl HierarchicalLayer {
    pub fn new(
        l: usize,
        d_prev: usize,
        d_curr: usize,
        vs: VarBuilder,
    ) -> Result<Self> {
        let f_pred = MLP::new(d_curr, d_prev, vs.pp("f_pred"))?;
        let w_up = linear(d_prev, d_curr, vs.pp("W_up"))?;
        let w_gate = linear(2 * d_curr, d_curr, vs.pp("W_gate"))?;
        let w_rec = LinearNoBias::new(d_curr, d_curr, vs.pp("W_rec"))?;
        Ok(Self { l, f_pred, w_up, w_gate, w_rec })
    }
}

// ============================================================================
// NetworkState: Beliefs μ, variance estimates σ², and gated working memory
// ============================================================================

#[derive(Clone, Debug)]
pub struct NetworkState {
    pub mu: Vec<Tensor>,
    pub sigma2: Vec<Tensor>,
    pub memory: Tensor,
}

impl NetworkState {
    pub fn init(batch_size: usize, d_layers: &[usize], d_mem: usize, device: &Device) -> Result<Self> {
        let mut mu = Vec::new();
        let mut sigma2 = Vec::new();
        for &d in d_layers {
            mu.push(Tensor::zeros((batch_size, d), DType::F32, device)?);
            sigma2.push(Tensor::ones((batch_size, d), DType::F32, device)?);
        }
        let memory = Tensor::zeros((batch_size, d_mem), DType::F32, device)?;
        Ok(Self { mu, sigma2, memory })
    }

    /// Compute mean precision π̄[l] = mean(1 / (σ²[l] + ε_min)) for each level.
    /// Used by PRISM optimizer for precision-scaled learning rates.
    pub fn mean_precisions(&self, epsilon_min: f64) -> Result<Vec<f64>> {
        let mut precisions = Vec::new();
        for s2 in &self.sigma2 {
            let denom = (s2 + epsilon_min)?;
            let pi = denom.recip()?;
            let mean_pi = pi.mean_all()?.to_scalar::<f32>()? as f64;
            precisions.push(mean_pi);
        }
        Ok(precisions)
    }
}

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_layers: Vec<usize>,
    pub kappa: f64,
    pub alpha: f64,
    pub epsilon_min: f64,
    pub max_drive: f64,
}

impl ModelConfig {
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let file = std::fs::File::create(path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create config file: {}", e)))?;
        serde_json::to_writer_pretty(file, self)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to serialize config: {}", e)))?;
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to open config file: {}", e)))?;
        let config = serde_json::from_reader(file)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to deserialize config: {}", e)))?;
        Ok(config)
    }
}

// ============================================================================
// GRUMemory: Gated Recurrent Unit working memory cell
// ============================================================================

#[derive(Clone, Debug)]
pub struct GRUMemory {
    pub w_z: Linear,
    pub u_z: Linear,
    pub w_r: Linear,
    pub u_r: Linear,
    pub w_h: Linear,
    pub u_h: Linear,
}

impl GRUMemory {
    pub fn new(in_dim: usize, mem_dim: usize, vs: VarBuilder) -> Result<Self> {
        let w_z = linear(in_dim, mem_dim, vs.pp("w_z"))?;
        let u_z = linear(mem_dim, mem_dim, vs.pp("u_z"))?;
        let w_r = linear(in_dim, mem_dim, vs.pp("w_r"))?;
        let u_r = linear(mem_dim, mem_dim, vs.pp("u_r"))?;
        let w_h = linear(in_dim, mem_dim, vs.pp("w_h"))?;
        let u_h = linear(mem_dim, mem_dim, vs.pp("u_h"))?;
        Ok(Self { w_z, w_r, w_h, u_z, u_r, u_h })
    }

    pub fn forward(&self, x: &Tensor, h: &Tensor) -> Result<Tensor> {
        let z = candle_nn::ops::sigmoid(&(self.w_z.forward(x)? + self.u_z.forward(h)?)?)?;
        let r = candle_nn::ops::sigmoid(&(self.w_r.forward(x)? + self.u_r.forward(h)?)?)?;
        
        let r_h = h.mul(&r)?;
        let h_tilde = (self.w_h.forward(x)? + self.u_h.forward(&r_h)?)?.tanh()?;
        
        let one_minus_z = z.neg()?.affine(1.0, 1.0)?;
        let h_new = (h.mul(&one_minus_z)? + z.mul(&h_tilde)?)?;
        Ok(h_new)
    }
}

// ============================================================================
// FreeEnergyRecurrentNetwork (FERN): The complete model
// ============================================================================

#[derive(Clone, Debug)]
pub struct FreeEnergyRecurrentNetwork {
    pub encoder: Embedding,
    pub layers: Vec<HierarchicalLayer>,
    pub decoder: Linear,
    pub d_layers: Vec<usize>,
    pub kappa: f64,
    pub alpha: f64,
    pub epsilon_min: f64,
    pub max_drive: f64,
    pub vocab_size: usize,
    
    // Gated Memory
    pub d_mem: usize,
    pub memory_cell: GRUMemory,
}

impl FreeEnergyRecurrentNetwork {
    pub fn new(
        config: ModelConfig,
        vs: VarBuilder,
    ) -> Result<Self> {
        assert!(config.d_layers.len() >= 2, "Need at least L0 (sensory) + L1");

        let encoder = embedding(config.vocab_size, config.d_layers[0], vs.pp("encoder"))?;

        let mut layers = Vec::new();
        for i in 1..config.d_layers.len() {
            let l_vs = vs.pp(&format!("layer_{}", i));
            layers.push(HierarchicalLayer::new(i, config.d_layers[i - 1], config.d_layers[i], l_vs)?);
        }

        let total_belief_dim: usize = config.d_layers[1..].iter().sum();
        let d_mem = 128;
        let memory_cell = GRUMemory::new(total_belief_dim, d_mem, vs.pp("memory_cell"))?;
        let decoder = linear(total_belief_dim + d_mem, config.vocab_size, vs.pp("decoder"))?;

        Ok(Self {
            encoder,
            layers,
            decoder,
            d_layers: config.d_layers,
            kappa: config.kappa,
            alpha: config.alpha,
            epsilon_min: config.epsilon_min,
            max_drive: config.max_drive,
            vocab_size: config.vocab_size,
            d_mem,
            memory_cell,
        })
    }
}

// ============================================================================
// LSTM Cell & Network Baselines
// ============================================================================

#[derive(Clone, Debug)]
pub struct LSTMCell {
    pub w_i: Linear, pub u_i: Linear,
    pub w_f: Linear, pub u_f: Linear,
    pub w_o: Linear, pub u_o: Linear,
    pub w_c: Linear, pub u_c: Linear,
}

impl LSTMCell {
    pub fn new(in_dim: usize, hidden_dim: usize, vs: VarBuilder) -> Result<Self> {
        let w_i = linear(in_dim, hidden_dim, vs.pp("w_i"))?;
        let u_i = linear(hidden_dim, hidden_dim, vs.pp("u_i"))?;
        let w_f = linear(in_dim, hidden_dim, vs.pp("w_f"))?;
        let u_f = linear(hidden_dim, hidden_dim, vs.pp("u_f"))?;
        let w_o = linear(in_dim, hidden_dim, vs.pp("w_o"))?;
        let u_o = linear(hidden_dim, hidden_dim, vs.pp("u_o"))?;
        let w_c = linear(in_dim, hidden_dim, vs.pp("w_c"))?;
        let u_c = linear(hidden_dim, hidden_dim, vs.pp("u_c"))?;
        Ok(Self { w_i, u_i, w_f, u_f, w_o, u_o, w_c, u_c })
    }

    pub fn forward(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
        let i = candle_nn::ops::sigmoid(&(self.w_i.forward(x)? + self.u_i.forward(h)?)?)?;
        let f = candle_nn::ops::sigmoid(&(self.w_f.forward(x)? + self.u_f.forward(h)?)?)?;
        let o = candle_nn::ops::sigmoid(&(self.w_o.forward(x)? + self.u_o.forward(h)?)?)?;
        let c_tilde = (self.w_c.forward(x)? + self.u_c.forward(h)?)?.tanh()?;
        
        let c_new = (c.mul(&f)? + c_tilde.mul(&i)?)?;
        let h_new = o.mul(&c_new.tanh()?)?;
        Ok((h_new, c_new))
    }
}

#[derive(Clone, Debug)]
pub struct LSTMNetwork {
    pub encoder: Embedding,
    pub lstm_cell: LSTMCell,
    pub decoder: Linear,
    pub d_embed: usize,
    pub d_hidden: usize,
    pub vocab_size: usize,
}

impl LSTMNetwork {
    pub fn new(vocab_size: usize, d_embed: usize, d_hidden: usize, vs: VarBuilder) -> Result<Self> {
        let encoder = embedding(vocab_size, d_embed, vs.pp("encoder"))?;
        let lstm_cell = LSTMCell::new(d_embed, d_hidden, vs.pp("lstm_cell"))?;
        let decoder = linear(d_hidden, vocab_size, vs.pp("decoder"))?;
        Ok(Self { encoder, lstm_cell, decoder, d_embed, d_hidden, vocab_size })
    }

    pub fn forward_sequence(&self, tokens: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = tokens.dims2()?;
        let device = tokens.device();
        let embeddings = self.encoder.forward(tokens)?;

        let mut h = Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?;
        let mut c = Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?;
        let mut all_logits = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let e_t = embeddings.narrow(1, t, 1)?.squeeze(1)?;
            let (h_new, c_new) = self.lstm_cell.forward(&e_t, &h, &c)?;
            h = h_new;
            c = c_new;
            let logits = self.decoder.forward(&h)?;
            all_logits.push(logits.unsqueeze(1)?);
        }

        Tensor::cat(&all_logits, 1)
    }
}

// ============================================================================
// GRU Network Baseline
// ============================================================================

#[derive(Clone, Debug)]
pub struct GRUNetwork {
    pub encoder: Embedding,
    pub gru_cell: GRUMemory,
    pub decoder: Linear,
    pub d_embed: usize,
    pub d_hidden: usize,
    pub vocab_size: usize,
}

impl GRUNetwork {
    pub fn new(vocab_size: usize, d_embed: usize, d_hidden: usize, vs: VarBuilder) -> Result<Self> {
        let encoder = embedding(vocab_size, d_embed, vs.pp("encoder"))?;
        let gru_cell = GRUMemory::new(d_embed, d_hidden, vs.pp("gru_cell"))?;
        let decoder = linear(d_hidden, vocab_size, vs.pp("decoder"))?;
        Ok(Self { encoder, gru_cell, decoder, d_embed, d_hidden, vocab_size })
    }

    pub fn forward_sequence(&self, tokens: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = tokens.dims2()?;
        let device = tokens.device();
        let embeddings = self.encoder.forward(tokens)?;

        let mut h = Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?;
        let mut all_logits = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let e_t = embeddings.narrow(1, t, 1)?.squeeze(1)?;
            h = self.gru_cell.forward(&e_t, &h)?;
            let logits = self.decoder.forward(&h)?;
            all_logits.push(logits.unsqueeze(1)?);
        }

        Tensor::cat(&all_logits, 1)
    }
}
