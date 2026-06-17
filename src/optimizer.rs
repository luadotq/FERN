//! # PRISM Optimizer
//! **PR**ecision-**I**nformed **S**tochastic **M**omentum
//!
//! A custom optimizer designed for the FERN (Free Energy Recurrent Network)
//! architecture. Key features:
//!
//! 1. **Parameter group separation**: Different learning rates for prediction
//!    (f_pred), error path (w_up), gate (w_gate), and I/O (encoder/decoder).
//! 2. **Precision-scaled learning**: f_pred learning rate scales with the
//!    average precision π̄ of its hierarchical level via lr * π̄/(π̄+1).
//! 3. **Cosine warmup for FE weight**: Smoothly ramps fe_weight from 0 to
//!    target over warmup_steps, preventing CE/FE conflict at training start.
//! 4. **Per-variable gradient clipping**: Each parameter tensor is clipped
//!    independently, preventing any single group from dominating.

use candle_core::{Result, Tensor, Var};
use candle_nn::VarMap;

// Parameter role classification

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamRole {
    /// Generative prediction network f_pred (top-down path)
    Prediction,
    /// Temporal transition model w_rec (recurrent path)
    Recurrent,
    /// Error projection matrix w_up (bottom-up path)
    ErrorPath,
    /// CfC temporal gate w_gate
    Gate,
    /// Encoder / decoder (I/O interface)
    IO,
}

// Per-variable Adam state

struct VarState {
    var: Var,
    m: Tensor,               // First moment estimate (mean of gradients)
    v: Tensor,               // Second moment estimate (mean of squared gradients)
    role: ParamRole,
    layer_idx: Option<usize>, // Hierarchical level index (1, 2, 3, ...)
}

// PRISM configuration

#[derive(Clone, Debug)]
pub struct PrismConfig {
    // Per-group learning rates
    pub pred_lr: f64,
    pub rec_lr: f64,
    pub error_lr: f64,
    pub gate_lr: f64,
    pub io_lr: f64,

    // Adam hyperparameters
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,

    // Regularization
    pub weight_decay: f64,
    pub grad_clip: f64,
    pub rec_grad_clip: f64,

    // FE warmup schedule
    pub fe_weight_target: f64,
    pub warmup_steps: usize,

    // Precision-aware scaling
    pub precision_scaling: bool,
}

impl Default for PrismConfig {
    fn default() -> Self {
        Self {
            pred_lr: 1e-3,
            rec_lr: 3e-4,
            error_lr: 3e-4,
            gate_lr: 1e-3,
            io_lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-4,
            grad_clip: 1.0,
            rec_grad_clip: 0.5,
            fe_weight_target: 1.0,
            warmup_steps: 50,
            precision_scaling: true,
        }
    }
}

// PRISM Optimizer

pub struct PrismOptimizer {
    vars: Vec<VarState>,
    config: PrismConfig,
    step_t: usize,
}

impl PrismOptimizer {
    /// Build the optimizer from a VarMap, automatically classifying parameters
    /// into groups based on their variable names.
    pub fn from_varmap(varmap: &VarMap, config: PrismConfig) -> Result<Self> {
        let data = varmap.data().lock().unwrap();
        let mut vars = Vec::with_capacity(data.len());

        for (name, var) in data.iter() {
            let (role, layer_idx) = Self::classify_param(name);
            let tensor = var.as_tensor();
            let m = Tensor::zeros_like(tensor)?;
            let v = Tensor::zeros_like(tensor)?;
            vars.push(VarState {
                var: var.clone(),
                m,
                v,
                role,
                layer_idx,
            });
        }

        Ok(Self {
            vars,
            config,
            step_t: 0,
        })
    }

    /// Classify a parameter by its variable name into a role and layer index.
    fn classify_param(name: &str) -> (ParamRole, Option<usize>) {
        let layer_idx = name
            .strip_prefix("layer_")
            .and_then(|rest| {
                rest.chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse::<usize>()
                    .ok()
            });

        let role = if name.contains("W_rec") {
            ParamRole::Recurrent
        } else if name.contains("f_pred") {
            ParamRole::Prediction
        } else if name.contains("W_up") {
            ParamRole::ErrorPath
        } else if name.contains("W_gate") || name.contains("memory_cell") {
            ParamRole::Gate
        } else {
            ParamRole::IO
        };

        (role, layer_idx)
    }

    // Cosine warmup schedule for FE weight

    /// Returns the current scheduled FE weight.
    ///
    /// Uses cosine warmup: `w(t) = w_target · 0.5 · (1 − cos(π · t / T_warmup))`
    ///
    /// This ensures:
    /// - At t=0: fe_weight = 0 (pure CE training)
    /// - At t=T_warmup: fe_weight = fe_weight_target (full variational objective)
    /// - Smooth, differentiable transition
    pub fn fe_weight(&self) -> f64 {
        if self.config.warmup_steps == 0 || self.step_t >= self.config.warmup_steps {
            return self.config.fe_weight_target;
        }
        let t = self.step_t as f64 / self.config.warmup_steps as f64;
        self.config.fe_weight_target * 0.5 * (1.0 - (std::f64::consts::PI * t).cos())
    }

    /// Current training step counter.
    #[allow(dead_code)]
    pub fn current_step(&self) -> usize {
        self.step_t
    }

    // Optimizer step: per-group Adam with precision scaling

    /// Performs one optimization step.
    /// * `grads` — Gradient store from `tensor.backward()`
    /// * `layer_precisions` — Mean precision π̄[l] for each hierarchical level,
    ///   computed from the network state after the forward pass.
    ///   Length should match `d_layers.len()`.
    pub fn step(
        &mut self,
        grads: &candle_core::backprop::GradStore,
        layer_precisions: &[f64],
    ) -> Result<()> {
        self.step_t += 1;
        let t = self.step_t as i32;

        // Precompute bias correction factors (shared across all vars)
        let bc1 = 1.0 - self.config.beta1.powi(t);
        let bc2 = 1.0 - self.config.beta2.powi(t);

        for vs in &mut self.vars {
            let theta = vs.var.as_tensor();
            let grad = match grads.get(theta) {
                Some(g) => g,
                None => continue,
            };

            let clip_limit = match vs.role {
                ParamRole::Recurrent => self.config.rec_grad_clip,
                _ => self.config.grad_clip,
            };
            let grad = Self::clip_gradient(&grad, clip_limit)?;

            let m_new = vs.m.affine(self.config.beta1, 0.0)?
                .add(&grad.affine(1.0 - self.config.beta1, 0.0)?)?;

            let grad_sq = grad.sqr()?;
            let v_new = vs.v.affine(self.config.beta2, 0.0)?
                .add(&grad_sq.affine(1.0 - self.config.beta2, 0.0)?)?;

            vs.m = m_new;
            vs.v = v_new;

            let m_hat = vs.m.affine(1.0 / bc1, 0.0)?;
            let v_hat = vs.v.affine(1.0 / bc2, 0.0)?;

            let base_lr = match vs.role {
                ParamRole::Prediction => self.config.pred_lr,
                ParamRole::Recurrent  => self.config.rec_lr,
                ParamRole::ErrorPath  => self.config.error_lr,
                ParamRole::Gate       => self.config.gate_lr,
                ParamRole::IO         => self.config.io_lr,
            };

            let lr = if self.config.precision_scaling
                && (vs.role == ParamRole::Prediction || vs.role == ParamRole::Recurrent)
            {
                if let Some(li) = vs.layer_idx {
                    if li < layer_precisions.len() {
                        let pi = layer_precisions[li];
                        let scaled = base_lr * pi / (pi + 1.0);
                        scaled.max(base_lr * 0.1) // floor: 10% of base_lr
                    } else {
                        base_lr
                    }
                } else {
                    base_lr
                }
            } else {
                base_lr
            };

            let denom = (v_hat.sqrt()? + self.config.eps)?;
            let update = m_hat.div(&denom)?.affine(lr, 0.0)?;

            let new_theta = if self.config.weight_decay > 0.0 {
                let decay = 1.0 - lr * self.config.weight_decay;
                (theta.affine(decay, 0.0)? - update)?
            } else {
                (theta - update)?
            };

            vs.var.set(&new_theta)?;
        }

        Ok(())
    }

    /// Clip gradient tensor to max L2 norm.
    /// Returns the original tensor if norm ≤ max_norm (zero-copy).
    fn clip_gradient(grad: &Tensor, max_norm: f64) -> Result<Tensor> {
        let norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let norm = norm_sq.sqrt();
        if norm > max_norm && norm > 0.0 {
            grad.affine(max_norm / norm, 0.0)
        } else {
            Ok(grad.clone())
        }
    }
}
