use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use crate::model::{FreeEnergyRecurrentNetwork, NetworkState, ModelConfig};
use crate::optimizer::{PrismConfig, PrismOptimizer};
use crate::step::forward_sequence;

pub struct Trainer {
    pub model: FreeEnergyRecurrentNetwork,
    #[allow(dead_code)]
    pub varmap: VarMap,
    pub inner_steps: usize,
    pub opt: PrismOptimizer,
}

impl Trainer {
    pub fn new(
        config: ModelConfig,
        prism_config: PrismConfig,
        inner_steps: usize,
        device: &Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = FreeEnergyRecurrentNetwork::new(
            config,
            vs,
        )?;

        {
            let data = varmap.data().lock().unwrap();
            for (name, var) in data.iter() {
                if name.contains("W_rec.weight") {
                    let shape = var.shape();
                    let dim = shape.dims()[0];
                    let dev = var.device();
                    let eye = Tensor::eye(dim, DType::F32, dev)?;
                    let rand = Tensor::rand(-1f32, 1f32, (dim, dim), dev)?;
                    let noise = rand.affine(0.01, -0.005)?; // uniform in [-0.005, 0.005]
                    let init_w = eye.affine(0.9, 0.0)?.add(&noise)?;
                    var.set(&init_w)?;
                }
            }
        }

        let opt = PrismOptimizer::from_varmap(&varmap, prism_config)?;

        Ok(Self {
            model,
            varmap,
            inner_steps,
            opt,
        })
    }

    /// Single training step.
    pub fn train_step(
        &mut self,
        input_tokens: &Tensor,
        target_tokens: &Tensor,
    ) -> Result<(f64, f64, f64, f64)> {
        let batch_size = input_tokens.dim(0)?;
        let device = input_tokens.device();

        // 1. Fresh state for this batch
        let mut state = NetworkState::init(batch_size, &self.model.d_layers, self.model.d_mem, device)?;

        // 2. Forward pass
        let (logits, avg_fe) = forward_sequence(
            &self.model,
            input_tokens,
            &mut state,
            self.inner_steps,
        )?;

        // 3. Cross-entropy loss
        let (b, s, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * s, v))?;
        let targets_flat = target_tokens.reshape(b * s)?;
        let loss_ce = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;

        // 4. Cosine-scheduled FE weight from PRISM
        let fe_w = self.opt.fe_weight();
        let total_loss = if fe_w > 0.0 {
            (loss_ce.clone() + avg_fe.clone().affine(fe_w, 0.0)?)?
        } else {
            loss_ce.clone()
        };

        // 5. Backward + PRISM step with precision info
        let grads = total_loss.backward()?;
        let precisions = state.mean_precisions(self.model.epsilon_min)?;
        self.opt.step(&grads, &precisions)?;

        // 6. Metrics
        let ce_val = loss_ce.to_scalar::<f32>()? as f64;
        let fe_val = avg_fe.to_scalar::<f32>()? as f64;
        let total_val = total_loss.to_scalar::<f32>()? as f64;

        Ok((total_val, ce_val, fe_val, fe_w))
    }
}
