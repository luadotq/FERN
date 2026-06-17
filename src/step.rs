use candle_core::{DType, Result, Tensor, Module};
use crate::model::{FreeEnergyRecurrentNetwork, NetworkState};

pub fn run_inner_inference(
    model: &FreeEnergyRecurrentNetwork,
    e_t: &Tensor,
    tokens_t: Option<&Tensor>,
    state: &mut NetworkState,
    inner_steps: usize,
) -> Result<Tensor> {
    let batch_size = e_t.dim(0)?;
    let num_layers = model.layers.len(); // number of Layer objects
    let device = e_t.device();

    let mut total_free_energy = Tensor::zeros((), DType::F32, device)?;

    for _step in 0..inner_steps {
        state.mu[0] = e_t.clone();

        // Top-down predictions
        // p_hat[i] = prediction FOR level i FROM level i+1
        let mut p_hat = Vec::with_capacity(num_layers + 1);
        for l in 0..num_layers {
            let pred = model.layers[l].f_pred.forward(&state.mu[l + 1])?;
            p_hat.push(pred);
        }
        // Top level (L_max)
        let top_dim = model.d_layers[num_layers];
        p_hat.push(Tensor::zeros((batch_size, top_dim), DType::F32, device)?);

        // Prediction errors, precision, and Free Energy
        let mut epsilon = Vec::with_capacity(num_layers + 1);
        let mut step_free_energy = Tensor::zeros((), DType::F32, device)?;

        // Compute mask for sensory error masking on pad tokens
        let mask = if let Some(t_tensor) = tokens_t {
            let f_tensor = t_tensor.to_dtype(DType::F32)?;
            f_tensor.minimum(&Tensor::ones_like(&f_tensor)?)?
        } else {
            Tensor::ones((batch_size, 1), DType::F32, device)?
        };

        for l in 0..=num_layers {
            let mu_l = &state.mu[l];
            let p_hat_l = &p_hat[l];
            let mut err_raw = (mu_l - p_hat_l)?;
            if l == 0 {
                err_raw = err_raw.broadcast_mul(&mask)?;
            }
            let err_sq = err_raw.sqr()?;

            let old_sigma2 = &state.sigma2[l];
            let new_sigma2 = (old_sigma2 * model.alpha)?
                .add(&(err_sq.clone() * (1.0 - model.alpha))?)?;

            let denom = (&new_sigma2 + model.epsilon_min)?;
            let pi_l = denom.recip()?;

            // detach σ² for state storage
            state.sigma2[l] = new_sigma2.detach();

            // Precision-weighted error
            let eps_l = pi_l.mul(&err_raw)?;
            if l > 0 {
                let d_l = model.d_layers[l] as f64;
                let accuracy = err_sq.mul(&pi_l)?;
                let mu_sq = mu_l.sqr()?;
                let log_denom = denom.log()?;
                let ones = Tensor::ones_like(&denom)?;
                let complexity = (&denom + &mu_sq)?.sub(&ones)?.sub(&log_denom)?;

                let f_l = if l < num_layers {
                    // Intermediate levels: full accuracy + KL
                    accuracy.add(&complexity)?
                } else {
                    // Top level (L_max): KL only
                    complexity
                };

                // Normalize by layer dimension and add to step total
                let sum_f_l = f_l.sum_all()?.affine(1.0 / d_l, 0.0)?;
                step_free_energy = step_free_energy.add(&sum_f_l)?;
            }

            epsilon.push(eps_l);
        }

        // Average FE over batch
        let step_fe = step_free_energy.affine(1.0 / (batch_size as f64), 0.0)?;
        total_free_energy = total_free_energy.add(&step_fe)?;

        // Gated Euler belief update
        let mut new_mu = state.mu.clone();
        for l in 1..=num_layers {
            let layer = &model.layers[l - 1];
            let mu_l = &state.mu[l];
            let eps_l = &epsilon[l];
            let eps_prev = &epsilon[l - 1];

            // Drive: d = -ε[l] + w_up · ε[l-1]
            let up_err = layer.w_up.forward(eps_prev)?;
            let drive = (up_err - eps_l)?;

            // BUG fix: clamp drive magnitude to prevent explosion
            let bound = Tensor::ones_like(&drive)?.affine(model.max_drive, 0.0)?;
            let neg_bound = bound.neg()?;
            let drive = drive.minimum(&bound)?.maximum(&neg_bound)?;

            let gate_in = Tensor::cat(&[mu_l, eps_l], 1)?;
            let g = candle_nn::ops::sigmoid(&layer.w_gate.forward(&gate_in)?)?;

            let scaled_drive = drive.affine(model.kappa, 0.0)?;
            let gated_step = g.mul(&scaled_drive)?;
            new_mu[l] = mu_l.add(&gated_step)?;
        }
        state.mu = new_mu;
    }

    // Average FE over inner steps
    let avg_fe = total_free_energy.affine(1.0 / (inner_steps as f64), 0.0)?;
    Ok(avg_fe)
}

pub fn forward_sequence(
    model: &FreeEnergyRecurrentNetwork,
    tokens: &Tensor,
    state: &mut NetworkState,
    inner_steps: usize,
) -> Result<(Tensor, Tensor)> {
    let (_batch_size, seq_len) = tokens.dims2()?;
    let device = tokens.device();

    let mut all_logits = Vec::with_capacity(seq_len);
    let mut total_fe = Tensor::zeros((), DType::F32, device)?;

    let embeddings = model.encoder.forward(tokens)?;

    for t in 0..seq_len {
        let e_t = embeddings.narrow(1, t, 1)?.squeeze(1)?;

        if t > 0 {
            let mut next_mu = state.mu.clone();
            for l in 1..model.d_layers.len() {
                let layer = &model.layers[l - 1];
                next_mu[l] = layer.w_rec.forward(&state.mu[l])?;
            }
            state.mu = next_mu;
        }

        let tokens_t = tokens.narrow(1, t, 1)?;
        let fe = run_inner_inference(model, &e_t, Some(&tokens_t), state, inner_steps)?;
        total_fe = total_fe.add(&fe)?;

        // Decode from beliefs μ[1..L_max]
        let beliefs: Vec<&Tensor> = (1..model.d_layers.len())
            .map(|l| &state.mu[l])
            .collect();
        let concat = Tensor::cat(&beliefs, 1)?;
        
        state.memory = model.memory_cell.forward(&concat, &state.memory)?;

        let decode_in = Tensor::cat(&[&concat, &state.memory], 1)?;
        let logits = model.decoder.forward(&decode_in)?;
        all_logits.push(logits.unsqueeze(1)?);
    }

    let seq_logits = Tensor::cat(&all_logits, 1)?;
    let avg_fe = total_fe.affine(1.0 / (seq_len as f64), 0.0)?;

    Ok((seq_logits, avg_fe))
}
