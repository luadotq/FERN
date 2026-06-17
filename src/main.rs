mod model;
mod optimizer;
mod step;
mod train;

use candle_core::{Device, DType, Result, Tensor, Module};
use candle_nn::{VarBuilder, VarMap, Optimizer};
use clap::{Parser, Subcommand};
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use std::io::{self, Write};

use crate::model::{FreeEnergyRecurrentNetwork, NetworkState, ModelConfig, LSTMNetwork, GRUNetwork};
use crate::optimizer::PrismConfig;
use crate::train::Trainer;

#[derive(Parser, Debug)]
#[command(name = "fern", version = "0.3.0", about = "Free Energy Recurrent Network Framework")]
struct CliArgs {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Train {
        #[arg(long, default_value = "500")]
        epochs: usize,

        #[arg(long, default_value = "32")]
        batch_size: usize,

        #[arg(long, default_value = "checkpoint")]
        checkpoint: String,
    },
    Generate {
        #[arg(long, default_value = "checkpoint")]
        checkpoint: String,

        #[arg(long, default_value = "the cat")]
        prompt: String,

        #[arg(long, default_value = "15")]
        max_tokens: usize,

        #[arg(long, default_value = "0.7")]
        temperature: f64,

        #[arg(long)]
        top_k: Option<usize>,
    },
    Repl {
        #[arg(long, default_value = "checkpoint")]
        checkpoint: String,

        #[arg(long, default_value = "0.7")]
        temperature: f64,

        #[arg(long)]
        top_k: Option<usize>,
    },
    Bench {
        #[arg(long, default_value = "checkpoint")]
        checkpoint: String,
    },
    Autobench {
        #[arg(long, default_value = "800")]
        epochs: usize,

        #[arg(long, default_value = "32")]
        batch_size: usize,

        #[arg(long, default_value = "autobench_report.md")]
        report: String,

        #[arg(long, default_value = "autobench_checkpoint")]
        checkpoint: String,
    },
    PertaskAutobench {
        #[arg(long, default_value = "300")]
        epochs: usize,

        #[arg(long, default_value = "32")]
        batch_size: usize,

        #[arg(long, default_value = "pertask_report.md")]
        report: String,
    },
    Compare {
        #[arg(long, default_value = "300")]
        epochs: usize,

        #[arg(long, default_value = "32")]
        batch_size: usize,

        #[arg(long, default_value = "compare_report.md")]
        report: String,
    },
}

fn vocab_to_word(id: u32) -> &'static str {
    match id {
        0 => "<pad>", 1 => "the", 2 => "cat", 3 => "sat", 4 => "on",
        5 => "mat",  6 => "dog", 7 => "ran", 8 => "in",  9 => "park",
        10 => "<eos>",
        11 => "a", 12 => "b", 13 => "c", 14 => "d", 15 => "copy",
        16 => "key_x", 17 => "key_y", 18 => "secret_x", 19 => "secret_y",
        20 => "key_z", 21 => "secret_z", 22 => "key_w", 23 => "secret_w",
        24 => "task_copy", 25 => "task_pattern", 26 => "task_secret",
        _ => "<unk>",
    }
}

fn word_to_vocab(word: &str) -> Option<u32> {
    match word {
        "<pad>" => Some(0),
        "the" => Some(1),
        "cat" => Some(2),
        "sat" => Some(3),
        "on" => Some(4),
        "mat" => Some(5),
        "dog" => Some(6),
        "ran" => Some(7),
        "in" => Some(8),
        "park" => Some(9),
        "<eos>" => Some(10),
        "a" => Some(11),
        "b" => Some(12),
        "c" => Some(13),
        "d" => Some(14),
        "copy" => Some(15),
        "key_x" => Some(16),
        "key_y" => Some(17),
        "secret_x" => Some(18),
        "secret_y" => Some(19),
        "key_z" => Some(20),
        "secret_z" => Some(21),
        "key_w" => Some(22),
        "secret_w" => Some(23),
        "task_copy" => Some(24),
        "task_pattern" => Some(25),
        "task_secret" => Some(26),
        _ => None,
    }
}

fn tokenize_prompt(prompt: &str) -> Vec<u32> {
    prompt
        .split_whitespace()
        .map(|w| {
            word_to_vocab(&w.to_lowercase())
                .unwrap_or(0) // Default to <pad> for unknown words
        })
        .collect()
}

fn generate_synthetic_data(
    batch_size: usize,
    epoch: usize,
    task_type_override: Option<usize>,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut rng = rand::thread_rng();
    let mut inputs: Vec<u32> = Vec::new();
    let mut targets: Vec<u32> = Vec::new();
    
    let task_type = if let Some(t) = task_type_override {
        t
    } else if epoch <= 150 {
        let choices = [0, 2, 4];
        let weights = [30, 40, 30];
        let dist = WeightedIndex::new(&weights).unwrap();
        choices[dist.sample(&mut rng)]
    } else if epoch <= 250 {
        let choices = [0, 2, 4, 5];
        let weights = [20, 30, 30, 20];
        let dist = WeightedIndex::new(&weights).unwrap();
        choices[dist.sample(&mut rng)]
    } else {
        rand::Rng::gen_range(&mut rng, 0..6)
    };
    
    let seq_len = match task_type {
        0 => 6,  // Standard Pattern Completion
        1 => 12, // Big Pattern Completion
        2 => {   // Standard Copy with Delay
            let delay_choices = vec![1, 3, 5];
            let d = delay_choices[rand::Rng::gen_range(&mut rng, 0..3)];
            5 + d
        },
        3 => {   // Big Copy with Delay
            let delay_choices = vec![5, 10, 15];
            let d = delay_choices[rand::Rng::gen_range(&mut rng, 0..3)];
            9 + d
        },
        4 => 6,  // Standard Secret Recall
        _ => {   // Big Secret Recall
            if rand::Rng::gen_bool(&mut rng, 0.5) { 10 } else { 8 }
        }
    };
    
    for _b in 0..batch_size {
        let (input, target) = match task_type {
            0 => {
                // Task 1: Standard Pattern Completion
                let pattern_a = [1u32, 2, 3, 4, 1, 5]; // "the cat sat on the mat"
                let pattern_b = [1u32, 6, 7, 8, 1, 9]; // "the dog ran in the park"
                let is_a = rand::Rng::gen_bool(&mut rng, 0.5);
                let pattern = if is_a { &pattern_a } else { &pattern_b };
                
                let mut input = vec![0u32; 6];
                input[0..6].copy_from_slice(pattern);

                let mut target = vec![0u32; 6];
                target[0..5].copy_from_slice(&pattern[1..6]);
                target[5] = 10; // <eos>
                (input, target)
            },
            1 => {
                // Task 2: Big Pattern Completion
                let pattern_a = [1u32, 2, 3, 4, 1, 5, 6, 7, 8, 1, 9, 2];
                let pattern_b = [11u32, 6, 7, 8, 1, 9, 2, 3, 4, 1, 5, 6];
                let pattern_c = [6u32, 3, 4, 1, 5, 2, 7, 8, 1, 9, 1, 2];
                let pattern_d = [2u32, 7, 8, 1, 9, 6, 3, 4, 1, 5, 1, 6];
                let r = rand::Rng::gen_range(&mut rng, 0..4);
                let pattern = match r {
                    0 => &pattern_a,
                    1 => &pattern_b,
                    2 => &pattern_c,
                    _ => &pattern_d,
                };
                
                let mut input = vec![0u32; 12];
                input[0..12].copy_from_slice(pattern);

                let mut target = vec![0u32; 12];
                target[0..11].copy_from_slice(&pattern[1..12]);
                target[11] = 10; // <eos>
                (input, target)
            },
            2 => {
                // Task 3: Standard Copy with Delay
                let d = seq_len - 5;
                let x1 = rand::Rng::gen_range(&mut rng, 11..=14);
                let x2 = rand::Rng::gen_range(&mut rng, 11..=14);
                
                let mut input = vec![0u32; seq_len];
                let mut target = vec![0u32; seq_len];

                input[0] = x1;
                input[1] = x2;
                target[0] = x2;
                
                input[2 + d] = 15;
                input[3 + d] = x1;
                input[4 + d] = x2;

                target[1 + d] = 15;
                target[2 + d] = x1;
                target[3 + d] = x2;
                target[4 + d] = 10;
                (input, target)
            },
            3 => {
                // Task 4: Big Copy with Delay
                let d = seq_len - 9;
                let x1 = rand::Rng::gen_range(&mut rng, 11..=14);
                let x2 = rand::Rng::gen_range(&mut rng, 11..=14);
                let x3 = rand::Rng::gen_range(&mut rng, 11..=14);
                let x4 = rand::Rng::gen_range(&mut rng, 11..=14);
                
                let mut input = vec![0u32; seq_len];
                let mut target = vec![0u32; seq_len];

                input[0] = x1;
                input[1] = x2;
                input[2] = x3;
                input[3] = x4;
                target[0] = x2;
                target[1] = x3;
                target[2] = x4;
                
                input[4 + d] = 15;
                input[5 + d] = x1;
                input[6 + d] = x2;
                input[7 + d] = x3;
                input[8 + d] = x4;

                target[3 + d] = 15;
                target[4 + d] = x1;
                target[5 + d] = x2;
                target[6 + d] = x3;
                target[7 + d] = x4;
                target[8 + d] = 10;
                (input, target)
            },
            4 => {
                // Task 5: Standard Secret Recall
                let k1 = rand::Rng::gen_range(&mut rng, 16..=17);
                let s1 = if k1 == 16 { 18 } else { 19 };
                let k2 = if k1 == 16 { 17 } else { 16 };
                let s2 = if k2 == 16 { 18 } else { 19 };

                let query_key = if rand::Rng::gen_bool(&mut rng, 0.5) { k1 } else { k2 };
                let query_secret = if query_key == 16 { 18 } else { 19 };

                let mut input = vec![0u32; 6];
                input[0] = k1;
                input[1] = s1;
                input[2] = k2;
                input[3] = s2;
                input[4] = query_key;

                let mut target = vec![0u32; 6];
                target[0] = s1;
                target[1] = k2;
                target[2] = s2;
                target[3] = query_key;
                target[4] = query_secret;
                target[5] = 10;
                (input, target)
            },
            _ => {
                // Task 6: Big Secret Recall
                let mut input = vec![0u32; seq_len];
                let mut target = vec![0u32; seq_len];
                if seq_len == 10 {
                    let mut pairs = vec![
                        (16, 18),
                        (17, 19),
                        (20, 21),
                        (22, 23),
                    ];
                    for i in (1..pairs.len()).rev() {
                        let j = rand::Rng::gen_range(&mut rng, 0..=i);
                        pairs.swap(i, j);
                    }

                    let q_idx = rand::Rng::gen_range(&mut rng, 0..4);
                    let (qkey, qsec) = pairs[q_idx];

                    input[0] = pairs[0].0; input[1] = pairs[0].1;
                    input[2] = pairs[1].0; input[3] = pairs[1].1;
                    input[4] = pairs[2].0; input[5] = pairs[2].1;
                    input[6] = pairs[3].0; input[7] = pairs[3].1;
                    input[8] = qkey;

                    target[0] = pairs[0].1;
                    target[1] = pairs[1].0; target[2] = pairs[1].1;
                    target[3] = pairs[2].0; target[4] = pairs[2].1;
                    target[5] = pairs[3].0; target[6] = pairs[3].1;
                    target[7] = qkey;
                    target[8] = qsec;
                    target[9] = 10;
                } else {
                    let mut pairs = vec![
                        (16, 18),
                        (17, 19),
                        (20, 21),
                    ];
                    for i in (1..pairs.len()).rev() {
                        let j = rand::Rng::gen_range(&mut rng, 0..=i);
                        pairs.swap(i, j);
                    }

                    let q_idx = rand::Rng::gen_range(&mut rng, 0..3);
                    let (qkey, qsec) = pairs[q_idx];

                    input[0] = pairs[0].0; input[1] = pairs[0].1;
                    input[2] = pairs[1].0; input[3] = pairs[1].1;
                    input[4] = pairs[2].0; input[5] = pairs[2].1;
                    input[6] = qkey;

                    target[0] = pairs[0].1;
                    target[1] = pairs[1].0; target[2] = pairs[1].1;
                    target[3] = pairs[2].0; target[4] = pairs[2].1;
                    target[5] = qkey;
                    target[6] = qsec;
                    target[7] = 10;
                }
                (input, target)
            }
        };

        let prefix = match task_type {
            0 | 1 => 25, // TASK_PATTERN
            2 | 3 => 24, // TASK_COPY
            4 | 5 => 26, // TASK_SECRET
            _ => unreachable!(),
        };

        let mut prepended_input = vec![prefix];
        prepended_input.extend(&input);
        
        let mut prepended_target = vec![input[0]];
        prepended_target.extend(&target);

        inputs.extend(prepended_input);
        targets.extend(prepended_target);
    }

    let inputs_tensor = Tensor::from_vec(inputs, (batch_size, seq_len + 1), device)?;
    let targets_tensor = Tensor::from_vec(targets, (batch_size, seq_len + 1), device)?;

    Ok((inputs_tensor, targets_tensor))
}

fn sample_from_logits(logits: &Tensor, temperature: f64, top_k: Option<usize>) -> Result<u32> {
    let logits = logits.squeeze(0)?; // shape: [vocab_size]
    if temperature <= 0.0 {
        let argmax = logits.argmax(0)?;
        return Ok(argmax.to_scalar::<u32>()?);
    }

    let scaled_logits = logits.affine(1.0 / temperature, 0.0)?;
    let probs = candle_nn::ops::softmax(&scaled_logits, 0)?;
    let mut probs_vec = probs.to_vec1::<f32>()?;

    if let Some(k) = top_k {
        if k > 0 && k < probs_vec.len() {
            let mut indexed_probs: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            let threshold = indexed_probs[k - 1].1;
            let mut sum = 0.0;
            for p in probs_vec.iter_mut() {
                if *p < threshold {
                    *p = 0.0;
                } else {
                    sum += *p;
                }
            }
            if sum > 0.0 {
                for p in probs_vec.iter_mut() {
                    *p /= sum;
                }
            }
        }
    }

    let mut rng = thread_rng();
    let dist = WeightedIndex::new(&probs_vec)
        .map_err(|e| candle_core::Error::Msg(format!("WeightedIndex error: {}", e)))?;
    let idx = dist.sample(&mut rng) as u32;
    Ok(idx)
}

pub trait EvaluableModel {
    fn generate(&self, prompt: &[u32], max_tokens: usize, device: &Device) -> Result<Vec<u32>>;
}

impl EvaluableModel for FreeEnergyRecurrentNetwork {
    fn generate(&self, prompt: &[u32], max_tokens: usize, device: &Device) -> Result<Vec<u32>> {
        generate_tokens(self, prompt, max_tokens, 3, 0.0, None, device)
    }
}

pub fn generate_tokens_lstm(
    model: &LSTMNetwork,
    prompt: &[u32],
    max_tokens: usize,
    device: &Device,
) -> Result<Vec<u32>> {
    let mut h = Tensor::zeros((1, model.d_hidden), DType::F32, device)?;
    let mut c = Tensor::zeros((1, model.d_hidden), DType::F32, device)?;
    let mut last_logits = Tensor::zeros((1, model.vocab_size), DType::F32, device)?;

    // Process prompt
    for &token in prompt {
        let next_tensor = Tensor::from_vec(vec![token], (1, 1), device)?;
        let e_t = model.encoder.forward(&next_tensor)?.squeeze(1)?;
        let (h_new, c_new) = model.lstm_cell.forward(&e_t, &h, &c)?;
        h = h_new;
        c = c_new;
        last_logits = model.decoder.forward(&h)?;
    }

    let mut next_logits = last_logits;
    let mut generated = Vec::new();

    // Generate autoregressively
    for _ in 0..max_tokens {
        let sampled_token = sample_from_logits(&next_logits, 0.0, None)?;
        generated.push(sampled_token);

        if sampled_token == 10 {
            break;
        }

        let next_tensor = Tensor::from_vec(vec![sampled_token], (1, 1), device)?;
        let e_t = model.encoder.forward(&next_tensor)?.squeeze(1)?;
        let (h_new, c_new) = model.lstm_cell.forward(&e_t, &h, &c)?;
        h = h_new;
        c = c_new;
        next_logits = model.decoder.forward(&h)?;
    }

    Ok(generated)
}

impl EvaluableModel for LSTMNetwork {
    fn generate(&self, prompt: &[u32], max_tokens: usize, device: &Device) -> Result<Vec<u32>> {
        generate_tokens_lstm(self, prompt, max_tokens, device)
    }
}

pub fn generate_tokens_gru(
    model: &GRUNetwork,
    prompt: &[u32],
    max_tokens: usize,
    device: &Device,
) -> Result<Vec<u32>> {
    let mut h = Tensor::zeros((1, model.d_hidden), DType::F32, device)?;
    let mut last_logits = Tensor::zeros((1, model.vocab_size), DType::F32, device)?;

    // Process prompt
    for &token in prompt {
        let next_tensor = Tensor::from_vec(vec![token], (1, 1), device)?;
        let e_t = model.encoder.forward(&next_tensor)?.squeeze(1)?;
        h = model.gru_cell.forward(&e_t, &h)?;
        last_logits = model.decoder.forward(&h)?;
    }

    let mut next_logits = last_logits;
    let mut generated = Vec::new();

    // Generate autoregressively
    for _ in 0..max_tokens {
        let sampled_token = sample_from_logits(&next_logits, 0.0, None)?;
        generated.push(sampled_token);

        if sampled_token == 10 {
            break;
        }

        let next_tensor = Tensor::from_vec(vec![sampled_token], (1, 1), device)?;
        let e_t = model.encoder.forward(&next_tensor)?.squeeze(1)?;
        h = model.gru_cell.forward(&e_t, &h)?;
        next_logits = model.decoder.forward(&h)?;
    }

    Ok(generated)
}

impl EvaluableModel for GRUNetwork {
    fn generate(&self, prompt: &[u32], max_tokens: usize, device: &Device) -> Result<Vec<u32>> {
        generate_tokens_gru(self, prompt, max_tokens, device)
    }
}

pub fn generate_tokens(
    model: &FreeEnergyRecurrentNetwork,
    prompt: &[u32],
    max_tokens: usize,
    inner_steps: usize,
    temperature: f64,
    top_k: Option<usize>,
    device: &Device,
) -> Result<Vec<u32>> {
    let mut state = NetworkState::init(1, &model.d_layers, model.d_mem, device)?;
    let mut generated = Vec::new();

    // Process prompt
    let prompt_tensor = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), device)?;
    let embeddings = model.encoder.forward(&prompt_tensor)?;

    let mut last_logits = None;
    for t in 0..prompt.len() {
        let e_t = embeddings.narrow(1, t, 1)?.squeeze(1)?;

        // Apply temporal prediction for t > 0
        if t > 0 {
            let mut next_mu = state.mu.clone();
            for l in 1..model.d_layers.len() {
                let layer = &model.layers[l - 1];
                next_mu[l] = layer.w_rec.forward(&state.mu[l])?;
            }
            state.mu = next_mu;
        }

        let tokens_t = prompt_tensor.narrow(1, t, 1)?;
        let _fe = crate::step::run_inner_inference(model, &e_t, Some(&tokens_t), &mut state, inner_steps)?;

        let beliefs: Vec<&Tensor> = (1..model.d_layers.len())
            .map(|l| &state.mu[l])
            .collect();
        let concat = Tensor::cat(&beliefs, 1)?;
        
        state.memory = model.memory_cell.forward(&concat, &state.memory)?;

        // Concatenate beliefs + memory for decoder
        let decode_in = Tensor::cat(&[&concat, &state.memory], 1)?;
        last_logits = Some(model.decoder.forward(&decode_in)?);
    }

    let mut next_logits = last_logits
        .ok_or_else(|| candle_core::Error::Msg("Empty prompt".to_string()))?;

    for _ in 0..max_tokens {
        let sampled_token = sample_from_logits(&next_logits, temperature, top_k)?;
        generated.push(sampled_token);

        if sampled_token == 10 { // Stop on EOS token (ID 10)
            break;
        }

        let next_tensor = Tensor::from_vec(vec![sampled_token], (1, 1), device)?;
        let e_t = model.encoder.forward(&next_tensor)?.squeeze(1)?;

        // Apply temporal recurrent transition (Option A)
        let mut next_mu = state.mu.clone();
        for l in 1..model.d_layers.len() {
            let layer = &model.layers[l - 1];
            next_mu[l] = layer.w_rec.forward(&state.mu[l])?;
        }
        state.mu = next_mu;

        let _fe = crate::step::run_inner_inference(model, &e_t, Some(&next_tensor), &mut state, inner_steps)?;

        let beliefs: Vec<&Tensor> = (1..model.d_layers.len())
            .map(|l| &state.mu[l])
            .collect();
        let concat = Tensor::cat(&beliefs, 1)?;
        
        state.memory = model.memory_cell.forward(&concat, &state.memory)?;

        // Concatenate beliefs + memory for decoder
        let decode_in = Tensor::cat(&[&concat, &state.memory], 1)?;
        next_logits = model.decoder.forward(&decode_in)?;
    }

    Ok(generated)
}

// Future work ( hardcoded :( )
fn run_repl(
    model: &FreeEnergyRecurrentNetwork,
    inner_steps: usize,
    temperature: f64,
    top_k: Option<usize>,
    device: &Device,
) -> Result<()> {
    println!("Type a prompt (e.g. 'the cat') and press Enter.");
    println!("Press Ctrl+C or type 'exit' or 'quit' to exit.");
    println!("Available words: the, cat, sat, on, mat, dog, ran, in, park, a, b, c, d, copy, key_x, key_y, secret_x, secret_y\n");

    let mut line = String::new();
    loop {
        print!("> ");
        io::stdout().flush()?;
        line.clear();
        let bytes_read = io::stdin().read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed == "exit" || trimmed == "quit" {
            break;
        }
        if trimmed.is_empty() {
            continue;
        }

        let prompt_tokens = tokenize_prompt(trimmed);
        if prompt_tokens.is_empty() {
            println!("Error: none of the words match the vocabulary.");
            continue;
        }

        let output_tokens = generate_tokens(model, &prompt_tokens, 15, inner_steps, temperature, top_k, device)?;
        let words: Vec<&str> = output_tokens.iter().map(|&id| vocab_to_word(id)).collect();
        println!("{}", words.join(" "));
    }
    Ok(())
}

fn test_pattern_completion(
    model: &impl EvaluableModel,
    device: &Device,
) -> Result<(usize, usize, Vec<String>)> {
    let mut passed = 0;
    let mut logs = Vec::new();

    let cases = vec![
        (vec![1u32, 2], vec![3u32, 4, 1, 5, 10], "the cat"),
        (vec![1u32, 6], vec![7u32, 8, 1, 9, 10], "the dog"),
    ];

    for (prompt, expected, prompt_str) in cases {
        let mut full_prompt = vec![25];
        full_prompt.extend(&prompt);
        let output = model.generate(&full_prompt, 10, device)?;
        let is_ok = output == expected;
        if is_ok {
            passed += 1;
        }
        let out_words: Vec<&str> = output.iter().map(|&id| vocab_to_word(id)).collect();
        let exp_words: Vec<&str> = expected.iter().map(|&id| vocab_to_word(id)).collect();
        logs.push(format!(
            "| `{}` | `{}` | `{}` | **{}** |",
            prompt_str,
            exp_words.join(" "),
            out_words.join(" "),
            if is_ok { "PASS" } else { "FAIL" }
        ));
    }

    Ok((passed, 2, logs))
}

fn test_big_pattern_completion(
    model: &impl EvaluableModel,
    device: &Device,
) -> Result<(usize, usize, Vec<String>)> {
    let mut passed = 0;
    let mut logs = Vec::new();

    let cases = vec![
        (vec![1u32, 2, 3], vec![4u32, 1, 5, 6, 7, 8, 1, 9, 2, 10], "the cat sat"),
        (vec![11u32, 6, 7], vec![8u32, 1, 9, 2, 3, 4, 1, 5, 6, 10], "a dog ran"),
    ];

    for (prompt, expected, prompt_str) in cases {
        let mut full_prompt = vec![25];
        full_prompt.extend(&prompt);
        let output = model.generate(&full_prompt, 10, device)?;
        let is_ok = output == expected;
        if is_ok {
            passed += 1;
        }
        let out_words: Vec<&str> = output.iter().map(|&id| vocab_to_word(id)).collect();
        let exp_words: Vec<&str> = expected.iter().map(|&id| vocab_to_word(id)).collect();
        logs.push(format!(
            "| `{}` | `{}` | `{}` | **{}** |",
            prompt_str,
            exp_words.join(" "),
            out_words.join(" "),
            if is_ok { "PASS" } else { "FAIL" }
        ));
    }

    Ok((passed, 2, logs))
}

fn test_copy_with_delay(
    model: &impl EvaluableModel,
    delay_len: usize,
    device: &Device,
) -> Result<(usize, usize, Vec<String>)> {
    let mut passed = 0;
    let mut total = 0;
    let mut logs = Vec::new();

    let symbols = vec![11u32, 12, 13, 14];
    for &x1 in &symbols {
        for &x2 in &symbols {
            total += 1;
            
            let mut prompt = vec![x1, x2];
            for _ in 0..delay_len {
                prompt.push(0);
            }
            prompt.push(15);

            let expected = vec![x1, x2];
            
            let mut full_prompt = vec![24];
            full_prompt.extend(&prompt);
            let output = model.generate(&full_prompt, 2, device)?;
            let is_ok = output.len() >= 2 && output[0] == x1 && output[1] == x2;
            if is_ok {
                passed += 1;
            }

            let prompt_words: Vec<&str> = prompt.iter().map(|&id| vocab_to_word(id)).collect();
            let out_words: Vec<&str> = output.iter().map(|&id| vocab_to_word(id)).collect();
            let exp_words: Vec<&str> = expected.iter().map(|&id| vocab_to_word(id)).collect();

            logs.push(format!(
                "| `{}` | `{}` | `{}` | **{}** |",
                prompt_words.join(" "),
                exp_words.join(" "),
                out_words.join(" "),
                if is_ok { "PASS" } else { "FAIL" }
            ));
        }
    }

    Ok((passed, total, logs))
}

fn test_big_copy_with_delay(
    model: &impl EvaluableModel,
    delay_len: usize,
    device: &Device,
) -> Result<(usize, usize, Vec<String>)> {
    let mut passed = 0;
    let mut total = 0;
    let mut logs = Vec::new();

    let symbols = vec![11u32, 12, 13, 14];
    for &x1 in &symbols {
        for &x2 in &symbols {
            for &x3 in &symbols {
                for &x4 in &symbols {
                    total += 1;
                    
                    let mut prompt = vec![x1, x2, x3, x4];
                    for _ in 0..delay_len {
                        prompt.push(0);
                    }
                    prompt.push(15);

                    let expected = vec![x1, x2, x3, x4];
                    
                    let mut full_prompt = vec![24];
                    full_prompt.extend(&prompt);
                    let output = model.generate(&full_prompt, 4, device)?;
                    let is_ok = output.len() >= 4 && output[0] == x1 && output[1] == x2 && output[2] == x3 && output[3] == x4;
                    if is_ok {
                        passed += 1;
                    }

                    if total <= 16 {
                        let prompt_words: Vec<&str> = prompt.iter().map(|&id| vocab_to_word(id)).collect();
                        let out_words: Vec<&str> = output.iter().map(|&id| vocab_to_word(id)).collect();
                        let exp_words: Vec<&str> = expected.iter().map(|&id| vocab_to_word(id)).collect();

                        logs.push(format!(
                            "| `{}` | `{}` | `{}` | **{}** |",
                            prompt_words.join(" "),
                            exp_words.join(" "),
                            out_words.join(" "),
                            if is_ok { "PASS" } else { "FAIL" }
                        ));
                    }
                }
            }
        }
    }

    Ok((passed, total, logs))
}

fn test_secret_recall(
    model: &impl EvaluableModel,
    device: &Device,
) -> Result<(usize, usize, Vec<String>)> {
    let mut passed = 0;
    let mut total = 0;
    let mut logs = Vec::new();

    let cases = vec![
        (16u32, 18u32, 17u32, 19u32, 16u32, 18u32),
        (16u32, 18u32, 17u32, 19u32, 17u32, 19u32),
        (17u32, 19u32, 16u32, 18u32, 16u32, 18u32),
        (17u32, 19u32, 16u32, 18u32, 17u32, 19u32),
    ];

    for (k1, s1, k2, s2, qkey, qsec) in cases {
        total += 1;
        let prompt = vec![k1, s1, k2, s2, qkey];
        let expected = vec![qsec];

        let mut full_prompt = vec![26];
        full_prompt.extend(&prompt);
        let output = model.generate(&full_prompt, 1, device)?;
        let is_ok = !output.is_empty() && output[0] == qsec;
        if is_ok {
            passed += 1;
        }

        let prompt_words: Vec<&str> = prompt.iter().map(|&id| vocab_to_word(id)).collect();
        let out_words: Vec<&str> = output.iter().map(|&id| vocab_to_word(id)).collect();
        let exp_words: Vec<&str> = expected.iter().map(|&id| vocab_to_word(id)).collect();

        logs.push(format!(
            "| `{}` | `{}` | `{}` | **{}** |",
            prompt_words.join(" "),
            exp_words.join(" "),
            out_words.join(" "),
            if is_ok { "PASS" } else { "FAIL" }
        ));
    }

    Ok((passed, total, logs))
}

fn test_big_secret_recall(
    model: &impl EvaluableModel,
    device: &Device,
) -> Result<(usize, usize, Vec<String>)> {
    let mut passed = 0;
    let mut total = 0;
    let mut logs = Vec::new();

    // 3-key recall cases
    let keys_3 = vec![16u32, 17, 20];
    let secrets_3 = vec![18u32, 19, 21];
    let perms_3 = vec![
        vec![0, 1, 2],
        vec![0, 2, 1],
        vec![1, 0, 2],
        vec![1, 2, 0],
        vec![2, 0, 1],
        vec![2, 1, 0],
    ];

    for perm in &perms_3 {
        let k1 = keys_3[perm[0]];
        let s1 = secrets_3[perm[0]];
        let k2 = keys_3[perm[1]];
        let s2 = secrets_3[perm[1]];
        let k3 = keys_3[perm[2]];
        let s3 = secrets_3[perm[2]];

        for &qidx in &[0, 1, 2] {
            total += 1;
            let qkey = keys_3[qidx];
            let qsec = secrets_3[qidx];

            let prompt = vec![k1, s1, k2, s2, k3, s3, qkey];
            let expected = vec![qsec];

            let mut full_prompt = vec![26];
            full_prompt.extend(&prompt);
            let output = model.generate(&full_prompt, 1, device)?;
            let is_ok = !output.is_empty() && output[0] == qsec;
            if is_ok {
                passed += 1;
            }

            let prompt_words: Vec<&str> = prompt.iter().map(|&id| vocab_to_word(id)).collect();
            let out_words: Vec<&str> = output.iter().map(|&id| vocab_to_word(id)).collect();
            let exp_words: Vec<&str> = expected.iter().map(|&id| vocab_to_word(id)).collect();

            logs.push(format!(
                "| `3-Key: {}` | `{}` | `{}` | **{}** |",
                prompt_words.join(" "),
                exp_words.join(" "),
                out_words.join(" "),
                if is_ok { "PASS" } else { "FAIL" }
            ));
        }
    }

    // 4-key recall cases
    let keys_4 = vec![16u32, 17, 20, 22];
    let secrets_4 = vec![18u32, 19, 21, 23];
    let perms_4 = vec![
        vec![0, 1, 2, 3],
        vec![1, 3, 0, 2],
        vec![3, 2, 1, 0],
    ];

    for perm in &perms_4 {
        let k1 = keys_4[perm[0]];
        let s1 = secrets_4[perm[0]];
        let k2 = keys_4[perm[1]];
        let s2 = secrets_4[perm[1]];
        let k3 = keys_4[perm[2]];
        let s3 = secrets_4[perm[2]];
        let k4 = keys_4[perm[3]];
        let s4 = secrets_4[perm[3]];

        for &qidx in &[0, 1, 2, 3] {
            total += 1;
            let qkey = keys_4[qidx];
            let qsec = secrets_4[qidx];

            let prompt = vec![k1, s1, k2, s2, k3, s3, k4, s4, qkey];
            let expected = vec![qsec];

            let mut full_prompt = vec![26];
            full_prompt.extend(&prompt);
            let output = model.generate(&full_prompt, 1, device)?;
            let is_ok = !output.is_empty() && output[0] == qsec;
            if is_ok {
                passed += 1;
            }

            let prompt_words: Vec<&str> = prompt.iter().map(|&id| vocab_to_word(id)).collect();
            let out_words: Vec<&str> = output.iter().map(|&id| vocab_to_word(id)).collect();
            let exp_words: Vec<&str> = expected.iter().map(|&id| vocab_to_word(id)).collect();

            logs.push(format!(
                "| `4-Key: {}` | `{}` | `{}` | **{}** |",
                prompt_words.join(" "),
                exp_words.join(" "),
                out_words.join(" "),
                if is_ok { "PASS" } else { "FAIL" }
            ));
        }
    }

    Ok((passed, total, logs))
}

struct BenchmarkScores {
    pattern_acc: f64,
    copy_d1_acc: f64,
    copy_d3_acc: f64,
    copy_d5_acc: f64,
    secret_acc: f64,
    
    // Big Bench
    big_pattern_acc: f64,
    big_copy_d5_acc: f64,
    big_copy_d10_acc: f64,
    big_copy_d15_acc: f64,
    big_secret_acc: f64,
}

fn run_train_and_eval(
    trainer: &mut Trainer,
    epochs: usize,
    batch_size: usize,
    _inner_steps: usize,
    task_type_override: Option<usize>,
    device: &Device,
    verbose: bool,
) -> Result<(BenchmarkScores, Vec<(usize, f64, f64, f64, f64)>, Option<usize>)> {
    let mut history = Vec::new();
    let mut ce_under_0_5_epoch = None;

    for epoch in 1..=epochs {
        let (inputs, targets) = generate_synthetic_data(batch_size, epoch, task_type_override, device)?;
        let (total, ce, fe, fe_w) = trainer.train_step(&inputs, &targets)?;

        if ce < 0.5 && ce_under_0_5_epoch.is_none() {
            ce_under_0_5_epoch = Some(epoch);
        }

        if verbose && (epoch == 1 || epoch % 50 == 0 || epoch == epochs) {
            println!(
                "  Epoch {:>3}/{} │ Loss: {:>7.4} │ CE: {:>6.4} │ FE: {:>7.4} │ fe_w: {:.3}",
                epoch, epochs, total, ce, fe, fe_w
            );
        }
        
        if epoch == 1 || epoch % 50 == 0 || epoch == epochs {
            history.push((epoch, total, ce, fe, fe_w));
        }
    }

    let pc = test_pattern_completion(&trainer.model, device)?;
    let cp_1 = test_copy_with_delay(&trainer.model, 1, device)?;
    let cp_3 = test_copy_with_delay(&trainer.model, 3, device)?;
    let cp_5 = test_copy_with_delay(&trainer.model, 5, device)?;
    let sr = test_secret_recall(&trainer.model, device)?;

    let bpc = test_big_pattern_completion(&trainer.model, device)?;
    let bcp_5 = test_big_copy_with_delay(&trainer.model, 5, device)?;
    let bcp_10 = test_big_copy_with_delay(&trainer.model, 10, device)?;
    let bcp_15 = test_big_copy_with_delay(&trainer.model, 15, device)?;
    let bsr = test_big_secret_recall(&trainer.model, device)?;

    let scores = BenchmarkScores {
        pattern_acc: (pc.0 as f64 / pc.1 as f64) * 100.0,
        copy_d1_acc: (cp_1.0 as f64 / cp_1.1 as f64) * 100.0,
        copy_d3_acc: (cp_3.0 as f64 / cp_3.1 as f64) * 100.0,
        copy_d5_acc: (cp_5.0 as f64 / cp_5.1 as f64) * 100.0,
        secret_acc: (sr.0 as f64 / sr.1 as f64) * 100.0,
        
        big_pattern_acc: (bpc.0 as f64 / bpc.1 as f64) * 100.0,
        big_copy_d5_acc: (bcp_5.0 as f64 / bcp_5.1 as f64) * 100.0,
        big_copy_d10_acc: (bcp_10.0 as f64 / bcp_10.1 as f64) * 100.0,
        big_copy_d15_acc: (bcp_15.0 as f64 / bcp_15.1 as f64) * 100.0,
        big_secret_acc: (bsr.0 as f64 / bsr.1 as f64) * 100.0,
    };

    Ok((scores, history, ce_under_0_5_epoch))
}

// ============================================================================
// Markdown Report Writer
// ============================================================================

fn write_report(
    report_path: &str,
    model_config: &ModelConfig,
    prism_config: &PrismConfig,
    epochs: usize,
    batch_size: usize,
    history: &[(usize, f64, f64, f64, f64)], // epoch, Loss, CE, FE, fe_w
    ce_under_0_5: Option<usize>,
    pattern_results: (usize, usize, Vec<String>),
    _copy_results_d1: (usize, usize, Vec<String>),
    copy_results_d3: (usize, usize, Vec<String>),
    _copy_results_d5: (usize, usize, Vec<String>),
    secret_results: (usize, usize, Vec<String>),
    
    // Big Bench Results
    big_pattern_results: (usize, usize, Vec<String>),
    _big_copy_results_d5: (usize, usize, Vec<String>),
    big_copy_results_d10: (usize, usize, Vec<String>),
    _big_copy_results_d15: (usize, usize, Vec<String>),
    big_secret_results: (usize, usize, Vec<String>),
    
    full_scores: &BenchmarkScores,
    no_fe_scores: &BenchmarkScores,
    no_prec_scores: &BenchmarkScores,
) -> Result<()> {
    let mut content = String::new();
    content.push_str("# FERN v0.4 Report\n\n");
    content.push_str("Report automatically generated after model training and evaluation.\n\n");

    content.push_str("##Model & Training Specifications\n\n");
    content.push_str("| Parameter | Value |\n");
    content.push_str("| --- | --- |\n");
    content.push_str(&format!("| **Vocab Size** | {} |\n", model_config.vocab_size));
    content.push_str(&format!("| **d_layers** | {:?} |\n", model_config.d_layers));
    content.push_str(&format!("| **Kappa (κ)** | {} |\n", model_config.kappa));
    content.push_str(&format!("| **Alpha (α)** | {} |\n", model_config.alpha));
    content.push_str(&format!("| **Max Drive** | {} |\n", model_config.max_drive));
    content.push_str(&format!("| **Epsilon Min** | {} |\n", model_config.epsilon_min));
    content.push_str(&format!("| **PRISM LR (f_pred)** | {} |\n", prism_config.pred_lr));
    content.push_str(&format!("| **PRISM LR (w_up)** | {} |\n", prism_config.error_lr));
    content.push_str(&format!("| **PRISM LR (w_gate)** | {} |\n", prism_config.gate_lr));
    content.push_str(&format!("| **PRISM LR (I/O)** | {} |\n", prism_config.io_lr));
    content.push_str(&format!("| **Warmup Steps** | {} |\n", prism_config.warmup_steps));
    content.push_str(&format!("| **Epochs** | {} |\n", epochs));
    content.push_str(&format!("| **Batch Size** | {} |\n\n", batch_size));

    content.push_str("## Training Convergence\n\n");
    let ce_threshold_str = ce_under_0_5
        .map(|e| e.to_string())
        .unwrap_or_else(|| "Never reached".to_string());
    content.push_str(&format!("- **Epochs to reach CE < 0.5**: {}\n", ce_threshold_str));
    if let Some(&(_, total, ce, fe, _)) = history.last() {
        content.push_str(&format!("- **Final Total Loss**: {:.4}\n", total));
        content.push_str(&format!("- **Final CE Loss**: {:.4}\n", ce));
        content.push_str(&format!("- **Final FE Loss**: {:.4}\n\n", fe));
    }

    content.push_str("| Epoch | Total Loss | CE Loss | FE Loss | fe_w |\n");
    content.push_str("| --- | --- | --- | --- | --- |\n");
    for &(epoch, total, ce, fe, fe_w) in history {
        content.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.3} |\n",
            epoch, total, ce, fe, fe_w
        ));
    }
    content.push_str("\n");

    content.push_str("## Baseline Comparison (Standard)\n\n");
    content.push_str("| Task | FERN (Full) | Random | Majority |\n");
    content.push_str("| --- | --- | --- | --- |\n");
    content.push_str(&format!(
        "| **Pattern (2/2)** | {:.1}% | 50.0% | 50.0% |\n",
        full_scores.pattern_acc
    ));
    content.push_str(&format!(
        "| **Copy (16/16, D3)** | {:.1}% | 6.3% | 25.0% |\n",
        full_scores.copy_d3_acc
    ));
    content.push_str(&format!(
        "| **Secret (4/4)** | {:.1}% | 50.0% | 50.0% |\n\n",
        full_scores.secret_acc
    ));

    content.push_str("## Baseline Comparison (Big Bench)\n\n");
    content.push_str("| Task | FERN (Full) | Random | Majority |\n");
    content.push_str("| --- | --- | --- | --- |\n");
    content.push_str(&format!(
        "| **Big Pattern (2/2)** | {:.1}% | 25.0% | 25.0% |\n",
        full_scores.big_pattern_acc
    ));
    content.push_str(&format!(
        "| **Big Copy (256/256, D10)** | {:.1}% | 0.4% | 25.0% |\n",
        full_scores.big_copy_d10_acc
    ));
    content.push_str(&format!(
        "| **Big Secret (30/30)** | {:.1}% | 3.3% | 25.0% |\n\n",
        full_scores.big_secret_acc
    ));

    content.push_str("## Ablation Analysis\n\n");
    content.push_str("| Component | Pattern | Copy (D3) | Secret | Big Pattern | Big Copy (D10) | Big Secret |\n");
    content.push_str("| --- | --- | --- | --- | --- | --- | --- |\n");
    content.push_str(&format!(
        "| **Full FERN** | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% |\n",
        full_scores.pattern_acc, full_scores.copy_d3_acc, full_scores.secret_acc,
        full_scores.big_pattern_acc, full_scores.big_copy_d10_acc, full_scores.big_secret_acc
    ));
    content.push_str(&format!(
        "| **No FE** | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% |\n",
        no_fe_scores.pattern_acc, no_fe_scores.copy_d3_acc, no_fe_scores.secret_acc,
        no_fe_scores.big_pattern_acc, no_fe_scores.big_copy_d10_acc, no_fe_scores.big_secret_acc
    ));
    content.push_str(&format!(
        "| **No Precision** | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% |\n\n",
        no_prec_scores.pattern_acc, no_prec_scores.copy_d3_acc, no_prec_scores.secret_acc,
        no_prec_scores.big_pattern_acc, no_prec_scores.big_copy_d10_acc, no_prec_scores.big_secret_acc
    ));

    content.push_str("## Detail Evaluation Results (Standard)\n\n");

    // 1. Pattern Completion
    content.push_str(&format!("### 1. Pattern Completion (Accuracy: {:.1}%)\n\n", full_scores.pattern_acc));
    content.push_str("| Prompt | Expected Output | Actual Output | Result |\n");
    content.push_str("| --- | --- | --- | --- |\n");
    for log in pattern_results.2 {
        content.push_str(&format!("{}\n", log));
    }
    content.push_str("\n");

    // 2. Copy with Delay
    content.push_str("### 2. Copy with Delay\n\n");
    content.push_str(&format!("- **Delay 1 Accuracy**: {:.1}%\n", full_scores.copy_d1_acc));
    content.push_str(&format!("- **Delay 3 Accuracy**: {:.1}%\n", full_scores.copy_d3_acc));
    content.push_str(&format!("- **Delay 5 Accuracy**: {:.1}%\n\n", full_scores.copy_d5_acc));
    
    content.push_str("#### Detail Logs for Delay 3:\n");
    content.push_str("| Prompt | Expected Output | Actual Output | Result |\n");
    content.push_str("| --- | --- | --- | --- |\n");
    for log in copy_results_d3.2 {
        content.push_str(&format!("{}\n", log));
    }
    content.push_str("\n");

    // 3. Secret Recall
    content.push_str(&format!("### 3. Secret Recall (Accuracy: {:.1}%)\n\n", full_scores.secret_acc));
    content.push_str("| Query Sequence | Expected Secret | Actual Output | Result |\n");
    content.push_str("| --- | --- | --- | --- |\n");
    for log in secret_results.2 {
        content.push_str(&format!("{}\n", log));
    }
    content.push_str("\n");

    content.push_str("## Detail Evaluation Results (Big Bench)\n\n");

    // 4. Big Pattern Completion
    content.push_str(&format!("### 4. Big Pattern Completion (Accuracy: {:.1}%)\n\n", full_scores.big_pattern_acc));
    content.push_str("| Prompt | Expected Output | Actual Output | Result |\n");
    content.push_str("| --- | --- | --- | --- |\n");
    for log in big_pattern_results.2 {
        content.push_str(&format!("{}\n", log));
    }
    content.push_str("\n");

    // 5. Big Copy with Delay
    content.push_str("### 5. Big Copy with Delay\n\n");
    content.push_str(&format!("- **Delay 5 Accuracy**: {:.1}%\n", full_scores.big_copy_d5_acc));
    content.push_str(&format!("- **Delay 10 Accuracy**: {:.1}%\n", full_scores.big_copy_d10_acc));
    content.push_str(&format!("- **Delay 15 Accuracy**: {:.1}%\n\n", full_scores.big_copy_d15_acc));
    
    content.push_str("#### Detail Logs for Delay 10 (First 16 tests):\n");
    content.push_str("| Prompt | Expected Output | Actual Output | Result |\n");
    content.push_str("| --- | --- | --- | --- |\n");
    for log in big_copy_results_d10.2 {
        content.push_str(&format!("{}\n", log));
    }
    content.push_str("\n");

    // 6. Big Secret Recall
    content.push_str(&format!("### 6. Big Secret Recall (Accuracy: {:.1}%)\n\n", full_scores.big_secret_acc));
    content.push_str("| Query Sequence | Expected Secret | Actual Output | Result |\n");
    content.push_str("| --- | --- | --- | --- |\n");
    for log in big_secret_results.2 {
        content.push_str(&format!("{}\n", log));
    }
    content.push_str("\n");

    content.push_str("> [!NOTE]\n");
    content.push_str("> The Gated Euler dynamics in FERN allow the higher-level belief states (layers L1-L3) to behave as a continuous-time memory cell.\n");
    content.push_str("> When sensory input is removed (clamped to `<pad>` 0), the gate $g$ shuts down incoming drive, enabling the beliefs to persist\n");
    content.push_str("> unchanged over multiple time steps. Once the trigger token (`copy` or the recall query `key`) is presented,\n");
    content.push_str("> the gate reopens and reads out the stored values through the decoder projection.\n\n");

    std::fs::write(report_path, content)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to write report file: {}", e)))?;
    
    Ok(())
}

fn write_pertask_report(
    report_path: &str,
    prism_config: &PrismConfig,
    user_epochs: usize,
    batch_size: usize,
    tasks_history: &[Vec<(usize, f64, f64, f64, f64)>],
    tasks_scores: &[BenchmarkScores],
    mixed_scores: Option<&BenchmarkScores>,
) -> Result<()> {
    let mut content = String::new();
    content.push_str("# FERN Per-Task Model Report\n\n");
    content.push_str("Report automatically generated after per-task model training and evaluation.\n\n");

    content.push_str("## Model & Training Specifications\n\n");
    content.push_str("| Parameter | Value |\n");
    content.push_str("| --- | --- |\n");
    content.push_str("| **Vocab Size** | 27 |\n");
    content.push_str("| **d_layers (Standard tasks)** | [32, 64, 64, 64] |\n");
    content.push_str("| **d_layers (Big tasks)** | [64, 128, 128, 128] |\n");
    content.push_str(&format!("| **PRISM LR (f_pred)** | {} |\n", prism_config.pred_lr));
    content.push_str(&format!("| **PRISM LR (w_up)** | {} |\n", prism_config.error_lr));
    content.push_str(&format!("| **PRISM LR (w_gate)** | {} |\n", prism_config.gate_lr));
    content.push_str(&format!("| **PRISM LR (I/O)** | {} |\n", prism_config.io_lr));
    content.push_str(&format!("| **Base Epochs** | {} (scaled by {:.2}x) |\n", user_epochs, user_epochs as f64 / 300.0));
    content.push_str(&format!("| **Batch Size** | {} |\n\n", batch_size));

    content.push_str("## Task Training Convergence\n\n");
    content.push_str("| Task | Final Total Loss | Final CE Loss | Final FE Loss |\n");
    content.push_str("| --- | --- | --- | --- |\n");
    let task_names = vec![
        "Standard Pattern Completion",
        "Big Pattern Completion",
        "Standard Copy with Delay",
        "Big Copy with Delay",
        "Standard Secret Recall",
        "Big Secret Recall",
    ];
    for (t, history) in tasks_history.iter().enumerate() {
        if let Some(&(_, total, ce, fe, _)) = history.last() {
            content.push_str(&format!(
                "| **{}** | {:.4} | {:.4} | {:.4} |\n",
                task_names[t], total, ce, fe
            ));
        }
    }
    content.push_str("\n");

    content.push_str("## Per-Task vs Mixed-Task Comparison\n\n");
    content.push_str("| Task | Per-Task | Mixed | Upper Bound Gap |\n");
    content.push_str("| --- | --- | --- | --- |\n");

    let format_row = |name: &str, per_task_val: f64, get_mixed_val: Option<f64>| {
        match get_mixed_val {
            Some(m) => {
                let gap = per_task_val - m;
                format!("| **{}** | {:.1}% | {:.1}% | {:+.1}% |\n", name, per_task_val, m, gap)
            }
            None => {
                format!("| **{}** | {:.1}% | N/A | N/A |\n", name, per_task_val)
            }
        }
    };

    content.push_str(&format_row("Standard Pattern", tasks_scores[0].pattern_acc, mixed_scores.map(|s| s.pattern_acc)));
    content.push_str(&format_row("Standard Copy D3", tasks_scores[2].copy_d3_acc, mixed_scores.map(|s| s.copy_d3_acc)));
    content.push_str(&format_row("Standard Secret", tasks_scores[4].secret_acc, mixed_scores.map(|s| s.secret_acc)));
    content.push_str(&format_row("Big Pattern", tasks_scores[1].big_pattern_acc, mixed_scores.map(|s| s.big_pattern_acc)));
    content.push_str(&format_row("Big Copy D10", tasks_scores[3].big_copy_d10_acc, mixed_scores.map(|s| s.big_copy_d10_acc)));
    content.push_str(&format_row("Big Secret", tasks_scores[5].big_secret_acc, mixed_scores.map(|s| s.big_secret_acc)));
    content.push_str("\n");

    content.push_str("> [!NOTE]\n");
    content.push_str("> By training a separate model for each benchmark task, we eliminate task-interference (mixed learning gradient conflict).\n");
    content.push_str("> This allows the belief dynamics and temporal prediction weights ($W_{rec}$) to dedicate 100% of the network capacity to the specific\n");
    content.push_str("> task, leading to much higher convergence and final accuracy scores.\n\n");

    std::fs::write(report_path, content)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to write report file: {}", e)))?;
    
    Ok(())
}

fn count_parameters(varmap: &VarMap) -> usize {
    let data = varmap.data().lock().unwrap();
    data.values().map(|v| v.as_tensor().elem_count()).sum()
}

fn find_lstm_hidden_size(target_params: usize, vocab_size: usize, d_embed: usize, device: &Device) -> Result<usize> {
    let mut best_hidden = 64;
    let mut best_diff = usize::MAX;
    for h in (32..1024).step_by(8) {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let _model = LSTMNetwork::new(vocab_size, d_embed, h, vs)?;
        let count = count_parameters(&varmap);
        let diff = (count as isize - target_params as isize).abs() as usize;
        if diff < best_diff {
            best_diff = diff;
            best_hidden = h;
        }
    }
    Ok(best_hidden)
}

fn find_gru_hidden_size(target_params: usize, vocab_size: usize, d_embed: usize, device: &Device) -> Result<usize> {
    let mut best_hidden = 64;
    let mut best_diff = usize::MAX;
    for h in (32..1024).step_by(8) {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let _model = GRUNetwork::new(vocab_size, d_embed, h, vs)?;
        let count = count_parameters(&varmap);
        let diff = (count as isize - target_params as isize).abs() as usize;
        if diff < best_diff {
            best_diff = diff;
            best_hidden = h;
        }
    }
    Ok(best_hidden)
}

fn train_lstm(
    epochs: usize,
    batch_size: usize,
    d_embed: usize,
    d_hidden: usize,
    device: &Device,
) -> Result<(LSTMNetwork, BenchmarkScores, usize)> {
    let lstm_varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&lstm_varmap, DType::F32, device);
    let model = LSTMNetwork::new(27, d_embed, d_hidden, vs)?;
    let total_params = count_parameters(&lstm_varmap);

    println!("\nTraining Baseline: LSTM");
    println!("  Parameters:              {}", total_params);
    println!("  Hidden size:             {}", d_hidden);
    println!("  Epochs:                  {}", epochs);
    println!("  Batch size:              {}", batch_size);
    println!();

    let vars = {
        let data = lstm_varmap.data().lock().unwrap();
        data.values().cloned().collect::<Vec<_>>()
    };
    
    let mut opt = candle_nn::optim::AdamW::new(
        vars,
        candle_nn::optim::ParamsAdamW {
            lr: 1e-3,
            weight_decay: 1e-4,
            ..Default::default()
        }
    )?;

    for epoch in 1..=epochs {
        let (inputs, targets) = generate_synthetic_data(batch_size, epoch, None, device)?;
        let logits = model.forward_sequence(&inputs)?;
        let (b, s, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * s, v))?;
        let targets_flat = targets.reshape(b * s)?;
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
        
        opt.backward_step(&loss)?;
        
        if epoch == 1 || epoch % 50 == 0 || epoch == epochs {
            let loss_val = loss.to_scalar::<f32>()? as f64;
            println!("  LSTM Epoch {:>3}/{} │ Loss: {:>7.4}", epoch, epochs, loss_val);
        }
    }

    // Evaluate
    let pc = test_pattern_completion(&model, device)?;
    let cp_1 = test_copy_with_delay(&model, 1, device)?;
    let cp_3 = test_copy_with_delay(&model, 3, device)?;
    let cp_5 = test_copy_with_delay(&model, 5, device)?;
    let sr = test_secret_recall(&model, device)?;

    let bpc = test_big_pattern_completion(&model, device)?;
    let bcp_5 = test_big_copy_with_delay(&model, 5, device)?;
    let bcp_10 = test_big_copy_with_delay(&model, 10, device)?;
    let bcp_15 = test_big_copy_with_delay(&model, 15, device)?;
    let bsr = test_big_secret_recall(&model, device)?;

    let scores = BenchmarkScores {
        pattern_acc: (pc.0 as f64 / pc.1 as f64) * 100.0,
        copy_d1_acc: (cp_1.0 as f64 / cp_1.1 as f64) * 100.0,
        copy_d3_acc: (cp_3.0 as f64 / cp_3.1 as f64) * 100.0,
        copy_d5_acc: (cp_5.0 as f64 / cp_5.1 as f64) * 100.0,
        secret_acc: (sr.0 as f64 / sr.1 as f64) * 100.0,
        
        big_pattern_acc: (bpc.0 as f64 / bpc.1 as f64) * 100.0,
        big_copy_d5_acc: (bcp_5.0 as f64 / bcp_5.1 as f64) * 100.0,
        big_copy_d10_acc: (bcp_10.0 as f64 / bcp_10.1 as f64) * 100.0,
        big_copy_d15_acc: (bcp_15.0 as f64 / bcp_15.1 as f64) * 100.0,
        big_secret_acc: (bsr.0 as f64 / bsr.1 as f64) * 100.0,
    };

    Ok((model, scores, total_params))
}

fn train_gru(
    epochs: usize,
    batch_size: usize,
    d_embed: usize,
    d_hidden: usize,
    device: &Device,
) -> Result<(GRUNetwork, BenchmarkScores, usize)> {
    let gru_varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&gru_varmap, DType::F32, device);
    let model = GRUNetwork::new(27, d_embed, d_hidden, vs)?;
    let total_params = count_parameters(&gru_varmap);

    println!("\nTraining Baseline: GRU");
    println!("  Parameters:              {}", total_params);
    println!("  Hidden size:             {}", d_hidden);
    println!("  Epochs:                  {}", epochs);
    println!("  Batch size:              {}", batch_size);
    println!();

    let vars = {
        let data = gru_varmap.data().lock().unwrap();
        data.values().cloned().collect::<Vec<_>>()
    };
    
    let mut opt = candle_nn::optim::AdamW::new(
        vars,
        candle_nn::optim::ParamsAdamW {
            lr: 1e-3,
            weight_decay: 1e-4,
            ..Default::default()
        }
    )?;

    for epoch in 1..=epochs {
        let (inputs, targets) = generate_synthetic_data(batch_size, epoch, None, device)?;
        let logits = model.forward_sequence(&inputs)?;
        let (b, s, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * s, v))?;
        let targets_flat = targets.reshape(b * s)?;
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
        
        opt.backward_step(&loss)?;
        
        if epoch == 1 || epoch % 50 == 0 || epoch == epochs {
            let loss_val = loss.to_scalar::<f32>()? as f64;
            println!("  GRU Epoch {:>3}/{} │ Loss: {:>7.4}", epoch, epochs, loss_val);
        }
    }

    // Evaluate
    let pc = test_pattern_completion(&model, device)?;
    let cp_1 = test_copy_with_delay(&model, 1, device)?;
    let cp_3 = test_copy_with_delay(&model, 3, device)?;
    let cp_5 = test_copy_with_delay(&model, 5, device)?;
    let sr = test_secret_recall(&model, device)?;

    let bpc = test_big_pattern_completion(&model, device)?;
    let bcp_5 = test_big_copy_with_delay(&model, 5, device)?;
    let bcp_10 = test_big_copy_with_delay(&model, 10, device)?;
    let bcp_15 = test_big_copy_with_delay(&model, 15, device)?;
    let bsr = test_big_secret_recall(&model, device)?;

    let scores = BenchmarkScores {
        pattern_acc: (pc.0 as f64 / pc.1 as f64) * 100.0,
        copy_d1_acc: (cp_1.0 as f64 / cp_1.1 as f64) * 100.0,
        copy_d3_acc: (cp_3.0 as f64 / cp_3.1 as f64) * 100.0,
        copy_d5_acc: (cp_5.0 as f64 / cp_5.1 as f64) * 100.0,
        secret_acc: (sr.0 as f64 / sr.1 as f64) * 100.0,
        
        big_pattern_acc: (bpc.0 as f64 / bpc.1 as f64) * 100.0,
        big_copy_d5_acc: (bcp_5.0 as f64 / bcp_5.1 as f64) * 100.0,
        big_copy_d10_acc: (bcp_10.0 as f64 / bcp_10.1 as f64) * 100.0,
        big_copy_d15_acc: (bcp_15.0 as f64 / bcp_15.1 as f64) * 100.0,
        big_secret_acc: (bsr.0 as f64 / bsr.1 as f64) * 100.0,
    };

    Ok((model, scores, total_params))
}

fn write_compare_report(
    report_path: &str,
    epochs: usize,
    batch_size: usize,
    fern_params: usize,
    lstm_params: usize,
    gru_params: usize,
    fern_scores: &BenchmarkScores,
    lstm_scores: &BenchmarkScores,
    gru_scores: &BenchmarkScores,
) -> Result<()> {
    let mut content = String::new();
    content.push_str("# FERN vs RNN Baselines (LSTM / GRU) Report\n\n");
    content.push_str("Report automatically generated after model training and evaluation.\n\n");

    content.push_str("## Training Specifications\n\n");
    content.push_str("| Parameter | Value |\n");
    content.push_str("| --- | --- |\n");
    content.push_str(&format!("| **Epochs** | {} |\n", epochs));
    content.push_str(&format!("| **Batch Size** | {} |\n\n", batch_size));

    content.push_str("## Baseline Comparison Table\n\n");
    content.push_str("| Model | Parameters | Standard Pattern | Standard Copy D3 | Standard Secret | Big Pattern | Big Copy D10 | Big Secret |\n");
    content.push_str("| --- | --- | --- | --- | --- | --- | --- | --- |\n");

    let format_row = |name: &str, params: usize, s: &BenchmarkScores| {
        let param_str = if params >= 1_000_000 {
            format!("{:.2}M", params as f64 / 1_000_000.0)
        } else {
            format!("{}K", params / 1000)
        };
        format!(
            "| **{}** | {} | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% |\n",
            name, param_str, s.pattern_acc, s.copy_d3_acc, s.secret_acc,
            s.big_pattern_acc, s.big_copy_d10_acc, s.big_secret_acc
        )
    };

    content.push_str(&format_row("FERN (Active Inference)", fern_params, fern_scores));
    content.push_str(&format_row("LSTM (Standard)", lstm_params, lstm_scores));
    content.push_str(&format_row("GRU (Standard)", gru_params, gru_scores));
    content.push_str("\n");

    content.push_str("> [!NOTE]\n");
    content.push_str("> This benchmark compares the Free Energy Recurrent Network (FERN) under hierarchical active inference\n");
    content.push_str("> against standard recurrent neural architectures (LSTM and GRU) optimized with AdamW.\n");
    content.push_str("> All models are compared under matched parameter budgets to ensure scientific fairness.\n\n");

    std::fs::write(report_path, content)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to write comparison report file: {}", e)))?;
    
    Ok(())
}

fn main() -> Result<()> {
    let args = CliArgs::parse();
    let device = Device::Cpu;

    let default_config = ModelConfig {
        vocab_size: 27,
        d_layers: vec![64, 128, 128, 128],
        kappa: 0.3,
        alpha: 0.9,
        epsilon_min: 1e-4,
        max_drive: 5.0,
    };

    let prism_config = PrismConfig {
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
        fe_weight_target: 0.1,
        warmup_steps: 100,
        precision_scaling: true,
    };

    let inner_steps = 3;

    match args.command {
        Some(Commands::Train { epochs, batch_size, checkpoint }) => {
            println!("Train FERN");
            let mut trainer = Trainer::new(
                default_config.clone(),
                prism_config,
                inner_steps,
                &device,
            )?;
            
            println!("\nTraining");
            println!("  Epochs:                  {}", epochs);
            println!("  Batch size:              {}", batch_size);
            println!("  Patterns: Multi-Task Mixture (Pattern Completion / Copy / Recall)");
            println!();

            for epoch in 1..=epochs {
                let (inputs, targets) = generate_synthetic_data(batch_size, epoch, None, &device)?;
                let (total, ce, fe, fe_w) = trainer.train_step(&inputs, &targets)?;

                if epoch == 1 || epoch % 50 == 0 || epoch == epochs {
                    println!(
                        "  Epoch {:>3}/{} │ Loss: {:>7.4} │ CE: {:>6.4} │ FE: {:>7.4} │ fe_w: {:.3}",
                        epoch, epochs, total, ce, fe, fe_w
                    );
                }
            }

            println!("\nSaving Checkpoint");
            default_config.save(format!("{}.json", checkpoint))?;
            trainer.varmap.save(format!("{}.safetensors", checkpoint))?;
            println!("  Saved config to {}.json", checkpoint);
            println!("  Saved weights to {}.safetensors", checkpoint);
        }
        Some(Commands::Generate { checkpoint, prompt, max_tokens, temperature, top_k }) => {
            let config = ModelConfig::load(format!("{}.json", checkpoint))?;
            let mut varmap = VarMap::new();
            let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let model = FreeEnergyRecurrentNetwork::new(config, vs)?;
            varmap.load(format!("{}.safetensors", checkpoint))?;

            let prompt_tokens = tokenize_prompt(&prompt);
            let output_tokens = generate_tokens(&model, &prompt_tokens, max_tokens, inner_steps, temperature, top_k, &device)?;
            let words: Vec<&str> = output_tokens.iter().map(|&id| vocab_to_word(id)).collect();
            println!("\nPrompt: '{}'", prompt);
            println!("Output: {:?}", words);
        }
        Some(Commands::Repl { checkpoint, temperature, top_k }) => {
            let config = ModelConfig::load(format!("{}.json", checkpoint))?;
            let mut varmap = VarMap::new();
            let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let model = FreeEnergyRecurrentNetwork::new(config, vs)?;
            varmap.load(format!("{}.safetensors", checkpoint))?;

            run_repl(&model, inner_steps, temperature, top_k, &device)?;
        }
        Some(Commands::Bench { checkpoint }) => {
            let config = ModelConfig::load(format!("{}.json", checkpoint))?;
            let mut varmap = VarMap::new();
            let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let model = FreeEnergyRecurrentNetwork::new(config, vs)?;
            varmap.load(format!("{}.safetensors", checkpoint))?;

            println!("\n─── Individual Tasks Recalls (Standard) ───");
            let pc = test_pattern_completion(&model, &device)?;
            println!("  Pattern Completion: {}/{} passed", pc.0, pc.1);
            for log in pc.2 { println!("    {}", log); }
            let cp = test_copy_with_delay(&model, 3, &device)?;
            println!("  Copy with Delay (D3): {}/{} passed", cp.0, cp.1);
            for log in cp.2 { println!("    {}", log); }
            let sr = test_secret_recall(&model, &device)?;
            println!("  Secret Recall: {}/{} passed", sr.0, sr.1);
            for log in sr.2 { println!("    {}", log); }

            println!("\n─── Individual Tasks Recalls (Big Bench) ───");
            let bpc = test_big_pattern_completion(&model, &device)?;
            println!("  Big Pattern Completion: {}/{} passed", bpc.0, bpc.1);
            for log in bpc.2 { println!("    {}", log); }
            let bcp = test_big_copy_with_delay(&model, 10, &device)?;
            println!("  Big Copy with Delay (D10): {}/{} passed", bcp.0, bcp.1);
            for log in bcp.2 { println!("    {}", log); }
            let bsr = test_big_secret_recall(&model, &device)?;
            println!("  Big Secret Recall: {}/{} passed", bsr.0, bsr.1);
            for log in bsr.2 { println!("    {}", log); }
        }
        Some(Commands::Autobench { epochs, batch_size, report, checkpoint }) => {
            println!("FERN Autobenchmark");
            
            let mut trainer = Trainer::new(
                default_config.clone(),
                prism_config.clone(),
                inner_steps,
                &device,
            )?;

            println!("\n1. Training Multi-Task Network (Full FERN)");
            let (full_scores, history, ce_under_0_5) = run_train_and_eval(
                &mut trainer,
                epochs,
                batch_size,
                inner_steps,
                None,
                &device,
                true,
            )?;

            println!("\n2. Training Ablation Baseline: No FE");
            let mut p_no_fe = prism_config.clone();
            p_no_fe.fe_weight_target = 0.0;
            let mut trainer_no_fe = Trainer::new(
                default_config.clone(),
                p_no_fe,
                inner_steps,
                &device,
            )?;
            let (no_fe_scores, _, _) = run_train_and_eval(
                &mut trainer_no_fe,
                epochs,
                batch_size,
                inner_steps,
                None,
                &device,
                false,
            )?;
            println!("  Completed.");

            println!("\n3. Training Ablation Baseline: No Precision");
            let mut p_no_prec = prism_config.clone();
            p_no_prec.precision_scaling = false;
            let mut trainer_no_prec = Trainer::new(
                default_config.clone(),
                p_no_prec,
                inner_steps,
                &device,
            )?;
            let (no_prec_scores, _, _) = run_train_and_eval(
                &mut trainer_no_prec,
                epochs,
                batch_size,
                inner_steps,
                None,
                &device,
                false,
            )?;
            println!("  Completed.");

            // Save the checkpoints of Full FERN
            default_config.save(format!("{}.json", checkpoint))?;
            trainer.varmap.save(format!("{}.safetensors", checkpoint))?;
            println!("\n4. Saved Full FERN checkpoint to prefix: {}", checkpoint);

            println!("\n5. Collecting Full Evaluations");
            let mut loaded_varmap = VarMap::new();
            let vs = VarBuilder::from_varmap(&loaded_varmap, DType::F32, &device);
            let loaded_model = FreeEnergyRecurrentNetwork::new(default_config.clone(), vs)?;
            loaded_varmap.load(format!("{}.safetensors", checkpoint))?;

            let pc = test_pattern_completion(&loaded_model, &device)?;
            let cp_d1 = test_copy_with_delay(&loaded_model, 1, &device)?;
            let cp_d3 = test_copy_with_delay(&loaded_model, 3, &device)?;
            let cp_d5 = test_copy_with_delay(&loaded_model, 5, &device)?;
            let sr = test_secret_recall(&loaded_model, &device)?;

            let bpc = test_big_pattern_completion(&loaded_model, &device)?;
            let bcp_d5 = test_big_copy_with_delay(&loaded_model, 5, &device)?;
            let bcp_d10 = test_big_copy_with_delay(&loaded_model, 10, &device)?;
            let bcp_d15 = test_big_copy_with_delay(&loaded_model, 15, &device)?;
            let bsr = test_big_secret_recall(&loaded_model, &device)?;

            println!("\n6. Generating Report");
            write_report(
                &report,
                &default_config,
                &prism_config,
                epochs,
                batch_size,
                &history,
                ce_under_0_5,
                pc,
                cp_d1,
                cp_d3,
                cp_d5,
                sr,
                bpc,
                bcp_d5,
                bcp_d10,
                bcp_d15,
                bsr,
                &full_scores,
                &no_fe_scores,
                &no_prec_scores,
            )?;
            println!("  Report successfully written to {}", report);
        }
        Some(Commands::PertaskAutobench { epochs: user_epochs, batch_size, report }) => {
            println!("FERN Per-Task Autobenchmark");
            
            let task_names = vec![
                "Standard Pattern Completion",
                "Big Pattern Completion",
                "Standard Copy with Delay",
                "Big Copy with Delay",
                "Standard Secret Recall",
                "Big Secret Recall",
            ];
            
            let checkpoints = vec![
                "pertask_pattern_standard",
                "pertask_pattern_big",
                "pertask_copy_standard",
                "pertask_copy_big",
                "pertask_secret_standard",
                "pertask_secret_big",
            ];
            
            let base_epochs = vec![150, 600, 400, 900, 200, 300];
            
            let mut tasks_history = Vec::new();
            let mut tasks_scores = Vec::new();
            
            for t in 0..6 {
                println!("\nTraining Per-Task Model: {}", task_names[t]);
                
                // 1. Task-tailored configuration
                let d_layers = if t == 0 || t == 2 || t == 4 {
                    vec![32, 64, 64, 64]
                } else {
                    vec![64, 128, 128, 128]
                };
                
                let model_config = ModelConfig {
                    vocab_size: 27,
                    d_layers,
                    kappa: 0.3,
                    alpha: 0.9,
                    epsilon_min: 1e-4,
                    max_drive: 5.0,
                };
                
                let task_fe_weight = match t {
                    0 | 1 => 0.2,   // Pattern tasks
                    2 | 3 => 0.10,  // Copy tasks (adjusted to 0.10 for stable copy)
                    4 | 5 => 0.1,   // Secret recall tasks
                    _ => unreachable!(),
                };
                let mut task_prism_config = prism_config.clone();
                task_prism_config.fe_weight_target = task_fe_weight;

                let mut trainer = Trainer::new(
                    model_config.clone(),
                    task_prism_config,
                    inner_steps,
                    &device,
                )?;
                
                let task_epochs = ((base_epochs[t] as f64) * (user_epochs as f64 / 300.0)) as usize;
                let task_epochs = std::cmp::max(1, task_epochs);
                
                println!("  Epochs:                  {}", task_epochs);
                println!("  Batch size:              {}", batch_size);
                println!("  Checkpoint:              {}", checkpoints[t]);
                println!();
                
                let (scores, history, _) = run_train_and_eval(
                    &mut trainer,
                    task_epochs,
                    batch_size,
                    inner_steps,
                    Some(t),
                    &device,
                    true,
                )?;
                
                // Save model checkpoints
                model_config.save(format!("{}.json", checkpoints[t]))?;
                trainer.varmap.save(format!("{}.safetensors", checkpoints[t]))?;
                println!("  Saved checkpoint to prefix: {}", checkpoints[t]);
                
                tasks_history.push(history);
                tasks_scores.push(scores);
            }
            
            // Train a fresh Mixed-Task model for comparison
            println!("\nTraining Comparison Mixed-Task Model (Full FERN)");
            let mixed_config = ModelConfig {
                vocab_size: 27,
                d_layers: vec![64, 128, 128, 128], // comparison mixed uses scaled architecture
                kappa: 0.3,
                alpha: 0.9,
                epsilon_min: 1e-4,
                max_drive: 5.0,
            };
            
            let mut mixed_trainer = Trainer::new(
                mixed_config.clone(),
                prism_config.clone(),
                inner_steps,
                &device,
            )?;
            
            println!("  Epochs:                  {}", user_epochs);
            println!("  Batch size:              {}", batch_size);
            println!();
            
            let (mixed_scores, _, _) = run_train_and_eval(
                &mut mixed_trainer,
                user_epochs,
                batch_size,
                inner_steps,
                None, // mixed training uses curriculum
                &device,
                true,
            )?;
            
            // Save mixed model checkpoint
            mixed_config.save("pertask_mixed_comparison.json".to_string())?;
            mixed_trainer.varmap.save("pertask_mixed_comparison.safetensors".to_string())?;
            println!("  Saved mixed comparison checkpoint to prefix: pertask_mixed_comparison");
            let mixed_scores = Some(mixed_scores);
            
            println!("\nGenerating Per-Task Report");
            write_pertask_report(
                &report,
                &prism_config,
                user_epochs,
                batch_size,
                &tasks_history,
                &tasks_scores,
                mixed_scores.as_ref(),
            )?;
            println!("  Per-Task Report successfully written to {}", report);
        }
        Some(Commands::Compare { epochs, batch_size, report }) => {
            println!("FERN Comparison Benchmark Mode");
            
            // 1. Train FERN model on the mixture
            println!("\n1. Training FERN");
            let mut trainer = Trainer::new(
                default_config.clone(),
                prism_config.clone(),
                inner_steps,
                &device,
            )?;
            let (fern_scores, _, _) = run_train_and_eval(
                &mut trainer,
                epochs,
                batch_size,
                inner_steps,
                None,
                &device,
                true,
            )?;
            let fern_params = count_parameters(&trainer.varmap);
            println!("  FERN Training Completed. Parameters: {}", fern_params);

            // 2. Find matching LSTM hidden size and train LSTM
            println!("\n2. Finding Matching LSTM Hidden Size");
            let d_embed = default_config.d_layers[0];
            let lstm_hidden = find_lstm_hidden_size(fern_params, 27, d_embed, &device)?;
            println!("  Selected LSTM Hidden Size: {}", lstm_hidden);
            
            let (_lstm_model, lstm_scores, lstm_params) = train_lstm(
                epochs,
                batch_size,
                d_embed,
                lstm_hidden,
                &device,
            )?;

            // 3. Find matching GRU hidden size and train GRU
            println!("\n3. Finding Matching GRU Hidden Size");
            let gru_hidden = find_gru_hidden_size(fern_params, 27, d_embed, &device)?;
            println!("  Selected GRU Hidden Size: {}", gru_hidden);
            
            let (_gru_model, gru_scores, gru_params) = train_gru(
                epochs,
                batch_size,
                d_embed,
                gru_hidden,
                &device,
            )?;

            // Print the parameter count of each model to stdout as requested by the user
            println!("\nFinal Parameter Counts");
            println!("- FERN (Active Inference): {} parameters", fern_params);
            println!("- LSTM (Standard):         {} parameters", lstm_params);
            println!("- GRU (Standard):          {} parameters", gru_params);

            // 4. Write comparison report
            println!("\n4. Generating Comparison Report");
            write_compare_report(
                &report,
                epochs,
                batch_size,
                fern_params,
                lstm_params,
                gru_params,
                &fern_scores,
                &lstm_scores,
                &gru_scores,
            )?;
            println!("  Comparison report successfully written to {}", report);
        }
        None => {
            // Default: Run the full Autobench!
            let report_file = "autobench_report.md";
            let checkpoint_prefix = "autobench_checkpoint";
            let epochs = 800;
            let batch_size = 32;

            println!("FERN: Autobench (Default)");
            println!("  No command specified. Running full autobenchmark pipeline...\n");

            let mut trainer = Trainer::new(
                default_config.clone(),
                prism_config.clone(),
                inner_steps,
                &device,
            )?;

            println!("1. Training Multi-Task Network (Full FERN)");
            let (full_scores, history, ce_under_0_5) = run_train_and_eval(
                &mut trainer,
                epochs,
                batch_size,
                inner_steps,
                None,
                &device,
                true,
            )?;

            println!("\n2. Training Ablation Baseline: No FE");
            let mut p_no_fe = prism_config.clone();
            p_no_fe.fe_weight_target = 0.0;
            let mut trainer_no_fe = Trainer::new(
                default_config.clone(),
                p_no_fe,
                inner_steps,
                &device,
            )?;
            let (no_fe_scores, _, _) = run_train_and_eval(
                &mut trainer_no_fe,
                epochs,
                batch_size,
                inner_steps,
                None,
                &device,
                false,
            )?;
            println!("  Completed.");

            println!("\n3. Training Ablation Baseline: No Precision");
            let mut p_no_prec = prism_config.clone();
            p_no_prec.precision_scaling = false;
            let mut trainer_no_prec = Trainer::new(
                default_config.clone(),
                p_no_prec,
                inner_steps,
                &device,
            )?;
            let (no_prec_scores, _, _) = run_train_and_eval(
                &mut trainer_no_prec,
                epochs,
                batch_size,
                inner_steps,
                None,
                &device,
                false,
            )?;
            println!("  Completed.");

            // Save the checkpoints of Full FERN
            default_config.save(format!("{}.json", checkpoint_prefix))?;
            trainer.varmap.save(format!("{}.safetensors", checkpoint_prefix))?;
            println!("\n4. Saved Full FERN checkpoint to prefix: {}", checkpoint_prefix);

            println!("\n5. Collecting Full Evaluations");
            let mut loaded_varmap = VarMap::new();
            let vs = VarBuilder::from_varmap(&loaded_varmap, DType::F32, &device);
            let loaded_model = FreeEnergyRecurrentNetwork::new(default_config.clone(), vs)?;
            loaded_varmap.load(format!("{}.safetensors", checkpoint_prefix))?;

            let pc = test_pattern_completion(&loaded_model, &device)?;
            let cp_d1 = test_copy_with_delay(&loaded_model, 1, &device)?;
            let cp_d3 = test_copy_with_delay(&loaded_model, 3, &device)?;
            let cp_d5 = test_copy_with_delay(&loaded_model, 5, &device)?;
            let sr = test_secret_recall(&loaded_model, &device)?;

            let bpc = test_big_pattern_completion(&loaded_model, &device)?;
            let bcp_d5 = test_big_copy_with_delay(&loaded_model, 5, &device)?;
            let bcp_d10 = test_big_copy_with_delay(&loaded_model, 10, &device)?;
            let bcp_d15 = test_big_copy_with_delay(&loaded_model, 15, &device)?;
            let bsr = test_big_secret_recall(&loaded_model, &device)?;

            println!("\n6. Generating Report");
            write_report(
                report_file,
                &default_config,
                &prism_config,
                epochs,
                batch_size,
                &history,
                ce_under_0_5,
                pc,
                cp_d1,
                cp_d3,
                cp_d5,
                sr,
                bpc,
                bcp_d5,
                bcp_d10,
                bcp_d15,
                bsr,
                &full_scores,
                &no_fe_scores,
                &no_prec_scores,
            )?;
            println!("  Report successfully written to {}", report_file);
        }
    }
    
    println!("\nDone.");
    Ok(())
}
