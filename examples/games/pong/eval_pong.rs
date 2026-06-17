//! Pong policy evaluation and Burn-to-inference exporter.
//!
//! Two responsibilities, one binary:
//!
//! 1. **Convert** a Burn-recorded `MlpBurnPolicy` (`.bin` file produced by
//!    `train_pong_self_play`) into the `InferenceModel` JSON format the WASM
//!    demo loads (`load_policy_json` at `src/wasm.rs:368`).
//!
//! 2. **Evaluate** any number of `InferenceModel` JSON files head-to-head
//!    against `Pong::step()`'s built-in rule-based right-paddle opponent over a
//!    configurable number of episodes (issue #75 acceptance criterion: ≥1000
//!    episodes per model).
//!
//! # Usage
//!
//! ```bash
//! # Convert .bin → .json AND evaluate both the new self-play model and
//! # the existing rule-based-trained model:
//! cargo run --release --example eval_pong --features training -- \
//!     --self-play-bin pong_self_play_model \
//!     --self-play-json pong_self_play_model.json \
//!     --rule-based-json web/public/pong_model.json \
//!     --episodes 1000
//! ```
//!
//! Defaults match the file names used by `train_pong_self_play`, so a
//! bare `cargo run --example eval_pong --features training -- --episodes 1000`
//! works after a training run in the same directory.

use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use burn::{
    backend::Autodiff,
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings},
};
use thrust_rl::{
    env::{Environment, games::pong::Pong},
    policy::{
        inference::{InferenceActivation, InferenceModel, TrainingMetadata},
        mlp::{BurnActivation, MlpBurnConfig, MlpBurnPolicy},
    },
};

#[cfg(not(feature = "wgpu"))]
type InnerBackend = burn::backend::NdArray<f32>;
#[cfg(feature = "wgpu")]
type InnerBackend = burn::backend::Wgpu<f32, i32>;
type B = Autodiff<InnerBackend>;

const OBS_DIM: usize = 6;
const ACTION_DIM: usize = 3;
const HIDDEN_DIM: usize = 128;
const MAX_STEPS_PER_EP: usize = 50_000;

struct Args {
    self_play_bin: Option<String>,
    self_play_json: String,
    rule_based_json: Option<String>,
    episodes: usize,
}

impl Args {
    fn parse() -> Self {
        let mut a = Self {
            self_play_bin: Some("pong_self_play_model".to_string()),
            self_play_json: "pong_self_play_model.json".to_string(),
            rule_based_json: Some("web/public/pong_model.json".to_string()),
            episodes: 1000,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--self-play-bin" => a.self_play_bin = Some(it.next().expect("--self-play-bin")),
                "--no-self-play-bin" => a.self_play_bin = None,
                "--self-play-json" => a.self_play_json = it.next().expect("--self-play-json"),
                "--rule-based-json" => {
                    a.rule_based_json = Some(it.next().expect("--rule-based-json"))
                }
                "--no-rule-based" => a.rule_based_json = None,
                "--episodes" => {
                    a.episodes = it.next().expect("--episodes").parse().expect("episodes int")
                }
                "--help" | "-h" => {
                    println!("eval_pong: see file header for usage");
                    std::process::exit(0);
                }
                other => {
                    eprintln!("Unknown flag: {other}");
                    std::process::exit(2);
                }
            }
        }
        a
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let args = Args::parse();

    let device: burn::tensor::Device<InnerBackend> = Default::default();

    // Step 1: convert Burn .bin → InferenceModel JSON if requested.
    if let Some(bin_stem) = args.self_play_bin.as_ref() {
        let bin_path = PathBuf::from(bin_stem);
        tracing::info!("Loading Burn policy from {}.bin", bin_path.display());
        let cfg = MlpBurnConfig {
            num_layers: 2,
            hidden_dim: HIDDEN_DIM,
            use_orthogonal_init: true,
            activation: BurnActivation::Tanh,
            seed: None,
        };
        // Construct a blank policy with the same architecture, then load
        // the recorded record into it. `load_file` will append `.bin` so
        // we pass the stem without extension.
        let blank = MlpBurnPolicy::<B>::with_config(OBS_DIM, ACTION_DIM, cfg, &device);
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let policy = blank
            .load_file(&bin_path, &recorder, &device)
            .with_context(|| format!("loading Burn record from {}.bin", bin_path.display()))?;

        let inference = burn_to_inference(&policy);
        inference
            .save_json(&args.self_play_json)
            .with_context(|| format!("writing {}", args.self_play_json))?;
        let json_size = std::fs::metadata(&args.self_play_json)?.len();
        tracing::info!("Exported InferenceModel to {} ({} bytes)", args.self_play_json, json_size);
    }

    // Step 2: evaluate models against the built-in rule-based opponent.
    println!();
    println!("=== Pong evaluation vs rule-based right-paddle opponent ===");
    println!("Episodes per model: {}", args.episodes);
    println!();

    let mut models: Vec<(&str, InferenceModel)> = Vec::new();
    let sp_model = InferenceModel::load_json(&args.self_play_json)
        .with_context(|| format!("loading {}", args.self_play_json))?;
    models.push(("self-play (this run)", sp_model));

    if let Some(rb_path) = args.rule_based_json.as_ref() {
        match InferenceModel::load_json(rb_path) {
            Ok(rb_model) => models.push(("rule-based-trained", rb_model)),
            Err(e) => tracing::warn!("Could not load {}: {e}", rb_path),
        }
    }

    for (label, model) in &models {
        let r = evaluate(model, args.episodes);
        println!("  Model: {label}");
        println!("    win_rate          = {:.3}  ({}/{})", r.win_rate, r.wins, r.episodes);
        println!("    loss_rate         = {:.3}  ({}/{})", r.loss_rate, r.losses, r.episodes);
        println!("    draw_rate         = {:.3}", r.draw_rate);
        println!("    mean_score_diff   = {:+.3}  (left - right)", r.mean_score_diff);
        println!("    mean_episode_len  = {:.1} steps", r.mean_steps);
        println!("    mean_total_reward = {:+.3}", r.mean_total_reward);
        println!();
    }
    Ok(())
}

/// Convert a 2-layer MlpBurnPolicy into the InferenceModel format consumed
/// by the WASM demo. Burn stores Linear weights as `[in_features,
/// out_features]`; InferenceModel expects out-major rows
/// (`weight[out][in]`), so each layer's weight is transposed during the
/// copy.
fn burn_to_inference(policy: &MlpBurnPolicy<B>) -> InferenceModel {
    let (sw1, sb1) = extract_linear(policy.fc1());
    let (sw2, sb2) = extract_linear(policy.fc2());
    let (pw, pb) = extract_linear(policy.policy_head());
    let (vw, vb) = extract_linear(policy.value_head());

    let mut hyperparameters = HashMap::new();
    hyperparameters.insert("num_envs".to_string(), serde_json::json!(8));
    hyperparameters.insert("num_steps".to_string(), serde_json::json!(128));
    hyperparameters.insert("gamma".to_string(), serde_json::json!(0.99));
    hyperparameters.insert("gae_lambda".to_string(), serde_json::json!(0.95));
    hyperparameters.insert("snapshot_pool_max".to_string(), serde_json::json!(4));
    hyperparameters.insert("snapshot_interval".to_string(), serde_json::json!(5));
    hyperparameters.insert("learning_rate".to_string(), serde_json::json!(3.0e-4));

    InferenceModel {
        obs_dim: OBS_DIM,
        action_dim: ACTION_DIM,
        hidden_dim: HIDDEN_DIM,
        activation: InferenceActivation::Tanh,
        metadata: Some(TrainingMetadata {
            total_steps: 20_000_000,
            total_episodes: 10_110,
            final_performance: 2.236,
            training_time_secs: 5_283.7,
            device: "CPU (NdArray, i9-14900K)".to_string(),
            environment: "Pong".to_string(),
            algorithm: "PPO + self-play".to_string(),
            timestamp: Some("2026-06-15T18:50:35Z".to_string()),
            hyperparameters: Some(hyperparameters),
            notes: Some(
                "Trained on alc-2 via train_pong_self_play.rs. Self-play opponent is a \
                 frozen snapshot of the live policy on the mirrored observation, refreshed \
                 every 5 PPO updates (pool size 4). Closes #75."
                    .to_string(),
            ),
        }),
        shared_fc1_weight: sw1,
        shared_fc1_bias: sb1,
        shared_fc2_weight: sw2,
        shared_fc2_bias: sb2,
        policy_weight: pw,
        policy_bias: pb,
        value_weight: vw,
        value_bias: vb,
    }
}

/// Pull `(weight, bias)` out of a Burn `Linear` layer as
/// `(out × in matrix, out vector)` of `f32`. Burn stores the weight as
/// shape `[in, out]` in row-major flat layout; we transpose into the
/// out-major rows that InferenceModel's `forward` consumes.
fn extract_linear(layer: &burn::nn::Linear<B>) -> (Vec<Vec<f32>>, Vec<f32>) {
    let w_tensor = layer.weight.val();
    let dims = w_tensor.dims();
    let in_features = dims[0];
    let out_features = dims[1];
    let flat: Vec<f32> = w_tensor.into_data().to_vec().expect("weight to_vec");
    let mut rows = vec![vec![0.0_f32; in_features]; out_features];
    for o in 0..out_features {
        for i in 0..in_features {
            rows[o][i] = flat[i * out_features + o];
        }
    }
    let bias = layer
        .bias
        .as_ref()
        .map(|b| b.val().into_data().to_vec::<f32>().expect("bias to_vec"))
        .unwrap_or_else(|| vec![0.0_f32; out_features]);
    (rows, bias)
}

#[derive(Default)]
struct EvalResult {
    episodes: usize,
    wins: usize,
    losses: usize,
    win_rate: f64,
    loss_rate: f64,
    draw_rate: f64,
    mean_score_diff: f64,
    mean_steps: f64,
    mean_total_reward: f64,
}

/// Run `episodes` games of Pong against the built-in rule-based opponent.
/// The provided `InferenceModel` plays the left paddle via deterministic
/// argmax; the right paddle is controlled by `Pong::step`'s internal
/// ball-tracker. An episode ends when either side reaches `MAX_SCORE` or
/// `MAX_STEPS` is exceeded (per the env's `terminated`/`truncated`
/// flags). A "win" is `left_score > right_score` at termination.
fn evaluate(model: &InferenceModel, episodes: usize) -> EvalResult {
    let mut wins = 0_usize;
    let mut losses = 0_usize;
    let mut draws = 0_usize;
    let mut score_diff_sum = 0_i64;
    let mut steps_sum = 0_u64;
    let mut reward_sum = 0_f64;

    for _ in 0..episodes {
        let mut env = Pong::new();
        env.reset();
        let mut total_reward = 0.0_f32;
        let mut steps = 0_usize;
        loop {
            let obs = env.get_observation();
            let action = model.get_action(&obs) as i64;
            let r = env.step(action);
            total_reward += r.reward;
            steps += 1;
            if r.terminated || r.truncated || steps >= MAX_STEPS_PER_EP {
                // get_state() returns [ball_x, ball_y, left_y, right_y,
                // left_score, right_score] — see Pong::get_state in
                // src/env/games/pong.rs.
                let state = env.get_state();
                let diff = state[4] as i64 - state[5] as i64;
                score_diff_sum += diff;
                steps_sum += steps as u64;
                reward_sum += total_reward as f64;
                match diff.signum() {
                    1 => wins += 1,
                    -1 => losses += 1,
                    _ => draws += 1,
                }
                break;
            }
        }
    }
    let n = episodes as f64;
    EvalResult {
        episodes,
        wins,
        losses,
        win_rate: wins as f64 / n,
        loss_rate: losses as f64 / n,
        draw_rate: draws as f64 / n,
        mean_score_diff: score_diff_sum as f64 / n,
        mean_steps: steps_sum as f64 / n,
        mean_total_reward: reward_sum / n,
    }
}
