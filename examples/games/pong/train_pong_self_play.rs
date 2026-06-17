//! Pong self-play PPO training on the Burn backend.
//!
//! Resurrected from the pre-Burn deletion (PR #98) per issue #101. The
//! left paddle is trained with PPO; the right paddle is a frozen
//! snapshot of the live policy, refreshed every `SNAPSHOT_INTERVAL`
//! PPO updates. The snapshot's observation is the mirrored Pong
//! observation, so a single policy network plays either side.
//!
//! # Differences from the tch-era trainer
//!
//! - The snapshot pool stores `MlpBurnPolicy` modules cloned directly via
//!   Burn's `Module::clone()` (cheap — params are reference- counted on the
//!   autodiff backend).
//! - One snapshot is sampled per env at the start of training and refreshed
//!   when the episode terminates. The opponent's actions are produced through
//!   `get_action_host` on the mirrored observation.
//!
//! # Usage
//!
//! ```bash
//! # CPU (default — NdArray backend):
//! cargo run --example train_pong_self_play --features training --release
//!
//! # GPU (wgpu — Vulkan/Metal/DX12/WebGPU):
//! cargo run --example train_pong_self_play --features "training,wgpu" --release
//! ```
//!
//! `TOTAL_TIMESTEPS=200000` overrides the default budget.
//!
//! # Artifacts
//!
//! Final policy is saved at the end of training to:
//! - `pong_self_play_model.json` (PrettyJSON, human-readable)
//! - `pong_self_play_model.bin` (Burn binary, smaller / for code-load)
//!
//! Intermediate checkpoints are written every `CHECKPOINT_INTERVAL_UPDATES`
//! PPO updates to `pong_self_play_checkpoint_<env_steps>steps.bin` so a
//! crashed multi-hour run is recoverable to the last cadence.

use anyhow::Result;
use burn::{
    backend::Autodiff,
    module::Module,
    optim::AdamConfig,
    record::{BinFileRecorder, FullPrecisionSettings, PrettyJsonFileRecorder},
    tensor::{Int, Tensor, TensorData},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use thrust_rl::{
    env::{
        Environment,
        pong::{Pong, mirror_observation},
    },
    policy::mlp::{BurnActivation, MlpBurnConfig, MlpBurnPolicy},
    train::{
        optimizer::BurnOptimizer,
        ppo::{PPOConfig, PPOTrainerBurn},
    },
};

// Concrete backend stack — selected at compile time via Cargo features.
// `--features "training,wgpu"` swaps the CPU NdArray default for Burn's
// cross-platform GPU backend (Vulkan / Metal / DX12 / WebGPU). See issue
// #102 for the GPU validation run.
#[cfg(not(feature = "wgpu"))]
type InnerBackend = burn::backend::NdArray<f32>;
#[cfg(feature = "wgpu")]
type InnerBackend = burn::backend::Wgpu<f32, i32>;
type B = Autodiff<InnerBackend>;

#[cfg(not(feature = "wgpu"))]
const BACKEND_LABEL: &str = "NdArray<f32> + Autodiff (CPU)";
#[cfg(feature = "wgpu")]
const BACKEND_LABEL: &str = "Wgpu<f32, i32> + Autodiff (GPU: Vulkan/Metal/DX12/WebGPU)";

const NUM_ENVS: usize = 8;
const NUM_STEPS: usize = 128;
const DEFAULT_TIMESTEPS: usize = 200_000;
const LEARNING_RATE: f64 = 3e-4;
const HIDDEN_DIM: usize = 128;
const POOL_MAX: usize = 4;
const SNAPSHOT_INTERVAL: usize = 5;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;
/// PPO updates between intermediate checkpoint writes. At
/// NUM_ENVS=8, NUM_STEPS=128 this is ~1.0M env_steps per checkpoint
/// (so a 20M-step overnight run produces ~20 checkpoints).
const CHECKPOINT_INTERVAL_UPDATES: usize = 1_000;
const MODEL_NAME: &str = "pong_self_play_model";
const CHECKPOINT_PREFIX: &str = "pong_self_play_checkpoint";

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let total_timesteps: usize = std::env::var("TOTAL_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TIMESTEPS);

    tracing::info!("Starting Pong self-play PPO training (Burn backend: {})", BACKEND_LABEL);
    let training_start = std::time::Instant::now();

    let probe = Pong::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n,
        _ => panic!("Expected discrete action space"),
    };
    drop(probe);

    tracing::info!("Environment: Pong");
    tracing::info!("  obs_dim         = {}", obs_dim);
    tracing::info!("  action_dim      = {}", action_dim);
    tracing::info!("  num_envs        = {}", NUM_ENVS);
    tracing::info!("  num_steps       = {}", NUM_STEPS);
    tracing::info!("  total_timesteps = {}", total_timesteps);

    let device: burn::tensor::Device<InnerBackend> = Default::default();

    let policy_config = MlpBurnConfig {
        num_layers: 2,
        hidden_dim: HIDDEN_DIM,
        use_orthogonal_init: true,
        activation: BurnActivation::Tanh,
        seed: None,
    };
    let policy = MlpBurnPolicy::<B>::with_config(obs_dim, action_dim, policy_config, &device);

    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> =
        BurnOptimizer::new(inner_opt, LEARNING_RATE);

    let ppo_config = PPOConfig::new()
        .learning_rate(LEARNING_RATE)
        .n_epochs(4)
        .batch_size(64)
        .gamma(GAMMA as f64)
        .gae_lambda(GAE_LAMBDA as f64)
        .clip_range(0.2)
        .clip_range_vf(0.2)
        .vf_coef(0.5)
        .ent_coef(0.01)
        .max_grad_norm(0.5)
        .target_kl(1.0);

    let mut trainer = PPOTrainerBurn::new(ppo_config, policy, burn_opt)?;

    // Snapshot pool: FIFO of frozen MlpBurnPolicy modules used as the
    // right-paddle opponent.
    let mut pool: Vec<MlpBurnPolicy<B>> = Vec::with_capacity(POOL_MAX);
    pool.push(trainer.policy().clone());

    // Per-env environments and opponent indices (refresh on episode end).
    let mut envs: Vec<Pong> = (0..NUM_ENVS).map(|_| Pong::new()).collect();
    for e in envs.iter_mut() {
        e.reset();
    }
    let mut observations: Vec<Vec<f32>> = envs.iter().map(|e| e.get_observation()).collect();
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let mut opponent_idx: Vec<usize> =
        (0..NUM_ENVS).map(|_| rng.random_range(0..pool.len())).collect();

    let num_updates = total_timesteps / (NUM_STEPS * NUM_ENVS);
    tracing::info!("Planned PPO updates: {}", num_updates);
    tracing::info!("------------------------------------------------------------");

    // Rollout buffers (host).
    let cap = NUM_STEPS * NUM_ENVS;
    let mut buf_obs: Vec<f32> = Vec::with_capacity(cap * obs_dim);
    let mut buf_actions: Vec<i64> = Vec::with_capacity(cap);
    let mut buf_log_probs: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_values: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_rewards: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_dones: Vec<f32> = Vec::with_capacity(cap);

    let mut episode_returns: Vec<f32> = Vec::new();
    let mut current_returns = vec![0.0_f32; NUM_ENVS];
    let mut total_env_steps: usize = 0;

    for update in 0..num_updates {
        buf_obs.clear();
        buf_actions.clear();
        buf_log_probs.clear();
        buf_values.clear();
        buf_rewards.clear();
        buf_dones.clear();

        for _step in 0..NUM_STEPS {
            // --- Left paddle (agent under training) ----------------
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_t: Tensor<B, 2> =
                Tensor::from_data(TensorData::new(obs_flat, [NUM_ENVS, obs_dim]), &device);
            let (left_actions, log_probs, values) = trainer.policy().get_action_host(obs_t);

            // --- Right paddle (frozen snapshot, mirrored observation)
            // Sample each env's opponent action via its assigned pool index.
            let mut right_actions: Vec<i64> = Vec::with_capacity(NUM_ENVS);
            for env_id in 0..NUM_ENVS {
                let mirrored = mirror_observation(&observations[env_id]);
                let mirrored_t: Tensor<B, 2> =
                    Tensor::from_data(TensorData::new(mirrored, [1, obs_dim]), &device);
                let opp = &pool[opponent_idx[env_id]];
                let (a, _, _) = opp.get_action_host(mirrored_t);
                right_actions.push(a[0]);
            }

            for env_id in 0..NUM_ENVS {
                let result = envs[env_id].step_two(left_actions[env_id], right_actions[env_id]);

                buf_obs.extend_from_slice(&observations[env_id]);
                buf_actions.push(left_actions[env_id]);
                buf_log_probs.push(log_probs[env_id]);
                buf_values.push(values[env_id]);
                buf_rewards.push(result.reward);

                let done = result.terminated || result.truncated;
                buf_dones.push(if done { 1.0 } else { 0.0 });

                current_returns[env_id] += result.reward;
                observations[env_id] = result.observation.clone();

                if done {
                    episode_returns.push(current_returns[env_id]);
                    trainer.increment_episodes(1);
                    current_returns[env_id] = 0.0;
                    envs[env_id].reset();
                    observations[env_id] = envs[env_id].get_observation();
                    opponent_idx[env_id] = rng.random_range(0..pool.len());
                }
            }
            total_env_steps += NUM_ENVS;
        }

        // --- GAE ---------------------------------------------------
        let last_obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let last_obs_t: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(last_obs_flat, [NUM_ENVS, obs_dim]), &device);
        let (_, _, last_values_host) = trainer.policy().get_action_host(last_obs_t);

        let (advantages_host, returns_host) = compute_gae(
            &buf_rewards,
            &buf_values,
            &buf_dones,
            &last_values_host,
            GAMMA,
            GAE_LAMBDA,
            NUM_STEPS,
            NUM_ENVS,
        );

        let batch = NUM_STEPS * NUM_ENVS;
        let obs_b: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(buf_obs.clone(), [batch, obs_dim]), &device);
        let actions_b: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(buf_actions.clone(), [batch]), &device);
        let old_log_probs_b: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(buf_log_probs.clone(), [batch]), &device);
        let old_values_b: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(buf_values.clone(), [batch]), &device);
        let advantages_b: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(advantages_host, [batch]), &device);
        let returns_b: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(returns_host, [batch]), &device);

        let stats = trainer.train_step(
            obs_b,
            actions_b,
            old_log_probs_b,
            old_values_b,
            advantages_b,
            returns_b,
            |p, o, a| p.evaluate_actions(o, a),
        )?;

        // --- Snapshot refresh --------------------------------------
        if (update + 1) % SNAPSHOT_INTERVAL == 0 {
            if pool.len() >= POOL_MAX {
                pool.remove(0);
            }
            pool.push(trainer.policy().clone());
            tracing::info!(
                "  snapshot refreshed at update {} (pool size = {})",
                update + 1,
                pool.len()
            );
        }

        // --- Intermediate checkpoint -------------------------------
        if (update + 1) % CHECKPOINT_INTERVAL_UPDATES == 0 {
            let ckpt_path = format!("{CHECKPOINT_PREFIX}_{total_env_steps}steps");
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            trainer
                .policy()
                .clone()
                .save_file(&ckpt_path, &recorder)
                .map_err(|e| anyhow::anyhow!("checkpoint write failed: {e}"))?;
            tracing::info!(
                "  checkpoint written: {}.bin (env_steps={})",
                ckpt_path,
                total_env_steps
            );
        }

        let recent_avg = if !episode_returns.is_empty() {
            let n = episode_returns.len();
            let slice = &episode_returns[n.saturating_sub(50)..];
            slice.iter().copied().sum::<f32>() / slice.len() as f32
        } else {
            0.0
        };

        if update % 2 == 0 || update == num_updates - 1 {
            tracing::info!(
                "update {:>3}/{}  env_steps={:>7}  episodes={:>4}  avg_return(last≤50)={:7.3}  entropy={:5.3}",
                update + 1,
                num_updates,
                total_env_steps,
                trainer.total_episodes(),
                recent_avg,
                stats.entropy,
            );
        }
    }

    let final_avg = if !episode_returns.is_empty() {
        let n = episode_returns.len();
        let slice = &episode_returns[n.saturating_sub(50)..];
        slice.iter().copied().sum::<f32>() / slice.len() as f32
    } else {
        0.0
    };
    tracing::info!("------------------------------------------------------------");
    tracing::info!(
        "Training complete.  episodes={}  env_steps={}  final_avg_return(last≤50)={:.3}  pool_size={}  time={:.1}s",
        trainer.total_episodes(),
        total_env_steps,
        final_avg,
        pool.len(),
        training_start.elapsed().as_secs_f64(),
    );

    // --- Final model save ------------------------------------------
    // JSON is the portable, human-readable target consumed by the
    // web demo (`web/public/pong_self_play_model.json`); the .bin is
    // the compact form for fast Rust-side reload.
    let json_recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
    trainer
        .policy()
        .clone()
        .save_file(MODEL_NAME, &json_recorder)
        .map_err(|e| anyhow::anyhow!("final JSON save failed: {e}"))?;
    let bin_recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    trainer
        .policy()
        .clone()
        .save_file(MODEL_NAME, &bin_recorder)
        .map_err(|e| anyhow::anyhow!("final BIN save failed: {e}"))?;
    tracing::info!("Saved final model: {MODEL_NAME}.json + {MODEL_NAME}.bin");

    Ok(())
}

/// Per-env GAE computation (host-side).
#[allow(clippy::too_many_arguments)]
fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    last_values: &[f32],
    gamma: f32,
    gae_lambda: f32,
    num_steps: usize,
    num_envs: usize,
) -> (Vec<f32>, Vec<f32>) {
    let cap = num_steps * num_envs;
    let mut advantages = vec![0.0_f32; cap];
    let mut returns = vec![0.0_f32; cap];
    let mut last_gae = vec![0.0_f32; num_envs];
    for t in (0..num_steps).rev() {
        for n in 0..num_envs {
            let idx = t * num_envs + n;
            let next_value = if t == num_steps - 1 {
                last_values[n]
            } else {
                values[(t + 1) * num_envs + n]
            };
            let next_nonterminal = 1.0 - dones[idx];
            let delta = rewards[idx] + gamma * next_value * next_nonterminal - values[idx];
            last_gae[n] = delta + gamma * gae_lambda * next_nonterminal * last_gae[n];
            advantages[idx] = last_gae[n];
            returns[idx] = advantages[idx] + values[idx];
        }
    }
    (advantages, returns)
}
