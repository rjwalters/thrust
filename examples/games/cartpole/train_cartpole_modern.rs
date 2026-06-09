//! Modern CartPole PPO training on the Burn backend.
//!
//! End-to-end PPO trainer for CartPole-v1 using the
//! [`MlpBurnPolicy`](thrust_rl::policy::mlp::MlpBurnPolicy) +
//! [`PPOTrainerBurn`](thrust_rl::train::ppo::PPOTrainerBurn) stack on
//! `Autodiff<NdArray<f32>>` (CPU). This is one of the resurrected
//! trainers from issue #101 (PR #98 deleted the pre-Burn version).
//!
//! # Architecture
//!
//! - 2-layer MLP, 128 hidden units, ReLU activations, orthogonal init.
//! - `EnvPool` of 16 parallel CartPole envs.
//! - `n_steps = 256` rollout, `n_epochs = 10`, `batch_size = 128`.
//! - Total budget: 200k env steps (≈ 50 PPO updates).
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_cartpole_modern --features training --release
//! ```
//!
//! Override the total step budget via the `TOTAL_TIMESTEPS` env var:
//!
//! ```bash
//! TOTAL_TIMESTEPS=50000 cargo run --example train_cartpole_modern \
//!     --features training --release
//! ```
//!
//! Expected: average episode length climbs well above the random
//! baseline (~22 steps) within ~100k env steps.

use anyhow::Result;
use burn::{
    backend::Autodiff,
    optim::AdamConfig,
    tensor::{Int, Tensor, TensorData},
};
use thrust_rl::{
    env::{Environment, cartpole::CartPole, pool::EnvPool},
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
type Backend = Autodiff<InnerBackend>;

#[cfg(not(feature = "wgpu"))]
const BACKEND_LABEL: &str = "NdArray<f32> + Autodiff (CPU)";
#[cfg(feature = "wgpu")]
const BACKEND_LABEL: &str = "Wgpu<f32, i32> + Autodiff (GPU: Vulkan/Metal/DX12/WebGPU)";

const NUM_ENVS: usize = 16;
const NUM_STEPS: usize = 256;
const DEFAULT_TIMESTEPS: usize = 200_000;
const LEARNING_RATE: f64 = 3e-4;
const HIDDEN_DIM: usize = 128;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let total_timesteps: usize = std::env::var("TOTAL_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TIMESTEPS);

    tracing::info!("Starting Modern CartPole PPO Training (Burn backend: {})", BACKEND_LABEL);

    let training_start = std::time::Instant::now();

    // Probe environment dimensions.
    let probe = CartPole::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n,
        _ => panic!("Expected discrete action space"),
    };
    drop(probe);

    tracing::info!("Environment: CartPole-v1");
    tracing::info!("  obs_dim    = {}", obs_dim);
    tracing::info!("  action_dim = {}", action_dim);
    tracing::info!("  num_envs   = {}", NUM_ENVS);
    tracing::info!("  num_steps  = {}", NUM_STEPS);
    tracing::info!("  total_timesteps = {}", total_timesteps);

    let mut env_pool = EnvPool::new(CartPole::new, NUM_ENVS);
    let device = Default::default();

    // Policy: 2-layer ReLU MLP, orthogonal init.
    let policy_config = MlpBurnConfig {
        num_layers: 2,
        hidden_dim: HIDDEN_DIM,
        use_orthogonal_init: true,
        activation: BurnActivation::ReLU,
    };
    let policy = MlpBurnPolicy::<Backend>::with_config(obs_dim, action_dim, policy_config, &device);

    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<Backend, MlpBurnPolicy<Backend>, _> =
        BurnOptimizer::new(inner_opt, LEARNING_RATE);

    let ppo_config = PPOConfig::new()
        .learning_rate(LEARNING_RATE)
        .n_epochs(10)
        .batch_size(128)
        .gamma(GAMMA as f64)
        .gae_lambda(GAE_LAMBDA as f64)
        .clip_range(0.2)
        .clip_range_vf(0.2)
        .vf_coef(0.5)
        .ent_coef(0.01)
        .max_grad_norm(0.5)
        // Disable KL early-stop (the trainer's collapse guard handles
        // pathology; KL early-stop interferes with small-budget runs).
        .target_kl(1.0);

    let mut trainer = PPOTrainerBurn::new(ppo_config, policy, burn_opt)?;

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

    let mut observations = env_pool.reset();

    // Per-env running episode-length tracker. CartPole reward = +1/step.
    let mut episode_lengths = vec![0u32; NUM_ENVS];
    let mut completed_episode_lengths: Vec<u32> = Vec::new();
    let mut total_env_steps: usize = 0;

    let mut last_avg_len: f32 = 0.0;

    for update in 0..num_updates {
        buf_obs.clear();
        buf_actions.clear();
        buf_log_probs.clear();
        buf_values.clear();
        buf_rewards.clear();
        buf_dones.clear();

        // --- Collect rollout ---------------------------------------
        for _step in 0..NUM_STEPS {
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_t: Tensor<Backend, 2> =
                Tensor::from_data(TensorData::new(obs_flat, [NUM_ENVS, obs_dim]), &device);

            let (actions, log_probs, values) = trainer.policy().get_action_host(obs_t);

            let results = env_pool.step(&actions);

            for env_id in 0..NUM_ENVS {
                buf_obs.extend_from_slice(&observations[env_id]);
                buf_actions.push(actions[env_id]);
                buf_log_probs.push(log_probs[env_id]);
                buf_values.push(values[env_id]);
                buf_rewards.push(results[env_id].reward);

                let done = results[env_id].terminated || results[env_id].truncated;
                buf_dones.push(if done { 1.0 } else { 0.0 });

                episode_lengths[env_id] += 1;
                observations[env_id] = results[env_id].observation.clone();

                if done {
                    completed_episode_lengths.push(episode_lengths[env_id]);
                    trainer.increment_episodes(1);
                    episode_lengths[env_id] = 0;
                    observations[env_id] = env_pool.reset_env(env_id)?;
                }
            }
            total_env_steps += NUM_ENVS;
        }

        // --- Compute GAE ------------------------------------------
        // Bootstrap value for the last observation.
        let last_obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let last_obs_t: Tensor<Backend, 2> =
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

        // --- Build training tensors -------------------------------
        let batch = NUM_STEPS * NUM_ENVS;
        let obs_b: Tensor<Backend, 2> =
            Tensor::from_data(TensorData::new(buf_obs.clone(), [batch, obs_dim]), &device);
        let actions_b: Tensor<Backend, 1, Int> =
            Tensor::from_data(TensorData::new(buf_actions.clone(), [batch]), &device);
        let old_log_probs_b: Tensor<Backend, 1> =
            Tensor::from_data(TensorData::new(buf_log_probs.clone(), [batch]), &device);
        let old_values_b: Tensor<Backend, 1> =
            Tensor::from_data(TensorData::new(buf_values.clone(), [batch]), &device);
        let advantages_b: Tensor<Backend, 1> =
            Tensor::from_data(TensorData::new(advantages_host, [batch]), &device);
        let returns_b: Tensor<Backend, 1> =
            Tensor::from_data(TensorData::new(returns_host, [batch]), &device);

        // --- Train step --------------------------------------------
        let stats = trainer.train_step(
            obs_b,
            actions_b,
            old_log_probs_b,
            old_values_b,
            advantages_b,
            returns_b,
            |p, o, a| p.evaluate_actions(o, a),
        )?;

        // --- Log progress ------------------------------------------
        if !completed_episode_lengths.is_empty() {
            let n = completed_episode_lengths.len();
            let recent = &completed_episode_lengths[n.saturating_sub(100)..];
            let sum: u64 = recent.iter().map(|&x| x as u64).sum();
            last_avg_len = sum as f32 / recent.len() as f32;
        }

        if update % 5 == 0 || update == num_updates - 1 {
            tracing::info!(
                "update {:>3}/{}  env_steps={:>7}  episodes={:>4}  avg_len(last≤100)={:6.1}  policy_loss={:7.4}  entropy={:5.3}",
                update + 1,
                num_updates,
                total_env_steps,
                trainer.total_episodes(),
                last_avg_len,
                stats.policy_loss,
                stats.entropy,
            );
        }
    }

    let training_duration = training_start.elapsed();
    tracing::info!("------------------------------------------------------------");
    tracing::info!("Training complete.");
    tracing::info!("  total env steps : {}", total_env_steps);
    tracing::info!("  total episodes  : {}", trainer.total_episodes());
    tracing::info!("  final avg length(last≤100): {:.2}", last_avg_len);
    tracing::info!("  training time   : {:.1}s", training_duration.as_secs_f64());
    tracing::info!(
        "  steps/sec       : {:.0}",
        total_env_steps as f64 / training_duration.as_secs_f64()
    );

    Ok(())
}

/// Per-env GAE computation (host-side).
///
/// `rewards`, `values`, `dones` are flat `[T * N]` row-major (step-major).
/// `last_values[n]` is the value bootstrap for env `n` at step `T`.
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

    // Walk the rollout in reverse per env. Layout: index = step * num_envs +
    // env_id.
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
