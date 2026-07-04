//! CartPole A2C training on the Burn backend.
//!
//! End-to-end synchronous Advantage Actor-Critic (A2C) trainer for
//! CartPole-v1 using the
//! [`MlpBurnPolicy`](thrust_rl::policy::mlp::MlpBurnPolicy) +
//! [`A2cTrainer`](thrust_rl::train::a2c::A2cTrainer) stack on
//! `Autodiff<NdArray<f32>>` (CPU). This is PR C of the A2C epic (#150,
//! issue #153) and mirrors the PPO example
//! (`train_cartpole_modern.rs`): same `EnvPool` rollout-collect + GAE
//! pattern, swapping `PPOTrainerBurn`/`PPOConfig` for
//! `A2cTrainer`/`A2cConfig`.
//!
//! A2C differs from PPO in two places (see
//! [`thrust_rl::train::a2c::trainer`]): an un-clipped policy-gradient +
//! plain-MSE value loss, and exactly **one** gradient step per rollout (no
//! epoch loop, no minibatch shuffle, no importance ratio — hence
//! `train_step` takes no `old_log_probs` / `old_values`).
//!
//! # Architecture
//!
//! - 2-layer MLP, 128 hidden units, ReLU activations, orthogonal init.
//! - `EnvPool` of 16 parallel CartPole envs.
//! - `n_steps = 5` rollout, single update per rollout.
//! - `learning_rate = 7e-4`, seeded via `A2cConfig::seed`.
//! - Total budget: 500k env steps by default.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_cartpole_a2c --features training --release
//! ```
//!
//! Override the total step budget via the `TOTAL_TIMESTEPS` env var:
//!
//! ```bash
//! TOTAL_TIMESTEPS=200000 cargo run --example train_cartpole_a2c \
//!     --features training --release
//! ```
//!
//! Expected: average episode length climbs well above the random
//! baseline (~22 steps).
//!
//! # Learning-curve CSV (opt-in)
//!
//! Set `CURVE_CSV=<path>` to emit one `env_steps,mean_episode_reward` row
//! per logging interval (header row first). The policy init is seeded, so
//! re-runs reproduce. The same opt-in writer lives on the PPO example, so
//! A2C and PPO curves can be overlaid on the same env/seed/budget for the
//! benchmark comparison. When `CURVE_CSV` is unset, no file is written and
//! behavior is unchanged.
//!
//! ```bash
//! CURVE_CSV=/tmp/a2c.csv cargo run --example train_cartpole_a2c \
//!     --features training --release
//! ```
//!
//! # V-trace off-policy correction (opt-in)
//!
//! Set `USE_VTRACE=1` to replace GAE with V-trace targets (Espeholt et al.
//! 2018, issue #263) for advantage/return computation. Clipping thresholds
//! default to `1.0` and are overridable via `VTRACE_RHO_BAR` /
//! `VTRACE_C_BAR`. On this on-policy CartPole rollout the behavior policy
//! equals the target policy, so with `rho_bar = c_bar = 1.0` V-trace
//! reduces to n-step returns and tracks the GAE path — a regression check
//! for the V-trace wiring.
//!
//! ```bash
//! USE_VTRACE=1 cargo run --example train_cartpole_a2c \
//!     --features training --release
//! ```

use std::io::Write;

use anyhow::Result;
use burn::{
    backend::Autodiff,
    optim::AdamConfig,
    tensor::{Int, Tensor, TensorData},
};
use thrust_rl::{
    buffer::rollout::RolloutBuffer,
    env::{Environment, cartpole::CartPole, pool::EnvPool},
    policy::mlp::{BurnActivation, MlpBurnConfig, MlpBurnPolicy},
    train::{
        a2c::{A2cConfig, A2cTrainer},
        optimizer::BurnOptimizer,
    },
};

// Concrete backend stack — selected at compile time via Cargo features.
// `--features "training,wgpu"` swaps the CPU NdArray default for Burn's
// cross-platform GPU backend (Vulkan / Metal / DX12 / WebGPU).
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
const NUM_STEPS: usize = 5;
const DEFAULT_TIMESTEPS: usize = 500_000;
const LEARNING_RATE: f64 = 7e-4;
const HIDDEN_DIM: usize = 128;
const GAMMA: f32 = 0.99;
// Classic A2C uses full n-step returns (gae_lambda = 1.0).
const GAE_LAMBDA: f32 = 1.0;
/// Seed for reproducible policy init, threaded through both the policy
/// network init and `A2cConfig::seed`.
const SEED: u64 = 0;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let total_timesteps: usize = std::env::var("TOTAL_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TIMESTEPS);

    // Opt-in V-trace off-policy correction (issue #263). On-policy CartPole
    // data with rho_bar = c_bar = 1.0 reduces V-trace to n-step returns, so
    // `USE_VTRACE=1` should converge like the default GAE path — a
    // regression check for the V-trace wiring.
    let use_vtrace = std::env::var("USE_VTRACE").map(|v| v == "1" || v == "true").unwrap_or(false);
    let vtrace_rho_bar: f32 =
        std::env::var("VTRACE_RHO_BAR").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let vtrace_c_bar: f32 =
        std::env::var("VTRACE_C_BAR").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);

    tracing::info!("Starting CartPole A2C Training (Burn backend: {})", BACKEND_LABEL);
    tracing::info!(
        "  advantage estimator: {}",
        if use_vtrace {
            format!("V-trace (rho_bar={vtrace_rho_bar}, c_bar={vtrace_c_bar})")
        } else {
            "GAE".to_string()
        }
    );

    let training_start = std::time::Instant::now();

    // Probe environment dimensions.
    let probe = CartPole::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n,
        _ => panic!("Expected discrete action space"),
    };

    tracing::info!("Environment: CartPole-v1");
    tracing::info!("  obs_dim    = {}", obs_dim);
    tracing::info!("  action_dim = {}", action_dim);
    tracing::info!("  num_envs   = {}", NUM_ENVS);
    tracing::info!("  num_steps  = {}", NUM_STEPS);
    tracing::info!("  total_timesteps = {}", total_timesteps);

    let mut env_pool = EnvPool::new(CartPole::new, NUM_ENVS);
    let device = Default::default();

    // Optional learning-curve CSV (issue #153). When CURVE_CSV is set we
    // write one `env_steps,mean_episode_reward` row per logging interval.
    // CartPole reward is +1/step, so mean episode reward == mean episode
    // length.
    let mut curve_csv = open_curve_csv()?;

    // Policy: 2-layer ReLU MLP, orthogonal init, seeded for reproducible
    // initialization.
    let policy_config = MlpBurnConfig {
        num_layers: 2,
        hidden_dim: HIDDEN_DIM,
        use_orthogonal_init: true,
        activation: BurnActivation::ReLU,
        seed: Some(SEED),
    };
    let policy = MlpBurnPolicy::<Backend>::with_config(obs_dim, action_dim, policy_config, &device);

    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<Backend, MlpBurnPolicy<Backend>, _> =
        BurnOptimizer::new(inner_opt, LEARNING_RATE);

    let a2c_config = A2cConfig::new()
        .learning_rate(LEARNING_RATE)
        .gamma(GAMMA as f64)
        .gae_lambda(GAE_LAMBDA as f64)
        .value_coef(0.5)
        .entropy_coef(0.01)
        .n_steps(NUM_STEPS)
        .num_envs(NUM_ENVS)
        .max_grad_norm(0.5)
        .normalize_advantages(true)
        .use_vtrace(use_vtrace)
        .vtrace_rho_bar(vtrace_rho_bar)
        .vtrace_c_bar(vtrace_c_bar)
        .seed(SEED);

    let mut trainer = A2cTrainer::new(a2c_config, policy, burn_opt)?;

    let num_updates = total_timesteps / (NUM_STEPS * NUM_ENVS);
    tracing::info!("Planned A2C updates: {}", num_updates);
    tracing::info!("------------------------------------------------------------");

    // Rollout buffers (host).
    let cap = NUM_STEPS * NUM_ENVS;
    let mut buf_obs: Vec<f32> = Vec::with_capacity(cap * obs_dim);
    let mut buf_actions: Vec<i64> = Vec::with_capacity(cap);
    let mut buf_values: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_rewards: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_dones: Vec<f32> = Vec::with_capacity(cap);
    // Behavior-policy log-probs at collection time. Needed as the V-trace
    // importance-sampling denominator (unused by the GAE path).
    let mut buf_log_probs: Vec<f32> = Vec::with_capacity(cap);

    // Reusable rollout buffer for advantage/return computation. Populated
    // fresh each update from the flat rollout vecs, then handed to GAE or
    // V-trace depending on `A2cConfig::use_vtrace`.
    let mut rollout = RolloutBuffer::new(NUM_STEPS, NUM_ENVS, obs_dim);

    let mut observations = env_pool.reset();

    // Per-env running episode-length tracker. CartPole reward = +1/step.
    let mut episode_lengths = [0u32; NUM_ENVS];
    let mut completed_episode_lengths: Vec<u32> = Vec::new();
    let mut total_env_steps: usize = 0;

    let mut last_avg_len: f32 = 0.0;

    for update in 0..num_updates {
        buf_obs.clear();
        buf_actions.clear();
        buf_values.clear();
        buf_rewards.clear();
        buf_dones.clear();
        buf_log_probs.clear();

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
                buf_values.push(values[env_id]);
                buf_rewards.push(results[env_id].reward);
                buf_log_probs.push(log_probs[env_id]);

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

        // --- Compute advantages (GAE or V-trace) -------------------
        // Bootstrap value for the last observation.
        let last_obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let last_obs_t: Tensor<Backend, 2> =
            Tensor::from_data(TensorData::new(last_obs_flat, [NUM_ENVS, obs_dim]), &device);
        let (_, _, last_values_host) = trainer.policy().get_action_host(last_obs_t);

        // Repopulate the rollout buffer from the flat rollout vecs. The
        // flat layout is step-major (`idx = step * NUM_ENVS + env`), which
        // matches the buffer's `[step][env]` indexing and its `get_batch`
        // flatten order. `done = terminated || truncated` is mapped onto
        // the buffer's `terminated` flag so GAE and V-trace both treat an
        // env reset as an episode boundary.
        for step in 0..NUM_STEPS {
            for env in 0..NUM_ENVS {
                let idx = step * NUM_ENVS + env;
                let obs = &buf_obs[idx * obs_dim..(idx + 1) * obs_dim];
                let done = buf_dones[idx] != 0.0;
                rollout.add(
                    step,
                    env,
                    obs,
                    buf_actions[idx],
                    buf_rewards[idx],
                    buf_values[idx],
                    buf_log_probs[idx],
                    done,
                    false,
                );
            }
        }

        if trainer.config().use_vtrace {
            // On-policy rollout: the collection policy *is* the current
            // target policy (params are only updated after this step), so
            // the target log-probs equal the stored behavior log-probs.
            // With rho_bar = c_bar = 1.0 this recovers n-step returns.
            let target_log_probs: Vec<Vec<f32>> = (0..NUM_STEPS)
                .map(|step| (0..NUM_ENVS).map(|env| buf_log_probs[step * NUM_ENVS + env]).collect())
                .collect();
            let rho_bar = trainer.config().vtrace_rho_bar;
            let c_bar = trainer.config().vtrace_c_bar;
            rollout.compute_vtrace_advantages(
                &target_log_probs,
                &last_values_host,
                GAMMA,
                rho_bar,
                c_bar,
            );
        } else {
            rollout.compute_advantages(&last_values_host, GAMMA, GAE_LAMBDA);
        }

        // --- Build training tensors -------------------------------
        let training_batch = rollout.get_batch();
        let batch = NUM_STEPS * NUM_ENVS;
        let obs_b: Tensor<Backend, 2> = Tensor::from_data(
            TensorData::new(training_batch.observations.clone(), [batch, obs_dim]),
            &device,
        );
        let actions_b: Tensor<Backend, 1, Int> =
            Tensor::from_data(TensorData::new(training_batch.actions.clone(), [batch]), &device);
        let advantages_b: Tensor<Backend, 1> =
            Tensor::from_data(TensorData::new(training_batch.advantages.clone(), [batch]), &device);
        let returns_b: Tensor<Backend, 1> =
            Tensor::from_data(TensorData::new(training_batch.returns.clone(), [batch]), &device);

        // --- Train step (single A2C update) ------------------------
        // A2C drops old_log_probs / old_values: on-policy, one update per
        // rollout, no importance ratio, no value clipping.
        let stats = trainer.train_step(obs_b, actions_b, advantages_b, returns_b, |p, o, a| {
            p.evaluate_actions(o, a)
        })?;

        // --- Log progress ------------------------------------------
        if !completed_episode_lengths.is_empty() {
            let n = completed_episode_lengths.len();
            let recent = &completed_episode_lengths[n.saturating_sub(100)..];
            let sum: u64 = recent.iter().map(|&x| x as u64).sum();
            last_avg_len = sum as f32 / recent.len() as f32;
        }

        // Emit one learning-curve row per logging interval when CURVE_CSV
        // is set. A2C runs many short updates, so we throttle to keep the
        // CSV manageable while still dense.
        if let Some(w) = curve_csv.as_mut()
            && (update % 20 == 0 || update == num_updates - 1)
        {
            writeln!(w, "{},{:.4}", total_env_steps, last_avg_len)?;
        }

        if update % 200 == 0 || update == num_updates - 1 {
            tracing::info!(
                "update {:>5}/{}  env_steps={:>7}  episodes={:>5}  avg_len(last≤100)={:6.1}  policy_loss={:8.4}  value_loss={:8.4}  entropy={:5.3}",
                update + 1,
                num_updates,
                total_env_steps,
                trainer.total_episodes(),
                last_avg_len,
                stats.policy_loss,
                stats.value_loss,
                stats.entropy,
            );
        }
    }

    if let Some(mut w) = curve_csv.take() {
        w.flush()?;
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

/// Open the opt-in learning-curve CSV writer.
///
/// Returns `Ok(Some(writer))` with the header row already written when the
/// `CURVE_CSV` env var names a path, or `Ok(None)` when it is unset (no
/// file written, no behavior change).
fn open_curve_csv() -> Result<Option<std::io::BufWriter<std::fs::File>>> {
    match std::env::var("CURVE_CSV") {
        Ok(path) if !path.is_empty() => {
            let file = std::fs::File::create(&path)?;
            let mut w = std::io::BufWriter::new(file);
            writeln!(w, "env_steps,mean_episode_reward")?;
            tracing::info!("Writing learning-curve CSV to {}", path);
            Ok(Some(w))
        }
        _ => Ok(None),
    }
}
