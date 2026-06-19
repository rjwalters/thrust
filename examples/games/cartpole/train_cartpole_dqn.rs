//! CartPole DQN training on the Burn backend.
//!
//! End-to-end Double-DQN trainer for CartPole-v1 using the
//! [`QNetworkBurn`](thrust_rl::policy::q_network::QNetworkBurn) +
//! [`DQNTrainerBurn`](thrust_rl::train::dqn::DQNTrainerBurn) stack on
//! `Autodiff<NdArray<f32>>` (CPU). Resurrected from the pre-Burn
//! deletion (PR #98) per issue #101.
//!
//! # Configuration
//!
//! - 2-layer Tanh Q-network, 64 hidden units, orthogonal init.
//! - Replay buffer capacity 50k, min buffer 1k.
//! - ε linearly annealed `1.0 → 0.05` over 10k env steps.
//! - Polyak soft target updates with τ = 0.005.
//! - Default total budget: 60k env steps.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_cartpole_dqn --features training --release
//! ```
//!
//! Override the total step budget via the `TOTAL_TIMESTEPS` env var:
//!
//! ```bash
//! TOTAL_TIMESTEPS=20000 cargo run --example train_cartpole_dqn \
//!     --features training --release
//! ```
//!
//! Expected: avg return over the last 100 episodes climbs well above
//! the random baseline (~22) within ~30k env steps.
//!
//! # Learning-curve CSV (opt-in)
//!
//! Set `CURVE_CSV=<path>` to emit one `env_steps,mean_episode_reward` row
//! per logging interval (header row first). The Q-network weight init
//! (`QNetworkBurn::with_seed`) and the action/replay sampling (`StdRng`) are
//! both seeded, so a run is reproducible up to the `Autodiff<NdArray<f32>>`
//! backend's own run-to-run float-reduction nondeterminism (the same caveat
//! the A2C/PPO curve examples carry on this backend). DQN runs on the SAME
//! CartPole env/seed/budget as the A2C (`train_cartpole_a2c.rs`) and PPO
//! (`train_cartpole_modern.rs`) examples, so its curve overlays directly on
//! theirs for the benchmark comparison. CartPole reward is +1/step, so mean
//! episode reward == mean episode length, matching the A2C/PPO semantics.
//! When `CURVE_CSV` is unset, no file is written and behavior is unchanged.
//!
//! ```bash
//! CURVE_CSV=/tmp/dqn.csv cargo run --example train_cartpole_dqn \
//!     --features training --release
//! ```

use std::io::Write;

use anyhow::Result;
use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    optim::AdamConfig,
    tensor::{Tensor, TensorData},
};
use rand::{SeedableRng, rngs::StdRng};
use thrust_rl::{
    env::{Environment, cartpole::CartPole},
    policy::q_network::QNetworkBurn,
    train::{
        dqn::{DQNConfig, DQNTrainerBurn},
        optimizer::BurnOptimizer,
    },
};

type B = Autodiff<NdArray<f32>>;

const DEFAULT_TIMESTEPS: usize = 60_000;
const HIDDEN_DIM: usize = 64;
/// Seed for reproducible runs. Threaded through the Q-network weight init
/// (`QNetworkBurn::with_seed`) so that, together with the seeded action/replay
/// `StdRng`, the learning curve is identical run-to-run.
const SEED: u64 = 0;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let total_timesteps: usize = std::env::var("TOTAL_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TIMESTEPS);

    tracing::info!("Starting CartPole DQN Training (Burn backend)");

    let probe = CartPole::new();
    let obs_dim = probe.observation_space().shape[0];
    let n_actions = match probe.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n as i64,
        _ => panic!("Expected discrete action space"),
    };
    drop(probe);

    tracing::info!("Environment: CartPole-v1");
    tracing::info!("  obs_dim    = {}", obs_dim);
    tracing::info!("  n_actions  = {}", n_actions);
    tracing::info!("  total_timesteps = {}", total_timesteps);

    let device: NdArrayDevice = Default::default();

    // Online Q-network. Seeded weight init so runs are reproducible (the
    // action/replay sampling is seeded separately via `StdRng` below).
    let online =
        QNetworkBurn::<B>::with_seed(obs_dim, n_actions as usize, HIDDEN_DIM, SEED, &device);

    let config = DQNConfig::new()
        .learning_rate(1e-3)
        .batch_size(64)
        .buffer_capacity(50_000)
        .min_buffer_size(1_000)
        .target_update_interval(500)
        .gamma(0.99)
        .epsilon_start(1.0)
        .epsilon_end(0.05)
        .epsilon_decay_steps(10_000)
        .max_grad_norm(10.0)
        .soft_update_tau(0.005);

    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<B, QNetworkBurn<B>, _> =
        BurnOptimizer::new(inner_opt, config.learning_rate);

    let mut trainer =
        DQNTrainerBurn::new(config, online, burn_opt, obs_dim, n_actions, device.clone())?;

    let mut env = CartPole::new();
    env.reset();
    let mut obs = env.get_observation();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);

    // Optional learning-curve CSV (issue #160). When CURVE_CSV is set we write
    // one `env_steps,mean_episode_reward` row per logging interval. CartPole
    // reward is +1/step, so mean episode reward == mean episode length,
    // matching the A2C/PPO examples so curves overlay on the same env/budget.
    let mut curve_csv = open_curve_csv()?;

    let mut episode_return: f32 = 0.0;
    let mut episode_returns: Vec<f32> = Vec::new();
    let mut last_log_step = 0_usize;
    let log_interval = 2_000_usize;

    while trainer.total_env_steps() < total_timesteps {
        // ε-greedy action selection.
        let action = {
            let device_local = device.clone();
            trainer.select_action(&obs, &mut rng, |q: &QNetworkBurn<B>, o_host: &[f32]| {
                // Forward pass on the inner (non-autodiff) backend for speed
                // is not directly available — use the autodiff module's
                // forward; gradient bearing is irrelevant since we discard
                // the tensor after argmax.
                let o_t: Tensor<B, 2> = Tensor::from_data(
                    TensorData::new(o_host.to_vec(), [1, o_host.len()]),
                    &device_local,
                );
                let q_values = q.forward(o_t);
                // Argmax over actions.
                let q_host: Vec<f32> = q_values.into_data().to_vec().unwrap_or_default();
                let mut best = 0_i64;
                let mut best_v = f32::NEG_INFINITY;
                for (i, &v) in q_host.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        best = i as i64;
                    }
                }
                best
            })
        };

        let result = env.step(action);
        let next_obs = result.observation.clone();
        let done = result.terminated || result.truncated;
        trainer.buffer_mut().push(&obs, action, result.reward, &next_obs, done);

        episode_return += result.reward;
        obs = next_obs;

        trainer.increment_env_step();
        let _ = trainer.maybe_sync_target(|online, _target, _tau| online.clone());

        if done {
            episode_returns.push(episode_return);
            trainer.increment_episodes(1);
            episode_return = 0.0;
            env.reset();
            obs = env.get_observation();
        }

        // One gradient update per env step.
        let _ = trainer.train_step(
            &mut rng,
            |q: &QNetworkBurn<B>, o: Tensor<B, 2>| q.forward(o),
            |q: &QNetworkBurn<B>, o: Tensor<B, 2>| q.forward(o),
        )?;

        // Periodic logging.
        let step = trainer.total_env_steps();
        if step.saturating_sub(last_log_step) >= log_interval {
            last_log_step = step;
            let recent_avg = if !episode_returns.is_empty() {
                let n = episode_returns.len();
                let slice = &episode_returns[n.saturating_sub(100)..];
                slice.iter().copied().sum::<f32>() / slice.len() as f32
            } else {
                0.0
            };
            // Emit one learning-curve row per logging interval when CURVE_CSV
            // is set. `step` is monotonic and increases by `log_interval`.
            if let Some(w) = curve_csv.as_mut() {
                writeln!(w, "{},{:.4}", step, recent_avg)?;
            }
            tracing::info!(
                "step={:>6}  episodes={:>4}  avg(last≤100)={:7.2}  ε={:.3}  buf={:>6}",
                step,
                trainer.total_episodes(),
                recent_avg,
                trainer.last_epsilon(),
                trainer.buffer_len(),
            );
        }
    }

    if let Some(mut w) = curve_csv.take() {
        w.flush()?;
    }

    let final_avg = if !episode_returns.is_empty() {
        let n = episode_returns.len();
        let slice = &episode_returns[n.saturating_sub(100)..];
        slice.iter().copied().sum::<f32>() / slice.len() as f32
    } else {
        0.0
    };
    tracing::info!("------------------------------------------------------------");
    tracing::info!(
        "Training complete.  episodes={}  env_steps={}  train_steps={}  avg(last≤100)={:.2}",
        trainer.total_episodes(),
        trainer.total_env_steps(),
        trainer.total_train_steps(),
        final_avg,
    );

    Ok(())
}

/// Open the opt-in learning-curve CSV writer.
///
/// Returns `Ok(Some(writer))` with the header row already written when the
/// `CURVE_CSV` env var names a path, or `Ok(None)` when it is unset (no file
/// written, no behavior change). Mirrors the helper in `train_cartpole_a2c.rs`
/// so DQN/A2C/PPO curves share the `env_steps,mean_episode_reward` schema.
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
