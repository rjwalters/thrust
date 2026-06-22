//! GridWorld DQN training on the Burn backend.
//!
//! End-to-end Double-DQN trainer for the sparse-reward
//! [`GridWorld`](thrust_rl::env::games::grid_world::GridWorld) navigation env
//! (issue #185, PR B of the more-environments epic #180), using the
//! [`QNetworkBurn`](thrust_rl::policy::q_network::QNetworkBurn) +
//! [`DQNTrainerBurn`](thrust_rl::train::dqn::DQNTrainerBurn) stack on
//! `Autodiff<NdArray<f32>>` (CPU). Modeled on `train_cartpole_dqn.rs`.
//!
//! # Environment
//!
//! `GridWorld` is a `4x4` FrozenLake-style grid (default layout, no slip):
//!
//! ```text
//! S F F F
//! F H F H
//! F F F H
//! H F F G
//! ```
//!
//! - **Observation:** 16-dim one-hot over cells (tabular-like, DQN handles this
//!   well). `obs_dim = 16`.
//! - **Actions:** 4 discrete moves (`0=Up`, `1=Right`, `2=Down`, `3=Left`).
//!   `n_actions = 4`.
//! - **Reward:** `+1.0` at the goal (terminates), `-1.0` in a hole
//!   (terminates), `-0.01` per non-terminal step, truncation at 100 steps.
//!
//! Because the goal reward dominates, the highest-return policy is the
//! shortest path to the goal: a return of `+1 - 0.01 * path_len`, i.e. just
//! under `+1`. A random policy mostly falls into holes (negative return) or
//! times out (`-0.01 * 100 = -1.0`), so the learning signal is the mean
//! episode return climbing from a negative floor toward near-`+1`.
//!
//! # Configuration
//!
//! - 2-layer Tanh Q-network, 64 hidden units, orthogonal init.
//! - Replay buffer capacity 50k, min buffer 1k.
//! - ε linearly annealed `1.0 → 0.10` over 20k env steps.
//! - γ = 0.95 (short-horizon credit assignment for the shortest path).
//! - Polyak soft target updates with τ = 0.005.
//! - Default total budget: 80k env steps (matches the convergence test; the
//!   greedy policy converges to the optimal 6-step path, return ≈ +0.94).
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_dqn_grid_world --features training --release
//! ```
//!
//! Override the total step budget via the `TOTAL_TIMESTEPS` env var:
//!
//! ```bash
//! TOTAL_TIMESTEPS=20000 cargo run --example train_dqn_grid_world \
//!     --features training --release
//! ```
//!
//! Expected: the avg return over the last 100 episodes climbs from the
//! negative/timeout floor toward the near-`+1` optimal-path return within
//! the budget.
//!
//! # Learning-curve CSV (opt-in)
//!
//! Set `CURVE_CSV=<path>` to emit one `env_steps,mean_episode_reward` row
//! per logging interval (header row first), using the same `open_curve_csv()`
//! convention as the CartPole DQN/A2C examples. The Q-network weight init
//! (`QNetworkBurn::with_seed`) and the action/replay sampling (`StdRng`) are
//! both seeded, so a run is reproducible up to the `Autodiff<NdArray<f32>>`
//! backend's own run-to-run float-reduction nondeterminism. When `CURVE_CSV`
//! is unset, no file is written and behavior is unchanged.
//!
//! ```bash
//! CURVE_CSV=/tmp/dqn_grid_world.csv cargo run --example train_dqn_grid_world \
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
    env::{Environment, games::grid_world::GridWorld},
    policy::q_network::QNetworkBurn,
    train::{
        dqn::{DQNConfig, DQNTrainerBurn},
        optimizer::BurnOptimizer,
    },
};

type B = Autodiff<NdArray<f32>>;

const DEFAULT_TIMESTEPS: usize = 80_000;
const HIDDEN_DIM: usize = 64;
/// Seed for reproducible runs. Threaded through the Q-network weight init
/// (`QNetworkBurn::with_seed`) so that, together with the seeded action/replay
/// `StdRng`, the learning curve is identical run-to-run (up to backend
/// float-reduction nondeterminism).
const SEED: u64 = 0;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let total_timesteps: usize = std::env::var("TOTAL_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TIMESTEPS);

    tracing::info!("Starting GridWorld DQN Training (Burn backend)");

    let probe = GridWorld::new();
    let obs_dim = probe.observation_space().shape[0];
    let n_actions = match probe.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n as i64,
        _ => panic!("Expected discrete action space"),
    };

    tracing::info!("Environment: GridWorld 4x4 (FrozenLake-style, no slip)");
    tracing::info!("  obs_dim    = {}", obs_dim);
    tracing::info!("  n_actions  = {}", n_actions);
    tracing::info!("  total_timesteps = {}", total_timesteps);

    let device: NdArrayDevice = Default::default();

    // Online Q-network. Seeded weight init so runs are reproducible (the
    // action/replay sampling is seeded separately via `StdRng` below).
    let online =
        QNetworkBurn::<B>::with_seed(obs_dim, n_actions as usize, HIDDEN_DIM, SEED, &device);

    let config = DQNConfig::new()
        .learning_rate(5e-4)
        .batch_size(128)
        .buffer_capacity(50_000)
        .min_buffer_size(1_000)
        .target_update_interval(500)
        .gamma(0.95)
        .epsilon_start(1.0)
        .epsilon_end(0.10)
        .epsilon_decay_steps(20_000)
        .max_grad_norm(10.0)
        .soft_update_tau(0.005);

    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<B, QNetworkBurn<B>, _> =
        BurnOptimizer::new(inner_opt, config.learning_rate);

    let mut trainer = DQNTrainerBurn::new(config, online, burn_opt, obs_dim, n_actions, device)?;

    let mut env = GridWorld::new();
    env.reset();
    let mut obs = env.get_observation();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);

    // Optional learning-curve CSV (issue #160 convention). When CURVE_CSV is
    // set we write one `env_steps,mean_episode_reward` row per logging
    // interval; the mean episode return climbs from the negative/timeout floor
    // toward the near-+1 optimal-path return as the policy learns.
    let mut curve_csv = open_curve_csv()?;

    let mut episode_return: f32 = 0.0;
    let mut episode_returns: Vec<f32> = Vec::new();
    let mut last_log_step = 0_usize;
    let log_interval = 1_000_usize;

    while trainer.total_env_steps() < total_timesteps {
        // ε-greedy action selection.
        let action = {
            let device_local = device;
            trainer.select_action(&obs, &mut rng, |q: &QNetworkBurn<B>, o_host: &[f32]| {
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
            // is set. `step` is monotonic and increases by ~`log_interval`.
            if let Some(w) = curve_csv.as_mut() {
                writeln!(w, "{},{:.4}", step, recent_avg)?;
            }
            tracing::info!(
                "step={:>6}  episodes={:>4}  avg(last≤100)={:7.3}  ε={:.3}  buf={:>6}",
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
        "Training complete.  episodes={}  env_steps={}  train_steps={}  avg(last≤100)={:.3}",
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
/// written, no behavior change). Mirrors the helper in `train_cartpole_dqn.rs`
/// so the curve shares the `env_steps,mean_episode_reward` schema.
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
