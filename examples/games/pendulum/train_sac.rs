//! Pendulum SAC training on the Burn backend.
//!
//! End-to-end seeded Soft Actor-Critic (SAC) trainer for the continuous
//! [`PendulumSwingUp`](thrust_rl::env::games::pendulum::PendulumSwingUp)
//! env using [`SacTrainer`](thrust_rl::train::sac::SacTrainer) on
//! `Autodiff<NdArray<f32>>` (CPU). This is the runnable counterpart of the
//! convergence test (`tests/test_sac_pendulum.rs`, issue #136 item 6) and
//! the foundation the SAC benchmark groups reuse (issue #162, epic #159).
//!
//! Unlike the on-policy A2C/PPO CartPole examples, SAC is **off-policy and
//! continuous**: one env step + one replay-sampled gradient update + one
//! buffer push, every single env step. The actor emits a tanh-squashed
//! action in `(-1, 1)` which is rescaled by `MAX_TORQUE = 2.0` before the
//! env step.
//!
//! # Architecture
//!
//! - `PendulumSwingUp`: `obs_dim = 3` (cosθ, sinθ, θ̇), `action_dim = 1`.
//! - SAC actor (256-wide, 2 hidden layers) + twin critics + targets.
//! - Replay buffer warmed for `learning_starts` steps before updates fire.
//! - Auto-tuned entropy temperature `alpha`.
//! - Seeded via `SacConfig::seed` and `PendulumSwingUp::with_seed`.
//! - Total budget: 30k env steps by default (the convergence-test budget).
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_sac --features training --release
//! ```
//!
//! Override the total step budget via the `TOTAL_TIMESTEPS` env var (a
//! short 4k run is a fast smoke check; convergence needs ~30k):
//!
//! ```bash
//! TOTAL_TIMESTEPS=4000 cargo run --example train_sac \
//!     --features training --release
//! ```
//!
//! Expected: mean episode return climbs from a random baseline of roughly
//! -1200..-1600 toward near-optimal ~ -150 (the convergence bar is -200
//! over the final eval episodes after 30k steps).
//!
//! # Learning-curve CSV (opt-in)
//!
//! Set `CURVE_CSV=<path>` to emit one `env_steps,mean_episode_reward` row
//! per logging interval (header row first). Training is seeded, so re-runs
//! reproduce the same CSV byte-for-byte. **Pendulum returns are negative**
//! (mean episode return, not length), so these curves form a separate
//! negative-return group and are NOT overlaid on the CartPole A2C/PPO/DQN
//! curves. When `CURVE_CSV` is unset, no file is written and behavior is
//! unchanged.
//!
//! ```bash
//! CURVE_CSV=/tmp/sac.csv cargo run --example train_sac \
//!     --features training --release
//! ```

use std::io::Write;

use anyhow::Result;
use burn::backend::{Autodiff, NdArray};
use thrust_rl::{
    env::{Environment, games::pendulum::PendulumSwingUp},
    train::sac::{SacConfig, SacTrainer},
};

type Backend = Autodiff<NdArray<f32>>;

const BACKEND_LABEL: &str = "NdArray<f32> + Autodiff (CPU)";

/// Pendulum observation / action dimensions.
const OBS_DIM: usize = 3;
const ACTION_DIM: usize = 1;

/// Torque range the Pendulum env clamps to; the actor emits actions in
/// `(-1, 1)` so we rescale by this before stepping.
const MAX_TORQUE: f32 = 2.0;

/// Default total env-step budget (matches the convergence test).
const DEFAULT_TIMESTEPS: usize = 30_000;

/// SAC network / replay hyperparameters (mirror the convergence test).
const BUFFER_CAPACITY: usize = 50_000;
const MIN_BUFFER_SIZE: usize = 1_000;
const LEARNING_STARTS: usize = 1_000;
const BATCH_SIZE: usize = 256;
const HIDDEN_DIM: usize = 256;
const NUM_HIDDEN_LAYERS: usize = 2;

/// Seed for reproducible trainer init + env, threaded through both
/// `SacConfig::seed` and `PendulumSwingUp::with_seed`.
const SEED: u64 = 0;

/// Log (and emit a CSV row) every this many env steps.
const LOG_INTERVAL: usize = 1_000;

/// Rescale a tanh-squashed actor action in `(-1, 1)` to the env's torque
/// range `[-MAX_TORQUE, MAX_TORQUE]`.
fn scale_action(action: &[f32]) -> Vec<f32> {
    action.iter().map(|a| a * MAX_TORQUE).collect()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let total_timesteps: usize = std::env::var("TOTAL_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TIMESTEPS);

    tracing::info!("Starting Pendulum SAC Training (Burn backend: {})", BACKEND_LABEL);
    tracing::info!("Environment: PendulumSwingUp (continuous)");
    tracing::info!("  obs_dim    = {}", OBS_DIM);
    tracing::info!("  action_dim = {}", ACTION_DIM);
    tracing::info!("  max_torque = {}", MAX_TORQUE);
    tracing::info!("  total_timesteps = {}", total_timesteps);
    tracing::info!(
        "  batch_size = {}  hidden_dim = {}  layers = {}",
        BATCH_SIZE,
        HIDDEN_DIM,
        NUM_HIDDEN_LAYERS
    );

    let training_start = std::time::Instant::now();
    let device = Default::default();

    // Optional learning-curve CSV. When CURVE_CSV is set we write one
    // `env_steps,mean_episode_reward` row per logging interval. Pendulum
    // reward is the (negative) mean episode return.
    let mut curve_csv = open_curve_csv()?;

    let config = SacConfig::new()
        .buffer_capacity(BUFFER_CAPACITY)
        .min_buffer_size(MIN_BUFFER_SIZE)
        .learning_starts(LEARNING_STARTS)
        .batch_size(BATCH_SIZE)
        .hidden_dim(HIDDEN_DIM)
        .num_hidden_layers(NUM_HIDDEN_LAYERS)
        .seed(SEED);
    let mut trainer = SacTrainer::<Backend>::new(config, OBS_DIM, ACTION_DIM, device)?;

    tracing::info!("------------------------------------------------------------");

    // --- Seeded training loop --------------------------------------------
    let mut env = PendulumSwingUp::with_seed(SEED);
    env.reset();
    let mut obs = env.get_observation();

    // Episode-return tracking: SAC drives one env at a time.
    let mut current_return = 0.0_f32;
    let mut completed_returns: Vec<f32> = Vec::new();
    let mut mean_return = 0.0_f32;
    let mut last_alpha = 0.0_f64;
    let mut last_buffer_len = 0_usize;
    let mut next_log = LOG_INTERVAL;

    for step in 1..=total_timesteps {
        let action = trainer.select_action(&obs);
        let result = env.step(scale_action(&action));
        let done = result.terminated || result.truncated;

        trainer
            .buffer_mut()
            .push(&obs, &action, result.reward, &result.observation, done);
        trainer.increment_env_step();

        current_return += result.reward;

        if let Some(stats) = trainer.train()? {
            last_alpha = stats.alpha;
            last_buffer_len = stats.buffer_len;
        }

        if done {
            completed_returns.push(current_return);
            current_return = 0.0;
            trainer.increment_episodes(1);
            env.reset();
            obs = env.get_observation();
        } else {
            obs = result.observation;
        }

        // --- Log + emit a curve row per interval -------------------------
        if step >= next_log {
            if !completed_returns.is_empty() {
                let n = completed_returns.len();
                let recent = &completed_returns[n.saturating_sub(100)..];
                mean_return = recent.iter().sum::<f32>() / recent.len() as f32;
            }

            if let Some(w) = curve_csv.as_mut() {
                writeln!(w, "{},{:.4}", trainer.total_env_steps(), mean_return)?;
            }

            tracing::info!(
                "env_steps={:>7}/{}  episodes={:>5}  mean_return(last≤100)={:9.1}  alpha={:6.4}  buffer={:>6}",
                trainer.total_env_steps(),
                total_timesteps,
                trainer.total_episodes(),
                mean_return,
                last_alpha,
                last_buffer_len,
            );

            next_log += LOG_INTERVAL;
        }
    }

    if let Some(mut w) = curve_csv.take() {
        w.flush()?;
    }

    let training_duration = training_start.elapsed();
    tracing::info!("------------------------------------------------------------");
    tracing::info!("Training complete.");
    tracing::info!("  total env steps  : {}", trainer.total_env_steps());
    tracing::info!("  total episodes   : {}", trainer.total_episodes());
    tracing::info!("  total train steps: {}", trainer.total_train_steps());
    tracing::info!("  final mean return(last≤100): {:.1}", mean_return);
    tracing::info!("  training time    : {:.1}s", training_duration.as_secs_f64());
    tracing::info!(
        "  steps/sec        : {:.0}",
        total_timesteps as f64 / training_duration.as_secs_f64()
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
