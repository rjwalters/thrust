//! MountainCarContinuous SAC training on the Burn backend.
//!
//! End-to-end seeded Soft Actor-Critic (SAC) trainer for the continuous
//! [`MountainCarContinuous`](thrust_rl::env::games::mountain_car_continuous::MountainCarContinuous)
//! env using [`SacTrainer`](thrust_rl::train::sac::SacTrainer) on
//! `Autodiff<NdArray<f32>>` (CPU). This is the runnable counterpart of the
//! convergence test (`tests/test_sac_mountain_car.rs`, issue #168, epic
//! #163) and the deceptive-reward sibling of the Pendulum SAC example
//! (`examples/games/pendulum/train_sac.rs`).
//!
//! Like the Pendulum SAC example, SAC here is **off-policy and continuous**:
//! one env step + one replay-sampled gradient update + one buffer push,
//! every single env step. Unlike Pendulum (which rescales the actor action
//! by `MAX_TORQUE = 2.0`), MountainCarContinuous's force range is already
//! `[-1, 1]` and the SAC actor emits a tanh-squashed action in `(-1, 1)`, so
//! **no rescaling is needed** — the actor action is passed straight to
//! `env.step(action)`.
//!
//! # Exploration: momentum-pumping warmup
//!
//! MountainCarContinuous has a deceptive / sparse reward — the only positive
//! signal (+100) is at the goal. SAC's default per-step uniform warmup
//! averages to ~0 force and never builds enough momentum to summit, so the
//! replay buffer holds no goal transition and the policy collapses to
//! near-zero force (mean return ≈ -0.02; never solves). To make this a real
//! benchmark, the warmup window here drives an **energy-pumping** explorer
//! ([`pump_action`]): push at (near-)full force in the direction of the
//! current velocity, the textbook strategy that rocks the car up the slope.
//! This seeds the buffer with goal-reaching episodes, after which SAC
//! converges to ~ +94. `learning_starts` is therefore set to `0` (we supply
//! our own warmup actions).
//!
//! # Architecture
//!
//! - `MountainCarContinuous`: `obs_dim = 2` (position, velocity), `action_dim =
//!   1` (engine force).
//! - SAC actor (256-wide, 2 hidden layers) + twin critics + targets.
//! - Replay buffer warmed for `learning_starts` steps before updates fire.
//! - Auto-tuned entropy temperature `alpha`.
//! - Seeded via `SacConfig::seed` and `MountainCarContinuous::with_seed`.
//! - Energy-pumping warmup (see [`pump_action`]) seeds the buffer with
//!   goal-reaching transitions before the SAC policy takes over.
//! - Total budget: 30k env steps by default (the convergence-test budget, which
//!   reaches ~ +94 with the momentum warmup).
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_sac_mountain_car --features training --release
//! ```
//!
//! Override the total step budget via the `TOTAL_TIMESTEPS` env var (a
//! short run is a fast smoke check; convergence needs ~30k):
//!
//! ```bash
//! TOTAL_TIMESTEPS=4000 cargo run --example train_sac_mountain_car \
//!     --features training --release
//! ```
//!
//! Expected: mean episode return climbs from a near-zero/slightly-negative
//! baseline (a non-summiting policy only pays the small control-effort
//! penalty) toward the +90..+100 "solved" range once the policy learns to
//! reach the goal flag (the convergence bar is +90 over the final eval
//! episodes).
//!
//! # Learning-curve CSV (opt-in)
//!
//! Set `CURVE_CSV=<path>` to emit one `env_steps,mean_episode_reward` row
//! per logging interval (header row first). Training is seeded, so re-runs
//! reproduce the same CSV byte-for-byte. When `CURVE_CSV` is unset, no file
//! is written and behavior is unchanged.
//!
//! ```bash
//! CURVE_CSV=/tmp/mc.csv cargo run --example train_sac_mountain_car \
//!     --features training --release
//! ```

use std::io::Write;

use anyhow::Result;
use burn::backend::{Autodiff, NdArray};
use rand::{Rng, SeedableRng, rngs::StdRng};
use thrust_rl::{
    env::{Environment, games::mountain_car_continuous::MountainCarContinuous},
    train::sac::{SacConfig, SacTrainer},
};

type Backend = Autodiff<NdArray<f32>>;

const BACKEND_LABEL: &str = "NdArray<f32> + Autodiff (CPU)";

/// MountainCarContinuous observation / action dimensions.
const OBS_DIM: usize = 2;
const ACTION_DIM: usize = 1;

/// Default total env-step budget (matches the convergence test).
const DEFAULT_TIMESTEPS: usize = 30_000;

/// Energy-pumping warmup window: number of initial env steps driven by the
/// momentum explorer ([`pump_action`]) to seed the buffer with goal-reaching
/// transitions. We supply our own warmup, so `learning_starts` is `0`.
const WARMUP_STEPS: usize = 2_000;

/// SAC network / replay hyperparameters (mirror the convergence test).
const BUFFER_CAPACITY: usize = 100_000;
const MIN_BUFFER_SIZE: usize = 1_000;
const BATCH_SIZE: usize = 256;
const HIDDEN_DIM: usize = 256;
const NUM_HIDDEN_LAYERS: usize = 2;

/// Seed for reproducible trainer init + env, threaded through both
/// `SacConfig::seed` and `MountainCarContinuous::with_seed`.
const SEED: u64 = 0;

/// Log (and emit a CSV row) every this many env steps.
const LOG_INTERVAL: usize = 1_000;

/// Energy-pumping warmup exploration: push at (near-)full force in the
/// direction of the current velocity, with a little jitter to keep the
/// transitions stochastic. This is the textbook MountainCar strategy that
/// builds momentum and reaches the goal, seeding the replay buffer with the
/// positive-reward transitions SAC needs. `velocity` is `obs[1]`.
fn pump_action(velocity: f32, rng: &mut StdRng) -> Vec<f32> {
    let base = if velocity >= 0.0 { 1.0 } else { -1.0 };
    let jitter: f32 = rng.random_range(-0.2..0.2);
    vec![(base + jitter).clamp(-1.0, 1.0)]
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let total_timesteps: usize = std::env::var("TOTAL_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TIMESTEPS);

    tracing::info!("Starting MountainCarContinuous SAC Training (Burn backend: {})", BACKEND_LABEL);
    tracing::info!("Environment: MountainCarContinuous (continuous, deceptive reward)");
    tracing::info!("  obs_dim    = {}", OBS_DIM);
    tracing::info!("  action_dim = {}", ACTION_DIM);
    tracing::info!("  force_range = [-1, 1] (no action rescaling)");
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
    // `env_steps,mean_episode_reward` row per logging interval.
    let mut curve_csv = open_curve_csv()?;

    let config = SacConfig::new()
        .buffer_capacity(BUFFER_CAPACITY)
        .min_buffer_size(MIN_BUFFER_SIZE)
        // We drive the warmup ourselves with pump_action, so disable the
        // trainer's built-in uniform-random warmup.
        .learning_starts(0)
        .batch_size(BATCH_SIZE)
        .hidden_dim(HIDDEN_DIM)
        .num_hidden_layers(NUM_HIDDEN_LAYERS)
        .seed(SEED);
    let mut trainer = SacTrainer::<Backend>::new(config, OBS_DIM, ACTION_DIM, device)?;

    tracing::info!("------------------------------------------------------------");

    // --- Seeded training loop --------------------------------------------
    let mut env = MountainCarContinuous::with_seed(SEED);
    env.reset();
    let mut obs = env.get_observation();

    // Episode-return tracking: SAC drives one env at a time.
    let mut current_return = 0.0_f32;
    let mut completed_returns: Vec<f32> = Vec::new();
    let mut mean_return = 0.0_f32;
    let mut last_alpha = 0.0_f64;
    let mut last_buffer_len = 0_usize;
    let mut next_log = LOG_INTERVAL;
    let mut explore_rng = StdRng::seed_from_u64(12_345);

    for step in 1..=total_timesteps {
        // No action rescaling: the env's force range is already [-1, 1].
        // During warmup use the momentum-pumping explorer to seed the buffer
        // with goal-reaching transitions; afterward follow the SAC policy.
        let action = if step <= WARMUP_STEPS {
            pump_action(obs[1], &mut explore_rng)
        } else {
            trainer.select_action(&obs)
        };
        let result = env.step(action.clone());
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
