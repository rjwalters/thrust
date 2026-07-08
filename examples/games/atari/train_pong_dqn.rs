//! Nature-DQN on ALE **Pong** via the `AtariPreprocess` subprocess adapter
//! (Epic #306, Phase 4 — issue #329).
//!
//! End-to-end Double-DQN trainer for Pong-v5 using the
//! [`NatureDqnQNetwork`](thrust_rl::policy::atari_cnn::NatureDqnQNetwork) +
//! [`DQNTrainerBurn`](thrust_rl::train::dqn::DQNTrainerBurn) stack. The
//! environment is Farama's `ale-py` driven as a **subprocess** through
//! [`AtariPreprocess`](thrust_rl::env::games::atari::AtariPreprocess) (Option D
//! from `docs/ALE_BINDING_STRATEGY.md`), with Machado preprocessing defaults
//! (frame-skip 4, frame-stack 4, sticky-action p=0.25, 84×84 grayscale,
//! life-loss termination).
//!
//! # Backend selection (compile-time)
//!
//! The concrete backend is chosen by Cargo features via a `#[cfg]` type alias,
//! matching `docs/BURN_BACKENDS.md` and `train_cartpole_modern.rs`:
//!
//! - `--features "training,env-atari,cuda"` → `Autodiff<Cuda<f32, i32>>` (Linux
//!   + NVIDIA — the alc-2 RTX 4090 run).
//! - `--features "training,env-atari,wgpu"` → `Autodiff<Wgpu<f32, i32>>`
//!   (any-vendor GPU; Metal/Vulkan/DX12).
//! - `--features "training,env-atari"` (no GPU feature) →
//!   `Autodiff<NdArray<f32>>` (CPU — the smoke-test path; too slow for a full
//!   5M-frame run).
//!
//! `env-atari` is the compile guard (`required-features` in `Cargo.toml`); the
//! GPU features are additive run-time backend selectors.
//!
//! # Runtime requirements
//!
//! `ale-py` must be importable by the worker's Python interpreter, and a Pong
//! ROM must be resolvable. Point `ATARI_PYTHON` at the interpreter that has
//! `ale-py` installed (falls back to `python3`). Run from the repo root so the
//! default relative worker path (`envs/atari/ale_worker.py`) resolves, or set
//! `ATARI_WORKER_SCRIPT` to an absolute path. See the runbook in
//! `docs/PONG_DQN_RUNBOOK.md`.
//!
//! # Replay-buffer memory (honest f32 math)
//!
//! The in-tree [`ReplayBuffer`](thrust_rl::buffer::replay::ReplayBuffer) stores
//! both `obs` and `next_obs` as `Vec<f32>` (4 bytes/value). One Pong
//! observation is `4 * 84 * 84 = 28_224` values, so one transition's frame
//! storage is `2 * 28_224 * 4 = 225_792` bytes ≈ 221 KiB. Hence:
//!
//! ```text
//! buffer at 1M   : 2 * 28224 * 4 * 1_000_000 = ~210 GiB   (Mnih 2015 — does NOT fit)
//! buffer at 100k : 2 * 28224 * 4 *   100_000 = ~21.0 GiB  (host RAM — DEFAULT here)
//! buffer at 50k  : 2 * 28224 * 4 *    50_000 = ~10.5 GiB  (host RAM — fallback)
//! ```
//!
//! The buffer lives in **host RAM**, not VRAM. Confirm host RAM before the run;
//! override the capacity with `BUFFER_CAPACITY` if 100k is too large. A future
//! u8 frame store (before the 1/255 scale) would cut this 4× (~5.3 GiB at 100k)
//! but requires a new buffer type — out of scope here.
//!
//! # Hyperparameters (Mnih 2015 adapted for a ≤10M-frame budget)
//!
//! | Parameter | Mnih 2015 (50M) | This run (5–10M) |
//! |---|---|---|
//! | `learning_rate` | 2.5e-4 (RMSProp) | 2.5e-4 (Adam) |
//! | `batch_size` | 32 | 32 |
//! | `buffer_capacity` | 1M | 100_000 (f32 budget) |
//! | `min_buffer_size` | 50_000 | 10_000 |
//! | `target_update_interval` | 10_000 | 10_000 (hard copy) |
//! | `gamma` | 0.99 | 0.99 |
//! | `epsilon_start`/`end` | 1.0 / 0.1 | 1.0 / 0.1 |
//! | `epsilon_decay_steps` | 1M | 1_000_000 |
//! | `max_grad_norm` | 10.0 | 10.0 |
//! | `soft_update_tau` | None (hard sync) | None (hard sync) |
//!
//! DQN reaches ~18.9 at 50M frames (Mnih 2015); Pong typically crosses zero
//! well within 10M frames per the spike analysis in
//! `docs/ALE_BINDING_STRATEGY.md`. This run targets ≤10M frames to produce a
//! *positive* score (beating the ~−21 random floor), not a ceiling score.
//!
//! # Usage
//!
//! Full run (alc-2, CUDA — see `docs/PONG_DQN_RUNBOOK.md`):
//!
//! ```bash
//! ATARI_PYTHON=/usr/bin/python3 \
//! CURVE_CSV="$HOME/pong_dqn_curve.csv" \
//! CHECKPOINT_INTERVAL=500000 CHECKPOINT_DIR="$HOME/pong_dqn_checkpoints" \
//! cargo run --release --features "training,env-atari,cuda" \
//!     --example train_pong_dqn
//! ```
//!
//! CPU smoke run (tiny budget; requires a local `ale-py`):
//!
//! ```bash
//! TOTAL_TIMESTEPS=2000 MIN_BUFFER_SIZE=200 BUFFER_CAPACITY=2000 \
//! ATARI_PYTHON=/path/to/venv/bin/python3 CURVE_CSV=/tmp/pong_smoke.csv \
//! cargo run --release --features "training,env-atari" --example train_pong_dqn
//! ```
//!
//! # Env-var knobs
//!
//! - `TOTAL_TIMESTEPS` (default `5_000_000`) — total env steps.
//! - `BUFFER_CAPACITY` (default `100_000`) — replay capacity (see RAM math).
//! - `MIN_BUFFER_SIZE` (default `10_000`) — warmup transitions before updates.
//! - `CURVE_CSV` (unset) — path for the `env_steps,mean_episode_reward` CSV.
//! - `CHECKPOINT_INTERVAL` (unset) — env-step period for weight snapshots.
//! - `CHECKPOINT_DIR` (default `checkpoints`) — directory for snapshots.
//! - `ATARI_PYTHON` — interpreter with `ale-py` (falls back to `python3`).

use std::io::Write;

use anyhow::Result;
use burn::{
    optim::AdamConfig,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Tensor, TensorData},
};
use rand::{SeedableRng, rngs::StdRng};
use thrust_rl::{
    env::{Environment, games::atari::AtariPreprocess},
    policy::atari_cnn::{NatureDqnConfig, NatureDqnQNetwork},
    train::{
        dqn::{DQNConfig, DQNTrainerBurn},
        optimizer::BurnOptimizer,
    },
    utils::cuda::default_burn_device,
};

// --- Compile-time backend selection (see docs/BURN_BACKENDS.md) -----------
// `cuda` takes precedence over `wgpu`; with neither, fall back to CPU NdArray.
#[cfg(feature = "cuda")]
type Inner = burn::backend::Cuda<f32, i32>;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Inner = burn::backend::Wgpu<f32, i32>;
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
type Inner = burn::backend::NdArray<f32>;

type B = burn::backend::Autodiff<Inner>;

#[cfg(feature = "cuda")]
const BACKEND_LABEL: &str = "Cuda<f32, i32> + Autodiff (NVIDIA)";
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND_LABEL: &str = "Wgpu<f32, i32> + Autodiff (GPU: Vulkan/Metal/DX12/WebGPU)";
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
const BACKEND_LABEL: &str = "NdArray<f32> + Autodiff (CPU)";

/// Frame-stack depth × height × width — the preprocessor's fixed layout.
const CHANNELS: usize = 4;
const HEIGHT: usize = 84;
const WIDTH: usize = 84;

/// Total env steps (default; overridable via `TOTAL_TIMESTEPS`).
const DEFAULT_TIMESTEPS: usize = 5_000_000;
/// Replay capacity (default; overridable via `BUFFER_CAPACITY`).
const DEFAULT_BUFFER_CAPACITY: usize = 100_000;
/// Warmup transitions before updates begin (default; via `MIN_BUFFER_SIZE`).
const DEFAULT_MIN_BUFFER_SIZE: usize = 10_000;
/// Default checkpoint directory when `CHECKPOINT_INTERVAL` is set.
const DEFAULT_CHECKPOINT_DIR: &str = "checkpoints";

/// Seed threaded through the Q-network FC init and the action/replay `StdRng`
/// so the learning curve is reproducible up to the backend's own
/// float-reduction nondeterminism.
const SEED: u64 = 0;

/// Progress-logging period, in env steps (overridable via `LOG_INTERVAL`;
/// used for both stdout logging and learning-curve CSV rows). The default
/// suits a multi-million-step run; the smoke test lowers it so the CSV and
/// progress lines appear within a tiny budget.
const DEFAULT_LOG_INTERVAL: usize = 10_000;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let total_timesteps = env_usize("TOTAL_TIMESTEPS", DEFAULT_TIMESTEPS);
    let buffer_capacity = env_usize("BUFFER_CAPACITY", DEFAULT_BUFFER_CAPACITY);
    let min_buffer_size = env_usize("MIN_BUFFER_SIZE", DEFAULT_MIN_BUFFER_SIZE);
    let checkpoint_interval = std::env::var("CHECKPOINT_INTERVAL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0);
    let checkpoint_dir =
        std::env::var("CHECKPOINT_DIR").unwrap_or_else(|_| DEFAULT_CHECKPOINT_DIR.to_string());
    let log_interval = env_usize("LOG_INTERVAL", DEFAULT_LOG_INTERVAL).max(1);

    tracing::info!("Starting Pong DQN Training (Burn backend: {})", BACKEND_LABEL);

    // --- Environment (Machado defaults) ----------------------------------
    let mut env = AtariPreprocess::new("pong", SEED)?;
    let obs_shape = env.observation_space().shape;
    let obs_len = obs_shape.iter().product::<usize>();
    let n_actions = match env.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n as i64,
        thrust_rl::env::SpaceType::Box => {
            anyhow::bail!("Pong must expose a discrete action space")
        }
    };

    tracing::info!("Environment: ALE Pong (AtariPreprocess, Machado defaults)");
    tracing::info!("  obs_shape       = {:?}  (obs_len = {})", obs_shape, obs_len);
    tracing::info!("  n_actions       = {}", n_actions);
    tracing::info!("  total_timesteps = {}", total_timesteps);
    tracing::info!(
        "  buffer_capacity = {}  (~{:.1} GiB host RAM)",
        buffer_capacity,
        buffer_gib(buffer_capacity, obs_len)
    );
    tracing::info!("  min_buffer_size = {}", min_buffer_size);

    let device = default_burn_device::<B>();

    // --- Online Q-network (target is cloned internally by the trainer) ----
    let q_config = NatureDqnConfig::default().with_seed(SEED);
    let online = NatureDqnQNetwork::<B>::with_config(n_actions as usize, q_config, &device);

    // Mnih-2015-adapted hyperparameters (see module docs). Hard target sync
    // (no soft tau) matches the Nature-DQN spec.
    let config = DQNConfig::new()
        .learning_rate(2.5e-4)
        .batch_size(32)
        .buffer_capacity(buffer_capacity)
        .min_buffer_size(min_buffer_size)
        .target_update_interval(10_000)
        .gamma(0.99)
        .epsilon_start(1.0)
        .epsilon_end(0.1)
        .epsilon_decay_steps(1_000_000)
        .max_grad_norm(10.0);

    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<B, NatureDqnQNetwork<B>, _> =
        BurnOptimizer::new(inner_opt, config.learning_rate);

    // The trainer takes the device by value; the action-selection closure below
    // needs it too. Clone here so both hold a device. `clone_on_copy` is
    // silenced because on the CPU (`NdArray`) backend the device is `Copy` and
    // the clone is a no-op, whereas the `cuda`/`wgpu` devices are not `Copy` and
    // genuinely require it — one source across all three backends.
    #[allow(clippy::clone_on_copy)]
    let trainer_device = device.clone();
    let mut trainer =
        DQNTrainerBurn::new(config, online, burn_opt, obs_len, n_actions, trainer_device)?;

    // --- Forward closures: reshape the flat replay obs to NCHW ------------
    // Both the online and target forward passes share this reshape.
    let forward_fn = |q: &NatureDqnQNetwork<B>, o_flat: Tensor<B, 2>| -> Tensor<B, 2> {
        let b = o_flat.dims()[0];
        q.forward(o_flat.reshape([b, CHANNELS, HEIGHT, WIDTH]))
    };

    env.reset();
    let mut obs = env.get_observation();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);

    let mut curve_csv = open_curve_csv()?;

    let mut episode_return: f32 = 0.0;
    let mut episode_returns: Vec<f32> = Vec::new();
    let mut last_log_step = 0_usize;
    let mut last_checkpoint_step = 0_usize;
    let training_start = std::time::Instant::now();

    while trainer.total_env_steps() < total_timesteps {
        // ε-greedy action selection: greedy branch reshapes a single obs.
        let action = {
            trainer.select_action(&obs, &mut rng, |q: &NatureDqnQNetwork<B>, o_host: &[f32]| {
                let o_t: Tensor<B, 2> =
                    Tensor::from_data(TensorData::new(o_host.to_vec(), [1, o_host.len()]), &device);
                let q_values = q.forward(o_t.reshape([1, CHANNELS, HEIGHT, WIDTH]));
                let q_host: Vec<f32> = q_values.into_data().to_vec().unwrap_or_default();
                argmax(&q_host)
            })
        };

        let result = env.step(action);
        let next_obs = result.observation.clone();
        let done = result.terminated || result.truncated;
        trainer.buffer_mut().push(&obs, action, result.reward, &next_obs, done);

        episode_return += result.reward;
        obs = next_obs;

        trainer.increment_env_step();
        // Hard target sync on the interval (no soft blend — clone online).
        let _ = trainer.maybe_sync_target(|online, _target, _tau| online.clone());

        if done {
            episode_returns.push(episode_return);
            trainer.increment_episodes(1);
            episode_return = 0.0;
            env.reset();
            obs = env.get_observation();
        }

        // One gradient update per env step (skipped until warmup completes).
        let _ = trainer.train_step(&mut rng, forward_fn, forward_fn)?;

        let step = trainer.total_env_steps();

        // Periodic logging + learning-curve CSV row.
        if step.saturating_sub(last_log_step) >= log_interval {
            last_log_step = step;
            let recent_avg = mean_last(&episode_returns, 100);
            if let Some(w) = curve_csv.as_mut() {
                writeln!(w, "{},{:.4}", step, recent_avg)?;
                w.flush()?;
            }
            let fps = step as f64 / training_start.elapsed().as_secs_f64();
            tracing::info!(
                "step={:>8}  episodes={:>5}  avg(last≤100)={:7.2}  ε={:.3}  buf={:>7}  fps={:.0}",
                step,
                trainer.total_episodes(),
                recent_avg,
                trainer.last_epsilon(),
                trainer.buffer_len(),
                fps,
            );
        }

        // Periodic checkpointing (opt-in via CHECKPOINT_INTERVAL).
        if let Some(interval) = checkpoint_interval
            && step.saturating_sub(last_checkpoint_step) >= interval
        {
            last_checkpoint_step = step;
            save_checkpoint(trainer.online(), &checkpoint_dir, step)?;
        }
    }

    if let Some(mut w) = curve_csv.take() {
        w.flush()?;
    }

    let final_avg = mean_last(&episode_returns, 100);
    let elapsed = training_start.elapsed();
    tracing::info!("------------------------------------------------------------");
    tracing::info!("Training complete.");
    tracing::info!("  episodes         : {}", trainer.total_episodes());
    tracing::info!("  env_steps        : {}", trainer.total_env_steps());
    tracing::info!("  train_steps      : {}", trainer.total_train_steps());
    tracing::info!("  avg(last≤100)    : {:.2}", final_avg);
    tracing::info!("  wall clock       : {:.1}s", elapsed.as_secs_f64());
    tracing::info!(
        "  env-steps/sec    : {:.0}",
        trainer.total_env_steps() as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}

/// Argmax over a slice of Q-values (ties break to the lowest index).
fn argmax(qs: &[f32]) -> i64 {
    let mut best = 0_i64;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in qs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as i64;
        }
    }
    best
}

/// Mean of the last `window` entries (or all, if fewer). Returns `0.0` when
/// empty.
fn mean_last(xs: &[f32], window: usize) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let slice = &xs[xs.len().saturating_sub(window)..];
    slice.iter().copied().sum::<f32>() / slice.len() as f32
}

/// Host-RAM footprint (GiB) of a replay buffer holding `capacity` transitions,
/// storing both `obs` and `next_obs` as `f32`. `obs_len` values per frame.
fn buffer_gib(capacity: usize, obs_len: usize) -> f64 {
    (2 * obs_len * 4 * capacity) as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Parse a `usize` env var, falling back to `default` when unset or
/// unparseable.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Save the online Q-network weights to `<dir>/pong_dqn_<step>.bin`.
///
/// Uses Burn's [`BinFileRecorder`] (same recorder the self-play Pong example
/// uses). Creates the directory if missing.
fn save_checkpoint(q: &NatureDqnQNetwork<B>, dir: &str, step: usize) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    // `save_file` appends the recorder's own extension (`.bin`), so pass the
    // stem without it.
    let stem = format!("{dir}/pong_dqn_{step}");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    burn::module::Module::save_file(q.clone(), &stem, &recorder)
        .map_err(|e| anyhow::anyhow!("checkpoint write failed: {e}"))?;
    tracing::info!("  checkpoint written: {stem}.bin (env_steps={step})");
    Ok(())
}

/// Open the opt-in learning-curve CSV writer.
///
/// Returns `Ok(Some(writer))` with the header row already written when the
/// `CURVE_CSV` env var names a non-empty path, or `Ok(None)` when it is unset.
/// Schema (`env_steps,mean_episode_reward`) matches the CartPole DQN/A2C/PPO
/// curves so Pong's curve is tooling-compatible.
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
