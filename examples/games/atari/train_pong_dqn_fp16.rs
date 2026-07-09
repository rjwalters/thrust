//! Mixed-precision (**f16**) Nature-DQN on ALE **Pong** — the opt-in
//! reduced-precision sibling of [`train_pong_dqn`] (issue #270, epic #267).
//!
//! This binary trains the exact same Double-DQN recipe as
//! `examples/games/atari/train_pong_dqn.rs` (same
//! [`NatureDqnQNetwork`](thrust_rl::policy::atari_cnn::NatureDqnQNetwork) +
//! [`DQNTrainerBurn`](thrust_rl::train::dqn::DQNTrainerBurn) stack, same
//! `AtariPreprocess` subprocess env, same Mnih-2015-adapted hyperparameters and
//! env-var knobs) but pins the Burn backend's float element to **f16** on GPU
//! backends and wraps the backward pass in **manual dynamic loss scaling**.
//!
//! # Why a separate binary
//!
//! The f32 path in `train_pong_dqn.rs` is left byte-for-byte unchanged. This
//! example opts into the f16 machinery through the `training-fp16` Cargo
//! feature and a distinct backward path
//! ([`DQNTrainerBurn::train_step_scaled`]), so enabling mixed precision never
//! perturbs the reference f32 run.
//!
//! # Backend + dtype selection (compile-time)
//!
//! The concrete backend is chosen by Cargo features via `#[cfg]` type aliases.
//! Precedence `cuda > wgpu > NdArray(CPU)`; `training-fp16` swaps the float
//! element from `f32` to `f16` on the GPU backends only:
//!
//! - `--features "training,env-atari,cuda,training-fp16"` → `Autodiff<Cuda<f16,
//!   i32>>` — **the verified path** (alc-2 RTX 4090).
//! - `--features "training,env-atari,cuda"` → `Autodiff<Cuda<f32, i32>>` (f16
//!   off — identical backend to the reference example, useful for A/B).
//! - `--features "training,env-atari,wgpu,training-fp16"` → `Autodiff<Wgpu<f16,
//!   i32>>` — **compiles but is not runtime-verified**; wgpu/Metal has no bf16
//!   matmul in Burn 0.21 and f16 on Metal is untested (see #305). CUDA is the
//!   supported runtime.
//! - `--features "training,env-atari,wgpu"` → `Autodiff<Wgpu<f32, i32>>`.
//! - `--features "training,env-atari"` (no GPU feature) →
//!   `Autodiff<NdArray<f32>>` (CPU; f16 is unavailable on NdArray, so the
//!   `training-fp16` flag is a hard `compile_error!` without a GPU backend).
//!
//! Enabling `training-fp16` **without** `cuda` or `wgpu` is a `compile_error!`:
//! NdArray implements `FloatNdArrayElement` only for f32/f64, so `NdArray<f16>`
//! cannot compile.
//!
//! # Loss scaling — why f16 needs it and bf16 does not
//!
//! f16 has a ~5-bit exponent (min normal ≈ 6.1e-5). Gradients backpropagated
//! through the Nature-DQN's three conv layers routinely fall below that floor
//! and **underflow to zero**, stalling learning — especially early, when the
//! TD error is small relative to the conv-stack depth. The standard fix
//! (Micikevicius et al. 2018, "Mixed Precision Training") is **loss scaling**:
//! multiply the loss by a large factor `S` before `.backward()`, which shifts
//! every gradient up by `S` into f16's representable range, then divide the
//! gradients by `S` before the optimizer step to recover the true update.
//!
//! **bf16** has the full f32 8-bit exponent (same dynamic range, fewer mantissa
//! bits) and generally trains **without** loss scaling. It would be the
//! lower-risk dtype — but bf16 matmul is unavailable on the wgpu/Metal runtime
//! in Burn 0.21 (cubek-matmul 0.2.0), tracked in #305. This example is
//! therefore f16-on-CUDA; a bf16 variant can drop the scaler entirely once
//! #305 resolves.
//!
//! Burn 0.21 ships **no** `GradScaler`/AMP utility, so the scaling is manual.
//! The scale/unscale wrapper lives in [`DQNTrainerBurn::train_step_scaled`] (an
//! additive trainer method; the f32 `train_step` is untouched); this binary
//! owns the *dynamic* schedule:
//!
//! - Initial scale `2^15 = 32768`.
//! - Each step: if the host-side **unscaled** loss scalar is non-finite
//!   (NaN/inf — the overflow proxy, since Burn 0.21 exposes no per-gradient
//!   finiteness API), **halve** the scale and **skip** the optimizer step.
//! - After `2000` consecutive clean (applied) steps, **double** the scale up to
//!   a `2^24` ceiling — recovering headroom as the loss magnitude shrinks.
//! - Every scale change is logged.
//!
//! # Precision of parameters, activations, and Adam state (honest note)
//!
//! With `Autodiff<Cuda<f16, i32>>`, Burn's backend float element applies to
//! **everything**: conv/FC weights and biases, activations, gradients, **and
//! the Adam optimizer's first/second moment estimates are all f16**. There is
//! no automatic master-weight promotion.
//!
//! **Adam-second-moment underflow risk.** Adam's update is
//! `θ -= lr · m̂ / (√v̂ + ε)`. In f16, the second moment `v_t` (a running mean
//! of *squared* gradients) can underflow to zero for very small gradients,
//! making the effective step `m̂/(√0 + ε)` blow up. Loss scaling mitigates the
//! *gradient* underflow but not the *moment* underflow directly. The robust
//! fixes are (a) **bf16** (f32 dynamic range — pending #305), or (b) a
//! **master-weight ring** (store f32 params + f32 Adam moments, compute in
//! f16). Both are **out of scope** for this v1 — it is deliberately the
//! simplest "all-f16" surface so the type machinery and the CUDA runtime path
//! can be verified end-to-end first.
//!
//! **`flex32` as a lower-risk alternative.** cubecl exposes `flex32`, a
//! relaxed-precision f32 that *computes* in reduced precision but *stores* as
//! f32 — so Adam moments stay f32 and no loss scaling is needed. For a first
//! reduced-precision experiment it is arguably the safest option; swap the
//! `Inner` alias to `Cuda<burn::tensor::flex32, i32>` and skip the scaler to
//! try it (verify the alias against your Burn version first). This example
//! targets true f16 because NVIDIA tensor cores are optimized for it.
//!
//! # Runtime requirements & usage
//!
//! Identical to `train_pong_dqn.rs`: `ale-py` must be importable by the worker
//! Python interpreter (`ATARI_PYTHON`), and a Pong ROM must resolve. Run from
//! the repo root. See `docs/PONG_DQN_RUNBOOK.md`.
//!
//! Acceptance smoke run (alc-2, CUDA):
//!
//! ```bash
//! TOTAL_TIMESTEPS=500000 CHECKPOINT_INTERVAL=0 \
//! CURVE_CSV="$HOME/fp16_smoke_curve.csv" \
//! ATARI_PYTHON=/usr/bin/python3 \
//! cargo run --release --features "training,env-atari,cuda,training-fp16" \
//!     --example train_pong_dqn_fp16
//! ```
//!
//! The learning curve should track the f32 reference
//! (`docs/research/data/2026-07-pong-dqn-run2a-lr6.25e-5.csv`: ≈ −20.6 at 200k,
//! −19.5 at 430k) rather than sticking at the −21 random floor.
//!
//! # Env-var knobs
//!
//! Same as `train_pong_dqn.rs` (`LEARNING_RATE`, `TOTAL_TIMESTEPS`,
//! `BUFFER_CAPACITY`, `MIN_BUFFER_SIZE`, `CURVE_CSV`, `CHECKPOINT_INTERVAL`,
//! `CHECKPOINT_DIR`, `LOG_INTERVAL`, `ATARI_PYTHON`), plus fp16-specific:
//!
//! - `LOSS_SCALE_INIT` (default `32768` = 2^15) — initial dynamic loss scale.
//! - `LOSS_SCALE_GROWTH_INTERVAL` (default `2000`) — clean steps before the
//!   scale doubles.

// --- Compile-time feature guard ------------------------------------------
// `training-fp16` selects an f16 GPU backend; without a GPU backend feature
// there is no f16-capable backend to select (NdArray implements
// `FloatNdArrayElement` only for f32/f64). Fail loudly at build time.
#[cfg(all(
    feature = "training-fp16",
    not(any(feature = "cuda", feature = "wgpu"))
))]
compile_error!(
    "`training-fp16` requires a GPU backend: add `cuda` (verified) or `wgpu` \
     (compiles, unverified) to your --features. NdArray (CPU) has no f16 \
     support — `FloatNdArrayElement` is implemented only for f32 and f64."
);

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

// --- Compile-time backend + dtype selection ------------------------------
// Precedence: cuda > wgpu > NdArray(CPU). `training-fp16` swaps the float
// element f32 → f16 on the GPU backends only. See module docs.

// cuda + fp16 — the verified path.
#[cfg(all(feature = "cuda", feature = "training-fp16"))]
type Inner = burn::backend::Cuda<burn::tensor::f16, i32>;
// cuda + f32.
#[cfg(all(feature = "cuda", not(feature = "training-fp16")))]
type Inner = burn::backend::Cuda<f32, i32>;
// wgpu + fp16 (no cuda) — compiles, runtime-unverified (see #305).
#[cfg(all(feature = "wgpu", not(feature = "cuda"), feature = "training-fp16"))]
type Inner = burn::backend::Wgpu<burn::tensor::f16, i32>;
// wgpu + f32 (no cuda).
#[cfg(all(
    feature = "wgpu",
    not(feature = "cuda"),
    not(feature = "training-fp16")
))]
type Inner = burn::backend::Wgpu<f32, i32>;
// CPU fallback (no GPU feature; only reachable when training-fp16 is OFF —
// the compile_error! above forbids training-fp16 here).
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
type Inner = burn::backend::NdArray<f32>;

type B = burn::backend::Autodiff<Inner>;

// Human-readable backend label + whether the loss scaler is active.
#[cfg(all(feature = "cuda", feature = "training-fp16"))]
const BACKEND_LABEL: &str = "Cuda<f16, i32> + Autodiff (NVIDIA, mixed-precision)";
#[cfg(all(feature = "cuda", not(feature = "training-fp16")))]
const BACKEND_LABEL: &str = "Cuda<f32, i32> + Autodiff (NVIDIA)";
#[cfg(all(feature = "wgpu", not(feature = "cuda"), feature = "training-fp16"))]
const BACKEND_LABEL: &str = "Wgpu<f16, i32> + Autodiff (GPU, mixed-precision, UNVERIFIED)";
#[cfg(all(
    feature = "wgpu",
    not(feature = "cuda"),
    not(feature = "training-fp16")
))]
const BACKEND_LABEL: &str = "Wgpu<f32, i32> + Autodiff (GPU)";
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
const BACKEND_LABEL: &str = "NdArray<f32> + Autodiff (CPU)";

/// Whether the reduced-precision loss-scaling path is compiled in. True only
/// for the `training-fp16` + GPU builds.
const FP16_ENABLED: bool = cfg!(feature = "training-fp16");

/// Frame-stack depth × height × width — the preprocessor's fixed layout.
const CHANNELS: usize = 4;
const HEIGHT: usize = 84;
const WIDTH: usize = 84;

/// Atari-standard Adam LR (Rainbow / Dopamine defaults); matches the f32
/// reference example.
const DEFAULT_LEARNING_RATE: f64 = 6.25e-5;

const DEFAULT_TIMESTEPS: usize = 5_000_000;
const DEFAULT_BUFFER_CAPACITY: usize = 100_000;
const DEFAULT_MIN_BUFFER_SIZE: usize = 10_000;
const DEFAULT_CHECKPOINT_DIR: &str = "checkpoints";
const SEED: u64 = 0;
const DEFAULT_LOG_INTERVAL: usize = 10_000;

// --- Dynamic loss-scaling constants (fp16 path) --------------------------
/// Initial loss scale, 2^15. Large enough to lift conv-stack gradients out of
/// the f16 underflow region; halved on the first overflow.
const DEFAULT_LOSS_SCALE_INIT: f64 = 32_768.0;
/// Consecutive clean steps before the scale doubles (recovering headroom as
/// the loss magnitude shrinks over training).
const DEFAULT_LOSS_SCALE_GROWTH_INTERVAL: usize = 2000;
/// Upper bound on the loss scale, 2^24. Keeps `loss × scale` well inside f32
/// host-scalar range and prevents runaway growth on a long clean streak.
const LOSS_SCALE_MAX: f64 = 16_777_216.0;
/// Lower bound on the loss scale. Below this, scaling is not helping and the
/// run is effectively broken; we clamp rather than collapse to zero.
const LOSS_SCALE_MIN: f64 = 1.0;

/// Dynamic loss-scale state for the f16 path (Micikevicius et al. 2018).
struct LossScaler {
    scale: f64,
    clean_steps: usize,
    growth_interval: usize,
    /// Count of skipped (overflowed) steps, for end-of-run diagnostics.
    total_skips: usize,
}

impl LossScaler {
    fn new(init: f64, growth_interval: usize) -> Self {
        Self { scale: init, clean_steps: 0, growth_interval, total_skips: 0 }
    }

    /// Halve the scale after an overflow (non-finite unscaled loss). Logs the
    /// change and resets the clean streak.
    fn on_overflow(&mut self) {
        let old = self.scale;
        self.scale = (self.scale / 2.0).max(LOSS_SCALE_MIN);
        self.clean_steps = 0;
        self.total_skips += 1;
        tracing::warn!(
            "loss-scale: overflow (non-finite loss) → halving {} → {} (skip #{}, step skipped)",
            old,
            self.scale,
            self.total_skips,
        );
    }

    /// Record a clean applied step; double the scale after `growth_interval`
    /// consecutive clean steps (up to the ceiling). Logs on growth.
    fn on_clean_step(&mut self) {
        self.clean_steps += 1;
        if self.clean_steps >= self.growth_interval && self.scale < LOSS_SCALE_MAX {
            let old = self.scale;
            self.scale = (self.scale * 2.0).min(LOSS_SCALE_MAX);
            self.clean_steps = 0;
            tracing::info!(
                "loss-scale: {} clean steps → doubling {} → {}",
                self.growth_interval,
                old,
                self.scale,
            );
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let learning_rate = env_f64("LEARNING_RATE", DEFAULT_LEARNING_RATE)?;
    let total_timesteps = env_usize("TOTAL_TIMESTEPS", DEFAULT_TIMESTEPS)?;
    let buffer_capacity = env_usize("BUFFER_CAPACITY", DEFAULT_BUFFER_CAPACITY)?;
    let min_buffer_size = env_usize("MIN_BUFFER_SIZE", DEFAULT_MIN_BUFFER_SIZE)?;
    let checkpoint_interval =
        match std::env::var("CHECKPOINT_INTERVAL") {
            Ok(s) => Some(s.parse::<usize>().map_err(|e| {
                anyhow::anyhow!("CHECKPOINT_INTERVAL={s:?} is not a valid usize: {e}")
            })?)
            .filter(|&n| n > 0),
            Err(_) => None,
        };
    let checkpoint_dir =
        std::env::var("CHECKPOINT_DIR").unwrap_or_else(|_| DEFAULT_CHECKPOINT_DIR.to_string());
    let log_interval = env_usize("LOG_INTERVAL", DEFAULT_LOG_INTERVAL)?.max(1);
    let loss_scale_init = env_f64("LOSS_SCALE_INIT", DEFAULT_LOSS_SCALE_INIT)?;
    let loss_scale_growth =
        env_usize("LOSS_SCALE_GROWTH_INTERVAL", DEFAULT_LOSS_SCALE_GROWTH_INTERVAL)?.max(1);

    tracing::info!("Starting Pong DQN Training (Burn backend: {})", BACKEND_LABEL);
    if FP16_ENABLED {
        tracing::info!(
            "  mixed-precision: f16 with dynamic loss scaling (init={}, growth_interval={})",
            loss_scale_init,
            loss_scale_growth,
        );
    } else {
        tracing::info!(
            "  mixed-precision: DISABLED (f32 path). Add `training-fp16` + a GPU feature to enable."
        );
    }

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
    tracing::info!("  learning_rate   = {}  (Adam)", learning_rate);

    let device = default_burn_device::<B>();

    let q_config = NatureDqnConfig::default().with_seed(SEED);
    let online = NatureDqnQNetwork::<B>::with_config(n_actions as usize, q_config, &device);

    let config = DQNConfig::new()
        .learning_rate(learning_rate)
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

    #[allow(clippy::clone_on_copy)]
    let trainer_device = device.clone();
    let mut trainer =
        DQNTrainerBurn::new(config, online, burn_opt, obs_len, n_actions, trainer_device)?;

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
    let mut scaler = LossScaler::new(loss_scale_init, loss_scale_growth);
    let training_start = std::time::Instant::now();

    while trainer.total_env_steps() < total_timesteps {
        let action = {
            trainer.select_action(&obs, &mut rng, |q: &NatureDqnQNetwork<B>, o_host: &[f32]| {
                let o_t: Tensor<B, 2> =
                    Tensor::from_data(TensorData::new(o_host.to_vec(), [1, o_host.len()]), &device);
                let q_values = q.forward(o_t.reshape([1, CHANNELS, HEIGHT, WIDTH]));
                // f16 read-back: convert to f32 host data before argmax so the
                // host scalar path is dtype-agnostic.
                let q_host: Vec<f32> =
                    q_values.into_data().convert::<f32>().to_vec().unwrap_or_default();
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
        let _ = trainer.maybe_sync_target(|online, _target, _tau| online.clone());

        if done {
            episode_returns.push(episode_return);
            trainer.increment_episodes(1);
            episode_return = 0.0;
            env.reset();
            obs = env.get_observation();
        }

        // One gradient update per env step. On the fp16 path, wrap it in the
        // dynamic loss scaler; the f32 path uses the plain step so its numerics
        // are identical to `train_pong_dqn.rs`.
        if FP16_ENABLED {
            if let Some((_stats, applied)) =
                trainer.train_step_scaled(&mut rng, scaler.scale, forward_fn, forward_fn)?
            {
                if applied {
                    scaler.on_clean_step();
                } else {
                    scaler.on_overflow();
                }
            }
        } else {
            let _ = trainer.train_step(&mut rng, forward_fn, forward_fn)?;
        }

        let step = trainer.total_env_steps();

        if step.saturating_sub(last_log_step) >= log_interval {
            last_log_step = step;
            let recent_avg = mean_last(&episode_returns, 100);
            if let Some(w) = curve_csv.as_mut() {
                writeln!(w, "{},{:.4}", step, recent_avg)?;
                w.flush()?;
            }
            let fps = step as f64 / training_start.elapsed().as_secs_f64();
            tracing::info!(
                "step={:>8}  episodes={:>5}  avg(last≤100)={:7.2}  ε={:.3}  buf={:>7}  fps={:.0}  scale={}",
                step,
                trainer.total_episodes(),
                recent_avg,
                trainer.last_epsilon(),
                trainer.buffer_len(),
                fps,
                scaler.scale,
            );
        }

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
    if FP16_ENABLED {
        tracing::info!("  final loss scale : {}", scaler.scale);
        tracing::info!("  skipped steps    : {}  (loss-scale overflows)", scaler.total_skips);
    }
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

/// Parse a `usize` env var, falling back to `default` only when unset; a
/// present-but-unparseable value is a hard error (operator surprise on a long
/// run).
fn env_usize(key: &str, default: usize) -> Result<usize> {
    match std::env::var(key) {
        Ok(s) => s
            .parse::<usize>()
            .map_err(|e| anyhow::anyhow!("{key}={s:?} is not a valid usize: {e}")),
        Err(_) => Ok(default),
    }
}

/// Parse an `f64` env var (fail-loud on unparseable), requiring a finite
/// positive value — matching the reference example's LR guard, and correct for
/// both `LEARNING_RATE` and the positive loss-scale knobs here.
fn env_f64(key: &str, default: f64) -> Result<f64> {
    match std::env::var(key) {
        Ok(s) => {
            let v = s
                .parse::<f64>()
                .map_err(|e| anyhow::anyhow!("{key}={s:?} is not a valid f64: {e}"))?;
            if !v.is_finite() || v <= 0.0 {
                anyhow::bail!("{key}={s:?} must be a finite positive f64");
            }
            Ok(v)
        }
        Err(_) => Ok(default),
    }
}

/// Save the online Q-network weights to `<dir>/pong_dqn_fp16_<step>.bin`.
fn save_checkpoint(q: &NatureDqnQNetwork<B>, dir: &str, step: usize) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let stem = format!("{dir}/pong_dqn_fp16_{step}");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    burn::module::Module::save_file(q.clone(), &stem, &recorder)
        .map_err(|e| anyhow::anyhow!("checkpoint write failed: {e}"))?;
    tracing::info!("  checkpoint written: {stem}.bin (env_steps={step})");
    Ok(())
}

/// Open the opt-in learning-curve CSV writer.
///
/// Schema (`env_steps,mean_episode_reward`) matches the f32 reference so the
/// fp16 curve is directly comparable against
/// `docs/research/data/2026-07-pong-dqn-run2a-lr6.25e-5.csv`.
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
