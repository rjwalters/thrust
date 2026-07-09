//! Throughput / wall-clock benchmarks for the trainers.
//!
//! Measures steps/sec and per-update wall-clock for the on-policy A2C
//! ([`A2cTrainer`]) and PPO ([`PPOTrainerBurn`]) trainers and the off-policy
//! DQN ([`DQNTrainerBurn`]) and SAC ([`SacTrainer`]) trainers. The harness is
//! **generic over the Burn backend** `B: AutodiffBackend`: every fixture
//! builder, data builder, and `bench_*` function is parameterized over `B` and
//! takes an explicit `&B::Device`, so the exact same bench bodies can be run on
//! any compiled backend with no logic duplication. The trainers/policies are
//! themselves already generic over `B: AutodiffBackend`.
//!
//! # Backend registration
//!
//! [`register_all`] runs the whole twelve-group suite for one backend, tagging
//! each criterion group with a `suffix` (e.g. `a2c_train_step/ndarray`) so
//! different backends produce distinct, side-by-side groups. The driver
//! registers the CPU baseline unconditionally:
//!
//! ```text
//! type Cpu = Autodiff<NdArray<f32>>;
//! register_all::<Cpu>(c, &default_burn_device::<Cpu>(), "ndarray");
//! ```
//!
//! The CPU `NdArray` baseline is always registered. GPU backends are added at
//! the same seam behind Cargo-feature gates, reusing the exact same generic
//! bench bodies with no logic duplication:
//!
//! ```text
//! #[cfg(feature = "wgpu")]
//! { type Gpu = Autodiff<Wgpu<f32, i32>>;
//!   register_all::<Gpu>(c, &default_burn_device::<Gpu>(), "wgpu"); }
//! ```
//!
//! Because CI is CPU-host only and never sets `wgpu`/`cuda`, the default
//! `cargo bench --features training` path always emits exactly the same twelve
//! logical groups, now suffixed `/ndarray`, and never compiles or
//! constructs a GPU backend. The GPU registration code only compiles when the
//! corresponding feature is enabled — see the "Throughput benchmarking on GPU
//! backends" section of `docs/BURN_BACKENDS.md`.
//!
//! All benchmarks are seeded so the numbers are comparable run-to-run.
//!
//! # Fairness caveat: on-policy vs off-policy steps/sec
//!
//! The `*_steps_per_sec` groups measure full-loop environment throughput and
//! are comparable WITHIN an algorithm class only (on-policy: A2C-vs-PPO;
//! off-policy: DQN-vs-SAC). They are NOT comparable across classes, because
//! the two classes do a different amount of gradient work per environment
//! step: an on-policy trainer collects `n_steps * num_envs` env interactions
//! and then amortizes ONE update over all of them, whereas an off-policy
//! trainer does ONE replay-sampled gradient update (on a `batch_size`
//! minibatch) for EVERY single env step. So off-policy does roughly
//! `batch_size` more gradient work per env step than on-policy — a raw
//! "DQN steps/sec vs A2C steps/sec" comparison is therefore meaningless.
//! Use the `*_train_step` groups (per-update cost on a fixed batch) as the
//! only cross-class comparable number. To keep the off-policy steps/sec
//! numbers honest, the replay buffer is pre-filled to `min_buffer_size` in
//! the untimed setup closure so the timed region measures steady-state
//! throughput (one real update per step) rather than the warmup transient
//! (env steps whose update is a `None` no-op).
//!
//! SAC is the second member of the off-policy class, so its per-update cost
//! (`sac_train_step`) is in the same cross-class-comparable family as
//! `dqn_train_step` (though SAC's update is heavier: it backprops a stochastic
//! actor, twin critics, and an entropy-temperature term per step, versus DQN's
//! single Q-network). SAC, however, runs on the *continuous* `PendulumSwingUp`
//! env (obs_dim=3, action_dim=1, action rescaled by `MAX_TORQUE = 2.0`), a
//! distinct environment from the CartPole groups. The
//! `sac_pendulum_steps_per_sec` steps/sec number is therefore additionally NOT
//! comparable across *environments*: it must not be read against the CartPole
//! `*_steps_per_sec` groups (different env dynamics, obs/action shapes, and
//! episode lengths).
//!
//! Benchmark groups (each suffixed with the backend tag, e.g. `/ndarray`):
//! - `a2c_cartpole_steps_per_sec` — a full rollout-collect (CartPole env
//!   stepping + host-side action sampling) plus exactly one A2C update,
//!   reported via `Throughput::Elements(n_steps * num_envs)` so criterion
//!   prints environment-steps/sec.
//! - `a2c_train_step` — a single [`A2cTrainer::train_step`] on a fixed
//!   synthetic batch, isolating the gradient-step cost.
//! - `ppo_train_step` — a single [`PPOTrainerBurn::train_step`] on the *same*
//!   synthetic batch, for a head-to-head per-update comparison.
//! - `ppo_cartpole_steps_per_sec` — a full-loop PPO comparison mirroring the
//!   A2C full-loop benchmark.
//! - `dqn_train_step` — a single Double-DQN [`DQNTrainerBurn::train_step`] on a
//!   replay minibatch (buffer pre-filled untimed in setup), isolating the
//!   off-policy per-update cost. Cross-class comparable against
//!   `a2c_train_step` / `ppo_train_step`.
//! - `dqn_cartpole_steps_per_sec` — a full DQN env-loop (one CartPole step +
//!   buffer push + one gradient update per env step), reported in
//!   environment-steps/sec. Comparable within the off-policy class only.
//! - `sac_train_step` — a single [`SacTrainer::train`] update on a replay
//!   minibatch (buffer pre-filled and `learning_starts` satisfied untimed in
//!   setup), isolating the off-policy continuous-control per-update cost.
//!   Cross-class comparable against the other `*_train_step` groups.
//! - `sac_pendulum_steps_per_sec` — a full SAC env-loop on the continuous
//!   `PendulumSwingUp` env (one env step + `scale_action` + buffer push + one
//!   gradient update per env step), reported in environment-steps/sec.
//!   Comparable within the off-policy class only, and NOT across environments
//!   (Pendulum, not CartPole).
//!
//! The final four groups are the large-net Nature-DQN CNN crossover
//! benchmarks (issue #328). They feed synthetic NCHW `[B, 4, 84, 84]`
//! observations directly to the Nature-DQN conv stack (decoupled from env
//! stepping / subprocess IPC per the epic Phase 1 spike), at `b32` and `b256`:
//! - `nature_dqn_policy_forward` — isolated inference cost (no autograd) of the
//!   actor-critic `NatureDqnBurnPolicy::forward`.
//! - `nature_dqn_qnet_forward` — isolated inference cost of the
//!   `NatureDqnQNetwork::forward` Q-network.
//! - `nature_dqn_policy_train_step` — forward + backward + Adam step for the
//!   actor-critic policy (the per-minibatch cost PPO pays on Atari frames).
//! - `nature_dqn_qnet_train_step` — forward + backward + Adam step for the
//!   Q-network (the CNN analogue of `dqn_train_step`).
//!
//! Run quickly with, e.g.:
//! ```text
//! cargo bench --features training --bench trainer_throughput -- \
//!     --warm-up-time 1 --measurement-time 5
//! ```

use std::hint::black_box;

// GPU backend types are pulled in only when the corresponding feature is
// enabled, so the default CPU build never references (or compiles) them. CI is
// CPU-host only and never sets `wgpu`/`cuda` — see the driver `benches()`.
#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;
use burn::{
    backend::{Autodiff, NdArray},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend},
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};
use thrust_rl::{
    env::{
        Environment,
        games::{cartpole::CartPole, pendulum::PendulumSwingUp},
    },
    policy::{
        atari_cnn::{NatureDqnBurnPolicy, NatureDqnConfig, NatureDqnQNetwork},
        mlp::MlpBurnPolicy,
        q_network::QNetworkBurn,
    },
    train::{
        A2cConfig, A2cTrainer, BurnOptimizer, DQNConfig, DQNTrainerBurn, PPOConfig, PPOTrainerBurn,
        SacConfig, SacTrainer,
    },
    utils::cuda::default_burn_device,
};

// CartPole is a 4-dim observation, 2-action discrete control task.
const OBS_DIM: usize = 4;
const ACTION_DIM: usize = 2;
const HIDDEN_DIM: usize = 64;

// Synthetic-batch dimensions for the isolated `train_step` benchmarks.
// Kept small so each iteration completes in well under a millisecond on
// CPU while still exercising the full backward/optimizer path.
const SYNTH_BATCH: usize = 64;

// Full-loop rollout shape (mirrors A2C defaults: n_steps=5, num_envs=16).
const ROLLOUT_N_STEPS: usize = 5;
const ROLLOUT_NUM_ENVS: usize = 16;

// DQN replay/minibatch shape. `DQN_BATCH` is the per-update minibatch sampled
// from the replay buffer; `DQN_MIN_BUFFER` is the warmup threshold below which
// `train_step` is a no-op (`Ok(None)`). The fixtures pre-fill the buffer to
// `DQN_MIN_BUFFER` so every timed `train_step` performs a real gradient update.
const DQN_BATCH: usize = 64;
const DQN_MIN_BUFFER: usize = 256;
const DQN_BUFFER_CAPACITY: usize = 4_096;

// Number of timed env steps in the DQN full-loop benchmark. Each step is one
// CartPole transition + buffer push + one gradient update.
const DQN_LOOP_STEPS: usize = 16;

// SAC runs on the continuous PendulumSwingUp env: 3-dim observation
// (cosθ, sinθ, θ̇), 1-dim action. The actor emits a tanh-squashed action in
// (-1, 1) which is rescaled by `SAC_MAX_TORQUE` before the env step.
const SAC_OBS_DIM: usize = 3;
const SAC_ACTION_DIM: usize = 1;
const SAC_MAX_TORQUE: f32 = 2.0;

// SAC replay/minibatch shape. `SAC_BATCH` is the per-update minibatch sampled
// from the replay buffer; `SAC_MIN_BUFFER` is the warmup threshold below which
// `train()` is a no-op (`Ok(None)`). `SAC_LEARNING_STARTS` is the random-action
// warmup window — the fixtures advance `total_env_steps` past it so the actor
// (not uniform noise) drives `select_action`. The fixtures pre-fill the buffer
// to `SAC_MIN_BUFFER` so every timed `train()` performs a real gradient update.
// `hidden_dim`/`num_hidden_layers` are kept small relative to the SAC default
// (256-wide, 2 layers) so each CPU iteration completes quickly while still
// exercising the full actor + twin-critic + alpha backward path.
const SAC_BATCH: usize = 64;
const SAC_MIN_BUFFER: usize = 256;
const SAC_BUFFER_CAPACITY: usize = 4_096;
const SAC_LEARNING_STARTS: usize = 128;
const SAC_HIDDEN_DIM: usize = 64;
const SAC_NUM_HIDDEN_LAYERS: usize = 2;

// Number of timed env steps in the SAC full-loop benchmark. Each step is one
// Pendulum transition + buffer push + one gradient update.
const SAC_LOOP_STEPS: usize = 16;

// Nature-DQN CNN bench batch sizes.
// B=32 mirrors the Nature-DQN paper (Mnih 2015); B=256 probes the GPU
// crossover point (Atari PPO rollout minibatch upper range). In-tree trainer
// defaults are batch_size=64 for both DQN and PPO, so these are deliberately
// distinct from `SYNTH_BATCH`/`DQN_BATCH`. The CNN groups feed synthetic
// NCHW `[B, 4, 84, 84]` observations directly to the Nature-DQN conv stack.
const CNN_BATCH_DQN: usize = 32;
const CNN_BATCH_PPO: usize = 256;
// Standard ALE legal-action count (policy-/Q-head width).
const CNN_N_ACTIONS: usize = 18;
// Learning rate for the CNN train-step optimizer (typical Atari PPO/DQN lr).
const CNN_LR: f64 = 2.5e-4;
// Entropy-bonus coefficient in the synthetic actor-critic loss.
const CNN_ENTROPY_COEF: f32 = 0.01;

/// Build a fresh, seeded A2C trainer on backend `B`.
fn make_a2c_trainer<B: AutodiffBackend>(
    device: &B::Device,
) -> A2cTrainer<B, MlpBurnPolicy<B>, impl burn::optim::Optimizer<MlpBurnPolicy<B>, B>> {
    let policy = MlpBurnPolicy::<B>::new(OBS_DIM, ACTION_DIM, HIDDEN_DIM, device);
    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 7e-4);
    let config = A2cConfig::default();
    A2cTrainer::new(config, policy, burn_opt).expect("valid A2C config")
}

/// Build a fresh, seeded PPO trainer on backend `B`.
///
/// `n_epochs = 1` keeps the PPO per-update cost a like-for-like single
/// gradient pass against A2C's single update; the surrogate clip / value
/// clip remain to reflect PPO's real per-step overhead.
fn make_ppo_trainer<B: AutodiffBackend>(
    device: &B::Device,
) -> PPOTrainerBurn<B, MlpBurnPolicy<B>, impl burn::optim::Optimizer<MlpBurnPolicy<B>, B>> {
    let policy = MlpBurnPolicy::<B>::new(OBS_DIM, ACTION_DIM, HIDDEN_DIM, device);
    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 3e-4);
    let config = PPOConfig::new().n_epochs(1);
    PPOTrainerBurn::new(config, policy, burn_opt).expect("valid PPO config")
}

/// A deterministic synthetic training batch shared by the isolated
/// `train_step` benchmarks. Returns
/// `(observations, actions, old_log_probs, old_values, advantages, returns)`.
#[allow(clippy::type_complexity)]
fn synthetic_batch<B: AutodiffBackend>(
    device: &B::Device,
) -> (
    Tensor<B, 2>,
    Tensor<B, 1, Int>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
) {
    let mut rng = StdRng::seed_from_u64(12345);
    use rand::Rng;

    let obs_data: Vec<f32> =
        (0..SYNTH_BATCH * OBS_DIM).map(|_| rng.random_range(-1.0..1.0)).collect();
    let observations =
        Tensor::<B, 2>::from_data(TensorData::new(obs_data, [SYNTH_BATCH, OBS_DIM]), device);

    let action_data: Vec<i64> =
        (0..SYNTH_BATCH).map(|_| rng.random_range(0..ACTION_DIM as i64)).collect();
    let actions =
        Tensor::<B, 1, Int>::from_data(TensorData::new(action_data, [SYNTH_BATCH]), device);

    let old_log_probs_data: Vec<f32> =
        (0..SYNTH_BATCH).map(|_| -rng.random_range(0.5..1.5)).collect();
    let old_log_probs =
        Tensor::<B, 1>::from_data(TensorData::new(old_log_probs_data, [SYNTH_BATCH]), device);

    let old_values_data: Vec<f32> = (0..SYNTH_BATCH).map(|_| rng.random_range(-1.0..1.0)).collect();
    let old_values =
        Tensor::<B, 1>::from_data(TensorData::new(old_values_data, [SYNTH_BATCH]), device);

    let adv_data: Vec<f32> = (0..SYNTH_BATCH).map(|_| rng.random_range(-2.0..2.0)).collect();
    let advantages = Tensor::<B, 1>::from_data(TensorData::new(adv_data, [SYNTH_BATCH]), device);

    let returns_data: Vec<f32> = (0..SYNTH_BATCH).map(|_| rng.random_range(-1.0..1.0)).collect();
    let returns = Tensor::<B, 1>::from_data(TensorData::new(returns_data, [SYNTH_BATCH]), device);

    (observations, actions, old_log_probs, old_values, advantages, returns)
}

/// One rollout's worth of transitions collected from `num_envs` CartPole
/// instances stepped `n_steps` times under the given policy.
struct Rollout<B: AutodiffBackend> {
    observations: Tensor<B, 2>,
    actions: Tensor<B, 1, Int>,
    old_log_probs: Tensor<B, 1>,
    old_values: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
    returns: Tensor<B, 1>,
}

/// Collect a full rollout: step `num_envs` seeded CartPole envs for
/// `n_steps`, sampling actions from the policy on the host. Computes plain
/// n-step (Monte-Carlo, `gamma`-discounted) returns and advantages so the
/// downstream `train_step` has well-formed inputs. This is the env-bound
/// portion of the full-loop benchmark.
fn collect_rollout<B: AutodiffBackend>(
    policy: &MlpBurnPolicy<B>,
    device: &B::Device,
    rng: &mut StdRng,
) -> Rollout<B> {
    const GAMMA: f32 = 0.99;

    let mut envs: Vec<CartPole> = (0..ROLLOUT_NUM_ENVS)
        .map(|_| {
            let mut e = CartPole::new();
            e.reset();
            e
        })
        .collect();

    let total = ROLLOUT_N_STEPS * ROLLOUT_NUM_ENVS;
    let mut obs_flat: Vec<f32> = Vec::with_capacity(total * OBS_DIM);
    let mut actions_flat: Vec<i64> = Vec::with_capacity(total);
    let mut log_probs_flat: Vec<f32> = Vec::with_capacity(total);
    let mut values_flat: Vec<f32> = Vec::with_capacity(total);
    let mut rewards_flat: Vec<f32> = Vec::with_capacity(total);
    let mut dones_flat: Vec<bool> = Vec::with_capacity(total);

    for _ in 0..ROLLOUT_N_STEPS {
        // Batch the current observations across all envs for one policy
        // forward pass.
        let mut step_obs: Vec<f32> = Vec::with_capacity(ROLLOUT_NUM_ENVS * OBS_DIM);
        for env in &envs {
            step_obs.extend_from_slice(&env.get_observation());
        }
        let obs_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(step_obs.clone(), [ROLLOUT_NUM_ENVS, OBS_DIM]),
            device,
        );
        let (acts, lps, vals) = policy.get_action_host_seeded(obs_tensor, rng);

        for (i, env) in envs.iter_mut().enumerate() {
            obs_flat.extend_from_slice(&step_obs[i * OBS_DIM..(i + 1) * OBS_DIM]);
            actions_flat.push(acts[i]);
            log_probs_flat.push(lps[i]);
            values_flat.push(vals[i]);

            let result = env.step(acts[i]);
            rewards_flat.push(result.reward);
            let done = result.terminated || result.truncated;
            dones_flat.push(done);
            if done {
                env.reset();
            }
        }
    }

    // Plain discounted Monte-Carlo returns per env, bootstrapping with the
    // last value estimate when the trajectory does not terminate. Layout is
    // step-major (all envs at step 0, then all envs at step 1, ...), so we
    // walk each env's column backwards.
    let mut returns_vec = vec![0.0f32; total];
    let mut advantages_vec = vec![0.0f32; total];
    for env_idx in 0..ROLLOUT_NUM_ENVS {
        let mut running = 0.0f32;
        for step in (0..ROLLOUT_N_STEPS).rev() {
            let idx = step * ROLLOUT_NUM_ENVS + env_idx;
            if dones_flat[idx] {
                running = 0.0;
            }
            running = rewards_flat[idx] + GAMMA * running;
            returns_vec[idx] = running;
            advantages_vec[idx] = running - values_flat[idx];
        }
    }

    Rollout {
        observations: Tensor::<B, 2>::from_data(
            TensorData::new(obs_flat, [total, OBS_DIM]),
            device,
        ),
        actions: Tensor::<B, 1, Int>::from_data(TensorData::new(actions_flat, [total]), device),
        old_log_probs: Tensor::<B, 1>::from_data(TensorData::new(log_probs_flat, [total]), device),
        old_values: Tensor::<B, 1>::from_data(TensorData::new(values_flat, [total]), device),
        advantages: Tensor::<B, 1>::from_data(TensorData::new(advantages_vec, [total]), device),
        returns: Tensor::<B, 1>::from_data(TensorData::new(returns_vec, [total]), device),
    }
}

/// `a2c_train_step` — isolated single A2C gradient step on a synthetic batch.
fn bench_a2c_train_step<B: AutodiffBackend>(c: &mut Criterion, device: &B::Device, suffix: &str) {
    let (obs, actions, _old_lp, _old_v, advantages, returns) = synthetic_batch::<B>(device);

    let mut group = c.benchmark_group(format!("a2c_train_step/{suffix}"));
    group.throughput(Throughput::Elements(SYNTH_BATCH as u64));
    group.bench_function("synthetic_batch", |b| {
        b.iter_batched(
            || make_a2c_trainer::<B>(device),
            |mut trainer| {
                let stats = trainer
                    .train_step(
                        obs.clone(),
                        actions.clone(),
                        advantages.clone(),
                        returns.clone(),
                        |p, o, a| p.evaluate_actions(o, a),
                    )
                    .expect("A2C train_step");
                black_box(stats)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// `ppo_train_step` — isolated single PPO gradient step on the *same*
/// synthetic batch, for a head-to-head per-update comparison.
fn bench_ppo_train_step<B: AutodiffBackend>(c: &mut Criterion, device: &B::Device, suffix: &str) {
    let (obs, actions, old_lp, old_v, advantages, returns) = synthetic_batch::<B>(device);

    let mut group = c.benchmark_group(format!("ppo_train_step/{suffix}"));
    group.throughput(Throughput::Elements(SYNTH_BATCH as u64));
    group.bench_function("synthetic_batch", |b| {
        b.iter_batched(
            || make_ppo_trainer::<B>(device),
            |mut trainer| {
                let stats = trainer
                    .train_step(
                        obs.clone(),
                        actions.clone(),
                        old_lp.clone(),
                        old_v.clone(),
                        advantages.clone(),
                        returns.clone(),
                        |p, o, a| p.evaluate_actions(o, a),
                    )
                    .expect("PPO train_step");
                black_box(stats)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// `a2c_cartpole_steps_per_sec` — full rollout collection + one A2C update,
/// reported in environment-steps/sec via `Throughput::Elements`.
fn bench_a2c_cartpole_steps_per_sec<B: AutodiffBackend>(
    c: &mut Criterion,
    device: &B::Device,
    suffix: &str,
) {
    let total_steps = (ROLLOUT_N_STEPS * ROLLOUT_NUM_ENVS) as u64;

    let mut group = c.benchmark_group(format!("a2c_cartpole_steps_per_sec/{suffix}"));
    group.throughput(Throughput::Elements(total_steps));
    group.bench_function("rollout_plus_update", |b| {
        b.iter_batched(
            || (make_a2c_trainer::<B>(device), StdRng::seed_from_u64(777)),
            |(mut trainer, mut rng)| {
                let rollout = collect_rollout::<B>(trainer.policy(), device, &mut rng);
                let stats = trainer
                    .train_step(
                        rollout.observations,
                        rollout.actions,
                        rollout.advantages,
                        rollout.returns,
                        |p, o, a| p.evaluate_actions(o, a),
                    )
                    .expect("A2C train_step");
                black_box(stats)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// `ppo_cartpole_steps_per_sec` — full-loop PPO comparison mirroring the
/// A2C full-loop benchmark.
fn bench_ppo_cartpole_steps_per_sec<B: AutodiffBackend>(
    c: &mut Criterion,
    device: &B::Device,
    suffix: &str,
) {
    let total_steps = (ROLLOUT_N_STEPS * ROLLOUT_NUM_ENVS) as u64;

    let mut group = c.benchmark_group(format!("ppo_cartpole_steps_per_sec/{suffix}"));
    group.throughput(Throughput::Elements(total_steps));
    group.bench_function("rollout_plus_update", |b| {
        b.iter_batched(
            || (make_ppo_trainer::<B>(device), StdRng::seed_from_u64(777)),
            |(mut trainer, mut rng)| {
                let rollout = collect_rollout::<B>(trainer.policy(), device, &mut rng);
                let stats = trainer
                    .train_step(
                        rollout.observations,
                        rollout.actions,
                        rollout.old_log_probs,
                        rollout.old_values,
                        rollout.advantages,
                        rollout.returns,
                        |p, o, a| p.evaluate_actions(o, a),
                    )
                    .expect("PPO train_step");
                black_box(stats)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// DQN-specific configuration shared by both DQN benchmark groups.
///
/// Small replay buffer / minibatch so each timed iteration completes quickly
/// on CPU while still exercising the full Double-DQN backward pass. Hard
/// target sync (no soft update) so the timed region is just the gradient step.
fn dqn_config() -> DQNConfig {
    DQNConfig::new()
        .learning_rate(1e-3)
        .batch_size(DQN_BATCH)
        .buffer_capacity(DQN_BUFFER_CAPACITY)
        .min_buffer_size(DQN_MIN_BUFFER)
        .target_update_interval(500)
        .gamma(0.99)
        .epsilon_start(1.0)
        .epsilon_end(0.05)
        .epsilon_decay_steps(10_000)
}

/// Build a fresh, seeded Burn DQN trainer on backend `B`.
fn make_dqn_trainer<B: AutodiffBackend>(
    device: &B::Device,
) -> DQNTrainerBurn<B, QNetworkBurn<B>, impl burn::optim::Optimizer<QNetworkBurn<B>, B>> {
    let online = QNetworkBurn::<B>::new(OBS_DIM, ACTION_DIM, HIDDEN_DIM, device);
    let inner_opt = AdamConfig::new().init();
    let config = dqn_config();
    let burn_opt: BurnOptimizer<B, QNetworkBurn<B>, _> =
        BurnOptimizer::new(inner_opt, config.learning_rate);
    DQNTrainerBurn::new(config, online, burn_opt, OBS_DIM, ACTION_DIM as i64, device.clone())
        .expect("valid DQN config")
}

/// A single deterministic synthetic CartPole-shaped transition derived from a
/// step index. Mirrors the trainer's own unit-test fixture so the pushed data
/// is plausible (bounded floats, alternating actions, periodic dones).
fn synthetic_transition(i: usize) -> ([f32; OBS_DIM], i64, f32, [f32; OBS_DIM], bool) {
    let phase = (i as f32) * 0.1;
    let obs = [phase.sin(), phase.cos(), phase * 0.5, phase * -0.3];
    let next_obs = [(phase + 0.1).sin(), (phase + 0.1).cos(), phase * 0.5, phase * -0.3];
    let action = (i % ACTION_DIM) as i64;
    let reward = if action == 0 { 1.0 } else { -1.0 };
    let done = i % 32 == 31;
    (obs, action, reward, next_obs, done)
}

/// Pre-fill a trainer's replay buffer with `n` seeded synthetic transitions
/// so `train_step` performs a real gradient update instead of the warmup
/// `None` no-op.
fn prefill_buffer<B: AutodiffBackend>(
    trainer: &mut DQNTrainerBurn<
        B,
        QNetworkBurn<B>,
        impl burn::optim::Optimizer<QNetworkBurn<B>, B>,
    >,
    n: usize,
) {
    for i in 0..n {
        let (obs, action, reward, next_obs, done) = synthetic_transition(i);
        trainer.buffer_mut().push(&obs, action, reward, &next_obs, done);
    }
}

/// `dqn_train_step` — isolated single Double-DQN gradient step on a replay
/// minibatch. The `iter_batched` setup closure (untimed) builds a fresh
/// seeded trainer and pre-fills its replay buffer past `min_buffer_size`, so
/// every timed `train_step` performs exactly one real gradient update (never
/// `None`). This is the off-policy per-update number, cross-class comparable
/// against `a2c_train_step` / `ppo_train_step`.
fn bench_dqn_train_step<B: AutodiffBackend>(c: &mut Criterion, device: &B::Device, suffix: &str) {
    let mut group = c.benchmark_group(format!("dqn_train_step/{suffix}"));
    group.throughput(Throughput::Elements(DQN_BATCH as u64));
    group.bench_function("replay_minibatch", |b| {
        b.iter_batched(
            || {
                let mut trainer = make_dqn_trainer::<B>(device);
                prefill_buffer::<B>(&mut trainer, DQN_MIN_BUFFER);
                (trainer, StdRng::seed_from_u64(99))
            },
            |(mut trainer, mut rng)| {
                let stats = trainer
                    .train_step(
                        &mut rng,
                        |q: &QNetworkBurn<B>, o: Tensor<B, 2>| q.forward(o),
                        |q: &QNetworkBurn<B>, o: Tensor<B, 2>| q.forward(o),
                    )
                    .expect("DQN train_step")
                    .expect("buffer pre-filled, train_step must update (not None)");
                black_box(stats)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// `dqn_cartpole_steps_per_sec` — full DQN env-loop: step a seeded CartPole
/// env, push the transition, and run one gradient update per env step
/// (mirroring `train_cartpole_dqn`). The buffer is pre-filled to
/// `min_buffer_size` in the untimed setup so the timed region measures
/// steady-state throughput (one update per step), not the warmup transient.
/// Reported in environment-steps/sec; comparable within the off-policy class
/// only (see module header).
fn bench_dqn_cartpole_steps_per_sec<B: AutodiffBackend>(
    c: &mut Criterion,
    device: &B::Device,
    suffix: &str,
) {
    let mut group = c.benchmark_group(format!("dqn_cartpole_steps_per_sec/{suffix}"));
    group.throughput(Throughput::Elements(DQN_LOOP_STEPS as u64));
    group.bench_function("env_step_plus_update", |b| {
        b.iter_batched(
            || {
                let mut trainer = make_dqn_trainer::<B>(device);
                prefill_buffer::<B>(&mut trainer, DQN_MIN_BUFFER);
                let mut env = CartPole::new();
                env.reset();
                (trainer, env, StdRng::seed_from_u64(0xC0FFEE))
            },
            |(mut trainer, mut env, mut rng)| {
                let mut obs = env.get_observation();
                for _ in 0..DQN_LOOP_STEPS {
                    let action = trainer.select_action(
                        &obs,
                        &mut rng,
                        |q: &QNetworkBurn<B>, o_host: &[f32]| {
                            let o_t: Tensor<B, 2> = Tensor::from_data(
                                TensorData::new(o_host.to_vec(), [1, o_host.len()]),
                                device,
                            );
                            let q_host: Vec<f32> =
                                q.forward(o_t).into_data().to_vec().unwrap_or_default();
                            let mut best = 0_i64;
                            let mut best_v = f32::NEG_INFINITY;
                            for (i, &v) in q_host.iter().enumerate() {
                                if v > best_v {
                                    best_v = v;
                                    best = i as i64;
                                }
                            }
                            best
                        },
                    );

                    let result = env.step(action);
                    let next_obs = result.observation.clone();
                    let done = result.terminated || result.truncated;
                    trainer.buffer_mut().push(&obs, action, result.reward, &next_obs, done);
                    obs = next_obs;

                    trainer.increment_env_step();

                    let stats = trainer
                        .train_step(
                            &mut rng,
                            |q: &QNetworkBurn<B>, o: Tensor<B, 2>| q.forward(o),
                            |q: &QNetworkBurn<B>, o: Tensor<B, 2>| q.forward(o),
                        )
                        .expect("DQN train_step")
                        .expect("buffer pre-filled, train_step must update (not None)");
                    black_box(stats);

                    if done {
                        env.reset();
                        obs = env.get_observation();
                    }
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Build a fresh, seeded SAC trainer on backend `B` for the continuous
/// PendulumSwingUp task (obs_dim=3, action_dim=1).
///
/// Replay / minibatch / network sizes are shrunk from the SAC defaults (see
/// the `SAC_*` constants) so each CPU iteration completes quickly while still
/// driving the full actor + twin-critic + alpha backward path.
fn make_sac_trainer<B: AutodiffBackend>(device: &B::Device) -> SacTrainer<B> {
    let config = SacConfig::new()
        .batch_size(SAC_BATCH)
        .buffer_capacity(SAC_BUFFER_CAPACITY)
        .min_buffer_size(SAC_MIN_BUFFER)
        .learning_starts(SAC_LEARNING_STARTS)
        .hidden_dim(SAC_HIDDEN_DIM)
        .num_hidden_layers(SAC_NUM_HIDDEN_LAYERS)
        .seed(0);
    SacTrainer::<B>::new(config, SAC_OBS_DIM, SAC_ACTION_DIM, device.clone())
        .expect("valid SAC config")
}

/// A single deterministic synthetic Pendulum-shaped transition derived from a
/// step index. Mirrors the SAC trainer's own unit-test fixture so the pushed
/// data is plausible: a unit-circle (cosθ, sinθ) plus a bounded angular
/// velocity, a tanh-range action in (-1, 1), a negative quadratic reward, and
/// periodic dones.
fn sac_synthetic_transition(
    i: usize,
) -> ([f32; SAC_OBS_DIM], [f32; SAC_ACTION_DIM], f32, [f32; SAC_OBS_DIM], bool) {
    let phase = (i as f32) * 0.1;
    let obs = [phase.cos(), phase.sin(), phase * 0.2];
    let next_obs = [(phase + 0.1).cos(), (phase + 0.1).sin(), phase * 0.2];
    let action = [phase.sin().clamp(-0.99, 0.99)];
    let reward = -(phase * phase);
    let done = i % 5 == 4;
    (obs, action, reward, next_obs, done)
}

/// Pre-fill a SAC trainer's replay buffer with `n` seeded synthetic Pendulum
/// transitions so `train()` performs a real gradient update instead of the
/// warmup `None` no-op.
fn sac_prefill_buffer<B: AutodiffBackend>(trainer: &mut SacTrainer<B>, n: usize) {
    for i in 0..n {
        let (obs, action, reward, next_obs, done) = sac_synthetic_transition(i);
        trainer.buffer_mut().push(&obs, &action, reward, &next_obs, done);
    }
}

/// Rescale a tanh-squashed actor action in `(-1, 1)` to the Pendulum torque
/// range `[-SAC_MAX_TORQUE, SAC_MAX_TORQUE]` (mirrors the `train_sac` example's
/// `scale_action`).
fn sac_scale_action(action: &[f32]) -> Vec<f32> {
    action.iter().map(|a| a * SAC_MAX_TORQUE).collect()
}

/// `sac_train_step` — isolated single SAC gradient step on a replay minibatch.
/// The `iter_batched` setup closure (untimed) builds a fresh seeded trainer,
/// pre-fills its replay buffer past `min_buffer_size`, and advances
/// `total_env_steps` past `learning_starts`, so every timed `train()` performs
/// exactly one real gradient update (never `None`). This is the off-policy
/// continuous-control per-update number, cross-class comparable against the
/// other `*_train_step` groups (SAC's update is heavier than DQN's — see the
/// module header).
fn bench_sac_train_step<B: AutodiffBackend>(c: &mut Criterion, device: &B::Device, suffix: &str) {
    let mut group = c.benchmark_group(format!("sac_train_step/{suffix}"));
    group.throughput(Throughput::Elements(SAC_BATCH as u64));
    group.bench_function("replay_minibatch", |b| {
        b.iter_batched(
            || {
                let mut trainer = make_sac_trainer::<B>(device);
                sac_prefill_buffer::<B>(&mut trainer, SAC_MIN_BUFFER);
                // Advance past learning_starts so the trainer is in its
                // steady-state regime (not the random-action warmup window).
                for _ in 0..SAC_LEARNING_STARTS {
                    trainer.increment_env_step();
                }
                trainer
            },
            |mut trainer| {
                let stats = trainer
                    .train()
                    .expect("SAC train()")
                    .expect("buffer pre-filled, train() must update (not None)");
                black_box(stats)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// `sac_pendulum_steps_per_sec` — full SAC env-loop on the continuous
/// PendulumSwingUp env: select an action, rescale it by `SAC_MAX_TORQUE`, step
/// the env, push the (unscaled) transition, and run one gradient update per env
/// step (mirroring the `train_sac` example). The buffer is pre-filled to
/// `min_buffer_size` and `total_env_steps` advanced past `learning_starts` in
/// the untimed setup so the timed region measures steady-state throughput (one
/// real update per step), not the warmup transient. Reported in
/// environment-steps/sec; comparable within the off-policy class only and NOT
/// across environments (Pendulum, not CartPole — see module header).
fn bench_sac_pendulum_steps_per_sec<B: AutodiffBackend>(
    c: &mut Criterion,
    device: &B::Device,
    suffix: &str,
) {
    let mut group = c.benchmark_group(format!("sac_pendulum_steps_per_sec/{suffix}"));
    group.throughput(Throughput::Elements(SAC_LOOP_STEPS as u64));
    group.bench_function("env_step_plus_update", |b| {
        b.iter_batched(
            || {
                let mut trainer = make_sac_trainer::<B>(device);
                sac_prefill_buffer::<B>(&mut trainer, SAC_MIN_BUFFER);
                for _ in 0..SAC_LEARNING_STARTS {
                    trainer.increment_env_step();
                }
                let mut env = PendulumSwingUp::with_seed(0);
                env.reset();
                (trainer, env)
            },
            |(mut trainer, mut env)| {
                let mut obs = env.get_observation();
                for _ in 0..SAC_LOOP_STEPS {
                    let action = trainer.select_action(&obs);
                    let result = env.step(sac_scale_action(&action));
                    let done = result.terminated || result.truncated;
                    trainer.buffer_mut().push(
                        &obs,
                        &action,
                        result.reward,
                        &result.observation,
                        done,
                    );
                    trainer.increment_env_step();

                    let stats = trainer
                        .train()
                        .expect("SAC train()")
                        .expect("buffer pre-filled, train() must update (not None)");
                    black_box(stats);

                    if done {
                        env.reset();
                        obs = env.get_observation();
                    } else {
                        obs = result.observation;
                    }
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Nature-DQN CNN benchmarks (large-net crossover measurement, issue #328)
// ---------------------------------------------------------------------------

/// A deterministic synthetic NCHW observation batch `[batch, 4, 84, 84]` for
/// the Nature-DQN CNN benches. Pixels are a smooth, bounded, seed-free pattern
/// in `[0, 1]` so the numbers are comparable run-to-run and across backends;
/// the network is scale-agnostic, so the exact values do not matter.
fn cnn_obs<B: AutodiffBackend>(batch: usize, device: &B::Device) -> Tensor<B, 4> {
    let data: Vec<f32> = (0..batch * 4 * 84 * 84).map(|i| (i as f32 * 0.001).sin().abs()).collect();
    Tensor::<B, 4>::from_data(TensorData::new(data, [batch, 4, 84, 84]), device)
}

/// A deterministic synthetic action index vector `[batch]` in `0..n_actions`,
/// used to drive the actor-critic `evaluate_actions` path in the policy
/// train-step bench.
fn cnn_actions<B: AutodiffBackend>(batch: usize, device: &B::Device) -> Tensor<B, 1, Int> {
    let data: Vec<i64> = (0..batch).map(|i| (i % CNN_N_ACTIONS) as i64).collect();
    Tensor::<B, 1, Int>::from_data(TensorData::new(data, [batch]), device)
}

/// Build a fresh, seeded Nature-DQN actor-critic policy on backend `B`.
fn make_cnn_policy<B: AutodiffBackend>(device: &B::Device) -> NatureDqnBurnPolicy<B> {
    NatureDqnBurnPolicy::<B>::with_config(
        CNN_N_ACTIONS,
        NatureDqnConfig::default().with_seed(42),
        device,
    )
}

/// Build a fresh, seeded Nature-DQN Q-network on backend `B`.
fn make_cnn_qnet<B: AutodiffBackend>(device: &B::Device) -> NatureDqnQNetwork<B> {
    NatureDqnQNetwork::<B>::with_config(
        CNN_N_ACTIONS,
        NatureDqnConfig::default().with_seed(42),
        device,
    )
}

/// `nature_dqn_policy_forward` — isolated **inference** cost (no autograd) of
/// [`NatureDqnBurnPolicy::forward`] on synthetic NCHW batches at B=32 and
/// B=256. The setup closure (untimed) builds the policy and input; the timed
/// closure runs one forward pass over the full Nature-DQN conv stack.
fn bench_nature_dqn_policy_forward<B: AutodiffBackend>(
    c: &mut Criterion,
    device: &B::Device,
    suffix: &str,
) {
    let mut group = c.benchmark_group(format!("nature_dqn_policy_forward/{suffix}"));
    for &batch in &[CNN_BATCH_DQN, CNN_BATCH_PPO] {
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_function(format!("b{batch}"), |b| {
            b.iter_batched(
                || (make_cnn_policy::<B>(device), cnn_obs::<B>(batch, device)),
                |(policy, obs)| {
                    let (logits, values) = policy.forward(obs);
                    black_box((logits, values));
                    // Drain the backend command queue so criterion times the full
                    // GPU work, not just kernel enqueue. No-op on NdArray.
                    let _ = B::sync(device);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// `nature_dqn_qnet_forward` — isolated **inference** cost (no autograd) of
/// [`NatureDqnQNetwork::forward`] on synthetic NCHW batches at B=32 and B=256.
fn bench_nature_dqn_qnet_forward<B: AutodiffBackend>(
    c: &mut Criterion,
    device: &B::Device,
    suffix: &str,
) {
    let mut group = c.benchmark_group(format!("nature_dqn_qnet_forward/{suffix}"));
    for &batch in &[CNN_BATCH_DQN, CNN_BATCH_PPO] {
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_function(format!("b{batch}"), |b| {
            b.iter_batched(
                || (make_cnn_qnet::<B>(device), cnn_obs::<B>(batch, device)),
                |(qnet, obs)| {
                    let q_values = qnet.forward(obs);
                    black_box(q_values);
                    // Drain the backend command queue so criterion times the full
                    // GPU work, not just kernel enqueue. No-op on NdArray.
                    let _ = B::sync(device);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// `nature_dqn_policy_train_step` — **forward + backward + Adam step** for the
/// Nature-DQN actor-critic policy: the per-minibatch training cost the PPO
/// trainer would pay on Atari observations. The setup closure (untimed) builds
/// a fresh seeded policy + Adam optimizer + synthetic obs/action batch; the
/// timed closure runs a synthetic actor-critic loss (policy-gradient + value
/// MSE-to-zero + entropy bonus), a full `backward()`, and one optimizer step.
fn bench_nature_dqn_policy_train_step<B: AutodiffBackend>(
    c: &mut Criterion,
    device: &B::Device,
    suffix: &str,
) {
    let mut group = c.benchmark_group(format!("nature_dqn_policy_train_step/{suffix}"));
    for &batch in &[CNN_BATCH_DQN, CNN_BATCH_PPO] {
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_function(format!("b{batch}"), |b| {
            b.iter_batched(
                || {
                    let policy = make_cnn_policy::<B>(device);
                    let inner_opt = AdamConfig::new().init();
                    let opt: BurnOptimizer<B, NatureDqnBurnPolicy<B>, _> =
                        BurnOptimizer::new(inner_opt, CNN_LR);
                    (policy, opt, cnn_obs::<B>(batch, device), cnn_actions::<B>(batch, device))
                },
                |(policy, mut opt, obs, actions)| {
                    let (log_probs, entropy, values) = policy.evaluate_actions(obs, actions);
                    // Synthetic actor-critic loss: policy-gradient surrogate +
                    // value MSE-to-zero + entropy bonus. Exercises both heads and
                    // the full conv-trunk backward graph.
                    let loss = log_probs.mean().neg() + values.powf_scalar(2.0).mean()
                        - entropy.mean().mul_scalar(CNN_ENTROPY_COEF);
                    let grads = GradientsParams::from_grads(loss.backward(), &policy);
                    let policy = opt.inner_mut().step(CNN_LR, policy, grads);
                    black_box(policy);
                    // Drain the backend command queue so criterion times the full
                    // GPU work, not just kernel enqueue. No-op on NdArray.
                    let _ = B::sync(device);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// `nature_dqn_qnet_train_step` — **forward + backward + Adam step** for the
/// Nature-DQN Q-network (the CNN analogue of the MLP `dqn_train_step` group).
/// Dummy loss `q_values.powf_scalar(2.0).mean()` (MSE against zero) is enough
/// to exercise the full backward graph. B=32 and B=256.
fn bench_nature_dqn_qnet_train_step<B: AutodiffBackend>(
    c: &mut Criterion,
    device: &B::Device,
    suffix: &str,
) {
    let mut group = c.benchmark_group(format!("nature_dqn_qnet_train_step/{suffix}"));
    for &batch in &[CNN_BATCH_DQN, CNN_BATCH_PPO] {
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_function(format!("b{batch}"), |b| {
            b.iter_batched(
                || {
                    let qnet = make_cnn_qnet::<B>(device);
                    let inner_opt = AdamConfig::new().init();
                    let opt: BurnOptimizer<B, NatureDqnQNetwork<B>, _> =
                        BurnOptimizer::new(inner_opt, CNN_LR);
                    (qnet, opt, cnn_obs::<B>(batch, device))
                },
                |(qnet, mut opt, obs)| {
                    let q_values = qnet.forward(obs);
                    let loss = q_values.powf_scalar(2.0).mean();
                    let grads = GradientsParams::from_grads(loss.backward(), &qnet);
                    let qnet = opt.inner_mut().step(CNN_LR, qnet, grads);
                    black_box(qnet);
                    // Drain the backend command queue so criterion times the full
                    // GPU work, not just kernel enqueue. No-op on NdArray.
                    let _ = B::sync(device);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Run the full twelve-group throughput suite for backend `B`, tagging every
/// criterion group with `suffix` (e.g. `"ndarray"`) so distinct backends
/// produce side-by-side, non-colliding groups. This is the single seam through
/// which a backend is registered — the GPU driver calls this once more behind a
/// `#[cfg(feature = "wgpu")]` / `#[cfg(feature = "cuda")]` gate without
/// touching any bench body.
fn register_all<B: AutodiffBackend>(c: &mut Criterion, device: &B::Device, suffix: &str) {
    bench_a2c_train_step::<B>(c, device, suffix);
    bench_ppo_train_step::<B>(c, device, suffix);
    bench_a2c_cartpole_steps_per_sec::<B>(c, device, suffix);
    bench_ppo_cartpole_steps_per_sec::<B>(c, device, suffix);
    bench_dqn_train_step::<B>(c, device, suffix);
    bench_dqn_cartpole_steps_per_sec::<B>(c, device, suffix);
    bench_sac_train_step::<B>(c, device, suffix);
    bench_sac_pendulum_steps_per_sec::<B>(c, device, suffix);
    bench_nature_dqn_policy_forward::<B>(c, device, suffix);
    bench_nature_dqn_qnet_forward::<B>(c, device, suffix);
    bench_nature_dqn_policy_train_step::<B>(c, device, suffix);
    bench_nature_dqn_qnet_train_step::<B>(c, device, suffix);
}

/// Criterion driver. Registers the CPU `ndarray` baseline unconditionally, then
/// registers the GPU backends behind Cargo-feature gates.
///
/// The CPU baseline is always present, so a single run produces the paired
/// CPU-vs-GPU comparison (e.g. `a2c_train_step/ndarray` alongside
/// `a2c_train_step/wgpu`). CI is CPU-host only and never sets `wgpu`/`cuda`, so
/// the default `cargo bench --features training` path compiles and runs only
/// the `ndarray` registration — the GPU branches below are not compiled.
///
/// No runtime adapter detection is performed: criterion has no skip primitive,
/// and a host without a GPU would never enable the feature in the first place.
/// If a GPU feature is compiled where no adapter exists, Burn panics on first
/// device use — acceptable because the feature is opt-in and only compiled on
/// operator GPU hosts (the actual GPU run is operator-gated, see #184).
fn benches(c: &mut Criterion) {
    type Cpu = Autodiff<NdArray<f32>>;
    register_all::<Cpu>(c, &default_burn_device::<Cpu>(), "ndarray");

    #[cfg(feature = "wgpu")]
    {
        // Cross-platform GPU (Vulkan / Metal / DX12 / WebGPU). Mirrors the
        // `Wgpu<f32, i32>` InnerBackend alias used in examples/games/**.
        type Gpu = Autodiff<Wgpu<f32, i32>>;
        register_all::<Gpu>(c, &default_burn_device::<Gpu>(), "wgpu");
    }

    #[cfg(feature = "cuda")]
    {
        // Linux + NVIDIA. `Cuda` defaults to `<f32, i32>` element/index types.
        type Gpu = Autodiff<Cuda<f32, i32>>;
        register_all::<Gpu>(c, &default_burn_device::<Gpu>(), "cuda");
    }

    // Mixed-precision (f16) registrations (issue #272, epic #267). These reuse
    // the exact same generic `register_all` bench bodies with the backend's
    // float element pinned to `f16`, so they sit side-by-side with the `/cuda`
    // (or `/wgpu`) f32 groups in one criterion run — the paired f16-vs-f32
    // comparison. Only compiled when `training-fp16` is enabled, which is opt-in
    // (see the `training-fp16` feature in Cargo.toml). The signal is in the four
    // `nature_dqn_*` CNN groups; the small-MLP groups register too (harmless)
    // and serve as a sanity check where no f16 win is expected.
    //
    // CUDA (NVIDIA Ampere+) is the runtime-verified f16 path — tensor cores make
    // this the intended target. See #270 for the verified training run.
    #[cfg(all(feature = "cuda", feature = "training-fp16"))]
    {
        type Fp16Cuda = Autodiff<Cuda<burn::tensor::f16, i32>>;
        register_all::<Fp16Cuda>(c, &default_burn_device::<Fp16Cuda>(), "cuda-f16");
    }

    // wgpu/Metal f16 compiles but is NOT runtime-verified: Metal has no bf16
    // matmul in Burn 0.21 and f16 on Metal is untested (see #305). Gated behind
    // `not(feature = "cuda")` so a combined `cuda,wgpu,training-fp16` build
    // measures the verified CUDA f16 path rather than registering both.
    #[cfg(all(feature = "wgpu", not(feature = "cuda"), feature = "training-fp16"))]
    {
        type Fp16Wgpu = Autodiff<Wgpu<burn::tensor::f16, i32>>;
        register_all::<Fp16Wgpu>(c, &default_burn_device::<Fp16Wgpu>(), "wgpu-f16");
    }
}

criterion_group!(benches_group, benches);
criterion_main!(benches_group);
