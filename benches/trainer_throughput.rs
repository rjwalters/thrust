//! Throughput / wall-clock benchmarks for the on-policy trainers.
//!
//! Measures steps/sec and per-update wall-clock for the A2C trainer
//! ([`A2cTrainer`]) and, for an apples-to-apples comparison, the PPO
//! trainer ([`PPOTrainerBurn`]). All benchmarks run on the pure-Rust
//! `Autodiff<NdArray<f32>>` (CPU) backend and are seeded so the numbers
//! are comparable run-to-run.
//!
//! Benchmark groups:
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
//!
//! Run quickly with, e.g.:
//! ```text
//! cargo bench --features training --bench trainer_throughput -- \
//!     --warm-up-time 1 --measurement-time 5
//! ```

use std::hint::black_box;

use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    optim::AdamConfig,
    tensor::{Int, Tensor, TensorData},
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};
use thrust_rl::{
    env::{Environment, games::cartpole::CartPole},
    policy::mlp::MlpBurnPolicy,
    train::{A2cConfig, A2cTrainer, BurnOptimizer, PPOConfig, PPOTrainerBurn},
};

/// CPU autodiff backend used for every benchmark.
type B = Autodiff<NdArray<f32>>;

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

/// Build a fresh, seeded A2C trainer on the CPU backend.
fn make_a2c_trainer()
-> A2cTrainer<B, MlpBurnPolicy<B>, impl burn::optim::Optimizer<MlpBurnPolicy<B>, B>> {
    let device = Default::default();
    let policy = MlpBurnPolicy::<B>::new(OBS_DIM, ACTION_DIM, HIDDEN_DIM, &device);
    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 7e-4);
    let config = A2cConfig::default();
    A2cTrainer::new(config, policy, burn_opt).expect("valid A2C config")
}

/// Build a fresh, seeded PPO trainer on the CPU backend.
///
/// `n_epochs = 1` keeps the PPO per-update cost a like-for-like single
/// gradient pass against A2C's single update; the surrogate clip / value
/// clip remain to reflect PPO's real per-step overhead.
fn make_ppo_trainer()
-> PPOTrainerBurn<B, MlpBurnPolicy<B>, impl burn::optim::Optimizer<MlpBurnPolicy<B>, B>> {
    let device = Default::default();
    let policy = MlpBurnPolicy::<B>::new(OBS_DIM, ACTION_DIM, HIDDEN_DIM, &device);
    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 3e-4);
    let config = PPOConfig::new().n_epochs(1);
    PPOTrainerBurn::new(config, policy, burn_opt).expect("valid PPO config")
}

/// A deterministic synthetic training batch shared by the isolated
/// `train_step` benchmarks. Returns
/// `(observations, actions, old_log_probs, old_values, advantages, returns)`.
fn synthetic_batch(
    device: &NdArrayDevice,
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
struct Rollout {
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
fn collect_rollout(policy: &MlpBurnPolicy<B>, device: &NdArrayDevice, rng: &mut StdRng) -> Rollout {
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
fn bench_a2c_train_step(c: &mut Criterion) {
    let device = Default::default();
    let (obs, actions, _old_lp, _old_v, advantages, returns) = synthetic_batch(&device);

    let mut group = c.benchmark_group("a2c_train_step");
    group.throughput(Throughput::Elements(SYNTH_BATCH as u64));
    group.bench_function("synthetic_batch", |b| {
        b.iter_batched(
            make_a2c_trainer,
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
fn bench_ppo_train_step(c: &mut Criterion) {
    let device = Default::default();
    let (obs, actions, old_lp, old_v, advantages, returns) = synthetic_batch(&device);

    let mut group = c.benchmark_group("ppo_train_step");
    group.throughput(Throughput::Elements(SYNTH_BATCH as u64));
    group.bench_function("synthetic_batch", |b| {
        b.iter_batched(
            make_ppo_trainer,
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
fn bench_a2c_cartpole_steps_per_sec(c: &mut Criterion) {
    let device = NdArrayDevice::default();
    let total_steps = (ROLLOUT_N_STEPS * ROLLOUT_NUM_ENVS) as u64;

    let mut group = c.benchmark_group("a2c_cartpole_steps_per_sec");
    group.throughput(Throughput::Elements(total_steps));
    group.bench_function("rollout_plus_update", |b| {
        b.iter_batched(
            || (make_a2c_trainer(), StdRng::seed_from_u64(777)),
            |(mut trainer, mut rng)| {
                let rollout = collect_rollout(trainer.policy(), &device, &mut rng);
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
fn bench_ppo_cartpole_steps_per_sec(c: &mut Criterion) {
    let device = NdArrayDevice::default();
    let total_steps = (ROLLOUT_N_STEPS * ROLLOUT_NUM_ENVS) as u64;

    let mut group = c.benchmark_group("ppo_cartpole_steps_per_sec");
    group.throughput(Throughput::Elements(total_steps));
    group.bench_function("rollout_plus_update", |b| {
        b.iter_batched(
            || (make_ppo_trainer(), StdRng::seed_from_u64(777)),
            |(mut trainer, mut rng)| {
                let rollout = collect_rollout(trainer.policy(), &device, &mut rng);
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

criterion_group!(
    benches,
    bench_a2c_train_step,
    bench_ppo_train_step,
    bench_a2c_cartpole_steps_per_sec,
    bench_ppo_cartpole_steps_per_sec,
);
criterion_main!(benches);
