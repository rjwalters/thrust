//! A2C convergence tests on the [`CartPole`] discrete-control env.
//!
//! This is PR C of the A2C epic (#150, issue #153). Two tiers, mirroring
//! the SAC Pendulum test convention (`tests/test_sac_pendulum.rs`):
//!
//! 1. **`a2c_cartpole_training_step_runs`** (always runs) — a fast, unit-level
//!    check that the A2C env loop wires together end-to-end: a handful of
//!    `EnvPool` rollouts push real CartPole transitions, at least one gradient
//!    update fires, and every reported loss / entropy stat is finite. This is
//!    the CI default — it executes in well under a second and asserts no
//!    convergence bar.
//!
//! 2. **`a2c_cartpole_reaches_reward_bar`** (`#[ignore]`) — the architect's
//!    convergence bar: seeded A2C solves CartPole, reaching **mean episode
//!    length >= 195 over the final 100 episodes** within a fixed env-step
//!    budget. CartPole reward is +1/step (max 500), so episode length ==
//!    episode reward; a random policy sits near ~22 steps, so 195 cleanly
//!    separates "learned" from "did not learn".
//!
//! ## Why the heavy bar is `#[ignore]`d
//!
//! The convergence run trains for hundreds of thousands of env steps on
//! the CPU `NdArray` backend — multi-minute wall-clock, too slow to gate
//! every `cargo test` run. It is kept opt-in:
//!
//! ```text
//! cargo test --release --features training --test test_a2c_cartpole -- --ignored
//! ```
//!
//! The fast step test above guarantees the trainer keeps working on every
//! CI run; the convergence test is the periodic / release-gate check.

#![cfg(feature = "training")]

use burn::{
    backend::{Autodiff, NdArray},
    optim::AdamConfig,
    tensor::{Int, Tensor, TensorData},
};
use thrust_rl::{
    env::{Environment, cartpole::CartPole, pool::EnvPool},
    policy::mlp::{BurnActivation, MlpBurnConfig, MlpBurnPolicy},
    train::{
        a2c::{A2cConfig, A2cTrainer},
        optimizer::BurnOptimizer,
    },
};

type B = Autodiff<NdArray<f32>>;

const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 1.0;

/// Build a seeded A2C trainer + EnvPool for CartPole.
#[allow(clippy::type_complexity)]
fn build(
    num_envs: usize,
    n_steps: usize,
    seed: u64,
) -> (
    A2cTrainer<B, MlpBurnPolicy<B>, impl burn::optim::Optimizer<MlpBurnPolicy<B>, B>>,
    EnvPool<CartPole>,
    usize,
    usize,
) {
    let device = Default::default();

    let probe = CartPole::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n,
        _ => panic!("expected discrete action space"),
    };

    let policy_config = MlpBurnConfig {
        num_layers: 2,
        hidden_dim: 128,
        use_orthogonal_init: true,
        activation: BurnActivation::ReLU,
        seed: Some(seed),
    };
    let policy = MlpBurnPolicy::<B>::with_config(obs_dim, action_dim, policy_config, &device);

    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 7e-4);

    let config = A2cConfig::new()
        .learning_rate(7e-4)
        .gamma(GAMMA as f64)
        .gae_lambda(GAE_LAMBDA as f64)
        .n_steps(n_steps)
        .num_envs(num_envs)
        .seed(seed);

    let trainer = A2cTrainer::new(config, policy, burn_opt).expect("trainer constructs");
    let env_pool = EnvPool::new(CartPole::new, num_envs);
    (trainer, env_pool, obs_dim, num_envs)
}

/// Host-side per-env GAE (step-major `[T * N]` layout). Mirrors the
/// example's `compute_gae`.
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

/// Run `num_updates` A2C updates against the CartPole `EnvPool`, returning
/// the completed episode lengths (in completion order) and the stats from
/// the final update. CartPole reward is +1/step so length == reward.
fn train(
    trainer: &mut A2cTrainer<B, MlpBurnPolicy<B>, impl burn::optim::Optimizer<MlpBurnPolicy<B>, B>>,
    env_pool: &mut EnvPool<CartPole>,
    obs_dim: usize,
    num_envs: usize,
    n_steps: usize,
    num_updates: usize,
) -> (Vec<u32>, thrust_rl::train::a2c::A2cStats) {
    let device = Default::default();
    let cap = n_steps * num_envs;
    let mut buf_obs: Vec<f32> = Vec::with_capacity(cap * obs_dim);
    let mut buf_actions: Vec<i64> = Vec::with_capacity(cap);
    let mut buf_values: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_rewards: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_dones: Vec<f32> = Vec::with_capacity(cap);

    let mut observations = env_pool.reset();
    let mut episode_lengths = vec![0u32; num_envs];
    let mut completed: Vec<u32> = Vec::new();
    let mut last_stats = thrust_rl::train::a2c::A2cStats::default();

    for _update in 0..num_updates {
        buf_obs.clear();
        buf_actions.clear();
        buf_values.clear();
        buf_rewards.clear();
        buf_dones.clear();

        for _step in 0..n_steps {
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_t: Tensor<B, 2> =
                Tensor::from_data(TensorData::new(obs_flat, [num_envs, obs_dim]), &device);
            let (actions, _log_probs, values) = trainer.policy().get_action_host(obs_t);
            let results = env_pool.step(&actions);

            for env_id in 0..num_envs {
                buf_obs.extend_from_slice(&observations[env_id]);
                buf_actions.push(actions[env_id]);
                buf_values.push(values[env_id]);
                buf_rewards.push(results[env_id].reward);
                let done = results[env_id].terminated || results[env_id].truncated;
                buf_dones.push(if done { 1.0 } else { 0.0 });
                episode_lengths[env_id] += 1;
                observations[env_id] = results[env_id].observation.clone();
                if done {
                    completed.push(episode_lengths[env_id]);
                    trainer.increment_episodes(1);
                    episode_lengths[env_id] = 0;
                    observations[env_id] = env_pool.reset_env(env_id).expect("reset env");
                }
            }
        }

        let last_obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let last_obs_t: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(last_obs_flat, [num_envs, obs_dim]), &device);
        let (_, _, last_values) = trainer.policy().get_action_host(last_obs_t);

        let (adv_host, ret_host) = compute_gae(
            &buf_rewards,
            &buf_values,
            &buf_dones,
            &last_values,
            GAMMA,
            GAE_LAMBDA,
            n_steps,
            num_envs,
        );

        let batch = n_steps * num_envs;
        let obs_b: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(buf_obs.clone(), [batch, obs_dim]), &device);
        let actions_b: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(buf_actions.clone(), [batch]), &device);
        let adv_b: Tensor<B, 1> = Tensor::from_data(TensorData::new(adv_host, [batch]), &device);
        let ret_b: Tensor<B, 1> = Tensor::from_data(TensorData::new(ret_host, [batch]), &device);

        last_stats = trainer
            .train_step(obs_b, actions_b, adv_b, ret_b, |p, o, a| p.evaluate_actions(o, a))
            .expect("train step is finite");
    }

    (completed, last_stats)
}

/// Fast, always-on smoke test: drive a few real CartPole rollouts through
/// the A2C trainer and assert finite gradient steps occur. Keeps the
/// trainer wired up on every CI run without a convergence bar.
#[test]
fn a2c_cartpole_training_step_runs() {
    let num_envs = 8;
    let n_steps = 5;
    let (mut trainer, mut env_pool, obs_dim, num_envs) = build(num_envs, n_steps, 0);

    let (_completed, stats) = train(&mut trainer, &mut env_pool, obs_dim, num_envs, n_steps, 50);

    assert!(trainer.total_steps() >= 50, "expected at least 50 gradient updates");
    assert!(stats.policy_loss.is_finite(), "policy loss finite");
    assert!(stats.value_loss.is_finite(), "value loss finite");
    assert!(stats.entropy.is_finite(), "entropy finite");
    assert!(stats.total_loss.is_finite(), "total loss finite");
}

/// Heavy convergence test (opt-in via `--ignored`): seeded A2C solves
/// CartPole, reaching mean episode length >= 195 over the final 100
/// episodes within a fixed env-step budget.
///
/// Run with:
/// ```text
/// cargo test --release --features training --test test_a2c_cartpole -- --ignored
/// ```
#[test]
#[ignore = "multi-minute convergence run; opt in with --ignored (prefer --release)"]
fn a2c_cartpole_reaches_reward_bar() {
    let num_envs = 16;
    let n_steps = 5;
    let total_timesteps = 600_000;
    let num_updates = total_timesteps / (n_steps * num_envs);

    let (mut trainer, mut env_pool, obs_dim, num_envs) = build(num_envs, n_steps, 0);
    let (completed, _stats) =
        train(&mut trainer, &mut env_pool, obs_dim, num_envs, n_steps, num_updates);

    assert!(
        completed.len() >= 100,
        "expected at least 100 completed episodes, got {}",
        completed.len()
    );

    let n = completed.len();
    let recent = &completed[n - 100..];
    let mean_len: f32 = recent.iter().map(|&x| x as f32).sum::<f32>() / 100.0;

    assert!(
        mean_len >= 195.0,
        "A2C mean episode length over final 100 episodes was {mean_len:.1}, expected >= 195.0 \
         (random baseline ~22)"
    );
}
