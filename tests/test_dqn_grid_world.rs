//! DQN smoke + convergence tests on the
//! [`GridWorld`](thrust_rl::env::games::grid_world::GridWorld) sparse-reward
//! navigation env.
//!
//! This is PR B of the more-environments epic (#180, issue #185), the
//! discrete-navigation counterpart to the existing DQN CartPole stack. Two
//! tiers, mirroring `tests/test_a2c_cartpole.rs` /
//! `tests/test_sac_mountain_car.rs`:
//!
//! 1. **`dqn_grid_world_training_step_runs`** (always runs) — a fast,
//!    unit-level check that the DQN env loop wires together end-to-end on
//!    GridWorld: a few hundred real env steps push transitions, at least one
//!    gradient update fires, and every reported loss / Q stat / ε is finite.
//!    This is the CI default — it executes in well under a second and never
//!    asserts a convergence bar.
//!
//! 2. **`dqn_grid_world_reaches_reward_bar`** (`#[ignore]`) — the convergence
//!    bar: seeded DQN solves the default `4x4` layout, reaching **mean eval
//!    return >= +0.90 over the final 20 greedy evaluation episodes** within a
//!    fixed env-step budget. The optimal shortest path earns `+1 - 0.01 *
//!    path_len` (≈ `+0.94` for the 6-step Manhattan path), so `+0.90` cleanly
//!    separates "found a reliable path to the goal" from "wandered / fell in
//!    holes / timed out" (negative floor: a hole is `-1.0`, a timeout is `-0.01
//!    * 100 = -1.0`, and a random policy on this deceptive layout averages well
//!      below zero).
//!
//! ## Why the heavy bar is `#[ignore]`d
//!
//! The convergence run trains for 80k env steps with one gradient update per
//! step on the CPU `NdArray` backend (~35 s wall-clock in `--release`,
//! measured) — too slow to gate every `cargo test` run. It is kept opt-in:
//!
//! ```text
//! cargo test --release --features training --test test_dqn_grid_world -- --ignored
//! ```
//!
//! The fast step test above guarantees the trainer keeps working on every CI
//! run; the convergence test is the periodic / release-gate check.

#![cfg(feature = "training")]

use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    optim::AdamConfig,
    tensor::{Tensor, TensorData},
};
use rand::{SeedableRng, rngs::StdRng};
use thrust_rl::{
    env::{Environment, SpaceType, games::grid_world::GridWorld},
    policy::q_network::QNetworkBurn,
    train::{
        dqn::{DQNConfig, DQNTrainerBurn},
        optimizer::BurnOptimizer,
    },
};

type B = Autodiff<NdArray<f32>>;

const HIDDEN_DIM: usize = 64;

/// Greedy argmax over the Q-network's action values for a single observation.
fn greedy_action(q: &QNetworkBurn<B>, obs: &[f32], device: &NdArrayDevice) -> i64 {
    let o_t: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(obs.to_vec(), [1, obs.len()]), device);
    let q_host: Vec<f32> = q.forward(o_t).into_data().to_vec().unwrap_or_default();
    let mut best = 0_i64;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in q_host.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as i64;
        }
    }
    best
}

/// Build a seeded DQN trainer for GridWorld.
#[allow(clippy::type_complexity)]
fn build(
    config: DQNConfig,
    seed: u64,
) -> (
    DQNTrainerBurn<B, QNetworkBurn<B>, impl burn::optim::Optimizer<QNetworkBurn<B>, B>>,
    NdArrayDevice,
    usize,
) {
    let device: NdArrayDevice = Default::default();

    let probe = GridWorld::new();
    let obs_dim = probe.observation_space().shape[0];
    let n_actions = match probe.action_space().space_type {
        SpaceType::Discrete(n) => n as i64,
        _ => panic!("expected discrete action space"),
    };

    let lr = config.learning_rate;
    let online =
        QNetworkBurn::<B>::with_seed(obs_dim, n_actions as usize, HIDDEN_DIM, seed, &device);
    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<B, QNetworkBurn<B>, _> = BurnOptimizer::new(inner_opt, lr);

    let trainer = DQNTrainerBurn::new(config, online, burn_opt, obs_dim, n_actions, device)
        .expect("trainer constructs");
    (trainer, device, obs_dim)
}

/// Fast, always-on smoke test: drive a few hundred real GridWorld steps
/// through the DQN trainer and assert a finite gradient step occurs. Keeps the
/// trainer wired up on every CI run without a convergence bar.
#[test]
fn dqn_grid_world_training_step_runs() {
    let config = DQNConfig::new()
        .learning_rate(1e-3)
        .batch_size(32)
        .buffer_capacity(2_000)
        .min_buffer_size(64)
        .target_update_interval(100)
        .gamma(0.99)
        .epsilon_start(1.0)
        .epsilon_end(0.05)
        .epsilon_decay_steps(500)
        .max_grad_norm(10.0)
        .soft_update_tau(0.01);
    let (mut trainer, device, _obs_dim) = build(config, 0);

    let mut env = GridWorld::new();
    env.reset();
    let mut obs = env.get_observation();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);

    let mut performed_update = false;
    let mut last_stats = None;
    let mut valid_actions = true;

    for _ in 0..400 {
        let dev = device;
        let action = trainer.select_action(&obs, &mut rng, |q: &QNetworkBurn<B>, o: &[f32]| {
            greedy_action(q, o, &dev)
        });
        if !(0..4).contains(&action) {
            valid_actions = false;
        }

        let result = env.step(action);
        let done = result.terminated || result.truncated;
        trainer
            .buffer_mut()
            .push(&obs, action, result.reward, &result.observation, done);

        trainer.increment_env_step();
        let _ = trainer.maybe_sync_target(|online, _target, _tau| online.clone());

        if let Some(stats) = trainer
            .train_step(
                &mut rng,
                |q: &QNetworkBurn<B>, o: Tensor<B, 2>| q.forward(o),
                |q: &QNetworkBurn<B>, o: Tensor<B, 2>| q.forward(o),
            )
            .expect("train step is finite")
        {
            performed_update = true;
            last_stats = Some(stats);
        }

        if done {
            trainer.increment_episodes(1);
            env.reset();
            obs = env.get_observation();
        } else {
            obs = result.observation;
        }
    }

    assert!(valid_actions, "every selected action must be a valid move in 0..4");
    assert!(performed_update, "at least one gradient update should have fired");
    let stats = last_stats.unwrap();
    assert!(stats.td_loss.is_finite(), "td loss finite");
    assert!(stats.mean_q.is_finite(), "mean_q finite");
    assert!(stats.epsilon.is_finite(), "epsilon finite");
    assert!(trainer.total_train_steps() > 0, "trainer recorded gradient updates");
}

/// Heavy convergence test (opt-in via `--ignored`): seeded DQN solves the
/// default GridWorld layout, reaching mean greedy eval return >= +0.90 over
/// the final 20 evaluation episodes within a fixed env-step budget.
///
/// Run with:
/// ```text
/// cargo test --release --features training --test test_dqn_grid_world -- --ignored
/// ```
#[test]
#[ignore = "convergence run (tens of thousands of env steps); opt in with --ignored (prefer --release)"]
fn dqn_grid_world_reaches_reward_bar() {
    /// Total env-step budget. Empirically ample for seeded DQN to learn the
    /// optimal shortest path to the goal on the deterministic 4x4 layout: the
    /// greedy policy converges to the 6-step path 0→4→8→9→13→14→15 and every
    /// eval episode returns +0.94 (well above the +0.90 bar). ~35 s release.
    const TOTAL_TIMESTEPS: usize = 80_000;
    /// Solved bar: the optimal shortest path earns ~+0.94 (+1 minus a handful
    /// of -0.01 step penalties); +0.90 cleanly separates "reaches the goal
    /// reliably" from any hole/timeout outcome (negative floor).
    const REWARD_BAR: f32 = 0.90;

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
    let (mut trainer, device, _obs_dim) = build(config, 0);

    // ----- Training loop -----
    let mut env = GridWorld::new();
    env.reset();
    let mut obs = env.get_observation();
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);

    while trainer.total_env_steps() < TOTAL_TIMESTEPS {
        let dev = device;
        let action = trainer.select_action(&obs, &mut rng, |q: &QNetworkBurn<B>, o: &[f32]| {
            greedy_action(q, o, &dev)
        });

        let result = env.step(action);
        let done = result.terminated || result.truncated;
        trainer
            .buffer_mut()
            .push(&obs, action, result.reward, &result.observation, done);

        trainer.increment_env_step();
        let _ = trainer.maybe_sync_target(|online, _target, _tau| online.clone());

        if done {
            trainer.increment_episodes(1);
            env.reset();
            obs = env.get_observation();
        } else {
            obs = result.observation;
        }

        trainer
            .train_step(
                &mut rng,
                |q: &QNetworkBurn<B>, o: Tensor<B, 2>| q.forward(o),
                |q: &QNetworkBurn<B>, o: Tensor<B, 2>| q.forward(o),
            )
            .expect("train step is finite");
    }

    // ----- Evaluation: 20 greedy episodes on the deterministic layout -----
    let n_eval = 20;
    let mut returns = Vec::with_capacity(n_eval);
    for _ in 0..n_eval {
        let mut eval_env = GridWorld::new();
        eval_env.reset();
        let mut eval_obs = eval_env.get_observation();
        let mut ep_return = 0.0_f32;
        loop {
            let action = greedy_action(trainer.online(), &eval_obs, &device);
            let result = eval_env.step(action);
            ep_return += result.reward;
            if result.terminated || result.truncated {
                break;
            }
            eval_obs = result.observation;
        }
        returns.push(ep_return);
    }

    let mean_return: f32 = returns.iter().sum::<f32>() / returns.len() as f32;
    assert!(
        mean_return >= REWARD_BAR,
        "DQN mean greedy eval return over {n_eval} episodes was {mean_return:.3}, expected >= {REWARD_BAR:.3}; \
         per-episode returns: {returns:?}"
    );
}
