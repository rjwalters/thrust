//! SAC smoke tests on the [`PendulumSwingUp`] continuous-control env.
//!
//! This is item 6 of the SAC decomposition (#136, PR E). Two tiers:
//!
//! 1. **`sac_pendulum_training_step_runs`** (always runs) — a fast, unit-level
//!    check that the SAC env loop wires together end-to-end: a few hundred env
//!    steps push real Pendulum transitions, at least one gradient update fires,
//!    and every reported loss / alpha / Q stat is finite. This is the CI
//!    default — it executes in well under a second and never asserts a
//!    convergence bar.
//!
//! 2. **`sac_pendulum_reaches_reward_bar`** (`#[ignore]`) — the architect's
//!    convergence bar: seeded SAC on `PendulumSwingUp` reaches **mean episode
//!    reward >= -200 over the final 20 evaluation episodes after 30k env
//!    steps** (`buffer_capacity` overridden to 50k for the test). Pendulum's
//!    near-optimal return is roughly -150 and a random policy sits around
//!    -1200..-1600, so -200 cleanly separates "learned" from "did not learn".
//!
//! ## Why the heavy bar is `#[ignore]`d
//!
//! 30k env steps with 256-wide actor + twin critics + targets on the CPU
//! `NdArray` backend is multi-minute wall-clock — too slow to gate every
//! `cargo test` run. It is kept as an opt-in test runnable with
//! `cargo test --features training --release -- --ignored
//! sac_pendulum_reaches_reward_bar`. The fast step test above guarantees
//! the trainer keeps working on every CI run; the convergence test is the
//! periodic / release-gate check.

#![cfg(feature = "training")]

use burn::backend::{Autodiff, NdArray};
use thrust_rl::{
    env::{Environment, games::pendulum::PendulumSwingUp},
    train::sac::{SacConfig, SacTrainer},
};

type B = Autodiff<NdArray<f32>>;

/// Torque range the Pendulum env clamps to; the actor emits actions in
/// `(-1, 1)` so we rescale by this before stepping.
const MAX_TORQUE: f32 = 2.0;

/// Rescale a tanh-squashed actor action in `(-1, 1)` to the env's torque
/// range `[-MAX_TORQUE, MAX_TORQUE]`.
fn scale_action(action: &[f32]) -> Vec<f32> {
    action.iter().map(|a| a * MAX_TORQUE).collect()
}

/// Fast, always-on smoke test: drive a few hundred real Pendulum steps
/// through the SAC trainer and assert a finite gradient step occurs.
#[test]
fn sac_pendulum_training_step_runs() {
    let device = Default::default();
    let config = SacConfig::new()
        .buffer_capacity(4_000)
        .min_buffer_size(100)
        .learning_starts(100)
        .batch_size(64)
        .hidden_dim(32)
        .seed(0);
    let mut trainer = SacTrainer::<B>::new(config, 3, 1, device).expect("trainer constructs");

    let mut env = PendulumSwingUp::with_seed(0);
    env.reset();
    let mut obs = env.get_observation();

    let mut performed_update = false;
    let mut last_stats = None;
    for _ in 0..400 {
        let action = trainer.select_action(&obs);
        let result = env.step(scale_action(&action));
        let done = result.terminated || result.truncated;
        trainer
            .buffer_mut()
            .push(&obs, &action, result.reward, &result.observation, done);
        trainer.increment_env_step();

        if let Some(stats) = trainer.train().expect("train step is finite") {
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

    assert!(performed_update, "at least one gradient update should have fired");
    let stats = last_stats.unwrap();
    assert!(stats.critic_loss.is_finite(), "critic loss finite");
    assert!(stats.actor_loss.is_finite(), "actor loss finite");
    assert!(stats.alpha_loss.is_finite(), "alpha loss finite");
    assert!(stats.alpha.is_finite() && stats.alpha > 0.0, "alpha positive finite");
    assert!(stats.mean_q.is_finite(), "mean_q finite");
    assert!(trainer.total_train_steps() > 0);
}

/// Heavy convergence test (opt-in via `--ignored`): seeded SAC reaches
/// mean reward >= -200 over the final 20 eval episodes after 30k env
/// steps.
///
/// Run with:
/// ```text
/// cargo test --features training --release -- --ignored sac_pendulum_reaches_reward_bar
/// ```
#[test]
#[ignore = "multi-minute convergence run; opt in with --ignored (prefer --release)"]
fn sac_pendulum_reaches_reward_bar() {
    let device = Default::default();
    let config = SacConfig::new()
        .buffer_capacity(50_000)
        .min_buffer_size(1_000)
        .learning_starts(1_000)
        .batch_size(256)
        .hidden_dim(256)
        .num_hidden_layers(2)
        .seed(0);
    let mut trainer = SacTrainer::<B>::new(config, 3, 1, device).expect("trainer constructs");

    // ----- Training loop: 30k env steps -----
    let mut env = PendulumSwingUp::with_seed(0);
    env.reset();
    let mut obs = env.get_observation();
    for _ in 0..30_000 {
        let action = trainer.select_action(&obs);
        let result = env.step(scale_action(&action));
        let done = result.terminated || result.truncated;
        trainer
            .buffer_mut()
            .push(&obs, &action, result.reward, &result.observation, done);
        trainer.increment_env_step();
        trainer.train().expect("train step is finite");
        if done {
            trainer.increment_episodes(1);
            env.reset();
            obs = env.get_observation();
        } else {
            obs = result.observation;
        }
    }

    // ----- Evaluation: 20 greedy episodes on fresh seeds -----
    let n_eval = 20;
    let mut returns = Vec::with_capacity(n_eval);
    for ep in 0..n_eval {
        let mut eval_env = PendulumSwingUp::with_seed(1_000 + ep as u64);
        eval_env.reset();
        let mut eval_obs = eval_env.get_observation();
        let mut ep_return = 0.0_f32;
        loop {
            let action = trainer.eval_action(&eval_obs);
            let result = eval_env.step(scale_action(&action));
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
        mean_return >= -200.0,
        "SAC mean eval return over {n_eval} episodes was {mean_return:.1}, expected >= -200.0; \
         per-episode returns: {returns:?}"
    );
}
