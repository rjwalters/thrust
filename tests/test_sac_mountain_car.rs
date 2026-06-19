//! SAC smoke + convergence tests on the
//! [`MountainCarContinuous`](thrust_rl::env::games::mountain_car_continuous::MountainCarContinuous)
//! continuous-control env.
//!
//! This is PR B of the SAC env-pool epic (#163, issue #168), the
//! deceptive-reward counterpart to `tests/test_sac_pendulum.rs`. Two tiers:
//!
//! 1. **`sac_mountain_car_training_step_runs`** (always runs) — a fast,
//!    unit-level check that the SAC env loop wires together end-to-end on
//!    MountainCarContinuous: a few hundred real env steps push transitions, at
//!    least one gradient update fires, and every reported loss / alpha / Q stat
//!    is finite. This is the CI default — it executes in well under a second
//!    and never asserts a convergence bar.
//!
//! 2. **`sac_mountain_car_reaches_reward_bar`** (`#[ignore]`) — the convergence
//!    bar: seeded SAC on `MountainCarContinuous` reaches **mean eval return >=
//!    +90 over the final 20 evaluation episodes** after the documented env-step
//!    budget. MountainCarContinuous-v0's published "solved" threshold is +90; a
//!    policy that never reaches the goal scores roughly `-(control cost)`
//!    (slightly negative to near-zero), so +90 cleanly separates "learned to
//!    summit" from "never summits". The bar is comfortably above the best
//!    non-summiting return.
//!
//! ## Action scaling
//!
//! Unlike Pendulum (which scales the actor action by `MAX_TORQUE = 2.0`),
//! MountainCarContinuous's force range is already `[-1, 1]` and the SAC actor
//! emits tanh-squashed actions in `(-1, 1)`, so **no rescaling is needed** —
//! the actor action is passed straight to `env.step(action)`.
//!
//! ## Exploration: why a momentum-pumping warmup
//!
//! MountainCarContinuous has the canonical *deceptive / sparse* reward: the
//! only positive signal (+100) is at the goal, and every other step just pays
//! a small control-effort penalty. SAC's default warmup draws actions
//! uniformly per-step; that force averages to ~0 and **never builds enough
//! momentum to summit** (verified empirically — 0 goal-reaches in 10k uniform
//! warmup steps). With no goal transition in the replay buffer, SAC has no
//! gradient toward the flag and collapses to near-zero force (mean eval return
//! ≈ -0.02, i.e. it never summits).
//!
//! The fix lives entirely in the *training loop* (no `src/` changes): during
//! the warmup window we drive an **energy-pumping** exploration policy that
//! pushes at (near-)full force in the direction of the current velocity — the
//! textbook strategy that solves MountainCar by rocking the car up the slope.
//! This seeds the buffer with hundreds of goal-reaching episodes, after which
//! SAC converges cleanly to ~ +94. We therefore set `learning_starts(0)` (we
//! supply our own warmup actions) and hand-roll the warmup via
//! [`pump_action`].
//!
//! ## Why the heavy bar is `#[ignore]`d
//!
//! A multi-tens-of-thousands env-step run with a 256-wide actor + twin
//! critics + targets on the CPU `NdArray` backend is multi-minute
//! wall-clock — too slow to gate every `cargo test` run. It is kept as an
//! opt-in test runnable with
//! `cargo test --release --features training --test test_sac_mountain_car
//! -- --ignored sac_mountain_car_reaches_reward_bar`. The fast step test
//! above guarantees the trainer keeps working on every CI run; the
//! convergence test is the periodic / release-gate check.

#![cfg(feature = "training")]

use burn::backend::{Autodiff, NdArray};
use rand::{Rng, SeedableRng, rngs::StdRng};
use thrust_rl::{
    env::{Environment, games::mountain_car_continuous::MountainCarContinuous},
    train::sac::{SacConfig, SacTrainer},
};

type B = Autodiff<NdArray<f32>>;

/// MountainCarContinuous observation / action dimensions.
const OBS_DIM: usize = 2;
const ACTION_DIM: usize = 1;

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

/// Fast, always-on smoke test: drive a few hundred real MountainCarContinuous
/// steps through the SAC trainer and assert a finite gradient step occurs.
#[test]
fn sac_mountain_car_training_step_runs() {
    let device = Default::default();
    let config = SacConfig::new()
        .buffer_capacity(4_000)
        .min_buffer_size(100)
        .learning_starts(100)
        .batch_size(64)
        .hidden_dim(32)
        .seed(0);
    let mut trainer =
        SacTrainer::<B>::new(config, OBS_DIM, ACTION_DIM, device).expect("trainer constructs");

    let mut env = MountainCarContinuous::with_seed(0);
    env.reset();
    let mut obs = env.get_observation();

    let mut performed_update = false;
    let mut last_stats = None;
    for _ in 0..400 {
        // No action rescaling: the env's force range is already [-1, 1].
        let action = trainer.select_action(&obs);
        let result = env.step(action.clone());
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

/// Heavy convergence test (opt-in via `--ignored`): seeded SAC reaches mean
/// eval return >= +90 over the final 20 eval episodes after the documented
/// env-step budget.
///
/// Run with:
/// ```text
/// cargo test --release --features training --test test_sac_mountain_car \
///     -- --ignored sac_mountain_car_reaches_reward_bar
/// ```
#[test]
#[ignore = "multi-minute convergence run; opt in with --ignored (prefer --release)"]
fn sac_mountain_car_reaches_reward_bar() {
    /// Total env-step budget. With the energy-pumping warmup seeding the
    /// buffer, 30k steps is ample to converge to ~ +94 (empirically measured
    /// at ~5 min release on the CPU NdArray backend).
    const TOTAL_TIMESTEPS: usize = 30_000;
    /// Energy-pumping warmup window. We supply our own exploration actions
    /// (see [`pump_action`]) during this window, hence `learning_starts(0)`.
    const WARMUP_STEPS: usize = 2_000;
    /// Solved bar: MountainCarContinuous-v0's published threshold.
    const REWARD_BAR: f32 = 90.0;

    let device = Default::default();
    let config = SacConfig::new()
        .buffer_capacity(100_000)
        .min_buffer_size(1_000)
        // We drive the warmup ourselves with pump_action, so disable the
        // trainer's built-in uniform-random warmup.
        .learning_starts(0)
        .batch_size(256)
        .hidden_dim(256)
        .num_hidden_layers(2)
        .seed(0);
    let mut trainer =
        SacTrainer::<B>::new(config, OBS_DIM, ACTION_DIM, device).expect("trainer constructs");

    // ----- Training loop -----
    let mut env = MountainCarContinuous::with_seed(0);
    env.reset();
    let mut obs = env.get_observation();
    let mut explore_rng = StdRng::seed_from_u64(12_345);
    for step in 0..TOTAL_TIMESTEPS {
        // No action rescaling: the env's force range is already [-1, 1].
        // During warmup use the momentum-pumping explorer to seed the buffer
        // with goal-reaching transitions; afterward follow the SAC policy.
        let action = if step < WARMUP_STEPS {
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
        let mut eval_env = MountainCarContinuous::with_seed(1_000 + ep as u64);
        eval_env.reset();
        let mut eval_obs = eval_env.get_observation();
        let mut ep_return = 0.0_f32;
        loop {
            let action = trainer.eval_action(&eval_obs);
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
        "SAC mean eval return over {n_eval} episodes was {mean_return:.1}, expected >= {REWARD_BAR:.1}; \
         per-episode returns: {returns:?}"
    );
}
