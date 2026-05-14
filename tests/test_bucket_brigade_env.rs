//! Tests for the Bucket Brigade thrust env adapter.
//!
//! Gated on the `env-bucket-brigade` feature. Run with:
//! `cargo test --test test_bucket_brigade_env --features env-bucket-brigade`.
//!
//! Lives in `tests/` (separate binary) rather than as in-module
//! `#[cfg(test)]` for the same reason as the other thrust-port tests in
//! this branch: pre-existing in-module test failures on `main` (see
//! upstream issue #7) prevent `cargo test --lib` from compiling.

#![cfg(feature = "env-bucket-brigade")]

use bucket_brigade_core::SCENARIOS;
use thrust_rl::env::games::bucket_brigade::BucketBrigadeMaEnv;

#[test]
fn env_constructs_with_correct_obs_dim() {
    let scenario = SCENARIOS["default"].clone();
    let env = BucketBrigadeMaEnv::new(scenario, 4, Some(42));
    // 10 (houses) + 4 (signals) + 4 (locations) + 8 (last_actions) + 12 (scenario) = 38
    assert_eq!(env.obs_dim(), 38);
    assert_eq!(env.action_dims(), [10, 2]);
}

#[test]
fn reset_returns_one_obs_per_agent() {
    let scenario = SCENARIOS["default"].clone();
    let mut env = BucketBrigadeMaEnv::new(scenario, 4, Some(42));
    let obs = env.reset(Some(42));
    assert_eq!(obs.len(), 4);
    let expected_dim = env.obs_dim();
    for o in &obs {
        assert_eq!(o.len(), expected_dim);
    }
}

#[test]
fn step_returns_one_reward_and_obs_per_agent() {
    let scenario = SCENARIOS["default"].clone();
    let mut env = BucketBrigadeMaEnv::new(scenario, 4, Some(42));
    env.reset(Some(42));
    let actions = vec![[0u8, 0], [1, 1], [2, 0], [3, 1]];
    let result = env.step(&actions);
    assert_eq!(result.rewards.len(), 4);
    assert_eq!(result.observations.len(), 4);
    let expected_dim = env.obs_dim();
    for o in &result.observations {
        assert_eq!(o.len(), expected_dim);
    }
}

#[test]
fn deterministic_with_seed() {
    let scenario = SCENARIOS["default"].clone();
    let mut env_a = BucketBrigadeMaEnv::new(scenario.clone(), 4, Some(99));
    let mut env_b = BucketBrigadeMaEnv::new(scenario, 4, Some(99));
    let actions = vec![[0u8, 0]; 4];
    for _ in 0..5 {
        let a = env_a.step(&actions);
        let b = env_b.step(&actions);
        assert_eq!(a.rewards, b.rewards);
        assert_eq!(a.done, b.done);
        if a.done {
            break;
        }
    }
}

#[test]
fn reset_recovers_after_done() {
    // Regression test for the bug we found in the *Python* env (see
    // rjwalters/bucket-brigade#132): an env that has reached `done` must
    // surface a usable state after `reset()`. The Rust core's `reset` already
    // re-zeroes the houses array, so this should just work, but we check
    // it explicitly because our trainer relies on the behavior.
    let scenario = SCENARIOS["default"].clone();
    let mut env = BucketBrigadeMaEnv::new(scenario, 4, Some(123));
    let rest_actions = vec![[0u8, 0]; 4];
    // Step the env many times --- the engine will eventually mark done.
    let mut steps = 0;
    loop {
        let r = env.step(&rest_actions);
        steps += 1;
        if r.done || steps > 200 {
            break;
        }
    }
    // After reset(), get_observation should return a fresh layout.
    let obs = env.reset(None);
    assert_eq!(obs.len(), 4);
    // The first observation field block is the houses array; after reset it
    // can be all-zero or contain initial-fire BURNING (=1) entries, but
    // never RUINED (=2).
    for o in &obs {
        for (i, &h) in o[..10].iter().enumerate() {
            assert!(
                h == 0.0 || h == 1.0,
                "house {i} = {h} after reset (must be SAFE or BURNING)",
            );
        }
    }
}
