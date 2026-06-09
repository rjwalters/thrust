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
use thrust_rl::env::games::bucket_brigade::{
    ACTION_DIMS, BucketBrigadeMaEnv, NUM_HOUSES, SCENARIO_INFO_LEN,
};

/// 10 (houses) + 4 (signals) + 4 (locations) + 12 (last_actions: 3*4)
///   + 4 (round1_signals) + 12 (scenario_info) = 46
fn expected_obs_dim_default_4_agents() -> usize {
    NUM_HOUSES + 4 + 4 + ACTION_DIMS * 4 + 4 + SCENARIO_INFO_LEN
}

#[test]
fn env_constructs_with_correct_obs_dim() {
    let scenario = SCENARIOS["default"].clone();
    let env = BucketBrigadeMaEnv::new(scenario, 4, Some(42));
    assert_eq!(env.obs_dim(), expected_obs_dim_default_4_agents());
    assert_eq!(env.action_dims(), vec![NUM_HOUSES as i64, 2, 2]);
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
    // [house, mode, signal] per agent
    let actions = vec![[0u8, 0, 0], [1, 1, 1], [2, 0, 1], [3, 1, 0]];
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
    let actions = vec![[0u8, 0, 0]; 4];
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
    let rest_actions = vec![[0u8, 0, 0]; 4];
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
        for (i, &h) in o[..NUM_HOUSES].iter().enumerate() {
            assert!(h == 0.0 || h == 1.0, "house {i} = {h} after reset (must be SAFE or BURNING)",);
        }
    }
}

#[test]
fn from_scenario_id_loads_minimal_specialization_v1() {
    // Acceptance criterion: env loads scenarios via the versioned
    // registry (no scenario duplication).
    let mut env = BucketBrigadeMaEnv::from_scenario_id(
        "minimal_specialization-v1",
        None, // pick up the frozen default num_agents
        Some(42),
    )
    .expect("registered scenario ID resolves");
    assert_eq!(env.num_agents(), 4);
    let obs = env.reset(Some(42));
    assert_eq!(obs.len(), 4);
    for o in &obs {
        assert_eq!(o.len(), env.obs_dim());
    }
}

#[test]
fn from_scenario_id_overrides_num_agents() {
    let env = BucketBrigadeMaEnv::from_scenario_id("minimal_specialization-v1", Some(3), Some(42))
        .expect("registered scenario ID resolves");
    assert_eq!(env.num_agents(), 3);
}

#[test]
fn from_scenario_id_unknown_returns_error() {
    let result = BucketBrigadeMaEnv::from_scenario_id("totally_made_up-v1", None, None);
    match result {
        Ok(_) => panic!("expected unknown scenario ID to error"),
        Err(err) => assert!(err.contains("totally_made_up-v1"), "unexpected error: {err}"),
    }
}

#[test]
fn smoke_step_through_episode_with_versioned_scenario() {
    // Acceptance criterion (smoke): a versioned-ID-constructed env runs an
    // entire episode end to end without panicking. This is the "Thrust PPO
    // trainer could plug in here" surface — we don't pull in the PPO
    // trainer itself (it's gated behind the `training` feature), but we
    // exercise every API a trainer would touch.
    let mut env =
        BucketBrigadeMaEnv::from_scenario_id("minimal_specialization-v1", None, Some(7)).unwrap();
    let num_agents = env.num_agents();
    env.reset(Some(7));

    // Tight loop: all REST, no signal, agents stay at house 0. Episode
    // ends when min_nights is reached and all houses are either all safe
    // or all ruined; cap at 1000 to make the test bounded.
    let actions = vec![[0u8, 0, 0]; num_agents];
    let mut done = false;
    for _ in 0..1000 {
        let r = env.step(&actions);
        assert_eq!(r.rewards.len(), num_agents);
        assert_eq!(r.observations.len(), num_agents);
        if r.done {
            done = true;
            break;
        }
    }
    assert!(done, "episode did not terminate within 1000 steps");
}
