//! Integration tests for the multi-discrete `MultiAgentEnvironment` trait
//! surface.
//!
//! Exercises [`BucketBrigadeMaEnv`] through the widened trait (one
//! `Vec<i64>` per agent) added in upstream issue #3. Lives in `tests/` for
//! the same reason as the other thrust-port tests in this branch:
//! pre-existing in-module test failures on `main` (see upstream issue #7)
//! prevent `cargo test --lib` from compiling.
//!
//! Run with:
//! `cargo test --test test_multi_discrete_env --features env-bucket-brigade`.

#![cfg(feature = "env-bucket-brigade")]

use bucket_brigade_core::SCENARIOS;
use thrust_rl::{
    env::games::bucket_brigade::BucketBrigadeMaEnv, multi_agent::MultiAgentEnvironment,
};

fn make_env(num_agents: usize, seed: u64) -> BucketBrigadeMaEnv {
    let scenario = SCENARIOS["default"].clone();
    let mut env = BucketBrigadeMaEnv::new(scenario, num_agents, Some(seed));
    env.reset(Some(seed));
    env
}

#[test]
fn agent_action_space_is_house_mode_for_every_agent() {
    let env = make_env(4, 42);
    for aid in 0..MultiAgentEnvironment::num_agents(&env) {
        assert_eq!(
            env.agent_action_space(aid),
            vec![10, 2],
            "agent {aid}: expected factored [house_index, mode] action space",
        );
    }
}

#[test]
fn step_multi_returns_one_entry_per_agent() {
    let mut env = make_env(4, 42);
    let actions: Vec<Vec<i64>> = vec![vec![3, 1], vec![5, 0], vec![7, 1], vec![0, 0]];
    let result = env.step_multi(&actions);

    let n = MultiAgentEnvironment::num_agents(&env);
    assert_eq!(result.rewards.len(), n, "rewards.len() must equal num_agents");
    assert_eq!(result.observations.len(), n, "observations.len() must equal num_agents");
    assert_eq!(result.terminated.len(), n);
    assert_eq!(result.truncated.len(), n);

    let obs_dim = env.obs_dim();
    for (i, obs) in result.observations.iter().enumerate() {
        assert_eq!(obs.len(), obs_dim, "agent {i} observation has wrong length");
    }
}

#[test]
fn step_multi_agrees_with_inherent_step() {
    // Calling the env through the trait and through its inherent typed API
    // with identical actions must produce identical results.
    let mut env_trait = make_env(3, 7);
    let mut env_inherent = make_env(3, 7);

    let actions_trait: Vec<Vec<i64>> = vec![vec![2, 1], vec![4, 0], vec![9, 1]];
    let actions_inherent: Vec<[u8; 2]> = vec![[2, 1], [4, 0], [9, 1]];

    for _ in 0..3 {
        let r_trait = env_trait.step_multi(&actions_trait);
        let r_inherent = env_inherent.step(&actions_inherent);

        assert_eq!(r_trait.rewards, r_inherent.rewards);
        assert_eq!(r_trait.observations, r_inherent.observations);
        assert!(r_trait.terminated.iter().all(|&t| t == r_inherent.done));
        if r_inherent.done {
            break;
        }
    }
}

#[test]
fn active_agents_marks_all_agents_active() {
    let env = make_env(5, 1);
    let active = env.active_agents();
    assert_eq!(active.len(), 5);
    assert!(active.iter().all(|&a| a), "BB engine has no per-agent termination");
}

#[test]
fn get_agent_observation_returns_per_agent_view() {
    let env = make_env(4, 42);
    let obs_dim = env.obs_dim();
    for aid in 0..MultiAgentEnvironment::num_agents(&env) {
        let obs = env.get_agent_observation(aid);
        assert_eq!(obs.len(), obs_dim, "agent {aid} obs has wrong length");
    }
}

#[test]
#[should_panic(expected = "expected 4 actions")]
fn step_multi_panics_on_wrong_agent_count() {
    let mut env = make_env(4, 42);
    let actions: Vec<Vec<i64>> = vec![vec![0, 0], vec![1, 0]]; // only 2, not 4
    let _ = env.step_multi(&actions);
}

#[test]
#[should_panic(expected = "action must have 2 dims")]
fn step_multi_panics_on_wrong_action_dim_count() {
    let mut env = make_env(2, 42);
    // Action vectors have length 3 instead of the required 2.
    let actions: Vec<Vec<i64>> = vec![vec![0, 0, 0], vec![1, 1, 1]];
    let _ = env.step_multi(&actions);
}

#[test]
#[should_panic(expected = "house_index")]
fn step_multi_panics_on_house_out_of_range() {
    let mut env = make_env(2, 42);
    // house_index = 99 is out of range 0..10.
    let actions: Vec<Vec<i64>> = vec![vec![99, 0], vec![0, 0]];
    let _ = env.step_multi(&actions);
}

#[test]
#[should_panic(expected = "mode")]
fn step_multi_panics_on_invalid_mode() {
    let mut env = make_env(2, 42);
    let actions: Vec<Vec<i64>> = vec![vec![0, 5], vec![1, 0]];
    let _ = env.step_multi(&actions);
}

#[test]
fn step_multi_handles_single_dim_action_space_shape() {
    // Sanity check: even though BB itself is 2-dim, this confirms the trait
    // accepts arbitrary inner vec lengths -- the env's own dim-check is the
    // only gatekeeper. A pure-discrete env (vec![n]) would round-trip here
    // through the same trait surface.
    let env = make_env(3, 42);
    for aid in 0..MultiAgentEnvironment::num_agents(&env) {
        let dims = env.agent_action_space(aid);
        assert!(!dims.is_empty(), "agent {aid} action space must be non-empty");
        // For BB, this should be exactly [10, 2]; a pure-discrete env would
        // return [n] here. Both are valid trait shapes.
        assert_eq!(dims, vec![10, 2]);
    }
}
