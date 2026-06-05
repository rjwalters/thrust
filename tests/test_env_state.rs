//! Smoke tests for the `Environment::clone_state` /
//! `Environment::restore_state` pair added in issue #62.
//!
//! These exercise the snapshot/restore contract from the perspective of a
//! caller that would build a real MCTS-style rollout-and-return loop on top of
//! the trait.

use thrust_rl::env::{CartPole, Environment};

/// MCTS-like rollout: snapshot, take a few steps, restore, replay the first
/// step from the snapshot, and verify it matches the first step of the
/// rollout. CartPole is deterministic per step, so observation, reward,
/// terminated, and truncated must all match exactly.
#[test]
fn cartpole_mcts_like_rollout() {
    let mut env = CartPole::new();
    env.reset();

    // Drive the env into a deterministic, non-initial state by stepping past
    // the random `reset_state` perturbation a few times.
    for _ in 0..3 {
        env.step(1);
    }

    let snap = env.clone_state();

    // Rollout: 3 deterministic actions from the snapshot.
    let actions = [1_i64, 0, 1];
    let mut results = Vec::with_capacity(actions.len());
    for &a in &actions {
        results.push(env.step(a));
    }

    // Restore and verify a single step from the original snapshot produces
    // the same first result as the rollout.
    env.restore_state(&snap);
    let same_first = env.step(actions[0]);

    assert_eq!(same_first.observation, results[0].observation);
    assert_eq!(same_first.reward, results[0].reward);
    assert_eq!(same_first.terminated, results[0].terminated);
    assert_eq!(same_first.truncated, results[0].truncated);
}

/// Snapshot must allow multiple independent rollouts from the same base state
/// (the canonical MCTS use case). Each rollout must produce identical results
/// when given the same action sequence.
#[test]
fn cartpole_repeated_rollouts_from_snapshot() {
    let mut env = CartPole::new();
    env.reset();
    for _ in 0..2 {
        env.step(0);
    }
    let snap = env.clone_state();

    let actions = [1_i64, 1, 0, 0, 1];

    // Rollout #1
    let mut run1 = Vec::with_capacity(actions.len());
    for &a in &actions {
        run1.push(env.step(a));
    }

    // Restore and roll out a second time with the same actions.
    env.restore_state(&snap);
    let mut run2 = Vec::with_capacity(actions.len());
    for &a in &actions {
        run2.push(env.step(a));
    }

    assert_eq!(run1.len(), run2.len());
    for (i, (a, b)) in run1.iter().zip(run2.iter()).enumerate() {
        assert_eq!(a.observation, b.observation, "obs mismatch at step {}", i);
        assert_eq!(a.reward, b.reward, "reward mismatch at step {}", i);
        assert_eq!(a.terminated, b.terminated, "terminated mismatch at step {}", i);
        assert_eq!(a.truncated, b.truncated, "truncated mismatch at step {}", i);
    }
}
