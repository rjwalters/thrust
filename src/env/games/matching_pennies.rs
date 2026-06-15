//! Matching Pennies — 2-agent zero-sum smoke env.
//!
//! Canonical mixed-strategy testbed for multi-agent RL: agent 0 (the
//! "matcher") wins +1 when both actions match; agent 1 (the
//! "mismatcher") wins +1 when they differ. The unique Nash equilibrium
//! is the uniform distribution `(0.5, 0.5)` for both agents, which makes
//! this env the standard smoke test for any mixed-equilibrium learner
//! (PSRO, NFSP, fictitious play).
//!
//! # Why a fresh env instead of reusing `SimpleBandit`?
//!
//! `SimpleBandit` is a single-agent contextual bandit with `Environment`
//! semantics. The PSRO trainer ingests a [`JointEnv`] (see
//! [`crate::multi_agent::joint`]) where every agent receives the same
//! observation and a synchronized step result with per-agent rewards.
//! Matching pennies is the smallest 2-agent zero-sum env that fits the
//! [`JointEnv`] surface, so we implement it directly here without
//! touching the heavier `Environment` / `MultiAgentEnvironment` traits.
//!
//! # Observation
//!
//! Observation is a constant scalar `[0.0]` (single-dim). Both agents
//! see the same observation on every step — this matches the
//! "globally-shared observation" constraint inherited from
//! [`crate::multi_agent::joint::JointMultiAgentTrainer`]. Matching
//! pennies is a *stateless* simultaneous-move game; the observation is
//! present only to satisfy the trainer's interface, not because the
//! agents need state.
//!
//! # Action / reward
//!
//! - Action: `0` or `1`, single discrete dim per agent.
//! - Reward: `+1` to the matcher (agent 0) and `-1` to the mismatcher (agent 1)
//!   when actions match, and the inverse when they differ. The game is zero-sum
//!   every step.
//!
//! # Episode length
//!
//! Each `reset_joint` starts a fresh episode of length
//! [`MatchingPennies::EPISODE_LEN`] steps; `step_joint` sets
//! `done = true` on the final step. The trainer wraps this with its own
//! rollout-length loop, so the episode length only affects how
//! frequently the `done` flag fires; the optimal mixed strategy is the
//! same regardless of episode length.

use crate::multi_agent::joint::{JointEnv, JointStepResult};

/// 2-agent zero-sum matching-pennies env.
///
/// Agent 0 ("matcher"): reward `+1` when both actions agree, `-1` otherwise.
/// Agent 1 ("mismatcher"): the negation.
#[derive(Debug, Clone)]
pub struct MatchingPennies {
    /// Number of steps elapsed in the current episode.
    step: usize,
}

impl MatchingPennies {
    /// Episode length used by `step_joint` to fire the `done` flag.
    pub const EPISODE_LEN: usize = 16;

    /// Number of agents in this env (always 2).
    pub const NUM_AGENTS: usize = 2;

    /// Observation dimensionality (always 1 — a constant scalar).
    pub const OBS_DIM: usize = 1;

    /// Per-agent action cardinality (always 2 — heads/tails).
    pub const ACTION_DIM: usize = 2;

    /// Construct a fresh env.
    pub fn new() -> Self {
        Self { step: 0 }
    }
}

impl Default for MatchingPennies {
    fn default() -> Self {
        Self::new()
    }
}

impl JointEnv for MatchingPennies {
    fn reset_joint(&mut self, _seed: Option<u64>) -> Vec<Vec<f32>> {
        self.step = 0;
        // Two agents, each gets the same single-dim constant observation.
        vec![vec![0.0; Self::OBS_DIM]; Self::NUM_AGENTS]
    }

    fn step_joint(&mut self, actions: &[Vec<i64>]) -> JointStepResult {
        debug_assert_eq!(actions.len(), Self::NUM_AGENTS, "matching pennies requires 2 agents");
        debug_assert_eq!(actions[0].len(), 1, "agent 0 must supply a 1-d action");
        debug_assert_eq!(actions[1].len(), 1, "agent 1 must supply a 1-d action");

        // Matcher wins (+1) when actions agree; mismatcher wins (+1) when
        // they disagree. Always zero-sum: rewards sum to zero every step.
        let a0 = actions[0][0];
        let a1 = actions[1][0];
        let matcher_reward = if a0 == a1 { 1.0_f32 } else { -1.0_f32 };
        let rewards = vec![matcher_reward, -matcher_reward];

        self.step += 1;
        let done = self.step >= Self::EPISODE_LEN;

        JointStepResult {
            rewards,
            done,
            observations: vec![vec![0.0; Self::OBS_DIM]; Self::NUM_AGENTS],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_sum_every_step() {
        // Reward symmetry: matcher_reward + mismatcher_reward == 0 for
        // every combination of (a0, a1).
        for a0 in 0..2 {
            for a1 in 0..2 {
                let mut env = MatchingPennies::new();
                env.reset_joint(None);
                let res = env.step_joint(&[vec![a0], vec![a1]]);
                let total: f32 = res.rewards.iter().sum();
                assert!(
                    total.abs() < 1e-6,
                    "rewards not zero-sum for ({a0}, {a1}): {:?}",
                    res.rewards
                );
            }
        }
    }

    #[test]
    fn test_matcher_wins_on_agreement() {
        for a in 0..2 {
            let mut env = MatchingPennies::new();
            env.reset_joint(None);
            let res = env.step_joint(&[vec![a], vec![a]]);
            assert_eq!(res.rewards[0], 1.0, "matcher should win on agreement (action {a})");
            assert_eq!(res.rewards[1], -1.0, "mismatcher should lose on agreement (action {a})");
        }
    }

    #[test]
    fn test_mismatcher_wins_on_disagreement() {
        let pairs = [(0_i64, 1_i64), (1, 0)];
        for (a0, a1) in pairs {
            let mut env = MatchingPennies::new();
            env.reset_joint(None);
            let res = env.step_joint(&[vec![a0], vec![a1]]);
            assert_eq!(res.rewards[0], -1.0, "matcher should lose on ({a0}, {a1})");
            assert_eq!(res.rewards[1], 1.0, "mismatcher should win on ({a0}, {a1})");
        }
    }

    #[test]
    fn test_episode_done_after_episode_len() {
        let mut env = MatchingPennies::new();
        env.reset_joint(None);
        for step in 0..MatchingPennies::EPISODE_LEN {
            let res = env.step_joint(&[vec![0], vec![0]]);
            let expected_done = step + 1 >= MatchingPennies::EPISODE_LEN;
            assert_eq!(res.done, expected_done, "done flag wrong at step {step}");
        }
    }

    #[test]
    fn test_reset_restores_step_counter() {
        let mut env = MatchingPennies::new();
        env.reset_joint(None);
        for _ in 0..MatchingPennies::EPISODE_LEN {
            env.step_joint(&[vec![0], vec![0]]);
        }
        // Reset and verify we get EPISODE_LEN more steps before done.
        env.reset_joint(None);
        for step in 0..MatchingPennies::EPISODE_LEN - 1 {
            let res = env.step_joint(&[vec![0], vec![0]]);
            assert!(!res.done, "should not be done at step {step} after reset");
        }
        let last = env.step_joint(&[vec![0], vec![0]]);
        assert!(last.done, "should be done on final step after reset");
    }

    #[test]
    fn test_observation_shape() {
        let mut env = MatchingPennies::new();
        let obs = env.reset_joint(None);
        assert_eq!(obs.len(), MatchingPennies::NUM_AGENTS);
        for o in &obs {
            assert_eq!(o.len(), MatchingPennies::OBS_DIM);
        }
        let res = env.step_joint(&[vec![0], vec![0]]);
        assert_eq!(res.observations.len(), MatchingPennies::NUM_AGENTS);
        for o in &res.observations {
            assert_eq!(o.len(), MatchingPennies::OBS_DIM);
        }
    }
}
