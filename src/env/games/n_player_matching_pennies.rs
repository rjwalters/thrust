//! N-player matching pennies ("majority game") — smoke env.
//!
//! N ≥ 2 "majority game" generalization: at each step every agent picks 0 or 1;
//! agent `i`'s reward is:
//!
//! - `+1` if agent `i`'s action matches the strict majority of the *other*
//!   agents' actions,
//! - `-1` if it matches against the strict majority,
//! - `0` if the "others" split evenly (only possible when `N − 1` is even, i.e.
//!   N is odd).
//!
//! # Relationship to 2-player matching pennies
//!
//! This is the *symmetric* majority game: every agent wants to match
//! the majority of the others. For N = 2 it collapses to a pure
//! *coordination* game (both agents want to agree → both get +1 on
//! match, both −1 on mismatch) — the symmetric counterpart of the
//! asymmetric matcher-vs-mismatcher 2-player
//! [`crate::env::games::matching_pennies::MatchingPennies`] env. The
//! action-marginal Nash equilibrium is `(0.5, 0.5)` in both cases (the
//! symmetric mixed Nash on the coordination game; the unique mixed
//! Nash on the asymmetric zero-sum game), so both envs serve as
//! mixed-equilibrium smoke tests for PSRO/NFSP.
//!
//! For N = 4 the unique symmetric mixed equilibrium is `(0.5, 0.5)`
//! per agent. The 0-on-tie rule preserves the `(0.5, 0.5)` symmetric
//! equilibrium for all odd N ≥ 3 as well.
//!
//! # Per-agent observation
//!
//! Observation is agent-index dependent: `obs_i = [i as f32 / (num_agents − 1)
//! as f32]` (length 1). This is the minimum-LOC encoding that exercises the
//! per-agent-observation plumbing from PR #122 — agent 0 sees `[0.0]`,
//! the last agent sees `[1.0]`, and intermediate agents see linearly
//! interpolated values. The observation is *stateless* across the
//! episode; it carries only the agent's identity.
//!
//! # Action / reward
//!
//! - Action: `0` or `1`, single discrete dim per agent.
//! - Reward: per-agent majority-of-others rule above.
//!
//! # Episode length
//!
//! Each `reset_joint` starts a fresh episode of length
//! [`NPlayerMatchingPennies::EPISODE_LEN`] steps; `step_joint` sets
//! `done = true` on the final step. Matches the 2-agent
//! [`crate::env::games::matching_pennies::MatchingPennies`] convention.

use crate::multi_agent::joint::{JointEnv, JointStepResult};

/// N-player matching-pennies "majority game" env (N ≥ 2).
#[derive(Debug, Clone)]
pub struct NPlayerMatchingPennies {
    num_agents: usize,
    /// Number of steps elapsed in the current episode.
    step: usize,
}

impl NPlayerMatchingPennies {
    /// Episode length used by `step_joint` to fire the `done` flag.
    pub const EPISODE_LEN: usize = 16;

    /// Observation dimensionality (always 1 — a normalized agent index).
    pub const OBS_DIM: usize = 1;

    /// Per-agent action cardinality (always 2 — heads/tails).
    pub const ACTION_DIM: usize = 2;

    /// Construct a fresh N-player env. Requires `num_agents >= 2`.
    pub fn new(num_agents: usize) -> Self {
        debug_assert!(num_agents >= 2, "NPlayerMatchingPennies requires num_agents >= 2");
        Self { num_agents, step: 0 }
    }

    /// Per-agent observation: `[i / (num_agents − 1)]` (length 1).
    fn per_agent_obs(&self) -> Vec<Vec<f32>> {
        let denom = (self.num_agents.saturating_sub(1)).max(1) as f32;
        (0..self.num_agents).map(|i| vec![(i as f32) / denom]).collect()
    }

    /// Number of agents this env was constructed for.
    pub fn num_agents(&self) -> usize {
        self.num_agents
    }
}

impl JointEnv for NPlayerMatchingPennies {
    fn reset_joint(&mut self, _seed: Option<u64>) -> Vec<Vec<f32>> {
        self.step = 0;
        self.per_agent_obs()
    }

    fn step_joint(&mut self, actions: &[Vec<i64>]) -> JointStepResult {
        debug_assert_eq!(
            actions.len(),
            self.num_agents,
            "n-player matching pennies requires {} agents, got {}",
            self.num_agents,
            actions.len()
        );
        for (i, a) in actions.iter().enumerate() {
            debug_assert_eq!(a.len(), 1, "agent {i} must supply a 1-d action");
            debug_assert!(a[0] == 0 || a[0] == 1, "agent {i} action must be 0 or 1, got {}", a[0]);
        }

        // For each agent `i`: compute the count of action-1 picks among
        // the *other* agents (size N − 1). The majority is action 1 if
        // count > (N − 1) / 2, action 0 if count < (N − 1) / 2, tie
        // otherwise. The reward is:
        //   matches strict majority → +1
        //   against strict majority → −1
        //   tied                    →  0
        let n = self.num_agents;
        let mut rewards = vec![0.0_f32; n];
        // Pre-compute total ones to avoid recomputing for every agent.
        let total_ones: i64 = actions.iter().map(|a| a[0]).sum();
        for i in 0..n {
            let others_ones = total_ones - actions[i][0];
            let n_others = (n - 1) as i64;
            let n_zeros = n_others - others_ones;
            let majority: Option<i64> = match others_ones.cmp(&n_zeros) {
                std::cmp::Ordering::Greater => Some(1),
                std::cmp::Ordering::Less => Some(0),
                std::cmp::Ordering::Equal => None,
            };
            rewards[i] = match majority {
                Some(m) if m == actions[i][0] => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            };
        }

        self.step += 1;
        let done = self.step >= Self::EPISODE_LEN;

        JointStepResult { rewards, done, observations: self.per_agent_obs() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For N = 2 the "majority of others" is the single other agent's
    /// action. Both agents get `+1` on agreement and `−1` on
    /// disagreement: a symmetric coordination game (NOT the asymmetric
    /// 2-player matcher-vs-mismatcher game in
    /// [`crate::env::games::matching_pennies::MatchingPennies`]).
    /// The mixed Nash on this coordination game is still `(0.5, 0.5)`,
    /// which is the property PSRO/NFSP relies on as the smoke target.
    #[test]
    fn test_n2_coordination_reward_table() {
        for a0 in 0..2_i64 {
            for a1 in 0..2_i64 {
                let mut env = NPlayerMatchingPennies::new(2);
                env.reset_joint(None);
                let res = env.step_joint(&[vec![a0], vec![a1]]);
                let expected = if a0 == a1 { 1.0_f32 } else { -1.0_f32 };
                assert_eq!(
                    res.rewards,
                    vec![expected, expected],
                    "N=2 coordination reward wrong for (a0={a0}, a1={a1}): {:?}",
                    res.rewards
                );
            }
        }
    }

    /// Enumerate all 2^4 = 16 joint actions for N = 4 and verify
    /// agent 0's reward against a hand-computed table.
    #[test]
    fn test_majority_reward_n4() {
        // For N=4, agent 0's "others" are agents 1, 2, 3 (3 agents).
        // Strict majority of 3 → 2 or 3 ones means majority = 1; 0 or 1
        // ones means majority = 0. No ties (3 is odd).
        for mask in 0..16_u32 {
            let bits: Vec<i64> = (0..4).map(|i| ((mask >> i) & 1) as i64).collect();
            let mut env = NPlayerMatchingPennies::new(4);
            env.reset_joint(None);
            let actions: Vec<Vec<i64>> = bits.iter().map(|b| vec![*b]).collect();
            let res = env.step_joint(&actions);
            // Hand-compute agent 0's expected reward.
            let others_ones: i64 = bits[1] + bits[2] + bits[3];
            let majority_other: i64 = if others_ones >= 2 { 1 } else { 0 };
            let expected_a0: f32 = if bits[0] == majority_other { 1.0 } else { -1.0 };
            assert_eq!(
                res.rewards[0], expected_a0,
                "N=4 agent-0 reward wrong for bits={bits:?}: got {} expected {}",
                res.rewards[0], expected_a0
            );
        }
    }

    /// Tie-break for odd N: `N = 3` with agent 0's "others" being
    /// agents 1 and 2 — if they pick `[0, 1]` the others split evenly,
    /// so agent 0's reward must be 0 regardless of agent 0's action.
    #[test]
    fn test_tie_break_returns_zero_on_odd_n_3() {
        // others = (a1, a2) = (0, 1) ⇒ tie ⇒ agent 0 reward = 0.
        for a0 in 0..2_i64 {
            let mut env = NPlayerMatchingPennies::new(3);
            env.reset_joint(None);
            let res = env.step_joint(&[vec![a0], vec![0], vec![1]]);
            assert_eq!(
                res.rewards[0], 0.0,
                "N=3 tie should give agent 0 reward 0 (a0={a0}), got {}",
                res.rewards[0]
            );
        }
    }

    /// Tie-break for odd N = 5: agent 0's "others" are agents
    /// {1,2,3,4}; if they split 2/2 → tie → agent 0 reward = 0.
    #[test]
    fn test_tie_break_returns_zero_on_odd_n_5() {
        for a0 in 0..2_i64 {
            let mut env = NPlayerMatchingPennies::new(5);
            env.reset_joint(None);
            // others: two 0s and two 1s ⇒ tied.
            let res = env.step_joint(&[vec![a0], vec![0], vec![0], vec![1], vec![1]]);
            assert_eq!(
                res.rewards[0], 0.0,
                "N=5 tie should give agent 0 reward 0 (a0={a0}), got {}",
                res.rewards[0]
            );
        }
    }

    /// Per-agent observations must be distinct across agents when
    /// `num_agents >= 2`.
    #[test]
    fn test_per_agent_obs_differ_by_index() {
        for n in [2_usize, 3, 4, 8] {
            let mut env = NPlayerMatchingPennies::new(n);
            let obs = env.reset_joint(None);
            assert_eq!(obs.len(), n);
            // All distinct.
            for i in 0..n {
                for j in (i + 1)..n {
                    assert_ne!(
                        obs[i], obs[j],
                        "obs for N={n} agents {i} and {j} must differ: {:?} == {:?}",
                        obs[i], obs[j]
                    );
                }
                // Length 1 each.
                assert_eq!(obs[i].len(), NPlayerMatchingPennies::OBS_DIM);
            }
        }
    }

    /// Episode `done` flag fires after exactly `EPISODE_LEN` steps,
    /// matching the 2-agent MatchingPennies convention.
    #[test]
    fn test_done_flag_at_episode_len() {
        let mut env = NPlayerMatchingPennies::new(4);
        env.reset_joint(None);
        let actions: Vec<Vec<i64>> = (0..4).map(|_| vec![0]).collect();
        for step in 0..NPlayerMatchingPennies::EPISODE_LEN {
            let res = env.step_joint(&actions);
            let expected_done = step + 1 >= NPlayerMatchingPennies::EPISODE_LEN;
            assert_eq!(res.done, expected_done, "done flag wrong at step {step}");
        }
    }

    /// At N = 4 with all-ones (a global consensus on action 1), every
    /// agent's "others" are unanimous on 1 → agent matching gets +1
    /// for everyone.
    #[test]
    fn test_n4_all_ones_unanimous_reward() {
        let mut env = NPlayerMatchingPennies::new(4);
        env.reset_joint(None);
        let actions: Vec<Vec<i64>> = (0..4).map(|_| vec![1]).collect();
        let res = env.step_joint(&actions);
        for (i, &r) in res.rewards.iter().enumerate() {
            assert_eq!(r, 1.0, "agent {i} should get +1 in unanimous-1 case, got {r}");
        }
    }

    /// Reset restores the step counter.
    #[test]
    fn test_reset_restores_step_counter() {
        let mut env = NPlayerMatchingPennies::new(3);
        env.reset_joint(None);
        let actions: Vec<Vec<i64>> = (0..3).map(|_| vec![0]).collect();
        for _ in 0..NPlayerMatchingPennies::EPISODE_LEN {
            env.step_joint(&actions);
        }
        env.reset_joint(None);
        for step in 0..NPlayerMatchingPennies::EPISODE_LEN - 1 {
            let res = env.step_joint(&actions);
            assert!(!res.done, "should not be done at step {step} after reset");
        }
        let last = env.step_joint(&actions);
        assert!(last.done, "should be done on final step after reset");
    }

    /// Observation shape is `(num_agents, OBS_DIM=1)` after both reset
    /// and step.
    #[test]
    fn test_observation_shape() {
        let n = 5;
        let mut env = NPlayerMatchingPennies::new(n);
        let obs = env.reset_joint(None);
        assert_eq!(obs.len(), n);
        for o in &obs {
            assert_eq!(o.len(), NPlayerMatchingPennies::OBS_DIM);
        }
        let actions: Vec<Vec<i64>> = (0..n).map(|_| vec![0]).collect();
        let res = env.step_joint(&actions);
        assert_eq!(res.observations.len(), n);
        for o in &res.observations {
            assert_eq!(o.len(), NPlayerMatchingPennies::OBS_DIM);
        }
    }
}
