//! Multi-agent environment trait.
//!
//! Extends the base [`crate::env::Environment`] trait to support multiple
//! agents interacting in the same game instance, enabling cooperative,
//! competitive, and mixed-motive scenarios.
//!
//! # Multi-discrete action spaces
//!
//! This trait supports both pure-discrete and *factored* (multi-discrete)
//! action spaces. Each agent's action is a `Vec<i64>` whose length equals the
//! number of action dimensions for that agent:
//!
//! - Pure-discrete agent with `n` choices: action vector has length 1 (e.g.
//!   `vec![3]` to pick action `3`).
//! - Factored Bucket-Brigade-style agent with `[house_index, mode]`: action
//!   vector has length 2 (e.g. `vec![7, 1]` for "house 7, mode 1").
//!
//! The per-agent layout is published via
//! [`MultiAgentEnvironment::agent_action_space`].
//!
//! # Burn migration note
//!
//! This trait holds no tensor types — it only deals in `Vec<f32>` host
//! observations and `Vec<i64>` host actions. The Burn-native
//! [`crate::multi_agent::joint`] trainer is the place where Burn tensors enter
//! the picture; the environment surface stays backend-agnostic.

use std::collections::HashMap;

use crate::env::Environment;

/// Multi-agent environment trait.
///
/// Environments implementing this trait support multiple agents interacting
/// in the same game instance. The base [`Environment`] trait provides the
/// single-agent reset / observation / action-space methods; this trait
/// extends it with the per-agent variants needed by the multi-agent
/// trainers.
pub trait MultiAgentEnvironment: Environment {
    /// Number of agents in this environment.
    fn num_agents(&self) -> usize;

    /// Get observation for a specific agent.
    ///
    /// # Arguments
    ///
    /// * `agent_id` - Index of the agent (`0..num_agents`).
    fn get_agent_observation(&self, agent_id: usize) -> Vec<f32>;

    /// Per-agent action-space layout.
    ///
    /// Returns one bin count per action dimension for the agent. For a
    /// pure-discrete agent with `n` choices this is `vec![n]`; for a
    /// factored agent with `[house_index, mode]` this is `vec![10, 2]`.
    ///
    /// The length of this vector must match the length of each per-agent
    /// action passed to [`MultiAgentEnvironment::step_multi`], and it must
    /// match the per-dim cardinalities of any policy driving this agent.
    fn agent_action_space(&self, agent_id: usize) -> Vec<usize>;

    /// Step the environment with multiple actions (one per agent).
    ///
    /// # Arguments
    ///
    /// * `actions` - One action vector per agent. `actions.len()` must equal
    ///   [`MultiAgentEnvironment::num_agents`]. Each inner `Vec<i64>` carries
    ///   one entry per action dimension and must match the layout reported by
    ///   [`MultiAgentEnvironment::agent_action_space`].
    ///
    /// # Returns
    ///
    /// Multi-agent result containing observations, rewards, and termination
    /// flags for each agent.
    fn step_multi(&mut self, actions: &[Vec<i64>]) -> MultiAgentResult;

    /// Get which agents are currently active (not terminated).
    fn active_agents(&self) -> Vec<bool>;
}

/// Result of a multi-agent environment step.
///
/// Contains per-agent observations, rewards, and termination flags.
#[derive(Debug, Clone)]
pub struct MultiAgentResult {
    /// Observations for each agent.
    pub observations: Vec<Vec<f32>>,

    /// Rewards for each agent.
    pub rewards: Vec<f32>,

    /// Terminal states for each agent (true = episode ended naturally,
    /// e.g. agent died or goal reached). GAE zeroes the value-function
    /// bootstrap across these transitions.
    pub terminated: Vec<bool>,

    /// Truncation flags for each agent (true = episode ended due to a
    /// time limit or external reset, not a terminal state). GAE keeps
    /// the value-function bootstrap across these transitions because
    /// the trajectory was still "alive" in the value sense. Follows the
    /// gym/gymnasium convention.
    pub truncated: Vec<bool>,

    /// Additional information (shared across all agents).
    pub info: HashMap<String, String>,
}

impl MultiAgentResult {
    /// Create a new multi-agent result.
    pub fn new(
        observations: Vec<Vec<f32>>,
        rewards: Vec<f32>,
        terminated: Vec<bool>,
        truncated: Vec<bool>,
    ) -> Self {
        Self { observations, rewards, terminated, truncated, info: HashMap::new() }
    }

    /// Check if all agents are done (either terminated or truncated).
    pub fn all_done(&self) -> bool {
        self.terminated.iter().zip(&self.truncated).all(|(term, trunc)| *term || *trunc)
    }

    /// Check if any agent is done.
    pub fn any_done(&self) -> bool {
        self.terminated.iter().zip(&self.truncated).any(|(term, trunc)| *term || *trunc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_agent_result_all_done() {
        let result = MultiAgentResult::new(
            vec![vec![0.0; 4], vec![0.0; 4]],
            vec![0.0, 0.0],
            vec![true, true],
            vec![false, false],
        );

        assert!(result.all_done());
        assert!(result.any_done());
    }

    #[test]
    fn test_multi_agent_result_any_done() {
        let result = MultiAgentResult::new(
            vec![vec![0.0; 4], vec![0.0; 4]],
            vec![0.0, 0.0],
            vec![true, false],
            vec![false, false],
        );

        assert!(!result.all_done());
        assert!(result.any_done());
    }

    #[test]
    fn test_multi_agent_result_none_done() {
        let result = MultiAgentResult::new(
            vec![vec![0.0; 4], vec![0.0; 4]],
            vec![0.0, 0.0],
            vec![false, false],
            vec![false, false],
        );

        assert!(!result.all_done());
        assert!(!result.any_done());
    }
}
