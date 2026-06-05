//! Multi-agent environment trait
//!
//! Extends the base Environment trait to support multiple agents playing
//! simultaneously in the same game instance.

use std::collections::HashMap;

use crate::env::Environment;

/// Multi-agent environment trait
///
/// Environments implementing this trait support multiple agents interacting
/// in the same game instance, enabling cooperative, competitive, and
/// mixed-motive scenarios.
///
/// # Multi-discrete action spaces
///
/// This trait supports both pure-discrete and *factored* (multi-discrete)
/// action spaces. Each agent's action is a `Vec<i64>` whose length equals the
/// number of action dimensions for that agent:
///
/// - Pure-discrete agent with `n` choices: action vector has length 1 (e.g.
///   `vec![3]` to pick action `3`).
/// - Factored Bucket-Brigade agent with `[house_index, mode]`: action vector
///   has length 2 (e.g. `vec![7, 1]` for "house 7, mode 1").
///
/// The per-agent layout is published via
/// [`agent_action_space`](MultiAgentEnvironment::agent_action_space).
pub trait MultiAgentEnvironment: Environment {
    /// Number of agents in this environment
    fn num_agents(&self) -> usize;

    /// Get observation for a specific agent
    ///
    /// # Arguments
    ///
    /// * `agent_id` - Index of the agent (0 to num_agents - 1)
    fn get_agent_observation(&self, agent_id: usize) -> Vec<f32>;

    /// Per-agent action-space layout.
    ///
    /// Returns one bin count per action dimension for the agent. For a
    /// pure-discrete agent with `n` choices this is `vec![n]`; for a factored
    /// Bucket-Brigade agent with `[house_index, mode]` this is `vec![10, 2]`.
    ///
    /// The length of this vector must match the length of each per-agent
    /// action passed to [`step_multi`](MultiAgentEnvironment::step_multi), and
    /// it must match the `action_dims` of any policy driving this agent.
    fn agent_action_space(&self, agent_id: usize) -> Vec<usize>;

    /// Step the environment with multiple actions (one per agent).
    ///
    /// # Arguments
    ///
    /// * `actions` - One action vector per agent. `actions.len()` must equal
    ///   [`num_agents`](MultiAgentEnvironment::num_agents). Each inner
    ///   `Vec<i64>` carries one entry per action dimension and must match the
    ///   layout reported by
    ///   [`agent_action_space`](MultiAgentEnvironment::agent_action_space).
    ///
    /// # Returns
    ///
    /// Multi-agent result containing observations, rewards, and termination
    /// flags for each agent.
    fn step_multi(&mut self, actions: &[Vec<i64>]) -> MultiAgentResult;

    /// Get which agents are currently active (not terminated)
    fn active_agents(&self) -> Vec<bool>;
}

/// Result of a multi-agent environment step
///
/// Contains per-agent observations, rewards, and termination flags.
#[derive(Debug, Clone)]
pub struct MultiAgentResult {
    /// Observations for each agent
    pub observations: Vec<Vec<f32>>,

    /// Rewards for each agent
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

    /// Additional information (shared across all agents)
    pub info: HashMap<String, String>,
}

impl MultiAgentResult {
    /// Create a new multi-agent result
    pub fn new(
        observations: Vec<Vec<f32>>,
        rewards: Vec<f32>,
        terminated: Vec<bool>,
        truncated: Vec<bool>,
    ) -> Self {
        Self { observations, rewards, terminated, truncated, info: HashMap::new() }
    }

    /// Check if all agents are done (either terminated or truncated)
    pub fn all_done(&self) -> bool {
        self.terminated.iter().zip(&self.truncated).all(|(term, trunc)| *term || *trunc)
    }

    /// Check if any agent is done
    pub fn any_done(&self) -> bool {
        self.terminated.iter().zip(&self.truncated).any(|(term, trunc)| *term || *trunc)
    }
}

// Tests disabled - multi-agent code needs updating
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::env::{SpaceInfo, StepInfo};
//     use anyhow::Result;
//
//     // Mock types for testing
//     #[derive(Clone)]
//     struct MockObs;
//
//     struct MockEnv;
//
//     impl Environment for MockEnv {
//         type Observation = MockObs;
//         type Action = i64;
//
//         fn reset(&mut self) -> Result<Self::Observation> {
//             Ok(MockObs)
//         }
//
//         fn step(&mut self, _action: Self::Action) ->
// Result<crate::env::StepResult<Self::Observation>> {
// Ok(crate::env::StepResult {                 observation: MockObs,
//                 reward: 0.0,
//                 terminated: false,
//                 truncated: false,
//                 info: StepInfo::default(),
//             })
//         }
//
//         fn observation_space(&self) -> SpaceInfo {
//             SpaceInfo {
//                 shape: vec![4],
//                 dtype: crate::env::SpaceType::Continuous,
//             }
//         }
//
//         fn action_space(&self) -> SpaceInfo {
//             SpaceInfo {
//                 shape: vec![],
//                 dtype: crate::env::SpaceType::Discrete(2),
//             }
//         }
//     }
//
//     impl MultiAgentEnvironment for MockEnv {
//         fn num_agents(&self) -> usize {
//             4
//         }
//
//         fn get_observation(&self, _agent_id: usize) -> Self::Observation {
//             MockObs
//         }
//
//         fn step_multi(&mut self, actions: &[Self::Action]) ->
// MultiAgentResult<Self> {             MultiAgentResult::new(
//                 vec![MockObs; actions.len()],
//                 vec![1.0; actions.len()],
//                 vec![false; actions.len()],
//                 vec![false; actions.len()],
//             )
//         }
//
//         fn active_agents(&self) -> Vec<bool> {
//             vec![true; 4]
//         }
//     }
//
//     #[test]
//     fn test_multi_agent_result_all_done() {
//         let result: MultiAgentResult<MockEnv> = MultiAgentResult::new(
//             vec![MockObs, MockObs],
//             vec![0.0, 0.0],
//             vec![true, true],
//             vec![false, false],
//         );
//
//         assert!(result.all_done());
//         assert!(result.any_done());
//     }
//
//     #[test]
//     fn test_multi_agent_result_any_done() {
//         let result: MultiAgentResult<MockEnv> = MultiAgentResult::new(
//             vec![MockObs, MockObs],
//             vec![0.0, 0.0],
//             vec![true, false],
//             vec![false, false],
//         );
//
//         assert!(!result.all_done());
//         assert!(result.any_done());
//     }
//
//     #[test]
//     fn test_multi_agent_result_none_done() {
//         let result: MultiAgentResult<MockEnv> = MultiAgentResult::new(
//             vec![MockObs, MockObs],
//             vec![0.0, 0.0],
//             vec![false, false],
//             vec![false, false],
//         );
//
//         assert!(!result.all_done());
//         assert!(!result.any_done());
//     }
// }
