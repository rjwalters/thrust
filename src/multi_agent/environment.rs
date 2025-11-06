//! Multi-agent environment trait
//!
//! Extends the base Environment trait to support multiple agents playing
//! simultaneously in the same game instance.

use crate::env::Environment;
use std::collections::HashMap;

/// Multi-agent environment trait
///
/// Environments implementing this trait support multiple agents interacting
/// in the same game instance, enabling cooperative, competitive, and
/// mixed-motive scenarios.
pub trait MultiAgentEnvironment: Environment {
    /// Number of agents in this environment
    fn num_agents(&self) -> usize;

    /// Get observation for a specific agent
    ///
    /// # Arguments
    ///
    /// * `agent_id` - Index of the agent (0 to num_agents - 1)
    fn get_agent_observation(&self, agent_id: usize) -> Vec<f32>;

    /// Step the environment with multiple actions (one per agent)
    ///
    /// # Arguments
    ///
    /// * `actions` - Slice of actions, one for each agent
    ///
    /// # Returns
    ///
    /// Multi-agent result containing observations, rewards, and termination
    /// flags for each agent.
    fn step_multi(&mut self, actions: &[i64]) -> MultiAgentResult;

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

    /// Terminal states for each agent
    pub terminated: Vec<bool>,
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
        Self {
            observations,
            rewards,
            terminated,
            truncated,
            info: HashMap::new(),
        }
    }

    /// Check if all agents are done (either terminated or truncated)
    pub fn all_done(&self) -> bool {
        self.terminated
            .iter()
            .zip(&self.truncated)
            .all(|(term, trunc)| *term || *trunc)
    }

    /// Check if any agent is done
    pub fn any_done(&self) -> bool {
        self.terminated
            .iter()
            .zip(&self.truncated)
            .any(|(term, trunc)| *term || *trunc)
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
//         fn step(&mut self, _action: Self::Action) -> Result<crate::env::StepResult<Self::Observation>> {
//             Ok(crate::env::StepResult {
//                 observation: MockObs,
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
//         fn step_multi(&mut self, actions: &[Self::Action]) -> MultiAgentResult<Self> {
//             MultiAgentResult::new(
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
