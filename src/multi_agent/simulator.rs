//! Game simulator thread for multi-agent training
//!
//! Manages parallel environment execution, matchmaking, and experience routing.

use super::{
    environment::{MultiAgentEnvironment, MultiAgentResult},
    matchmaking::Matchmaker,
    population::{AgentId, Population},
};
use std::sync::{Arc, RwLock};

/// Game simulator - runs environments and routes experiences to learners
///
/// This component runs in its own thread and is responsible for:
/// - Creating matches (which agents play together)
/// - Running parallel game instances
/// - Collecting trajectories
/// - Routing experiences to the correct learner threads
pub struct GameSimulator<E: MultiAgentEnvironment> {
    /// Pool of parallel environments
    env_pool: Vec<E>,

    /// Population of agents
    population: Arc<RwLock<Population>>,

    /// Matchmaking strategy
    matchmaker: Box<dyn Matchmaker>,

    /// Number of agents per game
    agents_per_game: usize,
}

impl<E: MultiAgentEnvironment> GameSimulator<E> {
    /// Create a new game simulator
    pub fn new(
        env_factory: &dyn Fn() -> E,
        num_envs: usize,
        population: Arc<RwLock<Population>>,
        matchmaker: Box<dyn Matchmaker>,
        agents_per_game: usize,
    ) -> Self {
        let env_pool = (0..num_envs).map(|_| env_factory()).collect();

        Self {
            env_pool,
            population,
            matchmaker,
            agents_per_game,
        }
    }

    // TODO: Implement run() method
    // TODO: Implement run_episodes() method
    // TODO: Implement route_experiences() method
    // TODO: Implement sync_policies() method
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        env::{Environment, SpaceInfo, StepResult, SpaceType, StepInfo},
        multi_agent::population::PopulationConfig,
    };
    use anyhow::Result;

    #[derive(Clone)]
    struct MockObs;

    struct MockEnv;

    impl Environment for MockEnv {
        type Observation = MockObs;
        type Action = i64;

        fn reset(&mut self) -> Result<Self::Observation> {
            Ok(MockObs)
        }

        fn step(&mut self, _action: Self::Action) -> Result<StepResult<Self::Observation>> {
            Ok(StepResult {
                observation: MockObs,
                reward: 0.0,
                terminated: false,
                truncated: false,
                info: StepInfo::default(),
            })
        }

        fn observation_space(&self) -> SpaceInfo {
            SpaceInfo {
                shape: vec![],
                dtype: SpaceType::Discrete(2),
            }
        }

        fn action_space(&self) -> SpaceInfo {
            SpaceInfo {
                shape: vec![],
                dtype: SpaceType::Discrete(2),
            }
        }
    }

    impl MultiAgentEnvironment for MockEnv {
        fn num_agents(&self) -> usize {
            4
        }

        fn get_observation(&self, _agent_id: usize) -> Self::Observation {
            MockObs
        }

        fn step_multi(&mut self, actions: &[Self::Action]) -> MultiAgentResult<Self> {
            MultiAgentResult::new(
                vec![MockObs; actions.len()],
                vec![1.0; actions.len()],
                vec![false; actions.len()],
                vec![false; actions.len()],
            )
        }

        fn active_agents(&self) -> Vec<bool> {
            vec![true; 4]
        }
    }

    #[test]
    fn test_simulator_creation() {
        let config = PopulationConfig::default();
        let population = Arc::new(RwLock::new(Population::new(config.clone(), 4, 2, 64)));
        let matchmaker = config.matchmaking.create_matchmaker();

        let _simulator = GameSimulator::new(
            &|| MockEnv,
            4, // num_envs
            population,
            matchmaker,
            4, // agents_per_game
        );
    }
}
