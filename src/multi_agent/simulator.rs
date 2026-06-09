//! Game simulator thread for multi-agent training
//!
//! Manages parallel environment execution, matchmaking, and experience routing.

use std::sync::{Arc, RwLock};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use tch::{Device, Kind, Tensor, no_grad};

use super::{
    environment::MultiAgentEnvironment,
    matchmaking::Matchmaker,
    messages::{Experience, PolicyUpdate},
    population::{AgentId, Population},
};

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

    /// Send experiences to learners (one channel per agent)
    experience_senders: Vec<Sender<Experience>>,

    /// Receive policy updates from learners
    policy_receiver: Receiver<PolicyUpdate>,

    /// Current matches (which agents in which env)
    current_matches: Vec<Vec<AgentId>>,

    /// Device for tensor operations
    device: Device,
}

impl<E> GameSimulator<E>
where
    E: MultiAgentEnvironment,
{
    /// Create a new game simulator
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        env_factory: &dyn Fn() -> E,
        num_envs: usize,
        population: Arc<RwLock<Population>>,
        matchmaker: Box<dyn Matchmaker>,
        agents_per_game: usize,
        experience_senders: Vec<Sender<Experience>>,
        policy_receiver: Receiver<PolicyUpdate>,
    ) -> Self {
        let env_pool = (0..num_envs).map(|_| env_factory()).collect();

        Self {
            env_pool,
            population,
            matchmaker,
            agents_per_game,
            experience_senders,
            policy_receiver,
            current_matches: Vec::new(),
            device: Device::cuda_if_available(),
        }
    }

    /// Main simulation loop
    pub fn run(&mut self, total_steps: usize) -> Result<()> {
        let mut steps = 0;

        while steps < total_steps {
            // 1. Create new matches
            self.create_matches()?;

            // 2. Reset all environments
            self.reset_environments()?;

            // 3. Run episodes and collect experiences
            let episode_steps = self.run_episodes()?;
            steps += episode_steps;

            // 4. Sync policies from learners (non-blocking)
            self.sync_policies()?;

            tracing::info!(
                "Simulator: {} steps completed ({:.1}%)",
                steps,
                100.0 * steps as f64 / total_steps as f64
            );
        }

        Ok(())
    }

    /// Create new matches using matchmaker
    fn create_matches(&mut self) -> Result<()> {
        let pop = self.population.read().unwrap();
        self.current_matches =
            self.matchmaker.create_matches(&pop, self.env_pool.len(), self.agents_per_game);
        Ok(())
    }

    /// Reset all environments to initial state
    fn reset_environments(&mut self) -> Result<()> {
        for env in &mut self.env_pool {
            env.reset();
        }
        Ok(())
    }

    /// Run episodes until all environments are done
    fn run_episodes(&mut self) -> Result<usize> {
        let mut total_steps = 0;
        let mut env_done = vec![false; self.env_pool.len()];
        let device = self.device; // Extract device to avoid borrow conflicts

        while !env_done.iter().all(|&done| done) {
            for (env_id, env) in self.env_pool.iter_mut().enumerate() {
                if env_done[env_id] {
                    continue;
                }

                // Get agents for this environment
                let agent_ids = &self.current_matches[env_id];

                // Get observations for all agents
                let observations: Vec<_> =
                    (0..self.agents_per_game).map(|i| env.get_agent_observation(i)).collect();

                // Each agent selects an action
                let mut actions = Vec::new();
                let mut values = Vec::new();
                let mut log_probs = Vec::new();

                for (i, &agent_id) in agent_ids.iter().enumerate() {
                    // Get observation as tensor (use extracted device)
                    let obs_tensor = Self::obs_to_tensor(&observations[i], device)?;

                    // Get action from agent's policy (with no_grad)
                    let (action, log_prob, value) = no_grad(|| {
                        let pop = self.population.read().unwrap();
                        let policy = pop.agents[agent_id].policy.read().unwrap();
                        policy.get_action(&obs_tensor)
                    });

                    // Extract scalar values
                    let action_val: i64 = action.int64_value(&[]);
                    let log_prob_val: f64 = log_prob.double_value(&[]);
                    let value_val: f64 = value.double_value(&[]);

                    actions.push(action_val);
                    log_probs.push(log_prob_val as f32);
                    values.push(value_val as f32);
                }

                // Step environment with all actions.
                //
                // The simulator's `Experience::action` is a scalar `i64`, so
                // for now we only drive pure-discrete agents through the
                // `MultiAgentEnvironment` trait. Each scalar action is wrapped
                // in a length-1 `Vec<i64>` to match the multi-discrete trait
                // signature. End-to-end multi-discrete support (widening
                // `Experience::action` to `Vec<i64>`) is tracked separately
                // -- see issue #3 scope notes.
                let actions_md: Vec<Vec<i64>> = actions.iter().map(|&a| vec![a]).collect();
                let result = env.step_multi(&actions_md);

                // Send experiences to learners. Observations cross the
                // channel as `Vec<f32>` host buffers (phase 3 of the
                // Burn migration, #80) — backend-agnostic message
                // protocol, no tensor round-trip on the wire.
                for (i, &agent_id) in agent_ids.iter().enumerate() {
                    let obs_vec = observations[i].clone();
                    let next_obs_vec = result.observations[i].clone();

                    let exp = Experience::new(
                        agent_id,
                        obs_vec,
                        actions[i],
                        result.rewards[i],
                        next_obs_vec,
                        result.terminated[i],
                        result.truncated[i],
                        values[i],
                        log_probs[i],
                    );

                    // Send to appropriate learner (non-blocking)
                    if let Err(e) = self.experience_senders[agent_id].try_send(exp) {
                        tracing::warn!("Failed to send experience to agent {}: {}", agent_id, e);
                    }
                }

                total_steps += 1;

                // Check if episode is done
                if result.all_done() {
                    env_done[env_id] = true;
                }
            }
        }

        Ok(total_steps)
    }

    /// Sync policies from learners (non-blocking)
    fn sync_policies(&mut self) -> Result<()> {
        // Drain all available policy updates
        while let Ok(update) = self.policy_receiver.try_recv() {
            tracing::info!(
                "Simulator: Syncing policy for agent {} (version {})",
                update.agent_id,
                update.version
            );

            // Load updated policy
            let mut pop = self.population.write().unwrap();
            let agent = &mut pop.agents[update.agent_id];

            // Load from saved model file
            agent.policy.write().unwrap().load(&update.model_path)?;

            // Update version
            agent.increment_version();

            // Log training stats
            tracing::info!(
                "Agent {} | Loss: {:.3} | Policy: {:.3} | Value: {:.3} | Entropy: {:.3}",
                update.agent_id,
                update.stats.total_loss,
                update.stats.policy_loss,
                update.stats.value_loss,
                update.stats.entropy,
            );
        }

        Ok(())
    }

    /// Convert observation to tensor
    /// This is a placeholder - real implementation depends on observation type
    fn obs_to_tensor(_obs: &Vec<f32>, device: Device) -> Result<Tensor> {
        // TODO: This needs to be generic over observation types
        // For now, assume observations are already Vec<f32>-like
        // In real implementation, we'd need trait bounds or conversion logic

        // Placeholder: create a dummy tensor
        // Real implementation would extract data from obs
        Ok(Tensor::zeros(&[4], (Kind::Float, device)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult},
        multi_agent::{environment::MultiAgentResult, population::PopulationConfig},
    };

    /// Minimal `Environment` impl for tests.
    ///
    /// `num_agents` is fixed at 4 and each agent observes a 4-dim zero vector.
    struct MockEnv {
        num_agents: usize,
        obs_dim: usize,
    }

    impl MockEnv {
        fn new() -> Self {
            Self { num_agents: 4, obs_dim: 4 }
        }
    }

    impl Environment for MockEnv {
        type Action = i64;
        type State = ();

        fn reset(&mut self) {}

        fn get_observation(&self) -> Vec<f32> {
            vec![0.0; self.obs_dim]
        }

        fn step(&mut self, _action: i64) -> StepResult {
            StepResult {
                observation: vec![0.0; self.obs_dim],
                reward: 0.0,
                terminated: false,
                truncated: false,
                info: StepInfo::default(),
            }
        }

        fn observation_space(&self) -> SpaceInfo {
            SpaceInfo { shape: vec![self.obs_dim], space_type: SpaceType::Box }
        }

        fn action_space(&self) -> SpaceInfo {
            SpaceInfo { shape: vec![], space_type: SpaceType::Discrete(2) }
        }

        fn render(&self) -> Vec<u8> {
            Vec::new()
        }

        fn close(&mut self) {}

        fn clone_state(&self) {}

        fn restore_state(&mut self, _state: &()) {}
    }

    impl MultiAgentEnvironment for MockEnv {
        fn num_agents(&self) -> usize {
            self.num_agents
        }

        fn get_agent_observation(&self, _agent_id: usize) -> Vec<f32> {
            vec![0.0; self.obs_dim]
        }

        fn agent_action_space(&self, _agent_id: usize) -> Vec<usize> {
            // Pure-discrete agent with 2 choices: a single action dim.
            vec![2]
        }

        fn step_multi(&mut self, actions: &[Vec<i64>]) -> MultiAgentResult {
            assert_eq!(
                actions.len(),
                self.num_agents,
                "MockEnv::step_multi: expected {} actions, got {}",
                self.num_agents,
                actions.len()
            );
            MultiAgentResult::new(
                vec![vec![0.0; self.obs_dim]; actions.len()],
                vec![1.0; actions.len()],
                vec![false; actions.len()],
                vec![false; actions.len()],
            )
        }

        fn active_agents(&self) -> Vec<bool> {
            vec![true; self.num_agents]
        }
    }

    #[test]
    fn test_simulator_creation() {
        let config = PopulationConfig::default();
        let population = Arc::new(RwLock::new(Population::new(config.clone(), 4, 2, 64)));
        let matchmaker = config.matchmaking.create_matchmaker();

        let (exp_senders, _exp_receivers): (Vec<_>, Vec<_>) =
            (0..8).map(|_| crossbeam_channel::unbounded()).unzip();
        let (_policy_sender, policy_receiver) = crossbeam_channel::unbounded();

        let _simulator = GameSimulator::new(
            &MockEnv::new,
            4, // num_envs
            population,
            matchmaker,
            4, // agents_per_game
            exp_senders,
            policy_receiver,
        );
    }

    #[test]
    fn test_mock_env_step_multi_shape() {
        // Sanity check that the test mock satisfies the trait contract:
        // each agent's action vec has length 1 (pure-discrete) and the
        // result has one entry per agent.
        let mut env = MockEnv::new();
        let actions: Vec<Vec<i64>> = (0..env.num_agents()).map(|_| vec![0]).collect();
        let result = env.step_multi(&actions);
        assert_eq!(result.rewards.len(), env.num_agents());
        assert_eq!(result.observations.len(), env.num_agents());
        for obs in &result.observations {
            assert_eq!(obs.len(), 4);
        }
        for aid in 0..env.num_agents() {
            assert_eq!(env.agent_action_space(aid), vec![2]);
        }
    }
}
