//! Population management for multi-agent training
//!
//! Manages a collection of diverse agent policies and their fitness metrics.

use crate::policy::mlp::MlpPolicy;
use std::sync::{Arc, RwLock, atomic::{AtomicU64, Ordering}};

/// Unique identifier for an agent in the population
pub type AgentId = usize;

/// Population of agents training together
pub struct Population {
    /// Individual agent policies
    pub agents: Vec<Agent>,

    /// Population configuration
    pub config: PopulationConfig,

    /// Shared metrics across all agents
    pub metrics: Arc<RwLock<PopulationMetrics>>,
}

impl Population {
    /// Create a new population with random initialization
    pub fn new(config: PopulationConfig, obs_dim: i64, action_dim: i64, hidden_dim: i64) -> Self {
        let agents = (0..config.size)
            .map(|id| Agent::new(id, obs_dim, action_dim, hidden_dim))
            .collect();

        let metrics = Arc::new(RwLock::new(PopulationMetrics::default()));

        Self { agents, config, metrics }
    }

    /// Get agent by ID
    pub fn get_agent(&self, id: AgentId) -> Option<&Agent> {
        self.agents.get(id)
    }

    /// Get mutable agent by ID
    pub fn get_agent_mut(&mut self, id: AgentId) -> Option<&mut Agent> {
        self.agents.get_mut(id)
    }

    /// Update agent fitness after game completion
    pub fn update_fitness(&mut self, agent_id: AgentId, reward: f64) {
        if let Some(agent) = self.get_agent_mut(agent_id) {
            agent.total_reward += reward;
            agent.games_played += 1;
            agent.fitness = agent.total_reward / agent.games_played as f64;
        }
    }

    /// Get population statistics
    pub fn stats(&self) -> PopulationStats {
        let fitnesses: Vec<f64> = self.agents.iter().map(|a| a.fitness).collect();
        let games: Vec<usize> = self.agents.iter().map(|a| a.games_played).collect();

        let mean_fitness = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
        let max_fitness = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_fitness = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
        let total_games = games.iter().sum();

        PopulationStats {
            mean_fitness,
            max_fitness,
            min_fitness,
            total_games,
        }
    }
}

/// Individual agent in the population
pub struct Agent {
    /// Unique agent identifier
    pub id: AgentId,

    /// Policy network (CPU copy for inference)
    pub policy: Arc<RwLock<MlpPolicy>>,

    /// Agent-specific metrics
    pub fitness: f64,
    pub games_played: usize,
    pub total_reward: f64,

    /// Policy version (for staleness tracking)
    pub version: AtomicU64,
}

impl Agent {
    /// Create a new agent with randomly initialized policy
    pub fn new(id: AgentId, obs_dim: i64, action_dim: i64, hidden_dim: i64) -> Self {
        let policy = MlpPolicy::new(obs_dim, action_dim, hidden_dim);

        Self {
            id,
            policy: Arc::new(RwLock::new(policy)),
            fitness: 0.0,
            games_played: 0,
            total_reward: 0.0,
            version: AtomicU64::new(0),
        }
    }

    /// Increment policy version (after update)
    pub fn increment_version(&self) -> u64 {
        self.version.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Get current policy version
    pub fn get_version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }
}

/// Configuration for population training
#[derive(Debug, Clone)]
pub struct PopulationConfig {
    /// Number of agents in population
    pub size: usize,

    /// Matchmaking strategy
    pub matchmaking: super::matchmaking::MatchmakingStrategy,

    /// Learning mode
    pub learning_mode: LearningMode,

    /// Update frequency (steps between policy syncs)
    pub update_interval: usize,
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            size: 8,
            matchmaking: super::matchmaking::MatchmakingStrategy::RoundRobin,
            learning_mode: LearningMode::OnPolicy,
            update_interval: 100,
        }
    }
}

/// Learning mode for population training
#[derive(Debug, Clone)]
pub enum LearningMode {
    /// Each agent learns only from own experience
    OnPolicy,

    /// Agents can sample from shared buffer
    OffPolicy { buffer_size: usize },

    /// Mix of both
    Hybrid { on_policy_ratio: f64 },
}

/// Shared metrics across population
#[derive(Debug, Default)]
pub struct PopulationMetrics {
    pub total_steps: usize,
    pub total_episodes: usize,
}

/// Population statistics
#[derive(Debug)]
pub struct PopulationStats {
    pub mean_fitness: f64,
    pub max_fitness: f64,
    pub min_fitness: f64,
    pub total_games: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_population_creation() {
        let config = PopulationConfig::default();
        let pop = Population::new(config.clone(), 4, 2, 64);

        assert_eq!(pop.agents.len(), config.size);
        assert_eq!(pop.config.size, config.size);
    }

    #[test]
    fn test_agent_creation() {
        let agent = Agent::new(0, 4, 2, 64);

        assert_eq!(agent.id, 0);
        assert_eq!(agent.fitness, 0.0);
        assert_eq!(agent.games_played, 0);
        assert_eq!(agent.get_version(), 0);
    }

    #[test]
    fn test_fitness_update() {
        let config = PopulationConfig::default();
        let mut pop = Population::new(config, 4, 2, 64);

        pop.update_fitness(0, 10.0);
        pop.update_fitness(0, 20.0);

        let agent = pop.get_agent(0).unwrap();
        assert_eq!(agent.games_played, 2);
        assert_eq!(agent.total_reward, 30.0);
        assert_eq!(agent.fitness, 15.0);
    }

    #[test]
    fn test_population_stats() {
        let config = PopulationConfig {
            size: 3,
            ..Default::default()
        };
        let mut pop = Population::new(config, 4, 2, 64);

        pop.update_fitness(0, 10.0);
        pop.update_fitness(1, 20.0);
        pop.update_fitness(2, 30.0);

        let stats = pop.stats();
        assert_eq!(stats.mean_fitness, 20.0);
        assert_eq!(stats.max_fitness, 30.0);
        assert_eq!(stats.min_fitness, 10.0);
        assert_eq!(stats.total_games, 3);
    }

    #[test]
    fn test_version_increment() {
        let agent = Agent::new(0, 4, 2, 64);

        assert_eq!(agent.get_version(), 0);
        assert_eq!(agent.increment_version(), 1);
        assert_eq!(agent.get_version(), 1);
        assert_eq!(agent.increment_version(), 2);
        assert_eq!(agent.get_version(), 2);
    }
}
