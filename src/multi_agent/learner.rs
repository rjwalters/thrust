//! Policy learner thread for multi-agent training
//!
//! Each learner trains a single agent's policy using PPO.

use crate::policy::mlp::MlpPolicy;
use super::population::AgentId;

/// Policy learner - trains one agent's policy
///
/// This component runs in its own thread and is responsible for:
/// - Receiving experiences from the game simulator
/// - Computing advantages using GAE
/// - Updating policy parameters using PPO
/// - Periodically sending updated weights back to simulator
pub struct PolicyLearner {
    /// Agent ID this learner is training
    pub agent_id: AgentId,

    /// Policy network (GPU copy)
    policy: MlpPolicy,

    /// Training configuration
    config: LearnerConfig,
}

impl PolicyLearner {
    /// Create a new policy learner
    pub fn new(agent_id: AgentId, policy: MlpPolicy, config: LearnerConfig) -> Self {
        Self {
            agent_id,
            policy,
            config,
        }
    }

    // TODO: Implement train() method - main training loop
    // TODO: Implement receive_experiences() method
    // TODO: Implement compute_advantages() method
    // TODO: Implement ppo_update() method
    // TODO: Implement send_policy_update() method
}

/// Configuration for policy learner
#[derive(Debug, Clone)]
pub struct LearnerConfig {
    /// Learning rate
    pub learning_rate: f64,

    /// Discount factor
    pub gamma: f64,

    /// GAE lambda
    pub gae_lambda: f64,

    /// PPO clip epsilon
    pub clip_epsilon: f64,

    /// Value loss coefficient
    pub value_loss_coef: f64,

    /// Entropy bonus coefficient
    pub entropy_coef: f64,

    /// Batch size
    pub batch_size: usize,

    /// Number of PPO epochs per batch
    pub n_epochs: usize,

    /// Update interval (send policy updates every N steps)
    pub update_interval: usize,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
            batch_size: 256,
            n_epochs: 4,
            update_interval: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learner_config_default() {
        let config = LearnerConfig::default();
        assert_eq!(config.learning_rate, 3e-4);
        assert_eq!(config.gamma, 0.99);
        assert_eq!(config.batch_size, 256);
    }

    #[test]
    fn test_learner_creation() {
        let policy = MlpPolicy::new(4, 2, 64);
        let config = LearnerConfig::default();
        let learner = PolicyLearner::new(0, policy, config);

        assert_eq!(learner.agent_id, 0);
    }
}
