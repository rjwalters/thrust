//! Message types for multi-agent communication
//!
//! Defines the message formats for communication between:
//! - GameSimulator → PolicyLearner: Experience data
//! - PolicyLearner → GameSimulator: Policy updates

use tch::Tensor;

use super::population::AgentId;

/// Experience tuple sent from simulator to learner
#[derive(Debug)]
pub struct Experience {
    /// Agent that generated this experience
    pub agent_id: AgentId,

    /// Observation tensor [obs_dim]
    pub observation: Tensor,

    /// Action taken
    pub action: i64,

    /// Reward received
    pub reward: f32,

    /// Next observation tensor [obs_dim]
    pub next_observation: Tensor,

    /// Whether episode terminated
    pub terminated: bool,

    /// Whether episode was truncated
    pub truncated: bool,

    /// Value estimate at this state (from policy)
    pub value: f32,

    /// Log probability of the action taken
    pub log_prob: f32,
}

impl Experience {
    /// Create a new experience tuple
    pub fn new(
        agent_id: AgentId,
        observation: Tensor,
        action: i64,
        reward: f32,
        next_observation: Tensor,
        terminated: bool,
        truncated: bool,
        value: f32,
        log_prob: f32,
    ) -> Self {
        Self {
            agent_id,
            observation,
            action,
            reward,
            next_observation,
            terminated,
            truncated,
            value,
            log_prob,
        }
    }

    /// Check if this experience marks the end of an episode
    pub fn is_done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// Policy update message sent from learner to simulator
#[derive(Debug)]
pub struct PolicyUpdate {
    /// Agent whose policy was updated
    pub agent_id: AgentId,

    /// New policy version number
    pub version: u64,

    /// Path to saved model file (learner saves, simulator loads)
    /// This avoids sending large tensors through channels
    pub model_path: String,

    /// Training statistics for logging
    pub stats: TrainingStats,
}

/// Training statistics from a policy update
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Total loss
    pub total_loss: f64,

    /// Policy loss component
    pub policy_loss: f64,

    /// Value loss component
    pub value_loss: f64,

    /// Entropy bonus
    pub entropy: f64,

    /// KL divergence (for monitoring)
    pub kl_divergence: f64,

    /// Number of training steps completed
    pub step: usize,

    /// Average episode reward (if available)
    pub avg_reward: Option<f64>,
}

impl Default for TrainingStats {
    fn default() -> Self {
        Self {
            total_loss: 0.0,
            policy_loss: 0.0,
            value_loss: 0.0,
            entropy: 0.0,
            kl_divergence: 0.0,
            step: 0,
            avg_reward: None,
        }
    }
}

/// Control message for coordinating training
#[derive(Debug, Clone)]
pub enum ControlMessage {
    /// Stop training and shutdown
    Shutdown,

    /// Save checkpoint
    SaveCheckpoint { path: String },

    /// Load checkpoint
    LoadCheckpoint { path: String },

    /// Adjust learning rate
    SetLearningRate { rate: f64 },
}

#[cfg(test)]
mod tests {
    use tch::Kind;

    use super::*;

    #[test]
    fn test_experience_creation() {
        let obs = Tensor::randn([4], (Kind::Float, tch::Device::Cpu));
        let next_obs = Tensor::randn([4], (Kind::Float, tch::Device::Cpu));

        let exp = Experience::new(0, obs, 1, 1.0, next_obs, false, false, 0.5, -0.69);

        assert_eq!(exp.agent_id, 0);
        assert_eq!(exp.action, 1);
        assert_eq!(exp.reward, 1.0);
        assert!(!exp.is_done());
    }

    #[test]
    fn test_experience_done() {
        let obs = Tensor::randn([4], (Kind::Float, tch::Device::Cpu));
        let next_obs = Tensor::randn([4], (Kind::Float, tch::Device::Cpu));

        let exp_term = Experience::new(
            0,
            obs.shallow_clone(),
            1,
            1.0,
            next_obs.shallow_clone(),
            true,
            false,
            0.5,
            -0.69,
        );
        let exp_trunc = Experience::new(
            0,
            obs.shallow_clone(),
            1,
            1.0,
            next_obs.shallow_clone(),
            false,
            true,
            0.5,
            -0.69,
        );
        let exp_both = Experience::new(0, obs, 1, 1.0, next_obs, true, true, 0.5, -0.69);

        assert!(exp_term.is_done());
        assert!(exp_trunc.is_done());
        assert!(exp_both.is_done());
    }

    #[test]
    fn test_training_stats_default() {
        let stats = TrainingStats::default();
        assert_eq!(stats.step, 0);
        assert_eq!(stats.total_loss, 0.0);
        assert!(stats.avg_reward.is_none());
    }

    #[test]
    fn test_policy_update_creation() {
        let update = PolicyUpdate {
            agent_id: 0,
            version: 1,
            model_path: "/tmp/model_0_v1.pt".to_string(),
            stats: TrainingStats::default(),
        };

        assert_eq!(update.agent_id, 0);
        assert_eq!(update.version, 1);
    }
}
