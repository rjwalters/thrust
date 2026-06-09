//! Message types for multi-agent communication
//!
//! Defines the message formats for communication between:
//! - GameSimulator → PolicyLearner: Experience data
//! - PolicyLearner → GameSimulator: Policy updates
//!
//! # Backend-agnostic storage (phase 3 of the Burn migration, #80)
//!
//! Prior to phase 3 the `Experience` struct stored `tch::Tensor`
//! observations directly. That tied every channel-shaped message to
//! the tch backend even though the multi_agent simulator/learner do
//! not need autograd on these tensors — they always immediately
//! convert to `Vec<f32>` host data before pushing into the replay
//! buffer.
//!
//! Phase 3 inlines that conversion: `Experience::observation` /
//! `next_observation` are now `Vec<f32>` host primitives. The
//! tch-using simulator builds them with `Vec::<f32>::try_from(&tensor)`
//! at the boundary (was: send the tensor, receive-and-convert on the
//! other side); the receiver consumes them directly. This:
//!
//! 1. Makes `messages.rs` backend-agnostic (no `tch::Tensor` fields).
//! 2. Removes a redundant host→tch→host round-trip on every message.
//! 3. Sets up phase 4 to swap `MlpPolicy` for `MlpBurnPolicy` in the simulator
//!    without touching the message protocol.

use super::population::AgentId;

/// Experience tuple sent from simulator to learner.
///
/// Observations are stored as `Vec<f32>` host primitives so the
/// message protocol is backend-agnostic; see the module-level note
/// for the rationale.
#[derive(Debug, Clone)]
pub struct Experience {
    /// Agent that generated this experience
    pub agent_id: AgentId,

    /// Observation, flattened to `[obs_dim]` f32. Backend-neutral.
    pub observation: Vec<f32>,

    /// Action taken
    pub action: i64,

    /// Reward received
    pub reward: f32,

    /// Next observation, flattened to `[obs_dim]` f32.
    pub next_observation: Vec<f32>,

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
    /// Create a new experience tuple.
    ///
    /// Observations are `Vec<f32>` host buffers; callers using tch
    /// should convert at the boundary with
    /// `Vec::<f32>::try_from(&tensor)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        agent_id: AgentId,
        observation: Vec<f32>,
        action: i64,
        reward: f32,
        next_observation: Vec<f32>,
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
    SaveCheckpoint {
        /// Destination path for the checkpoint blob. Format is determined
        /// by the receiving learner; typically a `.safetensors` or
        /// `tch`-native `.ot` file.
        path: String,
    },

    /// Load checkpoint
    LoadCheckpoint {
        /// Source path of the checkpoint blob to restore into the
        /// receiving learner's var-store.
        path: String,
    },

    /// Adjust learning rate
    SetLearningRate {
        /// New learning rate (replaces the optimizer's current rate; not
        /// applied as a delta). Effective on the next optimizer step.
        rate: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(seed: usize) -> Vec<f32> {
        (0..4).map(|i| (seed + i) as f32 * 0.1).collect()
    }

    #[test]
    fn test_experience_creation() {
        let exp = Experience::new(0, obs(0), 1, 1.0, obs(1), false, false, 0.5, -0.69);
        assert_eq!(exp.agent_id, 0);
        assert_eq!(exp.action, 1);
        assert_eq!(exp.reward, 1.0);
        assert!(!exp.is_done());
    }

    #[test]
    fn test_experience_done() {
        let exp_term = Experience::new(0, obs(0), 1, 1.0, obs(1), true, false, 0.5, -0.69);
        let exp_trunc = Experience::new(0, obs(0), 1, 1.0, obs(1), false, true, 0.5, -0.69);
        let exp_both = Experience::new(0, obs(0), 1, 1.0, obs(1), true, true, 0.5, -0.69);
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
