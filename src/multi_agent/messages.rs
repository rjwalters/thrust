//! Message types for multi-agent coordination.
//!
//! Defines the per-agent message formats exchanged between the components
//! of a population-based / threaded multi-agent training setup
//! (simulator threads producing experience, learner threads consuming
//! it). After the Burn migration the message payloads are all plain
//! `Vec<f32>` / `Vec<i64>` host data — no tensor types cross the channel
//! boundary, which keeps the producer and consumer free to choose
//! different Burn backends if they want to.
//!
//! # Scope
//!
//! Only the data carriers and control envelope live here. The threaded
//! learner / simulator implementations that consume these messages live
//! one layer up (and are intentionally minimal in this first Burn-native
//! port — see [`crate::multi_agent`] for the module-level scope).

use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Unique identifier for an agent across a multi-agent training run.
pub type AgentId = usize;

/// Experience tuple sent from a simulator to a learner.
///
/// All tensor payloads from the pre-Burn (`tch`-coupled) implementation are
/// replaced with plain `Vec<f32>` host buffers. Construct Burn tensors at
/// the learner-side rollout buffer with `Tensor::from_floats(...)` once
/// the buffer is full.
///
/// `Serialize`/`Deserialize` back the networked actor-learner transport's
/// wire encoding ([`crate::train::ppo::transport`], issue #281 / epic #265
/// Phase 4) — the in-process `crossbeam_channel` path never touches these
/// impls, it passes `Experience` values directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Agent that generated this experience.
    pub agent_id: AgentId,

    /// Observation vector `[obs_dim]`.
    pub observation: Vec<f32>,

    /// Action taken. Length 1 for scalar discrete; `num_dims` for
    /// multi-discrete (matches
    /// [`crate::multi_agent::environment::MultiAgentEnvironment::agent_action_space`]).
    pub action: Vec<i64>,

    /// Reward received.
    pub reward: f32,

    /// Next observation vector `[obs_dim]`.
    pub next_observation: Vec<f32>,

    /// Whether episode terminated (natural episode end).
    pub terminated: bool,

    /// Whether episode was truncated (time limit / external reset).
    pub truncated: bool,

    /// Value estimate at this state (from the rollout-time policy).
    pub value: f32,

    /// Log probability of the action taken under the rollout-time policy.
    pub log_prob: f32,
}

impl Experience {
    /// Create a new experience tuple.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        agent_id: AgentId,
        observation: Vec<f32>,
        action: Vec<i64>,
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
    /// (terminated or truncated).
    pub fn is_done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// Policy update message sent from a learner back to a simulator.
///
/// Pre-Burn this carried a path to a saved `tch` model file so the
/// simulator could reload via `VarStore`. Post-Burn the same pattern
/// works: the learner saves a Burn `BinFileRecorder` checkpoint and
/// publishes the path. The simulator may also pull updates in-process
/// by `clone()`ing the learner's [`burn::module::Module`] directly —
/// this struct is the cross-thread path.
#[derive(Debug, Clone)]
pub struct PolicyUpdate {
    /// Agent whose policy was updated.
    pub agent_id: AgentId,

    /// New policy version number (monotonically increasing).
    pub version: u64,

    /// Path to saved model file (learner saves, simulator loads).
    /// Avoids sending large parameter blobs through channels.
    pub model_path: String,

    /// Training statistics for logging.
    pub stats: TrainingStats,
}

/// In-process policy broadcast from a learner to its actors.
///
/// Cross-thread, in-process counterpart to [`PolicyUpdate`] (which
/// publishes a saved-model *file path*). This enum is what the
/// single-host asynchronous actor-learner runner
/// ([`crate::train::ppo::actor_learner`]) sends over each per-actor
/// `crossbeam_channel` broadcast channel after a learner update. The
/// payload representation is chosen by the learner implementation; the
/// actor side stays agnostic by matching on the variant.
///
/// `Serialize`/`Deserialize` (via serde's `rc` feature for the `Arc<Vec<u8>>`
/// payload) back the networked actor-learner transport's wire encoding
/// ([`crate::train::ppo::transport`], issue #281 / epic #265 Phase 4). The
/// in-process `crossbeam_channel` path never touches these impls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyBroadcast {
    /// Serialized Burn module record bytes, produced by
    /// [`burn::record::BinBytesRecorder`] with
    /// [`burn::record::FullPrecisionSettings`].
    ///
    /// Bytes are always `Send` regardless of the tensor backend (unlike
    /// raw module clones, whose `Send`-ness depends on the backend's
    /// tensor primitives), and the [`Arc`] keeps the N-actor fan-out
    /// allocation-free — each actor channel receives a cheap pointer
    /// clone of the same serialized blob. Over the wire, deserializing
    /// allocates a fresh `Arc` per remote actor (no sharing across the
    /// network), which is the correct behavior for a byte payload received
    /// independently by each connection.
    Bytes {
        /// Monotonically increasing policy version. Incremented by the
        /// learner once per broadcast so actors (and tests) can observe
        /// how fresh their local policy copy is.
        version: u64,
        /// Shared serialized module record.
        bytes: Arc<Vec<u8>>,
    },
}

impl PolicyBroadcast {
    /// The policy version carried by this broadcast.
    pub fn version(&self) -> u64 {
        match self {
            Self::Bytes { version, .. } => *version,
        }
    }
}

/// Training statistics from a policy update.
///
/// Mirrors [`crate::train::ppo::TrainingStats`] field-for-field for the
/// quantities a simulator typically needs to log. Stays a separate
/// struct so the multi-agent message surface does not pull in the
/// per-update aggregator.
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Total loss (weighted sum of policy / value / entropy / aux).
    pub total_loss: f64,

    /// Policy loss component.
    pub policy_loss: f64,

    /// Value loss component.
    pub value_loss: f64,

    /// Entropy bonus.
    pub entropy: f64,

    /// KL divergence (for monitoring).
    pub kl_divergence: f64,

    /// Number of gradient updates completed.
    pub step: usize,

    /// Average episode reward (if available).
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

/// Control message for coordinating training.
///
/// The Burn-native receivers handle these as best-effort hints — there
/// is no global broadcast required to participate in the multi-agent
/// surface, but a runner that wants to support checkpointing and
/// learning-rate scheduling can plumb this enum through its channel
/// network.
///
/// `Serialize`/`Deserialize` back the networked actor-learner transport's
/// wire encoding ([`crate::train::ppo::transport`], issue #281 / epic #265
/// Phase 4). The in-process `crossbeam_channel` path never touches these
/// impls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    /// Stop training and shut down.
    Shutdown,

    /// Save checkpoint.
    SaveCheckpoint {
        /// Destination path for the checkpoint blob. Format is determined
        /// by the receiving learner; typical is a Burn
        /// `BinFileRecorder` `.bin` file.
        path: String,
    },

    /// Load checkpoint.
    LoadCheckpoint {
        /// Source path of the checkpoint blob to restore into the
        /// receiving learner's module.
        path: String,
    },

    /// Adjust learning rate.
    SetLearningRate {
        /// New learning rate (replaces the optimizer's current rate;
        /// not applied as a delta). Effective on the next optimizer
        /// step.
        rate: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experience_creation() {
        let exp = Experience::new(
            0,
            vec![0.1, 0.2, 0.3, 0.4],
            vec![1],
            1.0,
            vec![0.5, 0.6, 0.7, 0.8],
            false,
            false,
            0.5,
            -0.69,
        );

        assert_eq!(exp.agent_id, 0);
        assert_eq!(exp.action, vec![1]);
        assert_eq!(exp.reward, 1.0);
        assert!(!exp.is_done());
    }

    #[test]
    fn test_experience_done() {
        let make = |term, trunc| {
            Experience::new(0, vec![0.0; 4], vec![1], 1.0, vec![0.0; 4], term, trunc, 0.5, -0.69)
        };

        assert!(make(true, false).is_done());
        assert!(make(false, true).is_done());
        assert!(make(true, true).is_done());
        assert!(!make(false, false).is_done());
    }

    #[test]
    fn test_training_stats_default() {
        let stats = TrainingStats::default();
        assert_eq!(stats.step, 0);
        assert_eq!(stats.total_loss, 0.0);
        assert!(stats.avg_reward.is_none());
    }

    #[test]
    fn test_policy_broadcast_version_and_shared_bytes() {
        let bytes = Arc::new(vec![1u8, 2, 3]);
        let broadcast = PolicyBroadcast::Bytes { version: 7, bytes: Arc::clone(&bytes) };

        assert_eq!(broadcast.version(), 7);

        // Cloning the broadcast shares the underlying blob (no deep copy).
        let cloned = broadcast.clone();
        let PolicyBroadcast::Bytes { bytes: cloned_bytes, .. } = cloned;
        assert!(Arc::ptr_eq(&bytes, &cloned_bytes));
    }

    #[test]
    fn test_policy_update_creation() {
        let update = PolicyUpdate {
            agent_id: 0,
            version: 1,
            model_path: "/tmp/model_0_v1.bin".to_string(),
            stats: TrainingStats::default(),
        };

        assert_eq!(update.agent_id, 0);
        assert_eq!(update.version, 1);
    }
}
