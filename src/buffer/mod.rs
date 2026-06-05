//! Experience buffers and replay management
//!
//! This module handles storage and sampling of experience for training.

pub mod replay;
pub mod rollout;

// Convenience re-export so multi-agent training scripts can pull the
// flat-buffer GAE helper directly off `thrust_rl::buffer`. Single-agent
// users continue to call `RolloutBuffer::compute_advantages`.
pub use rollout::compute_advantages_multi_agent;
