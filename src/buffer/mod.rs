//! Experience buffers and replay management
//!
//! This module handles storage and sampling of experience for training.

// Training-feature modules still owe public docs; tracked as the
// `--features training` follow-up to #33. Re-enable as `#![warn(missing_docs)]`
// once those items are documented.
#![allow(missing_docs)]

pub mod rollout;

// Convenience re-export so multi-agent training scripts can pull the
// flat-buffer GAE helper directly off `thrust_rl::buffer`. Single-agent
// users continue to call `RolloutBuffer::compute_advantages`.
pub use rollout::compute_advantages_multi_agent;
