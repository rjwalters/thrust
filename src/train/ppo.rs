//! Proximal Policy Optimization (PPO) algorithm
//!
//! This module implements the PPO algorithm for training RL agents.
//! PPO is a policy gradient method that uses a clipped surrogate objective
//! to ensure stable, reliable policy updates.
//!
//! # Algorithm Overview
//!
//! ```text
//! For each epoch:
//!   1. Collect trajectories using current policy
//!   2. Compute advantages using GAE
//!   3. For multiple epochs:
//!      a. Sample minibatches from buffer
//!      b. Compute PPO loss (clipped objective)
//!      c. Update policy via gradient descent
//! ```
//!
//! # References
//!
//! - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
//! - [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

// Re-export main components. `PPOConfig`, `TrainingStats`,
// `AggregatedStats`, and `generate_minibatch_indices` are
// backend-agnostic (they do not touch `tch::Tensor`) and are
// available under either `training` (tch) or `training-burn`. The
// tch-specific bits (`compute_gae`, `compute_policy_loss`,
// `compute_value_loss`, `PPOTrainer`) stay gated on `training`.
pub use config::PPOConfig;
#[cfg(feature = "training")]
pub use loss::{compute_entropy_loss, compute_gae, compute_policy_loss, compute_value_loss};
pub use loss::generate_minibatch_indices;
pub use stats::{AggregatedStats, TrainingStats};
#[cfg(feature = "training")]
pub use trainer::PPOTrainer;

// Submodules
mod config;
mod loss;
mod stats;
#[cfg(feature = "training")]
mod trainer;
