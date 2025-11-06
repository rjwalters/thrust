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

// Re-export main components
pub use config::PPOConfig;
pub use loss::{compute_gae, compute_policy_loss, compute_value_loss, generate_minibatch_indices};
pub use stats::{AggregatedStats, TrainingStats};
pub use trainer::PPOTrainer;

// Submodules
mod config;
mod loss;
mod stats;
mod trainer;
