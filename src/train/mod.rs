//! Training algorithms
//!
//! This module implements RL training algorithms like PPO.

pub mod ppo;

pub use ppo::{
    AggregatedStats, PPOConfig, PPOTrainer, TrainingStats, compute_gae, compute_policy_loss,
    compute_value_loss, generate_minibatch_indices,
};
