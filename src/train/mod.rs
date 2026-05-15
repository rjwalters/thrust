//! Training algorithms
//!
//! This module implements RL training algorithms like PPO.

// Training-feature modules still owe public docs; tracked as the
// `--features training` follow-up to #33. Re-enable as `#![warn(missing_docs)]`
// once those items are documented.
#![allow(missing_docs)]

pub mod ppo;

pub use ppo::{
    AggregatedStats, PPOConfig, PPOTrainer, TrainingStats, compute_gae, compute_policy_loss,
    compute_value_loss, generate_minibatch_indices,
};
