//! Training algorithms
//!
//! This module implements RL training algorithms like PPO.

pub mod ppo;

pub use ppo::{PPOConfig, PPOTrainer, TrainingStats};
