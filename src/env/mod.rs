//! Environment traits and implementations
//!
//! This module defines the core environment interface and provides
//! built-in environments for reinforcement learning.

use anyhow::Result;

/// Core trait for RL environments
pub trait Environment {
    /// Reset the environment and return initial observation
    fn reset(&mut self);

    /// Get the current observation
    fn get_observation(&self) -> Vec<f32>;

    /// Step the environment with an action
    fn step(&mut self, action: i64) -> StepResult;

    /// Get the observation space dimensions
    fn observation_space(&self) -> SpaceInfo;

    /// Get the action space dimensions
    fn action_space(&self) -> SpaceInfo;

    /// Render the current environment state
    fn render(&self) -> Vec<u8>;

    /// Close the environment
    fn close(&mut self);
}

/// Result of an environment step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Next observation
    pub observation: Vec<f32>,

    /// Reward received
    pub reward: f32,

    /// Whether the episode terminated
    pub terminated: bool,

    /// Whether the episode was truncated
    pub truncated: bool,

    /// Additional info
    pub info: StepInfo,
}

/// Space information for observations and actions
#[derive(Debug, Clone)]
pub struct SpaceInfo {
    /// Shape of the space
    pub shape: Vec<usize>,

    /// Data type
    pub space_type: SpaceType,
}

/// Space data types
#[derive(Debug, Clone, Copy)]
pub enum SpaceType {
    /// Discrete space with n options
    Discrete(usize),

    /// Continuous space (Box)
    Box,
}

/// Additional step information
#[derive(Debug, Clone, Default)]
pub struct StepInfo {
    // Add custom fields as needed
}

// Game environments
pub mod games;

// Re-export game environments for backwards compatibility
pub use games::{cartpole, pong, simple_bandit, snake};
pub use games::{CartPole, Pong, SimpleBandit, SnakeEnv};

// Training utilities
#[cfg(feature = "training")]
pub mod pool;
