//! Environment traits and implementations
//!
//! This module defines the core environment interface and provides
//! built-in environments for reinforcement learning.

use anyhow::Result;

/// Core trait for RL environments
pub trait Environment {
    /// Observation type
    type Observation;

    /// Action type
    type Action;

    /// Reset the environment and return initial observation
    fn reset(&mut self) -> Result<Self::Observation>;

    /// Step the environment with an action
    fn step(&mut self, action: Self::Action) -> Result<StepResult<Self::Observation>>;

    /// Get the observation space dimensions
    fn observation_space(&self) -> SpaceInfo;

    /// Get the action space dimensions
    fn action_space(&self) -> SpaceInfo;
}

/// Result of an environment step
#[derive(Debug, Clone)]
pub struct StepResult<O> {
    /// Next observation
    pub observation: O,

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
    pub dtype: SpaceType,
}

/// Space data types
#[derive(Debug, Clone, Copy)]
pub enum SpaceType {
    /// Discrete space with n options
    Discrete(usize),

    /// Continuous space (Box)
    Continuous,

    /// Multi-discrete space
    MultiDiscrete,
}

/// Additional step information
#[derive(Debug, Clone, Default)]
pub struct StepInfo {
    // Add custom fields as needed
}

// Built-in environments will go in submodules
pub mod cartpole;
pub mod pool;
// pub mod snake;
