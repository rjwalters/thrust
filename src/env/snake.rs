//! Multi-player Snake environment
//!
//! A competitive N-player snake game where agents compete for food.
//! - Grid-based world
//! - Each snake grows when eating food
//! - Game ends when all snakes die (collision with wall/self/others)
//! - Rewards: +1 for food, -1 for death, small time penalty to encourage
//!   efficiency

// TODO: Fix multi-agent implementation - temporarily disabled
// #[cfg(feature = "training")]
// use crate::multi_agent::environment::{MultiAgentEnvironment, MultiAgentResult};
use anyhow::Result;
pub use environment::SnakeEnv;
use rand::Rng;
pub use snake::{Food, Snake};
// Re-export main components
pub use types::{Cell, Direction, GameState, Position};

use super::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};
// TODO: Re-enable after fixing multi-agent code
// #[cfg(feature = "training")]
// pub use multi_agent::MultiAgentSnakeEnv;

// Submodules
mod environment;
mod snake;
mod types;
// TODO: Re-enable after fixing multi-agent code
// #[cfg(feature = "training")]
// mod multi_agent;

// Legacy aliases for backward compatibility
pub type SnakeEnvSingle = SnakeEnv;
// TODO: Re-enable after fixing multi-agent code
// #[cfg(feature = "training")]
// pub type SnakeEnvMulti = MultiAgentSnakeEnv;

/// Create single-agent snake environment
pub fn make_snake_env(width: i32, height: i32) -> SnakeEnv {
    SnakeEnv::new(width, height)
}

// TODO: Re-enable multi-agent snake environment after fixing multi-agent code
// #[cfg(feature = "training")]
// pub fn make_multi_snake_env(width: i32, height: i32, num_agents: usize) ->
// MultiAgentSnakeEnv {     MultiAgentSnakeEnv::new(width, height, num_agents)
// }
