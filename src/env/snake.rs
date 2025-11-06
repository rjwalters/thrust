//! Multi-player Snake environment
//!
//! A competitive N-player snake game where agents compete for food.
//! - Grid-based world
//! - Each snake grows when eating food
//! - Game ends when all snakes die (collision with wall/self/others)
//! - Rewards: +1 for food, -1 for death, small time penalty to encourage efficiency

use super::{Environment, SpaceInfo, SpaceType, StepResult, StepInfo};
// TODO: Fix multi-agent implementation - temporarily disabled
// #[cfg(feature = "training")]
// use crate::multi_agent::environment::{MultiAgentEnvironment, MultiAgentResult};
use anyhow::Result;
use rand::Rng;

// Re-export main components
pub use types::{Direction, Position, GameState, Cell};
pub use snake::{Snake, Food};
pub use environment::SnakeEnv;
// TODO: Re-enable after fixing multi-agent code
// #[cfg(feature = "training")]
// pub use multi_agent::MultiAgentSnakeEnv;

// Submodules
mod types;
mod snake;
mod environment;
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
// pub fn make_multi_snake_env(width: i32, height: i32, num_agents: usize) -> MultiAgentSnakeEnv {
//     MultiAgentSnakeEnv::new(width, height, num_agents)
// }
