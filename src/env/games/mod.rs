//! Game environment implementations
//!
//! This module contains various game environments for reinforcement learning:
//! - CartPole: Classic cart-pole balancing task
//! - Snake: Snake game with configurable grid size
//! - SimpleBandit: Simple multi-armed bandit for testing

pub mod cartpole;
pub mod simple_bandit;
pub mod snake;

// Re-export main types for convenience
pub use cartpole::CartPole;
pub use simple_bandit::SimpleBandit;
pub use snake::SnakeEnv;
