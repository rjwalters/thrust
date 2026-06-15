//! Game environment implementations
//!
//! This module contains various game environments for reinforcement learning:
//! - CartPole: Classic cart-pole balancing task
//! - Snake: Snake game with configurable grid size
//! - SimpleBandit: Simple multi-armed bandit for testing
//! - Pong: Single-player Pong vs rule-based opponent
//! - ContinuousLqr: 1D LQR placeholder env that exercises the continuous
//!   (`Vec<f32>`) action surface added in issue #61
//! - BucketBrigade (feature `env-bucket-brigade`): Slepian-Wolf MARL research
//!   env wrapping `bucket_brigade_core` with the versioned scenario registry
//!   from bucket-brigade PR #379
//! - MatchingPennies (feature `training`): 2-agent zero-sum smoke env
//!   implementing `JointEnv` for the multi-agent + PSRO trainers; canonical
//!   testbed for mixed-equilibrium learners (issue #107).

pub mod cartpole;
pub mod continuous_lqr;
#[cfg(feature = "training")]
pub mod matching_pennies;
pub mod pong;
pub mod simple_bandit;
pub mod snake;

#[cfg(feature = "env-bucket-brigade")]
pub mod bucket_brigade;

// Re-export main types for convenience
#[cfg(feature = "env-bucket-brigade")]
pub use bucket_brigade::BucketBrigadeMaEnv;
pub use cartpole::CartPole;
pub use continuous_lqr::ContinuousLqr;
#[cfg(feature = "training")]
pub use matching_pennies::MatchingPennies;
pub use pong::Pong;
pub use simple_bandit::SimpleBandit;
pub use snake::SnakeEnv;
