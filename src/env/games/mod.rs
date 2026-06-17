//! Game environment implementations
//!
//! This module contains various game environments for reinforcement learning:
//! - CartPole: Classic cart-pole balancing task
//! - Snake: Snake game with configurable grid size
//! - SimpleBandit: Simple multi-armed bandit for testing
//! - Pong: Single-player Pong vs rule-based opponent
//! - ContinuousLqr: 1D LQR placeholder env that exercises the continuous
//!   (`Vec<f32>`) action surface added in issue #61
//! - PendulumSwingUp: in-tree `Pendulum-v1` swing-up task; the canonical
//!   continuous-control SAC benchmark (issue #139, length-1 `Vec<f32>` torque
//!   action, 3-dim `[cos θ, sin θ, θ̇]` observation)
//! - BucketBrigade (feature `env-bucket-brigade`): Slepian-Wolf MARL research
//!   env wrapping `bucket_brigade_core` with the versioned scenario registry
//!   from bucket-brigade PR #379
//! - MatchingPennies (feature `training`): 2-agent zero-sum smoke env
//!   implementing `JointEnv` for the multi-agent + PSRO trainers; canonical
//!   testbed for mixed-equilibrium learners (issue #107).
//! - NPlayerMatchingPennies (feature `training`): N-player "majority game"
//!   generalization. Each agent's reward is `+1` if its action matches the
//!   strict majority of *other* agents' actions, `-1` if against, `0` on tie
//!   (odd N only). Smoke env for N-player PSRO/NFSP testing (issue #119).

pub mod cartpole;
pub mod continuous_lqr;
#[cfg(feature = "training")]
pub mod matching_pennies;
#[cfg(feature = "training")]
pub mod n_player_matching_pennies;
pub mod pendulum;
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
#[cfg(feature = "training")]
pub use n_player_matching_pennies::NPlayerMatchingPennies;
pub use pendulum::PendulumSwingUp;
pub use pong::Pong;
pub use simple_bandit::SimpleBandit;
pub use snake::SnakeEnv;
