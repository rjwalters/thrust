//! Game environment implementations
//!
//! This module contains various game environments for reinforcement learning:
//! - CartPole: Classic cart-pole balancing task
//! - FlickeringCartPole: CartPole whose 4-D observation is blanked to zeros
//!   with a seeded probability `p` (default 0.5) — the Hausknecht & Stone
//!   (2015) flickering-Atari POMDP protocol. Memory is load-bearing: a
//!   feedforward policy cannot act on a blanked frame, so recurrence wins by
//!   construction (issue #287, epic #262)
//! - MaskedCartPole: CartPole with the two velocity coordinates dropped from
//!   the observation. Retained as a documented NEGATIVE result — a 500k-step
//!   run showed a memoryless `[x, theta]` controller solves it, so
//!   velocity-masking alone does NOT make memory load-bearing (issue #287)
//! - TMaze: the provably memory-hard T-maze POMDP (Bakker 2001). A cue is
//!   shown only at step 0; the agent must recall it `N` steps later to turn
//!   correctly at the junction. A memoryless policy is provably at chance (50%)
//!   — the clean qualitative memory contrast that FlickeringCartPole only
//!   approximates (issue #302, epic #262)
//! - GridWorld: in-tree `4x4` FrozenLake-style sparse-reward navigation task
//!   (issue #182, `i64` discrete action `0=Up/1=Right/2=Down/3=Left`, 16-dim
//!   one-hot observation, absorbing goal/hole terminals + step-cap truncation)
//! - Snake: Snake game with configurable grid size
//! - SimpleBandit: Simple multi-armed bandit for testing
//! - Pong: Single-player Pong vs rule-based opponent
//! - ContinuousLqr: 1D LQR placeholder env that exercises the continuous
//!   (`Vec<f32>`) action surface added in issue #61
//! - PendulumSwingUp: in-tree `Pendulum-v1` swing-up task; the canonical
//!   continuous-control SAC benchmark (issue #139, length-1 `Vec<f32>` torque
//!   action, 3-dim `[cos θ, sin θ, θ̇]` observation)
//! - MountainCarContinuous: in-tree `MountainCarContinuous-v0` classic-control
//!   task; a sparse/deceptive-reward continuous-control benchmark (issue #166,
//!   length-1 `Vec<f32>` force action, 2-dim `[position, velocity]`
//!   observation, real terminal state at the goal)
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
pub mod flickering_cartpole;
pub mod grid_world;
pub mod masked_cartpole;
#[cfg(feature = "training")]
pub mod matching_pennies;
pub mod mountain_car_continuous;
#[cfg(feature = "training")]
pub mod n_player_matching_pennies;
pub mod pendulum;
pub mod pong;
#[cfg(feature = "training")]
pub mod signaling;
pub mod simple_bandit;
pub mod snake;
pub mod t_maze;

#[cfg(feature = "env-bucket-brigade")]
pub mod bucket_brigade;

// Re-export main types for convenience
#[cfg(feature = "env-bucket-brigade")]
pub use bucket_brigade::BucketBrigadeMaEnv;
pub use cartpole::CartPole;
pub use continuous_lqr::ContinuousLqr;
pub use flickering_cartpole::FlickeringCartPole;
pub use grid_world::GridWorld;
pub use masked_cartpole::MaskedCartPole;
#[cfg(feature = "training")]
pub use matching_pennies::MatchingPennies;
pub use mountain_car_continuous::MountainCarContinuous;
#[cfg(feature = "training")]
pub use n_player_matching_pennies::NPlayerMatchingPennies;
pub use pendulum::PendulumSwingUp;
pub use pong::Pong;
#[cfg(feature = "training")]
pub use signaling::SignalingGame;
pub use simple_bandit::SimpleBandit;
pub use snake::SnakeEnv;
pub use t_maze::TMaze;
