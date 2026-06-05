//! Environment traits and implementations
//!
//! This module defines the core environment interface and provides
//! built-in environments for reinforcement learning.
//!
//! # Action types and continuous (Box) action spaces
//!
//! The `Environment` trait exposes its action type as an associated
//! type `Environment::Action`. Discrete envs (CartPole, Snake, Pong,
//! SimpleBandit, BucketBrigade) set `type Action = i64;`. Continuous
//! envs set `type Action = Vec<f32>;` (or another float type for
//! single-dim cases). See `games::continuous_lqr::ContinuousLqr` for
//! a minimal continuous-action existence proof.
//!
//! ## Why the associated-type strategy (Strategy A)?
//!
//! When this trait was extended to support continuous-control
//! algorithms (SAC, TD3, DDPG, PPO with Gaussian heads), there were
//! two candidate designs:
//!
//! - **A. Associated `Action` type on `Environment`** (chosen). One trait,
//!   parameterised by the action type. Discrete envs default to `type Action =
//!   i64;` so existing call sites and trainers migrate by adding a single line
//!   per env impl.
//! - **B. Parallel `ContinuousEnvironment` trait.** Two distinct traits; envs
//!   implement one or the other (or both). The trainer dispatches per-trait.
//!
//! Strategy A was picked because:
//!
//! 1. **One trait.** Downstream code (snapshot/restore, pool, multi-agent
//!    simulator) does not have to branch on which trait an env implements.
//!    There is exactly one `Environment` to reason about.
//! 2. **Cheap default for the common case.** Every discrete env in the repo
//!    today gets `type Action = i64;` as a one-line addition. Continuous envs
//!    opt in to `Vec<f32>`.
//! 3. **Orthogonal to other trait extensions.** The associated-type approach
//!    composes cleanly with planned additions like a `type State` slot for env
//!    snapshotting (issue #62 / clone_state / restore_state). Strategy B would
//!    have forced snapshot machinery to be implemented twice or behind a third
//!    trait.
//! 4. **Generic trainers stay generic.** Discrete trainers add a `<Action =
//!    i64>` bound on their `E: Environment` type parameter, which is purely
//!    additive.
//!
//! The trade-off: any code that wants to handle *both* discrete and
//! continuous actions polymorphically must either be parametric on
//! `E::Action` or use a sum-type wrapper. Until SAC lands, no such
//! call site exists in Thrust.

use anyhow::Result;

/// Core trait for RL environments
pub trait Environment {
    /// Action type accepted by [`Environment::step`].
    ///
    /// - **Discrete envs** set this to `i64`. The index identifies which of the
    ///   [`SpaceType::Discrete`] options was chosen.
    /// - **Continuous envs** set this to `Vec<f32>` (or `f32` for a single-dim
    ///   Box action). Each element is a real-valued coordinate of the
    ///   [`SpaceType::Box`] action vector.
    ///
    /// Existing call sites that bind actions as `i64` should
    /// constrain `E: Environment<Action = i64>` to keep the
    /// type signature compatible.
    type Action;

    /// Reset the environment and return initial observation
    fn reset(&mut self);

    /// Get the current observation
    fn get_observation(&self) -> Vec<f32>;

    /// Step the environment with an action
    fn step(&mut self, action: Self::Action) -> StepResult;

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
pub use games::{
    CartPole, ContinuousLqr, Pong, SimpleBandit, SnakeEnv, cartpole, continuous_lqr, pong,
    simple_bandit, snake,
};

// Training utilities
#[cfg(feature = "training")]
pub mod pool;
