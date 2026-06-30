//! # Thrust
//!
//! High-performance reinforcement learning in Rust.
//!
//! Thrust is a modern RL library built on top of the [Burn](https://burn.dev)
//! tensor framework. After phase 5 of the Burn migration (#65), Burn is the
//! only tensor backend in the workspace; the previous `tch`/libtorch path
//! has been removed in favour of Burn's multi-backend stack (CPU NdArray,
//! WebGPU, CUDA, ROCm, Metal, Vulkan).
//!
//! ## Quick Start
//!
//! The [`prelude`] re-exports the types you need to stand up a training
//! run: the [`Environment`](crate::env::Environment) trait and a couple of
//! built-in environments, the
//! [`MlpBurnPolicy`](crate::policy::mlp::MlpBurnPolicy) actor-critic network,
//! and the trainer configs/trainers (A2C, PPO, DQN, SAC, BC). The example below
//! builds a CartPole env, an A2C config, and a policy on Burn's CPU `NdArray`
//! backend — the same pieces the `train_cartpole_a2c` example wires into a full
//! rollout/update loop.
//!
//! ```rust
//! # #[cfg(feature = "training")]
//! # fn main() {
//! use burn::backend::{Autodiff, NdArray};
//! use thrust_rl::prelude::*;
//!
//! type Backend = Autodiff<NdArray<f32>>;
//!
//! // 1. Build an environment and read its observation / action dims.
//! let env = CartPole::new();
//! let obs_dim = env.observation_space().shape[0];
//! let action_dim = match env.action_space().space_type {
//!     SpaceType::Discrete(n) => n,
//!     SpaceType::Box => panic!("CartPole is discrete"),
//! };
//!
//! // 2. Construct the actor-critic policy on the CPU backend.
//! let device = Default::default();
//! let policy = MlpBurnPolicy::<Backend>::with_config(
//!     obs_dim,
//!     action_dim,
//!     MlpBurnConfig { hidden_dim: 64, ..Default::default() },
//!     &device,
//! );
//! let _ = policy;
//!
//! // 3. Pick a trainer config. See the `train_cartpole_a2c` example for
//! //    the full rollout-collect + GAE + `A2cTrainer::train_step` loop.
//! let config = A2cConfig { num_envs: 16, n_steps: 5, ..Default::default() };
//! assert_eq!(config.num_envs, 16);
//! # }
//! # #[cfg(not(feature = "training"))]
//! # fn main() {}
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

/// Environment traits and implementations
pub mod env;

/// Policy and neural network implementations
/// inference submodule available for WASM, training modules require training
/// feature
pub mod policy;

/// Experience buffers and replay management (requires training feature).
///
/// The storage layer is plain `Vec<f32>`/`Vec<i64>` regardless of which
/// tensor backend the trainer ultimately uses; tensor materialization
/// happens via `to_burn_tensors` on the batch types when the trainer
/// calls into Burn.
#[cfg(feature = "training")]
pub mod buffer;

/// Training algorithms (PPO, DQN).
#[cfg(feature = "training")]
pub mod train;

/// Multi-agent training infrastructure (Burn backend).
///
/// Synchronized joint trainer plus the multi-agent environment trait and
/// cross-thread message payloads. Restored on top of the Burn stack in
/// issue #100 after PR #98 removed the pre-Burn tch-coupled module.
#[cfg(feature = "training")]
pub mod multi_agent;

/// Utility functions and helpers
pub mod utils;

/// Pure Rust inference for WASM compilation
pub mod inference;

/// WebAssembly bindings for browser visualization
#[cfg(feature = "wasm")]
pub mod wasm;

/// Prelude module for convenient imports.
///
/// Brings the highest-value public types into scope with a single
/// `use thrust_rl::prelude::*;`. The set is curated, not a blanket glob:
///
/// - **Environment API** (always available): the
///   [`Environment`](crate::env::Environment) trait plus
///   [`StepResult`](crate::env::StepResult),
///   [`SpaceInfo`](crate::env::SpaceInfo),
///   [`SpaceType`](crate::env::SpaceType), and the built-in
///   [`CartPole`](crate::env::CartPole) /
///   [`SimpleBandit`](crate::env::SimpleBandit) environments.
/// - **Training stack** (requires the `training` feature): the
///   [`MlpBurnPolicy`](crate::policy::mlp::MlpBurnPolicy) actor-critic network
///   and its [`MlpBurnConfig`](crate::policy::mlp::MlpBurnConfig), the trainer
///   configs/trainers (A2C, PPO, DQN, SAC, BC), and the
///   [`RolloutBuffer`](crate::buffer::rollout::RolloutBuffer) /
///   [`ReplayBuffer`](crate::buffer::replay::ReplayBuffer) experience buffers.
///
/// Training-only re-exports are `#[cfg(feature = "training")]`-gated so
/// `--no-default-features` still compiles.
pub mod prelude {
    // The `crate::env::*` re-export is unconditional; every other line is
    // `#[cfg(feature = "training")]`-gated so `--no-default-features` still
    // compiles. (`cargo fmt` alphabetizes the block, so the env import lands
    // in the middle rather than first.)
    #[cfg(feature = "training")]
    pub use crate::buffer::replay::ReplayBuffer;
    #[cfg(feature = "training")]
    pub use crate::buffer::rollout::RolloutBuffer;
    pub use crate::env::{CartPole, Environment, SimpleBandit, SpaceInfo, SpaceType, StepResult};
    #[cfg(feature = "training")]
    pub use crate::policy::mlp::{BurnActivation, MlpBurnConfig, MlpBurnPolicy};
    #[cfg(feature = "training")]
    pub use crate::train::{
        A2cConfig, A2cTrainer, BcConfig, BcTrainer, DQNConfig, DQNTrainerBurn, PPOConfig,
        PPOTrainerBurn, SacConfig, SacTrainer,
    };
}

/// Current version of thrust-rl
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.2.0");
    }
}
