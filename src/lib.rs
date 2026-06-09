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
//! ```rust,no_run
//! use thrust_rl::prelude::*;
//!
//! // Coming soon: Simple training example
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

/// Prelude module for convenient imports
///
/// This module re-exports commonly used types and traits for convenience.
pub mod prelude {
    // Re-export key types here as we build them
}

/// Current version of thrust-rl
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
    }
}
