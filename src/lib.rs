//! # Thrust
//!
//! High-performance reinforcement learning in Rust + CUDA
//!
//! Thrust is a modern RL library that combines Rust's performance and safety
//! with PyTorch's proven neural network capabilities (via tch-rs).
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

/// Experience buffers and replay management (requires training feature)
///
/// Gated on either backend (tch `training` or burn `training-burn`) —
/// the storage layer is plain `Vec<f32>`/`Vec<i64>` regardless of which
/// tensor framework the trainer ultimately uses. The Burn migration
/// (#65) keeps the two tensor-emitting surfaces in this module behind
/// disjoint cfgs so that `--features training-burn` can lean on the
/// existing buffers without dragging in tch.
#[cfg(any(feature = "training", feature = "training-burn"))]
pub mod buffer;

/// Training algorithms (PPO, etc.) (requires training feature)
#[cfg(feature = "training")]
pub mod train;

/// Utility functions and helpers
pub mod utils;

/// Pure Rust inference for WASM compilation
pub mod inference;

/// Multi-agent training infrastructure (requires training feature)
#[cfg(feature = "training")]
pub mod multi_agent;

/// WebAssembly bindings for browser visualization
#[cfg(feature = "wasm")]
pub mod wasm;

/// Hyperparameter optimization infrastructure
#[cfg(feature = "training")]
pub mod optimize;

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
