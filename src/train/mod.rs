//! Training algorithms
//!
//! This module implements RL training algorithms like PPO.

/// Backend-agnostic optimizer abstraction.
///
/// Exposed under both `training` (tch impl available) and `training-burn`
/// (Burn impl available) so the trait surface is the seam phase 3 (#80)
/// plugs Burn loss math into. See `optimizer.rs` for the pattern-choice
/// write-up.
pub mod optimizer;

#[cfg(feature = "training")]
pub mod dqn;
#[cfg(feature = "training")]
pub mod ppo;

#[cfg(feature = "training")]
pub use dqn::{DQNConfig, DQNStepStats, DQNTrainer};
// The optimizer abstraction is available regardless of which backend
// feature is selected. The tch impl (`TchOptimizer`) and Burn impl
// (`BurnOptimizer`) are individually feature-gated inside the module.
pub use optimizer::BackendOptimizer;
#[cfg(feature = "training-burn")]
pub use optimizer::BurnOptimizer;
#[cfg(feature = "training")]
pub use optimizer::TchOptimizer;
#[cfg(feature = "training")]
pub use ppo::{
    AggregatedStats, PPOConfig, PPOTrainer, TrainingStats, compute_gae, compute_policy_loss,
    compute_value_loss, generate_minibatch_indices,
};
