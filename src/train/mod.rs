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

#[cfg(any(feature = "training", feature = "training-burn"))]
pub mod dqn;
#[cfg(any(feature = "training", feature = "training-burn"))]
pub mod ppo;

/// Burn-backend PPO module (phase 3 of #65). Sibling to the tch
/// [`ppo`] module; both implement the same algorithm but use different
/// tensor backends. Collapses into a single module in phase 5 (#82).
#[cfg(feature = "training-burn")]
pub mod ppo_burn;

/// Burn-backend DQN module (phase 3 of #65). Sibling to the tch
/// [`dqn`] module.
#[cfg(feature = "training-burn")]
pub mod dqn_burn;

/// Numerical-parity tests between the tch and Burn loss
/// implementations. Active only when **both** backends are compiled
/// in. See the module-level doc on
/// [`crate::train::ppo_burn::loss`] / [`crate::train::dqn_burn::loss`]
/// for the 1e-4 tolerance contract.
#[cfg(all(feature = "training", feature = "training-burn", test))]
mod parity_tests;

#[cfg(feature = "training")]
pub use dqn::{DQNConfig, DQNStepStats, DQNTrainer};
#[cfg(feature = "training-burn")]
pub use dqn_burn::{DQNStepStatsBurn, DQNTrainerBurn};
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
#[cfg(feature = "training-burn")]
pub use ppo_burn::PPOTrainerBurn;
