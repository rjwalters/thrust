//! Policy and neural network wrappers
//!
//! This module provides policy implementations using tch-rs for neural
//! networks.

pub mod inference;
pub mod universal_inference;

#[cfg(feature = "training")]
pub mod mlp;

#[cfg(feature = "training")]
pub mod multi_discrete_mlp;

#[cfg(feature = "training")]
pub mod q_network;

#[cfg(feature = "training")]
pub mod snake_cnn;

#[cfg(feature = "training")]
pub use q_network::QNetwork;

// Burn-backend MLP policy — scout for issue #78 / phase 1 of the
// Burn migration. Deliberately minimal (only what the bandit trainer
// needs) and decoupled from the tch `MlpPolicy`; both can coexist.
#[cfg(feature = "training-burn")]
pub mod mlp_burn;
