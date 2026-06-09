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
// Burn migration. Phase 4 (#81) productionized this to mirror the full
// `MlpPolicy` surface (orthogonal init, 2/3-layer, encoder tap), and
// added the sibling modules below for the other three policy networks
// the training stack uses. The tch and Burn variants coexist until
// phase 5 (#82) deletes the tch path.
#[cfg(feature = "training-burn")]
pub mod mlp_burn;

/// Burn-backend multi-discrete MLP — phase 4 sibling of
/// [`multi_discrete_mlp`]. Used by environments like Bucket Brigade
/// and Snake/Pong's multi-discrete heads.
#[cfg(feature = "training-burn")]
pub mod multi_discrete_mlp_burn;

/// Burn-backend Snake CNN — phase 4 sibling of [`snake_cnn`]. Same
/// 3-conv + 2-fc topology.
#[cfg(feature = "training-burn")]
pub mod snake_cnn_burn;

/// Burn-backend Q-Network — phase 4 sibling of [`q_network`]. Same
/// MLP backbone as `MlpPolicy` but with a single Q-head (no value
/// head); DQN target-net sync uses Burn's record-based clone instead
/// of `VarStore::copy`.
#[cfg(feature = "training-burn")]
pub mod q_network_burn;
