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
