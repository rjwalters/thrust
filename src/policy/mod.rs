//! Policy and neural network wrappers
//!
//! This module provides policy implementations using tch-rs for neural
//! networks.

pub mod inference;
pub mod universal_inference;

#[cfg(feature = "training")]
pub mod mlp;

#[cfg(feature = "training")]
pub mod snake_cnn;
