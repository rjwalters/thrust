//! Behavioral Cloning (BC) — supervised imitation learning.
//!
//! Behavioral cloning trains a policy to reproduce expert actions from a
//! fixed dataset of `(observation, action)` pairs via supervised
//! cross-entropy. It lands as a sibling module to [`crate::train::a2c`],
//! reusing the same policy/optimizer infrastructure with a plain supervised
//! epoch loop in place of an environment-interaction loop.
//!
//! This module is built incrementally by the BC decomposition (#161):
//! - [`BcConfig`] — supervised hyperparameters (builder + `validate()`),
//!   mirroring [`crate::train::a2c::A2cConfig`] (#164).
//! - [`Demonstrations`] — the fixed expert dataset serving seeded minibatches
//!   (#164).
//! - The cross-entropy loss and `BcTrainer` epoch loop land in #167.

pub mod config;
pub mod dataset;

pub use config::BcConfig;
pub use dataset::Demonstrations;
