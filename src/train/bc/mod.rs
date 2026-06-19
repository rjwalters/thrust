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
//! - [`compute_bc_loss`] — the supervised cross-entropy loss, and [`BcTrainer`]
//!   — the supervised epoch loop over the dataset (#167).

pub mod config;
pub mod dataset;
pub mod loss;
pub mod trainer;

pub use config::BcConfig;
pub use dataset::Demonstrations;
pub use loss::compute_bc_loss;
pub use trainer::{BcEpochStats, BcTrainer};
