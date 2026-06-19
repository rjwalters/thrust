//! Synchronous Advantage Actor-Critic (A2C) trainer.
//!
//! A2C is the synchronous variant of A3C (Mnih et al. 2016,
//! "Asynchronous Methods for Deep Reinforcement Learning",
//! arXiv:1602.01783): a shared actor-critic network collects short
//! `n_steps`-long rollouts across `num_envs` parallel actors and performs
//! a single gradient update per rollout. It lands as a sibling module to
//! [`crate::train::ppo`], reusing the same policy/optimizer/GAE infra with
//! a simpler single-update-per-rollout loop.
//!
//! This module is built incrementally by the A2C decomposition (#150):
//! - [`A2cConfig`] — hyperparameters (builder + `validate()`), mirroring
//!   [`crate::train::ppo::PPOConfig`] (#151).
//! - [`loss`] — the un-clipped policy-gradient + plain-MSE value loss
//!   ([`compute_a2c_policy_loss`], [`compute_a2c_value_loss`]) and the
//!   [`A2cTrainer`] single-update-per-rollout loop (#152).

pub mod config;
pub mod loss;
pub mod trainer;

pub use config::A2cConfig;
pub use loss::{compute_a2c_policy_loss, compute_a2c_value_loss};
pub use trainer::{A2cStats, A2cTrainer};
