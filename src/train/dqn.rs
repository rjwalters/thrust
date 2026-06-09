//! DQN trainer (Burn backend).
//!
//! After phase 5 of the Burn migration (#82), Burn is the only tensor
//! backend in the workspace. The historical `train::dqn_burn` parallel
//! sibling has been collapsed back into `train::dqn`. See the parallel
//! `crate::train::ppo` module for the same rationale.
//!
//! # Contents
//!
//! - [`config`] — `DQNConfig` hyperparameters / builder API.
//! - [`loss`] — backend-generic DQN / Double-DQN loss math.
//! - [`trainer`] — `DQNTrainerBurn<B, Q, O>` that owns the online Q-network
//!   module (Burn's optimizer-consumes-module ownership model) and exposes a
//!   `train_step` that runs Smooth-L1 loss / gradient-step logic.

pub mod config;
pub mod loss;
pub mod trainer;

pub use config::DQNConfig;
pub use loss::{
    compute_dqn_loss, compute_dqn_loss_double, compute_loss, compute_td_target,
    compute_td_target_double, gather_action_q, huber_per_sample,
};
pub use trainer::{DQNStepStatsBurn, DQNTrainerBurn};
