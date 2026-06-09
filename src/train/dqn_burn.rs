//! Burn-backend DQN module (phase 3 of the Burn migration, #80).
//!
//! Sibling to [`crate::train::dqn`] (tch path). The two modules expose
//! deliberately parallel APIs so phase 5 (#82) can collapse them into
//! one when the tch path is dropped.
//!
//! # Contents
//!
//! - [`crate::train::dqn_burn::loss`] — backend-generic DQN / Double-DQN loss
//!   math.
//! - [`crate::train::dqn_burn::trainer`] — `DQNTrainerBurn<B, Q, O>` that owns
//!   the online Q-network module (Burn's optimizer-consumes-module ownership
//!   model) and exposes a `train_step` that runs Smooth-L1 loss /
//!   gradient-step logic.
//!
//! # Why a separate module
//!
//! Same reasoning as `crate::train::ppo_burn`: the pre-phase-3
//! `train/dqn` module is gated on `feature = "training"` and pulls
//! tch in at the top of every file. The parallel-sibling pattern keeps
//! both backends cleanly separated; phase 5 collapses them when the
//! tch path is dropped.

pub mod loss;
pub mod trainer;

pub use loss::{
    compute_dqn_loss, compute_dqn_loss_double, compute_loss, compute_td_target,
    compute_td_target_double, gather_action_q, huber_per_sample,
};
pub use trainer::{DQNStepStatsBurn, DQNTrainerBurn};
