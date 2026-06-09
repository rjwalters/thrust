//! Burn-backend PPO module (phase 3 of the Burn migration, #80).
//!
//! Sibling to [`crate::train::ppo`] (tch path). The two modules expose
//! deliberately parallel APIs so phase 5 (#82) can collapse them into
//! one when the tch path is dropped.
//!
//! # Contents
//!
//! - [`loss`] — backend-generic PPO loss math (policy/value/entropy).
//! - [`trainer`] — `PPOTrainerBurn<B, P, O>` that owns the policy module
//!   (Burn's optimizer-consumes-module ownership model) and exposes a
//!   `train_step` that runs the same surrogate-loss / gradient-step /
//!   KL-early-stop logic as the tch trainer.
//!
//! # Numerical parity
//!
//! Identical inputs to the tch PPO loss surface must produce the same
//! scalar values to **1e-4 absolute tolerance** (DoD on issue #80).
//! That is enforced by the integration parity tests in
//! [`crate::train::parity_tests`] (active only when both `training` and
//! `training-burn` features are enabled).
//!
//! # Why a separate module
//!
//! The pre-phase-3 `train/ppo` module is gated on `feature = "training"`
//! and pulls in tch types at the top of every file. Putting the Burn
//! port under `train/ppo` would mean either:
//!
//! 1. Loosening the feature gate on the whole tch module, which forces every
//!    tch-gated file to also be valid under `training-burn` alone; or
//! 2. Pervasive cfg-attr scaffolding inside each file.
//!
//! Both are noisy; the parallel-sibling pattern (`ppo` for tch,
//! `ppo_burn` for Burn) is the same pattern phase 2b used for the
//! optimizer abstraction (`TchOptimizer` / `BurnOptimizer`) and the
//! same one the policy module already adopted with `mlp.rs` /
//! `mlp_burn.rs`. Phase 5 collapses both.

pub mod loss;
pub mod trainer;

pub use loss::{compute_entropy_loss, compute_policy_loss, compute_value_loss, scalar_f64};
pub use trainer::PPOTrainerBurn;
