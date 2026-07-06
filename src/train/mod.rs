//! Training algorithms (Burn backend).
//!
//! Hosts the PPO and DQN trainers, the backend-agnostic optimizer
//! abstraction, and the loss math shared between them. After phase 5 of
//! the Burn migration (#82), Burn is the only tensor backend in the
//! workspace.

/// Burn optimizer wrapper used by both PPO and DQN.
pub mod optimizer;

/// Global gradient-norm clipping shared by the PPO trainers and the joint
/// multi-agent trainer (issues #239, #299).
pub mod grad_clip;

pub mod a2c;
pub mod bc;
pub mod dqn;
pub mod ppo;
pub mod sac;

pub use a2c::{A2cConfig, A2cStats, A2cTrainer, compute_a2c_policy_loss, compute_a2c_value_loss};
pub use bc::{BcConfig, BcEpochStats, BcTrainer, Demonstrations, compute_bc_loss};
pub use dqn::{DQNConfig, DQNStepStatsBurn, DQNTrainerBurn};
pub use grad_clip::{clip_grads_by_global_norm, global_grad_norm};
pub use optimizer::{BackendOptimizer, BurnOptimizer};
pub use ppo::{
    AggregatedStats, AsyncActorLearnerConfig, PPOConfig, PPOTrainerBurn, TrainingStats,
    compute_entropy_loss, compute_policy_loss, compute_value_loss, generate_minibatch_indices,
};
pub use sac::{SacConfig, SacStepStats, SacTrainer};
