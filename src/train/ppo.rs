//! PPO trainer (Burn backend).
//!
//! After phase 5 of the Burn migration (#82), Burn is the only tensor
//! backend in the workspace. The historical `train::ppo_burn` parallel
//! sibling has been collapsed back into `train::ppo`. The trainer struct
//! retains its `PPOTrainerBurn<B, P, O>` name because Burn's
//! optimizer-consumes-module ownership model is structurally different
//! from a hypothetical in-place trainer.
//!
//! # Contents
//!
//! - [`actor_learner`] — single-host asynchronous actor-learner runner (N
//!   inference-only actor threads feeding one learner over `crossbeam-channel`;
//!   Phase 2 of the distributed-training epic #265).
//! - [`config`] — `PPOConfig` hyperparameters / builder API.
//! - [`stats`] — `TrainingStats` / `AggregatedStats` per-update metrics.
//! - [`loss`] — backend-generic PPO loss math (policy/value/entropy) and the
//!   `generate_minibatch_indices` helper.
//! - [`trainer`] — `PPOTrainerBurn<B, P, O>` that owns the policy module
//!   (Burn's optimizer-consumes-module ownership model) and exposes a
//!   `train_step` that runs the surrogate-loss / gradient-step / KL early stop
//!   logic.

pub mod actor_learner;
pub mod config;
pub mod loss;
pub mod stats;
pub mod trainer;

pub use actor_learner::{
    ActorChannels, ActorHandle, ActorStats, AsyncActorLearnerConfig, LearnerReport, actor_thread,
    learner_loop, load_policy_from_broadcast, serialize_policy, spawn_actor,
};
pub use config::PPOConfig;
pub use loss::{
    compute_entropy_loss, compute_policy_loss, compute_value_loss, generate_minibatch_indices,
    scalar_f64,
};
pub use stats::{AggregatedStats, TrainingStats};
pub use trainer::PPOTrainerBurn;
