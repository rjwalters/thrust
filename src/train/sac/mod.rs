//! Soft Actor-Critic (SAC) trainer for continuous control.
//!
//! This module is the integration capstone of the SAC decomposition
//! (#136): it assembles the four building blocks —
//! [`SacActor`](crate::policy::sac_actor::SacActor),
//! [`ContinuousQNetwork`](crate::policy::continuous_q::ContinuousQNetwork),
//! [`ContinuousReplayBuffer`](crate::buffer::replay::ContinuousReplayBuffer),
//! and the [`PendulumSwingUp`](crate::env::games::pendulum::PendulumSwingUp)
//! env — into a working SAC trainer following Haarnoja et al. 2018 v2
//! (arXiv:1812.05905).
//!
//! - [`SacConfig`] — hyperparameters (builder + `validate()`), mirroring
//!   [`crate::train::dqn::DQNConfig`].
//! - [`SacTrainer`] — twin critics + targets, stochastic actor, automatic
//!   entropy-temperature tuning, replay buffer, and Polyak target updates,
//!   driven by three independent Adam optimizers.

mod config;
mod trainer;

pub use config::SacConfig;
pub use trainer::{LogAlpha, SacStepStats, SacTrainer};
