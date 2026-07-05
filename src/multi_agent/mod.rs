//! Multi-agent training infrastructure for Thrust (Burn backend).
//!
//! PR #98 (Burn phase 5) deleted the previous `tch`-coupled `multi_agent`
//! module wholesale. This module is the Burn-native rebuild called out by
//! issue #100, sitting on top of the Burn policy networks
//! ([`crate::policy::mlp::MlpBurnPolicy`],
//! [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpBurnPolicy`]) and the
//! backend-agnostic [`crate::train::optimizer::BurnOptimizer`].
//!
//! # Scope of the initial Burn port
//!
//! This first cut restores the pieces of the multi-agent surface that the
//! resurrected trainer examples (`train_snake_multi_v2`,
//! `train_pong_self_play`, the Slepian-Wolf bucket-brigade experiments)
//! and the multi-discrete integration tests will need:
//!
//! - [`crate::multi_agent::environment::MultiAgentEnvironment`] — extension to
//!   the base [`crate::env::Environment`] trait for per-agent observation /
//!   action / reward routing. No tensor dependency.
//! - [`crate::multi_agent::messages`] — Burn-native experience / policy-update
//!   / control payload types. Pre-Burn these carried `tch::Tensor`; post-Burn
//!   they carry plain `Vec<f32>` host buffers so the producer and consumer can
//!   pick different Burn backends.
//! - [`crate::multi_agent::joint::JointMultiAgentTrainer`] — synchronized joint
//!   trainer used when a loss term depends on **all** agents' parameters
//!   evaluated on the **same minibatch** (the Slepian-Wolf cross-agent
//!   redundancy penalty is the canonical motivation). See the module doc for
//!   the Burn-specific single-graph-multiple-optimizer pattern.
//!
//! # What's intentionally NOT here yet
//!
//! The pre-Burn module also shipped:
//!
//! - `learner.rs` / `population.rs` / `simulator.rs` / `matchmaking.rs` —
//!   per-thread independent learners with shared experience channels. The
//!   threading layer is orthogonal to the Burn migration but its old
//!   implementation was tightly coupled to `tch`'s shared `VarStore` model; a
//!   Burn-native port should design around Burn's move-through optimizer
//!   semantics and is left to a follow-up.
//! - `centralized_critic.rs` — optional centralized critic for MAPPO-style
//!   training. The Burn analog already lives at [`crate::train::optimizer`] one
//!   level up; threading it through the joint trainer is a small follow-up
//!   extension.
//!
//! Issue #100's definition of done only requires the joint trainer plus
//! its synthetic-env smoke test; those follow-ups are explicitly out of
//! scope for the first port.

/// Bucket-brigade reference baseline policies (specialist, cell enum).
#[cfg(feature = "env-bucket-brigade")]
pub mod bucket_brigade_baselines;
/// Bucket-brigade-specific scoring metrics (`gap_closed`).
#[cfg(feature = "env-bucket-brigade")]
pub mod bucket_brigade_metrics;
/// Bucket-brigade best-response improvability oracle (issue #259).
#[cfg(feature = "env-bucket-brigade")]
pub mod bucket_brigade_oracle;
/// Fixed-vocab agent-to-agent communication (comms) surface (issue #274).
pub mod comms;
/// Multi-agent environment trait.
pub mod environment;
/// Synchronized joint multi-agent PPO trainer.
pub mod joint;
/// Cross-thread message payloads.
pub mod messages;
/// Neural Fictitious Self-Play (NFSP) trainer (issue #106).
pub mod nfsp;
/// Policy-Space Response Oracles (PSRO) meta-game trainer (issue #107).
pub mod psro;

pub use comms::{AgentMessage, CommunicatingEnvironment, Delivery};
pub use environment::{MultiAgentEnvironment, MultiAgentResult};
pub use joint::{
    JointEnv, JointMultiAgentTrainer, JointPolicy, JointRollout, JointStats, JointStepResult,
    JointTrainerConfig,
};
pub use messages::{
    AgentId, ControlMessage, Experience, PolicyBroadcast, PolicyUpdate, TrainingStats,
};
pub use nfsp::{NfspConfig, NfspIterationStats, NfspStats, NfspTrainer, ReservoirBuffer};
pub use psro::{
    AlphaRankMetaSolver, FictitiousPlayMetaSolver, MetaSolver, PayoffCache, PsroConfig,
    PsroIterationStats, PsroStats, PsroTrainer, ReplicatorDynamicsMetaSolver, UniformMetaSolver,
};
