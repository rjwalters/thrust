//! Multi-agent training infrastructure for Thrust
//!
//! This module provides components for training multiple agents simultaneously
//! in cooperative, competitive, and mixed-motive scenarios.
//!
//! # Architecture
//!
//! The multi-agent system consists of:
//! - **Population**: Collection of diverse agent policies
//! - **GameSimulator**: Thread that runs parallel games and routes experiences
//! - **PolicyLearner**: Per-agent training thread with PPO
//! - **Matchmaker**: Strategy for assigning agents to games
//!
//! # Example
//!
//! ```rust,no_run
//! use thrust_rl::multi_agent::*;
//!
//! let config = PopulationConfig {
//!     size: 8,
//!     matchmaking: MatchmakingStrategy::RoundRobin,
//!     learning_mode: LearningMode::OnPolicy,
//!     update_interval: 100,
//! };
//!
//! let trainer = MultiAgentTrainer::new(
//!     config,
//!     Box::new(|| create_environment()),
//!     num_games: 64,
//! );
//!
//! trainer.train()?;
//! ```

pub mod environment;
pub mod learner;
pub mod matchmaking;
pub mod messages;
pub mod population;
pub mod simulator;

pub use environment::{MultiAgentEnvironment, MultiAgentResult};
pub use learner::PolicyLearner;
pub use matchmaking::{Matchmaker, MatchmakingStrategy};
pub use messages::{ControlMessage, Experience, PolicyUpdate, TrainingStats};
pub use population::{Agent, AgentId, LearningMode, Population, PopulationConfig};
pub use simulator::GameSimulator;
