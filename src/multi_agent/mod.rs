//! Multi-agent training infrastructure for Thrust
//!
//! This module provides two complementary multi-agent training architectures:
//!
//! ## 1. Per-thread independent learners ([`crate::multi_agent::learner::PolicyLearner`], [`crate::multi_agent::population::Population`], [`crate::multi_agent::simulator::GameSimulator`])
//!
//! Each agent runs in its own learner thread and updates from a shared rollout
//! pool. Good for league play, self-play, evolutionary tournaments, and any
//! setup where agents update *independently* without needing batched access
//! to one another's parameters.
//!
//! - **Population**: Collection of diverse agent policies
//! - **GameSimulator**: Thread that runs parallel games and routes experiences
//! - **PolicyLearner**: Per-agent training thread with PPO
//! - **Matchmaker**: Strategy for assigning agents to games
//!
//! ## 2. Synchronized joint trainer ([`crate::multi_agent::joint::JointMultiAgentTrainer`])
//!
//! A single process owns `N` policies and `N` optimizers and runs them in
//! lockstep on a shared rollout buffer. Required when the loss function
//! contains a term that depends on **all** agents' parameters evaluated on
//! the **same** minibatch (e.g. cross-agent representational regularizers
//! in the Slepian-Wolf MARL P3 experiments). One `.backward()` couples
//! every agent's encoder through the auxiliary term while leaving per-agent
//! gradient updates isolated (each optimizer reads only its own var-store).
//!
//! See [`crate::multi_agent::joint`] for the detailed semantics and acceptance
//! criteria.

pub mod centralized_critic;
pub mod environment;
pub mod joint;
pub mod learner;
pub mod matchmaking;
pub mod messages;
pub mod population;
pub mod simulator;

pub use centralized_critic::{
    CentralizedCritic, CentralizedCriticConfig, compute_centralized_value_loss,
};
pub use environment::{MultiAgentEnvironment, MultiAgentResult};
pub use joint::{
    JointEnv, JointMultiAgentTrainer, JointPolicy, JointRollout, JointStats, JointStepResult,
    JointTrainerConfig,
};
pub use learner::PolicyLearner;
pub use matchmaking::{Matchmaker, MatchmakingStrategy};
pub use messages::{ControlMessage, Experience, PolicyUpdate, TrainingStats};
pub use population::{Agent, AgentId, LearningMode, Population, PopulationConfig};
pub use simulator::GameSimulator;
