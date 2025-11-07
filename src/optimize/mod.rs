//! Hyperparameter Optimization Framework
//!
//! This module provides a flexible, high-performance hyperparameter
//! optimization system for RL training. It supports:
//!
//! - Bayesian optimization with Gaussian Processes
//! - Multi-objective optimization (Pareto frontiers)
//! - Parallel trial evaluation
//! - Mixed continuous/discrete parameter spaces
//! - Checkpointing and resumption
//!
//! # Example
//!
//! ```rust,no_run
//! use thrust_rl::optimize::*;
//!
//! // Define search space
//! let space = SearchSpace::new()
//!     .add_continuous("learning_rate", 1e-4, 1e-3, true) // log scale
//!     .add_discrete("n_steps", vec![64, 128, 256, 512])
//!     .add_discrete("hidden_dim", vec![64, 128, 256]);
//!
//! // Define objectives (maximize performance, minimize time)
//! let objectives = vec![Objective::Maximize("performance"), Objective::Minimize("training_time")];
//!
//! // Run optimization
//! let optimizer = BayesianOptimizer::new(space, objectives);
//! // ... run trials
//! ```

pub mod bayesian;
pub mod pareto;
pub mod scheduler;
pub mod space;
pub mod trial;

pub use bayesian::BayesianOptimizer;
pub use pareto::ParetoFrontier;
pub use scheduler::TrialScheduler;
pub use space::{Parameter, ParameterValue, SearchSpace};
pub use trial::{Objective, Trial, TrialResult};
