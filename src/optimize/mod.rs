//! Hyperparameter Optimization Framework
//!
//! Currently exposes the data shapes used by a search-driven training loop
//! (`SearchSpace`, `Parameter`, `Trial`, `Objective`, ...). The intended
//! optimizers (Bayesian / Pareto frontier / parallel scheduler) are not yet
//! implemented — see the TODOs in `bayesian`, `pareto`, and `scheduler`.

// Training-feature modules still owe public docs; tracked as the
// `--features training` follow-up to #33. Re-enable as `#![warn(missing_docs)]`
// once those items are documented.
#![allow(missing_docs)]

pub mod bayesian;
pub mod pareto;
pub mod scheduler;
pub mod space;
pub mod trial;

pub use space::{Parameter, ParameterValue, SearchSpace};
pub use trial::{Objective, Trial, TrialResult};
