//! Bucket Brigade env module.
//!
//! Thrust-native env trait implementation for the Bucket Brigade MARL
//! research env (see <https://github.com/rjwalters/bucket-brigade>).
//!
//! # Submodules
//!
//! - [`env`]: the core [`env::BucketBrigadeMaEnv`] adapter, which wraps
//!   [`bucket_brigade_core::BucketBrigade`] with a flat observation and
//!   multi-discrete action interface. Implements [`crate::env::Environment`] so
//!   the standard Thrust pool/snapshot plumbing works.
//! - [`registry`]: a Rust mirror of `bucket_brigade.envs.registry` (PR #379 on
//!   the bucket-brigade side). Maps frozen versioned scenario IDs (e.g.
//!   `"minimal_specialization-v1"`) to base scenarios in
//!   `bucket_brigade_core::SCENARIOS`. The versioned registry is the
//!   reproducibility surface shared between Python and Rust trainers.
//! - [`multi_agent`] (training-only): the
//!   [`crate::multi_agent::MultiAgentEnvironment`] impl. Gated behind the
//!   `training` feature because that's where the multi-agent trait lives.
//!
//! # Quick start
//!
//! ```no_run
//! use thrust_rl::env::games::bucket_brigade::BucketBrigadeMaEnv;
//!
//! let mut env = BucketBrigadeMaEnv::from_scenario_id(
//!     "minimal_specialization-v1",
//!     None, // use the frozen default (4 agents)
//!     Some(42),
//! )
//! .expect("registered scenario ID");
//! let initial_obs = env.reset(Some(42));
//! ```

pub mod env;
pub mod registry;

#[cfg(feature = "training")]
pub mod multi_agent;

pub use env::{ACTION_DIMS, BucketBrigadeMaEnv, MaStepResult, NUM_HOUSES, SCENARIO_INFO_LEN};
pub use registry::{
    DEFAULT_NUM_AGENTS, default_num_agents_for, get_scenario_by_id, list_versioned_scenarios,
    parse_scenario_id,
};
