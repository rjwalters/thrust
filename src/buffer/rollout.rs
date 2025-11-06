//! Rollout buffer for storing and processing trajectories
//!
//! This module implements experience storage for PPO training, including:
//! - Trajectory storage (observations, actions, rewards, etc.)
//! - Generalized Advantage Estimation (GAE) computation
//! - Efficient sampling for minibatch training
//!
//! # Buffer Layout
//!
//! The buffer uses a `[num_steps, num_envs]` layout where:
//! - `num_steps`: Number of timesteps per rollout (typically 128-2048)
//! - `num_envs`: Number of parallel environments
//!
//! This layout provides good cache locality for forward passes and
//! efficient computation of advantages.

// Re-export main components
pub use storage::{RolloutBuffer, RolloutBatch};
pub use gae::{compute_advantages, compute_mc_returns, compute_nstep_returns, normalize_advantages};
pub use sampling::{generate_minibatch_indices, sample_minibatch, Minibatch, MinibatchIterator, shuffle_indices, train_val_split};

// Submodules
mod storage;
mod gae;
mod sampling;

// Legacy interface - re-export compute_advantages as a method on RolloutBuffer
impl RolloutBuffer {
    /// Compute advantages using Generalized Advantage Estimation
    ///
    /// This is a convenience method that calls the module-level function.
    ///
    /// # Arguments
    /// * `last_values` - Value estimates for the final states [num_envs]
    /// * `gamma` - Discount factor
    /// * `gae_lambda` - GAE lambda parameter
    pub fn compute_advantages(&mut self, last_values: &[f32], gamma: f32, gae_lambda: f32) {
        gae::compute_advantages(self, last_values, gamma, gae_lambda);
    }

    /// Get a batch of all data from the buffer
    ///
    /// This is a convenience method that calls the module-level function.
    pub fn get_batch(&self) -> RolloutBatch {
        RolloutBatch::from_buffer(self)
    }
}
