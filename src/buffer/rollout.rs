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
pub use gae::{
    compute_advantages, compute_advantages_multi_agent, compute_advantages_partial,
    compute_mc_returns, compute_nstep_returns, normalize_advantages,
};
pub use sampling::{
    Minibatch, MinibatchIterator, generate_minibatch_indices, sample_minibatch, shuffle_indices,
    train_val_split,
};
pub use storage::{RolloutBatch, RolloutBuffer, RolloutBurnTensors};

// Submodules
mod gae;
mod sampling;
mod storage;

#[cfg(test)]
mod tests;

// Legacy interface - re-export compute_advantages as a method on RolloutBuffer
impl RolloutBuffer {
    /// Compute advantages using Generalized Advantage Estimation
    ///
    /// This is a convenience method that calls the module-level function.
    /// Iterates the full `[num_steps, num_envs]` capacity; use
    /// [`Self::compute_advantages_partial`] when the buffer is only
    /// partially filled.
    ///
    /// # Arguments
    /// * `last_values` - Value estimates for the final states `[num_envs]`
    /// * `gamma` - Discount factor
    /// * `gae_lambda` - GAE lambda parameter
    pub fn compute_advantages(&mut self, last_values: &[f32], gamma: f32, gae_lambda: f32) {
        gae::compute_advantages(self, last_values, gamma, gae_lambda);
    }

    /// Compute advantages over the first `valid_steps` rows only.
    ///
    /// Convenience wrapper around the module-level
    /// [`compute_advantages_partial`]. Use this when the rollout buffer
    /// has been partially filled (`valid_steps < num_steps`) to prevent
    /// zero-padded tail rows from contaminating GAE on the real prefix.
    ///
    /// # Arguments
    /// * `valid_steps` - Number of filled rows at the start of the buffer
    /// * `last_values` - Bootstrap `V(s_{T+1})` for the state after row
    ///   `valid_steps - 1`, per environment `[num_envs]`
    /// * `gamma` - Discount factor
    /// * `gae_lambda` - GAE lambda parameter
    pub fn compute_advantages_partial(
        &mut self,
        valid_steps: usize,
        last_values: &[f32],
        gamma: f32,
        gae_lambda: f32,
    ) {
        gae::compute_advantages_partial(self, valid_steps, last_values, gamma, gae_lambda);
    }

    /// Get a batch of all data from the buffer
    ///
    /// This is a convenience method that calls the module-level function.
    /// Returns the full `[num_steps, num_envs]` capacity, including any
    /// zero-padded unwritten tail; use [`Self::get_filled_batch`] when
    /// the buffer is only partially filled.
    pub fn get_batch(&self) -> RolloutBatch {
        RolloutBatch::from_buffer(self)
    }

    /// Get a batch covering only the first `valid_steps` rows.
    ///
    /// Use this when the buffer was filled with fewer than `num_steps`
    /// transitions (e.g. an early-terminating rollout). It prevents the
    /// zero-initialized tail of the buffer from being handed to PPO as
    /// fake training data.
    ///
    /// # Panics
    /// Panics if `valid_steps > self.shape().0`.
    pub fn get_filled_batch(&self, valid_steps: usize) -> RolloutBatch {
        RolloutBatch::from_buffer_partial(self, valid_steps)
    }
}
