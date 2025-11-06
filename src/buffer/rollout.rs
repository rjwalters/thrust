//! Rollout buffer for storing and processing trajectories
//!
//! This module implements experience storage for PPO training, including:
//! - Trajectory storage (observations, actions, rewards, etc.)
//! - GAE (Generalized Advantage Estimation) computation
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

/// Rollout buffer for PPO training
///
/// Stores trajectories collected from environment interactions and
/// computes advantages using Generalized Advantage Estimation (GAE).
///
/// # Example
///
/// ```rust
/// use thrust_rl::buffer::rollout::RolloutBuffer;
///
/// // Create buffer for 128 steps across 4 environments with 4D observations
/// let mut buffer = RolloutBuffer::new(128, 4, 4);
///
/// // Store a transition
/// buffer.add(
///     0,                        // step
///     0,                        // env_id
///     vec![0.1, 0.2, 0.3, 0.4], // observation
///     0,                        // action
///     1.0,                      // reward
///     0.5,                      // value estimate
///     -0.1,                     // log probability
///     false,                    // terminated
///     false,                    // truncated
/// );
/// ```
#[derive(Debug)]
pub struct RolloutBuffer {
    /// Number of steps per rollout
    num_steps: usize,

    /// Number of parallel environments
    num_envs: usize,

    /// Observation dimensionality
    obs_dim: usize,

    /// Observations: [num_steps, num_envs, obs_dim]
    observations: Vec<Vec<Vec<f32>>>,

    /// Actions: [num_steps, num_envs]
    actions: Vec<Vec<i64>>,

    /// Rewards: [num_steps, num_envs]
    rewards: Vec<Vec<f32>>,

    /// Value estimates: [num_steps, num_envs]
    values: Vec<Vec<f32>>,

    /// Log probabilities: [num_steps, num_envs]
    log_probs: Vec<Vec<f32>>,

    /// Episode termination flags: [num_steps, num_envs]
    terminated: Vec<Vec<bool>>,

    /// Episode truncation flags: [num_steps, num_envs]
    truncated: Vec<Vec<bool>>,

    /// Computed advantages: [num_steps, num_envs]
    advantages: Vec<Vec<f32>>,

    /// Computed returns: [num_steps, num_envs]
    returns: Vec<Vec<f32>>,

    /// Current position in buffer
    pos: usize,

    /// Whether advantages have been computed
    advantages_computed: bool,
}

impl RolloutBuffer {
    /// Create a new rollout buffer
    ///
    /// # Arguments
    ///
    /// * `num_steps` - Number of timesteps per rollout
    /// * `num_envs` - Number of parallel environments
    /// * `obs_dim` - Dimensionality of observations
    pub fn new(num_steps: usize, num_envs: usize, obs_dim: usize) -> Self {
        // Pre-allocate all buffers
        let observations = vec![vec![vec![0.0; obs_dim]; num_envs]; num_steps];
        let actions = vec![vec![0; num_envs]; num_steps];
        let rewards = vec![vec![0.0; num_envs]; num_steps];
        let values = vec![vec![0.0; num_envs]; num_steps];
        let log_probs = vec![vec![0.0; num_envs]; num_steps];
        let terminated = vec![vec![false; num_envs]; num_steps];
        let truncated = vec![vec![false; num_envs]; num_steps];
        let advantages = vec![vec![0.0; num_envs]; num_steps];
        let returns = vec![vec![0.0; num_envs]; num_steps];

        Self {
            num_steps,
            num_envs,
            obs_dim,
            observations,
            actions,
            rewards,
            values,
            log_probs,
            terminated,
            truncated,
            advantages,
            returns,
            pos: 0,
            advantages_computed: false,
        }
    }

    /// Add a transition to the buffer
    ///
    /// # Arguments
    ///
    /// * `step` - Current step index (0 to num_steps-1)
    /// * `env_id` - Environment index (0 to num_envs-1)
    /// * `obs` - Observation vector
    /// * `action` - Action taken
    /// * `reward` - Reward received
    /// * `value` - Value estimate from policy
    /// * `log_prob` - Log probability of action
    /// * `term` - Whether episode terminated
    /// * `trunc` - Whether episode was truncated
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        step: usize,
        env_id: usize,
        obs: Vec<f32>,
        action: i64,
        reward: f32,
        value: f32,
        log_prob: f32,
        term: bool,
        trunc: bool,
    ) {
        assert!(step < self.num_steps, "Step index out of bounds");
        assert!(env_id < self.num_envs, "Environment index out of bounds");
        assert_eq!(obs.len(), self.obs_dim, "Observation dimension mismatch");

        self.observations[step][env_id] = obs;
        self.actions[step][env_id] = action;
        self.rewards[step][env_id] = reward;
        self.values[step][env_id] = value;
        self.log_probs[step][env_id] = log_prob;
        self.terminated[step][env_id] = term;
        self.truncated[step][env_id] = trunc;

        // Mark advantages as needing recomputation
        self.advantages_computed = false;
    }

    /// Compute advantages using Generalized Advantage Estimation (GAE)
    ///
    /// Implements the GAE algorithm from "High-Dimensional Continuous Control
    /// Using Generalized Advantage Estimation" (Schulman et al., 2016).
    ///
    /// The advantage function is computed as:
    /// ```text
    /// δ_t = r_t + γ * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
    /// A_t = δ_t + γ * λ * A_{t+1} * (1 - done_{t+1})
    /// ```
    ///
    /// # Arguments
    ///
    /// * `last_values` - Value estimates for states after final step (bootstrap
    ///   values)
    /// * `gamma` - Discount factor (typically 0.99)
    /// * `gae_lambda` - GAE lambda parameter (typically 0.95)
    pub fn compute_advantages(&mut self, last_values: &[f32], gamma: f32, gae_lambda: f32) {
        assert_eq!(
            last_values.len(),
            self.num_envs,
            "Last values must match number of environments"
        );

        // Compute advantages for each environment independently
        for (env_id, &last_value) in last_values.iter().enumerate().take(self.num_envs) {
            let mut last_gae = 0.0;

            // Backward iteration through trajectory
            for step in (0..self.num_steps).rev() {
                // Determine next value (either next step or bootstrap value)
                let next_value = if step == self.num_steps - 1 {
                    last_value
                } else {
                    self.values[step + 1][env_id]
                };

                // Check if next state is terminal (use next step's flags if available)
                let next_done = if step == self.num_steps - 1 {
                    false // Bootstrap value is for continuing state
                } else {
                    self.terminated[step + 1][env_id] || self.truncated[step + 1][env_id]
                };

                let next_non_terminal = if next_done { 0.0 } else { 1.0 };

                // Compute TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
                let delta = self.rewards[step][env_id] + gamma * next_value * next_non_terminal
                    - self.values[step][env_id];

                // Compute GAE: A_t = δ_t + γ * λ * A_{t+1}
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae;

                self.advantages[step][env_id] = last_gae;

                // Returns are advantages + values
                self.returns[step][env_id] = last_gae + self.values[step][env_id];
            }
        }

        self.advantages_computed = true;
    }

    /// Get a flattened batch of all data for training
    ///
    /// Returns observations, actions, values, log_probs, advantages, and
    /// returns flattened into single vectors of length `num_steps *
    /// num_envs`.
    ///
    /// # Panics
    ///
    /// Panics if advantages haven't been computed yet. Call
    /// `compute_advantages` first.
    pub fn get_batch(&self) -> RolloutBatch {
        assert!(self.advantages_computed, "Must compute advantages before getting batch");

        let total_size = self.num_steps * self.num_envs;

        let mut observations = Vec::with_capacity(total_size);
        let mut actions = Vec::with_capacity(total_size);
        let mut values = Vec::with_capacity(total_size);
        let mut log_probs = Vec::with_capacity(total_size);
        let mut advantages = Vec::with_capacity(total_size);
        let mut returns = Vec::with_capacity(total_size);

        // Flatten [steps, envs] into single vectors
        for step in 0..self.num_steps {
            for env_id in 0..self.num_envs {
                observations.push(self.observations[step][env_id].clone());
                actions.push(self.actions[step][env_id]);
                values.push(self.values[step][env_id]);
                log_probs.push(self.log_probs[step][env_id]);
                advantages.push(self.advantages[step][env_id]);
                returns.push(self.returns[step][env_id]);
            }
        }

        RolloutBatch { observations, actions, values, log_probs, advantages, returns }
    }

    /// Reset buffer for new rollout collection
    pub fn reset(&mut self) {
        self.pos = 0;
        self.advantages_computed = false;
    }

    /// Get buffer dimensions
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.num_steps, self.num_envs, self.obs_dim)
    }

    /// Get number of samples currently in buffer
    pub fn len(&self) -> usize {
        self.pos * self.num_envs
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }
}

/// A batch of rollout data ready for training
#[derive(Debug, Clone)]
pub struct RolloutBatch {
    /// Observations: \[batch_size, obs_dim\]
    pub observations: Vec<Vec<f32>>,

    /// Actions: \[batch_size\]
    pub actions: Vec<i64>,

    /// Old value estimates: \[batch_size\]
    pub values: Vec<f32>,

    /// Old log probabilities: \[batch_size\]
    pub log_probs: Vec<f32>,

    /// Computed advantages: \[batch_size\]
    pub advantages: Vec<f32>,

    /// Computed returns: \[batch_size\]
    pub returns: Vec<f32>,
}

impl RolloutBatch {
    /// Get batch size
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let buffer = RolloutBuffer::new(128, 4, 4);
        assert_eq!(buffer.shape(), (128, 4, 4));
    }

    #[test]
    fn test_add_transition() {
        let mut buffer = RolloutBuffer::new(10, 2, 4);

        buffer.add(0, 0, vec![1.0, 2.0, 3.0, 4.0], 1, 1.0, 0.5, -0.693, false, false);

        assert_eq!(buffer.observations[0][0], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(buffer.actions[0][0], 1);
        assert_eq!(buffer.rewards[0][0], 1.0);
    }

    #[test]
    fn test_compute_advantages() {
        let mut buffer = RolloutBuffer::new(4, 1, 2);

        // Add a simple trajectory
        for step in 0..4 {
            buffer.add(
                step,
                0,
                vec![0.0, 0.0],
                0,
                1.0, // constant reward
                0.0, // zero value estimates for simplicity
                0.0,
                false,
                false,
            );
        }

        let last_values = vec![0.0];
        buffer.compute_advantages(&last_values, 0.99, 0.95);

        assert!(buffer.advantages_computed);

        // With constant rewards and zero values, advantages should be positive
        for step in 0..4 {
            assert!(buffer.advantages[step][0] > 0.0);
        }
    }

    #[test]
    fn test_gae_with_terminal() {
        let mut buffer = RolloutBuffer::new(3, 1, 2);

        // Trajectory that terminates in the middle
        buffer.add(0, 0, vec![0.0, 0.0], 0, 1.0, 0.0, 0.0, false, false);
        buffer.add(1, 0, vec![0.0, 0.0], 0, 1.0, 0.0, 0.0, true, false); // Terminal
        buffer.add(2, 0, vec![0.0, 0.0], 0, 1.0, 0.0, 0.0, false, false);

        let last_values = vec![0.0];
        buffer.compute_advantages(&last_values, 0.99, 0.95);

        // Advantage at step 1 should not propagate to step 0 due to terminal
        assert!(buffer.advantages[1][0] > 0.0);
    }

    #[test]
    fn test_get_batch() {
        let mut buffer = RolloutBuffer::new(2, 2, 3);

        // Fill buffer with test data
        for step in 0..2 {
            for env in 0..2 {
                buffer.add(
                    step,
                    env,
                    vec![step as f32, env as f32, 1.0],
                    step as i64,
                    1.0,
                    0.5,
                    -0.1,
                    false,
                    false,
                );
            }
        }

        let last_values = vec![0.0, 0.0];
        buffer.compute_advantages(&last_values, 0.99, 0.95);

        let batch = buffer.get_batch();

        assert_eq!(batch.len(), 4); // 2 steps * 2 envs
        assert_eq!(batch.observations.len(), 4);
        assert_eq!(batch.actions.len(), 4);
    }

    #[test]
    #[should_panic(expected = "Must compute advantages before getting batch")]
    fn test_get_batch_without_advantages() {
        let buffer = RolloutBuffer::new(2, 2, 3);
        buffer.get_batch(); // Should panic
    }

    #[test]
    fn test_buffer_reset() {
        let mut buffer = RolloutBuffer::new(10, 2, 4);
        buffer.add(0, 0, vec![1.0, 2.0, 3.0, 4.0], 1, 1.0, 0.5, -0.1, false, false);

        buffer.reset();
        assert_eq!(buffer.pos, 0);
        assert!(!buffer.advantages_computed);
    }

    #[test]
    fn test_multiple_environments() {
        let mut buffer = RolloutBuffer::new(5, 3, 2);

        // Add data for 3 environments
        for step in 0..5 {
            for env in 0..3 {
                buffer.add(
                    step,
                    env,
                    vec![step as f32, env as f32],
                    env as i64,
                    1.0,
                    0.0,
                    0.0,
                    false,
                    false,
                );
            }
        }

        let last_values = vec![0.0, 0.0, 0.0];
        buffer.compute_advantages(&last_values, 0.99, 0.95);

        let batch = buffer.get_batch();
        assert_eq!(batch.len(), 15); // 5 steps * 3 envs
    }
}
