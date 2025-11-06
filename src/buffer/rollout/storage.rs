//! Rollout buffer storage and data management
//!
//! This module handles the core storage functionality for rollout buffers,
//! including data insertion, retrieval, and buffer management.

/// Rollout buffer for storing trajectories
///
/// Stores trajectories collected from environment interactions and
/// computes advantages using Generalized Advantage Estimation (GAE).
///
/// # Buffer Layout
///
/// The buffer uses a `[num_steps, num_envs]` layout where:
/// - `num_steps`: Number of timesteps per rollout (typically 128-2048)
/// - `num_envs`: Number of parallel environments
///
/// This layout provides good cache locality for forward passes and
/// efficient computation of advantages.
#[derive(Debug, Clone)]
pub struct RolloutBuffer {
    /// Number of steps per rollout
    num_steps: usize,

    /// Number of parallel environments
    num_envs: usize,

    /// Dimensionality of observations
    obs_dim: usize,

    /// Observations [num_steps, num_envs, obs_dim]
    observations: Vec<Vec<Vec<f32>>>,

    /// Actions taken [num_steps, num_envs]
    actions: Vec<Vec<i64>>,

    /// Rewards received [num_steps, num_envs]
    rewards: Vec<Vec<f32>>,

    /// Value estimates [num_steps, num_envs]
    values: Vec<Vec<f32>>,

    /// Log probabilities [num_steps, num_envs]
    log_probs: Vec<Vec<f32>>,

    /// Episode termination flags [num_steps, num_envs]
    terminated: Vec<Vec<bool>>,

    /// Episode truncation flags [num_steps, num_envs]
    truncated: Vec<Vec<bool>>,

    /// Computed advantages [num_steps, num_envs]
    advantages: Vec<Vec<f32>>,

    /// Computed returns [num_steps, num_envs]
    returns: Vec<Vec<f32>>,
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
        }
    }

    /// Add a transition to the buffer
    ///
    /// # Arguments
    ///
    /// * `step` - Timestep within the rollout (0 to num_steps-1)
    /// * `env_id` - Environment ID (0 to num_envs-1)
    /// * `observation` - Current observation
    /// * `action` - Action taken
    /// * `reward` - Reward received
    /// * `value` - Value estimate for current state
    /// * `log_prob` - Log probability of the action
    /// * `terminated` - Whether the episode terminated
    /// * `truncated` - Whether the episode was truncated
    pub fn add(
        &mut self,
        step: usize,
        env_id: usize,
        observation: &[f32],
        action: i64,
        reward: f32,
        value: f32,
        log_prob: f32,
        terminated: bool,
        truncated: bool,
    ) {
        debug_assert!(step < self.num_steps, "step {} >= num_steps {}", step, self.num_steps);
        debug_assert!(env_id < self.num_envs, "env_id {} >= num_envs {}", env_id, self.num_envs);
        debug_assert_eq!(observation.len(), self.obs_dim, "observation dimension mismatch");

        self.observations[step][env_id].copy_from_slice(observation);
        self.actions[step][env_id] = action;
        self.rewards[step][env_id] = reward;
        self.values[step][env_id] = value;
        self.log_probs[step][env_id] = log_prob;
        self.terminated[step][env_id] = terminated;
        self.truncated[step][env_id] = truncated;
    }

    /// Reset the buffer for a new rollout
    pub fn reset(&mut self) {
        // Clear computed advantages and returns
        for step in 0..self.num_steps {
            for env in 0..self.num_envs {
                self.advantages[step][env] = 0.0;
                self.returns[step][env] = 0.0;
            }
        }
    }

    /// Get buffer shape (num_steps, num_envs, obs_dim)
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.num_steps, self.num_envs, self.obs_dim)
    }

    /// Get total number of transitions in buffer
    pub fn len(&self) -> usize {
        self.num_steps * self.num_envs
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get observations tensor shape for neural network input
    pub fn obs_shape(&self) -> (usize, usize) {
        (self.num_steps * self.num_envs, self.obs_dim)
    }

    // Getters for raw data access
    pub fn observations(&self) -> &[Vec<Vec<f32>>] { &self.observations }
    pub fn actions(&self) -> &[Vec<i64>] { &self.actions }
    pub fn rewards(&self) -> &[Vec<f32>] { &self.rewards }
    pub fn values(&self) -> &[Vec<f32>] { &self.values }
    pub fn log_probs(&self) -> &[Vec<f32>] { &self.log_probs }
    pub fn terminated(&self) -> &[Vec<bool>] { &self.terminated }
    pub fn truncated(&self) -> &[Vec<bool>] { &self.truncated }
    pub fn advantages(&self) -> &[Vec<f32>] { &self.advantages }
    pub fn returns(&self) -> &[Vec<f32>] { &self.returns }

    // Mutable getters for advantage/return computation
    pub fn advantages_mut(&mut self) -> &mut [Vec<f32>] { &mut self.advantages }
    pub fn returns_mut(&mut self) -> &mut [Vec<f32>] { &mut self.returns }

    /// Get mutable references to both advantages and returns
    /// This is needed to avoid double mutable borrow in GAE computation
    pub fn advantages_and_returns_mut(&mut self) -> (&mut [Vec<f32>], &mut [Vec<f32>]) {
        (&mut self.advantages, &mut self.returns)
    }
}

/// Batch of rollout data for training
///
/// Contains flattened tensors suitable for neural network training.
/// All arrays have shape [batch_size].
#[derive(Debug, Clone)]
pub struct RolloutBatch {
    /// Flattened observations [batch_size, obs_dim]
    pub observations: Vec<f32>,

    /// Actions taken [batch_size]
    pub actions: Vec<i64>,

    /// Old log probabilities [batch_size]
    pub old_log_probs: Vec<f32>,

    /// Old value estimates [batch_size]
    pub old_values: Vec<f32>,

    /// Computed advantages [batch_size]
    pub advantages: Vec<f32>,

    /// Computed returns [batch_size]
    pub returns: Vec<f32>,
}

impl RolloutBatch {
    /// Create a new batch from rollout buffer
    pub fn from_buffer(buffer: &RolloutBuffer) -> Self {
        let batch_size = buffer.len();
        let obs_size = batch_size * buffer.obs_dim;

        let mut observations = Vec::with_capacity(obs_size);
        let mut actions = Vec::with_capacity(batch_size);
        let mut old_log_probs = Vec::with_capacity(batch_size);
        let mut old_values = Vec::with_capacity(batch_size);
        let mut advantages = Vec::with_capacity(batch_size);
        let mut returns = Vec::with_capacity(batch_size);

        // Flatten all data into 1D arrays
        for step in 0..buffer.num_steps {
            for env in 0..buffer.num_envs {
                observations.extend_from_slice(&buffer.observations[step][env]);
                actions.push(buffer.actions[step][env]);
                old_log_probs.push(buffer.log_probs[step][env]);
                old_values.push(buffer.values[step][env]);
                advantages.push(buffer.advantages[step][env]);
                returns.push(buffer.returns[step][env]);
            }
        }

        Self {
            observations,
            actions,
            old_log_probs,
            old_values,
            advantages,
            returns,
        }
    }

    /// Get batch size
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rollout_buffer_creation() {
        let buffer = RolloutBuffer::new(10, 2, 4);

        assert_eq!(buffer.shape(), (10, 2, 4));
        assert_eq!(buffer.len(), 20); // 10 steps * 2 envs
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_rollout_buffer_add_and_reset() {
        let mut buffer = RolloutBuffer::new(5, 1, 2);

        // Add some data
        buffer.add(0, 0, &[1.0, 2.0], 1, 1.5, 0.8, -0.2, false, false);
        buffer.add(1, 0, &[2.0, 3.0], 0, 2.0, 1.2, -0.1, false, false);

        // Check data was stored
        assert_eq!(buffer.actions()[0][0], 1);
        assert_eq!(buffer.rewards()[0][0], 1.5);
        assert_eq!(buffer.observations()[0][0], vec![1.0, 2.0]);

        // Reset and check advantages/returns are cleared
        buffer.reset();
        assert_eq!(buffer.advantages()[0][0], 0.0);
        assert_eq!(buffer.returns()[0][0], 0.0);
    }

    #[test]
    fn test_rollout_batch_from_buffer() {
        let mut buffer = RolloutBuffer::new(2, 1, 2);

        // Add test data
        buffer.add(0, 0, &[1.0, 2.0], 1, 1.5, 0.8, -0.2, false, false);
        buffer.add(1, 0, &[2.0, 3.0], 0, 2.0, 1.2, -0.1, false, false);

        // Set some advantages and returns
        buffer.advantages_mut()[0][0] = 0.5;
        buffer.returns_mut()[0][0] = 1.3;
        buffer.advantages_mut()[1][0] = 0.8;
        buffer.returns_mut()[1][0] = 2.0;

        let batch = RolloutBatch::from_buffer(&buffer);

        assert_eq!(batch.size(), 2);
        assert_eq!(batch.actions, vec![1, 0]);
        assert_eq!(batch.advantages, vec![0.5, 0.8]);
        assert_eq!(batch.returns, vec![1.3, 2.0]);
        assert_eq!(batch.observations, vec![1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn test_rollout_batch_properties() {
        let batch = RolloutBatch {
            observations: vec![1.0, 2.0, 3.0, 4.0],
            actions: vec![0, 1],
            old_log_probs: vec![-0.1, -0.2],
            old_values: vec![0.5, 0.8],
            advantages: vec![0.3, 0.6],
            returns: vec![1.0, 1.5],
        };

        assert_eq!(batch.size(), 2);
        assert_eq!(batch.obs_shape(), (2, 2)); // 2 samples, 2 obs dims each
        assert!(!batch.is_empty());
    }
}
