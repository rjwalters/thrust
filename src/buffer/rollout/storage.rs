//! Rollout buffer storage and data management
//!
//! This module handles the core storage functionality for rollout buffers,
//! including data insertion, retrieval, and buffer management.
//!
//! The host-side storage layout (`Vec<f32>` / `Vec<i64>`) is deliberately
//! backend-agnostic; tensor materialization happens via
//! [`RolloutBatch::to_burn_tensors`] when the trainer pushes the batch
//! to a device.

use burn::tensor::{Int, Tensor as BurnTensor, TensorData, backend::Backend};

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

    // ---- Getters for raw data access ----
    //
    // All getters return a slice indexed as `[step][env]` (outer = time,
    // inner = parallel env). GAE, log-prob masking, and minibatch sampling
    // all rely on this layout; do not transpose at the call site without
    // adjusting [`compute_advantages`](super::gae::compute_advantages) accordingly.

    /// Borrow the per-step observations, indexed `[step][env]` and
    /// then by observation dimension (final inner `Vec<f32>` is one
    /// observation vector of length `obs_dim`).
    pub fn observations(&self) -> &[Vec<Vec<f32>>] {
        &self.observations
    }
    /// Borrow the per-step discrete actions, indexed `[step][env]`.
    pub fn actions(&self) -> &[Vec<i64>] {
        &self.actions
    }
    /// Borrow the per-step rewards (raw, un-discounted), indexed
    /// `[step][env]`.
    pub fn rewards(&self) -> &[Vec<f32>] {
        &self.rewards
    }
    /// Borrow the per-step bootstrap value estimates `V(s_t)` produced
    /// by the policy at collection time, indexed `[step][env]`. Used by
    /// [`compute_advantages`](super::gae::compute_advantages) as the value
    /// baseline.
    pub fn values(&self) -> &[Vec<f32>] {
        &self.values
    }
    /// Borrow the per-step action log-probabilities under the
    /// behavior policy at collection time, indexed `[step][env]`. PPO
    /// uses these as the denominator of the importance-sampling ratio.
    pub fn log_probs(&self) -> &[Vec<f32>] {
        &self.log_probs
    }
    /// Borrow the per-step terminal-state flags (true on episode end),
    /// indexed `[step][env]`. GAE zeroes the bootstrap across terminal
    /// transitions but keeps the realized reward.
    pub fn terminated(&self) -> &[Vec<bool>] {
        &self.terminated
    }
    /// Borrow the per-step truncation flags (true on time-limit / external
    /// reset that is not a terminal state), indexed `[step][env]`. GAE
    /// retains the bootstrap value across truncated transitions because the
    /// trajectory is still "alive" in the value-function sense.
    pub fn truncated(&self) -> &[Vec<bool>] {
        &self.truncated
    }
    /// Borrow the per-step GAE advantages computed by
    /// [`compute_advantages`](super::gae::compute_advantages), indexed
    /// `[step][env]`. Zero-initialized until GAE has run.
    pub fn advantages(&self) -> &[Vec<f32>] {
        &self.advantages
    }
    /// Borrow the per-step value-function targets (advantages + values),
    /// indexed `[step][env]`. Zero-initialized until GAE has run.
    pub fn returns(&self) -> &[Vec<f32>] {
        &self.returns
    }

    // ---- Mutable getters for advantage/return computation ----

    /// Mutable view of the advantages buffer, indexed `[step][env]`.
    /// Used when in-place normalizing advantages or applying a custom
    /// advantage estimator that does not touch `returns`; use
    /// [`Self::advantages_and_returns_mut`] when both must be borrowed
    /// at the same time (e.g. inside
    /// [`compute_advantages`](super::gae::compute_advantages)).
    pub fn advantages_mut(&mut self) -> &mut [Vec<f32>] {
        &mut self.advantages
    }
    /// Mutable view of the returns / value-target buffer, indexed
    /// `[step][env]`. Use [`Self::advantages_and_returns_mut`] instead
    /// when both must be borrowed at the same time.
    pub fn returns_mut(&mut self) -> &mut [Vec<f32>] {
        &mut self.returns
    }

    /// Get mutable references to both advantages and returns
    /// This is needed to avoid double mutable borrow in GAE computation
    pub fn advantages_and_returns_mut(&mut self) -> (&mut [Vec<f32>], &mut [Vec<f32>]) {
        (&mut self.advantages, &mut self.returns)
    }
}

/// Batch of rollout data for training
///
/// Contains flattened tensors suitable for neural network training.
/// All arrays have shape `[batch_size]`.
#[derive(Debug, Clone)]
pub struct RolloutBatch {
    /// Flattened observations [batch_size, obs_dim]
    pub observations: Vec<f32>,

    /// Actions taken `[batch_size]`
    pub actions: Vec<i64>,

    /// Old log probabilities `[batch_size]`
    pub old_log_probs: Vec<f32>,

    /// Old value estimates `[batch_size]`
    pub old_values: Vec<f32>,

    /// Computed advantages `[batch_size]`
    pub advantages: Vec<f32>,

    /// Computed returns `[batch_size]`
    pub returns: Vec<f32>,
}

impl RolloutBatch {
    /// Create a new batch from rollout buffer
    ///
    /// Iterates the full `[num_steps, num_envs]` capacity. If the buffer
    /// was only partially filled, the unwritten tail surfaces as
    /// zero-initialized rows. Use [`Self::from_buffer_partial`] (or
    /// [`super::RolloutBuffer::get_filled_batch`]) when the caller
    /// knows the fill count.
    pub fn from_buffer(buffer: &RolloutBuffer) -> Self {
        Self::from_buffer_partial(buffer, buffer.num_steps)
    }

    /// Create a new batch from the first `valid_steps` rows of the buffer.
    ///
    /// Rows in `valid_steps..num_steps` (the unfilled tail of a partial
    /// rollout) are skipped, preventing zero-padded rows from
    /// contaminating PPO gradients.
    ///
    /// # Panics
    /// Panics if `valid_steps > buffer.num_steps`.
    pub fn from_buffer_partial(buffer: &RolloutBuffer, valid_steps: usize) -> Self {
        assert!(
            valid_steps <= buffer.num_steps,
            "valid_steps ({}) must not exceed buffer.num_steps ({})",
            valid_steps,
            buffer.num_steps
        );

        let batch_size = valid_steps * buffer.num_envs;
        let obs_size = batch_size * buffer.obs_dim;

        let mut observations = Vec::with_capacity(obs_size);
        let mut actions = Vec::with_capacity(batch_size);
        let mut old_log_probs = Vec::with_capacity(batch_size);
        let mut old_values = Vec::with_capacity(batch_size);
        let mut advantages = Vec::with_capacity(batch_size);
        let mut returns = Vec::with_capacity(batch_size);

        // Flatten the filled prefix into 1D arrays
        for step in 0..valid_steps {
            for env in 0..buffer.num_envs {
                observations.extend_from_slice(&buffer.observations[step][env]);
                actions.push(buffer.actions[step][env]);
                old_log_probs.push(buffer.log_probs[step][env]);
                old_values.push(buffer.values[step][env]);
                advantages.push(buffer.advantages[step][env]);
                returns.push(buffer.returns[step][env]);
            }
        }

        Self { observations, actions, old_log_probs, old_values, advantages, returns }
    }

    /// Get batch size
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the observation shape as (batch_size, obs_dim)
    pub fn obs_shape(&self) -> (usize, usize) {
        let batch_size = self.len();
        let obs_dim = if batch_size == 0 {
            0
        } else {
            self.observations.len() / batch_size
        };
        (batch_size, obs_dim)
    }

    /// Construct the full set of training tensors from this batch in a
    /// Construct the full set of training tensors as Burn tensors on
    /// `device`.
    ///
    /// Returns a named [`RolloutBurnTensors`] bundle so trainers can
    /// pattern-match named fields rather than positional tuple elements.
    ///
    /// Shapes (all on `device`):
    /// - `observations`: `[batch, obs_dim]`, `f32`
    /// - `actions`: `[batch]`, `i64` (Burn `Int` kind)
    /// - `old_log_probs`: `[batch]`, `f32`
    /// - `old_values`: `[batch]`, `f32`
    /// - `advantages`: `[batch]`, `f32`
    /// - `returns`: `[batch]`, `f32`
    ///
    /// Empty batches still produce well-formed `[0, obs_dim]` /
    /// `[0]` tensors. The `obs_dim` is derived from
    /// [`Self::obs_shape`] (which returns `0` for an empty batch), so
    /// the observation tensor for the empty case is shaped `[0, 0]`.
    pub fn to_burn_tensors<B: Backend>(&self, device: &B::Device) -> RolloutBurnTensors<B> {
        let (batch_size, obs_dim) = self.obs_shape();

        // Construct the rank-2 observation tensor directly rather than
        // building a rank-1 tensor and reshaping. The reshape path hits a
        // panic deep in cubecl-zspace when both dims are zero (empty
        // batch), and direct rank-2 construction sidesteps that edge case.
        let observations = BurnTensor::<B, 2>::from_data(
            TensorData::new(self.observations.clone(), [batch_size, obs_dim]),
            device,
        );
        let actions = BurnTensor::<B, 1, Int>::from_data(
            TensorData::new(self.actions.clone(), [batch_size]),
            device,
        );
        let old_log_probs = BurnTensor::<B, 1>::from_data(
            TensorData::new(self.old_log_probs.clone(), [batch_size]),
            device,
        );
        let old_values = BurnTensor::<B, 1>::from_data(
            TensorData::new(self.old_values.clone(), [batch_size]),
            device,
        );
        let advantages = BurnTensor::<B, 1>::from_data(
            TensorData::new(self.advantages.clone(), [batch_size]),
            device,
        );
        let returns = BurnTensor::<B, 1>::from_data(
            TensorData::new(self.returns.clone(), [batch_size]),
            device,
        );

        RolloutBurnTensors { observations, actions, old_log_probs, old_values, advantages, returns }
    }
}

/// Bundle of Burn tensors produced by [`RolloutBatch::to_burn_tensors`].
///
/// Fields are in the order PPO trainers consume them: policy/value
/// inputs first (observations, actions), then the old policy outputs
/// (log-probs and values used for the importance ratio and value clip),
/// then the GAE outputs (advantages and returns). Generic over the
/// backend `B` so the same trainer surface works for CPU (`NdArray`),
/// GPU (`Wgpu`, `Cuda`), and `Autodiff<_>` wrappers.
#[derive(Debug)]
pub struct RolloutBurnTensors<B: Backend> {
    /// Observations, shape `[batch_size, obs_dim]`, dtype `f32`.
    pub observations: BurnTensor<B, 2>,

    /// Discrete actions, shape `[batch_size]`, dtype `i64`.
    pub actions: BurnTensor<B, 1, Int>,

    /// Behavior-policy log-probabilities, shape `[batch_size]`,
    /// dtype `f32`.
    pub old_log_probs: BurnTensor<B, 1>,

    /// Behavior-policy value estimates `V(s_t)`, shape `[batch_size]`,
    /// dtype `f32`.
    pub old_values: BurnTensor<B, 1>,

    /// GAE advantages, shape `[batch_size]`, dtype `f32`.
    pub advantages: BurnTensor<B, 1>,

    /// Value-function targets (advantages + values), shape
    /// `[batch_size]`, dtype `f32`.
    pub returns: BurnTensor<B, 1>,
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

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.actions, vec![1, 0]);
        assert_eq!(batch.advantages, vec![0.5, 0.8]);
        assert_eq!(batch.returns, vec![1.3, 2.0]);
        assert_eq!(batch.observations, vec![1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn test_rollout_batch_from_buffer_partial_skips_unfilled_tail() {
        // 4-step buffer, single env, only the first 2 rows filled.
        let mut buffer = RolloutBuffer::new(4, 1, 2);

        buffer.add(0, 0, &[1.0, 2.0], 1, 1.5, 0.8, -0.2, false, false);
        buffer.add(1, 0, &[2.0, 3.0], 0, 2.0, 1.2, -0.1, false, false);
        buffer.advantages_mut()[0][0] = 0.5;
        buffer.returns_mut()[0][0] = 1.3;
        buffer.advantages_mut()[1][0] = 0.8;
        buffer.returns_mut()[1][0] = 2.0;

        let batch = RolloutBatch::from_buffer_partial(&buffer, 2);

        // Batch has exactly 2 rows — the zero-initialized tail (rows 2-3)
        // is skipped.
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.actions, vec![1, 0]);
        assert_eq!(batch.old_values, vec![0.8, 1.2]);
        assert_eq!(batch.old_log_probs, vec![-0.2, -0.1]);
        assert_eq!(batch.advantages, vec![0.5, 0.8]);
        assert_eq!(batch.returns, vec![1.3, 2.0]);
        assert_eq!(batch.observations, vec![1.0, 2.0, 2.0, 3.0]);

        // `from_buffer` (full capacity) still emits 4 rows; the 2-row
        // partial batch is a strict subset.
        let full_batch = RolloutBatch::from_buffer(&buffer);
        assert_eq!(full_batch.len(), 4);
    }

    #[test]
    fn test_rollout_batch_from_buffer_partial_zero_valid() {
        let buffer = RolloutBuffer::new(4, 2, 3);
        let batch = RolloutBatch::from_buffer_partial(&buffer, 0);
        assert!(batch.is_empty());
        assert_eq!(batch.actions.len(), 0);
        assert_eq!(batch.observations.len(), 0);
    }

    #[test]
    #[should_panic(expected = "valid_steps")]
    fn test_rollout_batch_from_buffer_partial_panics_on_overflow() {
        let buffer = RolloutBuffer::new(4, 1, 2);
        let _ = RolloutBatch::from_buffer_partial(&buffer, 5);
    }

    mod burn_tests {
        use burn::backend::NdArray;

        use super::*;

        type B = NdArray<f32>;

        #[test]
        fn test_to_burn_tensors_matches_inline_construction() {
            // Mirrors the tch round-trip test: shapes + element-wise equality
            // against the source `Vec`s the batch was built from.
            let batch = RolloutBatch {
                observations: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                actions: vec![0, 1, 0],
                old_log_probs: vec![-0.1, -0.2, -0.3],
                old_values: vec![0.5, 0.7, 0.9],
                advantages: vec![0.1, -0.2, 0.3],
                returns: vec![0.6, 0.5, 1.2],
            };

            let device = crate::utils::cuda::default_burn_device::<B>();
            let t = batch.to_burn_tensors::<B>(&device);

            // Shapes.
            assert_eq!(t.observations.dims(), [3, 2]);
            assert_eq!(t.actions.dims(), [3]);
            assert_eq!(t.old_log_probs.dims(), [3]);
            assert_eq!(t.old_values.dims(), [3]);
            assert_eq!(t.advantages.dims(), [3]);
            assert_eq!(t.returns.dims(), [3]);

            // Round-trip values. Observations are row-major flatten, so
            // copying the `[3, 2]` tensor back to a `Vec<f32>` must equal
            // the original 1-D buffer.
            let obs_flat: Vec<f32> = t.observations.into_data().to_vec().unwrap();
            assert_eq!(obs_flat, batch.observations);
            let acts: Vec<i64> = t.actions.into_data().to_vec().unwrap();
            assert_eq!(acts, batch.actions);
            let lp: Vec<f32> = t.old_log_probs.into_data().to_vec().unwrap();
            assert_eq!(lp, batch.old_log_probs);
            let v: Vec<f32> = t.old_values.into_data().to_vec().unwrap();
            assert_eq!(v, batch.old_values);
            let adv: Vec<f32> = t.advantages.into_data().to_vec().unwrap();
            assert_eq!(adv, batch.advantages);
            let ret: Vec<f32> = t.returns.into_data().to_vec().unwrap();
            assert_eq!(ret, batch.returns);
        }

        #[test]
        fn test_to_burn_tensors_empty_batch() {
            // Empty batch → `[0, 0]` observation tensor and `[0]`
            // scalar-per-row tensors, matching the tch path's edge case.
            let batch = RolloutBatch {
                observations: vec![],
                actions: vec![],
                old_log_probs: vec![],
                old_values: vec![],
                advantages: vec![],
                returns: vec![],
            };

            let device = crate::utils::cuda::default_burn_device::<B>();
            let t = batch.to_burn_tensors::<B>(&device);

            assert_eq!(t.observations.dims(), [0, 0]);
            assert_eq!(t.actions.dims(), [0]);
            assert_eq!(t.old_log_probs.dims(), [0]);
            assert_eq!(t.old_values.dims(), [0]);
            assert_eq!(t.advantages.dims(), [0]);
            assert_eq!(t.returns.dims(), [0]);
        }
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

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.obs_shape(), (2, 2)); // 2 samples, 2 obs dims each
        assert!(!batch.is_empty());
    }
}
