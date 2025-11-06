//! Data sampling and batching for rollout buffers
//!
//! This module provides utilities for creating training batches from
//! rollout buffers, including minibatch sampling and data shuffling.

use super::storage::RolloutBuffer;

/// Generate minibatch indices for PPO training
///
/// Creates shuffled minibatches from a buffer of the given size.
/// Each minibatch contains approximately `batch_size` samples.
///
/// # Arguments
/// * `buffer_size` - Total number of samples in buffer
/// * `batch_size` - Desired size of each minibatch
///
/// # Returns
/// Vector of vectors, where each inner vector contains indices for one minibatch
pub fn generate_minibatch_indices(buffer_size: usize, batch_size: usize) -> Vec<Vec<usize>> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let mut indices: Vec<usize> = (0..buffer_size).collect();
    indices.shuffle(&mut thread_rng());

    indices
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Sample a minibatch from the rollout buffer
///
/// # Arguments
/// * `buffer` - Source rollout buffer
/// * `indices` - Indices of samples to include in the batch
///
/// # Returns
/// Minibatch data suitable for training
pub fn sample_minibatch(
    buffer: &RolloutBuffer,
    indices: &[usize],
) -> Minibatch {
    let batch_size = indices.len();

    let mut observations = vec![0.0; batch_size * buffer.shape().2];
    let mut actions = vec![0; batch_size];
    let mut old_log_probs = vec![0.0; batch_size];
    let mut old_values = vec![0.0; batch_size];
    let mut advantages = vec![0.0; batch_size];
    let mut returns = vec![0.0; batch_size];

    // Flatten buffer data
    let flat_obs = buffer.observations().iter()
        .flatten()
        .flatten()
        .cloned()
        .collect::<Vec<f32>>();

    let flat_actions = buffer.actions().iter()
        .flatten()
        .cloned()
        .collect::<Vec<i64>>();

    let flat_log_probs = buffer.log_probs().iter()
        .flatten()
        .cloned()
        .collect::<Vec<f32>>();

    let flat_values = buffer.values().iter()
        .flatten()
        .cloned()
        .collect::<Vec<f32>>();

    let flat_advantages = buffer.advantages().iter()
        .flatten()
        .cloned()
        .collect::<Vec<f32>>();

    let flat_returns = buffer.returns().iter()
        .flatten()
        .cloned()
        .collect::<Vec<f32>>();

    // Sample the minibatch
    for (i, &idx) in indices.iter().enumerate() {
        // Copy observation (multiple elements per sample)
        let obs_start = idx * buffer.shape().2;
        let obs_end = obs_start + buffer.shape().2;
        observations[i * buffer.shape().2..(i + 1) * buffer.shape().2]
            .copy_from_slice(&flat_obs[obs_start..obs_end]);

        actions[i] = flat_actions[idx];
        old_log_probs[i] = flat_log_probs[idx];
        old_values[i] = flat_values[idx];
        advantages[i] = flat_advantages[idx];
        returns[i] = flat_returns[idx];
    }

    Minibatch {
        observations,
        actions,
        old_log_probs,
        old_values,
        advantages,
        returns,
        obs_dim: buffer.shape().2,
    }
}

/// Minibatch data for training
///
/// Contains a subset of rollout data arranged for efficient training.
#[derive(Debug, Clone)]
pub struct Minibatch {
    /// Observations [batch_size * obs_dim]
    pub observations: Vec<f32>,

    /// Actions [batch_size]
    pub actions: Vec<i64>,

    /// Old log probabilities [batch_size]
    pub old_log_probs: Vec<f32>,

    /// Old value estimates [batch_size]
    pub old_values: Vec<f32>,

    /// Advantages [batch_size]
    pub advantages: Vec<f32>,

    /// Returns [batch_size]
    pub returns: Vec<f32>,

    /// Observation dimension
    obs_dim: usize,
}

impl Minibatch {
    /// Get batch size
    pub fn size(&self) -> usize {
        self.actions.len()
    }

    /// Get observation shape for neural network input
    pub fn obs_shape(&self) -> (usize, usize) {
        (self.size(), self.obs_dim)
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }
}

/// Iterator for generating minibatches from a rollout buffer
///
/// Provides an iterator interface for processing rollout data in minibatches.
/// Useful for training loops that process data incrementally.
pub struct MinibatchIterator<'a> {
    buffer: &'a RolloutBuffer,
    batch_size: usize,
    indices: Vec<Vec<usize>>,
    current_batch: usize,
}

impl<'a> MinibatchIterator<'a> {
    /// Create a new minibatch iterator
    ///
    /// # Arguments
    /// * `buffer` - Rollout buffer to iterate over
    /// * `batch_size` - Size of each minibatch
    /// * `shuffle` - Whether to shuffle the data order
    pub fn new(buffer: &'a RolloutBuffer, batch_size: usize, shuffle: bool) -> Self {
        let buffer_size = buffer.len();
        let indices = if shuffle {
            generate_minibatch_indices(buffer_size, batch_size)
        } else {
            (0..buffer_size)
                .collect::<Vec<_>>()
                .chunks(batch_size)
                .map(|chunk| chunk.to_vec())
                .collect()
        };

        Self {
            buffer,
            batch_size,
            indices,
            current_batch: 0,
        }
    }
}

impl<'a> Iterator for MinibatchIterator<'a> {
    type Item = Minibatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_batch >= self.indices.len() {
            return None;
        }

        let batch_indices = &self.indices[self.current_batch];
        self.current_batch += 1;

        Some(sample_minibatch(self.buffer, batch_indices))
    }
}

/// Create a shuffled sequence of indices for experience replay
///
/// This can be used to randomize the order of experience samples
/// for more robust training.
///
/// # Arguments
/// * `size` - Total number of samples
///
/// # Returns
/// Vector of shuffled indices
pub fn shuffle_indices(size: usize) -> Vec<usize> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let mut indices: Vec<usize> = (0..size).collect();
    indices.shuffle(&mut thread_rng());
    indices
}

/// Split buffer into train/validation sets
///
/// Useful for hyperparameter tuning or model validation.
///
/// # Arguments
/// * `buffer` - Source buffer to split
/// * `train_ratio` - Fraction of data to use for training (0.0 to 1.0)
///
/// # Returns
/// (train_indices, val_indices)
pub fn train_val_split(buffer: &RolloutBuffer, train_ratio: f32) -> (Vec<usize>, Vec<usize>) {
    let total_size = buffer.len();
    let train_size = ((total_size as f32) * train_ratio) as usize;

    let mut indices: Vec<usize> = (0..total_size).collect();
    // Note: In a real implementation, you'd want to shuffle here
    // but we'll keep it deterministic for reproducibility

    let train_indices = indices[..train_size].to_vec();
    let val_indices = indices[train_size..].to_vec();

    (train_indices, val_indices)
}
