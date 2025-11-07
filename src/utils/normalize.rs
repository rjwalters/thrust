//! Observation normalization for stable RL training
//!
//! This module provides running mean and standard deviation normalization,
//! which is crucial for stable training in many RL environments.

/// Running mean and standard deviation normalizer
///
/// Tracks mean and variance using Welford's online algorithm
/// for numerical stability.
#[derive(Debug, Clone)]
pub struct RunningMeanStd {
    mean: Vec<f32>,
    var: Vec<f32>,
    count: f64,
    epsilon: f32,
}

impl RunningMeanStd {
    /// Create a new normalizer
    ///
    /// # Arguments
    /// * `size` - Dimension of observations
    /// * `epsilon` - Small constant for numerical stability
    pub fn new(size: usize, epsilon: f32) -> Self {
        Self { mean: vec![0.0; size], var: vec![1.0; size], count: 1e-4, epsilon }
    }

    /// Update statistics with a batch of observations
    ///
    /// # Arguments
    /// * `observations` - Batch of observations [batch_size][obs_dim]
    pub fn update(&mut self, observations: &[Vec<f32>]) {
        if observations.is_empty() {
            return;
        }

        let batch_size = observations.len() as f64;
        let obs_dim = self.mean.len();

        // Compute batch statistics
        let mut batch_mean = vec![0.0; obs_dim];
        for obs in observations {
            for (i, &val) in obs.iter().enumerate() {
                batch_mean[i] += val as f64;
            }
        }
        for val in &mut batch_mean {
            *val /= batch_size;
        }

        let mut batch_var = vec![0.0; obs_dim];
        for obs in observations {
            for (i, &val) in obs.iter().enumerate() {
                let diff = val as f64 - batch_mean[i];
                batch_var[i] += diff * diff;
            }
        }
        for val in &mut batch_var {
            *val /= batch_size;
        }

        // Update running statistics using parallel axis theorem
        let delta = batch_mean
            .iter()
            .zip(&self.mean)
            .map(|(b, m)| b - *m as f64)
            .collect::<Vec<_>>();

        let total_count = self.count + batch_size;

        // Update mean
        for i in 0..obs_dim {
            self.mean[i] = (self.mean[i] as f64 + delta[i] * batch_size / total_count) as f32;
        }

        // Update variance
        for i in 0..obs_dim {
            let m_a = self.var[i] as f64 * self.count;
            let m_b = batch_var[i] * batch_size;
            let m2 = m_a + m_b + delta[i].powi(2) * self.count * batch_size / total_count;
            self.var[i] = (m2 / total_count) as f32;
        }

        self.count = total_count;
    }

    /// Normalize observations
    ///
    /// # Arguments
    /// * `observations` - Observations to normalize [obs_dim]
    ///
    /// # Returns
    /// * Normalized observations (mean=0, std=1)
    pub fn normalize(&self, observations: &[f32]) -> Vec<f32> {
        observations
            .iter()
            .zip(&self.mean)
            .zip(&self.var)
            .map(|((&obs, &mean), &var)| (obs - mean) / (var.sqrt() + self.epsilon))
            .collect()
    }

    /// Normalize a batch of observations in-place
    ///
    /// # Arguments
    /// * `observations` - Batch of observations [batch_size][obs_dim]
    pub fn normalize_batch(&self, observations: &mut [Vec<f32>]) {
        for obs in observations {
            *obs = self.normalize(obs);
        }
    }

    /// Get current mean
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    /// Get current standard deviation
    pub fn std(&self) -> Vec<f32> {
        self.var.iter().map(|v| (v + self.epsilon).sqrt()).collect()
    }

    /// Get number of samples seen
    pub fn count(&self) -> f64 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_basic() {
        let mut normalizer = RunningMeanStd::new(2, 1e-8);

        // Update with some data
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        normalizer.update(&data);

        // Mean should be approximately [2.0, 4.0]
        assert!((normalizer.mean()[0] - 2.0).abs() < 0.1);
        assert!((normalizer.mean()[1] - 4.0).abs() < 0.1);

        // Normalize a new observation
        let obs = vec![2.0, 4.0];
        let normalized = normalizer.normalize(&obs);

        // Should be close to zero (at the mean)
        assert!(normalized[0].abs() < 0.5);
        assert!(normalized[1].abs() < 0.5);
    }

    #[test]
    fn test_normalize_batch() {
        let mut normalizer = RunningMeanStd::new(2, 1e-8);

        let mut batch = vec![vec![0.0, 0.0], vec![1.0, 1.0]];

        normalizer.update(&batch);
        normalizer.normalize_batch(&mut batch);

        // After normalization, mean should be close to 0
        let sum: f32 = batch.iter().flatten().sum();
        assert!(sum.abs() < 0.5);
    }

    #[test]
    fn test_incremental_update() {
        let mut normalizer = RunningMeanStd::new(1, 1e-8);

        // Add observations incrementally
        normalizer.update(&[vec![1.0]]);
        normalizer.update(&[vec![2.0]]);
        normalizer.update(&[vec![3.0]]);

        // Mean should be approximately 2.0
        assert!((normalizer.mean()[0] - 2.0).abs() < 0.1);
    }
}
