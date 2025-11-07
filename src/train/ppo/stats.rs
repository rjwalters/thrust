//! Training statistics for PPO
//!
//! This module defines structures for tracking and aggregating
//! training metrics during PPO training.

use std::ops::AddAssign;

/// Training statistics for a PPO update
///
/// Tracks various metrics from a single training step including
/// losses, KL divergence, and other diagnostic information.
#[derive(Debug, Clone, Default)]
pub struct TrainingStats {
    /// Policy loss
    pub policy_loss: f64,

    /// Value function loss
    pub value_loss: f64,

    /// Entropy bonus
    pub entropy: f64,

    /// Total loss (weighted sum of policy, value, and entropy losses)
    pub total_loss: f64,

    /// Fraction of clipped policy updates
    pub clip_fraction: f64,

    /// Approximate KL divergence between old and new policies
    pub approx_kl: f64,

    /// Explained variance of value function predictions
    pub explained_var: f64,

    /// Number of gradient updates performed
    pub num_updates: usize,
}

impl TrainingStats {
    /// Create zero-initialized statistics
    pub fn zeros() -> Self {
        Self::default()
    }

    /// Create statistics from scalar values
    pub fn new(
        policy_loss: f64,
        value_loss: f64,
        entropy: f64,
        total_loss: f64,
        clip_fraction: f64,
        approx_kl: f64,
        explained_var: f64,
    ) -> Self {
        Self {
            policy_loss,
            value_loss,
            entropy,
            total_loss,
            clip_fraction,
            approx_kl,
            explained_var,
            num_updates: 1,
        }
    }

    /// Add another statistics instance to this one
    pub fn add(&mut self, other: &TrainingStats) {
        self.policy_loss += other.policy_loss;
        self.value_loss += other.value_loss;
        self.entropy += other.entropy;
        self.total_loss += other.total_loss;
        self.clip_fraction += other.clip_fraction;
        self.approx_kl += other.approx_kl;
        self.explained_var += other.explained_var;
        self.num_updates += other.num_updates;
    }

    /// Compute average statistics across multiple updates
    pub fn average(&self) -> Self {
        let scale = self.num_updates as f64;
        if scale == 0.0 {
            return Self::zeros();
        }

        Self {
            policy_loss: self.policy_loss / scale,
            value_loss: self.value_loss / scale,
            entropy: self.entropy / scale,
            total_loss: self.total_loss / scale,
            clip_fraction: self.clip_fraction / scale,
            approx_kl: self.approx_kl / scale,
            explained_var: self.explained_var / scale,
            num_updates: 1,
        }
    }
}

impl AddAssign<&TrainingStats> for TrainingStats {
    fn add_assign(&mut self, other: &TrainingStats) {
        self.add(other);
    }
}

/// Aggregated training statistics across multiple training steps
///
/// Provides summary statistics and trends for monitoring training progress.
#[derive(Debug, Clone)]
pub struct AggregatedStats {
    /// Statistics from the current training step
    pub current: TrainingStats,

    /// Running average of recent statistics
    pub running_avg: TrainingStats,

    /// Best policy loss achieved so far
    pub best_policy_loss: f64,

    /// Best value loss achieved so far
    pub best_value_loss: f64,

    /// Total number of training steps
    pub total_steps: usize,

    /// Learning rate (for logging)
    pub learning_rate: f64,
}

impl AggregatedStats {
    /// Create new aggregated statistics
    pub fn new(learning_rate: f64) -> Self {
        Self {
            current: TrainingStats::zeros(),
            running_avg: TrainingStats::zeros(),
            best_policy_loss: f64::INFINITY,
            best_value_loss: f64::INFINITY,
            total_steps: 0,
            learning_rate,
        }
    }

    /// Update with new training statistics
    pub fn update(&mut self, stats: TrainingStats) {
        self.current = stats.clone();
        self.total_steps += 1;

        // Update running average (exponential moving average with alpha=0.1)
        let alpha = 0.1;
        self.running_avg.policy_loss =
            alpha * stats.policy_loss + (1.0 - alpha) * self.running_avg.policy_loss;
        self.running_avg.value_loss =
            alpha * stats.value_loss + (1.0 - alpha) * self.running_avg.value_loss;
        self.running_avg.entropy = alpha * stats.entropy + (1.0 - alpha) * self.running_avg.entropy;
        self.running_avg.total_loss =
            alpha * stats.total_loss + (1.0 - alpha) * self.running_avg.total_loss;
        self.running_avg.clip_fraction =
            alpha * stats.clip_fraction + (1.0 - alpha) * self.running_avg.clip_fraction;
        self.running_avg.approx_kl =
            alpha * stats.approx_kl + (1.0 - alpha) * self.running_avg.approx_kl;
        self.running_avg.explained_var =
            alpha * stats.explained_var + (1.0 - alpha) * self.running_avg.explained_var;

        // Update best losses
        if stats.policy_loss < self.best_policy_loss {
            self.best_policy_loss = stats.policy_loss;
        }
        if stats.value_loss < self.best_value_loss {
            self.best_value_loss = stats.value_loss;
        }
    }
}
