//! DQN configuration and hyperparameters
//!
//! This module defines the configuration parameters for DQN training
//! and provides validation and builder-pattern methods. Mirrors the
//! structure of [`crate::train::ppo::PPOConfig`].

use anyhow::{Result, anyhow};

/// DQN configuration parameters.
///
/// Default values target classic CartPole-style discrete control:
/// 50k-capacity replay, 64-sample batches, hard target sync every
/// 500 env steps, linear ε-decay over 10k env steps. These are the
/// same defaults the Stable-Baselines3 DQN baseline uses on CartPole.
#[derive(Debug, Clone)]
pub struct DQNConfig {
    /// Adam learning rate for the online Q-network.
    pub learning_rate: f64,

    /// Number of transitions sampled per gradient update.
    pub batch_size: usize,

    /// Maximum number of transitions stored in the replay buffer.
    /// Older transitions are evicted FIFO once capacity is reached.
    pub buffer_capacity: usize,

    /// Minimum number of transitions required before training starts.
    /// Until the buffer holds this many transitions the trainer only
    /// collects experience (via random or ε-greedy actions) and skips
    /// gradient updates.
    pub min_buffer_size: usize,

    /// Number of environment steps between hard target-net syncs
    /// (target ← online).
    pub target_update_interval: usize,

    /// Discount factor used in the TD target
    /// `y = r + γ · (1 - done) · max_a' Q_target(s', a')`.
    pub gamma: f64,

    /// Initial value of the ε-greedy exploration parameter.
    pub epsilon_start: f64,

    /// Final value of ε after the linear decay completes.
    pub epsilon_end: f64,

    /// Number of environment steps over which ε linearly anneals from
    /// `epsilon_start` to `epsilon_end`. After this many steps ε stays
    /// at `epsilon_end`.
    pub epsilon_decay_steps: usize,

    /// Maximum gradient norm for clipping the Q-network update.
    pub max_grad_norm: f64,
}

impl Default for DQNConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            batch_size: 64,
            buffer_capacity: 50_000,
            min_buffer_size: 1_000,
            target_update_interval: 500,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 10_000,
            max_grad_norm: 10.0,
        }
    }
}

impl DQNConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate configuration parameters.
    ///
    /// Returns an `Err` describing the first invalid field encountered.
    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 {
            return Err(anyhow!("learning_rate must be positive"));
        }
        if self.batch_size == 0 {
            return Err(anyhow!("batch_size must be positive"));
        }
        if self.buffer_capacity == 0 {
            return Err(anyhow!("buffer_capacity must be positive"));
        }
        if self.buffer_capacity < self.batch_size {
            return Err(anyhow!(
                "buffer_capacity ({}) must be at least batch_size ({})",
                self.buffer_capacity,
                self.batch_size
            ));
        }
        if self.min_buffer_size > self.buffer_capacity {
            return Err(anyhow!(
                "min_buffer_size ({}) must be <= buffer_capacity ({})",
                self.min_buffer_size,
                self.buffer_capacity
            ));
        }
        if self.target_update_interval == 0 {
            return Err(anyhow!("target_update_interval must be positive"));
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err(anyhow!("gamma must be in [0, 1]"));
        }
        if !(0.0..=1.0).contains(&self.epsilon_start) {
            return Err(anyhow!("epsilon_start must be in [0, 1]"));
        }
        if !(0.0..=1.0).contains(&self.epsilon_end) {
            return Err(anyhow!("epsilon_end must be in [0, 1]"));
        }
        if self.epsilon_end > self.epsilon_start {
            return Err(anyhow!(
                "epsilon_end ({}) must be <= epsilon_start ({})",
                self.epsilon_end,
                self.epsilon_start
            ));
        }
        if self.epsilon_decay_steps == 0 {
            return Err(anyhow!("epsilon_decay_steps must be positive"));
        }
        if self.max_grad_norm <= 0.0 {
            return Err(anyhow!("max_grad_norm must be positive"));
        }
        Ok(())
    }

    /// Compute the ε used at a given env-step count under the linear
    /// schedule:
    ///
    /// ```text
    /// ε(t) = max(ε_end, ε_start - (ε_start - ε_end) · t / decay_steps)
    /// ```
    pub fn epsilon_at(&self, env_steps: usize) -> f64 {
        if self.epsilon_decay_steps == 0 {
            return self.epsilon_end;
        }
        let fraction = (env_steps as f64) / (self.epsilon_decay_steps as f64);
        let eps = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * fraction;
        eps.max(self.epsilon_end)
    }

    // ----- Builder-style setters (mirroring PPOConfig) -----

    /// Set learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set minibatch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set replay buffer capacity.
    pub fn buffer_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }

    /// Set minimum buffer size before training starts.
    pub fn min_buffer_size(mut self, size: usize) -> Self {
        self.min_buffer_size = size;
        self
    }

    /// Set target update interval (env steps between hard target syncs).
    pub fn target_update_interval(mut self, steps: usize) -> Self {
        self.target_update_interval = steps;
        self
    }

    /// Set discount factor γ.
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set initial ε for ε-greedy exploration.
    pub fn epsilon_start(mut self, eps: f64) -> Self {
        self.epsilon_start = eps;
        self
    }

    /// Set final ε for ε-greedy exploration.
    pub fn epsilon_end(mut self, eps: f64) -> Self {
        self.epsilon_end = eps;
        self
    }

    /// Set number of env steps over which ε anneals.
    pub fn epsilon_decay_steps(mut self, steps: usize) -> Self {
        self.epsilon_decay_steps = steps;
        self
    }

    /// Set maximum gradient norm.
    pub fn max_grad_norm(mut self, norm: f64) -> Self {
        self.max_grad_norm = norm;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validates() {
        let cfg = DQNConfig::default();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.learning_rate, 1e-3);
        assert_eq!(cfg.batch_size, 64);
        assert_eq!(cfg.buffer_capacity, 50_000);
        assert_eq!(cfg.min_buffer_size, 1_000);
        assert_eq!(cfg.target_update_interval, 500);
        assert_eq!(cfg.gamma, 0.99);
        assert_eq!(cfg.epsilon_start, 1.0);
        assert_eq!(cfg.epsilon_end, 0.05);
        assert_eq!(cfg.epsilon_decay_steps, 10_000);
        assert_eq!(cfg.max_grad_norm, 10.0);
    }

    #[test]
    fn test_builder() {
        let cfg = DQNConfig::new()
            .learning_rate(5e-4)
            .batch_size(128)
            .buffer_capacity(20_000)
            .min_buffer_size(500)
            .target_update_interval(250)
            .gamma(0.95)
            .epsilon_start(0.5)
            .epsilon_end(0.01)
            .epsilon_decay_steps(5_000)
            .max_grad_norm(5.0);
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.learning_rate, 5e-4);
        assert_eq!(cfg.batch_size, 128);
        assert_eq!(cfg.buffer_capacity, 20_000);
        assert_eq!(cfg.min_buffer_size, 500);
        assert_eq!(cfg.target_update_interval, 250);
        assert_eq!(cfg.gamma, 0.95);
        assert_eq!(cfg.epsilon_start, 0.5);
        assert_eq!(cfg.epsilon_end, 0.01);
        assert_eq!(cfg.epsilon_decay_steps, 5_000);
        assert_eq!(cfg.max_grad_norm, 5.0);
    }

    #[test]
    fn test_validate_rejects_negative_lr() {
        let cfg = DQNConfig::new().learning_rate(-1.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_batch() {
        let cfg = DQNConfig::new().batch_size(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_gamma_out_of_range() {
        assert!(DQNConfig::new().gamma(-0.1).validate().is_err());
        assert!(DQNConfig::new().gamma(1.5).validate().is_err());
        assert!(DQNConfig::new().gamma(0.0).validate().is_ok());
        assert!(DQNConfig::new().gamma(1.0).validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_epsilon_end_above_start() {
        let cfg = DQNConfig::new().epsilon_start(0.1).epsilon_end(0.5);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_target_update() {
        let cfg = DQNConfig::new().target_update_interval(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_epsilon_decay() {
        let cfg = DQNConfig::new().epsilon_decay_steps(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_capacity_below_batch() {
        let cfg = DQNConfig::new().buffer_capacity(32).batch_size(64);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_min_buffer_above_capacity() {
        let cfg = DQNConfig::new().buffer_capacity(100).min_buffer_size(1000);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_max_grad_norm() {
        let cfg = DQNConfig::new().max_grad_norm(0.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_epsilon_schedule_linear() {
        let cfg = DQNConfig::new().epsilon_start(1.0).epsilon_end(0.1).epsilon_decay_steps(1000);

        assert!((cfg.epsilon_at(0) - 1.0).abs() < 1e-9);
        // At halfway, ε should be 0.55.
        assert!((cfg.epsilon_at(500) - 0.55).abs() < 1e-6);
        // At full decay, ε should be at ε_end.
        assert!((cfg.epsilon_at(1000) - 0.1).abs() < 1e-9);
        // Past the decay window, ε floors at ε_end.
        assert!((cfg.epsilon_at(10_000) - 0.1).abs() < 1e-9);
    }
}
