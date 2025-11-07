//! PPO configuration and hyperparameters
//!
//! This module defines the configuration parameters for PPO training
//! and provides validation and builder pattern methods.

use anyhow::{Result, anyhow};

/// PPO configuration parameters
///
/// These hyperparameters control the PPO training process.
/// Default values are based on common settings that work well
/// for simple environments like CartPole.
#[derive(Debug, Clone)]
pub struct PPOConfig {
    /// Learning rate for policy and value function
    pub learning_rate: f64,

    /// Number of training epochs per rollout
    pub n_epochs: usize,

    /// Minibatch size for training
    pub batch_size: usize,

    /// Discount factor (gamma)
    pub gamma: f64,

    /// GAE lambda parameter
    pub gae_lambda: f64,

    /// PPO clipping parameter (epsilon)
    pub clip_range: f64,

    /// Value function clipping parameter
    pub clip_range_vf: f64,

    /// Value function loss coefficient
    pub vf_coef: f64,

    /// Entropy bonus coefficient
    pub ent_coef: f64,

    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,

    /// Target KL divergence for early stopping
    pub target_kl: f64,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            n_epochs: 10,
            batch_size: 64,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_range: 0.2,
            clip_range_vf: 0.2,
            vf_coef: 0.5,
            ent_coef: 0.01,
            max_grad_norm: 0.5,
            target_kl: 0.01,
        }
    }
}

impl PPOConfig {
    /// Create a new default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 {
            return Err(anyhow!("learning_rate must be positive"));
        }
        if self.n_epochs == 0 {
            return Err(anyhow!("n_epochs must be positive"));
        }
        if self.batch_size == 0 {
            return Err(anyhow!("batch_size must be positive"));
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err(anyhow!("gamma must be in [0, 1]"));
        }
        if !(0.0..=1.0).contains(&self.gae_lambda) {
            return Err(anyhow!("gae_lambda must be in [0, 1]"));
        }
        if self.clip_range <= 0.0 {
            return Err(anyhow!("clip_range must be positive"));
        }
        if self.clip_range_vf <= 0.0 {
            return Err(anyhow!("clip_range_vf must be positive"));
        }
        if self.vf_coef < 0.0 {
            return Err(anyhow!("vf_coef must be non-negative"));
        }
        if self.ent_coef < 0.0 {
            return Err(anyhow!("ent_coef must be non-negative"));
        }
        if self.max_grad_norm <= 0.0 {
            return Err(anyhow!("max_grad_norm must be positive"));
        }
        if self.target_kl < 0.0 {
            return Err(anyhow!("target_kl must be non-negative"));
        }
        Ok(())
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set number of training epochs
    pub fn n_epochs(mut self, epochs: usize) -> Self {
        self.n_epochs = epochs;
        self
    }

    /// Set minibatch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set discount factor
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set GAE lambda
    pub fn gae_lambda(mut self, lambda: f64) -> Self {
        self.gae_lambda = lambda;
        self
    }

    /// Set PPO clipping parameter
    pub fn clip_range(mut self, clip: f64) -> Self {
        self.clip_range = clip;
        self
    }

    /// Set value function clipping parameter
    pub fn clip_range_vf(mut self, clip: f64) -> Self {
        self.clip_range_vf = clip;
        self
    }

    /// Set value function loss coefficient
    pub fn vf_coef(mut self, coef: f64) -> Self {
        self.vf_coef = coef;
        self
    }

    /// Set entropy bonus coefficient
    pub fn ent_coef(mut self, coef: f64) -> Self {
        self.ent_coef = coef;
        self
    }

    /// Set maximum gradient norm
    pub fn max_grad_norm(mut self, norm: f64) -> Self {
        self.max_grad_norm = norm;
        self
    }

    /// Set target KL divergence
    pub fn target_kl(mut self, kl: f64) -> Self {
        self.target_kl = kl;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PPOConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.learning_rate, 3e-4);
        assert_eq!(config.n_epochs, 10);
        assert_eq!(config.batch_size, 64);
    }

    #[test]
    fn test_config_validation() {
        // Valid config should pass
        let config = PPOConfig::new();
        assert!(config.validate().is_ok());

        // Invalid learning rate
        let config = PPOConfig::new().learning_rate(-1.0);
        assert!(config.validate().is_err());

        // Invalid gamma
        let config = PPOConfig::new().gamma(1.5);
        assert!(config.validate().is_err());

        // Invalid n_epochs
        let config = PPOConfig::new().n_epochs(0);
        assert!(config.validate().is_err());

        // Invalid batch_size
        let config = PPOConfig::new().batch_size(0);
        assert!(config.validate().is_err());

        // Invalid clip_range
        let config = PPOConfig::new().clip_range(-0.1);
        assert!(config.validate().is_err());

        // Invalid vf_coef (should allow 0.0)
        let config = PPOConfig::new().vf_coef(-0.1);
        assert!(config.validate().is_err());

        // Valid vf_coef = 0.0
        let config = PPOConfig::new().vf_coef(0.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = PPOConfig::new()
            .learning_rate(1e-3)
            .n_epochs(5)
            .batch_size(128)
            .gamma(0.95)
            .clip_range(0.1);

        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.n_epochs, 5);
        assert_eq!(config.batch_size, 128);
        assert_eq!(config.gamma, 0.95);
        assert_eq!(config.clip_range, 0.1);

        // Other values should remain default
        assert_eq!(config.gae_lambda, 0.95);
        assert_eq!(config.vf_coef, 0.5);
    }
}
