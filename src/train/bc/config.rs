//! Behavioral Cloning (BC) configuration and hyperparameters.
//!
//! [`BcConfig`] mirrors the builder + `validate()` style of
//! [`crate::train::a2c::A2cConfig`] and [`crate::train::sac::SacConfig`].
//! Behavioral cloning trains a policy to reproduce expert actions from a
//! fixed [`Demonstrations`](crate::train::bc::Demonstrations) dataset via
//! supervised cross-entropy. Defaults target classic discrete-control
//! tasks like CartPole: a `1e-3` learning rate over `64`-sample minibatches
//! for `10` epochs.
//!
//! This is the config half of PR A of the BC decomposition (#164); the BC
//! trainer (#167) consumes it.

use anyhow::{Result, anyhow};

/// Behavioral Cloning configuration parameters.
///
/// These hyperparameters control the supervised imitation-learning process:
/// the policy is trained for `epochs` passes over a fixed dataset of expert
/// `(observation, action)` pairs, drawing shuffled minibatches of
/// `batch_size` examples and minimizing cross-entropy against the expert
/// labels.
#[derive(Debug, Clone)]
pub struct BcConfig {
    /// Learning rate for the supervised policy optimizer.
    pub learning_rate: f64,

    /// Number of demonstration examples per minibatch.
    pub batch_size: usize,

    /// Number of full passes over the demonstration dataset.
    pub epochs: usize,

    /// Seed threaded into seeded network init and minibatch shuffling for
    /// reproducibility. Two trainers built with the same `seed` produce
    /// bit-identical training trajectories.
    pub seed: u64,
}

impl Default for BcConfig {
    fn default() -> Self {
        Self { learning_rate: 1e-3, batch_size: 64, epochs: 10, seed: 0 }
    }
}

impl BcConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate configuration parameters.
    ///
    /// Returns an `Err` describing the first invalid field encountered.
    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 {
            return Err(anyhow!("learning_rate must be positive, got {}", self.learning_rate));
        }
        if self.batch_size == 0 {
            return Err(anyhow!("batch_size must be positive"));
        }
        if self.epochs == 0 {
            return Err(anyhow!("epochs must be positive"));
        }
        Ok(())
    }

    // ----- Builder-style setters (mirroring A2cConfig / SacConfig) -----

    /// Set the learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the minibatch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the number of training epochs.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set the reproducibility seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BcConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.epochs, 10);
        assert_eq!(config.seed, 0);
    }

    #[test]
    fn test_config_validation() {
        // Valid config should pass
        let config = BcConfig::new();
        assert!(config.validate().is_ok());

        // Invalid learning rate
        assert!(BcConfig::new().learning_rate(0.0).validate().is_err());
        assert!(BcConfig::new().learning_rate(-1.0).validate().is_err());

        // Invalid batch_size
        assert!(BcConfig::new().batch_size(0).validate().is_err());

        // Invalid epochs
        assert!(BcConfig::new().epochs(0).validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = BcConfig::new().learning_rate(5e-4).batch_size(128).epochs(25).seed(42);

        assert!(config.validate().is_ok());
        assert_eq!(config.learning_rate, 5e-4);
        assert_eq!(config.batch_size, 128);
        assert_eq!(config.epochs, 25);
        assert_eq!(config.seed, 42);
    }
}
