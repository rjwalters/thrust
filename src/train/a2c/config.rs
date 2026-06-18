//! A2C configuration and hyperparameters.
//!
//! [`A2cConfig`] mirrors the builder + `validate()` style of
//! [`crate::train::ppo::PPOConfig`] and [`crate::train::sac::SacConfig`].
//! Defaults track classic synchronous Advantage Actor-Critic (Mnih et al.
//! 2016, "Asynchronous Methods for Deep Reinforcement Learning",
//! arXiv:1602.01783) as popularized by the synchronous A2C variant: a
//! `7e-4` learning rate, 5-step rollouts across 16 parallel actors, and
//! full n-step returns (`gae_lambda = 1.0`). This is the config half of
//! PR A of the A2C decomposition (#150); the A2C trainer (#152) consumes
//! it.

use anyhow::{Result, anyhow};

/// A2C configuration parameters.
///
/// These hyperparameters control the synchronous Advantage Actor-Critic
/// training process. Default values target classic discrete-control tasks
/// like CartPole: a single gradient update per `n_steps`-long rollout
/// collected across `num_envs` synchronous actors, full n-step returns
/// (`gae_lambda = 1.0`), and global gradient-norm clipping.
#[derive(Debug, Clone)]
pub struct A2cConfig {
    /// Learning rate for the shared actor-critic optimizer.
    pub learning_rate: f64,

    /// Discount factor (`gamma`).
    pub gamma: f64,

    /// GAE lambda parameter. The classic A2C default of `1.0` recovers
    /// full n-step returns (no GAE smoothing).
    pub gae_lambda: f64,

    /// Value function loss coefficient.
    pub value_coef: f64,

    /// Entropy bonus coefficient.
    pub entropy_coef: f64,

    /// Rollout length: number of environment steps collected per update
    /// (Mnih et al. 2016 use 5-step rollouts).
    pub n_steps: usize,

    /// Number of parallel synchronous actors.
    pub num_envs: usize,

    /// Maximum global gradient norm for clipping.
    pub max_grad_norm: f64,

    /// Whether to normalize advantages to zero-mean/unit-variance per
    /// rollout before computing the policy loss.
    pub normalize_advantages: bool,

    /// Seed threaded into seeded network init (e.g.
    /// [`MlpBurnConfig::with_seed`](crate::policy::mlp::MlpBurnConfig::with_seed))
    /// for reproducibility. Two trainers built with the same `seed`
    /// produce bit-identical initialization.
    pub seed: u64,
}

impl Default for A2cConfig {
    fn default() -> Self {
        Self {
            learning_rate: 7e-4,
            gamma: 0.99,
            gae_lambda: 1.0,
            value_coef: 0.5,
            entropy_coef: 0.01,
            n_steps: 5,
            num_envs: 16,
            max_grad_norm: 0.5,
            normalize_advantages: true,
            seed: 0,
        }
    }
}

impl A2cConfig {
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
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err(anyhow!("gamma must be in [0, 1], got {}", self.gamma));
        }
        if !(0.0..=1.0).contains(&self.gae_lambda) {
            return Err(anyhow!("gae_lambda must be in [0, 1], got {}", self.gae_lambda));
        }
        if self.value_coef < 0.0 {
            return Err(anyhow!("value_coef must be non-negative, got {}", self.value_coef));
        }
        if self.entropy_coef < 0.0 {
            return Err(anyhow!("entropy_coef must be non-negative, got {}", self.entropy_coef));
        }
        if self.n_steps == 0 {
            return Err(anyhow!("n_steps must be positive"));
        }
        if self.num_envs == 0 {
            return Err(anyhow!("num_envs must be positive"));
        }
        if self.max_grad_norm <= 0.0 {
            return Err(anyhow!("max_grad_norm must be positive, got {}", self.max_grad_norm));
        }
        Ok(())
    }

    // ----- Builder-style setters (mirroring PPOConfig / SacConfig) -----

    /// Set the learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the discount factor `gamma`.
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the GAE lambda parameter.
    pub fn gae_lambda(mut self, lambda: f64) -> Self {
        self.gae_lambda = lambda;
        self
    }

    /// Set the value function loss coefficient.
    pub fn value_coef(mut self, coef: f64) -> Self {
        self.value_coef = coef;
        self
    }

    /// Set the entropy bonus coefficient.
    pub fn entropy_coef(mut self, coef: f64) -> Self {
        self.entropy_coef = coef;
        self
    }

    /// Set the rollout length (steps collected per update).
    pub fn n_steps(mut self, steps: usize) -> Self {
        self.n_steps = steps;
        self
    }

    /// Set the number of parallel synchronous actors.
    pub fn num_envs(mut self, envs: usize) -> Self {
        self.num_envs = envs;
        self
    }

    /// Set the maximum global gradient norm for clipping.
    pub fn max_grad_norm(mut self, norm: f64) -> Self {
        self.max_grad_norm = norm;
        self
    }

    /// Enable or disable per-rollout advantage normalization.
    pub fn normalize_advantages(mut self, enabled: bool) -> Self {
        self.normalize_advantages = enabled;
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
        let config = A2cConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.learning_rate, 7e-4);
        assert_eq!(config.gamma, 0.99);
        assert_eq!(config.gae_lambda, 1.0);
        assert_eq!(config.value_coef, 0.5);
        assert_eq!(config.entropy_coef, 0.01);
        assert_eq!(config.n_steps, 5);
        assert_eq!(config.num_envs, 16);
        assert_eq!(config.max_grad_norm, 0.5);
        assert!(config.normalize_advantages);
        assert_eq!(config.seed, 0);
    }

    #[test]
    fn test_config_validation() {
        // Valid config should pass
        let config = A2cConfig::new();
        assert!(config.validate().is_ok());

        // Invalid learning rate
        assert!(A2cConfig::new().learning_rate(0.0).validate().is_err());
        assert!(A2cConfig::new().learning_rate(-1.0).validate().is_err());

        // Invalid gamma
        assert!(A2cConfig::new().gamma(-0.1).validate().is_err());
        assert!(A2cConfig::new().gamma(1.5).validate().is_err());
        assert!(A2cConfig::new().gamma(0.0).validate().is_ok());
        assert!(A2cConfig::new().gamma(1.0).validate().is_ok());

        // Invalid gae_lambda
        assert!(A2cConfig::new().gae_lambda(-0.1).validate().is_err());
        assert!(A2cConfig::new().gae_lambda(1.5).validate().is_err());
        assert!(A2cConfig::new().gae_lambda(0.0).validate().is_ok());
        assert!(A2cConfig::new().gae_lambda(1.0).validate().is_ok());

        // Invalid value_coef (negative rejected, 0.0 accepted edge)
        assert!(A2cConfig::new().value_coef(-0.1).validate().is_err());
        assert!(A2cConfig::new().value_coef(0.0).validate().is_ok());

        // Invalid entropy_coef (negative rejected, 0.0 accepted)
        assert!(A2cConfig::new().entropy_coef(-0.1).validate().is_err());
        assert!(A2cConfig::new().entropy_coef(0.0).validate().is_ok());

        // Invalid n_steps
        assert!(A2cConfig::new().n_steps(0).validate().is_err());

        // Invalid num_envs
        assert!(A2cConfig::new().num_envs(0).validate().is_err());

        // Invalid max_grad_norm
        assert!(A2cConfig::new().max_grad_norm(0.0).validate().is_err());
        assert!(A2cConfig::new().max_grad_norm(-1.0).validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = A2cConfig::new()
            .learning_rate(1e-3)
            .gamma(0.95)
            .gae_lambda(0.9)
            .value_coef(0.25)
            .entropy_coef(0.05)
            .n_steps(20)
            .num_envs(8)
            .max_grad_norm(1.0)
            .normalize_advantages(false)
            .seed(42);

        assert!(config.validate().is_ok());
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.gamma, 0.95);
        assert_eq!(config.gae_lambda, 0.9);
        assert_eq!(config.value_coef, 0.25);
        assert_eq!(config.entropy_coef, 0.05);
        assert_eq!(config.n_steps, 20);
        assert_eq!(config.num_envs, 8);
        assert_eq!(config.max_grad_norm, 1.0);
        assert!(!config.normalize_advantages);
        assert_eq!(config.seed, 42);
    }
}
