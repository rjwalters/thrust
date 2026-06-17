//! Soft Actor-Critic (SAC) configuration and hyperparameters.
//!
//! [`SacConfig`] mirrors the builder + `validate()` style of
//! [`crate::train::dqn::DQNConfig`] and
//! [`crate::train::ppo::PPOConfig`]. Defaults track Haarnoja et al. 2018
//! v2 ("Soft Actor-Critic Algorithms and Applications", arXiv:1812.05905)
//! on Pendulum-scale continuous-control tasks. This is the config half of
//! PR E of the SAC decomposition (#136); the [`crate::train::sac::SacTrainer`]
//! consumes it.

use anyhow::{Result, anyhow};

/// SAC configuration parameters.
///
/// Default values target classic Pendulum-scale continuous control:
/// 1M-capacity replay, 256-sample batches, always-soft Polyak target
/// updates (`tau = 0.005`), automatic entropy temperature tuning, and the
/// `3e-4` Adam learning rate the v2 paper uses for all three optimizers.
/// Smoke tests typically override `buffer_capacity` down to ~50k.
#[derive(Debug, Clone)]
pub struct SacConfig {
    /// Adam learning rate for the stochastic actor.
    pub actor_lr: f64,

    /// Adam learning rate for both online critics (`q1`, `q2`).
    pub critic_lr: f64,

    /// Adam learning rate for the entropy temperature `log_alpha`. Only
    /// used when [`Self::auto_alpha`] is `true`.
    pub alpha_lr: f64,

    /// Discount factor used in the critic TD target.
    pub gamma: f64,

    /// Polyak (soft) target update coefficient `tau`. SAC always performs
    /// a soft update of both target critics every gradient step:
    /// `theta_target <- tau * theta_online + (1 - tau) * theta_target`.
    pub tau: f64,

    /// Number of transitions sampled per gradient update.
    pub batch_size: usize,

    /// Maximum number of transitions stored in the replay buffer. Older
    /// transitions are evicted FIFO once capacity is reached.
    pub buffer_capacity: usize,

    /// Minimum number of transitions required before the first gradient
    /// update. Until the buffer holds this many transitions the trainer
    /// only collects experience.
    pub min_buffer_size: usize,

    /// Number of environment steps for which actions are drawn uniformly
    /// at random (pure exploration) before the actor starts choosing
    /// actions.
    pub learning_starts: usize,

    /// Number of gradient updates performed per environment step (the
    /// update-to-data ratio).
    pub gradient_steps_per_env_step: usize,

    /// Width of every hidden layer in the actor and critics.
    pub hidden_dim: usize,

    /// Number of hidden layers in the actor and critic trunks.
    pub num_hidden_layers: usize,

    /// Automatically tune the entropy temperature `alpha` (Haarnoja et al.
    /// 2018 v2). When `true`, `log_alpha` is optimized to drive the
    /// policy entropy toward [`Self::target_entropy`]. When `false`,
    /// `alpha` is held fixed at [`Self::init_alpha`].
    pub auto_alpha: bool,

    /// Initial entropy temperature `alpha`. When [`Self::auto_alpha`] is
    /// `false` this is the fixed value used throughout training; when
    /// `true` it is the starting point (`log_alpha = ln(init_alpha)`).
    pub init_alpha: f32,

    /// Target policy entropy for automatic temperature tuning. `None`
    /// resolves to the conventional heuristic `-action_dim` at trainer
    /// construction time.
    pub target_entropy: Option<f32>,

    /// Optional global gradient-norm clip applied to every optimizer
    /// step. `None` (the SAC default) leaves the updates unclipped.
    pub max_grad_norm: Option<f64>,

    /// Seed threaded through replay sampling, actor noise sampling, and
    /// seeded network init for bit-exact reproducibility.
    pub seed: u64,
}

impl Default for SacConfig {
    fn default() -> Self {
        Self {
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            alpha_lr: 3e-4,
            gamma: 0.99,
            tau: 0.005,
            batch_size: 256,
            buffer_capacity: 1_000_000,
            min_buffer_size: 1_000,
            learning_starts: 1_000,
            gradient_steps_per_env_step: 1,
            hidden_dim: 256,
            num_hidden_layers: 2,
            auto_alpha: true,
            init_alpha: 0.2,
            target_entropy: None,
            max_grad_norm: None,
            seed: 0,
        }
    }
}

impl SacConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate configuration parameters.
    ///
    /// Returns an `Err` describing the first invalid field encountered.
    pub fn validate(&self) -> Result<()> {
        if self.actor_lr <= 0.0 {
            return Err(anyhow!("actor_lr must be positive, got {}", self.actor_lr));
        }
        if self.critic_lr <= 0.0 {
            return Err(anyhow!("critic_lr must be positive, got {}", self.critic_lr));
        }
        if self.alpha_lr <= 0.0 {
            return Err(anyhow!("alpha_lr must be positive, got {}", self.alpha_lr));
        }
        if !(self.tau > 0.0 && self.tau <= 1.0) {
            return Err(anyhow!("tau must be in (0, 1], got {}", self.tau));
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err(anyhow!("gamma must be in [0, 1], got {}", self.gamma));
        }
        if self.batch_size == 0 {
            return Err(anyhow!("batch_size must be positive"));
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
        if self.init_alpha <= 0.0 {
            return Err(anyhow!("init_alpha must be positive, got {}", self.init_alpha));
        }
        if self.gradient_steps_per_env_step == 0 {
            return Err(anyhow!("gradient_steps_per_env_step must be positive"));
        }
        if self.hidden_dim == 0 {
            return Err(anyhow!("hidden_dim must be positive"));
        }
        if self.num_hidden_layers == 0 {
            return Err(anyhow!("num_hidden_layers must be positive"));
        }
        if let Some(clip) = self.max_grad_norm
            && clip <= 0.0
        {
            return Err(anyhow!("max_grad_norm must be positive when set, got {}", clip));
        }
        if let Some(te) = self.target_entropy
            && !te.is_finite()
        {
            return Err(anyhow!("target_entropy must be finite when set, got {}", te));
        }
        Ok(())
    }

    /// Resolve the effective target entropy for automatic temperature
    /// tuning, applying the `-action_dim` heuristic when
    /// [`Self::target_entropy`] is `None`.
    pub fn resolved_target_entropy(&self, action_dim: usize) -> f32 {
        self.target_entropy.unwrap_or(-(action_dim as f32))
    }

    // ----- Builder-style setters (mirroring DQNConfig / PPOConfig) -----

    /// Set the actor learning rate.
    pub fn actor_lr(mut self, lr: f64) -> Self {
        self.actor_lr = lr;
        self
    }

    /// Set the critic learning rate (applied to both online critics).
    pub fn critic_lr(mut self, lr: f64) -> Self {
        self.critic_lr = lr;
        self
    }

    /// Set the entropy-temperature learning rate.
    pub fn alpha_lr(mut self, lr: f64) -> Self {
        self.alpha_lr = lr;
        self
    }

    /// Set the discount factor `gamma`.
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the Polyak soft-update coefficient `tau`.
    pub fn tau(mut self, tau: f64) -> Self {
        self.tau = tau;
        self
    }

    /// Set the minibatch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set the replay buffer capacity.
    pub fn buffer_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }

    /// Set the minimum buffer size before the first gradient update.
    pub fn min_buffer_size(mut self, size: usize) -> Self {
        self.min_buffer_size = size;
        self
    }

    /// Set the number of random-action warmup steps.
    pub fn learning_starts(mut self, steps: usize) -> Self {
        self.learning_starts = steps;
        self
    }

    /// Set the number of gradient updates per environment step.
    pub fn gradient_steps_per_env_step(mut self, steps: usize) -> Self {
        self.gradient_steps_per_env_step = steps;
        self
    }

    /// Set the hidden-layer width for the actor and critics.
    pub fn hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self
    }

    /// Set the number of hidden layers for the actor and critics.
    pub fn num_hidden_layers(mut self, layers: usize) -> Self {
        self.num_hidden_layers = layers;
        self
    }

    /// Enable or disable automatic entropy-temperature tuning.
    pub fn auto_alpha(mut self, enabled: bool) -> Self {
        self.auto_alpha = enabled;
        self
    }

    /// Set the initial (or fixed) entropy temperature `alpha`.
    pub fn init_alpha(mut self, alpha: f32) -> Self {
        self.init_alpha = alpha;
        self
    }

    /// Set an explicit target entropy, overriding the `-action_dim`
    /// heuristic.
    pub fn target_entropy(mut self, entropy: f32) -> Self {
        self.target_entropy = Some(entropy);
        self
    }

    /// Enable global gradient-norm clipping with the given cap.
    pub fn max_grad_norm(mut self, norm: f64) -> Self {
        self.max_grad_norm = Some(norm);
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
    fn test_default_config_validates() {
        let cfg = SacConfig::default();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.actor_lr, 3e-4);
        assert_eq!(cfg.critic_lr, 3e-4);
        assert_eq!(cfg.alpha_lr, 3e-4);
        assert_eq!(cfg.gamma, 0.99);
        assert_eq!(cfg.tau, 0.005);
        assert_eq!(cfg.batch_size, 256);
        assert_eq!(cfg.buffer_capacity, 1_000_000);
        assert_eq!(cfg.min_buffer_size, 1_000);
        assert_eq!(cfg.learning_starts, 1_000);
        assert_eq!(cfg.gradient_steps_per_env_step, 1);
        assert_eq!(cfg.hidden_dim, 256);
        assert_eq!(cfg.num_hidden_layers, 2);
        assert!(cfg.auto_alpha);
        assert_eq!(cfg.init_alpha, 0.2);
        assert!(cfg.target_entropy.is_none());
        assert!(cfg.max_grad_norm.is_none());
        assert_eq!(cfg.seed, 0);
    }

    #[test]
    fn test_builder() {
        let cfg = SacConfig::new()
            .actor_lr(1e-3)
            .critic_lr(2e-3)
            .alpha_lr(5e-4)
            .gamma(0.95)
            .tau(0.01)
            .batch_size(128)
            .buffer_capacity(50_000)
            .min_buffer_size(500)
            .learning_starts(2_000)
            .gradient_steps_per_env_step(2)
            .hidden_dim(64)
            .num_hidden_layers(3)
            .auto_alpha(false)
            .init_alpha(0.1)
            .target_entropy(-2.0)
            .max_grad_norm(5.0)
            .seed(42);
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.actor_lr, 1e-3);
        assert_eq!(cfg.critic_lr, 2e-3);
        assert_eq!(cfg.alpha_lr, 5e-4);
        assert_eq!(cfg.gamma, 0.95);
        assert_eq!(cfg.tau, 0.01);
        assert_eq!(cfg.batch_size, 128);
        assert_eq!(cfg.buffer_capacity, 50_000);
        assert_eq!(cfg.min_buffer_size, 500);
        assert_eq!(cfg.learning_starts, 2_000);
        assert_eq!(cfg.gradient_steps_per_env_step, 2);
        assert_eq!(cfg.hidden_dim, 64);
        assert_eq!(cfg.num_hidden_layers, 3);
        assert!(!cfg.auto_alpha);
        assert_eq!(cfg.init_alpha, 0.1);
        assert_eq!(cfg.target_entropy, Some(-2.0));
        assert_eq!(cfg.max_grad_norm, Some(5.0));
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_resolved_target_entropy() {
        let cfg = SacConfig::default();
        assert_eq!(cfg.resolved_target_entropy(1), -1.0);
        assert_eq!(cfg.resolved_target_entropy(3), -3.0);
        let cfg = cfg.target_entropy(-7.5);
        assert_eq!(cfg.resolved_target_entropy(3), -7.5);
    }

    #[test]
    fn test_validate_rejects_non_positive_lrs() {
        assert!(SacConfig::new().actor_lr(0.0).validate().is_err());
        assert!(SacConfig::new().actor_lr(-1.0).validate().is_err());
        assert!(SacConfig::new().critic_lr(0.0).validate().is_err());
        assert!(SacConfig::new().alpha_lr(-1e-4).validate().is_err());
    }

    #[test]
    fn test_validate_rejects_tau_out_of_range() {
        assert!(SacConfig::new().tau(0.0).validate().is_err());
        assert!(SacConfig::new().tau(-0.1).validate().is_err());
        assert!(SacConfig::new().tau(1.5).validate().is_err());
        assert!(SacConfig::new().tau(1.0).validate().is_ok());
        assert!(SacConfig::new().tau(0.005).validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_gamma_out_of_range() {
        assert!(SacConfig::new().gamma(-0.1).validate().is_err());
        assert!(SacConfig::new().gamma(1.5).validate().is_err());
        assert!(SacConfig::new().gamma(0.0).validate().is_ok());
        assert!(SacConfig::new().gamma(1.0).validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_zero_batch() {
        assert!(SacConfig::new().batch_size(0).validate().is_err());
    }

    #[test]
    fn test_validate_rejects_capacity_below_batch() {
        let cfg = SacConfig::new().buffer_capacity(64).batch_size(256);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_min_buffer_above_capacity() {
        let cfg = SacConfig::new().buffer_capacity(1_000).min_buffer_size(5_000);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_non_positive_init_alpha() {
        assert!(SacConfig::new().init_alpha(0.0).validate().is_err());
        assert!(SacConfig::new().init_alpha(-0.2).validate().is_err());
        assert!(SacConfig::new().init_alpha(0.2).validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_zero_gradient_steps() {
        assert!(SacConfig::new().gradient_steps_per_env_step(0).validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_hidden_and_layers() {
        assert!(SacConfig::new().hidden_dim(0).validate().is_err());
        assert!(SacConfig::new().num_hidden_layers(0).validate().is_err());
    }

    #[test]
    fn test_validate_rejects_non_positive_grad_clip() {
        assert!(SacConfig::new().max_grad_norm(0.0).validate().is_err());
        assert!(SacConfig::new().max_grad_norm(-1.0).validate().is_err());
        assert!(SacConfig::new().max_grad_norm(10.0).validate().is_ok());
    }
}
