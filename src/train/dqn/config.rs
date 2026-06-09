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

    /// Discount factor used in the TD target. With Double-DQN
    /// (used unconditionally by [`crate::train::dqn::DQNTrainerBurn`]):
    /// `y = r + γ · (1 - done) · Q_target(s', argmax_a' Q_online(s', a'))`.
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

    /// Polyak (soft) target update coefficient `τ ∈ (0, 1]`.
    ///
    /// When `Some(τ)`, every call to
    /// [`crate::train::dqn::DQNTrainerBurn::maybe_sync_target`] performs the
    /// blend
    ///
    /// ```text
    /// θ_target ← τ · θ_online + (1 − τ) · θ_target
    /// ```
    ///
    /// across every parameter of the target network. This replaces the
    /// hard copy that fires every `target_update_interval` env steps; in
    /// soft-update mode `target_update_interval` is ignored.
    ///
    /// When `None` (default), the trainer falls back to the original hard
    /// copy on the interval, preserving byte-for-byte backward
    /// compatibility with vanilla DQN.
    ///
    /// A typical value is `0.005` (the SB3/Spinning Up default).
    pub soft_update_tau: Option<f64>,

    /// Use Prioritized Experience Replay (Schaul et al., 2015) in place
    /// of the uniform [`crate::buffer::replay::ReplayBuffer`].
    ///
    /// When `true`, the trainer holds a
    /// [`crate::buffer::replay::PrioritizedReplayBuffer`] and samples
    /// transitions proportionally to `(|TD error| + ε)^α`. Per-sample
    /// importance-sampling weights are applied to the Smooth-L1 loss
    /// (so high-priority transitions get correspondingly down-weighted
    /// updates), and the buffer's priorities are refreshed after each
    /// gradient step with the new TD-error magnitudes.
    ///
    /// Defaults to `false` to preserve the vanilla uniform behavior.
    pub prioritized_replay: bool,

    /// PER priority exponent `α ∈ [0, 1]`. `0` recovers uniform sampling
    /// (purely as a degenerate case — flip [`Self::prioritized_replay`]
    /// off for the same effect without the sum-tree overhead). `1` is
    /// fully proportional to priority. Typical value `0.6` (Schaul §3.2).
    pub per_alpha: f64,

    /// PER importance-sampling exponent at the start of training.
    /// Typical value `0.4` (Schaul §3.4).
    pub per_beta_start: f64,

    /// PER importance-sampling exponent at the end of the annealing
    /// schedule. Always `1.0` in the original paper — full bias
    /// correction by the end of training.
    pub per_beta_end: f64,

    /// Number of environment steps over which β linearly anneals from
    /// `per_beta_start` to `per_beta_end`. After this many steps β stays
    /// at `per_beta_end`. Defaults to `epsilon_decay_steps` if set to
    /// `0` (sentinel "follow the ε schedule").
    pub per_beta_steps: usize,

    /// Tiny constant added to `|TD error|` before raising to `α`, so
    /// transitions with vanishing TD error still have a small chance of
    /// being resampled. Typical value `1e-6`.
    pub per_epsilon: f64,
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
            soft_update_tau: None,
            prioritized_replay: false,
            per_alpha: 0.6,
            per_beta_start: 0.4,
            per_beta_end: 1.0,
            per_beta_steps: 0,
            per_epsilon: 1e-6,
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
        if let Some(tau) = self.soft_update_tau {
            if !(tau > 0.0 && tau <= 1.0) {
                return Err(anyhow!("soft_update_tau must be in (0, 1], got {}", tau));
            }
        }
        // Prioritized Experience Replay parameter ranges. We validate
        // even when `prioritized_replay = false` so callers that flip
        // the flag on later don't suddenly hit a runtime error from a
        // stale, half-built config.
        if !(0.0..=1.0).contains(&self.per_alpha) {
            return Err(anyhow!("per_alpha must be in [0, 1], got {}", self.per_alpha));
        }
        if !(0.0..=1.0).contains(&self.per_beta_start) {
            return Err(anyhow!("per_beta_start must be in [0, 1], got {}", self.per_beta_start));
        }
        if !(0.0..=1.0).contains(&self.per_beta_end) {
            return Err(anyhow!("per_beta_end must be in [0, 1], got {}", self.per_beta_end));
        }
        if self.per_beta_start > self.per_beta_end {
            return Err(anyhow!(
                "per_beta_start ({}) must be <= per_beta_end ({})",
                self.per_beta_start,
                self.per_beta_end
            ));
        }
        if self.per_epsilon < 0.0 || !self.per_epsilon.is_finite() {
            return Err(anyhow!(
                "per_epsilon must be finite and non-negative, got {}",
                self.per_epsilon
            ));
        }
        Ok(())
    }

    /// Compute β at a given env-step count under the linear schedule:
    ///
    /// ```text
    /// β(t) = β_start + (β_end − β_start) · min(t / β_steps, 1)
    /// ```
    ///
    /// If `per_beta_steps == 0` the trainer falls back to
    /// `epsilon_decay_steps` so callers can leave the field at its
    /// default and have β anneal over the same window as ε.
    pub fn beta_at(&self, env_steps: usize) -> f64 {
        let steps = if self.per_beta_steps == 0 {
            self.epsilon_decay_steps.max(1)
        } else {
            self.per_beta_steps
        };
        let fraction = ((env_steps as f64) / (steps as f64)).clamp(0.0, 1.0);
        self.per_beta_start + (self.per_beta_end - self.per_beta_start) * fraction
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

    /// Enable Polyak (soft) target updates with coefficient `τ`.
    ///
    /// When set, [`crate::train::dqn::DQNTrainerBurn::maybe_sync_target`]
    /// performs `θ_target ← τ · θ_online + (1 − τ) · θ_target` on every
    /// call (i.e. every env step in the standard rollout loop) instead of
    /// the hard copy gated by `target_update_interval`.
    ///
    /// A typical value is `0.005`.
    pub fn soft_update_tau(mut self, tau: f64) -> Self {
        self.soft_update_tau = Some(tau);
        self
    }

    /// Enable or disable Prioritized Experience Replay.
    pub fn prioritized_replay(mut self, enabled: bool) -> Self {
        self.prioritized_replay = enabled;
        self
    }

    /// Set the PER priority exponent `α`.
    pub fn per_alpha(mut self, alpha: f64) -> Self {
        self.per_alpha = alpha;
        self
    }

    /// Set the PER importance-sampling exponent at the start of training.
    pub fn per_beta_start(mut self, beta: f64) -> Self {
        self.per_beta_start = beta;
        self
    }

    /// Set the PER importance-sampling exponent at the end of training.
    pub fn per_beta_end(mut self, beta: f64) -> Self {
        self.per_beta_end = beta;
        self
    }

    /// Set the number of env steps over which β linearly anneals.
    /// Pass `0` to follow [`Self::epsilon_decay_steps`].
    pub fn per_beta_steps(mut self, steps: usize) -> Self {
        self.per_beta_steps = steps;
        self
    }

    /// Set the PER priority floor `ε`.
    pub fn per_epsilon(mut self, eps: f64) -> Self {
        self.per_epsilon = eps;
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
    fn test_default_soft_update_tau_is_none() {
        let cfg = DQNConfig::default();
        assert!(cfg.soft_update_tau.is_none());
    }

    #[test]
    fn test_soft_update_tau_builder() {
        let cfg = DQNConfig::new().soft_update_tau(0.005);
        assert_eq!(cfg.soft_update_tau, Some(0.005));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_soft_update_tau_out_of_range() {
        assert!(DQNConfig::new().soft_update_tau(0.0).validate().is_err());
        assert!(DQNConfig::new().soft_update_tau(-0.1).validate().is_err());
        assert!(DQNConfig::new().soft_update_tau(1.5).validate().is_err());
        assert!(DQNConfig::new().soft_update_tau(1.0).validate().is_ok());
        assert!(DQNConfig::new().soft_update_tau(0.005).validate().is_ok());
    }

    #[test]
    fn test_default_per_fields() {
        let cfg = DQNConfig::default();
        assert!(!cfg.prioritized_replay);
        assert!((cfg.per_alpha - 0.6).abs() < 1e-9);
        assert!((cfg.per_beta_start - 0.4).abs() < 1e-9);
        assert!((cfg.per_beta_end - 1.0).abs() < 1e-9);
        assert_eq!(cfg.per_beta_steps, 0);
        assert!((cfg.per_epsilon - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn test_per_builder_setters() {
        let cfg = DQNConfig::new()
            .prioritized_replay(true)
            .per_alpha(0.7)
            .per_beta_start(0.3)
            .per_beta_end(1.0)
            .per_beta_steps(20_000)
            .per_epsilon(1e-5);
        assert!(cfg.prioritized_replay);
        assert!((cfg.per_alpha - 0.7).abs() < 1e-9);
        assert!((cfg.per_beta_start - 0.3).abs() < 1e-9);
        assert!((cfg.per_beta_end - 1.0).abs() < 1e-9);
        assert_eq!(cfg.per_beta_steps, 20_000);
        assert!((cfg.per_epsilon - 1e-5).abs() < 1e-12);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_per_alpha_out_of_range() {
        assert!(DQNConfig::new().per_alpha(-0.1).validate().is_err());
        assert!(DQNConfig::new().per_alpha(1.5).validate().is_err());
        assert!(DQNConfig::new().per_alpha(0.0).validate().is_ok());
        assert!(DQNConfig::new().per_alpha(1.0).validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_per_beta_out_of_range() {
        assert!(DQNConfig::new().per_beta_start(-0.1).validate().is_err());
        assert!(DQNConfig::new().per_beta_end(1.5).validate().is_err());
    }

    #[test]
    fn test_validate_rejects_per_beta_start_above_end() {
        let cfg = DQNConfig::new().per_beta_start(0.9).per_beta_end(0.5);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_negative_per_epsilon() {
        let cfg = DQNConfig::new().per_epsilon(-1e-6);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_beta_schedule_linear() {
        let cfg = DQNConfig::new().per_beta_start(0.4).per_beta_end(1.0).per_beta_steps(1000);
        assert!((cfg.beta_at(0) - 0.4).abs() < 1e-9);
        assert!((cfg.beta_at(500) - 0.7).abs() < 1e-9);
        assert!((cfg.beta_at(1000) - 1.0).abs() < 1e-9);
        // Past the end of the schedule β floors at β_end.
        assert!((cfg.beta_at(10_000) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_beta_schedule_falls_back_to_epsilon_decay() {
        // If per_beta_steps == 0, β should anneal over epsilon_decay_steps.
        let cfg = DQNConfig::new()
            .per_beta_start(0.4)
            .per_beta_end(1.0)
            .per_beta_steps(0)
            .epsilon_decay_steps(2_000);
        assert!((cfg.beta_at(1_000) - 0.7).abs() < 1e-9);
        assert!((cfg.beta_at(2_000) - 1.0).abs() < 1e-9);
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
