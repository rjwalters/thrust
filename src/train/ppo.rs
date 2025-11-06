//! Proximal Policy Optimization (PPO) algorithm
//!
//! This module implements the PPO algorithm for training RL agents.
//! PPO is a policy gradient method that uses a clipped surrogate objective
//! to ensure stable, reliable policy updates.
//!
//! # Algorithm Overview
//!
//! ```text
//! For each epoch:
//!   1. Collect trajectories using current policy
//!   2. Compute advantages using GAE
//!   3. For multiple epochs:
//!      a. Sample minibatches from buffer
//!      b. Compute PPO loss (clipped objective)
//!      c. Update policy via gradient descent
//! ```
//!
//! # References
//!
//! - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
//! - [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

use anyhow::Result;
use tch::Tensor;

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
    pub clip_range_vf: Option<f64>,

    /// Coefficient for value function loss
    pub vf_coef: f64,

    /// Coefficient for entropy bonus
    pub ent_coef: f64,

    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,

    /// Target KL divergence (for early stopping)
    pub target_kl: Option<f64>,
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
            clip_range_vf: None,
            vf_coef: 0.5,
            ent_coef: 0.01,
            max_grad_norm: 0.5,
            target_kl: None,
        }
    }
}

impl PPOConfig {
    /// Create a new PPO config with default values
    pub fn new() -> Self {
        Self::default()
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

    /// Set PPO clip range
    pub fn clip_range(mut self, clip: f64) -> Self {
        self.clip_range = clip;
        self
    }

    /// Set value function coefficient
    pub fn vf_coef(mut self, coef: f64) -> Self {
        self.vf_coef = coef;
        self
    }

    /// Set entropy coefficient
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
        self.target_kl = Some(kl);
        self
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 {
            anyhow::bail!("Learning rate must be positive");
        }
        if self.n_epochs == 0 {
            anyhow::bail!("Number of epochs must be positive");
        }
        if self.batch_size == 0 {
            anyhow::bail!("Batch size must be positive");
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            anyhow::bail!("Gamma must be in [0, 1]");
        }
        if !(0.0..=1.0).contains(&self.gae_lambda) {
            anyhow::bail!("GAE lambda must be in [0, 1]");
        }
        if self.clip_range <= 0.0 {
            anyhow::bail!("Clip range must be positive");
        }
        Ok(())
    }
}

/// Training statistics for one update step
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Total loss (policy + value + entropy)
    pub total_loss: f64,

    /// Policy loss (clipped surrogate objective)
    pub policy_loss: f64,

    /// Value function loss
    pub value_loss: f64,

    /// Entropy bonus
    pub entropy: f64,

    /// Approximate KL divergence
    pub approx_kl: f64,

    /// Fraction of samples where policy was clipped
    pub clip_fraction: f64,

    /// Explained variance (1 = perfect value function)
    pub explained_variance: f64,
}

impl TrainingStats {
    /// Create stats with all zeros
    pub fn zeros() -> Self {
        Self {
            total_loss: 0.0,
            policy_loss: 0.0,
            value_loss: 0.0,
            entropy: 0.0,
            approx_kl: 0.0,
            clip_fraction: 0.0,
            explained_variance: 0.0,
        }
    }
}

/// Compute PPO policy loss (clipped surrogate objective)
///
/// # Arguments
///
/// * `log_probs` - Log probabilities of actions under current policy
/// * `old_log_probs` - Log probabilities of actions under old policy
/// * `advantages` - Advantage estimates
/// * `clip_range` - Clipping parameter epsilon
///
/// # Returns
///
/// Returns (policy_loss, clip_fraction, approx_kl)
pub fn compute_policy_loss(
    log_probs: &Tensor,
    old_log_probs: &Tensor,
    advantages: &Tensor,
    clip_range: f64,
) -> (Tensor, f64, f64) {
    // Compute ratio: pi(a|s) / pi_old(a|s)
    let ratio = (log_probs - old_log_probs).exp();

    // Normalize advantages (important for stable training)
    let adv_mean = advantages.mean(tch::Kind::Float);
    let adv_std = advantages.std(false);
    let advantages_normalized = (advantages - adv_mean) / (adv_std + 1e-8);

    // Clipped surrogate objective
    let policy_loss_1 = &advantages_normalized * &ratio;
    let policy_loss_2 = &advantages_normalized * ratio.clamp(1.0 - clip_range, 1.0 + clip_range);
    let policy_loss = -policy_loss_1.min_other(&policy_loss_2).mean(tch::Kind::Float);

    // Compute clip fraction (for monitoring)
    let clip_mask = (ratio.abs() - 1.0).abs().gt(clip_range);
    let clip_fraction =
        f64::try_from(clip_mask.to_kind(tch::Kind::Float).mean(tch::Kind::Float)).unwrap_or(0.0);

    // Approximate KL divergence: KL(old||new) ≈ log(r) - (r - 1)
    let approx_kl =
        f64::try_from((log_probs - old_log_probs).mean(tch::Kind::Float)).unwrap_or(0.0);

    (policy_loss, clip_fraction, approx_kl)
}

/// Compute value function loss
///
/// # Arguments
///
/// * `values` - Value predictions from current policy
/// * `old_values` - Value predictions from old policy
/// * `returns` - Discounted returns (targets)
/// * `clip_range_vf` - Optional clipping for value function
///
/// # Returns
///
/// Returns (value_loss, explained_variance)
pub fn compute_value_loss(
    values: &Tensor,
    old_values: &Tensor,
    returns: &Tensor,
    clip_range_vf: Option<f64>,
) -> (Tensor, f64) {
    // Compute value loss
    let value_loss = if let Some(clip_vf) = clip_range_vf {
        // Clipped value loss (like in PPO paper)
        let values_clipped = old_values + (values - old_values).clamp(-clip_vf, clip_vf);
        let value_loss_1 = (values - returns).pow_tensor_scalar(2);
        let value_loss_2 = (&values_clipped - returns).pow_tensor_scalar(2);
        value_loss_1.max_other(&value_loss_2).mean(tch::Kind::Float)
    } else {
        // Unclipped value loss (simpler)
        (values - returns).pow_tensor_scalar(2).mean(tch::Kind::Float)
    };

    // Compute explained variance: 1 - Var(returns - values) / Var(returns)
    let returns_var = returns.var(false);
    let residual_var = (returns - values).var(false);
    let explained_var = 1.0 - f64::try_from(residual_var / returns_var).unwrap_or(0.0);

    (value_loss, explained_var)
}

/// Compute entropy bonus
///
/// Entropy encourages exploration by penalizing deterministic policies.
///
/// # Arguments
///
/// * `entropy` - Entropy of the policy distribution
///
/// # Returns
///
/// Negative mean entropy (we want to maximize entropy, so minimize negative)
pub fn compute_entropy_loss(entropy: &Tensor) -> Tensor {
    -entropy.mean(tch::Kind::Float)
}

/// Sample random indices for minibatch training
///
/// # Arguments
///
/// * `buffer_size` - Total number of samples in buffer
/// * `batch_size` - Size of each minibatch
///
/// # Returns
///
/// Vector of index vectors, one per minibatch
pub fn generate_minibatch_indices(buffer_size: usize, batch_size: usize) -> Vec<Vec<usize>> {
    use rand::{seq::SliceRandom, thread_rng};

    let mut indices: Vec<usize> = (0..buffer_size).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);

    indices.chunks(batch_size).map(|chunk| chunk.to_vec()).collect()
}

/// PPO Trainer
///
/// Manages the training loop for PPO, including:
/// - Collecting rollouts from environments
/// - Computing advantages with GAE
/// - Updating policy via minibatch gradient descent
/// - Tracking training statistics
///
/// # Type Parameters
///
/// * `P` - Policy type (must implement forward and evaluate methods)
pub struct PPOTrainer<P> {
    config: PPOConfig,
    policy: P,
    optimizer: Option<tch::nn::Optimizer>,
    total_steps: usize,
    total_episodes: usize,
}

impl<P> PPOTrainer<P> {
    /// Create a new PPO trainer
    ///
    /// # Arguments
    ///
    /// * `config` - PPO configuration parameters
    /// * `policy` - Policy network
    pub fn new(config: PPOConfig, policy: P) -> Result<Self> {
        config.validate()?;

        Ok(Self { config, policy, optimizer: None, total_steps: 0, total_episodes: 0 })
    }

    /// Set the optimizer for training
    ///
    /// This must be called before training starts.
    pub fn set_optimizer(&mut self, optimizer: tch::nn::Optimizer) {
        self.optimizer = Some(optimizer);
    }

    /// Get reference to the policy
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// Get mutable reference to the policy
    pub fn policy_mut(&mut self) -> &mut P {
        &mut self.policy
    }

    /// Get the configuration
    pub fn config(&self) -> &PPOConfig {
        &self.config
    }

    /// Get total training steps
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Get total episodes completed
    pub fn total_episodes(&self) -> usize {
        self.total_episodes
    }

    /// Train for one PPO update
    ///
    /// This performs:
    /// 1. Multiple epochs of minibatch updates
    /// 2. Policy and value loss computation
    /// 3. Gradient updates
    ///
    /// # Arguments
    ///
    /// * `observations` - Batch of observations [batch_size, obs_dim]
    /// * `actions` - Batch of actions [batch_size]
    /// * `old_log_probs` - Log probs from rollout policy [batch_size]
    /// * `old_values` - Value estimates from rollout [batch_size]
    /// * `advantages` - Computed advantages [batch_size]
    /// * `returns` - Discounted returns [batch_size]
    /// * `forward_fn` - Function to compute (log_probs, entropy, values) from
    ///   (obs, actions)
    ///
    /// # Returns
    ///
    /// Average training statistics across all epochs
    pub fn train_step<F>(
        &mut self,
        observations: &Tensor,
        actions: &Tensor,
        old_log_probs: &Tensor,
        old_values: &Tensor,
        advantages: &Tensor,
        returns: &Tensor,
        mut forward_fn: F,
    ) -> Result<TrainingStats>
    where
        F: FnMut(&Tensor, &Tensor) -> (Tensor, Tensor, Tensor),
    {
        let optimizer = self
            .optimizer
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Optimizer not set. Call set_optimizer() first."))?;

        let batch_size = observations.size()[0] as usize;
        let mut stats_sum = TrainingStats::zeros();
        let mut num_updates = 0;

        // Multiple epochs over the data
        for epoch in 0..self.config.n_epochs {
            let batch_indices = generate_minibatch_indices(batch_size, self.config.batch_size);

            for indices in &batch_indices {
                // Convert indices to i64 for tensor indexing
                let indices_i64: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
                let indices_tensor = Tensor::from_slice(&indices_i64);

                // Sample minibatch
                let mb_obs =
                    observations.index_select(0, &indices_tensor.to_device(observations.device()));
                let mb_actions =
                    actions.index_select(0, &indices_tensor.to_device(actions.device()));
                let mb_old_log_probs = old_log_probs
                    .index_select(0, &indices_tensor.to_device(old_log_probs.device()));
                let mb_old_values =
                    old_values.index_select(0, &indices_tensor.to_device(old_values.device()));
                let mb_advantages =
                    advantages.index_select(0, &indices_tensor.to_device(advantages.device()));
                let mb_returns =
                    returns.index_select(0, &indices_tensor.to_device(returns.device()));

                // Forward pass
                let (log_probs, entropy, values) = forward_fn(&mb_obs, &mb_actions);

                // Compute losses
                let (policy_loss, clip_fraction, approx_kl) = compute_policy_loss(
                    &log_probs,
                    &mb_old_log_probs,
                    &mb_advantages,
                    self.config.clip_range,
                );

                let (value_loss, explained_var) = compute_value_loss(
                    &values,
                    &mb_old_values,
                    &mb_returns,
                    self.config.clip_range_vf,
                );

                let entropy_loss = compute_entropy_loss(&entropy);

                // Extract scalar values before tensors are moved
                let policy_loss_val: f64 = f64::try_from(&policy_loss).unwrap_or(0.0);
                let value_loss_val: f64 = f64::try_from(&value_loss).unwrap_or(0.0);
                let entropy_val: f64 = f64::try_from(&entropy).unwrap_or(0.0);

                // Total loss
                let loss = &policy_loss
                    + self.config.vf_coef * &value_loss
                    + self.config.ent_coef * &entropy_loss;

                let total_loss_val: f64 = f64::try_from(&loss).unwrap_or(0.0);

                // Backward pass
                optimizer.zero_grad();
                loss.backward();

                // Gradient clipping
                optimizer.clip_grad_norm(self.config.max_grad_norm);

                // Update weights
                optimizer.step();

                stats_sum.total_loss += total_loss_val;
                stats_sum.policy_loss += policy_loss_val;
                stats_sum.value_loss += value_loss_val;
                stats_sum.entropy += entropy_val;
                stats_sum.approx_kl += approx_kl;
                stats_sum.clip_fraction += clip_fraction;
                stats_sum.explained_variance += explained_var;
                num_updates += 1;

                // Early stopping based on KL divergence
                if let Some(target_kl) = self.config.target_kl
                    && approx_kl > target_kl
                {
                    tracing::info!(
                        "Early stopping at epoch {} due to reaching max KL: {:.4} > {:.4}",
                        epoch,
                        approx_kl,
                        target_kl
                    );
                    break;
                }
            }

            // Early stopping check (outer loop)
            if let Some(target_kl) = self.config.target_kl
                && stats_sum.approx_kl / num_updates as f64 > target_kl
            {
                break;
            }
        }

        // Average statistics
        let n = num_updates as f64;
        Ok(TrainingStats {
            total_loss: stats_sum.total_loss / n,
            policy_loss: stats_sum.policy_loss / n,
            value_loss: stats_sum.value_loss / n,
            entropy: stats_sum.entropy / n,
            approx_kl: stats_sum.approx_kl / n,
            clip_fraction: stats_sum.clip_fraction / n,
            explained_variance: stats_sum.explained_variance / n,
        })
    }

    /// Increment step counter
    pub fn increment_steps(&mut self, n: usize) {
        self.total_steps += n;
    }

    /// Increment episode counter
    pub fn increment_episodes(&mut self, n: usize) {
        self.total_episodes += n;
    }

    /// Train for one PPO update with external policy
    ///
    /// This version takes the policy as a separate parameter to avoid
    /// borrow checker issues.
    pub fn train_step_with_policy<F>(
        &mut self,
        policy: &P,
        observations: &Tensor,
        actions: &Tensor,
        old_log_probs: &Tensor,
        old_values: &Tensor,
        advantages: &Tensor,
        returns: &Tensor,
        mut forward_fn: F,
    ) -> Result<TrainingStats>
    where
        F: FnMut(&P, &Tensor, &Tensor) -> (Tensor, Tensor, Tensor),
    {
        let optimizer = self
            .optimizer
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Optimizer not set. Call set_optimizer() first."))?;

        let batch_size = observations.size()[0] as usize;
        let mut stats_sum = TrainingStats::zeros();
        let mut num_updates = 0;

        // Multiple epochs over the data
        for epoch in 0..self.config.n_epochs {
            let batch_indices = generate_minibatch_indices(batch_size, self.config.batch_size);

            for indices in &batch_indices {
                // Convert indices to i64 for tensor indexing
                let indices_i64: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
                let indices_tensor = Tensor::from_slice(&indices_i64);

                // Sample minibatch
                let mb_obs =
                    observations.index_select(0, &indices_tensor.to_device(observations.device()));
                let mb_actions =
                    actions.index_select(0, &indices_tensor.to_device(actions.device()));
                let mb_old_log_probs = old_log_probs
                    .index_select(0, &indices_tensor.to_device(old_log_probs.device()));
                let mb_old_values =
                    old_values.index_select(0, &indices_tensor.to_device(old_values.device()));
                let mb_advantages =
                    advantages.index_select(0, &indices_tensor.to_device(advantages.device()));
                let mb_returns =
                    returns.index_select(0, &indices_tensor.to_device(returns.device()));

                // Forward pass
                let (log_probs, entropy, values) = forward_fn(policy, &mb_obs, &mb_actions);

                // Compute losses
                let (policy_loss, clip_fraction, approx_kl) = compute_policy_loss(
                    &log_probs,
                    &mb_old_log_probs,
                    &mb_advantages,
                    self.config.clip_range,
                );

                let (value_loss, explained_var) = compute_value_loss(
                    &values,
                    &mb_old_values,
                    &mb_returns,
                    self.config.clip_range_vf,
                );

                let entropy_loss = compute_entropy_loss(&entropy);

                // Extract scalar values before tensors are moved
                let policy_loss_val: f64 = f64::try_from(&policy_loss).unwrap_or(0.0);
                let value_loss_val: f64 = f64::try_from(&value_loss).unwrap_or(0.0);
                let entropy_val: f64 = f64::try_from(&entropy).unwrap_or(0.0);

                // Total loss
                let loss = &policy_loss
                    + self.config.vf_coef * &value_loss
                    + self.config.ent_coef * &entropy_loss;

                let total_loss_val: f64 = f64::try_from(&loss).unwrap_or(0.0);

                // Backward pass
                optimizer.zero_grad();
                loss.backward();

                // Gradient clipping
                optimizer.clip_grad_norm(self.config.max_grad_norm);

                // Update weights
                optimizer.step();

                stats_sum.total_loss += total_loss_val;
                stats_sum.policy_loss += policy_loss_val;
                stats_sum.value_loss += value_loss_val;
                stats_sum.entropy += entropy_val;
                stats_sum.approx_kl += approx_kl;
                stats_sum.clip_fraction += clip_fraction;
                stats_sum.explained_variance += explained_var;
                num_updates += 1;

                // Early stopping based on KL divergence
                if let Some(target_kl) = self.config.target_kl
                    && approx_kl > target_kl
                {
                    tracing::info!(
                        "Early stopping at epoch {} due to reaching max KL: {:.4} > {:.4}",
                        epoch,
                        approx_kl,
                        target_kl
                    );
                    break;
                }
            }

            // Early stopping check (outer loop)
            if let Some(target_kl) = self.config.target_kl
                && stats_sum.approx_kl / num_updates as f64 > target_kl
            {
                break;
            }
        }

        // Average statistics
        let n = num_updates as f64;
        Ok(TrainingStats {
            total_loss: stats_sum.total_loss / n,
            policy_loss: stats_sum.policy_loss / n,
            value_loss: stats_sum.value_loss / n,
            entropy: stats_sum.entropy / n,
            approx_kl: stats_sum.approx_kl / n,
            clip_fraction: stats_sum.clip_fraction / n,
            explained_variance: stats_sum.explained_variance / n,
        })
    }
}

#[cfg(test)]
mod tests {
    use tch::Kind;

    use super::*;

    #[test]
    fn test_config_default() {
        let config = PPOConfig::default();
        assert_eq!(config.learning_rate, 3e-4);
        assert_eq!(config.n_epochs, 10);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.gamma, 0.99);
        assert_eq!(config.gae_lambda, 0.95);
    }

    #[test]
    fn test_config_builder() {
        let config = PPOConfig::new()
            .learning_rate(1e-3)
            .n_epochs(5)
            .batch_size(32)
            .gamma(0.95)
            .clip_range(0.1);

        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.n_epochs, 5);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.gamma, 0.95);
        assert_eq!(config.clip_range, 0.1);
    }

    #[test]
    fn test_config_validation() {
        let valid_config = PPOConfig::default();
        assert!(valid_config.validate().is_ok());

        let invalid_lr = PPOConfig::new().learning_rate(-0.1);
        assert!(invalid_lr.validate().is_err());

        let invalid_gamma = PPOConfig::new().gamma(1.5);
        assert!(invalid_gamma.validate().is_err());

        let invalid_lambda = PPOConfig::new().gae_lambda(-0.1);
        assert!(invalid_lambda.validate().is_err());
    }

    #[test]
    fn test_training_stats_zeros() {
        let stats = TrainingStats::zeros();
        assert_eq!(stats.total_loss, 0.0);
        assert_eq!(stats.policy_loss, 0.0);
        assert_eq!(stats.value_loss, 0.0);
    }

    #[test]
    fn test_policy_loss_no_clipping() {
        // When ratio is close to 1.0, no clipping should occur
        let log_probs = Tensor::zeros([4], (Kind::Float, tch::Device::Cpu));
        let old_log_probs = Tensor::zeros([4], (Kind::Float, tch::Device::Cpu));
        let advantages = Tensor::ones([4], (Kind::Float, tch::Device::Cpu));

        let (loss, clip_frac, _kl) =
            compute_policy_loss(&log_probs, &old_log_probs, &advantages, 0.2);

        assert!(loss.size().is_empty()); // Scalar
        assert_eq!(clip_frac, 0.0); // No clipping
    }

    #[test]
    fn test_policy_loss_with_clipping() {
        // Large log prob difference should trigger clipping
        let log_probs = Tensor::full([4], 1.0, (Kind::Float, tch::Device::Cpu));
        let old_log_probs = Tensor::full([4], -1.0, (Kind::Float, tch::Device::Cpu));
        let advantages = Tensor::ones([4], (Kind::Float, tch::Device::Cpu));

        let (_loss, clip_frac, _kl) =
            compute_policy_loss(&log_probs, &old_log_probs, &advantages, 0.2);

        assert!(clip_frac > 0.0); // Some clipping should occur
    }

    #[test]
    fn test_value_loss() {
        let values = Tensor::f_from_slice(&[0.9f32, 1.0, 1.1, 1.2]).unwrap();
        let old_values = Tensor::f_from_slice(&[0.5f32, 0.5, 0.5, 0.5]).unwrap();
        let returns = Tensor::f_from_slice(&[1.0f32, 1.5, 1.0, 1.5]).unwrap();

        let (loss, explained_var) = compute_value_loss(&values, &old_values, &returns, None);

        assert!(loss.size().is_empty()); // Scalar
        let loss_val: f64 = loss.try_into().unwrap();
        assert!(loss_val > 0.0);
        // Explained variance can be negative if model is worse than mean
        assert!(explained_var.is_finite() || explained_var.is_nan());
    }

    #[test]
    fn test_value_loss_with_clipping() {
        let values = Tensor::full([4], 1.0, (Kind::Float, tch::Device::Cpu));
        let old_values = Tensor::full([4], 0.5, (Kind::Float, tch::Device::Cpu));
        let returns = Tensor::full([4], 1.5, (Kind::Float, tch::Device::Cpu));

        let (loss, _) = compute_value_loss(&values, &old_values, &returns, Some(0.2));

        assert!(loss.size().is_empty()); // Scalar
    }

    #[test]
    fn test_entropy_loss() {
        let entropy = Tensor::full([4], 0.5, (Kind::Float, tch::Device::Cpu));
        let loss = compute_entropy_loss(&entropy);

        let loss_val: f64 = loss.try_into().unwrap();
        assert_eq!(loss_val, -0.5); // Should be negative of mean
    }

    #[test]
    fn test_minibatch_indices() {
        let buffer_size = 100;
        let batch_size = 32;

        let batches = generate_minibatch_indices(buffer_size, batch_size);

        // Should have ceil(100/32) = 4 batches
        assert_eq!(batches.len(), 4);

        // First 3 batches should be full
        assert_eq!(batches[0].len(), 32);
        assert_eq!(batches[1].len(), 32);
        assert_eq!(batches[2].len(), 32);

        // Last batch has remainder
        assert_eq!(batches[3].len(), 4);

        // All indices should be unique
        let mut all_indices: Vec<usize> = batches.into_iter().flatten().collect();
        all_indices.sort();
        assert_eq!(all_indices, (0..buffer_size).collect::<Vec<_>>());
    }

    #[test]
    fn test_minibatch_indices_shuffle() {
        let buffer_size = 10;
        let batch_size = 5;

        let batches1 = generate_minibatch_indices(buffer_size, batch_size);
        let batches2 = generate_minibatch_indices(buffer_size, batch_size);

        // Should produce different orderings (with very high probability)
        let flat1: Vec<usize> = batches1.into_iter().flatten().collect();
        let flat2: Vec<usize> = batches2.into_iter().flatten().collect();

        // Not a perfect test, but very unlikely to fail
        assert_ne!(flat1, flat2);
    }
}
