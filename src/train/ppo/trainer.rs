//! PPO Trainer implementation
//!
//! This module contains the main PPOTrainer struct and its training methods.

use super::{config::PPOConfig, loss::*, stats::TrainingStats};
use anyhow::{anyhow, Result};
use tch::Tensor;

/// PPO Trainer for policy optimization
///
/// Manages the training process including optimizer setup,
/// gradient computation, and parameter updates.
#[derive(Debug)]
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

        Ok(Self {
            config,
            policy,
            optimizer: None,
            total_steps: 0,
            total_episodes: 0,
        })
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
    /// 3. Gradient descent parameter updates
    ///
    /// # Arguments
    ///
    /// * `observations` - Observation tensor [batch_size, obs_dim]
    /// * `actions` - Action tensor [batch_size]
    /// * `old_log_probs` - Old log probabilities [batch_size]
    /// * `old_values` - Old value estimates [batch_size]
    /// * `advantages` - Computed advantages [batch_size]
    /// * `returns` - Computed returns [batch_size]
    /// * `forward_fn` - Function that performs forward pass through policy
    ///
    /// # Returns
    /// Training statistics for this update
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
            .ok_or_else(|| anyhow!("Optimizer not set. Call set_optimizer() first."))?;

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
                let mb_obs = observations
                    .index_select(0, &indices_tensor.to_device(observations.device()));
                let mb_actions = actions
                    .index_select(0, &indices_tensor.to_device(actions.device()));
                let mb_old_log_probs = old_log_probs
                    .index_select(0, &indices_tensor.to_device(old_log_probs.device()));
                let mb_old_values = old_values
                    .index_select(0, &indices_tensor.to_device(old_values.device()));
                let mb_advantages = advantages
                    .index_select(0, &indices_tensor.to_device(advantages.device()));
                let mb_returns = returns
                    .index_select(0, &indices_tensor.to_device(returns.device()));

                // Normalize advantages at minibatch level (SB3-style)
                let adv_mean = mb_advantages.mean(tch::Kind::Float);
                let adv_std = mb_advantages.std(false);
                let mb_advantages = (&mb_advantages - adv_mean) / (adv_std + 1e-8);

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

                // Optimizer step
                optimizer.step();

                // Accumulate statistics
                let step_stats = TrainingStats::new(
                    policy_loss_val,
                    value_loss_val,
                    entropy_val,
                    total_loss_val,
                    clip_fraction,
                    approx_kl,
                    explained_var,
                );
                stats_sum.add(&step_stats);
                num_updates += 1;

                // Early stopping based on KL divergence
                if approx_kl > self.config.target_kl {
                    break;
                }
            }
        }

        // Update training counters
        self.total_steps += num_updates;

        // Return average statistics
        Ok(stats_sum.average())
    }

    /// Train for one PPO update with explicit policy reference
    ///
    /// This is a convenience method that calls train_step with a closure
    /// that evaluates the policy.
    ///
    /// # Arguments
    ///
    /// * `policy` - Policy to evaluate
    /// * `observations` - Observation tensor [batch_size, obs_dim]
    /// * `actions` - Action tensor [batch_size]
    /// * `old_log_probs` - Old log probabilities [batch_size]
    /// * `old_values` - Old value estimates [batch_size]
    /// * `advantages` - Computed advantages [batch_size]
    /// * `returns` - Computed returns [batch_size]
    /// * `evaluate_fn` - Function that evaluates policy at (obs, actions)
    ///
    /// # Returns
    /// Training statistics for this update
    pub fn train_step_with_policy<Policy, F>(
        &mut self,
        policy: &Policy,
        observations: &Tensor,
        actions: &Tensor,
        old_log_probs: &Tensor,
        old_values: &Tensor,
        advantages: &Tensor,
        returns: &Tensor,
        evaluate_fn: F,
    ) -> Result<TrainingStats>
    where
        F: Fn(&Policy, &Tensor, &Tensor) -> (Tensor, Tensor, Tensor),
    {
        self.train_step(
            observations,
            actions,
            old_log_probs,
            old_values,
            advantages,
            returns,
            |obs, acts| evaluate_fn(policy, obs, acts),
        )
    }

    /// Increment step counter
    pub fn increment_steps(&mut self, steps: usize) {
        self.total_steps += steps;
    }

    /// Increment episode counter
    pub fn increment_episodes(&mut self, episodes: usize) {
        self.total_episodes += episodes;
    }
}
