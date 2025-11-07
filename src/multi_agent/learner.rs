//! Policy learner thread for multi-agent training
//!
//! Each learner trains a single agent's policy using PPO.

use std::time::Duration;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use tch::Tensor;

use super::{
    messages::{Experience, PolicyUpdate, TrainingStats},
    population::AgentId,
};
use crate::{buffer::rollout::RolloutBuffer, policy::mlp::MlpPolicy, train::ppo::PPOTrainer};

/// Policy learner - trains one agent's policy
///
/// This component runs in its own thread and is responsible for:
/// - Receiving experiences from the game simulator
/// - Computing advantages using GAE
/// - Updating policy parameters using PPO
/// - Periodically sending updated weights back to simulator
pub struct PolicyLearner {
    /// Agent ID this learner is training
    pub agent_id: AgentId,

    /// PPO trainer (owns the policy network)
    trainer: PPOTrainer<MlpPolicy>,

    /// Receive experiences from simulator
    experience_receiver: Receiver<Experience>,

    /// Send policy updates to simulator
    policy_sender: Sender<PolicyUpdate>,

    /// Local experience buffer
    buffer: RolloutBuffer,

    /// Training configuration
    config: LearnerConfig,

    /// Training step counter
    step: usize,

    /// Path for saving models
    model_save_dir: String,
}

impl PolicyLearner {
    /// Create a new policy learner
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        agent_id: AgentId,
        policy: MlpPolicy,
        experience_receiver: Receiver<Experience>,
        policy_sender: Sender<PolicyUpdate>,
        config: LearnerConfig,
        model_save_dir: String,
    ) -> Result<Self> {
        // Create optimizer from policy before moving it to trainer
        use tch::nn::{self, OptimizerConfig};
        let vs = &policy.var_store();
        let optimizer = nn::Adam::default().build(vs, config.learning_rate)?;

        // Create trainer with policy
        let mut trainer = PPOTrainer::new(config.clone().into(), policy)?;
        trainer.set_optimizer(optimizer);

        let buffer = RolloutBuffer::new(
            config.buffer_size,
            1, // num_envs
            4, // obs_dim - TODO: make configurable
        );

        Ok(Self {
            agent_id,
            trainer,
            experience_receiver,
            policy_sender,
            buffer,
            config,
            step: 0,
            model_save_dir,
        })
    }

    /// Main training loop
    pub fn train(mut self) -> Result<()> {
        tracing::info!("Learner {} starting training", self.agent_id);

        loop {
            // 1. Collect experiences until buffer is full or timeout
            if let Err(e) = self.collect_experiences() {
                tracing::warn!("Agent {} experience collection error: {}", self.agent_id, e);
                continue;
            }

            // 2. Check if we have enough data to train
            if self.buffer.len() < self.config.min_batch_size {
                continue;
            }

            // 3. Compute advantages
            // TODO: Fix API - compute_advantages needs last_values
            // self.buffer.compute_advantages(&last_values, self.config.gamma,
            // self.config.gae_lambda);
            let last_values = vec![0.0]; // Placeholder
            self.buffer.compute_advantages(
                &last_values,
                self.config.gamma as f32,
                self.config.gae_lambda as f32,
            );

            // 4. Train on batch
            let stats = self.train_step()?;

            // 5. Clear buffer for next batch
            self.buffer.reset();

            self.step += 1;

            // 6. Log progress
            if self.step % 10 == 0 {
                tracing::info!(
                    "Agent {} | Step {} | Loss: {:.3} | Policy: {:.3} | Entropy: {:.3}",
                    self.agent_id,
                    self.step,
                    stats.total_loss,
                    stats.policy_loss,
                    stats.entropy,
                );
            }

            // 7. Periodically send policy update to simulator
            if self.step % self.config.update_interval == 0 {
                if let Err(e) = self.send_policy_update(stats) {
                    tracing::warn!("Agent {} failed to send policy update: {}", self.agent_id, e);
                }
            }
        }
    }

    /// Collect experiences from simulator
    fn collect_experiences(&mut self) -> Result<()> {
        let timeout = Duration::from_millis(100);
        let start_len = self.buffer.len();
        // TODO: Fix - buffer doesn't have capacity() method
        let target_len = self.config.buffer_size;

        // Try to fill buffer, but don't block forever
        while self.buffer.len() < target_len {
            match self.experience_receiver.recv_timeout(timeout) {
                Ok(_exp) => {
                    // TODO: Fix - buffer.add() API doesn't match
                    // Need to adapt Experience format to buffer's expected
                    // format For now, just count the
                    // received experience self.buffer.add(.
                    // ..)?;
                }
                Err(_) => {
                    // Timeout - check if we have enough data
                    if self.buffer.len() > start_len {
                        break; // Got some new data, good enough
                    }
                }
            }
        }

        Ok(())
    }

    /// Run PPO training step
    fn train_step(&mut self) -> Result<TrainingStats> {
        // Get batch from buffer
        let batch = self.buffer.get_batch();

        // Convert Vec data to Tensors
        let device = self.trainer.policy().device();
        let batch_size = batch.actions.len() as i64;
        let obs_dim = (batch.observations.len() / batch.actions.len()) as i64;

        // Observations are already flattened: Vec<f32> -> Tensor [batch_size, obs_dim]
        let observations = Tensor::from_slice(&batch.observations)
            .view([batch_size, obs_dim])
            .to_device(device);

        let actions = Tensor::from_slice(&batch.actions).to_device(device);
        let old_log_probs = Tensor::from_slice(&batch.old_log_probs).to_device(device);
        let old_values = Tensor::from_slice(&batch.old_values).to_device(device);
        let advantages = Tensor::from_slice(&batch.advantages).to_device(device);
        let returns = Tensor::from_slice(&batch.returns).to_device(device);

        // Train for multiple epochs
        // Note: We can't use train_step() because it requires both &self.policy and
        // &mut self.trainer Instead, we'll use a workaround by calling the
        // trainer directly with its own policy
        let mut total_stats = TrainingStats::default();
        for _ in 0..self.config.n_epochs {
            // Safety: The trainer owns the policy, so this is safe as long as we don't
            // call any methods that would try to borrow trainer mutably during policy
            // access
            let trainer_ptr: *mut PPOTrainer<MlpPolicy> = &mut self.trainer;
            let policy_ptr: *const MlpPolicy = unsafe { &*trainer_ptr }.policy();

            let stats = unsafe {
                (*trainer_ptr).train_step_with_policy(
                    &*policy_ptr,
                    &observations,
                    &actions,
                    &old_log_probs,
                    &old_values,
                    &advantages,
                    &returns,
                    |policy: &MlpPolicy, obs: &Tensor, acts: &Tensor| {
                        policy.evaluate_actions(obs, acts)
                    },
                )?
            };

            // Accumulate stats
            total_stats.total_loss += stats.total_loss;
            total_stats.policy_loss += stats.policy_loss;
            total_stats.value_loss += stats.value_loss;
            total_stats.entropy += stats.entropy;
        }

        // Average over epochs
        let n = self.config.n_epochs as f64;
        total_stats.total_loss /= n;
        total_stats.policy_loss /= n;
        total_stats.value_loss /= n;
        total_stats.entropy /= n;
        total_stats.step = self.step;

        Ok(total_stats)
    }

    /// Send policy update to simulator
    fn send_policy_update(&mut self, stats: TrainingStats) -> Result<()> {
        // Save model to file
        let model_path =
            format!("{}/agent_{}_step_{}.pt", self.model_save_dir, self.agent_id, self.step);
        self.trainer.policy().save(&model_path)?;

        // Create update message
        let update =
            PolicyUpdate { agent_id: self.agent_id, version: self.step as u64, model_path, stats };

        // Send (non-blocking)
        self.policy_sender
            .try_send(update)
            .map_err(|e| anyhow::anyhow!("Failed to send policy update: {}", e))?;

        Ok(())
    }
}

/// Configuration for policy learner
#[derive(Debug, Clone)]
pub struct LearnerConfig {
    /// Learning rate
    pub learning_rate: f64,

    /// Discount factor
    pub gamma: f64,

    /// GAE lambda
    pub gae_lambda: f64,

    /// PPO clip epsilon
    pub clip_epsilon: f64,

    /// Value loss coefficient
    pub value_loss_coef: f64,

    /// Entropy bonus coefficient
    pub entropy_coef: f64,

    /// Buffer size
    pub buffer_size: usize,

    /// Minimum batch size before training
    pub min_batch_size: usize,

    /// Number of PPO epochs per batch
    pub n_epochs: usize,

    /// Update interval (send policy updates every N steps)
    pub update_interval: usize,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
            buffer_size: 2048,
            min_batch_size: 256,
            n_epochs: 4,
            update_interval: 10,
        }
    }
}

// Convert LearnerConfig to PPOConfig
impl From<LearnerConfig> for crate::train::ppo::PPOConfig {
    fn from(config: LearnerConfig) -> Self {
        crate::train::ppo::PPOConfig {
            learning_rate: config.learning_rate,
            n_epochs: config.n_epochs,
            batch_size: config.min_batch_size,
            gamma: config.gamma,
            gae_lambda: config.gae_lambda,
            clip_range: config.clip_epsilon,
            clip_range_vf: 0.2, // Use default value
            vf_coef: config.value_loss_coef,
            ent_coef: config.entropy_coef,
            max_grad_norm: 0.5,
            target_kl: 0.01, // Use default value
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learner_config_default() {
        let config = LearnerConfig::default();
        assert_eq!(config.learning_rate, 3e-4);
        assert_eq!(config.gamma, 0.99);
        assert_eq!(config.buffer_size, 2048);
    }

    #[test]
    fn test_learner_creation() {
        let policy = MlpPolicy::new(4, 2, 64);
        let (_exp_sender, exp_receiver) = crossbeam_channel::unbounded();
        let (policy_sender, _policy_receiver) = crossbeam_channel::unbounded();
        let config = LearnerConfig::default();

        let learner =
            PolicyLearner::new(0, policy, exp_receiver, policy_sender, config, "/tmp".to_string());

        assert!(learner.is_ok());
        assert_eq!(learner.unwrap().agent_id, 0);
    }
}
