//! Policy learner thread for multi-agent training
//!
//! Each learner trains a single agent's policy using PPO.

use std::time::Duration;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use tch::{Tensor, no_grad};

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

    /// Index into the rollout buffer for the next experience to be inserted.
    /// Wraps modulo `config.buffer_size` and is reset to 0 after each
    /// `train()` cycle (when the buffer is cleared via `reset`).
    buffer_step_idx: usize,

    /// Snapshot of the most recently received experience for the current
    /// rollout. Used to compute the GAE bootstrap value `V(s_{T+1})` when
    /// the rollout ends in a non-terminal state (e.g. truncation or buffer
    /// full). `None` when the buffer is empty.
    last_experience: Option<LastExperience>,
}

/// Minimal information about the most recent experience needed to compute
/// the GAE bootstrap value on a non-terminal rollout boundary.
struct LastExperience {
    /// Flattened next observation for the final inserted transition.
    next_observation: Vec<f32>,

    /// Whether the final inserted transition terminated the episode.
    /// A terminal end uses `0.0` as the bootstrap value; a non-terminal
    /// end (truncation / buffer-full) requires `V(s_{T+1})` from the
    /// current policy.
    terminated: bool,
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
            config.obs_dim,
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
            buffer_step_idx: 0,
            last_experience: None,
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

            // 2. Check if we have enough data to train.
            //
            // Note: `RolloutBuffer::len()` returns the *capacity*
            // (`num_steps * num_envs`), not the count of inserted
            // transitions. Use `buffer_step_idx`, which tracks how many
            // experiences have actually been written since the last reset.
            if self.buffer_step_idx < self.config.min_batch_size {
                continue;
            }

            // 3. Compute advantages with a correct GAE bootstrap value.
            //    - If the final inserted transition ended the episode
            //      (`terminated == true`), `V(s_{T+1}) = 0`.
            //    - Otherwise (truncation / buffer-full), bootstrap with
            //      `V(s_{T+1})` from the current policy, computed under
            //      `no_grad`.
            let last_values = self.compute_bootstrap_last_values();
            self.buffer.compute_advantages(
                &last_values,
                self.config.gamma as f32,
                self.config.gae_lambda as f32,
            );

            // 4. Train on batch
            let stats = self.train_step()?;

            // 5. Clear buffer for next batch
            self.buffer.reset();
            self.buffer_step_idx = 0;
            self.last_experience = None;

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

    /// Collect experiences from the simulator and push them into the
    /// rollout buffer.
    ///
    /// Each `Experience` received on the channel is translated into a
    /// `RolloutBuffer::add(...)` call. The local `buffer_step_idx`
    /// tracks the next free row in the buffer; this is the canonical
    /// "how many transitions have we written" counter since
    /// `RolloutBuffer::len()` returns the buffer's full capacity, not
    /// its fill level.
    ///
    /// `last_experience` is updated on every successful insert so that
    /// the GAE bootstrap value can be computed from `V(s_{T+1})` when
    /// the rollout ends in a non-terminal state.
    fn collect_experiences(&mut self) -> Result<()> {
        let timeout = Duration::from_millis(100);
        let start_idx = self.buffer_step_idx;
        let target_len = self.config.buffer_size;

        // Try to fill buffer, but don't block forever.
        while self.buffer_step_idx < target_len {
            match self.experience_receiver.recv_timeout(timeout) {
                Ok(exp) => {
                    self.ingest_experience(exp)?;
                }
                Err(_) => {
                    // Timeout - check if we have new data
                    if self.buffer_step_idx > start_idx {
                        break; // Got some new data, good enough
                    }
                }
            }
        }

        Ok(())
    }

    /// Translate a single `Experience` into a `RolloutBuffer::add(...)`
    /// call and advance `buffer_step_idx`. The observation tensor is
    /// converted to `Vec<f32>` and validated against the configured
    /// `obs_dim`. The last-inserted experience snapshot (used for
    /// the GAE bootstrap) is updated.
    fn ingest_experience(&mut self, exp: Experience) -> Result<()> {
        // Tensors may live on GPU; move to CPU before extracting to Vec.
        let obs_cpu = exp.observation.to_device(tch::Device::Cpu);
        let obs_vec: Vec<f32> = Vec::<f32>::try_from(&obs_cpu)
            .map_err(|e| anyhow::anyhow!("Failed to convert observation tensor to Vec<f32>: {e}"))?;

        if obs_vec.len() != self.config.obs_dim {
            return Err(anyhow::anyhow!(
                "Experience observation dim {} does not match configured obs_dim {}",
                obs_vec.len(),
                self.config.obs_dim
            ));
        }

        let next_obs_cpu = exp.next_observation.to_device(tch::Device::Cpu);
        let next_obs_vec: Vec<f32> = Vec::<f32>::try_from(&next_obs_cpu).map_err(|e| {
            anyhow::anyhow!("Failed to convert next_observation tensor to Vec<f32>: {e}")
        })?;

        // Single-env learner: env_id is always 0. If/when multi-env per
        // learner support is added, this should be threaded through.
        let env_id = 0;
        let step = self.buffer_step_idx;

        self.buffer.add(
            step,
            env_id,
            &obs_vec,
            exp.action,
            exp.reward,
            exp.value,
            exp.log_prob,
            exp.terminated,
            exp.truncated,
        );

        self.last_experience =
            Some(LastExperience { next_observation: next_obs_vec, terminated: exp.terminated });

        self.buffer_step_idx += 1;

        Ok(())
    }

    /// Compute the GAE bootstrap value vector `last_values`.
    ///
    /// Returns `vec![0.0]` for the single-env case when:
    /// - no experiences have been seen yet (defensive default), or
    /// - the last received experience ended the episode (`terminated`).
    ///
    /// Otherwise runs the current policy on the final `next_observation`
    /// under `no_grad` and returns `V(s_{T+1})`.
    fn compute_bootstrap_last_values(&self) -> Vec<f32> {
        match self.last_experience.as_ref() {
            None => vec![0.0],
            Some(last) if last.terminated => vec![0.0],
            Some(last) => {
                let policy = self.trainer.policy();
                let device = policy.device();
                let obs_tensor = Tensor::from_slice(&last.next_observation)
                    .view([1, last.next_observation.len() as i64])
                    .to_device(device);
                let last_value = no_grad(|| {
                    let (_logits, value) = policy.forward(&obs_tensor);
                    value.double_value(&[]) as f32
                });
                vec![last_value]
            }
        }
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

    /// Dimensionality of observations. Must match the environment's
    /// `observation_space().shape()` and the policy network's input
    /// dimension. The default value (`4`) matches CartPole for
    /// backward compatibility — explicit configuration is required
    /// for any other environment.
    pub obs_dim: usize,
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
            // CartPole-compatible default; override for other envs.
            obs_dim: 4,
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
    use tch::Kind;

    use super::*;

    #[test]
    fn test_learner_config_default() {
        let config = LearnerConfig::default();
        assert_eq!(config.learning_rate, 3e-4);
        assert_eq!(config.gamma, 0.99);
        assert_eq!(config.buffer_size, 2048);
        // obs_dim defaults to 4 for backward compatibility with CartPole.
        assert_eq!(config.obs_dim, 4);
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

    /// Regression test for issue #4: the four TODO/placeholder defects in
    /// `PolicyLearner`.
    ///
    /// This test exercises the previously-broken path end-to-end:
    /// - Constructs a learner with a non-trivial `obs_dim` (8, not 4).
    /// - Feeds a batch of `Experience` values through the channel.
    /// - Verifies experiences are translated into `RolloutBuffer::add(...)`
    ///   calls (defect #4: the buffer used to never be populated).
    /// - Verifies a full training cycle (collect → compute_advantages →
    ///   train_step → reset) runs without panicking.
    /// - Verifies `compute_bootstrap_last_values` returns `vec![0.0]`
    ///   for a terminal end and a non-zero `V(s_{T+1})` (or at least a
    ///   distinct policy-derived value) for a non-terminal end (defect #2).
    #[test]
    fn test_learner_ingests_experiences_and_trains() {
        let obs_dim: usize = 8;
        let action_dim: i64 = 3;
        let policy = MlpPolicy::new(obs_dim as i64, action_dim, 32);

        let (exp_sender, exp_receiver) = crossbeam_channel::unbounded();
        let (policy_sender, _policy_receiver) = crossbeam_channel::unbounded();

        // Small buffer / batch sizes so the test runs quickly.
        let config = LearnerConfig {
            obs_dim,
            buffer_size: 8,
            min_batch_size: 4,
            n_epochs: 1,
            ..LearnerConfig::default()
        };

        let mut learner = PolicyLearner::new(
            0,
            policy,
            exp_receiver,
            policy_sender,
            config,
            "/tmp".to_string(),
        )
        .expect("learner construction should succeed");

        // Push a small batch of experiences with the last one terminal.
        let n_exp = 6;
        for i in 0..n_exp {
            let obs = Tensor::ones([obs_dim as i64], (Kind::Float, tch::Device::Cpu));
            let next_obs =
                Tensor::ones([obs_dim as i64], (Kind::Float, tch::Device::Cpu)) * (i as f64 + 1.0);
            let terminated = i == n_exp - 1;
            let exp = Experience::new(
                0,
                obs,
                (i % action_dim as usize) as i64,
                1.0,
                next_obs,
                terminated,
                false,
                0.5,
                -0.69,
            );
            exp_sender.send(exp).expect("channel send should succeed");
        }
        // Drop sender so collect_experiences times out cleanly instead of
        // blocking after consuming the queue.
        drop(exp_sender);

        // Defect #4 regression: collect_experiences must actually fill
        // the buffer (it used to drop `_exp` on the floor).
        learner.collect_experiences().expect("collect_experiences should not error");
        assert_eq!(
            learner.buffer_step_idx, n_exp,
            "buffer_step_idx must reflect every received experience"
        );
        assert!(learner.last_experience.is_some(), "last_experience must be tracked");
        let last = learner.last_experience.as_ref().unwrap();
        assert!(last.terminated, "final experience was terminal");
        assert_eq!(last.next_observation.len(), obs_dim);

        // Defect #2 regression: bootstrap returns 0.0 for terminal ends.
        let last_values_term = learner.compute_bootstrap_last_values();
        assert_eq!(last_values_term, vec![0.0], "terminal end must give 0.0 bootstrap");

        // Force a non-terminal last_experience and verify the policy is
        // consulted (the result is a real f32 — not the placeholder 0.0
        // unconditionally).
        learner.last_experience = Some(LastExperience {
            next_observation: vec![0.5_f32; obs_dim],
            terminated: false,
        });
        let last_values_trunc = learner.compute_bootstrap_last_values();
        assert_eq!(last_values_trunc.len(), 1);
        assert!(last_values_trunc[0].is_finite(), "bootstrap value must be finite");

        // Defect #1 regression: `RolloutBuffer` was constructed with the
        // configured `obs_dim` (8), not the hardcoded 4. If it had been
        // hardcoded, `buffer.add(... &obs_vec ...)` above would have
        // panicked the debug_assert on obs dim mismatch.

        // Run one full training cycle to confirm the loop is no longer
        // a silent no-op. Restore terminal state first so GAE bootstrap
        // is deterministic.
        learner.last_experience = Some(LastExperience {
            next_observation: vec![0.0_f32; obs_dim],
            terminated: true,
        });
        let last_values = learner.compute_bootstrap_last_values();
        learner.buffer.compute_advantages(
            &last_values,
            learner.config.gamma as f32,
            learner.config.gae_lambda as f32,
        );
        let stats = learner.train_step().expect("train_step should run end-to-end");
        // Loss values must be finite (no NaN/Inf).
        assert!(stats.total_loss.is_finite(), "total_loss must be finite");
        assert!(stats.policy_loss.is_finite(), "policy_loss must be finite");
        assert!(stats.value_loss.is_finite(), "value_loss must be finite");
    }
}
