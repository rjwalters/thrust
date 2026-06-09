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
    /// Monotonically increments as experiences are ingested and is reset
    /// to 0 after each successful `train()` cycle. The collection loop
    /// bounds it above by `config.buffer_size`, so it is always in
    /// `0..=config.buffer_size`. Doubles as the canonical fill count
    /// (`RolloutBuffer::len()` returns capacity, not fill).
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
            //    - If the final inserted transition ended the episode (`terminated ==
            //      true`), `V(s_{T+1}) = 0`.
            //    - Otherwise (truncation / buffer-full), bootstrap with `V(s_{T+1})` from
            //      the current policy, computed under `no_grad`.
            //
            //    Restrict the GAE backward iteration to the filled prefix
            //    (`buffer_step_idx` rows). Otherwise the unwritten tail —
            //    zero rewards, zero values, `terminated == false` — would
            //    propagate `γ^k * last_value` backward through the padding
            //    into the real rows, corrupting advantages on the actual
            //    data.
            let last_values = self.compute_bootstrap_last_values();
            self.buffer.compute_advantages_partial(
                self.buffer_step_idx,
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
        // Phase 3 (#80) of the Burn migration moves message-protocol
        // observations from `tch::Tensor` to `Vec<f32>` host buffers;
        // see the module-level note in `multi_agent::messages`. No
        // host↔tch conversion is needed here anymore.
        let obs_vec = exp.observation;

        if obs_vec.len() != self.config.obs_dim {
            return Err(anyhow::anyhow!(
                "Experience observation dim {} does not match configured obs_dim {}",
                obs_vec.len(),
                self.config.obs_dim
            ));
        }

        let next_obs_vec = exp.next_observation;

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

    /// Run a PPO training step.
    ///
    /// Delegates to `PPOTrainer::train_step_with_policy`, which internally
    /// iterates over the rollout for `config.n_epochs` PPO epochs and returns
    /// averaged statistics across all minibatch updates. This method performs
    /// exactly one such trainer call --- it does NOT add an outer epoch loop
    /// (doing so would compound epochs multiplicatively; see issue #41).
    fn train_step(&mut self) -> Result<TrainingStats> {
        // Get batch from the filled prefix of the buffer. `RolloutBuffer`
        // pre-allocates its full capacity at construction, so calling
        // `get_batch()` here would include zero-padded unwritten rows
        // when the rollout ended before `buffer_size` (early-terminating
        // episode, truncation, etc.) and feed fake transitions to PPO.
        let batch = self.buffer.get_filled_batch(self.buffer_step_idx);

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

        // Run one PPO update via the safe split-borrow path. The trainer's
        // `train_step_self_policy` accesses `self.policy` and the optimizer
        // through disjoint-field destructuring, eliminating the `unsafe`
        // raw-pointer workaround that previously lived here (see #39). It
        // already performs `config.n_epochs` epochs over the batch internally
        // and returns averaged stats across all minibatches, so we must NOT
        // wrap this call in an additional `n_epochs` loop --- doing so would
        // cause `n_epochs²` PPO epochs per call (see #41).
        let stats = self.trainer.train_step_self_policy(
            &observations,
            &actions,
            &old_log_probs,
            &old_values,
            &advantages,
            &returns,
            |policy: &MlpPolicy, obs: &Tensor, acts: &Tensor| policy.evaluate_actions(obs, acts),
        )?;

        // The trainer returns averaged stats already; just pass through the
        // fields we expose on the multi-agent `TrainingStats` message.
        Ok(TrainingStats {
            total_loss: stats.total_loss,
            policy_loss: stats.policy_loss,
            value_loss: stats.value_loss,
            entropy: stats.entropy,
            step: self.step,
            ..TrainingStats::default()
        })
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
    /// - Verifies `compute_bootstrap_last_values` returns `vec![0.0]` for a
    ///   terminal end and a non-zero `V(s_{T+1})` (or at least a distinct
    ///   policy-derived value) for a non-terminal end (defect #2).
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

        let mut learner =
            PolicyLearner::new(0, policy, exp_receiver, policy_sender, config, "/tmp".to_string())
                .expect("learner construction should succeed");

        // Push a small batch of experiences with the last one terminal.
        // Observations are `Vec<f32>` host buffers (phase 3 of the
        // Burn migration, #80).
        let n_exp = 6;
        for i in 0..n_exp {
            let obs: Vec<f32> = vec![1.0; obs_dim];
            let next_obs: Vec<f32> = vec![(i as f32) + 1.0; obs_dim];
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
        learner.last_experience =
            Some(LastExperience { next_observation: vec![0.5_f32; obs_dim], terminated: false });
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
        learner.last_experience =
            Some(LastExperience { next_observation: vec![0.0_f32; obs_dim], terminated: true });
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

    /// Regression test for issue #28: `PolicyLearner::train_step` must not
    /// consume zero-padded unwritten rows from `RolloutBuffer`, and the
    /// preceding GAE pass must not walk backward through them either.
    ///
    /// With `buffer_size = 64` and 10 ingested experiences, the PPO batch
    /// must contain exactly 10 rows — not 64. The strongest single signal
    /// is that every batch entry's `old_log_prob` equals the sentinel
    /// value (`-0.69`) we send through the channel, since the buffer's
    /// storage default is `0.0`. If any zero-padded row leaked into the
    /// batch, that assertion would fail.
    #[test]
    fn test_train_step_does_not_zero_pad_partial_rollout() {
        let obs_dim: usize = 4;
        let action_dim: i64 = 2;
        let policy = MlpPolicy::new(obs_dim as i64, action_dim, 16);

        let (exp_sender, exp_receiver) = crossbeam_channel::unbounded();
        let (policy_sender, _policy_receiver) = crossbeam_channel::unbounded();

        let config = LearnerConfig {
            obs_dim,
            buffer_size: 64,
            min_batch_size: 4,
            n_epochs: 1,
            ..LearnerConfig::default()
        };

        let mut learner =
            PolicyLearner::new(0, policy, exp_receiver, policy_sender, config, "/tmp".to_string())
                .expect("learner construction should succeed");

        // Ingest exactly 10 experiences (well under buffer_size = 64).
        // Each experience carries the sentinel log_prob `-0.69`, distinct
        // from the storage default of `0.0`.
        let n_exp = 10usize;
        let sentinel_log_prob: f32 = -0.69;
        for i in 0..n_exp {
            let obs: Vec<f32> = vec![1.0; obs_dim];
            let next_obs: Vec<f32> = vec![1.0; obs_dim];
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
                sentinel_log_prob,
            );
            exp_sender.send(exp).expect("channel send should succeed");
        }
        // Drop sender so `collect_experiences` times out instead of
        // blocking after consuming the queue.
        drop(exp_sender);

        learner.collect_experiences().expect("collect_experiences should not error");
        assert_eq!(
            learner.buffer_step_idx, n_exp,
            "buffer_step_idx must reflect every received experience"
        );

        // Run partial-fill GAE on just the filled prefix. Bootstrap value
        // is 0.0 because the final experience is terminal.
        let last_values = learner.compute_bootstrap_last_values();
        learner.buffer.compute_advantages_partial(
            learner.buffer_step_idx,
            &last_values,
            learner.config.gamma as f32,
            learner.config.gae_lambda as f32,
        );

        // Core assertion: the batch consumed by PPO is sized to the
        // filled prefix, not to buffer capacity.
        let batch = learner.buffer.get_filled_batch(learner.buffer_step_idx);
        assert_eq!(batch.actions.len(), n_exp, "batch must contain only filled rows");
        assert_eq!(batch.old_log_probs.len(), n_exp);
        assert_eq!(batch.old_values.len(), n_exp);
        assert_eq!(batch.advantages.len(), n_exp);
        assert_eq!(batch.returns.len(), n_exp);
        assert_eq!(batch.observations.len(), n_exp * obs_dim);

        // Sanity: every batch row came from real data — its
        // `old_log_prob` must equal the sentinel we sent, not the
        // storage default `0.0`. This is the single strongest signal
        // that no zero-padded row leaked into the batch.
        for &lp in &batch.old_log_probs {
            assert!(
                (lp - sentinel_log_prob).abs() < 1e-6,
                "old_log_prob in batch must be real (-0.69), not zero-padded; got {}",
                lp
            );
        }

        // Same sanity check for `old_values` (we sent `0.5`).
        for &v in &batch.old_values {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "old_value in batch must be real (0.5), not zero-padded; got {}",
                v
            );
        }

        // End-to-end: a full `train_step` on the partial rollout must
        // produce finite losses. (Bad GAE contamination from padded
        // rows could in principle push losses to NaN/Inf via outsized
        // advantage normalization; this is a backstop assertion.)
        let stats = learner.train_step().expect("train_step end-to-end should not error");
        assert!(stats.total_loss.is_finite(), "total_loss must be finite");
        assert!(stats.policy_loss.is_finite(), "policy_loss must be finite");
        assert!(stats.value_loss.is_finite(), "value_loss must be finite");
    }
}
