//! Recurrent PPO trainer (Burn backend).
//!
//! Phase 3 of the recurrent-policy epic (#262). [`RecurrentPPOTrainer`] is the
//! rank-3 sibling of [`crate::train::ppo::trainer::PPOTrainerBurn`]: it drives
//! the same clipped-surrogate / value-clip / entropy-bonus / KL-early-stop
//! recipe, reusing the **unchanged** loss functions in
//! [`crate::train::ppo::loss`], but consumes whole-trajectory sequence batches
//! from a [`RecurrentRolloutBuffer`] instead of flattened rank-2 transitions.
//!
//! # Why a separate trainer
//!
//! The feedforward trainer's `evaluate_fn` closure is hard-typed to rank-2
//! observations (`Tensor<B, 2>`) and rank-1 actions — it flattens time and env
//! together, structurally erasing the temporal order a recurrent policy needs.
//! Recurrence requires a rank-3 `[N_env, T, obs_dim]` forward that runs the
//! LSTM step-by-step and resets `(h, c)` at episode boundaries. Rather than
//! overload the feedforward closure, this trainer takes a rank-3 analogue:
//!
//! ```text
//! evaluate_fn(&policy, obs_seq [N_env,T,obs_dim], actions [N_env,T], episode_starts [N_env,T])
//!     -> (log_probs [N_env,T], entropy [N_env,T], values [N_env,T])
//! ```
//!
//! # Rank-2 → rank-1 shape adapter
//!
//! [`crate::train::ppo::loss`]'s `compute_policy_loss` / `compute_value_loss` /
//! `compute_entropy_loss` all operate on rank-1 tensors. `evaluate_sequences`
//! returns rank-2 `[N_env, T]`. The trainer therefore flattens
//! (`reshape([N_env * T])`) **inside** the minibatch loop, *after* the forward
//! — the LSTM forward needs the `[N_env, T]` shape intact to run its
//! step-by-step loop with per-step state resets. Advantage normalization runs
//! on the flattened rank-1 view, exactly as in `PPOTrainerBurn`.
//!
//! # Episode-boundary contract
//!
//! `episode_starts` is consumed **directly** from the buffer's materialized
//! batch (`terminated[t-1] || truncated[t-1]` shifted one step, with the
//! cross-iteration carry at `t == 0`). GAE stays terminated-only, computed
//! upstream by the buffer. The two masks are intentionally distinct; this
//! trainer does not re-derive or touch either (see
//! `docs/RECURRENT_POLICY_DESIGN.md`, Q2).
//!
//! # Ownership model
//!
//! Identical to `PPOTrainerBurn`: the policy is held in `Option<P>` and swapped
//! through Burn's move-through optimizer on every gradient step.

use anyhow::{Result, anyhow};
use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend},
};

use super::{
    config::PPOConfig,
    loss::{compute_entropy_loss, compute_policy_loss, compute_value_loss, scalar_f64},
    stats::TrainingStats,
};
use crate::{
    buffer::rollout::RecurrentRolloutBuffer,
    train::optimizer::{BackendOptimizer, BurnOptimizer},
};

/// Recurrent PPO trainer (Burn backend).
///
/// Generic over the same three parameters as
/// [`crate::train::ppo::trainer::PPOTrainerBurn`]:
/// - `B: AutodiffBackend` — the Burn backend.
/// - `P: AutodiffModule<B>` — the recurrent policy module (e.g.
///   [`crate::policy::lstm::LstmBurnPolicy`]).
/// - `O: Optimizer<P, B>` — the Burn optimizer.
///
/// The policy is held in `Option<P>` because Burn's `Optimizer::step` consumes
/// the module by value; we `.take()` / put-back across each gradient step.
pub struct RecurrentPPOTrainer<B, P, O>
where
    B: AutodiffBackend,
    P: AutodiffModule<B>,
    O: Optimizer<P, B>,
{
    config: PPOConfig,
    policy: Option<P>,
    optimizer: BurnOptimizer<B, P, O>,
    device: B::Device,
    /// Optional learning-rate override applied on the next `train_step`,
    /// `None` ⇒ use the optimizer's configured rate. Set by
    /// [`Self::set_learning_rate`] to support caller-driven LR schedules
    /// (e.g. linear annealing), which stabilize the end of long PPO runs.
    lr_override: Option<f64>,
    total_steps: usize,
    total_episodes: usize,
    low_entropy_count: usize,
}

impl<B, P, O> RecurrentPPOTrainer<B, P, O>
where
    B: AutodiffBackend,
    P: AutodiffModule<B> + Clone,
    O: Optimizer<P, B>,
{
    /// Build a new recurrent PPO trainer.
    ///
    /// `device` is the Burn device the sequence-batch tensors are
    /// materialized on (the buffer stores host `Vec` data, so — unlike the
    /// feedforward trainer, which reads the device off the observation tensor
    /// passed to `train_step` — the recurrent trainer must be told which
    /// device to build minibatches on).
    pub fn new(
        config: PPOConfig,
        policy: P,
        optimizer: BurnOptimizer<B, P, O>,
        device: B::Device,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            policy: Some(policy),
            optimizer,
            device,
            lr_override: None,
            total_steps: 0,
            total_episodes: 0,
            low_entropy_count: 0,
        })
    }

    /// Override the learning rate used by subsequent `train_step` calls.
    ///
    /// Enables caller-driven schedules (linear annealing, warmup, …). Pass the
    /// per-update rate before each `train_step`. Without a call, the
    /// optimizer's configured rate is used.
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.lr_override = Some(lr);
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &PPOConfig {
        &self.config
    }

    /// Borrow the policy. Panics if called mid-step (the policy has been moved
    /// into the optimizer); only safe between `train_step` invocations.
    pub fn policy(&self) -> &P {
        self.policy.as_ref().expect("policy is None mid-step")
    }

    /// Total completed gradient updates.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Total completed episodes (caller increments).
    pub fn total_episodes(&self) -> usize {
        self.total_episodes
    }

    /// Increment the episode counter.
    pub fn increment_episodes(&mut self, n: usize) {
        self.total_episodes += n;
    }

    /// Train for one recurrent PPO update.
    ///
    /// Iterates `n_epochs` passes over the buffer's env-major minibatches
    /// (whole trajectories, `envs_per_minibatch` at a time), running the
    /// rank-3 `evaluate_fn` forward, flattening to rank-1, and applying the
    /// shared PPO surrogate / value / entropy losses. Mirrors
    /// [`crate::train::ppo::trainer::PPOTrainerBurn::train_step`] step-for-step
    /// except for the sequence-batch assembly and the rank-2 → rank-1 flatten.
    ///
    /// * `buffer` — the recurrent rollout buffer; advantages/returns must
    ///   already be computed (via `buffer.compute_advantages`).
    /// * `envs_per_minibatch` — whole env-trajectories per minibatch (the
    ///   recurrent analogue of the feedforward `batch_size`).
    /// * `evaluate_fn` — receives `(&policy, obs_seq, actions, episode_starts)`
    ///   and returns `(log_probs, entropy, values)`, each `[N_env, T]`.
    pub fn train_step<F>(
        &mut self,
        buffer: &RecurrentRolloutBuffer,
        envs_per_minibatch: usize,
        mut evaluate_fn: F,
    ) -> Result<TrainingStats>
    where
        F: FnMut(
            &P,
            Tensor<B, 3>,
            Tensor<B, 2, Int>,
            Tensor<B, 2>,
        ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
    {
        let device = self.device.clone();
        let mut stats_sum = TrainingStats::zeros();
        let mut num_updates = 0;

        for _epoch in 0..self.config.n_epochs {
            for batch in buffer.to_minibatches::<B>(envs_per_minibatch, true, &device) {
                // Take the policy out so we can move it through `step`.
                let policy = self
                    .policy
                    .take()
                    .ok_or_else(|| anyhow!("policy is None; concurrent train_step calls?"))?;

                // Rank-3 forward: the LSTM needs `[N_env, T]` intact to run its
                // per-step loop with episode-boundary resets.
                let (log_probs, entropy, values) = evaluate_fn(
                    &policy,
                    batch.obs_seq.clone(),
                    batch.actions.clone(),
                    batch.episode_starts.clone(),
                );

                // Flatten rank-2 `[N_env, T]` → rank-1 `[N_env * T]` for the
                // loss functions (which operate on rank-1 tensors).
                let [n_env, t] = log_probs.dims();
                let flat = n_env * t;
                let log_probs = log_probs.reshape([flat]);
                let entropy = entropy.reshape([flat]);
                let values = values.reshape([flat]);
                let old_log_probs = batch.old_log_probs.clone().reshape([flat]);
                let old_values = batch.old_values.clone().reshape([flat]);
                let advantages = batch.advantages.clone().reshape([flat]);
                let returns = batch.returns.clone().reshape([flat]);

                // Advantage normalization on the flattened 1D view (matches
                // `PPOTrainerBurn`). Advantages come from the buffer as data —
                // no autograd tape — so a host round-trip is safe.
                let adv_data: Vec<f32> = advantages.into_data().to_vec().unwrap_or_default();
                let adv_mean = host_mean(&adv_data);
                let adv_std = host_std_biased(&adv_data, adv_mean);
                let adv_norm: Vec<f32> = adv_data
                    .iter()
                    .map(|&a| (a - adv_mean as f32) / (adv_std as f32 + 1e-8))
                    .collect();
                let advantages =
                    Tensor::<B, 1>::from_data(TensorData::new(adv_norm, [flat]), &device);

                let (policy_loss, clip_fraction, approx_kl) = compute_policy_loss(
                    log_probs,
                    old_log_probs,
                    advantages,
                    self.config.clip_range,
                );

                let (value_loss, explained_var) =
                    compute_value_loss(values, old_values, returns, self.config.clip_range_vf);

                let entropy_loss = compute_entropy_loss(entropy.clone());

                // Scalars for stat collection.
                let policy_loss_val = scalar_f64(policy_loss.clone());
                let value_loss_val = scalar_f64(value_loss.clone());
                let entropy_val = scalar_f64(entropy.mean());

                // total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                let total_loss = policy_loss
                    + value_loss.mul_scalar(self.config.vf_coef as f32)
                    + entropy_loss.mul_scalar(self.config.ent_coef as f32);
                let total_loss_val = scalar_f64(total_loss.clone());

                // Burn gradient flow: backward → GradientsParams → step.
                let grads = total_loss.backward();
                let grads = GradientsParams::from_grads(grads, &policy);
                let lr = self.lr_override.unwrap_or_else(|| self.optimizer.learning_rate());
                let policy = self.optimizer.inner_mut().step(lr, policy, grads);
                self.policy = Some(policy);

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

                if approx_kl > self.config.target_kl {
                    break;
                }
            }
        }

        self.total_steps += num_updates;
        let avg_stats = stats_sum.average();

        // Entropy-collapse guard (matches the feedforward trainer).
        const ENTROPY_THRESHOLD: f64 = 0.05;
        const MAX_LOW_ENTROPY_COUNT: usize = 3;
        if avg_stats.entropy < ENTROPY_THRESHOLD {
            self.low_entropy_count += 1;
            if self.low_entropy_count >= MAX_LOW_ENTROPY_COUNT {
                return Err(anyhow!(
                    "Training stopped due to entropy collapse (entropy < {} for {} updates)",
                    ENTROPY_THRESHOLD,
                    MAX_LOW_ENTROPY_COUNT
                ));
            }
        } else {
            self.low_entropy_count = 0;
        }

        Ok(avg_stats)
    }
}

/// Host-side arithmetic mean.
fn host_mean(xs: &[f32]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().map(|&x| x as f64).sum::<f64>() / xs.len() as f64
}

/// Biased standard deviation (denominator `n`).
fn host_std_biased(xs: &[f32], mean: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let sq_dev = xs.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>();
    (sq_dev / n).sqrt()
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, NdArray},
        optim::AdamConfig,
    };

    use super::*;
    use crate::{policy::lstm::LstmBurnPolicy, train::optimizer::BurnOptimizer};

    type B = Autodiff<NdArray<f32>>;

    /// Build a small synthetic recurrent buffer with deterministic data and a
    /// computed GAE, ready to feed `train_step`.
    fn synthetic_buffer(
        num_steps: usize,
        num_envs: usize,
        obs_dim: usize,
    ) -> RecurrentRolloutBuffer {
        let hidden_dim = 8;
        let mut buf = RecurrentRolloutBuffer::new(num_steps, num_envs, obs_dim, hidden_dim);
        for step in 0..num_steps {
            for env in 0..num_envs {
                let obs: Vec<f32> = (0..obs_dim).map(|d| 0.1 * (step + env + d) as f32).collect();
                // Terminate one env midway to exercise the episode-start mask.
                let term = env == 0 && step == num_steps / 2;
                buf.add(step, env, &obs, (step % 2) as i64, 1.0, 0.0, -0.7, term, false);
            }
        }
        let last_values = vec![0.0_f32; num_envs];
        buf.compute_advantages(&last_values, 0.99, 0.95);
        buf
    }

    /// The recurrent trainer constructs and reports zeroed counters.
    #[test]
    fn recurrent_trainer_constructs() {
        let device = Default::default();
        let policy = LstmBurnPolicy::<B>::new(2, 2, 8, &device);
        let inner_opt = AdamConfig::new().init();
        let burn_opt: BurnOptimizer<B, LstmBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 3e-4);
        let trainer =
            RecurrentPPOTrainer::new(PPOConfig::default(), policy, burn_opt, device).unwrap();
        assert_eq!(trainer.total_steps(), 0);
    }

    /// End-to-end: a single `train_step` over a synthetic rank-3 batch runs
    /// without panic, moves the policy through the optimizer, and produces
    /// finite stats.
    #[test]
    fn recurrent_trainer_train_step_runs() {
        let device = Default::default();
        let (num_steps, num_envs, obs_dim, action_dim) = (4, 4, 2, 2);
        let policy = LstmBurnPolicy::<B>::new(obs_dim, action_dim, 8, &device);
        let inner_opt = AdamConfig::new().init();
        let burn_opt: BurnOptimizer<B, LstmBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 1e-3);
        let config = PPOConfig::default().n_epochs(2).target_kl(1.0);
        let mut trainer = RecurrentPPOTrainer::new(config, policy, burn_opt, device).unwrap();

        let buffer = synthetic_buffer(num_steps, num_envs, obs_dim);

        let stats = trainer
            .train_step(&buffer, 2, |p, obs_seq, actions, episode_starts| {
                p.evaluate_sequences(obs_seq, actions, None, episode_starts)
            })
            .unwrap();

        assert!(trainer.total_steps() > 0, "at least one gradient step taken");
        assert!(stats.policy_loss.is_finite());
        assert!(stats.value_loss.is_finite());
        assert!(stats.entropy.is_finite());
    }
}
