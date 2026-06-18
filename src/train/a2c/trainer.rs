//! Synchronous Advantage Actor-Critic (A2C) trainer (Burn backend).
//!
//! Sibling to [`crate::train::ppo::trainer::PPOTrainerBurn`]. A2C reuses
//! the same policy ([`MlpBurnPolicy`]), optimizer ([`BurnOptimizer`]) and
//! GAE/advantage path, and shares PPO's `Option<P>` move-through
//! ownership model (Burn's `Optimizer::step` consumes the module by
//! value). It diverges from PPO in exactly two places:
//!
//! 1. **Loss** — un-clipped policy-gradient loss + plain-MSE value loss (see
//!    [`crate::train::a2c::loss`]); no importance ratio, no value clipping.
//! 2. **Loop** — exactly **one** gradient step per rollout: no epoch loop, no
//!    minibatch shuffle, no KL early-stop. Consequently
//!    [`A2cTrainer::train_step`] drops the `old_log_probs` / `old_values`
//!    arguments PPO needs for its importance ratio and value clipping.
//!
//! [`MlpBurnPolicy`]: crate::policy::mlp::MlpBurnPolicy

use anyhow::{Result, anyhow};
use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    tensor::{Int, Tensor, backend::AutodiffBackend},
};
use rand::{SeedableRng, rngs::StdRng};

use super::{
    config::A2cConfig,
    loss::{compute_a2c_policy_loss, compute_a2c_value_loss, compute_entropy_loss, scalar_f64},
};
use crate::train::optimizer::{BackendOptimizer, BurnOptimizer};

/// Per-update A2C training statistics.
///
/// Purpose-built (rather than reusing PPO's
/// [`TrainingStats`](crate::train::ppo::TrainingStats)) because A2C has
/// no clip-fraction / KL / explained-variance diagnostics — the loop
/// performs a single un-clipped update. All fields are finite host-side
/// `f64`s pulled off the autograd tape after the gradient step.
#[derive(Debug, Clone, Copy, Default)]
pub struct A2cStats {
    /// Policy-gradient loss `-mean(log_prob * advantage)`.
    pub policy_loss: f64,
    /// Plain-MSE value loss `mean((V(s) - returns)^2)`.
    pub value_loss: f64,
    /// Mean per-row policy entropy (the bonus, before negation/scaling).
    pub entropy: f64,
    /// Total optimized loss
    /// `policy_loss + value_coef * value_loss + entropy_coef * entropy_loss`.
    pub total_loss: f64,
}

/// Burn-backend A2C trainer.
///
/// Generic over:
/// - `B: AutodiffBackend` — the Burn backend (e.g. `Autodiff<NdArray<f32>>`).
/// - `P: AutodiffModule<B>` — the shared actor-critic policy module.
/// - `O: Optimizer<P, B>` — the Burn optimizer (typically
///   `AdamConfig::new().init()`).
///
/// The policy is held in `Option<P>` because Burn's `Optimizer::step`
/// consumes the module by value; we `.take()` it and put back the updated
/// copy across the single gradient step.
pub struct A2cTrainer<B, P, O>
where
    B: AutodiffBackend,
    P: AutodiffModule<B>,
    O: Optimizer<P, B>,
{
    config: A2cConfig,
    policy: Option<P>,
    optimizer: BurnOptimizer<B, P, O>,
    total_steps: usize,
    total_episodes: usize,
    low_entropy_count: usize,
    /// Seedable RNG owned by the trainer so any stochastic step (e.g.
    /// future seeded advantage handling) is reproducible under
    /// [`A2cConfig::seed`]. Kept for parity with
    /// [`PPOTrainerBurn`](crate::train::ppo::trainer::PPOTrainerBurn),
    /// whose minibatch shuffle draws from this RNG.
    #[allow(dead_code)]
    rng: StdRng,
}

impl<B, P, O> A2cTrainer<B, P, O>
where
    B: AutodiffBackend,
    P: AutodiffModule<B> + Clone,
    O: Optimizer<P, B>,
{
    /// Build a new Burn A2C trainer.
    ///
    /// Validates the config and stages the global gradient-norm clip
    /// ([`A2cConfig::max_grad_norm`]) on the optimizer wrapper.
    pub fn new(
        config: A2cConfig,
        policy: P,
        mut optimizer: BurnOptimizer<B, P, O>,
    ) -> Result<Self> {
        config.validate()?;
        let rng = StdRng::seed_from_u64(config.seed);
        // Record the global-norm cap on the wrapper. The actual clip is
        // honored by the Burn optimizer built with
        // `AdamConfig::with_grad_clipping`; staging it here keeps the
        // trainer's view of the cap consistent with PPO.
        optimizer.clip_grad_norm(config.max_grad_norm);
        Ok(Self {
            config,
            policy: Some(policy),
            optimizer,
            total_steps: 0,
            total_episodes: 0,
            low_entropy_count: 0,
            rng,
        })
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &A2cConfig {
        &self.config
    }

    /// Borrow the policy. Panics if the trainer is mid-step (the policy
    /// has been moved into the optimizer); only safe to call between
    /// `train_step` invocations.
    pub fn policy(&self) -> &P {
        self.policy.as_ref().expect("policy is None mid-step")
    }

    /// Total completed gradient updates (one per `train_step`).
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

    /// Train for one A2C update.
    ///
    /// Performs exactly **one** gradient step over the whole rollout:
    /// 1. Optionally normalize advantages to zero-mean/unit-variance when
    ///    [`A2cConfig::normalize_advantages`] is set.
    /// 2. Evaluate the policy to get `(log_probs, entropy, values)`.
    /// 3. `policy_loss = -mean(log_prob * advantage)` (no ratio, no clip).
    /// 4. `value_loss = mean((V(s) - returns)^2)` (plain MSE, no clip).
    /// 5. `entropy_loss = -mean(entropy)`.
    /// 6. `total = policy_loss + value_coef * value_loss
    ///    + entropy_coef * entropy_loss`.
    /// 7. Backprop, build `GradientsParams`, step the optimizer once.
    /// 8. Entropy-collapse guard (shared with PPO).
    ///
    /// Note the **absence** of `old_log_probs` / `old_values`: A2C is
    /// on-policy with a single update, so there is no behaviour policy to
    /// form an importance ratio against, and no old value baseline to
    /// clip against.
    ///
    /// The `evaluate_fn` closure receives `(&policy, obs, actions)` and
    /// must return `(log_probs, entropy, values)` — exactly the shape of
    /// [`MlpBurnPolicy::evaluate_actions`](crate::policy::mlp::MlpBurnPolicy::evaluate_actions).
    pub fn train_step<F>(
        &mut self,
        observations: Tensor<B, 2>,
        actions: Tensor<B, 1, Int>,
        advantages: Tensor<B, 1>,
        returns: Tensor<B, 1>,
        mut evaluate_fn: F,
    ) -> Result<A2cStats>
    where
        F: FnMut(&P, Tensor<B, 2>, Tensor<B, 1, Int>) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>),
    {
        let device = observations.device();

        // Advantage normalization (matches the PPO trainer's host-side
        // biased normalization). Done on the host so the advantages stay
        // detached constants w.r.t. the policy parameters.
        let advantages = if self.config.normalize_advantages {
            let adv_mean = scalar_f64(advantages.clone().mean());
            let adv_data: Vec<f32> = advantages.into_data().to_vec().unwrap_or_default();
            let adv_std = host_std_biased(&adv_data, adv_mean) as f32;
            let normalized: Vec<f32> =
                adv_data.iter().map(|&a| (a - adv_mean as f32) / (adv_std + 1e-8)).collect();
            let n = normalized.len();
            Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(normalized, [n]), &device)
        } else {
            advantages
        };

        // Take the policy out so we can move it through `step`.
        let policy = self
            .policy
            .take()
            .ok_or_else(|| anyhow!("policy is None; concurrent train_step calls?"))?;

        let (log_probs, entropy, values) = evaluate_fn(&policy, observations, actions);

        let policy_loss = compute_a2c_policy_loss(log_probs, advantages);
        let value_loss = compute_a2c_value_loss(values, returns);
        let entropy_loss = compute_entropy_loss(entropy.clone());

        // Host-side scalars for stat collection.
        let policy_loss_val = scalar_f64(policy_loss.clone());
        let value_loss_val = scalar_f64(value_loss.clone());
        let entropy_val = scalar_f64(entropy.mean());

        // total = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        let total_loss = policy_loss
            + value_loss.mul_scalar(self.config.value_coef as f32)
            + entropy_loss.mul_scalar(self.config.entropy_coef as f32);
        let total_loss_val = scalar_f64(total_loss.clone());

        // Burn gradient flow: backward → GradientsParams → single step.
        let grads = total_loss.backward();
        let grads = GradientsParams::from_grads(grads, &policy);
        let lr = self.optimizer.learning_rate();
        let policy = self.optimizer.inner_mut().step(lr, policy, grads);
        self.policy = Some(policy);

        self.total_steps += 1;

        let stats = A2cStats {
            policy_loss: policy_loss_val,
            value_loss: value_loss_val,
            entropy: entropy_val,
            total_loss: total_loss_val,
        };

        // Entropy-collapse guard (matches the PPO trainer).
        const ENTROPY_THRESHOLD: f64 = 0.05;
        const MAX_LOW_ENTROPY_COUNT: usize = 3;
        if stats.entropy < ENTROPY_THRESHOLD {
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

        Ok(stats)
    }
}

/// Biased standard deviation (denominator `n`). Mirrors the PPO
/// trainer's host-side advantage-normalization helper.
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
        tensor::TensorData,
    };

    use super::*;
    use crate::{policy::mlp::MlpBurnPolicy, train::optimizer::BurnOptimizer};

    type B = Autodiff<NdArray<f32>>;

    /// Smoke test: an A2C trainer constructs and reports zero steps.
    #[test]
    fn a2c_trainer_constructs() {
        let device = Default::default();
        let policy = MlpBurnPolicy::<B>::new(4, 2, 32, &device);
        let inner_opt = AdamConfig::new().init();
        let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 7e-4);
        let trainer = A2cTrainer::new(A2cConfig::default(), policy, burn_opt).unwrap();
        assert_eq!(trainer.total_steps(), 0);
        assert_eq!(trainer.total_episodes(), 0);
    }

    /// End-to-end: a single `train_step` against a synthetic batch
    /// completes, performs exactly one update, and yields finite stats.
    #[test]
    fn a2c_train_step_runs() {
        let device = Default::default();
        let policy = MlpBurnPolicy::<B>::new(4, 2, 16, &device);
        let inner_opt = AdamConfig::new().init();
        let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 1e-3);
        let mut trainer = A2cTrainer::new(A2cConfig::default(), policy, burn_opt).unwrap();

        let batch = 8;
        let obs_dim = 4;
        let obs_data: Vec<f32> = (0..batch * obs_dim).map(|i| (i as f32) * 0.01).collect();
        let observations =
            Tensor::<B, 2>::from_data(TensorData::new(obs_data, [batch, obs_dim]), &device);
        let actions = Tensor::<B, 1, Int>::from_data(
            TensorData::new(vec![0i64, 1, 0, 1, 0, 1, 0, 1], [batch]),
            &device,
        );
        let advantages = Tensor::<B, 1>::from_data(
            TensorData::new(vec![1.0f32, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5], [batch]),
            &device,
        );
        let returns =
            Tensor::<B, 1>::from_data(TensorData::new(vec![1.0f32; batch], [batch]), &device);

        let stats = trainer
            .train_step(observations, actions, advantages, returns, |p, o, a| {
                p.evaluate_actions(o, a)
            })
            .unwrap();

        // Exactly one gradient step per rollout.
        assert_eq!(trainer.total_steps(), 1);
        assert!(stats.policy_loss.is_finite());
        assert!(stats.value_loss.is_finite());
        assert!(stats.entropy.is_finite());
        assert!(stats.total_loss.is_finite());
    }
}
