//! Burn-backend PPO trainer (phase 3 of the Burn migration, #80).
//!
//! Sibling to [`crate::train::ppo::PPOTrainerBurn`] (tch path). Both
//! trainers implement the same clipped-surrogate, value-clip,
//! entropy-bonus, KL-early-stop recipe; the only difference is the
//! tensor backend and the optimizer ownership model.
//!
//! # Ownership model
//!
//! Burn's `Optimizer<M, B>` is move-through: every gradient step
//! consumes the module by value and returns the updated copy. Phase
//! 1's scout (#78) confirmed this is the **single biggest** structural
//! divergence between the two backends (Burn-migration friction point #1).
//!
//! The Burn trainer therefore *owns* the policy module via an
//! `Option<P>` field and swaps it through the optimizer on every
//! step:
//!
//! ```text
//! let module = self.policy.take().unwrap();
//! let grads = loss.backward();
//! let grads = GradientsParams::from_grads(grads, &module);
//! let module = self.optimizer.inner_mut().step(lr, module, grads);
//! self.policy = Some(module);
//! ```
//!
//! The tch trainer is `struct PPOTrainer<P>` with `policy: P`; the
//! Burn trainer is `struct PPOTrainerBurn<B, P, O>` with the policy
//! held in `Option<P>`. Phase 5 (#82) collapses the two when the
//! ownership-model asymmetry goes away (only Burn remains).
//!
//! # Evaluating the policy
//!
//! The trainer takes a closure `evaluate_fn(&P, observations, actions)`
//! that returns `(log_probs, entropy, values)` exactly as the tch
//! trainer does (see `PPOTrainer::train_step_with_policy`). This keeps
//! the loss math identical and lets the caller plug in any module
//! whose forward pass yields the right tensor shapes — including, for
//! phase 4, the proper `MlpBurnPolicy`/`SnakeCnnBurn` ports.

use anyhow::{Result, anyhow};
use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    tensor::{Int, Tensor, backend::AutodiffBackend},
};
use rand::{SeedableRng, rngs::StdRng};

use super::{
    config::PPOConfig,
    loss::{
        compute_entropy_loss, compute_policy_loss, compute_value_loss,
        generate_minibatch_indices_with_rng, scalar_f64,
    },
    stats::TrainingStats,
};
use crate::train::{
    grad_clip::clip_grads_by_global_norm,
    optimizer::{BackendOptimizer, BurnOptimizer},
};

/// Burn-backend PPO trainer.
///
/// Generic over:
/// - `B: AutodiffBackend` — the Burn backend (e.g. `Autodiff<NdArray<f32>>`,
///   `Autodiff<Wgpu>`, etc.).
/// - `P: AutodiffModule<B>` — the policy module type.
/// - `O: Optimizer<P, B>` — the Burn optimizer (typically built from
///   `AdamConfig::new().init()`).
///
/// The policy is held in `Option<P>` because Burn's `Optimizer::step`
/// consumes the module by value. We use `.take()` / put-back across
/// each gradient step.
pub struct PPOTrainerBurn<B, P, O>
where
    B: AutodiffBackend,
    P: AutodiffModule<B>,
    O: Optimizer<P, B>,
{
    config: PPOConfig,
    policy: Option<P>,
    optimizer: BurnOptimizer<B, P, O>,
    total_steps: usize,
    total_episodes: usize,
    low_entropy_count: usize,
    /// Seedable RNG for the per-epoch minibatch shuffle. Owned by the
    /// trainer so the shuffle order is reproducible under
    /// `config.seed` (issue #109). Previously the shuffle drew from
    /// the thread-local `rand::rng()`, which defeated any
    /// upstream seed plumbing (e.g. `PsroConfig::seed`).
    rng: StdRng,
}

impl<B, P, O> PPOTrainerBurn<B, P, O>
where
    B: AutodiffBackend,
    P: AutodiffModule<B> + Clone,
    O: Optimizer<P, B>,
{
    /// Build a new Burn PPO trainer.
    ///
    /// Validates the config and stages the global gradient-norm cap
    /// ([`PPOConfig::max_grad_norm`]) on the optimizer wrapper; `train_step`
    /// applies it to the gradients of every minibatch step (issue #299).
    pub fn new(
        config: PPOConfig,
        policy: P,
        mut optimizer: BurnOptimizer<B, P, O>,
    ) -> Result<Self> {
        config.validate()?;
        let rng = StdRng::seed_from_u64(config.seed);
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
    pub fn config(&self) -> &PPOConfig {
        &self.config
    }

    /// Borrow the policy. Panics if the trainer is mid-step (the policy
    /// has been moved into the optimizer); only safe to call between
    /// `train_step` invocations.
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

    /// Train for one PPO update.
    ///
    /// Implements the same algorithm as
    /// `PPOTrainer::train_step_with_policy` on the tch path:
    /// 1. Normalize advantages to zero mean / unit variance.
    /// 2. For each of `n_epochs` epochs, shuffle the buffer and iterate
    ///    minibatches.
    /// 3. Compute surrogate / value / entropy losses.
    /// 4. `total_loss = policy + vf_coef * value + ent_coef * entropy`.
    /// 5. Backprop, build `GradientsParams`, step the optimizer.
    /// 6. KL early stop if `approx_kl > target_kl`.
    /// 7. Entropy-collapse guard.
    ///
    /// The `evaluate_fn` closure receives `(&policy, obs, actions)` and
    /// must return `(log_probs, entropy, values)` — exactly the same
    /// shape as the policy network's `evaluate_actions` method.
    #[allow(clippy::too_many_arguments)]
    pub fn train_step<F>(
        &mut self,
        observations: Tensor<B, 2>,
        actions: Tensor<B, 1, Int>,
        old_log_probs: Tensor<B, 1>,
        old_values: Tensor<B, 1>,
        advantages: Tensor<B, 1>,
        returns: Tensor<B, 1>,
        mut evaluate_fn: F,
    ) -> Result<TrainingStats>
    where
        F: FnMut(&P, Tensor<B, 2>, Tensor<B, 1, Int>) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>),
    {
        let device = observations.device();
        let batch_size = observations.dims()[0];
        let mut stats_sum = TrainingStats::zeros();
        let mut num_updates = 0;

        // Advantage normalization (matches the tch trainer).
        let adv_mean_scalar = scalar_f64(advantages.clone().mean()) as f32;
        let adv_data: Vec<f32> = advantages.into_data().to_vec().unwrap_or_default();
        let adv_std = host_std_biased(&adv_data, adv_mean_scalar as f64) as f32;
        let advantages_normalized_host: Vec<f32> =
            adv_data.iter().map(|&a| (a - adv_mean_scalar) / (adv_std + 1e-8)).collect();

        for _epoch in 0..self.config.n_epochs {
            // Seedable RNG → reproducible minibatch shuffle order
            // under `config.seed` (issue #109).
            let batch_indices = generate_minibatch_indices_with_rng(
                batch_size,
                self.config.batch_size,
                &mut self.rng,
            );

            for indices in &batch_indices {
                let mb_obs = select_rows_2d(observations.clone(), indices, &device);
                let mb_actions = select_rows_int(actions.clone(), indices, &device);
                let mb_old_log_probs = select_rows_1d(old_log_probs.clone(), indices, &device);
                let mb_old_values = select_rows_1d(old_values.clone(), indices, &device);
                let mb_returns = select_rows_1d(returns.clone(), indices, &device);
                let mb_adv: Vec<f32> =
                    indices.iter().map(|&i| advantages_normalized_host[i]).collect();
                let mb_advantages = Tensor::<B, 1>::from_data(
                    burn::tensor::TensorData::new(mb_adv, [indices.len()]),
                    &device,
                );

                // Take the policy out so we can move it through `step`.
                let policy = self
                    .policy
                    .take()
                    .ok_or_else(|| anyhow!("policy is None; concurrent train_step calls?"))?;

                let (log_probs, entropy, values) =
                    evaluate_fn(&policy, mb_obs.clone(), mb_actions.clone());

                let (policy_loss, clip_fraction, approx_kl) = compute_policy_loss(
                    log_probs,
                    mb_old_log_probs,
                    mb_advantages,
                    self.config.clip_range,
                );

                let (value_loss, explained_var) = compute_value_loss(
                    values,
                    mb_old_values,
                    mb_returns,
                    self.config.clip_range_vf,
                );

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
                // Global gradient-norm clip (issue #299):
                // `PPOConfig::max_grad_norm` is staged on the wrapper in
                // `new` and applied to the gradient slice before the
                // move-through step, mirroring the joint trainer (#239).
                let grads = match self.optimizer.grad_clip_norm() {
                    Some(max_norm) if max_norm > 0.0 => {
                        clip_grads_by_global_norm::<B, P>(&policy, grads, max_norm as f32)
                    }
                    _ => grads,
                };
                let lr = self.optimizer.learning_rate();
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

        // Entropy-collapse guard (matches the tch trainer).
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

/// Biased standard deviation (denominator `n`).
fn host_std_biased(xs: &[f32], mean: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let sq_dev = xs.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>();
    (sq_dev / n).sqrt()
}

/// Select `indices` rows from a rank-2 tensor.
fn select_rows_2d<B: AutodiffBackend>(
    tensor: Tensor<B, 2>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let cols = tensor.dims()[1];
    let host: Vec<f32> = tensor.into_data().to_vec().unwrap_or_default();
    let mut out = Vec::with_capacity(indices.len() * cols);
    for &i in indices {
        let start = i * cols;
        out.extend_from_slice(&host[start..start + cols]);
    }
    Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(out, [indices.len(), cols]), device)
}

/// Select `indices` rows from a rank-1 float tensor.
fn select_rows_1d<B: AutodiffBackend>(
    tensor: Tensor<B, 1>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 1> {
    let host: Vec<f32> = tensor.into_data().to_vec().unwrap_or_default();
    let out: Vec<f32> = indices.iter().map(|&i| host[i]).collect();
    Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(out, [indices.len()]), device)
}

/// Select `indices` rows from a rank-1 int tensor.
///
/// NOTE: We can't call `to_vec::<i64>()` directly here because Burn's
/// integer dtype is backend-dependent — `NdArray` uses `i64`, but `Wgpu`
/// uses `i32`. `to_vec` requires the requested `E` to match the stored
/// dtype exactly, so on wgpu it returns `DataError::TypeMismatch` and
/// `unwrap_or_default()` silently yields an empty vector, triggering an
/// out-of-bounds panic. Using `.iter::<i64>()` instead lets Burn handle
/// the per-element cast, so the host buffer is always populated.
fn select_rows_int<B: AutodiffBackend>(
    tensor: Tensor<B, 1, Int>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let data = tensor.into_data();
    let host: Vec<i64> = data.iter::<i64>().collect();
    let out: Vec<i64> = indices.iter().map(|&i| host[i]).collect();
    Tensor::<B, 1, Int>::from_data(burn::tensor::TensorData::new(out, [indices.len()]), device)
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, NdArray},
        module::{Module, ModuleVisitor, Param},
        optim::AdamConfig,
    };

    use super::*;
    use crate::{policy::mlp::MlpBurnPolicy, train::optimizer::BurnOptimizer};

    type B = Autodiff<NdArray<f32>>;

    /// Flatten every float parameter of a module into one host vector.
    fn params_flat<M: Module<B>>(module: &M) -> Vec<f32> {
        struct Collect {
            out: Vec<f32>,
        }
        impl ModuleVisitor<B> for Collect {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let host: Vec<f32> = param.val().into_data().to_vec().unwrap_or_default();
                self.out.extend(host);
            }
        }
        let mut c = Collect { out: Vec::new() };
        module.visit(&mut c);
        c.out
    }

    /// L2 norm of the parameter update `after - before`.
    fn update_norm(before: &[f32], after: &[f32]) -> f64 {
        assert_eq!(before.len(), after.len());
        before
            .iter()
            .zip(after)
            .map(|(&a, &b)| ((b - a) as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Smoke test: a Burn PPO trainer constructs and exposes the same
    /// config back through `config()`.
    #[test]
    fn ppo_trainer_burn_constructs() {
        let device = Default::default();
        let policy = MlpBurnPolicy::<B>::new(4, 2, 32, &device);
        let inner_opt = AdamConfig::new().init();
        let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 3e-4);
        let trainer = PPOTrainerBurn::new(PPOConfig::default(), policy, burn_opt).unwrap();
        assert_eq!(trainer.total_steps(), 0);
    }

    /// End-to-end: a single train_step against a synthetic batch
    /// completes without error, moves through the optimizer, and
    /// records `num_updates > 0`.
    #[test]
    fn ppo_trainer_burn_train_step_runs() {
        let device = Default::default();
        let policy = MlpBurnPolicy::<B>::new(4, 2, 16, &device);
        let inner_opt = AdamConfig::new().init();
        let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 1e-3);
        // Smaller batch_size so the synthetic 8-row batch produces > 1
        // minibatch per epoch.
        let config = PPOConfig::default().batch_size(4).n_epochs(1);
        let mut trainer = PPOTrainerBurn::new(config, policy, burn_opt).unwrap();

        let batch = 8;
        let obs_dim = 4;
        let mut obs_data = Vec::with_capacity(batch * obs_dim);
        for i in 0..batch * obs_dim {
            obs_data.push((i as f32) * 0.01);
        }
        let observations = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(obs_data, [batch, obs_dim]),
            &device,
        );
        let actions = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(vec![0i64, 1, 0, 1, 0, 1, 0, 1], [batch]),
            &device,
        );
        let old_log_probs = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(vec![-0.7f32; batch], [batch]),
            &device,
        );
        let old_values = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(vec![0.0f32; batch], [batch]),
            &device,
        );
        let advantages = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(
                vec![1.0f32, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5],
                [batch],
            ),
            &device,
        );
        let returns = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(vec![1.0f32; batch], [batch]),
            &device,
        );

        let stats = trainer
            .train_step(
                observations,
                actions,
                old_log_probs,
                old_values,
                advantages,
                returns,
                |p, o, a| p.evaluate_actions(o, a),
            )
            .unwrap();
        assert!(trainer.total_steps() > 0);
        // Stats should be finite.
        assert!(stats.policy_loss.is_finite());
        assert!(stats.value_loss.is_finite());
    }

    /// Issue #299: `PPOConfig::max_grad_norm` must actually be applied.
    ///
    /// Two trainers start from identical (cloned) policies, identical
    /// synthetic data, and identical seeds; the only difference is the cap.
    /// The tiny cap scales the gradients far below Adam's epsilon, so its
    /// parameter update must come out much smaller than the
    /// effectively-unbounded control's. The huge-cap control doubles as the
    /// no-clip baseline: gradients below the cap pass through untouched (see
    /// `train::grad_clip::tests` for the direct no-op assertion).
    #[test]
    fn ppo_trainer_burn_applies_max_grad_norm() {
        let device: burn::backend::ndarray::NdArrayDevice = Default::default();
        let policy = MlpBurnPolicy::<B>::new(4, 2, 16, &device);

        let batch = 8;
        let make_batch = || {
            let obs_dim = 4;
            let obs_data: Vec<f32> = (0..batch * obs_dim).map(|i| (i as f32) * 0.01).collect();
            let observations = Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(obs_data, [batch, obs_dim]),
                &device,
            );
            let actions = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::new(vec![0i64, 1, 0, 1, 0, 1, 0, 1], [batch]),
                &device,
            );
            let old_log_probs = Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(vec![-0.7f32; batch], [batch]),
                &device,
            );
            let old_values = Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(vec![0.0f32; batch], [batch]),
                &device,
            );
            let advantages = Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(
                    vec![1.0f32, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5],
                    [batch],
                ),
                &device,
            );
            let returns = Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(vec![1.0f32; batch], [batch]),
                &device,
            );
            (observations, actions, old_log_probs, old_values, advantages, returns)
        };

        let run = |config: PPOConfig, policy: MlpBurnPolicy<B>| -> f64 {
            let inner_opt = AdamConfig::new().init();
            let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> =
                BurnOptimizer::new(inner_opt, 1e-3);
            let mut trainer = PPOTrainerBurn::new(config, policy, burn_opt).unwrap();
            let before = params_flat(trainer.policy());
            let (observations, actions, old_log_probs, old_values, advantages, returns) =
                make_batch();
            trainer
                .train_step(
                    observations,
                    actions,
                    old_log_probs,
                    old_values,
                    advantages,
                    returns,
                    |p, o, a| p.evaluate_actions(o, a),
                )
                .unwrap();
            let after = params_flat(trainer.policy());
            update_norm(&before, &after)
        };

        // `batch_size == batch` → one minibatch; `n_epochs == 1` → exactly
        // one gradient step per trainer.
        let base = PPOConfig::default().batch_size(batch).n_epochs(1);
        let clipped = run(base.clone().max_grad_norm(1e-6), policy.clone());
        let unclipped = run(base.max_grad_norm(1e9), policy);

        assert!(unclipped > 0.0, "control update must move parameters");
        assert!(clipped > 0.0, "clipped update should still move parameters");
        assert!(
            clipped < 0.2 * unclipped,
            "tiny max_grad_norm must shrink the update: clipped {clipped} vs unclipped {unclipped}"
        );
    }
}
