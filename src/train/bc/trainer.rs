//! Behavioral Cloning trainer (Burn backend).
//!
//! Sibling to [`crate::train::a2c::trainer::A2cTrainer`] and
//! [`crate::train::ppo::trainer::PPOTrainerBurn`]. [`BcTrainer`] reuses the
//! same policy ([`MlpBurnPolicy`]), optimizer ([`BurnOptimizer`]) and the
//! `Option<P>` move-through-`step` ownership model (Burn's
//! `Optimizer::step` consumes the module by value). It DIVERGES from every
//! RL trainer in loop shape:
//!
//! - **No environment interaction.** Training is a supervised epoch loop over a
//!   fixed [`Demonstrations`] dataset.
//! - **No advantages / returns / entropy.** The objective is the plain
//!   cross-entropy of the expert action ([`compute_bc_loss`]); the policy's
//!   value head is ignored.
//! - **Many gradient steps per call.** [`BcTrainer::train_epoch`] performs one
//!   step per minibatch (`ceil(len / batch_size)` steps per epoch), shuffling
//!   the example order from the trainer's seeded RNG so two runs with the same
//!   [`BcConfig::seed`] produce identical minibatch order and stats.
//!
//! [`MlpBurnPolicy`]: crate::policy::mlp::MlpBurnPolicy

use anyhow::{Result, anyhow};
use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    tensor::{Tensor, backend::AutodiffBackend},
};
use rand::{SeedableRng, rngs::StdRng};

use super::{
    config::BcConfig,
    dataset::Demonstrations,
    loss::{compute_bc_loss, scalar_f64},
};
use crate::train::{
    optimizer::{BackendOptimizer, BurnOptimizer},
    ppo::loss::generate_minibatch_indices_with_rng,
};

/// Per-epoch behavioral-cloning statistics.
///
/// Both fields are finite host-side `f64`s aggregated over the epoch's
/// minibatches. Purpose-built (rather than reusing the RL stats structs)
/// because BC has no policy/value/entropy decomposition — just the
/// supervised loss and the action-match accuracy.
#[derive(Debug, Clone, Copy, Default)]
pub struct BcEpochStats {
    /// Mean cross-entropy loss over the epoch's minibatches.
    pub loss: f64,
    /// Mean action-match accuracy in `[0, 1]`: the fraction of examples whose
    /// argmax-logit action matched the expert label, averaged over the epoch.
    pub accuracy: f64,
}

/// Burn-backend Behavioral Cloning trainer.
///
/// Generic over:
/// - `B: AutodiffBackend` — the Burn backend (e.g. `Autodiff<NdArray<f32>>`).
/// - `P: AutodiffModule<B>` — the policy module (only its logits head is
///   trained; the value head is ignored).
/// - `O: Optimizer<P, B>` — the Burn optimizer (typically
///   `AdamConfig::new().init()`).
///
/// The policy is held in `Option<P>` because Burn's `Optimizer::step`
/// consumes the module by value; each minibatch `.take()`s it and puts back
/// the updated copy.
pub struct BcTrainer<B, P, O>
where
    B: AutodiffBackend,
    P: AutodiffModule<B>,
    O: Optimizer<P, B>,
{
    config: BcConfig,
    policy: Option<P>,
    optimizer: BurnOptimizer<B, P, O>,
    /// Seedable RNG owned by the trainer, seeded from [`BcConfig::seed`], so
    /// the per-epoch minibatch shuffle is reproducible.
    rng: StdRng,
    total_steps: usize,
    total_epochs: usize,
}

impl<B, P, O> BcTrainer<B, P, O>
where
    B: AutodiffBackend,
    P: AutodiffModule<B> + Clone,
    O: Optimizer<P, B>,
{
    /// Build a new Burn behavioral-cloning trainer.
    ///
    /// Validates the config and seeds the trainer's minibatch-shuffle RNG
    /// from [`BcConfig::seed`].
    pub fn new(config: BcConfig, policy: P, optimizer: BurnOptimizer<B, P, O>) -> Result<Self> {
        config.validate()?;
        let rng = StdRng::seed_from_u64(config.seed);
        Ok(Self { config, policy: Some(policy), optimizer, rng, total_steps: 0, total_epochs: 0 })
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &BcConfig {
        &self.config
    }

    /// Borrow the policy. Panics if the trainer is mid-step (the policy has
    /// been moved into the optimizer); only safe to call between
    /// `train_epoch` invocations.
    pub fn policy(&self) -> &P {
        self.policy.as_ref().expect("policy is None mid-step")
    }

    /// Total completed gradient updates (one per minibatch across all epochs).
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Total completed epochs.
    pub fn total_epochs(&self) -> usize {
        self.total_epochs
    }

    /// Train for one full supervised pass over the demonstration dataset.
    ///
    /// 1. Draw a seeded shuffle of all example indices, partitioned into
    ///    minibatches of [`BcConfig::batch_size`] (`ceil(len / batch_size)`
    ///    minibatches).
    /// 2. For each minibatch: gather `(obs, actions)`, produce logits via
    ///    `forward_fn`, compute the cross-entropy [`compute_bc_loss`],
    ///    backprop, and step the optimizer once.
    /// 3. Aggregate the example-weighted mean loss and action-match accuracy
    ///    over the epoch.
    ///
    /// `forward_fn` lets the caller pick how logits are produced from the
    /// policy — e.g. `|p, o| p.forward(o).0` to drop
    /// [`MlpBurnPolicy`](crate::policy::mlp::MlpBurnPolicy)'s value head.
    ///
    /// Returns finite `loss` and `accuracy in [0, 1]`. Returns an `Err` if the
    /// dataset is empty.
    pub fn train_epoch<F>(
        &mut self,
        demos: &Demonstrations,
        mut forward_fn: F,
    ) -> Result<BcEpochStats>
    where
        F: FnMut(&P, Tensor<B, 2>) -> Tensor<B, 2>,
    {
        if demos.is_empty() {
            return Err(anyhow!("cannot train on an empty Demonstrations dataset"));
        }

        let device = B::Device::default();
        let batches =
            generate_minibatch_indices_with_rng(demos.len(), self.config.batch_size, &mut self.rng);

        // Example-weighted accumulators so uneven final minibatches do not
        // skew the epoch mean.
        let mut loss_sum = 0.0_f64;
        let mut correct = 0_usize;
        let mut seen = 0_usize;

        for indices in &batches {
            if indices.is_empty() {
                continue;
            }
            let k = indices.len();
            let (obs, actions) = demos.batch::<B>(indices, &device);

            // Take the policy out so we can move it through `step`.
            let policy = self
                .policy
                .take()
                .ok_or_else(|| anyhow!("policy is None; concurrent train_epoch calls?"))?;

            let logits = forward_fn(&policy, obs);

            // Host-side accuracy: argmax over the action dim vs expert label.
            let predicted: Vec<i64> = logits.clone().argmax(1).into_data().to_vec().unwrap();
            let expected: Vec<i64> = actions.clone().into_data().to_vec().unwrap();
            correct += predicted.iter().zip(expected.iter()).filter(|(p, e)| p == e).count();

            let loss = compute_bc_loss(logits, actions);
            loss_sum += scalar_f64(loss.clone()) * k as f64;
            seen += k;

            // Burn gradient flow: backward -> GradientsParams -> single step.
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &policy);
            let lr = self.optimizer.learning_rate();
            let policy = self.optimizer.inner_mut().step(lr, policy, grads);
            self.policy = Some(policy);

            self.total_steps += 1;
        }

        self.total_epochs += 1;

        let n = seen.max(1) as f64;
        Ok(BcEpochStats { loss: loss_sum / n, accuracy: correct as f64 / n })
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, NdArray},
        optim::AdamConfig,
    };

    use super::*;
    use crate::{policy::mlp::MlpBurnPolicy, train::optimizer::BurnOptimizer};

    type B = Autodiff<NdArray<f32>>;

    /// Build a tiny synthetic demonstration set: 6 examples, obs_dim 4,
    /// 2 discrete actions. Linearly separable-ish so accuracy is meaningful.
    fn tiny_demos() -> Demonstrations {
        let obs = vec![
            0.0, 0.0, 0.0, 0.0, // ex 0 -> action 0
            0.1, 0.0, 0.1, 0.0, // ex 1 -> action 0
            0.0, 0.1, 0.0, 0.1, // ex 2 -> action 0
            1.0, 1.0, 1.0, 1.0, // ex 3 -> action 1
            0.9, 1.0, 0.9, 1.0, // ex 4 -> action 1
            1.0, 0.9, 1.0, 0.9, // ex 5 -> action 1
        ];
        let actions = vec![0i64, 0, 0, 1, 1, 1];
        Demonstrations::new(obs, actions, 4).unwrap()
    }

    fn build_trainer(
        config: BcConfig,
    ) -> BcTrainer<B, MlpBurnPolicy<B>, impl Optimizer<MlpBurnPolicy<B>, B>> {
        let device = Default::default();
        // Seed the policy init so reproducibility tests are deterministic.
        let policy = MlpBurnPolicy::<B>::new_seeded(4, 2, 16, config.seed, &device);
        let inner_opt = AdamConfig::new().init();
        let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> =
            BurnOptimizer::new(inner_opt, config.learning_rate);
        BcTrainer::new(config, policy, burn_opt).unwrap()
    }

    /// Smoke test: a BC trainer constructs and reports zero steps/epochs.
    #[test]
    fn bc_trainer_constructs() {
        let trainer = build_trainer(BcConfig::default());
        assert_eq!(trainer.total_steps(), 0);
        assert_eq!(trainer.total_epochs(), 0);
    }

    /// Fast always-on smoke: one epoch over a tiny dataset performs at least
    /// one gradient step and yields finite loss + accuracy in `[0, 1]`.
    #[test]
    fn bc_train_epoch_smoke() {
        let config = BcConfig::default().batch_size(4).epochs(1).seed(1);
        let mut trainer = build_trainer(config);
        let demos = tiny_demos();

        let stats = trainer.train_epoch(&demos, |p, o| p.forward(o).0).unwrap();

        // 6 examples, batch_size 4 -> ceil(6/4) = 2 minibatches -> 2 steps.
        assert_eq!(trainer.total_steps(), 2);
        assert_eq!(trainer.total_epochs(), 1);
        assert!(stats.loss.is_finite(), "loss should be finite, got {}", stats.loss);
        assert!(stats.accuracy.is_finite(), "accuracy should be finite, got {}", stats.accuracy);
        assert!(
            (0.0..=1.0).contains(&stats.accuracy),
            "accuracy must be in [0, 1], got {}",
            stats.accuracy
        );
    }

    /// Reproducibility: two trainers built with the same seed and the same
    /// demos produce identical first-epoch stats.
    #[test]
    fn bc_train_epoch_is_reproducible() {
        let config = BcConfig::default().batch_size(4).epochs(1).seed(99);
        let demos = tiny_demos();

        let mut a = build_trainer(config.clone());
        let mut b = build_trainer(config);

        let stats_a = a.train_epoch(&demos, |p, o| p.forward(o).0).unwrap();
        let stats_b = b.train_epoch(&demos, |p, o| p.forward(o).0).unwrap();

        assert_eq!(stats_a.loss, stats_b.loss);
        assert_eq!(stats_a.accuracy, stats_b.accuracy);
    }

    /// An empty dataset is rejected rather than silently producing NaNs.
    #[test]
    fn bc_train_epoch_rejects_empty() {
        let mut trainer = build_trainer(BcConfig::default());
        let empty = Demonstrations::new(vec![], vec![], 4).unwrap();
        assert!(trainer.train_epoch(&empty, |p, o| p.forward(o).0).is_err());
    }
}
