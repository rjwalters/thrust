//! Burn-backend stochastic Gaussian actor for SAC (Soft Actor-Critic).
//!
//! This module implements [`SacActor`](crate::policy::sac_actor::SacActor),
//! the policy network half of the
//! SAC trainer for **continuous** control (parent proposal #136, PR C of
//! the decomposition). SAC's actor is a *reparameterized, tanh-squashed
//! Gaussian* policy: a shared MLP trunk feeds two heads — a `mean` head
//! and a `log_std` head — and an action is drawn as
//!
//! ```text
//! u = mean + std * eps        (eps ~ N(0, I), reparameterization trick)
//! a = tanh(u)                 (squash into the (-1, 1) box)
//! ```
//!
//! The squashing requires the standard change-of-variables correction to
//! the log-probability (Haarnoja et al. 2018, "Soft Actor-Critic
//! Algorithms and Applications", arXiv:1812.05905, Appendix C):
//!
//! ```text
//! log_prob(a) = gaussian_log_prob(u) - sum_i log(1 - tanh(u_i)^2 + eps)
//! ```
//!
//! # Why a dedicated small module (not a monolithic 5-net `SacPolicy`)?
//!
//! Burn's optimizer is move-through — it consumes a `Module` by value
//! each `step`. SAC needs **three** independently-stepped optimizers
//! (actor, critics, alpha), so the actor, the four critics, and the
//! entropy temperature must be owned as *separate* module fields by the
//! future `SacTrainer`, not fused into one module. This file therefore
//! provides only the actor; the critics live in `continuous_q` (PR D)
//! and the trainer assembles them (PR E).
//!
//! # Reproducibility
//!
//! [`SacActor::sample`](crate::policy::sac_actor::SacActor::sample)
//! consumes a caller-supplied [`StdRng`](rand::rngs::StdRng) for the
//! Gaussian noise, mirroring the seeding contract of
//! [`MlpBurnPolicy::get_action_host_seeded`](crate::policy::mlp::MlpBurnPolicy::get_action_host_seeded):
//! two calls with the same actor state, same `obs`, and same-seeded RNG
//! produce bit-identical `(action, log_prob)`. Construction is likewise
//! seedable via
//! [`SacActorConfig::with_seed`](crate::policy::sac_actor::SacActorConfig::with_seed),
//! reusing the host-side init helpers from [`crate::policy::seeded_init`].

use burn::{
    nn::{Initializer, Linear},
    tensor::{Tensor, activation, backend::Backend},
};
use rand::{Rng, rngs::StdRng};

use crate::policy::mlp::{
    BurnActivation, derive_layer_seed, linear_from_weights, linear_with_init, seeded_layer_weights,
};

/// Lower clamp for the `log_std` head output.
///
/// Mirrors the canonical SAC reference implementations (`[-20, 2]`):
/// clamping keeps the standard deviation in a numerically sane range
/// (`std` in roughly `[2e-9, 7.4]`) so the reparameterized sample and
/// its log-probability never overflow or collapse to a degenerate
/// distribution early in training.
pub const LOG_STD_MIN: f32 = -20.0;

/// Upper clamp for the `log_std` head output. See [`LOG_STD_MIN`].
pub const LOG_STD_MAX: f32 = 2.0;

/// Numerical-stability epsilon added inside the `log(1 - tanh(u)^2)`
/// tanh-squash correction so the log never sees an exact zero argument
/// when `|tanh(u)|` is driven to `1`.
const TANH_CORRECTION_EPS: f32 = 1e-6;

/// Configuration for the [`SacActor`] architecture.
///
/// Analogous to [`crate::policy::mlp::MlpBurnConfig`]: depth, width,
/// activation, an orthogonal-vs-Kaiming init toggle, and an optional
/// construction seed for bit-exact reproducibility.
#[derive(Debug, Clone, Copy)]
pub struct SacActorConfig {
    /// Number of hidden layers in the shared trunk. Only `2` or `3` are
    /// supported; anything else is treated as `2` (matching
    /// [`crate::policy::mlp::MlpBurnConfig`]).
    pub num_layers: usize,
    /// Width of every hidden layer.
    pub hidden_dim: usize,
    /// If `true`, initialize hidden-layer weights with
    /// [`Initializer::Orthogonal`] (gain `sqrt(2)`) and the output heads
    /// with `gain = 0.01`. Set `false` to fall back to Burn's default
    /// Kaiming-uniform init.
    pub use_orthogonal_init: bool,
    /// Activation applied between hidden layers.
    pub activation: BurnActivation,
    /// Optional construction seed. When `Some`,
    /// [`SacActor::with_config`](crate::policy::sac_actor::SacActor::with_config)
    /// builds every layer from a deterministic, [`StdRng`]-driven weight
    /// buffer (see [`crate::policy::seeded_init`]) so two constructions
    /// with the same seed produce **bit-identical** actors. When `None`
    /// (the default) Burn's unseedable [`Initializer`] path is used
    /// verbatim.
    pub seed: Option<u64>,
}

impl Default for SacActorConfig {
    fn default() -> Self {
        // SAC convention: 256-wide, 2-layer ReLU trunk.
        Self {
            num_layers: 2,
            hidden_dim: 256,
            use_orthogonal_init: true,
            activation: BurnActivation::ReLU,
            seed: None,
        }
    }
}

impl SacActorConfig {
    /// Set the construction seed, enabling the deterministic host-side
    /// init path in [`SacActor::with_config`].
    ///
    /// Builder-style; returns `self` for chaining:
    /// `SacActorConfig::default().with_seed(42)`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Stochastic, tanh-squashed Gaussian actor for SAC continuous control.
///
/// Layout:
///
/// ```text
/// obs → fc1 →act→ fc2 →act→ (fc3 →act→)? ┬─ mean_head    (μ)
///                                        └─ log_std_head (log σ, clamped)
/// ```
///
/// Both heads share the trunk activations. Use
/// [`sample`](Self::sample) for a reparameterized stochastic action with
/// its (squash-corrected) log-probability, and
/// [`mean_action`](Self::mean_action) for the deterministic
/// evaluation action `tanh(μ)`.
#[derive(burn::module::Module, Debug)]
pub struct SacActor<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Option<Linear<B>>,
    mean_head: Linear<B>,
    log_std_head: Linear<B>,
    activation: BurnActivation,
}

impl<B: Backend> SacActor<B> {
    /// Build a fresh actor on `device` with the given configuration.
    ///
    /// `obs_dim` is the observation width; `action_dim` is the number of
    /// continuous action dimensions (each squashed into `(-1, 1)`).
    pub fn with_config(
        obs_dim: usize,
        action_dim: usize,
        config: SacActorConfig,
        device: &B::Device,
    ) -> Self {
        // Seeded path: build every layer from a deterministic
        // `StdRng`-driven weight buffer so two constructions with the same
        // seed are bit-identical. Layer indices are fixed:
        // 0=fc1, 1=fc2, 2=fc3, 3=mean_head, 4=log_std_head.
        if let Some(seed) = config.seed {
            let orth = config.use_orthogonal_init;
            let mk = |idx: u64, d_in: usize, d_out: usize, is_head: bool| {
                let s = derive_layer_seed(seed, idx);
                let w = seeded_layer_weights(s, d_in, d_out, orth, is_head);
                linear_from_weights::<B>(d_in, d_out, &w, device)
            };
            let fc1 = mk(0, obs_dim, config.hidden_dim, false);
            let fc2 = mk(1, config.hidden_dim, config.hidden_dim, false);
            let fc3 = if config.num_layers >= 3 {
                Some(mk(2, config.hidden_dim, config.hidden_dim, false))
            } else {
                None
            };
            let mean_head = mk(3, config.hidden_dim, action_dim, true);
            let log_std_head = mk(4, config.hidden_dim, action_dim, true);
            return Self { fc1, fc2, fc3, mean_head, log_std_head, activation: config.activation };
        }

        let hidden_init = if config.use_orthogonal_init {
            Initializer::Orthogonal { gain: 2.0_f64.sqrt() }
        } else {
            Initializer::KaimingUniform { gain: 1.0_f64 / 3.0_f64.sqrt(), fan_out_only: false }
        };
        let head_init = if config.use_orthogonal_init {
            Initializer::Orthogonal { gain: 0.01 }
        } else {
            Initializer::KaimingUniform { gain: 1.0_f64 / 3.0_f64.sqrt(), fan_out_only: false }
        };

        let fc1 = linear_with_init::<B>(obs_dim, config.hidden_dim, hidden_init.clone(), device);
        let fc2 = linear_with_init::<B>(
            config.hidden_dim,
            config.hidden_dim,
            hidden_init.clone(),
            device,
        );
        let fc3 = if config.num_layers >= 3 {
            Some(linear_with_init::<B>(config.hidden_dim, config.hidden_dim, hidden_init, device))
        } else {
            None
        };
        let mean_head =
            linear_with_init::<B>(config.hidden_dim, action_dim, head_init.clone(), device);
        let log_std_head = linear_with_init::<B>(config.hidden_dim, action_dim, head_init, device);

        Self { fc1, fc2, fc3, mean_head, log_std_head, activation: config.activation }
    }

    fn apply_activation<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        match self.activation {
            BurnActivation::ReLU => activation::relu(x),
            BurnActivation::Tanh => activation::tanh(x),
        }
    }

    /// Shared-trunk feature representation for `obs` (shape
    /// `[batch, obs_dim]` → `[batch, hidden_dim]`). Gradients flow back
    /// into the trunk.
    fn encoder_features(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.apply_activation(self.fc1.forward(obs));
        let h = self.apply_activation(self.fc2.forward(h));
        if let Some(fc3) = &self.fc3 {
            self.apply_activation(fc3.forward(h))
        } else {
            h
        }
    }

    /// Forward pass producing the Gaussian parameters
    /// `(mean, log_std)`, each shape `[batch, action_dim]`.
    ///
    /// `log_std` is clamped to `[`[`LOG_STD_MIN`]`, `[`LOG_STD_MAX`]`]`
    /// before being returned.
    pub fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.encoder_features(obs);
        let mean = self.mean_head.forward(h.clone());
        let log_std = self.log_std_head.forward(h).clamp(LOG_STD_MIN, LOG_STD_MAX);
        (mean, log_std)
    }

    /// Deterministic evaluation action `tanh(μ)`, shape
    /// `[batch, action_dim]`, lying in the open box `(-1, 1)`.
    ///
    /// Used at evaluation time when the stochastic exploration of
    /// [`sample`](Self::sample) is undesirable.
    pub fn mean_action(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let (mean, _log_std) = self.forward(obs);
        activation::tanh(mean)
    }

    /// Draw one reparameterized, tanh-squashed action per row and return
    /// `(action, log_prob)`.
    ///
    /// * `action` is shape `[batch, action_dim]`, every element in the open box
    ///   `(-1, 1)`.
    /// * `log_prob` is shape `[batch]` — the joint log-density over all action
    ///   dimensions, with the tanh-squash change-of-variables correction
    ///   applied.
    ///
    /// The Gaussian noise `eps` is drawn on the host from `rng` (via the
    /// Box–Muller transform, matching [`crate::policy::seeded_init`]) and
    /// injected as a constant tensor, so gradients still flow through
    /// `mean` and `std` (the reparameterization trick) while the sample
    /// stays bit-exactly reproducible for a given seed — mirroring
    /// [`crate::policy::mlp::MlpBurnPolicy::get_action_host_seeded`].
    pub fn sample(&self, obs: Tensor<B, 2>, rng: &mut StdRng) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let (mean, log_std) = self.forward(obs);
        let dims = mean.dims();
        let [batch, action_dim] = dims;
        let device = mean.device();

        // Host-side standard-normal noise, one draw per (row, dim).
        let mut eps_data = Vec::with_capacity(batch * action_dim);
        for _ in 0..(batch * action_dim) {
            eps_data.push(standard_normal(rng));
        }
        let eps = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(eps_data, [batch, action_dim]),
            &device,
        );

        let std = log_std.clone().exp();
        // Reparameterized pre-squash sample u = mean + std * eps.
        let u = mean.clone() + std.clone() * eps;
        let action = activation::tanh(u.clone());

        // Gaussian log-prob of u under N(mean, std^2), summed over dims:
        //   log N(u) = -0.5 * ((u - mean)/std)^2 - log_std - 0.5*log(2π)
        let norm = (u.clone() - mean) / std;
        let log_two_pi = (2.0 * std::f32::consts::PI).ln();
        let gaussian_log_prob = (norm.clone() * norm) * (-0.5) - log_std - (0.5 * log_two_pi);
        let gaussian_log_prob = gaussian_log_prob.sum_dim(1).squeeze_dim::<1>(1);

        // Tanh-squash correction: subtract sum_i log(1 - tanh(u_i)^2 + eps).
        let tanh_u = action.clone();
        let one_minus_sq = -(tanh_u.clone() * tanh_u) + 1.0 + TANH_CORRECTION_EPS;
        let correction = one_minus_sq.log().sum_dim(1).squeeze_dim::<1>(1);

        let log_prob = gaussian_log_prob - correction;
        (action, log_prob)
    }

    /// Borrow the first shared-trunk linear layer.
    pub fn fc1(&self) -> &Linear<B> {
        &self.fc1
    }

    /// Borrow the second shared-trunk linear layer.
    pub fn fc2(&self) -> &Linear<B> {
        &self.fc2
    }

    /// Borrow the mean head.
    pub fn mean_head(&self) -> &Linear<B> {
        &self.mean_head
    }

    /// Borrow the log-std head.
    pub fn log_std_head(&self) -> &Linear<B> {
        &self.log_std_head
    }
}

/// Draw a single `N(0, 1)` sample from `rng` via the Box–Muller
/// transform.
///
/// Identical recipe to [`crate::policy::seeded_init`]'s private sampler:
/// the only RNG draws are `f32` uniforms, so two identically-seeded
/// [`StdRng`](rand::rngs::StdRng)s yield identical noise sequences (and
/// hence identical reparameterized actions).
fn standard_normal(rng: &mut StdRng) -> f32 {
    let u1: f32 = {
        let x: f32 = rng.random();
        if x <= f32::MIN_POSITIVE {
            f32::MIN_POSITIVE
        } else {
            x
        }
    };
    let u2: f32 = rng.random();
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0_f32 * std::f32::consts::PI * u2;
    r * theta.cos()
}

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};
    use rand::SeedableRng;

    use super::*;

    type B = Autodiff<NdArray<f32>>;

    fn obs_batch(
        batch: usize,
        obs_dim: usize,
        device: &burn::backend::ndarray::NdArrayDevice,
    ) -> Tensor<B, 2> {
        let n = batch * obs_dim;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.3).collect();
        Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(data, [batch, obs_dim]), device)
    }

    #[test]
    fn test_construction_two_layer() {
        let device = Default::default();
        let actor = SacActor::<B>::with_config(3, 1, SacActorConfig::default(), &device);
        assert!(actor.fc3.is_none());
    }

    #[test]
    fn test_construction_three_layer() {
        let device = Default::default();
        let cfg = SacActorConfig { num_layers: 3, ..Default::default() };
        let actor = SacActor::<B>::with_config(3, 2, cfg, &device);
        assert!(actor.fc3.is_some());
    }

    #[test]
    fn test_forward_shapes_two_layer() {
        let device = Default::default();
        let actor = SacActor::<B>::with_config(5, 3, SacActorConfig::default(), &device);
        let obs = obs_batch(8, 5, &device);
        let (mean, log_std) = actor.forward(obs);
        assert_eq!(mean.dims(), [8, 3]);
        assert_eq!(log_std.dims(), [8, 3]);
    }

    #[test]
    fn test_forward_shapes_three_layer() {
        let device = Default::default();
        let cfg = SacActorConfig { num_layers: 3, hidden_dim: 32, ..Default::default() };
        let actor = SacActor::<B>::with_config(5, 3, cfg, &device);
        let obs = obs_batch(8, 5, &device);
        let (mean, log_std) = actor.forward(obs);
        assert_eq!(mean.dims(), [8, 3]);
        assert_eq!(log_std.dims(), [8, 3]);
    }

    #[test]
    fn test_log_std_is_clamped() {
        // Large-magnitude obs would push an unclamped log_std head far
        // out of range; verify the clamp holds.
        let device = Default::default();
        let cfg =
            SacActorConfig { hidden_dim: 16, use_orthogonal_init: false, ..Default::default() };
        let actor = SacActor::<B>::with_config(4, 2, cfg, &device);
        let data: Vec<f32> = vec![100.0; 4 * 4];
        let obs = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(data, [4, 4]), &device);
        let (_mean, log_std) = actor.forward(obs);
        let vals: Vec<f32> = log_std.into_data().to_vec().unwrap();
        for v in vals {
            assert!((LOG_STD_MIN..=LOG_STD_MAX).contains(&v), "log_std {v} outside clamp range");
        }
    }

    #[test]
    fn test_mean_action_in_range_and_shape() {
        let device = Default::default();
        let actor = SacActor::<B>::with_config(4, 3, SacActorConfig::default(), &device);
        let obs = obs_batch(6, 4, &device);
        let action = actor.mean_action(obs);
        assert_eq!(action.dims(), [6, 3]);
        let vals: Vec<f32> = action.into_data().to_vec().unwrap();
        for v in vals {
            assert!(v > -1.0 && v < 1.0, "mean action {v} not in (-1, 1)");
        }
    }

    #[test]
    fn test_sample_actions_in_range_logprob_finite() {
        let device = Default::default();
        let actor =
            SacActor::<B>::with_config(4, 3, SacActorConfig::default().with_seed(7), &device);
        let obs = obs_batch(10, 4, &device);
        let mut rng = StdRng::seed_from_u64(123);
        let (action, log_prob) = actor.sample(obs, &mut rng);
        assert_eq!(action.dims(), [10, 3]);
        assert_eq!(log_prob.dims(), [10]);

        let acts: Vec<f32> = action.into_data().to_vec().unwrap();
        for v in acts {
            assert!(v > -1.0 && v < 1.0, "sampled action {v} not in (-1, 1)");
        }
        let lps: Vec<f32> = log_prob.into_data().to_vec().unwrap();
        for v in lps {
            assert!(v.is_finite(), "log_prob {v} not finite");
        }
    }

    /// Same-seed [`SacActor::sample`] is bit-identical; a different seed
    /// differs — mirroring the `get_action_host_seeded` reproducibility
    /// guarantee that the SAC trainer's `seed` will rely on.
    #[test]
    fn test_sample_is_bit_exact_in_seed() {
        let device = Default::default();
        let actor =
            SacActor::<B>::with_config(4, 2, SacActorConfig::default().with_seed(11), &device);
        let obs = obs_batch(5, 4, &device);

        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        let (a_a, lp_a) = actor.sample(obs.clone(), &mut rng_a);
        let (a_b, lp_b) = actor.sample(obs.clone(), &mut rng_b);
        let a_a: Vec<f32> = a_a.into_data().to_vec().unwrap();
        let a_b: Vec<f32> = a_b.into_data().to_vec().unwrap();
        let lp_a: Vec<f32> = lp_a.into_data().to_vec().unwrap();
        let lp_b: Vec<f32> = lp_b.into_data().to_vec().unwrap();
        assert_eq!(a_a, a_b, "same-seed actions must be bit-identical");
        assert_eq!(lp_a, lp_b, "same-seed log_probs must be bit-identical");

        let mut rng_c = StdRng::seed_from_u64(99);
        let (a_c, _) = actor.sample(obs, &mut rng_c);
        let a_c: Vec<f32> = a_c.into_data().to_vec().unwrap();
        assert_ne!(a_a, a_c, "different-seed actions must differ");
    }

    /// Construction is bit-identical under the same seed and differs
    /// under a different seed.
    #[test]
    fn test_with_seed_construction_is_bit_exact() {
        let device = Default::default();
        let obs = obs_batch(4, 4, &device);

        let a = SacActor::<B>::with_config(4, 2, SacActorConfig::default().with_seed(5), &device);
        let b = SacActor::<B>::with_config(4, 2, SacActorConfig::default().with_seed(5), &device);
        let c = SacActor::<B>::with_config(4, 2, SacActorConfig::default().with_seed(6), &device);

        let (ma, _) = a.forward(obs.clone());
        let (mb, _) = b.forward(obs.clone());
        let (mc, _) = c.forward(obs);
        let ma: Vec<f32> = ma.into_data().to_vec().unwrap();
        let mb: Vec<f32> = mb.into_data().to_vec().unwrap();
        let mc: Vec<f32> = mc.into_data().to_vec().unwrap();
        assert_eq!(ma, mb, "same seed must build bit-identical actors");
        assert_ne!(ma, mc, "different seed must build different actors");
    }
}
