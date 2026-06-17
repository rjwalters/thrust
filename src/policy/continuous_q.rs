//! Burn-backend continuous-action Q-critic for SAC (part of #136).
//!
//! [`ContinuousQNetwork`](crate::policy::continuous_q::ContinuousQNetwork) is
//! the `Q(s, a)` critic SAC uses for its twin critics and the two target
//! critics. It is structurally close to the
//! discrete [`crate::policy::q_network::QNetworkBurn`], but differs in two
//! ways that match the continuous-control setting:
//!
//! - **Input width is `obs_dim + action_dim`** — the forward pass concatenates
//!   the observation and the continuous action along the feature axis before
//!   the trunk, so the critic scores a specific `(state, action)` pair rather
//!   than every discrete action at once.
//! - **Output width is `1`** — a single scalar `Q(s, a)` per batch row.
//!
//! # Architecture
//!
//! ```text
//! obs    [batch, obs_dim]   ┐
//!                           ├─ concat → [batch, obs_dim + action_dim]
//! action [batch, action_dim]┘
//!     → fc1 →act→ fc2 →act→ (fc3 →act→)? q_head
//!  Q(s, a) [batch]
//! ```
//!
//! The trunk depth (2 or 3 layers), hidden width, activation, and init
//! recipe are configurable via
//! [`ContinuousQNetworkConfig`](crate::policy::continuous_q::ContinuousQNetworkConfig),
//! mirroring [`crate::policy::mlp::MlpBurnConfig`]. Construction can be seeded
//! for bit-exact reproducibility (see [`crate::policy::seeded_init`]).
//!
//! # Twin critics + targets
//!
//! The module owns no optimizer and no target state of its own — Burn's
//! optimizer consumes a module by value per step, so SAC's future
//! `SacTrainer` owns four independent instances (`q1`, `q2`, `q1_target`,
//! `q2_target`) as separate fields and steps the two online critics
//! independently. Instantiating two twins with different seeds (or
//! different unseeded draws) yields decorrelated critics, which is the
//! standard clipped-double-Q setup.
//!
//! # Target-net sync
//!
//! Two helpers mirror the record-based approach documented on
//! [`crate::policy::q_network::QNetworkBurn`]:
//!
//! - [`copy_params_from`](crate::policy::continuous_q::ContinuousQNetwork::copy_params_from) — hard copy (`theta_target <-
//!   theta_online`), used to initialize the targets.
//! - [`soft_update_from`](crate::policy::continuous_q::ContinuousQNetwork::soft_update_from) — Polyak blend (`theta_target <-
//!   tau * theta_online + (1 - tau) * theta_target`), used every step in SAC.
//!   `tau = 1.0` reduces exactly to a hard copy.

use burn::{
    module::Module,
    nn::{Initializer, Linear},
    tensor::{Tensor, activation, backend::Backend},
};

use super::mlp::{
    BurnActivation, derive_layer_seed, linear_from_weights, linear_with_init, seeded_layer_weights,
};

/// Configuration for [`ContinuousQNetwork`] architecture.
///
/// Mirrors [`crate::policy::q_network::QNetworkBurnConfig`] but adds the
/// depth/activation knobs SAC critics want (the default SAC critic is a
/// 2-layer, 256-wide ReLU MLP per Haarnoja et al. 2018). Held as a
/// separate type so the critic can be tuned independently of the actor.
#[derive(Debug, Clone, Copy)]
pub struct ContinuousQNetworkConfig {
    /// Number of hidden layers in the trunk. Only `2` or `3` are
    /// supported; anything else is treated as `2`.
    pub num_layers: usize,
    /// Width of every hidden layer.
    pub hidden_dim: usize,
    /// If `true`, initialize hidden-layer weights with orthogonal
    /// (gain `sqrt(2)`) and the Q-head with `gain = 0.01`. Set `false`
    /// for Burn's stock Kaiming-uniform default.
    pub use_orthogonal_init: bool,
    /// Activation applied between hidden layers.
    pub activation: BurnActivation,
    /// Optional construction seed. When `Some`, every layer is built
    /// from a deterministic, `StdRng`-driven weight buffer (see
    /// [`crate::policy::seeded_init`]) so two constructions with the same
    /// seed produce **bit-identical** critics. When `None` (the default)
    /// Burn's unseedable [`Initializer`] path is used verbatim. Twin
    /// critics should be seeded differently (or left unseeded) so they
    /// are decorrelated.
    pub seed: Option<u64>,
}

impl Default for ContinuousQNetworkConfig {
    fn default() -> Self {
        Self {
            num_layers: 2,
            hidden_dim: 256,
            use_orthogonal_init: true,
            activation: BurnActivation::ReLU,
            seed: None,
        }
    }
}

impl ContinuousQNetworkConfig {
    /// Set the construction seed, enabling the deterministic host-side
    /// init path in [`ContinuousQNetwork::with_config`].
    ///
    /// Builder-style; returns `self` for chaining:
    /// `ContinuousQNetworkConfig::default().with_seed(42)`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Continuous-action Q-critic on Burn: `Q(s, a) -> scalar`.
#[derive(Module, Debug)]
pub struct ContinuousQNetwork<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Option<Linear<B>>,
    q_head: Linear<B>,
    activation: BurnActivation,
}

impl<B: Backend> ContinuousQNetwork<B> {
    /// Build a fresh critic with the default config (2-layer ReLU,
    /// orthogonal init) and the given hidden width.
    pub fn new(obs_dim: usize, action_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        Self::with_config(
            obs_dim,
            action_dim,
            ContinuousQNetworkConfig { hidden_dim, ..Default::default() },
            device,
        )
    }

    /// Build a fresh critic with the given configuration.
    ///
    /// The trunk input width is `obs_dim + action_dim` (the concatenated
    /// `(state, action)` feature vector). When `config.seed` is `Some`,
    /// every layer is drawn from a per-layer-derived `StdRng` stream so
    /// the construction is bit-exact across runs and machines.
    pub fn with_config(
        obs_dim: usize,
        action_dim: usize,
        config: ContinuousQNetworkConfig,
        device: &B::Device,
    ) -> Self {
        let input_dim = obs_dim + action_dim;
        let hidden = config.hidden_dim;
        let use_third = config.num_layers >= 3;

        let (fc1, fc2, fc3, q_head) = if let Some(base_seed) = config.seed {
            // Seeded host-side init: each layer pulls from a distinct,
            // deterministically-derived RNG stream so equal-shaped layers
            // don't collide.
            let mut layer_idx = 0u64;
            let mut next = || {
                let s = derive_layer_seed(base_seed, layer_idx);
                layer_idx += 1;
                s
            };

            let w1 =
                seeded_layer_weights(next(), input_dim, hidden, config.use_orthogonal_init, false);
            let fc1 = linear_from_weights::<B>(input_dim, hidden, &w1, device);

            let w2 =
                seeded_layer_weights(next(), hidden, hidden, config.use_orthogonal_init, false);
            let fc2 = linear_from_weights::<B>(hidden, hidden, &w2, device);

            let fc3 = if use_third {
                let w3 =
                    seeded_layer_weights(next(), hidden, hidden, config.use_orthogonal_init, false);
                Some(linear_from_weights::<B>(hidden, hidden, &w3, device))
            } else {
                None
            };

            let wq = seeded_layer_weights(next(), hidden, 1, config.use_orthogonal_init, true);
            let q_head = linear_from_weights::<B>(hidden, 1, &wq, device);

            (fc1, fc2, fc3, q_head)
        } else {
            // Unseeded: route through Burn's `Initializer` exactly like
            // the discrete Q-network.
            let hidden_init = if config.use_orthogonal_init {
                Initializer::Orthogonal { gain: 2.0_f64.sqrt() }
            } else {
                Initializer::KaimingUniform { gain: 1.0_f64 / 3.0_f64.sqrt(), fan_out_only: false }
            };
            let output_init = if config.use_orthogonal_init {
                Initializer::Orthogonal { gain: 0.01 }
            } else {
                Initializer::KaimingUniform { gain: 1.0_f64 / 3.0_f64.sqrt(), fan_out_only: false }
            };

            let fc1 = linear_with_init::<B>(input_dim, hidden, hidden_init.clone(), device);
            let fc2 = linear_with_init::<B>(hidden, hidden, hidden_init.clone(), device);
            let fc3 = if use_third {
                Some(linear_with_init::<B>(hidden, hidden, hidden_init, device))
            } else {
                None
            };
            let q_head = linear_with_init::<B>(hidden, 1, output_init, device);

            (fc1, fc2, fc3, q_head)
        };

        Self { fc1, fc2, fc3, q_head, activation: config.activation }
    }

    fn apply_activation<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        match self.activation {
            BurnActivation::ReLU => activation::relu(x),
            BurnActivation::Tanh => activation::tanh(x),
        }
    }

    /// Forward pass: compute `Q(s, a)` for a batch of `(state, action)`
    /// pairs.
    ///
    /// * `obs` shape `[batch, obs_dim]`.
    /// * `action` shape `[batch, action_dim]`.
    /// * Returns Q-values of shape `[batch]` (the trailing singleton head
    ///   dimension is squeezed).
    ///
    /// The observation and action are concatenated along the feature axis
    /// before the trunk, so gradients flow into both inputs — SAC's actor
    /// loss differentiates `Q(s, a)` with respect to the action.
    pub fn forward(&self, obs: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 1> {
        let input = Tensor::cat(vec![obs, action], 1);
        let h = self.apply_activation(self.fc1.forward(input));
        let h = self.apply_activation(self.fc2.forward(h));
        let h = if let Some(fc3) = &self.fc3 {
            self.apply_activation(fc3.forward(h))
        } else {
            h
        };
        self.q_head.forward(h).squeeze_dim::<1>(1)
    }

    /// Replace this critic's parameters with a deep copy of `source`'s
    /// (hard target sync, `theta_target <- theta_online`).
    ///
    /// Returns a new module with the same architecture but `source`'s
    /// records. Burn's optimizer ownership model (`step` consumes the
    /// module by value) means we return `Self` rather than mutating
    /// `&mut self`; the trainer holds each target critic as a field and
    /// swaps it through this call when initializing the targets.
    pub fn copy_params_from(self, source: &ContinuousQNetwork<B>) -> ContinuousQNetwork<B> {
        // Burn modules clone their record cheaply (a tree of `Param`s
        // over reference-counted tensors). `load_record` consumes the
        // receiver and returns a new module with the source's params.
        self.load_record(source.clone().into_record())
    }

    /// Polyak (soft) target update:
    /// `theta_target <- tau * theta_online + (1 - tau) * theta_target`.
    ///
    /// Applied to every parameter tensor (every trunk layer's weight and
    /// bias, plus the Q-head's). `tau = 1.0` reduces exactly to a hard
    /// copy ([`copy_params_from`](Self::copy_params_from)); a `tau` in
    /// `(0, 1)` nudges the target toward the online network — the
    /// always-soft update SAC performs every gradient step.
    ///
    /// Mutates `self` in place. `self` (the target) and `online` must
    /// share the same architecture (depth, widths); the trainer
    /// constructs the targets as clones of the online critics, so this
    /// holds by construction.
    pub fn soft_update_from(&mut self, online: &ContinuousQNetwork<B>, tau: f64) {
        debug_assert!(
            (0.0..=1.0).contains(&tau),
            "tau must lie in [0, 1] for a convex Polyak blend, got {tau}"
        );
        debug_assert_eq!(
            self.fc3.is_some(),
            online.fc3.is_some(),
            "target and online critics must have the same depth"
        );

        // Move the target's layers out so we can consume their tensors in
        // the blend (Burn tensor ops take ownership). We reconstruct the
        // module afterwards.
        let target = std::mem::replace(self, online.clone());

        let fc1 = blend_linear(target.fc1, &online.fc1, tau);
        let fc2 = blend_linear(target.fc2, &online.fc2, tau);
        let fc3 = match (target.fc3, &online.fc3) {
            (Some(t), Some(o)) => Some(blend_linear(t, o, tau)),
            _ => None,
        };
        let q_head = blend_linear(target.q_head, &online.q_head, tau);

        *self = Self { fc1, fc2, fc3, q_head, activation: target.activation };
    }
}

/// Blend a single [`Linear`] layer's parameters as
/// `tau * online + (1 - tau) * target`, returning a fresh layer.
///
/// Consumes `target` (Burn tensor arithmetic takes ownership) and reads
/// `online` via cheap `val()` clones of its reference-counted tensors.
fn blend_linear<B: Backend>(target: Linear<B>, online: &Linear<B>, tau: f64) -> Linear<B> {
    use burn::module::Param;

    let one_minus_tau = 1.0 - tau;

    let target_w = target.weight.val();
    let online_w = online.weight.val();
    // `detach()` drops the autodiff graph so the blended tensor is a leaf
    // again — `Param::from_tensor` requires a leaf, and target critics
    // carry no gradients in SAC anyway (they are updated only by this
    // Polyak blend, never by an optimizer step).
    let weight = online_w.mul_scalar(tau).add(target_w.mul_scalar(one_minus_tau)).detach();

    let bias = match (target.bias, &online.bias) {
        (Some(target_b), Some(online_b)) => {
            let blended = online_b
                .val()
                .mul_scalar(tau)
                .add(target_b.val().mul_scalar(one_minus_tau))
                .detach();
            Some(Param::from_tensor(blended))
        }
        _ => None,
    };

    Linear::<B> { weight: Param::from_tensor(weight), bias }
}

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};

    use super::*;

    type B = Autodiff<NdArray<f32>>;

    /// Build a deterministic `[batch, dim]` tensor with ascending values
    /// on the default device.
    fn ramp(batch: usize, dim: usize) -> Tensor<B, 2> {
        let device = Default::default();
        let data: Vec<f32> = (0..batch * dim).map(|i| 0.01 * i as f32).collect();
        Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(data, [batch, dim]), &device)
    }

    #[test]
    fn forward_shape_two_layer() {
        let device = Default::default();
        let q = ContinuousQNetwork::<B>::new(4, 2, 32, &device);
        let obs = ramp(8, 4);
        let action = ramp(8, 2);
        let out = q.forward(obs, action);
        assert_eq!(out.dims(), [8], "2-layer critic must return [batch]");
    }

    #[test]
    fn forward_shape_three_layer() {
        let device = Default::default();
        let cfg = ContinuousQNetworkConfig { num_layers: 3, ..Default::default() };
        let q = ContinuousQNetwork::<B>::with_config(5, 3, cfg, &device);
        assert!(q.fc3.is_some(), "num_layers=3 must build a third trunk layer");
        let obs = ramp(6, 5);
        let action = ramp(6, 3);
        let out = q.forward(obs, action);
        assert_eq!(out.dims(), [6], "3-layer critic must return [batch]");
    }

    #[test]
    fn tanh_activation_branch() {
        let device = Default::default();
        let cfg = ContinuousQNetworkConfig {
            activation: BurnActivation::Tanh,
            use_orthogonal_init: false,
            ..Default::default()
        };
        let q = ContinuousQNetwork::<B>::with_config(3, 1, cfg, &device);
        let obs = ramp(2, 3);
        let action = ramp(2, 1);
        assert_eq!(q.forward(obs, action).dims(), [2]);
    }

    #[test]
    fn seeded_construction_is_bit_exact() {
        let device = Default::default();
        let cfg = ContinuousQNetworkConfig::default().with_seed(7);
        let a = ContinuousQNetwork::<B>::with_config(4, 2, cfg, &device);
        let b = ContinuousQNetwork::<B>::with_config(4, 2, cfg, &device);

        let obs = ramp(4, 4);
        let action = ramp(4, 2);
        let qa: Vec<f32> = a.forward(obs.clone(), action.clone()).into_data().to_vec().unwrap();
        let qb: Vec<f32> = b.forward(obs, action).into_data().to_vec().unwrap();
        assert_eq!(qa, qb, "same seed must yield bit-identical critics");
    }

    /// `copy_params_from` makes the target reproduce the online critic's
    /// outputs exactly on the same input.
    #[test]
    fn copy_params_from_matches_online() {
        let device = Default::default();
        let cfg = ContinuousQNetworkConfig {
            hidden_dim: 16,
            use_orthogonal_init: false,
            ..Default::default()
        };
        let online = ContinuousQNetwork::<B>::with_config(4, 2, cfg, &device);
        let target = ContinuousQNetwork::<B>::with_config(4, 2, cfg, &device);

        let obs = ramp(3, 4);
        let action = ramp(3, 2);

        // Fresh independent draws should disagree.
        let on_before: Vec<f32> =
            online.forward(obs.clone(), action.clone()).into_data().to_vec().unwrap();
        let tg_before: Vec<f32> =
            target.forward(obs.clone(), action.clone()).into_data().to_vec().unwrap();
        assert!(
            on_before.iter().zip(&tg_before).any(|(a, b)| (a - b).abs() > 1e-6),
            "fresh critics should disagree before copy"
        );

        let target = target.copy_params_from(&online);
        let on_after: Vec<f32> =
            online.forward(obs.clone(), action.clone()).into_data().to_vec().unwrap();
        let tg_after: Vec<f32> = target.forward(obs, action).into_data().to_vec().unwrap();
        for (a, b) in on_after.iter().zip(&tg_after) {
            assert!((a - b).abs() < 1e-6, "after copy: online={a} target={b}");
        }
    }

    /// `soft_update_from` with `tau = 1.0` is exactly a hard copy.
    #[test]
    fn soft_update_tau_one_equals_hard_copy() {
        let device = Default::default();
        let cfg = ContinuousQNetworkConfig {
            hidden_dim: 16,
            use_orthogonal_init: false,
            ..Default::default()
        };
        let online = ContinuousQNetwork::<B>::with_config(4, 2, cfg, &device);
        let mut target = ContinuousQNetwork::<B>::with_config(4, 2, cfg, &device);

        let obs = ramp(3, 4);
        let action = ramp(3, 2);

        target.soft_update_from(&online, 1.0);
        let on: Vec<f32> =
            online.forward(obs.clone(), action.clone()).into_data().to_vec().unwrap();
        let tg: Vec<f32> = target.forward(obs, action).into_data().to_vec().unwrap();
        for (a, b) in on.iter().zip(&tg) {
            assert!((a - b).abs() < 1e-6, "tau=1 soft update: online={a} target={b}");
        }
    }

    /// `soft_update_from` with `tau` in `(0, 1)` moves the target toward
    /// the online network: the post-update output sits strictly between
    /// the original target output and the online output, and the
    /// distance to online shrinks.
    #[test]
    fn soft_update_moves_target_toward_online() {
        let device = Default::default();
        let cfg = ContinuousQNetworkConfig {
            hidden_dim: 16,
            use_orthogonal_init: false,
            ..Default::default()
        };
        let online = ContinuousQNetwork::<B>::with_config(4, 2, cfg, &device);
        let mut target = ContinuousQNetwork::<B>::with_config(4, 2, cfg, &device);

        let obs = ramp(3, 4);
        let action = ramp(3, 2);

        let online_out: Vec<f32> =
            online.forward(obs.clone(), action.clone()).into_data().to_vec().unwrap();
        let target_before: Vec<f32> =
            target.forward(obs.clone(), action.clone()).into_data().to_vec().unwrap();

        let dist_before: f32 =
            online_out.iter().zip(&target_before).map(|(o, t)| (o - t).abs()).sum();
        assert!(dist_before > 1e-4, "test needs distinct critics to start");

        let tau = 0.25;
        target.soft_update_from(&online, tau);
        let target_after: Vec<f32> = target.forward(obs, action).into_data().to_vec().unwrap();

        let dist_after: f32 =
            online_out.iter().zip(&target_after).map(|(o, t)| (o - t).abs()).sum();
        assert!(
            dist_after < dist_before,
            "soft update should shrink distance to online: before={dist_before} after={dist_after}"
        );

        // And the params should actually have changed from the original
        // target (tau > 0).
        assert!(
            target_before.iter().zip(&target_after).any(|(a, b)| (a - b).abs() > 1e-6),
            "soft update with tau>0 must change the target"
        );
    }
}
