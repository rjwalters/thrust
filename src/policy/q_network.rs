//! Burn-backend Q-Network for DQN training (phase 4 of the Burn
//! migration, #65).
//!
//! Sibling to [`crate::policy::q_network::QNetworkBurn`] (tch path). The two
//! modules share the same 2-layer Tanh backbone as
//! [`crate::policy::mlp::MlpBurnPolicy`] /
//! [`crate::policy::mlp::MlpBurnPolicy`] with PPO-style orthogonal
//! initialization (gain `sqrt(2)` on the trunk, `0.01` on the Q-head). Unlike
//! the MLP policy this network has a single output head — its outputs are
//! interpreted directly as `Q(s, a)` values (no softmax).
//!
//! # Architecture
//!
//! ```text
//! Input [batch, obs_dim]
//!     → fc1 → Tanh
//!     → fc2 → Tanh
//!     → q_head
//!  Q-values [batch, n_actions]
//! ```
//!
//! # Target-net sync
//!
//! The tch path's `VarStore::copy(&source)` is replaced by Burn's
//! record-based clone:
//!
//! ```ignore
//! let snapshot = online.clone();           // cheap — Burn Modules clone
//! target = target.load_record(snapshot.into_record());
//! ```
//!
//! This is exposed as
//! [`crate::policy::q_network::QNetworkBurn::copy_params_from`]
//! so the Burn DQN trainer (phase 5) can drop in the same
//! `target.copy_params_from(&online)` call site shape the tch trainer uses.

use burn::{
    module::Module,
    nn::{Initializer, Linear},
    tensor::{Tensor, activation, backend::Backend},
};

use super::mlp::{derive_layer_seed, linear_from_weights, linear_with_init, seeded_layer_weights};

/// Configuration for [`QNetworkBurn`] architecture.
///
/// Held as a separate type from
/// [`crate::policy::mlp::MlpBurnConfig`] so that callers can
/// independently tune the Q-network (e.g. wider hidden_dim for richer
/// observation spaces) without dragging the policy module along.
#[derive(Debug, Clone, Copy)]
pub struct QNetworkBurnConfig {
    /// Width of every hidden layer.
    pub hidden_dim: usize,
    /// If `true`, initialize hidden-layer weights with orthogonal
    /// (gain `sqrt(2)`) and the Q-head with `gain = 0.01`. Set
    /// `false` for Burn's stock Kaiming-uniform default.
    pub use_orthogonal_init: bool,
    /// Optional construction seed. When `Some`, every layer is built from a
    /// deterministically-derived host-side RNG stream (see
    /// [`crate::policy::seeded_init`]) so two constructions with the same seed
    /// produce **bit-identical** networks. When `None` (the default) Burn's
    /// unseedable [`Initializer`] path is used verbatim. This mirrors
    /// [`crate::policy::continuous_q::ContinuousQNetworkConfig::seed`] so the
    /// discrete and continuous Q-networks share the same reproducibility hook.
    pub seed: Option<u64>,
}

impl Default for QNetworkBurnConfig {
    fn default() -> Self {
        Self { hidden_dim: 64, use_orthogonal_init: true, seed: None }
    }
}

impl QNetworkBurnConfig {
    /// Set the construction seed, enabling the deterministic host-side init
    /// path in [`QNetworkBurn::with_config`].
    ///
    /// ```
    /// # use thrust_rl::policy::q_network::QNetworkBurnConfig;
    /// let cfg = QNetworkBurnConfig::default().with_seed(42);
    /// assert_eq!(cfg.seed, Some(42));
    /// ```
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Two-layer Tanh Q-network on Burn.
#[derive(Module, Debug)]
pub struct QNetworkBurn<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    q_head: Linear<B>,
}

impl<B: Backend> QNetworkBurn<B> {
    /// Build a fresh Q-network with the default orthogonal-init config.
    pub fn new(obs_dim: usize, n_actions: usize, hidden_dim: usize, device: &B::Device) -> Self {
        Self::with_config(
            obs_dim,
            n_actions,
            QNetworkBurnConfig { hidden_dim, ..Default::default() },
            device,
        )
    }

    /// Build a fresh Q-network whose weights are seeded for bit-exact
    /// reproducibility. Two calls with the same `seed` (and shapes) produce
    /// byte-identical networks; mirrors
    /// [`crate::policy::continuous_q::ContinuousQNetwork`].
    pub fn with_seed(
        obs_dim: usize,
        n_actions: usize,
        hidden_dim: usize,
        seed: u64,
        device: &B::Device,
    ) -> Self {
        Self::with_config(
            obs_dim,
            n_actions,
            QNetworkBurnConfig { hidden_dim, ..Default::default() }.with_seed(seed),
            device,
        )
    }

    /// Build a fresh Q-network with the given configuration.
    pub fn with_config(
        obs_dim: usize,
        n_actions: usize,
        config: QNetworkBurnConfig,
        device: &B::Device,
    ) -> Self {
        let hidden = config.hidden_dim;

        let (fc1, fc2, q_head) = if let Some(base_seed) = config.seed {
            // Seeded host-side init: each layer pulls from a distinct,
            // deterministically-derived RNG stream so equal-shaped layers
            // don't collide. Mirrors `ContinuousQNetwork::with_config`.
            let mut layer_idx = 0u64;
            let mut next = || {
                let s = derive_layer_seed(base_seed, layer_idx);
                layer_idx += 1;
                s
            };

            let w1 =
                seeded_layer_weights(next(), obs_dim, hidden, config.use_orthogonal_init, false);
            let fc1 = linear_from_weights::<B>(obs_dim, hidden, &w1, device);

            let w2 =
                seeded_layer_weights(next(), hidden, hidden, config.use_orthogonal_init, false);
            let fc2 = linear_from_weights::<B>(hidden, hidden, &w2, device);

            let wq =
                seeded_layer_weights(next(), hidden, n_actions, config.use_orthogonal_init, true);
            let q_head = linear_from_weights::<B>(hidden, n_actions, &wq, device);

            (fc1, fc2, q_head)
        } else {
            // Unseeded: route through Burn's `Initializer` verbatim.
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

            let fc1 = linear_with_init::<B>(obs_dim, hidden, hidden_init.clone(), device);
            let fc2 = linear_with_init::<B>(hidden, hidden, hidden_init, device);
            let q_head = linear_with_init::<B>(hidden, n_actions, output_init, device);

            (fc1, fc2, q_head)
        };

        Self { fc1, fc2, q_head }
    }

    /// Forward pass: compute `Q(s, a)` for every action `a`.
    ///
    /// * `obs` shape `[batch, obs_dim]`.
    /// * Returns Q-values of shape `[batch, n_actions]`.
    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::tanh(self.fc1.forward(obs));
        let h = activation::tanh(self.fc2.forward(h));
        self.q_head.forward(h)
    }

    /// Replace this network's parameters with a deep copy of `source`'s
    /// parameters.
    ///
    /// Returns a new module with the same architecture but the
    /// source's records. Burn's `Optimizer` ownership model (`step`
    /// consumes the module by value) means we return `Self` rather
    /// than mutating `&mut self`; the DQN trainer holds the target
    /// net in an `Option<Self>` and swaps it through this call.
    pub fn copy_params_from(self, source: &QNetworkBurn<B>) -> QNetworkBurn<B>
    where
        B: Backend,
    {
        // Burn modules can clone their record cheaply (the record is a
        // tree of `Param`s; each `Param` is cheap to clone since the
        // underlying tensors are reference-counted on the autodiff
        // path). `load_record` consumes the receiver and returns a new
        // module with the source's parameters.
        self.load_record(source.clone().into_record())
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};

    use super::*;

    type B = Autodiff<NdArray<f32>>;

    #[test]
    fn test_q_network_burn_creation() {
        let device = Default::default();
        let _q_net = QNetworkBurn::<B>::new(4, 2, 64, &device);
    }

    #[test]
    fn test_q_network_burn_forward_shape() {
        let device = Default::default();
        let q_net = QNetworkBurn::<B>::new(4, 3, 32, &device);
        let obs = Tensor::<B, 2>::zeros([8, 4], &device);
        let q_values = q_net.forward(obs);
        assert_eq!(q_values.dims(), [8, 3]);
    }

    /// Two seeded constructions with the same seed must yield bit-identical
    /// Q-values, while different seeds must disagree. Mirrors
    /// `ContinuousQNetwork`'s `seeded_construction_is_bit_exact`.
    #[test]
    fn seeded_construction_is_bit_exact() {
        let device = Default::default();
        let a = QNetworkBurn::<B>::with_seed(4, 2, 16, 7, &device);
        let b = QNetworkBurn::<B>::with_seed(4, 2, 16, 7, &device);
        let c = QNetworkBurn::<B>::with_seed(4, 2, 16, 8, &device);

        let obs = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [2, 4]),
            &device,
        );
        let qa: Vec<f32> = a.forward(obs.clone()).into_data().to_vec().unwrap();
        let qb: Vec<f32> = b.forward(obs.clone()).into_data().to_vec().unwrap();
        let qc: Vec<f32> = c.forward(obs).into_data().to_vec().unwrap();

        assert_eq!(qa, qb, "same seed must yield bit-identical Q-networks");
        assert!(
            qa.iter().zip(&qc).any(|(x, y)| (x - y).abs() > 1e-6),
            "different seeds should yield different Q-networks"
        );
    }

    /// Mirrors `q_network::tests::test_copy_params_from_byte_equal`
    /// from the tch path: after copying online → target, their forward
    /// outputs must agree exactly.
    #[test]
    fn test_copy_params_from_matches_online() {
        let device = Default::default();
        let online = QNetworkBurn::<B>::with_config(
            4,
            2,
            QNetworkBurnConfig { hidden_dim: 16, use_orthogonal_init: false, ..Default::default() },
            &device,
        );
        let target = QNetworkBurn::<B>::with_config(
            4,
            2,
            QNetworkBurnConfig { hidden_dim: 16, use_orthogonal_init: false, ..Default::default() },
            &device,
        );

        // Build a simple synthetic batch.
        let obs = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [2, 4]),
            &device,
        );

        // Sanity check: fresh nets should disagree (different orthogonal
        // draws). We compare via host floats since we don't have a
        // direct |a - b| reduction here.
        let q_online_before: Vec<f32> = online.forward(obs.clone()).into_data().to_vec().unwrap();
        let q_target_before: Vec<f32> = target.forward(obs.clone()).into_data().to_vec().unwrap();
        let any_diff_before =
            q_online_before.iter().zip(&q_target_before).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff_before, "expected fresh nets to disagree before copy");

        // Sync target ← online.
        let online_for_recall = QNetworkBurn::<B>::with_config(
            4,
            2,
            QNetworkBurnConfig { hidden_dim: 16, use_orthogonal_init: false, ..Default::default() },
            &device,
        );
        // To compare, we want the sync to make `target` match `online`
        // exactly. The Burn idiom returns a fresh module, which we
        // re-bind:
        let target_copied = target.copy_params_from(&online);
        let q_online_after: Vec<f32> = online.forward(obs.clone()).into_data().to_vec().unwrap();
        let q_target_after: Vec<f32> =
            target_copied.forward(obs.clone()).into_data().to_vec().unwrap();
        for (a, b) in q_online_after.iter().zip(&q_target_after) {
            assert!(
                (a - b).abs() < 1e-6,
                "Q output mismatch after copy_params_from: online={a} target={b}"
            );
        }

        // And a *fresh* `online_for_recall` (independent draws) should
        // still disagree with the synced target — confirms we copied
        // online's specific draws, not "any zero-init".
        let q_fresh: Vec<f32> = online_for_recall.forward(obs).into_data().to_vec().unwrap();
        let still_differs = q_fresh.iter().zip(&q_target_after).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(still_differs, "synced target unexpectedly matched a *different* fresh net");
    }
}
