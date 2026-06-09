//! Burn-backend Q-Network for DQN training (phase 4 of the Burn
//! migration, #65).
//!
//! Sibling to [`crate::policy::q_network::QNetwork`] (tch path). The two
//! modules share the same 2-layer Tanh backbone as
//! [`crate::policy::mlp::MlpPolicy`] /
//! [`crate::policy::mlp_burn::MlpBurnPolicy`] with PPO-style orthogonal
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
//! [`crate::policy::q_network_burn::QNetworkBurn::copy_params_from`]
//! so the Burn DQN trainer (phase 5) can drop in the same
//! `target.copy_params_from(&online)` call site shape the tch trainer uses.

use burn::{
    module::Module,
    nn::{Initializer, Linear},
    tensor::{Tensor, activation, backend::Backend},
};

use super::mlp_burn::linear_with_init;

/// Configuration for [`QNetworkBurn`] architecture.
///
/// Held as a separate type from
/// [`crate::policy::mlp_burn::MlpBurnConfig`] so that callers can
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
}

impl Default for QNetworkBurnConfig {
    fn default() -> Self {
        Self { hidden_dim: 64, use_orthogonal_init: true }
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

    /// Build a fresh Q-network with the given configuration.
    pub fn with_config(
        obs_dim: usize,
        n_actions: usize,
        config: QNetworkBurnConfig,
        device: &B::Device,
    ) -> Self {
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

        let fc1 = linear_with_init::<B>(obs_dim, config.hidden_dim, hidden_init.clone(), device);
        let fc2 = linear_with_init::<B>(config.hidden_dim, config.hidden_dim, hidden_init, device);
        let q_head = linear_with_init::<B>(config.hidden_dim, n_actions, output_init, device);

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
    /// Burn's idiomatic equivalent of `tch::nn::VarStore::copy` —
    /// returns a new module with the same architecture but the
    /// source's records. The Burn `Optimizer` ownership model
    /// (`step` consumes the module by value) means we return `Self`
    /// rather than mutating `&mut self`; the DQN trainer holds the
    /// target net in an `Option<Self>` and swaps it through this call.
    ///
    /// Mirrors [`crate::policy::q_network::QNetwork::copy_params_from`].
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

    /// Mirrors `q_network::tests::test_copy_params_from_byte_equal`
    /// from the tch path: after copying online → target, their forward
    /// outputs must agree exactly.
    #[test]
    fn test_copy_params_from_matches_online() {
        let device = Default::default();
        let online = QNetworkBurn::<B>::with_config(
            4,
            2,
            QNetworkBurnConfig { hidden_dim: 16, use_orthogonal_init: false },
            &device,
        );
        let target = QNetworkBurn::<B>::with_config(
            4,
            2,
            QNetworkBurnConfig { hidden_dim: 16, use_orthogonal_init: false },
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
            QNetworkBurnConfig { hidden_dim: 16, use_orthogonal_init: false },
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
