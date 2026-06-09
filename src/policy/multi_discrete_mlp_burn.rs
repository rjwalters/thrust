//! Burn-backend multi-discrete actor-critic MLP for factored action
//! spaces (phase 4 of the Burn migration, #65).
//!
//! Sibling to [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpPolicy`]
//! (tch path). The two modules implement the same shared-trunk +
//! per-dim head architecture; the only difference is the tensor
//! framework. Used by environments like Bucket Brigade and the
//! multi-agent self-play paths that need a factored
//! `[house_index, mode]` action.
//!
//! # Why a separate sibling instead of generalizing the tch version
//!
//! Same rationale as `mlp_burn.rs` — see the module-level doc in
//! `src/train/ppo_burn.rs`. The two backends have different generic
//! parameters (`<B: Backend>` for Burn vs no generic for tch + a
//! `VarStore` field), and phase 5 (#82) collapses the two when the
//! tch path is dropped.

use burn::{
    module::Module,
    nn::{Initializer, Linear},
    tensor::{Int, Tensor, activation, backend::Backend},
};

use super::mlp_burn::{BurnActivation, MlpBurnConfig, linear_with_init};

/// Multi-discrete MLP actor-critic policy on Burn.
///
/// Mirrors [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpPolicy`]:
/// shared trunk built from the same [`MlpBurnConfig`] knobs the
/// single-action [`crate::policy::mlp_burn::MlpBurnPolicy`] consumes,
/// plus one [`Linear`] action head per dimension. Per-step log-probs
/// are summed across dims (treating the dims as conditionally
/// independent given the state), and per-step entropies are averaged
/// — the same convention as the tch path so the parity tests can
/// compare both implementations on identical inputs.
#[derive(Module, Debug)]
pub struct MultiDiscreteMlpBurnPolicy<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Option<Linear<B>>,
    action_heads: Vec<Linear<B>>,
    value_head: Linear<B>,
    activation: BurnActivation,
}

impl<B: Backend> MultiDiscreteMlpBurnPolicy<B> {
    /// Build a fresh multi-discrete policy with the default 2-layer
    /// architecture (mirrors
    /// [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpPolicy::new`]).
    pub fn new(
        obs_dim: usize,
        action_dims: Vec<usize>,
        hidden_dim: usize,
        device: &B::Device,
    ) -> Self {
        let config = MlpBurnConfig { hidden_dim, ..Default::default() };
        Self::with_config(obs_dim, action_dims, config, device)
    }

    /// Build a fresh multi-discrete policy with custom configuration.
    pub fn with_config(
        obs_dim: usize,
        action_dims: Vec<usize>,
        config: MlpBurnConfig,
        device: &B::Device,
    ) -> Self {
        assert!(!action_dims.is_empty(), "action_dims must have at least one element");
        for (i, d) in action_dims.iter().enumerate() {
            assert!(*d >= 1, "action_dims[{i}] = {d}; must be >= 1");
        }

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

        let action_heads: Vec<Linear<B>> = action_dims
            .iter()
            .map(|&dim| linear_with_init::<B>(config.hidden_dim, dim, output_init.clone(), device))
            .collect();
        let value_head = linear_with_init::<B>(config.hidden_dim, 1, output_init, device);

        Self { fc1, fc2, fc3, action_heads, value_head, activation: config.activation }
    }

    fn apply_activation<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        match self.activation {
            BurnActivation::ReLU => activation::relu(x),
            BurnActivation::Tanh => activation::tanh(x),
        }
    }

    /// Shared-trunk features (mirrors
    /// [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpPolicy::encoder_features`]).
    pub fn encoder_features(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.apply_activation(self.fc1.forward(obs));
        let h = self.apply_activation(self.fc2.forward(h));
        if let Some(fc3) = &self.fc3 {
            self.apply_activation(fc3.forward(h))
        } else {
            h
        }
    }

    /// Forward pass: per-dim action logits plus value estimate.
    ///
    /// Returns `(Vec<logits_i>, value)` where
    /// `logits_i: [batch, action_dims[i]]` and `value: [batch]`.
    pub fn forward(&self, obs: Tensor<B, 2>) -> (Vec<Tensor<B, 2>>, Tensor<B, 1>) {
        let features = self.encoder_features(obs);
        let logits: Vec<Tensor<B, 2>> =
            self.action_heads.iter().map(|h| h.forward(features.clone())).collect();
        let value = self.value_head.forward(features).squeeze_dim::<1>(1);
        (logits, value)
    }

    /// Number of action dimensions (heads).
    pub fn num_action_dims(&self) -> usize {
        self.action_heads.len()
    }

    /// Evaluate given actions: per-step summed log-prob, per-step mean
    /// entropy (across dims), and value.
    ///
    /// # Arguments
    /// * `obs`     - `[batch, obs_dim]`
    /// * `actions` - `[batch, num_dims]` int (one action per dim per row)
    ///
    /// # Returns
    /// `(log_probs [batch], entropy [batch], values [batch])`.
    /// `log_probs` is summed across dims; `entropy` is averaged across
    /// dims (matching the tch convention so parity holds).
    pub fn evaluate_actions(
        &self,
        obs: Tensor<B, 2>,
        actions: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let (logits_per_dim, value) = self.forward(obs);

        let num_dims = logits_per_dim.len();
        assert!(num_dims > 0, "logits_per_dim must be non-empty");

        let mut summed_log_probs: Option<Tensor<B, 1>> = None;
        let mut summed_entropy: Option<Tensor<B, 1>> = None;

        for (i, logits) in logits_per_dim.into_iter().enumerate() {
            let log_probs = activation::log_softmax(logits, 1);
            let probs = log_probs.clone().exp();
            let per_dim_entropy: Tensor<B, 1> =
                -(probs * log_probs.clone()).sum_dim(1).squeeze_dim::<1>(1);

            // actions[:, i] as [batch, 1] int then gather → [batch]
            let actions_i: Tensor<B, 1, Int> =
                actions.clone().slice([0..actions.dims()[0], i..i + 1]).squeeze_dim::<1>(1);
            let per_dim_log_p: Tensor<B, 1> =
                log_probs.gather(1, actions_i.unsqueeze_dim::<2>(1)).squeeze_dim::<1>(1);

            summed_log_probs = Some(match summed_log_probs.take() {
                Some(acc) => acc + per_dim_log_p,
                None => per_dim_log_p,
            });
            summed_entropy = Some(match summed_entropy.take() {
                Some(acc) => acc + per_dim_entropy,
                None => per_dim_entropy,
            });
        }

        let log_probs = summed_log_probs.expect("at least one dim");
        // Mean entropy across dims (matches the tch trainer convention).
        let entropy = summed_entropy.expect("at least one dim").div_scalar(num_dims as f32);

        (log_probs, entropy, value)
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};

    use super::*;

    type B = Autodiff<NdArray<f32>>;

    #[test]
    fn test_creation_default() {
        let device = Default::default();
        let _policy = MultiDiscreteMlpBurnPolicy::<B>::new(4, vec![10, 2], 32, &device);
    }

    #[test]
    fn test_forward_shapes() {
        let device = Default::default();
        let policy = MultiDiscreteMlpBurnPolicy::<B>::with_config(
            4,
            vec![10, 2],
            MlpBurnConfig::default(),
            &device,
        );
        let obs = Tensor::<B, 2>::zeros([3, 4], &device);
        let (logits, value) = policy.forward(obs);
        assert_eq!(logits.len(), 2);
        assert_eq!(logits[0].dims(), [3, 10]);
        assert_eq!(logits[1].dims(), [3, 2]);
        assert_eq!(value.dims(), [3]);
    }

    #[test]
    fn test_evaluate_actions_shapes() {
        let device = Default::default();
        let policy = MultiDiscreteMlpBurnPolicy::<B>::new(4, vec![3, 4], 16, &device);
        let obs = Tensor::<B, 2>::zeros([5, 4], &device);
        let actions_data: Vec<i64> = vec![0, 1, 1, 2, 2, 0, 0, 3, 1, 2];
        let actions = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(actions_data, [5, 2]),
            &device,
        );
        let (log_probs, entropy, values) = policy.evaluate_actions(obs, actions);
        assert_eq!(log_probs.dims(), [5]);
        assert_eq!(entropy.dims(), [5]);
        assert_eq!(values.dims(), [5]);
    }

    #[test]
    fn test_num_action_dims() {
        let device = Default::default();
        let policy = MultiDiscreteMlpBurnPolicy::<B>::new(4, vec![10, 2, 5], 32, &device);
        assert_eq!(policy.num_action_dims(), 3);
    }
}
