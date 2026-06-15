//! Burn-backend multi-discrete actor-critic MLP for factored action
//! spaces.
//!
//! Shared-trunk + per-dim head architecture used by environments like
//! Bucket Brigade and the multi-agent self-play paths that need a
//! factored `[house_index, mode]` action.

use burn::{
    module::Module,
    nn::{Initializer, Linear},
    tensor::{Int, Tensor, activation, backend::Backend},
};

use super::mlp::{BurnActivation, MlpBurnConfig, linear_with_init};

/// Multi-discrete MLP actor-critic policy on Burn.
///
/// Shared trunk built from the same `MlpBurnConfig` knobs the
/// single-action `MlpBurnPolicy` consumes, plus one [`Linear`] action
/// head per dimension. Per-step log-probs are summed across dims
/// (treating the dims as conditionally independent given the state),
/// and per-step entropies are averaged.
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
    /// [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpBurnPolicy::new`]).
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
    /// [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpBurnPolicy::encoder_features`]).
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

    /// Per-dimension action cardinalities (one entry per head).
    ///
    /// Returns the same vector that was passed to
    /// [`MultiDiscreteMlpBurnPolicy::with_config`] /
    /// [`MultiDiscreteMlpBurnPolicy::new`]. Reads each action head's
    /// `weight` tensor shape — Burn's [`burn::nn::Linear`] stores `weight:
    /// Param<Tensor<B, 2>>` with shape `[d_input, d_output]`, so `d_output`
    /// is the per-dim cardinality. Used by the multi-agent joint trainer's
    /// [`crate::multi_agent::joint::JointPolicy::action_dims_joint`] impl to
    /// size the rollout action buffer.
    pub fn action_dim_cardinalities(&self) -> Vec<usize> {
        self.action_heads.iter().map(|h| h.weight.val().dims()[1]).collect()
    }

    /// Sample one action per row per dim from the per-dim categorical
    /// distributions and return `(actions_host, log_probs_host,
    /// values_host)` as plain `Vec`s.
    ///
    /// Thin backwards-compat wrapper around
    /// [`MultiDiscreteMlpBurnPolicy::get_action_host_seeded`] that
    /// constructs a thread-local RNG. **Not deterministic across
    /// calls** — use
    /// [`get_action_host_seeded`](Self::get_action_host_seeded) and
    /// pass a seeded [`rand::rngs::StdRng`] when reproducibility is
    /// required (PSRO/NFSP/joint trainer rollouts call the seeded form
    /// via the [`crate::multi_agent::joint::JointPolicy`] trait so that
    /// `PsroConfig::seed` / `NfspConfig::seed` produce bit-identical
    /// rollouts; see issue #114).
    pub fn get_action_host(&self, obs: Tensor<B, 2>) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
        use rand::SeedableRng;
        // Seed from OS entropy so the wrapper remains stochastic for
        // non-deterministic callers (same behavior pre-#114, routed
        // through `StdRng`).
        let mut rng = rand::rngs::StdRng::from_os_rng();
        self.get_action_host_seeded(obs, &mut rng)
    }

    /// Same contract as [`get_action_host`](Self::get_action_host) but
    /// the host-side categorical draws consume `rng` instead of the
    /// thread-local generator.
    ///
    /// Mirrors [`crate::policy::mlp::MlpBurnPolicy::get_action_host_seeded`] —
    /// the trainer-side rollout loop does not need gradient flow through
    /// the sampled action (only the eventual
    /// [`MultiDiscreteMlpBurnPolicy::evaluate_actions`] call on the stored
    /// transitions matters for the PPO surrogate). We therefore do the
    /// categorical draw on the host with `rand`, sidestepping Burn
    /// 0.21's lack of a first-class `multinomial` op.
    ///
    /// Layout:
    /// - `actions[row * num_dims + d]` is the action for dim `d` of row `row`.
    ///   Length is `batch * num_dims`.
    /// - `log_probs[row]` is the joint log-probability summed across dims.
    /// - `values[row]` is the value estimate.
    ///
    /// Bit-exactness contract: two calls with the same `obs`, same
    /// policy state, and same-seeded `rng` (`StdRng::seed_from_u64`)
    /// produce element-wise identical `(actions, log_probs, values)`.
    /// The per-row outer loop and per-dim inner loop consume RNG draws
    /// in a fixed `row major → dim major` order.
    pub fn get_action_host_seeded(
        &self,
        obs: Tensor<B, 2>,
        rng: &mut rand::rngs::StdRng,
    ) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
        use rand::Rng;
        let (logits_per_dim, value) = self.forward(obs);
        let num_dims = logits_per_dim.len();
        assert!(num_dims > 0, "at least one action dim");

        // Pre-extract host-side probs & log-probs for every head once. We
        // can sample dim-by-dim cheaply from there.
        let mut probs_per_dim: Vec<(usize, Vec<f32>, Vec<f32>)> = Vec::with_capacity(num_dims);
        let mut batch_opt: Option<usize> = None;
        for logits in logits_per_dim.into_iter() {
            let dims = logits.dims();
            let batch = dims[0];
            let n_actions = dims[1];
            batch_opt.get_or_insert(batch);
            let probs = activation::softmax(logits.clone(), 1);
            let log_probs = activation::log_softmax(logits, 1);
            let probs_flat: Vec<f32> = probs.into_data().to_vec().expect("probs to_vec");
            let log_probs_flat: Vec<f32> =
                log_probs.into_data().to_vec().expect("log_probs to_vec");
            probs_per_dim.push((n_actions, probs_flat, log_probs_flat));
        }
        let batch = batch_opt.unwrap_or(0);
        let values_host: Vec<f32> = value.into_data().to_vec().expect("values to_vec");

        let mut actions = vec![0_i64; batch * num_dims];
        let mut log_probs = vec![0.0_f32; batch];

        for row in 0..batch {
            let mut joint_lp = 0.0_f32;
            for (d, (n_actions, probs_flat, log_probs_flat)) in probs_per_dim.iter().enumerate() {
                let u: f32 = rng.random();
                let mut cum = 0.0;
                let mut chosen = (*n_actions - 1) as i64;
                for j in 0..*n_actions {
                    cum += probs_flat[row * n_actions + j];
                    if u < cum {
                        chosen = j as i64;
                        break;
                    }
                }
                actions[row * num_dims + d] = chosen;
                joint_lp += log_probs_flat[row * n_actions + chosen as usize];
            }
            log_probs[row] = joint_lp;
        }

        (actions, log_probs, values_host)
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

    /// Bit-exact reproducibility of
    /// [`MultiDiscreteMlpBurnPolicy::get_action_host_seeded`] across
    /// same-seeded `StdRng` invocations.
    ///
    /// Mirror of the unit test in `src/policy/mlp.rs` — the
    /// multi-discrete path consumes one RNG draw per (row, dim) pair
    /// in `row major → dim major` order, so two calls with the same
    /// observation and same-seeded RNG must produce element-wise
    /// identical `(actions, log_probs, values)`.
    #[test]
    fn test_get_action_host_seeded_is_bit_exact() {
        use rand::{SeedableRng, rngs::StdRng};

        let device = Default::default();
        let policy = MultiDiscreteMlpBurnPolicy::<B>::new(4, vec![3, 4], 16, &device);

        // 3-row batch, 2 action dims.
        let obs_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let obs_a = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(obs_data.clone(), [3, 4]),
            &device,
        );
        let obs_b =
            Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(obs_data, [3, 4]), &device);

        let mut rng_a = StdRng::seed_from_u64(123);
        let mut rng_b = StdRng::seed_from_u64(123);
        let (a_a, lp_a, v_a) = policy.get_action_host_seeded(obs_a, &mut rng_a);
        let (a_b, lp_b, v_b) = policy.get_action_host_seeded(obs_b, &mut rng_b);
        // 3 rows × 2 dims = 6 action ints.
        assert_eq!(a_a.len(), 6, "row-major (row, dim) layout = 3*2 entries");
        assert_eq!(a_a, a_b, "same-seed actions must be bit-identical");
        assert_eq!(lp_a, lp_b, "same-seed log_probs must be bit-identical");
        assert_eq!(v_a, v_b, "same-seed values must be bit-identical");
    }
}
