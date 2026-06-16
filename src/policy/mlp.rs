//! Burn-backend MLP actor-critic policy.
//!
//! Implements a 2/3-layer MLP actor-critic architecture with orthogonal
//! initialization (PPO recipe — gain `sqrt(2)` on the trunk, `0.01` on
//! the output heads).
//!
//! # Entry points
//!
//! - `MlpBurnPolicy::new` — the simple scout-era constructor (random Kaiming
//!   init, 2 layers).
//! - `MlpBurnConfig` — builder-style configuration with orthogonal init,
//!   activation, and depth knobs; supports the encoder-tap helper that
//!   downstream regularizers want.
//!
//! # Why generic over `B: Backend`?
//!
//! Burn's idiomatic pattern is to make every `Module` generic over a
//! `Backend` type parameter (CPU `NdArray`, GPU `Wgpu`/`Cuda`,
//! autodiff-decorated variants, etc.). Production trainers can re-use
//! the same modules with a different backend at the top of the binary
//! without touching the policy code.

use burn::{
    module::{Module, Param},
    nn::{Initializer, Linear, LinearConfig},
    tensor::{Int, Tensor, activation, backend::Backend},
};

/// Build a [`Linear`] layer with an explicit weight initializer and a
/// zeroed bias.
///
/// Burn's `LinearConfig::with_initializer` applies the same initializer
/// to both the weight and the bias, but [`Initializer::Orthogonal`]
/// requires a rank-≥2 tensor and panics on the 1D bias. The PPO recipe
/// (mirrored on the tch path) initializes biases to zero anyway, so the
/// idiomatic Burn analogue is "Orthogonal on the weight, zero on the
/// bias". This helper packages that two-step setup.
///
/// Re-used by [`MlpBurnPolicy`],
/// [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpBurnPolicy`],
/// [`crate::policy::q_network::QNetworkBurn`], and
/// [`crate::policy::snake_cnn::SnakeCnnBurnPolicy`].
pub(crate) fn linear_with_init<B: Backend>(
    d_input: usize,
    d_output: usize,
    initializer: Initializer,
    device: &B::Device,
) -> Linear<B> {
    // Build a 2D weight Param via the initializer, and a 1D zero bias
    // Param via Param::from_tensor. `LinearConfig::with_initializer`
    // can't help here because it applies the same initializer to both
    // weight and bias, and `Initializer::Orthogonal` panics on the
    // rank-1 bias tensor (it requires `D >= 2`).
    let weight: Param<Tensor<B, 2>> = initializer.init_with::<B, 2, _>(
        [d_input, d_output],
        Some(d_input),
        Some(d_output),
        device,
    );
    let bias_tensor = Tensor::<B, 1>::zeros([d_output], device);
    Linear::<B> { weight, bias: Some(Param::from_tensor(bias_tensor)) }
}

/// Activation function applied between hidden layers in
/// [`MlpBurnPolicy`] (and its multi-discrete sibling).
///
/// Mirrors [`crate::policy::mlp::BurnActivation`] on the tch path; the two
/// enums are deliberately separate so the Burn module does not pull in
/// `tch` types under `--features training-burn` alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnActivation {
    /// Rectified linear unit (`max(0, x)`).
    ReLU,
    /// Hyperbolic tangent (`tanh(x)`).
    Tanh,
}

/// Configuration for [`MlpBurnPolicy`] architecture.
///
/// Mirrors [`crate::policy::mlp::MlpBurnConfig`] on the tch path. Stored
/// inside the policy so the parity tests can compare both backends on
/// identical hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct MlpBurnConfig {
    /// Number of hidden layers in the shared trunk. Only `2` or `3` are
    /// supported; anything else is treated as `2`.
    pub num_layers: usize,
    /// Width of every hidden layer.
    pub hidden_dim: usize,
    /// If `true`, initialize hidden-layer weights with
    /// [`Initializer::Orthogonal`] (gain `sqrt(2)`) and output heads
    /// with `Initializer::Orthogonal { gain = 0.01 }`. Set `false` to
    /// fall back to Burn's default Kaiming-uniform init.
    pub use_orthogonal_init: bool,
    /// Activation applied between hidden layers.
    pub activation: BurnActivation,
}

impl Default for MlpBurnConfig {
    fn default() -> Self {
        Self {
            num_layers: 2,
            hidden_dim: 64,
            use_orthogonal_init: true,
            activation: BurnActivation::Tanh,
        }
    }
}

/// Two- or three-layer MLP actor-critic for **discrete** action spaces,
/// ported to Burn.
///
/// Layout mirrors [`crate::policy::mlp::MlpBurnPolicy`] at a high level:
///
/// ```text
/// obs → fc1 →act→ fc2 →act→ (fc3 →act→)? policy_head (logits)
///                                       └─ value_head  (V(s))
/// ```
///
/// Both heads share the trunk activations — standard PPO actor-critic.
///
/// # Numerical parity
///
/// When constructed with `use_orthogonal_init = true` (the default), the
/// trunk uses [`Initializer::Orthogonal { gain: sqrt(2) }`] and the
/// output heads use `gain = 0.01`. These match the tch policy's init
/// gains exactly (see [`crate::policy::mlp::MlpBurnPolicy::with_config`]),
/// which is the necessary precondition for the phase-4 numerical-parity
/// check called out on issue #81.
#[derive(Module, Debug)]
pub struct MlpBurnPolicy<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Option<Linear<B>>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
    activation: BurnActivation,
}

impl<B: Backend> MlpBurnPolicy<B> {
    /// Backward-compatible 2-layer constructor (the phase 1 scout
    /// signature). Uses Burn's default Kaiming-uniform init — kept so
    /// the existing bandit trainer and parity tests are not perturbed.
    ///
    /// New call sites that want PPO-style orthogonal init should call
    /// [`MlpBurnPolicy::with_config`] instead.
    pub fn new(obs_dim: usize, action_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let config = MlpBurnConfig {
            num_layers: 2,
            hidden_dim,
            // Preserve scout behavior — the phase 1 scout used the
            // default LinearConfig init (Kaiming uniform), not the
            // PPO orthogonal recipe.
            use_orthogonal_init: false,
            activation: BurnActivation::Tanh,
        };
        Self::with_config(obs_dim, action_dim, config, device)
    }

    /// Build a fresh policy on `device` with the given configuration.
    ///
    /// This is the production constructor for phase 4 onwards. Mirrors
    /// [`crate::policy::mlp::MlpBurnPolicy::with_config`].
    pub fn with_config(
        obs_dim: usize,
        action_dim: usize,
        config: MlpBurnConfig,
        device: &B::Device,
    ) -> Self {
        let hidden_init = if config.use_orthogonal_init {
            Initializer::Orthogonal { gain: 2.0_f64.sqrt() }
        } else {
            // Burn's default — see LinearConfig docs.
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

        let policy_head =
            linear_with_init::<B>(config.hidden_dim, action_dim, output_init.clone(), device);
        let value_head = linear_with_init::<B>(config.hidden_dim, 1, output_init, device);

        Self { fc1, fc2, fc3, policy_head, value_head, activation: config.activation }
    }

    fn apply_activation<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        match self.activation {
            BurnActivation::ReLU => activation::relu(x),
            BurnActivation::Tanh => activation::tanh(x),
        }
    }

    /// Forward pass: returns `(logits, value)`.
    ///
    /// * `obs` is shape `[batch, obs_dim]`.
    /// * `logits` is shape `[batch, action_dim]` (pre-softmax).
    /// * `value` is shape `[batch]` (squeezed from `[batch, 1]`).
    pub fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let h = self.encoder_features(obs);
        let logits = self.policy_head.forward(h.clone());
        let value = self.value_head.forward(h).squeeze_dim::<1>(1);
        (logits, value)
    }

    /// Compute the shared-trunk feature representation for `obs`.
    ///
    /// Mirrors [`crate::policy::mlp::MlpBurnPolicy::encoder_features`] —
    /// auxiliary regularizers (cross-agent redundancy penalties,
    /// behavioural-diversity bonuses) tap this directly.
    ///
    /// Gradients flow back into the trunk.
    pub fn encoder_features(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.apply_activation(self.fc1.forward(obs));
        let h = self.apply_activation(self.fc2.forward(h));
        if let Some(fc3) = &self.fc3 {
            self.apply_activation(fc3.forward(h))
        } else {
            h
        }
    }

    /// Action-head output dimensionality (number of discrete actions).
    ///
    /// Reads the `policy_head` weight tensor's shape — Burn's
    /// [`burn::nn::Linear`] stores `weight: Param<Tensor<B, 2>>` with
    /// shape `[d_input, d_output]`, so `d_output` is the action
    /// cardinality. Used by the multi-agent joint trainer's
    /// [`crate::multi_agent::joint::JointPolicy::action_dims_joint`] impl
    /// to size the rollout action buffer without consuming RNG draws.
    pub fn policy_head_action_dim(&self) -> usize {
        self.policy_head.weight.val().dims()[1]
    }

    /// Borrow the first shared-trunk linear layer.
    pub fn fc1(&self) -> &Linear<B> {
        &self.fc1
    }

    /// Borrow the second shared-trunk linear layer.
    pub fn fc2(&self) -> &Linear<B> {
        &self.fc2
    }

    /// Borrow the policy (action-logits) head.
    pub fn policy_head(&self) -> &Linear<B> {
        &self.policy_head
    }

    /// Borrow the value (`V(s)`) head.
    pub fn value_head(&self) -> &Linear<B> {
        &self.value_head
    }

    /// Sample one action per row from the policy's categorical
    /// distribution and return `(actions_host, log_probs_host,
    /// values_host)` as plain `Vec`s.
    ///
    /// Thin backwards-compat wrapper around
    /// [`MlpBurnPolicy::get_action_host_seeded`] that constructs a
    /// thread-local RNG. **Not deterministic across calls** — use
    /// [`get_action_host_seeded`](Self::get_action_host_seeded) and pass
    /// a seeded [`rand::rngs::StdRng`] when reproducibility is required
    /// (PSRO/NFSP/joint trainer rollouts call the seeded form via the
    /// [`crate::multi_agent::joint::JointPolicy`] trait so that
    /// `PsroConfig::seed` / `NfspConfig::seed` produce bit-identical
    /// rollouts; see issue #114).
    ///
    /// Retained for example-driver convenience where the caller does
    /// not need bit-exact reproducibility and would otherwise have to
    /// thread an `&mut StdRng` through bespoke rollout loops.
    pub fn get_action_host(&self, obs: Tensor<B, 2>) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
        use rand::SeedableRng;
        // Seed from OS entropy so the wrapper remains stochastic for
        // non-deterministic callers (the same behavior pre-#114, just
        // routed through `StdRng`).
        let mut rng = rand::rngs::StdRng::from_os_rng();
        self.get_action_host_seeded(obs, &mut rng)
    }

    /// Same contract as [`get_action_host`](Self::get_action_host) but
    /// the host-side categorical draws consume `rng` instead of the
    /// thread-local generator.
    ///
    /// The trainer-side rollout loop does not need gradient flow
    /// through the sampled action (only the eventual
    /// [`MlpBurnPolicy::evaluate_actions`] call on the stored
    /// transitions matters for the PPO surrogate). We therefore do the
    /// categorical draw on the host with `rand`, sidestepping Burn
    /// 0.21's lack of a first-class `multinomial` op.
    ///
    /// Bit-exactness contract: two calls with the same `obs`, same
    /// `policy` state, and same-seeded `rng` (`StdRng::seed_from_u64`)
    /// must produce element-wise identical
    /// `(actions, log_probs, values)`. This is the load-bearing
    /// guarantee `PsroConfig::seed` / `NfspConfig::seed` rely on after
    /// issue #114.
    pub fn get_action_host_seeded(
        &self,
        obs: Tensor<B, 2>,
        rng: &mut rand::rngs::StdRng,
    ) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
        use rand::Rng;
        let (logits, value) = self.forward(obs);
        let probs = activation::softmax(logits.clone(), 1);
        let log_probs_all = activation::log_softmax(logits, 1);

        let dims = probs.dims();
        let batch = dims[0];
        let n_actions = dims[1];

        let probs_flat: Vec<f32> = probs.into_data().to_vec().expect("probs to_vec");
        let log_probs_flat: Vec<f32> =
            log_probs_all.into_data().to_vec().expect("log_probs to_vec");
        let values_host: Vec<f32> = value.into_data().to_vec().expect("values to_vec");

        let mut actions = Vec::with_capacity(batch);
        let mut log_probs = Vec::with_capacity(batch);
        for row in 0..batch {
            let u: f32 = rng.random();
            let mut cum = 0.0;
            let mut chosen = (n_actions - 1) as i64;
            for j in 0..n_actions {
                cum += probs_flat[row * n_actions + j];
                if u < cum {
                    chosen = j as i64;
                    break;
                }
            }
            actions.push(chosen);
            log_probs.push(log_probs_flat[row * n_actions + chosen as usize]);
        }
        (actions, log_probs, values_host)
    }

    /// Evaluate a batch of `(obs, actions)` pairs.
    ///
    /// Returns `(action_log_probs, entropy_per_row, values)` — the
    /// quantities the PPO surrogate loss needs. Entropy is per-row here
    /// (not the mean): the caller decides how to aggregate. This
    /// matches the tch policy's contract (the tch
    /// `evaluate_actions` returns a scalar mean; the trainer reduces
    /// per-row entropy on the Burn path inside
    /// [`crate::train::ppo::trainer::PPOTrainerBurn::train_step`]).
    pub fn evaluate_actions(
        &self,
        obs: Tensor<B, 2>,
        actions: Tensor<B, 1, Int>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let (logits, value) = self.forward(obs);
        let log_probs = activation::log_softmax(logits, 1);
        let probs = log_probs.clone().exp();

        let action_log_probs =
            log_probs.clone().gather(1, actions.unsqueeze_dim::<2>(1)).squeeze_dim::<1>(1);
        // H = -Σ p * log p over the action axis.
        let entropy = -(probs * log_probs).sum_dim(1).squeeze_dim::<1>(1);

        (action_log_probs, entropy, value)
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};

    use super::*;

    type B = Autodiff<NdArray<f32>>;

    #[test]
    fn test_policy_creation_default() {
        let device = Default::default();
        let _policy = MlpBurnPolicy::<B>::new(4, 2, 64, &device);
    }

    #[test]
    fn test_with_config_two_layer() {
        let device = Default::default();
        let cfg = MlpBurnConfig::default();
        let policy = MlpBurnPolicy::<B>::with_config(4, 2, cfg, &device);
        assert!(policy.fc3.is_none());
    }

    #[test]
    fn test_with_config_three_layer() {
        let device = Default::default();
        let cfg = MlpBurnConfig { num_layers: 3, ..Default::default() };
        let policy = MlpBurnPolicy::<B>::with_config(4, 2, cfg, &device);
        assert!(policy.fc3.is_some());
    }

    #[test]
    fn test_forward_pass_two_layer() {
        let device = Default::default();
        let cfg = MlpBurnConfig::default();
        let policy = MlpBurnPolicy::<B>::with_config(4, 2, cfg, &device);
        let obs = Tensor::<B, 2>::zeros([8, 4], &device);
        let (logits, values) = policy.forward(obs);
        assert_eq!(logits.dims(), [8, 2]);
        assert_eq!(values.dims(), [8]);
    }

    #[test]
    fn test_forward_pass_three_layer() {
        let device = Default::default();
        let cfg = MlpBurnConfig { num_layers: 3, ..Default::default() };
        let policy = MlpBurnPolicy::<B>::with_config(4, 2, cfg, &device);
        let obs = Tensor::<B, 2>::zeros([8, 4], &device);
        let (logits, values) = policy.forward(obs);
        assert_eq!(logits.dims(), [8, 2]);
        assert_eq!(values.dims(), [8]);
    }

    #[test]
    fn test_evaluate_actions_shapes() {
        let device = Default::default();
        let policy = MlpBurnPolicy::<B>::with_config(4, 2, MlpBurnConfig::default(), &device);
        let obs = Tensor::<B, 2>::zeros([8, 4], &device);
        let actions = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(vec![0i64, 1, 0, 1, 0, 1, 0, 1], [8]),
            &device,
        );
        let (log_probs, entropy, values) = policy.evaluate_actions(obs, actions);
        assert_eq!(log_probs.dims(), [8]);
        assert_eq!(entropy.dims(), [8]);
        assert_eq!(values.dims(), [8]);
    }

    #[test]
    fn test_relu_activation_branch() {
        let device = Default::default();
        let cfg = MlpBurnConfig {
            activation: BurnActivation::ReLU,
            use_orthogonal_init: false,
            ..Default::default()
        };
        let policy = MlpBurnPolicy::<B>::with_config(4, 2, cfg, &device);
        let obs = Tensor::<B, 2>::zeros([2, 4], &device);
        let (logits, _values) = policy.forward(obs);
        assert_eq!(logits.dims(), [2, 2]);
    }

    /// Bit-exact reproducibility of [`MlpBurnPolicy::get_action_host_seeded`]
    /// across same-seeded `StdRng` invocations.
    ///
    /// This is the load-bearing guarantee for `PsroConfig::seed` /
    /// `NfspConfig::seed` after issue #114: two
    /// `get_action_host_seeded` calls with the same `obs`, same policy
    /// state, and same-seeded RNG must produce element-wise identical
    /// `(actions, log_probs, values)`. The PSRO/NFSP integration
    /// tests (`tests/test_psro_matching_pennies.rs` and
    /// `tests/test_nfsp_matching_pennies.rs`) build their bit-exact
    /// reproducibility chain on this primitive.
    #[test]
    fn test_get_action_host_seeded_is_bit_exact() {
        use rand::{SeedableRng, rngs::StdRng};

        let device = Default::default();
        let policy = MlpBurnPolicy::<B>::with_config(4, 3, MlpBurnConfig::default(), &device);

        // Two-row batch so we exercise the per-row loop body.
        let obs_data = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let obs_a = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(obs_data.clone(), [2, 4]),
            &device,
        );
        let obs_b =
            Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(obs_data, [2, 4]), &device);

        // Same seed → bit-identical output.
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        let (a_a, lp_a, v_a) = policy.get_action_host_seeded(obs_a, &mut rng_a);
        let (a_b, lp_b, v_b) = policy.get_action_host_seeded(obs_b, &mut rng_b);
        assert_eq!(a_a, a_b, "same-seed actions must be bit-identical");
        assert_eq!(lp_a, lp_b, "same-seed log_probs must be bit-identical");
        assert_eq!(v_a, v_b, "same-seed values must be bit-identical");

        // Different seed → at least one row's action should differ
        // (modulo the unlikely event of identical samples — for 3
        // actions, P(both rows match) = 1/9 in expectation under
        // uniform logits; we use orthogonal init which doesn't
        // produce uniform logits, so the probability is even lower).
        let obs_c = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [2, 4]),
            &device,
        );
        let mut rng_c = StdRng::seed_from_u64(99);
        let (a_c, _, _) = policy.get_action_host_seeded(obs_c, &mut rng_c);
        // We can't assert hard inequality (low-but-nonzero probability
        // of accidental match) — but at least the call must succeed
        // and produce a 2-row response.
        assert_eq!(a_c.len(), 2, "two-row batch returns two actions");
    }
}
