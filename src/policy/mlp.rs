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
    nn::{Initializer, Linear},
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

/// Build a [`Linear`] layer from a pre-computed, row-major
/// `[d_input, d_output]` weight buffer (and a zeroed bias).
///
/// This is the seeded counterpart to [`linear_with_init`]: instead of
/// routing through Burn's unseedable [`Initializer`], the caller
/// supplies weights produced by
/// [`crate::policy::seeded_init`] (driven by `StdRng::seed_from_u64`),
/// so two constructions with the same seed yield bit-identical layers.
/// Used by the `with_config` seeded path on both
/// [`MlpBurnPolicy`] and
/// [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpBurnPolicy`]
/// (issue #135).
pub(crate) fn linear_from_weights<B: Backend>(
    d_input: usize,
    d_output: usize,
    weights: &[f32],
    device: &B::Device,
) -> Linear<B> {
    debug_assert_eq!(weights.len(), d_input * d_output, "weight buffer must be d_input * d_output");
    let weight_tensor = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(weights.to_vec(), [d_input, d_output]),
        device,
    );
    let bias_tensor = Tensor::<B, 1>::zeros([d_output], device);
    Linear::<B> {
        weight: Param::from_tensor(weight_tensor),
        bias: Some(Param::from_tensor(bias_tensor)),
    }
}

/// Produce a seeded weight buffer for one layer, honoring the
/// orthogonal-vs-Kaiming recipe shared by both MLP policies.
///
/// Mirrors the gains used on the unseeded [`Initializer`] path:
/// orthogonal trunk `sqrt(2)` / head `0.01`; Kaiming `1/sqrt(3)` for
/// both heads and trunk (Burn's `KaimingUniform` default gain).
pub(crate) fn seeded_layer_weights(
    seed: u64,
    d_in: usize,
    d_out: usize,
    use_orthogonal: bool,
    is_head: bool,
) -> Vec<f32> {
    use crate::policy::seeded_init::{seeded_kaiming_uniform, seeded_orthogonal};
    if use_orthogonal {
        let gain = if is_head { 0.01_f32 } else { 2.0_f32.sqrt() };
        seeded_orthogonal(seed, d_in, d_out, gain)
    } else {
        let gain = 1.0_f32 / 3.0_f32.sqrt();
        seeded_kaiming_uniform(seed, d_in, d_out, gain)
    }
}

/// Derive a distinct per-layer seed from a base construction seed.
///
/// Each `Linear` layer in a policy must draw from a *different* RNG
/// stream — otherwise every layer of the same shape would get identical
/// weights. We mix the base seed with a small per-layer index using a
/// SplitMix64-style finalizer so the streams are decorrelated yet fully
/// determined by `(base_seed, layer_index)`.
pub(crate) fn derive_layer_seed(base_seed: u64, layer_index: u64) -> u64 {
    let mut z = base_seed.wrapping_add(layer_index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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

/// Host-side categorical distribution for one or more rows, produced by
/// [`MlpBurnPolicy::forward_to_host_dist`].
///
/// Holds the flattened per-row `probs` / `log_probs` (`[batch, n_actions]`
/// row-major) and per-row `values`. The seeded draw lives in
/// [`HostCategoricalDist::sample_actions`] so the tensor forward and the
/// RNG-consuming sample are cleanly separated — the precondition that lets
/// the batched sampler do one `[N, obs_dim]` forward while keeping the
/// per-row RNG draw order bit-identical (issue #235).
struct HostCategoricalDist {
    batch: usize,
    n_actions: usize,
    probs_flat: Vec<f32>,
    log_probs_flat: Vec<f32>,
    values_host: Vec<f32>,
}

impl HostCategoricalDist {
    /// Draw one action per row from the categorical distribution, consuming
    /// exactly one `rng.random()` per row in ascending row order. Returns
    /// `(actions, log_probs_of_chosen, values)`.
    ///
    /// The draw is byte-for-byte the loop that
    /// [`MlpBurnPolicy::get_action_host_seeded`] used before the
    /// forward/sample split, so a same-seeded `rng` reproduces the exact
    /// same action stream (issue #114 / #235).
    fn sample_actions(&self, rng: &mut rand::rngs::StdRng) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
        use rand::Rng;
        let mut actions = Vec::with_capacity(self.batch);
        let mut log_probs = Vec::with_capacity(self.batch);
        for row in 0..self.batch {
            let u: f32 = rng.random();
            let mut cum = 0.0;
            let mut chosen = (self.n_actions - 1) as i64;
            for j in 0..self.n_actions {
                cum += self.probs_flat[row * self.n_actions + j];
                if u < cum {
                    chosen = j as i64;
                    break;
                }
            }
            actions.push(chosen);
            log_probs.push(self.log_probs_flat[row * self.n_actions + chosen as usize]);
        }
        (actions, log_probs, self.values_host.clone())
    }
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
    /// Optional construction seed. When `Some`, `with_config` builds
    /// every layer from a deterministic, [`StdRng`](rand::rngs::StdRng)-
    /// driven weight buffer (see [`crate::policy::seeded_init`]) instead
    /// of Burn's unseedable [`Initializer`], so two constructions with
    /// the same seed produce **bit-identical** policies. When `None`
    /// (the default) the behavior is unchanged — Burn's `Initializer`
    /// path is used verbatim. This is the load-bearing knob behind the
    /// end-to-end `PsroConfig::seed` / `NfspConfig::seed` reproducibility
    /// contract (issue #135). The seeded path covers **both** the
    /// orthogonal and Kaiming-uniform recipes (selected by
    /// `use_orthogonal_init`), so seeding works regardless of which init
    /// the caller picks.
    pub seed: Option<u64>,
}

impl Default for MlpBurnConfig {
    fn default() -> Self {
        Self {
            num_layers: 2,
            hidden_dim: 64,
            use_orthogonal_init: true,
            activation: BurnActivation::Tanh,
            seed: None,
        }
    }
}

impl MlpBurnConfig {
    /// Set the construction seed, enabling the deterministic
    /// host-side init path in `with_config`.
    ///
    /// Builder-style; returns `self` for chaining:
    /// `MlpBurnConfig::default().with_seed(42)`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
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
            seed: None,
        };
        Self::with_config(obs_dim, action_dim, config, device)
    }

    /// Seeded variant of [`new`](Self::new): same 2-layer Kaiming
    /// architecture, but constructed deterministically from `seed` so
    /// two calls with the same seed produce bit-identical weights.
    ///
    /// Convenience wrapper for callers (and the PSRO/NFSP policy
    /// factories) that want reproducible policies without assembling a
    /// full [`MlpBurnConfig`]. Equivalent to
    /// `with_config(.., MlpBurnConfig { use_orthogonal_init: false,
    /// .., seed: Some(seed) }, ..)`.
    pub fn new_seeded(
        obs_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        seed: u64,
        device: &B::Device,
    ) -> Self {
        let config = MlpBurnConfig {
            num_layers: 2,
            hidden_dim,
            use_orthogonal_init: false,
            activation: BurnActivation::Tanh,
            seed: Some(seed),
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
        // Seeded path (issue #135): when `config.seed` is set, build
        // every layer from a deterministic `StdRng`-driven weight buffer
        // so two constructions with the same seed are bit-identical.
        // Each layer gets a distinct derived seed so layers of the same
        // shape don't collide. Layer indices are fixed:
        // 0=fc1, 1=fc2, 2=fc3, 3=policy_head, 4=value_head.
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
            let policy_head = mk(3, config.hidden_dim, action_dim, true);
            let value_head = mk(4, config.hidden_dim, 1, true);
            return Self { fc1, fc2, fc3, policy_head, value_head, activation: config.activation };
        }

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
        // Decoupled into a pure-tensor forward (`forward_to_host_dist`,
        // no RNG) followed by a host-side categorical draw
        // (`sample_actions_from_host_dist`, the only RNG-consuming half).
        // This split is what makes the **batched** entry point
        // (`get_actions_host_seeded_batched`) possible without changing
        // the per-row RNG draw order: a single forward over `[N, obs_dim]`
        // produces N rows of host probs, then the per-row loop draws RNG in
        // the exact same row-major sequence as N separate `[1, obs_dim]`
        // calls would. Bit-exactness (issue #114 / #235) is therefore
        // preserved by construction.
        let dist = self.forward_to_host_dist(obs);
        dist.sample_actions(rng)
    }

    /// Tensor half of [`get_action_host_seeded`]: run the policy forward
    /// and pull the host-side categorical distribution (`probs`,
    /// `log_probs`) plus values for every row. **Consumes no RNG.**
    ///
    /// Returns a [`HostCategoricalDist`] from which
    /// [`HostCategoricalDist::sample_actions`] performs the seeded draw.
    /// Splitting the forward (one batched tensor op over `[N, obs_dim]`)
    /// from the sample (a row-major host loop) lets the batched sampler
    /// replace N batch-1 forwards with a single `[N, obs_dim]` forward
    /// while keeping the RNG draw order — and therefore the sampled
    /// action stream — bit-identical (issue #235).
    fn forward_to_host_dist(&self, obs: Tensor<B, 2>) -> HostCategoricalDist {
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

        HostCategoricalDist { batch, n_actions, probs_flat, log_probs_flat, values_host }
    }

    /// Batched seeded sampler: one forward over `[N, obs_dim]`, then N
    /// host-side categorical draws in row-major order.
    ///
    /// Bit-identical to calling [`get_action_host_seeded`] once per row on
    /// `[1, obs_dim]` slices **provided the rows are drawn from the same
    /// policy** — the forward is a single matmul over all N rows (same
    /// weights), and the RNG is consumed one draw per row, ascending. This
    /// is the [`crate::multi_agent::joint::JointPolicy`]-trait batched
    /// entry point that eliminates per-call batch-1 overhead on the
    /// NdArray backend wherever many observations are scored through the
    /// **same** model in one step (issue #235).
    pub fn get_actions_host_seeded_batched(
        &self,
        obs: Tensor<B, 2>,
        rng: &mut rand::rngs::StdRng,
    ) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
        self.forward_to_host_dist(obs).sample_actions(rng)
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

    /// Flatten every weight + bias of a policy into one comparison
    /// vector (test helper for the bit-identity assertions below).
    fn collect_params(p: &MlpBurnPolicy<B>) -> Vec<f32> {
        let mut out = Vec::new();
        let mut push = |lin: &Linear<B>| {
            out.extend::<Vec<f32>>(lin.weight.val().into_data().to_vec().unwrap());
            if let Some(b) = &lin.bias {
                out.extend::<Vec<f32>>(b.val().into_data().to_vec().unwrap());
            }
        };
        push(&p.fc1);
        push(&p.fc2);
        if let Some(fc3) = &p.fc3 {
            push(fc3);
        }
        push(&p.policy_head);
        push(&p.value_head);
        out
    }

    /// Two seeded constructions (orthogonal init) with the same seed
    /// produce bit-identical weights; a different seed differs. This is
    /// the core guarantee behind end-to-end `PsroConfig::seed` /
    /// `NfspConfig::seed` reproducibility (issue #135).
    #[test]
    fn test_with_seed_is_bit_identical_orthogonal() {
        let device = Default::default();
        let cfg = MlpBurnConfig { num_layers: 3, ..Default::default() }.with_seed(42);
        let a = MlpBurnPolicy::<B>::with_config(4, 2, cfg, &device);
        let b = MlpBurnPolicy::<B>::with_config(4, 2, cfg, &device);
        assert_eq!(collect_params(&a), collect_params(&b), "same seed must be bit-identical");

        let cfg_diff = MlpBurnConfig { num_layers: 3, ..Default::default() }.with_seed(43);
        let c = MlpBurnPolicy::<B>::with_config(4, 2, cfg_diff, &device);
        assert_ne!(collect_params(&a), collect_params(&c), "different seed must differ");
    }

    /// Same as above but for the Kaiming-uniform path (`new_seeded`
    /// uses `use_orthogonal_init = false`). This is the path the
    /// matching-pennies tests exercise (issue #135, Correction 2).
    #[test]
    fn test_new_seeded_is_bit_identical_kaiming() {
        let device = Default::default();
        let a = MlpBurnPolicy::<B>::new_seeded(4, 2, 16, 7, &device);
        let b = MlpBurnPolicy::<B>::new_seeded(4, 2, 16, 7, &device);
        assert_eq!(collect_params(&a), collect_params(&b), "same seed must be bit-identical");
        let c = MlpBurnPolicy::<B>::new_seeded(4, 2, 16, 8, &device);
        assert_ne!(collect_params(&a), collect_params(&c), "different seed must differ");
    }

    /// Distinct layers within one policy must not share weights even
    /// when they have the same shape (the per-layer seed derivation
    /// must decorrelate them). fc2 and fc3 are both
    /// `[hidden, hidden]`; assert they differ.
    #[test]
    fn test_seeded_layers_are_decorrelated() {
        let device = Default::default();
        let cfg = MlpBurnConfig { num_layers: 3, hidden_dim: 8, ..Default::default() }.with_seed(1);
        let p = MlpBurnPolicy::<B>::with_config(8, 2, cfg, &device);
        let fc2: Vec<f32> = p.fc2.weight.val().into_data().to_vec().unwrap();
        let fc3: Vec<f32> = p.fc3.as_ref().unwrap().weight.val().into_data().to_vec().unwrap();
        assert_ne!(fc2, fc3, "same-shape trunk layers must get distinct seeded weights");
    }

    /// The unseeded path (`seed: None`) is unchanged — two
    /// constructions are *not* required to match (Burn's unseeded
    /// init), and the seeded path must not accidentally fire. We just
    /// assert construction succeeds and produces the right shapes, i.e.
    /// the `None` branch is still wired to Burn's `Initializer`.
    #[test]
    fn test_unseeded_path_still_constructs() {
        let device = Default::default();
        let cfg = MlpBurnConfig::default(); // seed: None
        assert!(cfg.seed.is_none());
        let p = MlpBurnPolicy::<B>::with_config(4, 2, cfg, &device);
        let obs = Tensor::<B, 2>::zeros([3, 4], &device);
        let (logits, values) = p.forward(obs);
        assert_eq!(logits.dims(), [3, 2]);
        assert_eq!(values.dims(), [3]);
    }
}
