//! Burn-backend Nature-DQN-scale CNN policy for the Atari (ALE) workload
//! (Epic #306, Phase 3 — issue #327).
//!
//! Implements the classic Nature-DQN convolutional stack (Mnih et al.,
//! *Human-level control through deep reinforcement learning*, 2015) as two
//! Burn modules that share the same conv trunk:
//!
//! - [`crate::policy::atari_cnn::NatureDqnBurnPolicy`] — actor-critic variant
//!   (policy + value heads), consumed by
//!   [`crate::train::ppo::trainer::PPOTrainerBurn`].
//! - [`crate::policy::atari_cnn::NatureDqnQNetwork`] — single-Q-head variant,
//!   consumed by [`crate::train::dqn::DQNTrainerBurn`] (with a
//!   `copy_params_from` target-net sync, mirroring
//!   [`crate::policy::q_network::QNetworkBurn`]).
//!
//! # Architecture
//!
//! ```text
//! obs [B, 4, 84, 84]
//!   → conv1 (32 ch, 8x8, stride 4)  → ReLU   → [B, 32, 20, 20]
//!   → conv2 (64 ch, 4x4, stride 2)  → ReLU   → [B, 64,  9,  9]
//!   → conv3 (64 ch, 3x3, stride 1)  → ReLU   → [B, 64,  7,  7]
//!   → flatten (64*7*7 = 3136)
//!   → fc_common (3136 -> 512)       → ReLU
//!   → heads:
//!       actor-critic: policy_head (512 -> A) [logits], value_head (512 -> 1)
//!       q-network:    q_head      (512 -> A) [Q(s, a)]
//! ```
//!
//! Convolutions use Burn's default `PaddingConfig2d::Valid` (no padding),
//! matching the Nature-DQN spec; the spatial reductions are therefore
//! `84 → 20 → 9 → 7`, giving a cached `flat_size` of `64 * 7 * 7 = 3136`.
//!
//! # Input contract
//!
//! - Layout: **NCHW** `[batch, channels, height, width]` — same convention as
//!   [`crate::policy::snake_cnn::SnakeCnnBurnPolicy`] and Burn's `Conv2d`.
//! - Channels: 4 (frame-stack dimension, produced by the preprocessor — not
//!   this module).
//! - Spatial: 84 × 84.
//! - Dtype: `f32`, pixel-scaled to **0.0–1.0** (uint8 ÷ 255). The network is
//!   scale-agnostic, but this is the expected convention.
//! - No batch-size constraint.
//!
//! # Trainer integration (closure-based, flat rollout buffers)
//!
//! Both Burn trainers are closure-based, not trait-based; the only module
//! bounds are `AutodiffModule<B> + Clone`, satisfied automatically by
//! `#[derive(Module, Debug)]`. The rollout buffers hand the closure a
//! **flat** observation tensor `[B, C*H*W]`, so the closure must reshape to
//! `[B, C, H, W]` before calling `forward`/`evaluate_actions` — the same
//! pattern used by `examples/games/snake/train_snake_multi_v2.rs`
//! (lines 239–253):
//!
//! ```ignore
//! // PPO (actor-critic):
//! let evaluate_fn = |p: &NatureDqnBurnPolicy<B>, o_flat: Tensor<B, 2>, acts: Tensor<B, 1, Int>| {
//!     let b = o_flat.dims()[0];
//!     let o4 = o_flat.reshape([b, 4, 84, 84]);
//!     p.evaluate_actions(o4, acts) // (log_probs [B], entropy [B], values [B])
//! };
//!
//! // DQN (Q-network):
//! let forward_fn = |q: &NatureDqnQNetwork<B>, o_flat: Tensor<B, 2>| {
//!     let b = o_flat.dims()[0];
//!     q.forward(o_flat.reshape([b, 4, 84, 84])) // Q-values [B, A]
//! };
//! ```
//!
//! # Seeded initialization
//!
//! Seeded construction (see [`crate::policy::atari_cnn::NatureDqnConfig`])
//! drives the three FC
//! layers from deterministically-derived host-side RNG streams via the
//! shared `mlp.rs` helpers (`derive_layer_seed` / `seeded_layer_weights`
//! / `linear_from_weights`), so two constructions with the same seed
//! produce **bit-identical** FC weights. Fixed per-variant layer indices:
//!
//! - `NatureDqnBurnPolicy`: `0 = fc_common`, `1 = policy_head`, `2 =
//!   value_head`
//! - `NatureDqnQNetwork`: `0 = fc_common`, `1 = q_head`
//!
//! **Conv layers are intentionally unseeded.** Burn's `Conv2dConfig` — like
//! `LinearConfig` — exposes no seed parameter, so the seeded path cannot
//! reach the convolutions. This is a deliberate, second-order concern: the
//! conv parameters total ~78K versus ~1.6M for `fc_common` alone, so the FC
//! layers dominate reproducibility. The seeded path therefore covers only
//! the three FC layers; the unseeded (`seed: None`) path routes every layer
//! through Burn's stock `Initializer`.

use burn::{
    module::Module,
    nn::{
        Initializer, Linear,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Int, Tensor, activation, backend::Backend},
};

use super::mlp::{derive_layer_seed, linear_from_weights, linear_with_init, seeded_layer_weights};

/// Number of input channels (frame-stack depth) the Nature-DQN policies
/// expect. Fixed by the Atari preprocessor convention.
const INPUT_CHANNELS: usize = 4;

/// Flattened post-conv feature width: `64 * 7 * 7`. Cached on each module as
/// a plain field so it survives `Module::load_record` (same trick as
/// [`crate::policy::snake_cnn::SnakeCnnBurnPolicy`]).
const FLAT_SIZE: usize = 64 * 7 * 7; // 3136

/// Hidden width of the shared `fc_common` layer.
const FC_HIDDEN: usize = 512;

/// Configuration for the Nature-DQN policies.
///
/// Deliberately minimal (only a seed) — the conv/FC topology is fixed by the
/// Nature-DQN spec, so unlike [`crate::policy::mlp::MlpBurnConfig`] there is
/// nothing else to tune. Mirrors the `seed` reproducibility hook on
/// [`crate::policy::q_network::QNetworkBurnConfig`].
#[derive(Debug, Clone, Copy, Default)]
pub struct NatureDqnConfig {
    /// Optional construction seed. When `Some`, the three FC layers are
    /// built from deterministically-derived host-side RNG streams (see
    /// [`crate::policy::seeded_init`]) so two constructions with the same
    /// seed produce **bit-identical** FC weights. When `None` (the default)
    /// Burn's unseedable [`Initializer`] path is used verbatim. Conv layers
    /// are unseeded in either case (see the module-level docs).
    pub seed: Option<u64>,
}

impl NatureDqnConfig {
    /// Set the construction seed, enabling the deterministic host-side FC
    /// init path in `with_config`.
    ///
    /// ```
    /// # use thrust_rl::policy::atari_cnn::NatureDqnConfig;
    /// let cfg = NatureDqnConfig::default().with_seed(42);
    /// assert_eq!(cfg.seed, Some(42));
    /// ```
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Build the three shared conv layers (unseeded — Burn `Conv2dConfig` has no
/// seed parameter). Kernels/strides fixed by the Nature-DQN spec; default
/// `PaddingConfig2d::Valid` (no padding).
fn build_convs<B: Backend>(device: &B::Device) -> (Conv2d<B>, Conv2d<B>, Conv2d<B>) {
    let conv1 = Conv2dConfig::new([INPUT_CHANNELS, 32], [8, 8]).with_stride([4, 4]).init(device);
    let conv2 = Conv2dConfig::new([32, 64], [4, 4]).with_stride([2, 2]).init(device);
    let conv3 = Conv2dConfig::new([64, 64], [3, 3]).with_stride([1, 1]).init(device);
    (conv1, conv2, conv3)
}

/// Default Kaiming-uniform initializer for the unseeded FC path — matches
/// Burn's stock `LinearConfig::default()` weight init (and the convention
/// used by [`crate::policy::snake_cnn::SnakeCnnBurnPolicy`]).
fn default_fc_init() -> Initializer {
    Initializer::KaimingUniform { gain: 1.0_f64 / 3.0_f64.sqrt(), fan_out_only: false }
}

/// Run the shared conv trunk and flatten to `[B, FLAT_SIZE]`.
fn conv_features<B: Backend>(
    conv1: &Conv2d<B>,
    conv2: &Conv2d<B>,
    conv3: &Conv2d<B>,
    flat_size: usize,
    obs: Tensor<B, 4>,
) -> Tensor<B, 2> {
    let x = activation::relu(conv1.forward(obs));
    let x = activation::relu(conv2.forward(x));
    let x = activation::relu(conv3.forward(x));
    let batch = x.dims()[0];
    x.reshape([batch, flat_size])
}

/// Nature-DQN-scale actor-critic CNN policy on Burn.
///
/// Conv trunk (`conv1/conv2/conv3`) → `fc_common` → `{policy_head,
/// value_head}`. See the [module docs](self) for the full architecture and I/O
/// contract.
#[derive(Module, Debug)]
pub struct NatureDqnBurnPolicy<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    fc_common: Linear<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
    /// Cached `64 * 7 * 7 = 3136`. Stored as a plain field so it lands in the
    /// `Record` and survives `Module::load_record`.
    flat_size: usize,
}

impl<B: Backend> NatureDqnBurnPolicy<B> {
    /// Construct a fresh actor-critic policy with unseeded (Burn-default)
    /// weight initialization.
    ///
    /// * `n_actions` — size of the discrete action space (policy-head width).
    /// * `device`    — Burn backend device.
    pub fn new(n_actions: usize, device: &B::Device) -> Self {
        Self::with_config(n_actions, NatureDqnConfig::default(), device)
    }

    /// Construct a fresh actor-critic policy with the given configuration.
    ///
    /// When `config.seed` is `Some`, the three FC layers are built from
    /// deterministically-derived host-side RNG streams (bit-exact across
    /// constructions with the same seed). Conv layers are always unseeded.
    pub fn with_config(n_actions: usize, config: NatureDqnConfig, device: &B::Device) -> Self {
        let (conv1, conv2, conv3) = build_convs::<B>(device);

        let (fc_common, policy_head, value_head) = if let Some(base_seed) = config.seed {
            // Seeded host-side FC init. Fixed layer indices:
            //   0 = fc_common, 1 = policy_head, 2 = value_head.
            let mut layer_idx = 0u64;
            let mut next = || {
                let s = derive_layer_seed(base_seed, layer_idx);
                layer_idx += 1;
                s
            };

            let wc = seeded_layer_weights(next(), FLAT_SIZE, FC_HIDDEN, false, false);
            let fc_common = linear_from_weights::<B>(FLAT_SIZE, FC_HIDDEN, &wc, device);

            let wp = seeded_layer_weights(next(), FC_HIDDEN, n_actions, false, true);
            let policy_head = linear_from_weights::<B>(FC_HIDDEN, n_actions, &wp, device);

            let wv = seeded_layer_weights(next(), FC_HIDDEN, 1, false, true);
            let value_head = linear_from_weights::<B>(FC_HIDDEN, 1, &wv, device);

            (fc_common, policy_head, value_head)
        } else {
            let init = default_fc_init();
            let fc_common = linear_with_init::<B>(FLAT_SIZE, FC_HIDDEN, init.clone(), device);
            let policy_head = linear_with_init::<B>(FC_HIDDEN, n_actions, init.clone(), device);
            let value_head = linear_with_init::<B>(FC_HIDDEN, 1, init, device);
            (fc_common, policy_head, value_head)
        };

        Self { conv1, conv2, conv3, fc_common, policy_head, value_head, flat_size: FLAT_SIZE }
    }

    /// Forward pass.
    ///
    /// * `obs` shape `[batch, 4, 84, 84]` (NCHW, pixels in `0.0..=1.0`).
    /// * Returns `(action_logits [batch, n_actions], values [batch, 1])` —
    ///   value retains the rank-2 layout used by
    ///   [`crate::policy::snake_cnn::SnakeCnnBurnPolicy`], so the trainer
    ///   closure squeezes it itself.
    pub fn forward(&self, obs: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let flat = conv_features(&self.conv1, &self.conv2, &self.conv3, self.flat_size, obs);
        let features = activation::relu(self.fc_common.forward(flat));
        let logits = self.policy_head.forward(features.clone());
        let values = self.value_head.forward(features);
        (logits, values)
    }

    /// PPO-facing evaluation, mirroring
    /// [`crate::policy::mlp::MlpBurnPolicy::evaluate_actions`].
    ///
    /// * `obs` shape `[batch, 4, 84, 84]`; `actions` shape `[batch]`.
    /// * Returns `(action_log_probs [batch], entropy [batch], values [batch])`
    ///   — all rank-1. The value head's rank-2 `[batch, 1]` output is squeezed
    ///   here.
    pub fn evaluate_actions(
        &self,
        obs: Tensor<B, 4>,
        actions: Tensor<B, 1, Int>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let (logits, values) = self.forward(obs);
        let log_probs = activation::log_softmax(logits, 1);
        let probs = log_probs.clone().exp();

        let action_log_probs =
            log_probs.clone().gather(1, actions.unsqueeze_dim::<2>(1)).squeeze_dim::<1>(1);
        // H = -Σ p * log p over the action axis.
        let entropy = -(probs * log_probs).sum_dim(1).squeeze_dim::<1>(1);
        let values = values.squeeze_dim::<1>(1);

        (action_log_probs, entropy, values)
    }
}

/// Nature-DQN-scale Q-network CNN on Burn.
///
/// Same conv trunk as [`NatureDqnBurnPolicy`], but with a single `q_head`
/// whose outputs are interpreted directly as `Q(s, a)` (no softmax). Includes
/// a record-based `copy_params_from` for target-net sync, mirroring
/// [`crate::policy::q_network::QNetworkBurn`].
#[derive(Module, Debug)]
pub struct NatureDqnQNetwork<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    fc_common: Linear<B>,
    q_head: Linear<B>,
    /// Cached `64 * 7 * 7 = 3136`; see `NatureDqnBurnPolicy`'s `flat_size`.
    flat_size: usize,
}

impl<B: Backend> NatureDqnQNetwork<B> {
    /// Construct a fresh Q-network with unseeded (Burn-default) init.
    pub fn new(n_actions: usize, device: &B::Device) -> Self {
        Self::with_config(n_actions, NatureDqnConfig::default(), device)
    }

    /// Construct a fresh Q-network with the given configuration.
    ///
    /// When `config.seed` is `Some`, the two FC layers (`0 = fc_common`,
    /// `1 = q_head`) are built from deterministically-derived host-side RNG
    /// streams. Conv layers are always unseeded.
    pub fn with_config(n_actions: usize, config: NatureDqnConfig, device: &B::Device) -> Self {
        let (conv1, conv2, conv3) = build_convs::<B>(device);

        let (fc_common, q_head) = if let Some(base_seed) = config.seed {
            // Seeded host-side FC init. Fixed layer indices:
            //   0 = fc_common, 1 = q_head.
            let mut layer_idx = 0u64;
            let mut next = || {
                let s = derive_layer_seed(base_seed, layer_idx);
                layer_idx += 1;
                s
            };

            let wc = seeded_layer_weights(next(), FLAT_SIZE, FC_HIDDEN, false, false);
            let fc_common = linear_from_weights::<B>(FLAT_SIZE, FC_HIDDEN, &wc, device);

            let wq = seeded_layer_weights(next(), FC_HIDDEN, n_actions, false, true);
            let q_head = linear_from_weights::<B>(FC_HIDDEN, n_actions, &wq, device);

            (fc_common, q_head)
        } else {
            let init = default_fc_init();
            let fc_common = linear_with_init::<B>(FLAT_SIZE, FC_HIDDEN, init.clone(), device);
            let q_head = linear_with_init::<B>(FC_HIDDEN, n_actions, init, device);
            (fc_common, q_head)
        };

        Self { conv1, conv2, conv3, fc_common, q_head, flat_size: FLAT_SIZE }
    }

    /// Forward pass: compute `Q(s, a)` for every action `a`.
    ///
    /// * `obs` shape `[batch, 4, 84, 84]` (NCHW, pixels in `0.0..=1.0`).
    /// * Returns Q-values of shape `[batch, n_actions]`.
    pub fn forward(&self, obs: Tensor<B, 4>) -> Tensor<B, 2> {
        let flat = conv_features(&self.conv1, &self.conv2, &self.conv3, self.flat_size, obs);
        let features = activation::relu(self.fc_common.forward(flat));
        self.q_head.forward(features)
    }

    /// Replace this network's parameters with a deep copy of `source`'s
    /// parameters (target-net sync). Returns a new module, mirroring
    /// [`crate::policy::q_network::QNetworkBurn::copy_params_from`].
    pub fn copy_params_from(self, source: &NatureDqnQNetwork<B>) -> NatureDqnQNetwork<B> {
        self.load_record(source.clone().into_record())
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, NdArray},
        module::Module,
    };

    use super::*;

    type B = Autodiff<NdArray<f32>>;

    /// Sum the element counts of every `Linear`/`Conv2d` weight and bias in a
    /// module, using Burn's `num_params` (which counts exactly the learnable
    /// parameter tensors). `flat_size` is a plain `usize` field, not a
    /// `Param`, so it is correctly excluded.
    fn count_params<M: Module<B>>(module: &M) -> usize {
        module.num_params()
    }

    #[test]
    fn test_nature_dqn_ac_forward_single() {
        let device = Default::default();
        let policy = NatureDqnBurnPolicy::<B>::new(4, &device);
        let obs = Tensor::<B, 4>::zeros([1, 4, 84, 84], &device);
        let (logits, values) = policy.forward(obs);
        assert_eq!(logits.dims(), [1, 4]);
        assert_eq!(values.dims(), [1, 1]);
    }

    #[test]
    fn test_nature_dqn_ac_forward_batch() {
        let device = Default::default();
        let policy = NatureDqnBurnPolicy::<B>::new(4, &device);
        let obs = Tensor::<B, 4>::zeros([32, 4, 84, 84], &device);
        let (logits, values) = policy.forward(obs);
        assert_eq!(logits.dims(), [32, 4]);
        assert_eq!(values.dims(), [32, 1]);
    }

    #[test]
    fn test_nature_dqn_q_forward() {
        let device = Default::default();
        let q_net = NatureDqnQNetwork::<B>::new(4, &device);
        let obs = Tensor::<B, 4>::zeros([1, 4, 84, 84], &device);
        let q_values = q_net.forward(obs);
        assert_eq!(q_values.dims(), [1, 4]);
    }

    #[test]
    fn test_nature_dqn_evaluate_actions_shapes() {
        let device = Default::default();
        let policy = NatureDqnBurnPolicy::<B>::new(4, &device);
        let obs = Tensor::<B, 4>::zeros([8, 4, 84, 84], &device);
        let actions = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(vec![0i64, 1, 2, 3, 0, 1, 2, 3], [8]),
            &device,
        );
        let (log_probs, entropy, values) = policy.evaluate_actions(obs, actions);
        assert_eq!(log_probs.dims(), [8]);
        assert_eq!(entropy.dims(), [8]);
        assert_eq!(values.dims(), [8]);
    }

    /// Two seeded constructions with the same seed must yield bit-identical FC
    /// weights (`fc_common`, `policy_head`, `value_head`); a different seed
    /// must differ. Conv weights are unseeded and not compared.
    #[test]
    fn test_nature_dqn_seeded_fc_identical() {
        let device = Default::default();
        let cfg = NatureDqnConfig::default().with_seed(42);
        let a = NatureDqnBurnPolicy::<B>::with_config(4, cfg, &device);
        let b = NatureDqnBurnPolicy::<B>::with_config(4, cfg, &device);
        let c = NatureDqnBurnPolicy::<B>::with_config(
            4,
            NatureDqnConfig::default().with_seed(43),
            &device,
        );

        let fc_a: Vec<f32> = a.fc_common.weight.val().into_data().to_vec().unwrap();
        let fc_b: Vec<f32> = b.fc_common.weight.val().into_data().to_vec().unwrap();
        let fc_c: Vec<f32> = c.fc_common.weight.val().into_data().to_vec().unwrap();
        assert_eq!(fc_a, fc_b, "same seed must yield identical fc_common weights");
        assert!(fc_a != fc_c, "different seed must yield different fc_common weights");

        let ph_a: Vec<f32> = a.policy_head.weight.val().into_data().to_vec().unwrap();
        let ph_b: Vec<f32> = b.policy_head.weight.val().into_data().to_vec().unwrap();
        assert_eq!(ph_a, ph_b, "same seed must yield identical policy_head weights");

        let vh_a: Vec<f32> = a.value_head.weight.val().into_data().to_vec().unwrap();
        let vh_b: Vec<f32> = b.value_head.weight.val().into_data().to_vec().unwrap();
        assert_eq!(vh_a, vh_b, "same seed must yield identical value_head weights");
    }

    /// Distinct per-layer seeds (via `derive_layer_seed`) must decorrelate
    /// layers of different shape: `fc_common` and `policy_head` share no
    /// weight values by construction. (Shapes differ, so we compare the
    /// leading overlap.)
    #[test]
    fn test_nature_dqn_seeded_layers_decorrelated() {
        let device = Default::default();
        let cfg = NatureDqnConfig::default().with_seed(7);
        let policy = NatureDqnBurnPolicy::<B>::with_config(4, cfg, &device);

        let fc: Vec<f32> = policy.fc_common.weight.val().into_data().to_vec().unwrap();
        let ph: Vec<f32> = policy.policy_head.weight.val().into_data().to_vec().unwrap();
        let n = ph.len().min(fc.len());
        assert!(
            fc[..n].iter().zip(&ph[..n]).any(|(x, y)| (x - y).abs() > 1e-9),
            "fc_common and policy_head must not share weights within one seeded construction"
        );
    }

    /// After `copy_params_from`, the target's forward output must match the
    /// source's element-wise.
    #[test]
    fn test_nature_dqn_q_copy_params_from() {
        let device = Default::default();
        let source = NatureDqnQNetwork::<B>::with_config(
            4,
            NatureDqnConfig::default().with_seed(11),
            &device,
        );
        let target = NatureDqnQNetwork::<B>::with_config(
            4,
            NatureDqnConfig::default().with_seed(99),
            &device,
        );

        let obs = Tensor::<B, 4>::ones([2, 4, 84, 84], &device) * 0.5;

        let q_source_before: Vec<f32> = source.forward(obs.clone()).into_data().to_vec().unwrap();
        let q_target_before: Vec<f32> = target.forward(obs.clone()).into_data().to_vec().unwrap();
        assert!(
            q_source_before.iter().zip(&q_target_before).any(|(a, b)| (a - b).abs() > 1e-6),
            "expected fresh nets to disagree before copy"
        );

        let target_copied = target.copy_params_from(&source);
        let q_source_after: Vec<f32> = source.forward(obs.clone()).into_data().to_vec().unwrap();
        let q_target_after: Vec<f32> = target_copied.forward(obs).into_data().to_vec().unwrap();
        for (a, b) in q_source_after.iter().zip(&q_target_after) {
            assert!(
                (a - b).abs() < 1e-6,
                "Q output mismatch after copy_params_from: source={a} target={b}"
            );
        }
    }

    #[test]
    fn test_nature_dqn_param_count_ac() {
        let device = Default::default();
        let policy = NatureDqnBurnPolicy::<B>::new(4, &device);
        assert_eq!(count_params(&policy), 1_686_693);
    }

    #[test]
    fn test_nature_dqn_param_count_q() {
        let device = Default::default();
        let q_net = NatureDqnQNetwork::<B>::new(4, &device);
        assert_eq!(count_params(&q_net), 1_686_180);
    }
}
