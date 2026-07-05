//! Burn-backend LSTM (recurrent) actor-critic policy.
//!
//! Phase 1 of the recurrent-policy epic (#262). Implements
//! [`LstmBurnPolicy`](crate::policy::lstm::LstmBurnPolicy), a concrete
//! `#[derive(Module)]` struct that mirrors
//! [`crate::policy::mlp::MlpBurnPolicy`] but replaces the feedforward
//! trunk with a Burn 0.21 [`Lstm`](burn::nn::Lstm) so the policy can carry
//! memory across timesteps. As with the MLP policy there is **no** formal
//! `Policy` trait — each policy is a standalone Burn module.
//!
//! # Architecture
//!
//! ```text
//! obs [.., seq, obs_dim] → Lstm → [.., seq, hidden] ┬─ policy_head → logits
//!                                                    └─ value_head  → V(s)
//! ```
//!
//! # Entry points
//!
//! - [`LstmBurnPolicy::forward_step`](crate::policy::lstm::LstmBurnPolicy::forward_step) — single-step inference for rollout
//!   collection. Threads the recurrent state `(h, c)` in and out so the caller
//!   can carry it across environment steps.
//! - [`LstmBurnPolicy::evaluate_sequences`](crate::policy::lstm::LstmBurnPolicy::evaluate_sequences) — the rank-3 training forward. Runs
//!   the LSTM over an entire `[n_env, T, obs_dim]` batch of trajectories,
//!   resetting the hidden/cell state to zero at **episode boundaries**
//!   (`episode_starts = terminated || truncated`).
//!
//! # Seeded initialization
//!
//! Burn's [`LstmConfig`](burn::nn::LstmConfig) initializer takes no seed, so
//! two constructions draw different weights. To honor the reproducibility
//! contract shared with [`crate::policy::mlp::MlpBurnConfig::seed`], the seeded
//! path overrides all **eight** gate `Linear` layers (four
//! [`burn::nn::GateController`]s × `{input_transform, hidden_transform}`)
//! plus the two heads from deterministic, `StdRng`-driven weight buffers
//! via the existing `seeded_layer_weights` / `derive_layer_seed`
//! helpers. `GateController`'s fields are `pub`, so the swap is plain
//! post-`init` field assignment — no `unsafe`, no new seeded-init
//! primitive (see `docs/RECURRENT_POLICY_DESIGN.md`, Q4).

use burn::{
    module::Module,
    nn::{Initializer, Linear, Lstm, LstmConfig, LstmState},
    tensor::{Int, Tensor, activation, backend::Backend},
};

use crate::policy::mlp::{
    derive_layer_seed, linear_from_weights, linear_with_init, seeded_layer_weights,
};

/// Configuration for [`LstmBurnPolicy`].
///
/// Mirrors [`crate::policy::mlp::MlpBurnConfig`]'s knobs that carry over
/// to a recurrent trunk. The LSTM has no depth knob (a single recurrent
/// layer for v1) and no activation knob (the gate/cell/hidden
/// activations are fixed by the LSTM cell equations).
#[derive(Debug, Clone, Copy)]
pub struct LstmBurnConfig {
    /// Width of the LSTM hidden/cell state (and therefore the feature
    /// vector fed to both heads).
    pub hidden_dim: usize,
    /// If `true`, use the PPO orthogonal recipe (gain `sqrt(2)` on the
    /// gate weights, `0.01` on the output heads). If `false`, fall back
    /// to Kaiming-uniform (seeded path) or Burn's default `XavierNormal`
    /// (unseeded path). Only affects the *distribution* the weights are
    /// drawn from, never the reproducibility guarantee.
    pub use_orthogonal_init: bool,
    /// Optional construction seed. When `Some`, `with_config` builds
    /// every gate `Linear` and both heads from a deterministic,
    /// [`StdRng`](rand::rngs::StdRng)-driven weight buffer instead of
    /// Burn's unseedable [`Initializer`], so two constructions with the
    /// same seed produce **bit-identical** policies. When `None` the
    /// LSTM uses Burn's default `XavierNormal` gate init (unseeded).
    pub seed: Option<u64>,
}

impl Default for LstmBurnConfig {
    fn default() -> Self {
        Self { hidden_dim: 64, use_orthogonal_init: true, seed: None }
    }
}

impl LstmBurnConfig {
    /// Set the construction seed, enabling the deterministic host-side
    /// init path in [`LstmBurnPolicy::with_config`].
    ///
    /// Builder-style; returns `self` for chaining:
    /// `LstmBurnConfig::default().with_seed(42)`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Single-layer LSTM actor-critic for **discrete** action spaces.
///
/// Both heads share the recurrent trunk — standard recurrent PPO
/// actor-critic. Derives Burn [`Module`] (and, transitively, the
/// `AutodiffModule` impl the training path needs) so gradients flow back
/// through the LSTM gates.
#[derive(Module, Debug)]
pub struct LstmBurnPolicy<B: Backend> {
    lstm: Lstm<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
}

impl<B: Backend> LstmBurnPolicy<B> {
    /// Convenience constructor with the default configuration
    /// (`hidden_dim = 64`, orthogonal init, unseeded).
    pub fn new(obs_dim: usize, action_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let config = LstmBurnConfig { hidden_dim, ..Default::default() };
        Self::with_config(obs_dim, action_dim, config, device)
    }

    /// Build a fresh recurrent policy on `device` with the given
    /// configuration.
    ///
    /// When `config.seed` is `Some`, every gate `Linear` (eight total)
    /// and both heads are built from deterministic `StdRng`-driven weight
    /// buffers so two constructions with the same seed are bit-identical.
    /// Gate `Linear` layer indices are fixed so distinct-but-same-shape
    /// layers (e.g. every `hidden_transform`) draw decorrelated streams:
    ///
    /// ```text
    /// 0 input_gate.input     1 input_gate.hidden
    /// 2 forget_gate.input    3 forget_gate.hidden
    /// 4 cell_gate.input      5 cell_gate.hidden
    /// 6 output_gate.input    7 output_gate.hidden
    /// 8 policy_head          9 value_head
    /// ```
    pub fn with_config(
        obs_dim: usize,
        action_dim: usize,
        config: LstmBurnConfig,
        device: &B::Device,
    ) -> Self {
        let hidden = config.hidden_dim;
        // Build the LSTM with bias enabled. On the seeded path the gate
        // weights are overwritten below; on the unseeded path Burn's
        // default `XavierNormal` gate init is used verbatim (it handles
        // the rank-1 bias, unlike `Orthogonal`, which is why we don't
        // route orthogonal gate init through `LstmConfig`).
        let mut lstm = LstmConfig::new(obs_dim, hidden, true).init::<B>(device);

        if let Some(seed) = config.seed {
            let orth = config.use_orthogonal_init;
            // Overwrite all eight gate `Linear`s from seeded buffers.
            // `GateController`'s fields are `pub`, so this is plain
            // field assignment (see design note Q4). `is_head = false`
            // for gate layers.
            let mk = |idx: u64, d_in: usize, d_out: usize| -> Linear<B> {
                let s = derive_layer_seed(seed, idx);
                let w = seeded_layer_weights(s, d_in, d_out, orth, false);
                linear_from_weights::<B>(d_in, d_out, &w, device)
            };
            lstm.input_gate.input_transform = mk(0, obs_dim, hidden);
            lstm.input_gate.hidden_transform = mk(1, hidden, hidden);
            lstm.forget_gate.input_transform = mk(2, obs_dim, hidden);
            lstm.forget_gate.hidden_transform = mk(3, hidden, hidden);
            lstm.cell_gate.input_transform = mk(4, obs_dim, hidden);
            lstm.cell_gate.hidden_transform = mk(5, hidden, hidden);
            lstm.output_gate.input_transform = mk(6, obs_dim, hidden);
            lstm.output_gate.hidden_transform = mk(7, hidden, hidden);

            let mk_head = |idx: u64, d_in: usize, d_out: usize| -> Linear<B> {
                let s = derive_layer_seed(seed, idx);
                let w = seeded_layer_weights(s, d_in, d_out, orth, true);
                linear_from_weights::<B>(d_in, d_out, &w, device)
            };
            let policy_head = mk_head(8, hidden, action_dim);
            let value_head = mk_head(9, hidden, 1);
            return Self { lstm, policy_head, value_head };
        }

        // Unseeded path: Burn default gate init (already applied above),
        // orthogonal (or Kaiming) heads via the shared `linear_with_init`
        // helper, which zeroes the bias and thus tolerates `Orthogonal`.
        let head_init = if config.use_orthogonal_init {
            Initializer::Orthogonal { gain: 0.01 }
        } else {
            Initializer::KaimingUniform { gain: 1.0_f64 / 3.0_f64.sqrt(), fan_out_only: false }
        };
        let policy_head = linear_with_init::<B>(hidden, action_dim, head_init.clone(), device);
        let value_head = linear_with_init::<B>(hidden, 1, head_init, device);
        Self { lstm, policy_head, value_head }
    }

    /// Hidden/cell state width (`config.hidden_dim`).
    pub fn hidden_dim(&self) -> usize {
        self.lstm.d_hidden
    }

    /// Action-head output dimensionality (number of discrete actions).
    ///
    /// Reads the `policy_head` weight shape (`[hidden, action_dim]`), the
    /// same trick [`crate::policy::mlp::MlpBurnPolicy::policy_head_action_dim`]
    /// uses to size buffers without consuming RNG.
    pub fn policy_head_action_dim(&self) -> usize {
        self.policy_head.weight.val().dims()[1]
    }

    /// Single-step forward for rollout collection.
    ///
    /// * `obs` — one timestep, shape `[batch, obs_dim]`.
    /// * `state` — the incoming recurrent state; `None` starts from a zeroed
    ///   `(h, c)`.
    ///
    /// Returns `(logits, value, new_state)`:
    /// * `logits` — `[batch, action_dim]` (pre-softmax).
    /// * `value` — `[batch]` (squeezed from `[batch, 1]`).
    /// * `new_state` — [`LstmState`] with `cell`/`hidden` each `[batch,
    ///   hidden_dim]`, to thread into the next step.
    pub fn forward_step(
        &self,
        obs: Tensor<B, 2>,
        state: Option<LstmState<B, 2>>,
    ) -> (Tensor<B, 2>, Tensor<B, 1>, LstmState<B, 2>) {
        // Add a length-1 sequence axis: [batch, 1, obs_dim].
        let obs_seq = obs.unsqueeze_dim::<3>(1);
        let (out, new_state) = self.lstm.forward(obs_seq, state);
        // [batch, 1, hidden] → [batch, hidden].
        let feats = out.squeeze_dim::<2>(1);
        let logits = self.policy_head.forward(feats.clone());
        let value = self.value_head.forward(feats).squeeze_dim::<1>(1);
        (logits, value, new_state)
    }

    /// Rank-3 training forward over full trajectories.
    ///
    /// Runs the LSTM step-by-step across the `T` axis, resetting the
    /// hidden/cell state to zero at **episode boundaries** so stale memory
    /// never leaks across the boundary into a fresh episode.
    ///
    /// * `obs_seq` — `[n_env, T, obs_dim]`. One env-trajectory per row,
    ///   temporally intact (Strategy A, see design note Q2).
    /// * `actions` — `[n_env, T]` discrete actions actually taken, used to
    ///   gather their log-probabilities.
    /// * `initial_state` — recurrent state entering step 0; `None` starts from
    ///   a zeroed `(h, c)`.
    /// * `episode_starts` — `[n_env, T]` per-step episode-boundary flags (`1.0`
    ///   = reset, `0.0` = carry). These encode `terminated || truncated`,
    ///   **not** terminated alone. Thrust's rollout resets envs on either flag
    ///   and CartPole/MaskedCartPole truncate at `max_steps = 500`, so a
    ///   truncated step still ends the episode and its state must reset. This
    ///   is the deliberate asymmetry with GAE, which bootstraps on `terminated`
    ///   only (see `docs/RECURRENT_POLICY_DESIGN.md`, Q2 "GAE bootstrapping vs.
    ///   hidden-state reset"). At each step `t`, if `episode_starts[:, t]` is
    ///   `1.0` the incoming `(h, c)` is zeroed *before* the step is processed.
    ///
    /// Returns `(action_log_probs, entropy, values)`, each `[n_env, T]`:
    /// the per-step quantities the recurrent PPO surrogate needs. Entropy
    /// is per-step (not reduced); the trainer decides how to aggregate.
    pub fn evaluate_sequences(
        &self,
        obs_seq: Tensor<B, 3>,
        actions: Tensor<B, 2, Int>,
        initial_state: Option<LstmState<B, 2>>,
        episode_starts: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let device = obs_seq.device();
        let [n_env, seq_len, obs_dim] = obs_seq.dims();
        let hidden = self.hidden_dim();

        // Recurrent state carried across the T loop. `None` ⇒ zeros.
        let (mut cell, mut hidden_state) = match initial_state {
            Some(s) => (s.cell, s.hidden),
            None => (
                Tensor::<B, 2>::zeros([n_env, hidden], &device),
                Tensor::<B, 2>::zeros([n_env, hidden], &device),
            ),
        };

        let mut feats: Vec<Tensor<B, 3>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            // `keep = 1 - episode_starts[:, t]`, shape [n_env, 1]. Zeroing
            // the state per-env (rather than resetting the whole batch)
            // is what lets one minibatch hold envs at different episode
            // phases; the [n_env, 1] factor broadcasts over `hidden`.
            let flag = episode_starts.clone().slice([0..n_env, t..(t + 1)]);
            let keep = flag.neg().add_scalar(1.0);
            cell = cell.mul(keep.clone());
            hidden_state = hidden_state.mul(keep);

            // Single-step LSTM: [n_env, 1, obs_dim] → [n_env, 1, hidden].
            let step_in = obs_seq.clone().slice([0..n_env, t..(t + 1), 0..obs_dim]);
            let (out, new_state) =
                self.lstm.forward(step_in, Some(LstmState::new(cell, hidden_state)));
            feats.push(out);
            cell = new_state.cell;
            hidden_state = new_state.hidden;
        }

        // [n_env, T, hidden].
        let features = Tensor::cat(feats, 1);
        let logits = self.policy_head.forward(features.clone());
        // [n_env, T, 1] → [n_env, T].
        let values = self.value_head.forward(features).squeeze_dim::<2>(2);

        // Categorical log-probs over the action axis (dim 2).
        let log_probs_all = activation::log_softmax(logits, 2);
        let probs = log_probs_all.clone().exp();
        let action_log_probs = log_probs_all
            .clone()
            .gather(2, actions.unsqueeze_dim::<3>(2))
            .squeeze_dim::<2>(2);
        // H = -Σ p·log p over the action axis.
        let entropy = -(probs * log_probs_all).sum_dim(2).squeeze_dim::<2>(2);

        (action_log_probs, entropy, values)
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};

    use super::*;

    type B = Autodiff<NdArray<f32>>;

    /// Unseeded construction succeeds and a single-step forward produces
    /// output tensors of the correct rank/shape.
    #[test]
    fn test_constructs_unseeded() {
        let device = Default::default();
        let policy = LstmBurnPolicy::<B>::with_config(4, 2, LstmBurnConfig::default(), &device);
        assert_eq!(policy.hidden_dim(), 64);
        assert_eq!(policy.policy_head_action_dim(), 2);

        let obs = Tensor::<B, 2>::zeros([3, 4], &device);
        let (logits, value, _state) = policy.forward_step(obs, None);
        assert_eq!(logits.dims(), [3, 2]);
        assert_eq!(value.dims(), [3]);
    }

    /// `forward_step(obs, None)` returns logits `[1, action_dim]`, value
    /// `[1]`, and a new state whose `cell`/`hidden` are each
    /// `[1, hidden_dim]` for a single observation.
    #[test]
    fn test_forward_step_shapes() {
        let device = Default::default();
        let cfg = LstmBurnConfig { hidden_dim: 8, ..Default::default() };
        let policy = LstmBurnPolicy::<B>::with_config(4, 3, cfg, &device);

        let obs = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.1_f32, 0.2, 0.3, 0.4], [1, 4]),
            &device,
        );
        let (logits, value, state) = policy.forward_step(obs, None);
        assert_eq!(logits.dims(), [1, 3]);
        assert_eq!(value.dims(), [1]);
        assert_eq!(state.cell.dims(), [1, 8]);
        assert_eq!(state.hidden.dims(), [1, 8]);
    }

    /// `evaluate_sequences` returns the three per-step quantities with the
    /// expected `[n_env, T]` shape.
    #[test]
    fn test_evaluate_sequences_shapes() {
        let device = Default::default();
        let cfg = LstmBurnConfig { hidden_dim: 8, ..Default::default() };
        let policy = LstmBurnPolicy::<B>::with_config(4, 2, cfg, &device);

        let obs_seq = Tensor::<B, 3>::zeros([2, 3, 4], &device); // [n_env=2, T=3, obs=4]
        let actions = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(vec![0i64, 1, 0, 1, 0, 1], [2, 3]),
            &device,
        );
        let episode_starts = Tensor::<B, 2>::zeros([2, 3], &device);
        let (log_probs, entropy, values) =
            policy.evaluate_sequences(obs_seq, actions, None, episode_starts);
        assert_eq!(log_probs.dims(), [2, 3]);
        assert_eq!(entropy.dims(), [2, 3]);
        assert_eq!(values.dims(), [2, 3]);
    }

    /// Flatten every gate `Linear` (8) + both heads (2) weight and bias
    /// into one comparison vector (test helper for the bit-identity
    /// assertions below).
    fn collect_params(p: &LstmBurnPolicy<B>) -> Vec<f32> {
        let mut out = Vec::new();
        let mut push = |lin: &Linear<B>| {
            out.extend::<Vec<f32>>(lin.weight.val().into_data().to_vec().unwrap());
            if let Some(b) = &lin.bias {
                out.extend::<Vec<f32>>(b.val().into_data().to_vec().unwrap());
            }
        };
        push(&p.lstm.input_gate.input_transform);
        push(&p.lstm.input_gate.hidden_transform);
        push(&p.lstm.forget_gate.input_transform);
        push(&p.lstm.forget_gate.hidden_transform);
        push(&p.lstm.cell_gate.input_transform);
        push(&p.lstm.cell_gate.hidden_transform);
        push(&p.lstm.output_gate.input_transform);
        push(&p.lstm.output_gate.hidden_transform);
        push(&p.policy_head);
        push(&p.value_head);
        out
    }

    /// Two seeded constructions with the same seed produce bit-identical
    /// weights across all 8 gate `Linear`s and both heads; a different
    /// seed differs. Mirrors `test_with_seed_is_bit_identical_orthogonal`
    /// in `mlp.rs` — the reproducibility contract Phase 2/3 depend on.
    #[test]
    fn test_with_seed_is_bit_identical() {
        let device = Default::default();
        let cfg = LstmBurnConfig { hidden_dim: 8, ..Default::default() }.with_seed(42);
        let a = LstmBurnPolicy::<B>::with_config(4, 2, cfg, &device);
        let b = LstmBurnPolicy::<B>::with_config(4, 2, cfg, &device);
        assert_eq!(collect_params(&a), collect_params(&b), "same seed must be bit-identical");

        let cfg_diff = LstmBurnConfig { hidden_dim: 8, ..Default::default() }.with_seed(43);
        let c = LstmBurnPolicy::<B>::with_config(4, 2, cfg_diff, &device);
        assert_ne!(collect_params(&a), collect_params(&c), "different seed must differ");
    }

    /// Same-shape gate layers must draw decorrelated seeded streams: the
    /// four `hidden_transform`s (all `[hidden, hidden]`) must not be
    /// identical. Guards against a per-layer-seed derivation bug.
    #[test]
    fn test_seeded_gates_are_decorrelated() {
        let device = Default::default();
        let cfg = LstmBurnConfig { hidden_dim: 6, ..Default::default() }.with_seed(1);
        let p = LstmBurnPolicy::<B>::with_config(4, 2, cfg, &device);
        let i: Vec<f32> =
            p.lstm.input_gate.hidden_transform.weight.val().into_data().to_vec().unwrap();
        let f: Vec<f32> =
            p.lstm.forget_gate.hidden_transform.weight.val().into_data().to_vec().unwrap();
        assert_ne!(i, f, "same-shape gate layers must get distinct seeded weights");
    }

    /// Hidden state resets to zeros at an episode boundary
    /// (`episode_starts[:, 1] = 1`) and carries forward otherwise
    /// (`episode_starts[:, 1] = 0`).
    ///
    /// Construction:
    /// - A 2-step sequence with a nonzero step-0 observation (so step 0
    ///   produces a nonzero recurrent state).
    /// - When step 1 is flagged as an episode start, its output must equal a
    ///   fresh single-step forward from a zeroed state (`forward_step(o1,
    ///   None)`), proving the carried state was zeroed.
    /// - When step 1 is *not* flagged, its output must differ (the step-0 state
    ///   was carried in), proving reset is conditional.
    #[test]
    fn test_evaluate_sequences_episode_boundary_reset() {
        let device = Default::default();
        // Seeded so weights are fixed across the three forwards compared.
        let cfg = LstmBurnConfig { hidden_dim: 8, ..Default::default() }.with_seed(7);
        let policy = LstmBurnPolicy::<B>::with_config(4, 2, cfg, &device);

        // obs_seq [1, 2, 4]: distinct nonzero steps.
        let o0 = [0.9_f32, -0.4, 0.6, 0.2];
        let o1 = [0.1_f32, 0.5, -0.3, 0.8];
        let seq_data: Vec<f32> = o0.iter().chain(o1.iter()).copied().collect();
        let obs_seq =
            Tensor::<B, 3>::from_data(burn::tensor::TensorData::new(seq_data, [1, 2, 4]), &device);
        let actions = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(vec![0i64, 0], [1, 2]),
            &device,
        );

        // Reset at step 1: episode_starts = [0, 1].
        let starts_reset = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.0_f32, 1.0], [1, 2]),
            &device,
        );
        let (_, _, values_reset) =
            policy.evaluate_sequences(obs_seq.clone(), actions.clone(), None, starts_reset);
        let v_reset: Vec<f32> = values_reset.into_data().to_vec().unwrap();

        // Fresh single-step forward from zeroed state on o1.
        let o1_tensor =
            Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(o1.to_vec(), [1, 4]), &device);
        let (_, value_fresh, _) = policy.forward_step(o1_tensor, None);
        let v_fresh: Vec<f32> = value_fresh.into_data().to_vec().unwrap();

        // Step-1 value under reset must equal the fresh (zero-state) value.
        assert!(
            (v_reset[1] - v_fresh[0]).abs() < 1e-5,
            "reset step-1 value {} should match fresh zero-state value {}",
            v_reset[1],
            v_fresh[0]
        );

        // No reset at step 1: episode_starts = [0, 0]. The step-0 state is
        // carried in, so the step-1 value must differ from the reset case.
        let starts_carry = Tensor::<B, 2>::zeros([1, 2], &device);
        let (_, _, values_carry) = policy.evaluate_sequences(obs_seq, actions, None, starts_carry);
        let v_carry: Vec<f32> = values_carry.into_data().to_vec().unwrap();
        assert!(
            (v_carry[1] - v_reset[1]).abs() > 1e-6,
            "carried step-1 value {} should differ from reset value {} (stale state must flow)",
            v_carry[1],
            v_reset[1]
        );
    }
}
