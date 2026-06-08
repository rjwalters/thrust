//! Minimal MLP actor-critic policy implemented over the Burn 0.21 tensor
//! framework.
//!
//! # Scope
//!
//! This module is **scoped to the phase-1 Burn migration scout** (issue #78).
//! It is intentionally not feature-complete: it provides only the surface area
//! the bandit trainer needs (forward pass returning `(logits, value)`, action
//! sampling, action evaluation). It is **not** intended to be the production
//! design — that work lands in later phases of the migration epic (#65).
//!
//! Compared to [`crate::policy::mlp::MlpPolicy`] (tch path), the salient
//! differences are documented in `docs/BURN_MIGRATION_PHASE1_REPORT.md`.
//!
//! # Why generic over `B: Backend`?
//!
//! Burn's idiomatic pattern is to make every `Module` generic over a `Backend`
//! type parameter (CPU `NdArray`, GPU `Wgpu`/`Cuda`, autodiff-decorated
//! variants, etc.). This mirrors `geode-fem`'s pattern. The bandit trainer
//! picks `NdArray<f32>` wrapped in `Autodiff` at the top level.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Int, Tensor, activation, backend::Backend},
};

/// Two-layer MLP actor-critic for **discrete** action spaces, ported to Burn.
///
/// Layout matches [`crate::policy::mlp::MlpPolicy`] at a high level:
///
/// ```text
/// obs ──► fc1 ─tanh─► fc2 ─tanh─► policy_head (logits over n actions)
///                              └─► value_head  (scalar V(s))
/// ```
///
/// The two heads share the trunk activations, which is the standard PPO
/// actor-critic recipe. `hidden_dim` is the width of both hidden layers.
#[derive(Module, Debug)]
pub struct MlpBurnPolicy<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
}

impl<B: Backend> MlpBurnPolicy<B> {
    /// Build a fresh policy on `device` with the given dims.
    ///
    /// Burn's default `LinearConfig` initializes weights with a Kaiming-uniform
    /// scheme — adequate for the bandit smoke test; the orthogonal init the
    /// tch policy uses is a known follow-up (documented in the friction
    /// report).
    pub fn new(obs_dim: usize, action_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(obs_dim, hidden_dim).init(device),
            fc2: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            policy_head: LinearConfig::new(hidden_dim, action_dim).init(device),
            value_head: LinearConfig::new(hidden_dim, 1).init(device),
        }
    }

    /// Forward pass: returns `(logits, value)`.
    ///
    /// * `obs` is shape `[batch, obs_dim]`.
    /// * `logits` is shape `[batch, action_dim]` (pre-softmax).
    /// * `value` is shape `[batch]` (squeezed from `[batch, 1]`).
    pub fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let h = activation::tanh(self.fc1.forward(obs));
        let h = activation::tanh(self.fc2.forward(h));
        let logits = self.policy_head.forward(h.clone());
        let value = self.value_head.forward(h).squeeze_dim::<1>(1);
        (logits, value)
    }

    /// Sample one action per row from the policy's categorical distribution
    /// and return `(actions_host, log_probs_host, values_host)` as plain
    /// `Vec`s.
    ///
    /// The trainer-side rollout loop does not need gradient flow through the
    /// sampled action (only the eventual `evaluate_actions` call on the
    /// stored transitions matters for the PPO surrogate). We therefore do
    /// the categorical draw on the host with `rand`, which sidesteps Burn
    /// 0.21's lack of a first-class `multinomial` op — documented in the
    /// friction report as a real gap for any port that needs on-device
    /// sampling (Snake CNN, multi-agent self-play).
    pub fn get_action_host(&self, obs: Tensor<B, 2>) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
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

        let mut rng = rand::thread_rng();
        let mut actions = Vec::with_capacity(batch);
        let mut log_probs = Vec::with_capacity(batch);
        for row in 0..batch {
            let u: f32 = rng.r#gen();
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
    /// Returns `(action_log_probs, entropy_per_row, values)` — the quantities
    /// the PPO surrogate loss needs. Entropy is per-row here (not the mean):
    /// the caller decides how to aggregate.
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
