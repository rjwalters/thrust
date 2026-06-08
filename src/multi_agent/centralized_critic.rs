//! Centralized critic for MAPPO-style multi-agent training.
//!
//! # Motivation
//!
//! Independent value functions in multi-agent PPO observe only the local
//! agent's view (or, in the joint trainer's case, a shared but un-coordinated
//! per-agent value head). The advantage estimates each agent uses for its
//! policy update are therefore high-variance — every agent's value baseline is
//! noisy *and* uncorrelated with the team's actual return signal.
//!
//! A **centralized critic** is a single value function `V_c(s_joint)` that
//! conditions on the joint observation. During training it provides a
//! lower-variance value baseline for every agent's GAE / advantage
//! computation; at execution time each agent still acts independently on its
//! own observation. This is the **CT-DE** (centralized training, decentralized
//! execution) pattern formalized by MAPPO (Yu et al., 2021) and COMA
//! (Foerster et al., 2018).
//!
//! # When to use this versus alternatives
//!
//! - **Use the centralized critic** when (a) the agents share a global state
//!   you can compute, (b) you want lower-variance advantages without changing
//!   the actor's decentralized execution, and (c) you don't need explicit
//!   counterfactual reasoning (use COMA for that).
//! - **Use the joint trainer's `aux_fn` hook** for *representation*-level
//!   regularizers (e.g. cross-agent redundancy penalties) where you want the
//!   gradient to flow back into every agent's encoder. The aux hook is the
//!   wrong tool for value-loss insertion: it doesn't see advantages/returns,
//!   and its gradient flows through every encoder which is not what a
//!   centralized value loss wants.
//! - **Use independent per-agent value heads** (the joint trainer's default)
//!   when the policies are heterogeneous and there is no meaningful global
//!   state — i.e. when "joint observation" is not well-defined for the env.
//!
//! # v1 architectural choices
//!
//! - The centralized critic owns **its own** [`nn::VarStore`] and is paired
//!   with a caller-constructed [`nn::Optimizer`]. This mirrors how each
//!   per-agent policy in [`crate::multi_agent::joint::JointMultiAgentTrainer`]
//!   owns its var-store and optimizer, and keeps gradient flow strictly
//!   parameter-isolated: the critic's backward pass only updates critic
//!   parameters; per-agent backward passes only update per-agent parameters.
//! - Per-agent value heads are **kept** even when the centralized critic is
//!   active. `JointPolicy::get_action` still consults them at rollout time. The
//!   centralized critic provides an *additional* value-loss contribution; it
//!   does not replace the per-agent heads. Dropping the per-agent heads
//!   entirely is a possible future change.
//! - The forward network is a small MLP (Linear → Tanh → Linear → Tanh →
//!   Linear) returning a `[batch]` scalar value. The architecture mirrors the
//!   per-agent value-head style used elsewhere in `src/policy/mlp.rs`.

use tch::{Device, Kind, Tensor, nn};

/// Configuration for the centralized critic.
///
/// Defaults match the PPO baseline values used elsewhere in the codebase.
#[derive(Debug, Clone)]
pub struct CentralizedCriticConfig {
    /// Hidden-layer width for the critic MLP. `64` is the default; mirrors
    /// `train_p3.rs::Config::hidden_dim`.
    pub hidden_dim: i64,
    /// Adam learning rate used when the caller constructs the optimizer via
    /// [`CentralizedCritic::build_optimizer`]. `3e-4` is the PPO baseline.
    pub learning_rate: f64,
    /// Weight on the centralized-critic value loss inside the joint loss.
    /// `0.5` is the PPO baseline (same as `vf_coef`).
    pub vf_coef: f64,
    /// Value-function clip range. `0.0` disables clipping (loss falls back to
    /// plain MSE against the target). Matches
    /// `JointTrainerConfig::clip_range_vf`.
    pub clip_range_vf: f64,
}

impl Default for CentralizedCriticConfig {
    fn default() -> Self {
        Self { hidden_dim: 64, learning_rate: 3e-4, vf_coef: 0.5, clip_range_vf: 0.0 }
    }
}

/// Centralized critic: a value function `V_c(s_joint) -> R` over the joint
/// observation.
///
/// Owns its own [`nn::VarStore`]; gradients from the critic's value loss
/// flow only into the critic's own parameters.
pub struct CentralizedCritic {
    vs: nn::VarStore,
    fc1: nn::Linear,
    fc2: nn::Linear,
    head: nn::Linear,
    /// Cached device the critic's parameters live on. Set from `vs.device()`
    /// at construction and treated as an invariant for the critic's lifetime:
    /// nothing in this module calls `vs.set_device(...)` after construction,
    /// so `self.device == self.vs.device()` always holds. Stored here so that
    /// `device()` does not need to round-trip through `vs`.
    device: Device,
}

impl CentralizedCritic {
    /// Build a new centralized critic.
    ///
    /// The critic accepts `[batch, joint_obs_dim]` and returns `[batch]`
    /// scalar values. Parameters live on the caller-chosen device (CUDA if
    /// available, else CPU). Use [`Self::new_on_device`] to pin to a specific
    /// device (e.g. in tests that need CPU regardless of CUDA availability).
    pub fn new(joint_obs_dim: i64, hidden_dim: i64) -> Self {
        Self::new_on_device(joint_obs_dim, hidden_dim, Device::cuda_if_available())
    }

    /// Build a new centralized critic with explicit device (useful in tests
    /// that need to pin to CPU regardless of CUDA availability).
    pub fn new_on_device(joint_obs_dim: i64, hidden_dim: i64, device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        let lc = nn::LinearConfig::default();
        let fc1 = nn::linear(&root / "fc1", joint_obs_dim, hidden_dim, lc);
        let fc2 = nn::linear(&root / "fc2", hidden_dim, hidden_dim, lc);
        let head = nn::linear(&root / "head", hidden_dim, 1, lc);
        let device = vs.device();

        Self { vs, fc1, fc2, head, device }
    }

    /// Device the critic lives on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Immutable view of the critic's var-store.
    pub fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }

    /// Mutable view of the critic's var-store. Used by callers to construct
    /// an [`nn::Optimizer`] (see [`CentralizedCritic::build_optimizer`]).
    pub fn var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    /// Convenience: build an Adam optimizer over the critic's parameters.
    ///
    /// Equivalent to `tch::nn::Adam::default().build(critic.var_store_mut(),
    /// lr)`.
    pub fn build_optimizer(&mut self, lr: f64) -> anyhow::Result<nn::Optimizer> {
        use tch::nn::OptimizerConfig;
        let opt = nn::Adam::default()
            .build(self.var_store_mut(), lr)
            .map_err(|e| anyhow::anyhow!("CentralizedCritic::build_optimizer: {}", e))?;
        Ok(opt)
    }

    /// Forward pass.
    ///
    /// `joint_obs`: `[batch, joint_obs_dim]`. Returns `[batch]` (the
    /// trailing singleton dim from the head is squeezed).
    pub fn forward(&self, joint_obs: &Tensor) -> Tensor {
        let h = joint_obs.apply(&self.fc1).tanh();
        let h = h.apply(&self.fc2).tanh();
        h.apply(&self.head).squeeze_dim(-1)
    }
}

/// Centralized-critic value loss.
///
/// Identical shape to [`crate::train::ppo::compute_value_loss`] so that the
/// centralized critic's loss term is interchangeable with the per-agent value
/// loss when assembling the joint loss. Returns `(loss, explained_variance)`.
///
/// `clip_range_vf <= 0.0` (or non-finite) disables clipping and reduces to
/// plain MSE.
pub fn compute_centralized_value_loss(
    values: &Tensor,
    old_values: &Tensor,
    returns: &Tensor,
    clip_range_vf: f64,
) -> (Tensor, f64) {
    let value_loss = if clip_range_vf > 0.0 && clip_range_vf.is_finite() {
        let values_clipped =
            old_values + (values - old_values).clamp(-clip_range_vf, clip_range_vf);
        let vf_loss_1 = (values - returns).square();
        let vf_loss_2 = (values_clipped - returns).square();
        // Pessimistic clip: take the worse (larger) per-sample loss.
        vf_loss_1.maximum(&vf_loss_2).mean(Kind::Float)
    } else {
        (values - returns).square().mean(Kind::Float)
    };

    // Explained variance: 1 - Var(returns - values) / Var(returns).
    let var_returns = returns.var(false);
    let var_returns_val: f64 = f64::try_from(&var_returns).unwrap_or(0.0);
    let explained_var = if var_returns_val == 0.0 {
        1.0
    } else {
        let residual_var = (returns - values).var(false);
        let residual_var_val: f64 = f64::try_from(&residual_var).unwrap_or(0.0);
        1.0 - residual_var_val / var_returns_val
    };

    (value_loss, explained_var)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forward pass shape: `[B, joint_obs_dim] -> [B]`.
    #[test]
    fn test_centralized_critic_forward_shape() {
        let joint_obs_dim = 8;
        let hidden_dim = 16;
        let critic = CentralizedCritic::new_on_device(joint_obs_dim, hidden_dim, Device::Cpu);

        let batch = 4_i64;
        let obs = Tensor::randn([batch, joint_obs_dim], (Kind::Float, Device::Cpu));
        let values = critic.forward(&obs);
        assert_eq!(values.size(), vec![batch], "centralized critic output should be [batch]");
    }

    /// Hand-computable value loss case (no clipping): predicted = [1.0, 2.0],
    /// returns = [1.5, 1.5]. MSE = mean((1.0-1.5)^2, (2.0-1.5)^2)
    ///                          = mean(0.25, 0.25)
    ///                          = 0.25.
    #[test]
    fn test_compute_centralized_value_loss_unclipped() {
        let predicted = Tensor::from_slice(&[1.0_f32, 2.0]).to_device(Device::Cpu);
        let old = Tensor::from_slice(&[1.0_f32, 2.0]).to_device(Device::Cpu);
        let returns = Tensor::from_slice(&[1.5_f32, 1.5]).to_device(Device::Cpu);

        let (loss, _ev) = compute_centralized_value_loss(&predicted, &old, &returns, 0.0);
        let loss_val: f64 = f64::try_from(&loss).unwrap();
        assert!((loss_val - 0.25).abs() < 1e-5, "expected MSE = 0.25, got {loss_val}");
    }

    /// Random-input sanity: the returned loss is finite and explained_var is
    /// in `(-inf, 1]`.
    #[test]
    fn test_compute_centralized_value_loss_finite() {
        let predicted = Tensor::randn([16], (Kind::Float, Device::Cpu));
        let old = predicted.detach().copy();
        let returns = Tensor::randn([16], (Kind::Float, Device::Cpu));

        let (loss, ev) = compute_centralized_value_loss(&predicted, &old, &returns, 0.2);
        let loss_val: f64 = f64::try_from(&loss).unwrap();
        assert!(loss_val.is_finite(), "loss must be finite");
        assert!(loss_val >= 0.0, "MSE-style loss must be non-negative");
        assert!(ev.is_finite() && ev <= 1.0, "explained_var in (-inf, 1], got {ev}");
    }

    /// Loss-decrease smoke: optimize the centralized critic over a fixed
    /// target for ~100 steps and assert the loss strictly decreases.
    /// Guards against accidentally breaking the optimizer-wiring pattern.
    #[test]
    fn test_centralized_critic_loss_decreases() {
        let joint_obs_dim = 6;
        let hidden_dim = 8;
        let mut critic = CentralizedCritic::new_on_device(joint_obs_dim, hidden_dim, Device::Cpu);
        let mut opt = critic.build_optimizer(1e-2).unwrap();

        let obs = Tensor::randn([32, joint_obs_dim], (Kind::Float, Device::Cpu));
        let target = Tensor::randn([32], (Kind::Float, Device::Cpu));

        let initial_loss: f64 = {
            let pred = critic.forward(&obs);
            let (loss, _) = compute_centralized_value_loss(&pred, &pred.detach(), &target, 0.0);
            f64::try_from(&loss).unwrap()
        };

        for _ in 0..100 {
            let pred = critic.forward(&obs);
            let old_v = pred.detach();
            let (loss, _) = compute_centralized_value_loss(&pred, &old_v, &target, 0.0);
            opt.zero_grad();
            loss.backward();
            opt.step();
        }

        let final_loss: f64 = {
            let pred = critic.forward(&obs);
            let (loss, _) = compute_centralized_value_loss(&pred, &pred.detach(), &target, 0.0);
            f64::try_from(&loss).unwrap()
        };

        assert!(
            final_loss < initial_loss,
            "centralized critic loss did not decrease: initial={initial_loss}, final={final_loss}"
        );
    }
}
