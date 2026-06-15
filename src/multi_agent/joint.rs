//! Synchronized joint multi-agent PPO trainer (Burn backend).
//!
//! Burn-native rebuild of the pre-Burn `JointMultiAgentTrainer`. The previous
//! `tch`-coupled implementation was deleted in PR #98 along with the rest of
//! `src/multi_agent/`; this module re-establishes the *synchronized* (one
//! shared backward pass) joint trainer on top of the Burn policy networks
//! and the [`crate::train::optimizer::BurnOptimizer`] wrapper.
//!
//! # When to use this module
//!
//! Use the joint trainer when you need a loss term that depends on **all**
//! agents' parameters evaluated on the **same minibatch** at the **same**
//! optimization step. The canonical motivating example is the Slepian-Wolf
//! MARL P3 cross-agent representational redundancy penalty
//!
//! ```text
//! L_red   = λ * Σ_{i<j} || corr(Z_i(obs), Z_j(obs)) ||_F² / d²
//! L_total = Σ_i L_ppo[i](obs, a_i, ...) + L_red
//! ```
//!
//! where `Z_i = encoder_features_i(obs)`. The penalty couples every policy's
//! encoder through one shared backward pass; per-thread learners cannot
//! express this without heavy synchronization.
//!
//! # Single-graph, multiple-optimizer semantics under Burn
//!
//! Burn's [`burn::optim::Optimizer::step`] consumes the module by value and
//! returns the updated copy. That makes the tch-style "one `.backward()` plus
//! N independent `Optimizer::step` calls touching disjoint var-stores"
//! pattern slightly different in Burn:
//!
//! 1. Each policy computes its own `(policy_loss, value_loss, entropy)` on the
//!    minibatch.
//! 2. The caller-supplied `aux_fn` is invoked on every policy's encoder
//!    features and may return an additional scalar loss.
//! 3. All per-agent losses plus the aux loss are summed into one scalar
//!    `joint_loss`; we call `.backward()` once.
//! 4. For each policy `i`, we slice the joint gradients down to policy `i`'s
//!    parameters with [`burn::optim::GradientsParams::from_grads(grads.clone(),
//!    &policies[i])`], then call `optimizer_i.step(lr, policies[i], slice_i)`.
//!
//! That last step is the Burn analog of "every optimizer reads only its own
//! var-store's `.grad`" on the tch path — gradient flow stays
//! parameter-isolated because each `from_grads` slice only contains the
//! ids of one policy's params. The aux term's contribution flows into every
//! policy's slice through the shared backward pass, by construction.
//!
//! # Minibatch sampling
//!
//! For simplicity (and to keep the smoke test deterministic) this first
//! Burn-native cut takes **one** minibatch per epoch, sampled via
//! [`crate::train::ppo::loss::generate_minibatch_indices`] and truncated to
//! `config.minibatch_size`. Iterating *all* minibatches per epoch is the
//! more conventional PPO pattern and can be added without changing the
//! public API.

use anyhow::{Result, anyhow};
use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    tensor::{Int, Tensor, backend::AutodiffBackend},
};
use rand::rngs::StdRng;

use crate::train::{
    optimizer::{BackendOptimizer, BurnOptimizer},
    ppo::loss::{compute_policy_loss, compute_value_loss, scalar_f64},
};

// -----------------------------------------------------------------------
// Trait: what a policy must support to participate.
// -----------------------------------------------------------------------

/// Capabilities a policy must expose to participate in
/// [`JointMultiAgentTrainer`].
///
/// The trait pins exactly the surface the trainer needs:
///
/// - **`get_action_host`** — rollout-time sampling. Returns `(actions_per_dim,
///   log_probs, values)` on the host so the trainer can build the rollout
///   buffer without tying it to a particular backend tensor.
/// - **`evaluate_actions`** — re-evaluate the current policy on previously
///   sampled actions to compute updated log-probs / entropy / value for the PPO
///   loss; this is the only place autograd-bearing tensors are produced.
/// - **`encoder_features`** — shared-trunk activations for the auxiliary loss.
/// - **`action_dims`** — per-dim action cardinalities, used to size action
///   buffers without invoking the policy.
pub trait JointPolicy<B: AutodiffBackend>: AutodiffModule<B> + Clone {
    /// Sample actions for a single rollout step.
    ///
    /// `obs` carries one row per environment in the rollout batch. Returns
    /// host-side `(actions, log_probs, values)` where:
    ///
    /// - `actions` is laid out flat per-row: `actions[row * num_dims + d]` is
    ///   the action sampled for dim `d` of row `row`. Length is
    ///   `obs.dims()\[0\] * num_dims`.
    /// - `log_probs[row]` is the joint log-probability summed across dims.
    /// - `values[row]` is the value estimate.
    fn get_action_host(&self, obs: Tensor<B, 2>) -> (Vec<i64>, Vec<f32>, Vec<f32>);

    /// Re-evaluate the policy on previously-sampled actions.
    ///
    /// `actions` is shape `[batch, num_dims]`. For scalar discrete policies
    /// (`num_dims == 1`) pass actions reshaped to `[batch, 1]`. Returns
    /// `(log_probs, entropy, values)` where every tensor has shape
    /// `[batch]`.
    fn evaluate_actions_joint(
        &self,
        obs: Tensor<B, 2>,
        actions: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>);

    /// Shared-trunk feature representation; gradients flow back into the
    /// encoder. The natural quantity to feed into cross-agent regularizers.
    ///
    /// Shape: `[batch, hidden_dim]`.
    fn encoder_features_joint(&self, obs: Tensor<B, 2>) -> Tensor<B, 2>;

    /// Per-dimension action cardinalities.
    ///
    /// - Scalar discrete (e.g. [`crate::policy::mlp::MlpBurnPolicy`] with
    ///   `action_dim = 5`): returns `vec![5]` (one dim, cardinality 5). The
    ///   rollout buffer uses `num_action_dims = 1`.
    /// - Multi-discrete (e.g.
    ///   [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpBurnPolicy`] with
    ///   action dims `[10, 2]`): returns `vec![10, 2]`. The rollout buffer uses
    ///   `num_action_dims = 2`.
    fn action_dims_joint(&self) -> Vec<i64>;
}

// MlpBurnPolicy: scalar discrete. We don't have direct access to the policy's
// action_dim through `Module`, so the impl forces callers to record the
// cardinality at construction time via a thin wrapper. To stay zero-friction
// here we expose the impls on the concrete policy types directly: the MLP
// policy's `policy_head` output dimension is what `action_dims_joint` would
// return, but Burn's `Linear` doesn't surface that directly through the
// `Module` API. We track action dims by inspecting the `policy_head`'s
// weight tensor shape — it lives on the module so this is parity-preserving.
impl<B: AutodiffBackend> JointPolicy<B> for crate::policy::mlp::MlpBurnPolicy<B>
where
    Self: AutodiffModule<B> + Clone,
{
    fn get_action_host(&self, obs: Tensor<B, 2>) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
        let (actions, log_probs, values) = self.get_action_host(obs);
        // Scalar discrete: actions is already 1-per-row.
        (actions, log_probs, values)
    }

    fn evaluate_actions_joint(
        &self,
        obs: Tensor<B, 2>,
        actions: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        // The MLP policy's `evaluate_actions` takes rank-1 actions. Squeeze
        // the dim-1 axis to match.
        let actions_1d: Tensor<B, 1, Int> = actions.squeeze_dim::<1>(1);
        self.evaluate_actions(obs, actions_1d)
    }

    fn encoder_features_joint(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.encoder_features(obs)
    }

    fn action_dims_joint(&self) -> Vec<i64> {
        // `policy_head.weight` has shape `[hidden_dim, action_dim]` in Burn's
        // `Linear` layout; we pull `action_dim` off the second axis.
        let head_dims = self.policy_head_action_dim();
        vec![head_dims as i64]
    }
}

impl<B: AutodiffBackend> JointPolicy<B>
    for crate::policy::multi_discrete_mlp::MultiDiscreteMlpBurnPolicy<B>
where
    Self: AutodiffModule<B> + Clone,
{
    fn get_action_host(&self, obs: Tensor<B, 2>) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
        self.get_action_host(obs)
    }

    fn evaluate_actions_joint(
        &self,
        obs: Tensor<B, 2>,
        actions: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        self.evaluate_actions(obs, actions)
    }

    fn encoder_features_joint(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.encoder_features(obs)
    }

    fn action_dims_joint(&self) -> Vec<i64> {
        self.action_dim_cardinalities().into_iter().map(|d| d as i64).collect()
    }
}

// -----------------------------------------------------------------------
// Trait: minimal joint environment for rollout collection.
// -----------------------------------------------------------------------

/// Per-step result returned by a [`JointEnv`] implementation.
#[derive(Debug, Clone)]
pub struct JointStepResult {
    /// Per-agent rewards.
    pub rewards: Vec<f32>,
    /// Whether the episode has terminated.
    pub done: bool,
    /// Per-agent observations after the step (length = `num_agents`).
    pub observations: Vec<Vec<f32>>,
}

/// Minimal joint-environment surface needed by
/// [`JointMultiAgentTrainer::collect_rollout`].
///
/// Why a fresh trait instead of
/// [`crate::multi_agent::environment::MultiAgentEnvironment`]? The base trait's
/// `step_multi` already takes `actions: &[Vec<i64>]` so it nominally fits, but
/// it also requires a full `Environment` impl (single-action `step`, action
/// spaces, snapshot/restore, etc.). The joint trainer only needs `reset_joint`
/// / `step_joint`, and adapter envs can implement this trait directly without
/// touching the wider trait hierarchy.
pub trait JointEnv {
    /// Reset the env in-place. Returns per-agent observations.
    fn reset_joint(&mut self, seed: Option<u64>) -> Vec<Vec<f32>>;

    /// Step the env with per-agent actions.
    ///
    /// `actions[i]` is the full per-dim action vector for agent `i`:
    /// - Length 1 for scalar discrete (e.g. `[3]`).
    /// - Length `num_dims` for multi-discrete (e.g. `[house_index, mode]`).
    fn step_joint(&mut self, actions: &[Vec<i64>]) -> JointStepResult;
}

// -----------------------------------------------------------------------
// Config / Rollout / Stats
// -----------------------------------------------------------------------

/// Trainer configuration. Plain data; defaults match the tch-era
/// `JointTrainerConfig` field-for-field so the smoke-test parameters
/// stay portable.
#[derive(Debug, Clone)]
pub struct JointTrainerConfig {
    /// Number of agents trained jointly. Must match the length of
    /// [`JointMultiAgentTrainer`]'s policy / optimizer slots.
    pub num_agents: usize,
    /// Steps collected per rollout before each PPO update.
    pub rollout_steps: usize,
    /// Discount factor `γ ∈ [0, 1]`.
    pub gamma: f64,
    /// GAE smoothing parameter `λ ∈ [0, 1]`.
    pub gae_lambda: f64,
    /// PPO policy-ratio clip range `ε`.
    pub clip_range: f64,
    /// PPO value-function clip range. Use `f64::INFINITY` to fall back to
    /// plain MSE (matches the Burn [`compute_value_loss`] contract).
    pub clip_range_vf: f64,
    /// Weight on the value-function loss term inside the joint loss.
    pub vf_coef: f64,
    /// Weight on the entropy bonus.
    pub ent_coef: f64,
    /// Number of PPO epochs per update.
    pub n_epochs: usize,
    /// Minibatch size for SGD within each PPO epoch.
    pub minibatch_size: usize,
    /// Global gradient-norm clip applied through each per-policy optimizer.
    pub max_grad_norm: f64,
    /// Standardize advantages to zero mean / unit variance per minibatch
    /// before computing the surrogate.
    pub normalize_advantages: bool,
}

impl Default for JointTrainerConfig {
    fn default() -> Self {
        Self {
            num_agents: 4,
            rollout_steps: 2048,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_range: 0.2,
            clip_range_vf: 0.2,
            vf_coef: 0.5,
            ent_coef: 0.01,
            n_epochs: 4,
            minibatch_size: 256,
            max_grad_norm: 0.5,
            normalize_advantages: true,
        }
    }
}

/// Synchronized rollout buffer (host-side).
///
/// Every agent sees the *same* observation on every step — the trainer
/// assumes a globally-shared observation; environments with distinct
/// per-agent views should pre-concatenate or use only agent 0's view as
/// the trainer input. Per-agent actions / log-probs / values / rewards
/// are stored as parallel host buffers and materialized into Burn
/// tensors lazily inside [`JointMultiAgentTrainer::update`].
#[derive(Debug, Clone)]
pub struct JointRollout {
    /// Shared observations: flat `[T * obs_dim]`. Same for every agent.
    pub observations: Vec<f32>,
    /// Observation dimensionality.
    pub obs_dim: usize,
    /// Per-agent actions: `Vec<N>[T * num_action_dims]`. `num_action_dims`
    /// is 1 for scalar discrete, `num_dims` for multi-discrete.
    pub actions: Vec<Vec<i64>>,
    /// Number of action dimensions (uniform across agents in this first cut).
    pub num_action_dims: usize,
    /// Per-agent rollout-time log-probs: `Vec<N>[T]`.
    pub log_probs: Vec<Vec<f32>>,
    /// Per-agent value estimates: `Vec<N>[T]`.
    pub values: Vec<Vec<f32>>,
    /// Per-agent rewards: `Vec<N>[T]`.
    pub rewards: Vec<Vec<f32>>,
    /// Episode-termination flag (shared across agents): `[T]`.
    pub dones: Vec<f32>,
}

impl JointRollout {
    /// Rollout length in steps.
    pub fn num_steps(&self) -> usize {
        self.dones.len()
    }

    /// Number of agents represented in this rollout.
    pub fn num_agents(&self) -> usize {
        self.actions.len()
    }
}

/// Per-update training statistics for the joint trainer.
///
/// Mirrors [`crate::train::ppo::TrainingStats`] but with per-agent
/// vectors for the agent-local quantities and a single shared scalar
/// for the auxiliary cross-agent term.
#[derive(Debug, Clone, Default)]
pub struct JointStats {
    /// Per-agent policy loss (averaged across PPO epochs).
    pub policy_loss: Vec<f64>,
    /// Per-agent value-function loss.
    pub value_loss: Vec<f64>,
    /// Per-agent entropy.
    pub entropy: Vec<f64>,
    /// Per-agent fraction of clipped updates.
    pub clip_fraction: Vec<f64>,
    /// Per-agent approximate KL divergence between old and new policy.
    pub approx_kl: Vec<f64>,
    /// Per-agent explained variance of the value function.
    pub explained_var: Vec<f64>,
    /// Auxiliary cross-agent loss (e.g. λ * redundancy_penalty). Scalar
    /// shared by all agents because it's computed jointly on the same
    /// minibatch features.
    pub aux_loss: f64,
    /// Total summed loss `Σ_i agent_loss_i + aux_loss` (averaged across
    /// PPO epochs).
    pub total_loss: f64,
}

impl JointStats {
    /// Construct a fully-zeroed [`JointStats`] sized for `num_agents`
    /// agents.
    pub fn zeros(num_agents: usize) -> Self {
        Self {
            policy_loss: vec![0.0; num_agents],
            value_loss: vec![0.0; num_agents],
            entropy: vec![0.0; num_agents],
            clip_fraction: vec![0.0; num_agents],
            approx_kl: vec![0.0; num_agents],
            explained_var: vec![0.0; num_agents],
            aux_loss: 0.0,
            total_loss: 0.0,
        }
    }
}

// -----------------------------------------------------------------------
// Trainer
// -----------------------------------------------------------------------

/// Synchronized joint multi-agent PPO trainer (Burn backend).
///
/// Generic over:
/// - `B: AutodiffBackend` — the Burn backend.
/// - `P: JointPolicy<B>` — the per-agent policy module type.
/// - `O: Optimizer<P, B>` — the Burn optimizer type (typically built from
///   `AdamConfig::new().init()`).
///
/// The trainer owns `N` policies and `N` optimizers; gradient flow is
/// parameter-isolated because each [`burn::optim::GradientsParams::from_grads`]
/// slice extracts only one policy's parameters from the shared autograd
/// gradients.
pub struct JointMultiAgentTrainer<B, P, O>
where
    B: AutodiffBackend,
    P: JointPolicy<B>,
    O: Optimizer<P, B>,
{
    /// Owned policies. Stored in `Option<P>` slots because Burn's
    /// `Optimizer::step` consumes the module by value; we `.take()` and
    /// put back across each step.
    policies: Vec<Option<P>>,
    /// One optimizer per policy.
    optimizers: Vec<BurnOptimizer<B, P, O>>,
    /// Trainer configuration.
    config: JointTrainerConfig,
    /// Device the policies live on.
    device: B::Device,
}

impl<B, P, O> JointMultiAgentTrainer<B, P, O>
where
    B: AutodiffBackend,
    P: JointPolicy<B>,
    O: Optimizer<P, B>,
{
    /// Construct a trainer from a fully-initialized set of policies and
    /// optimizers.
    ///
    /// `optimizers[i]` is paired with `policies[i]` and only ever updates
    /// `policies[i]`'s parameters.
    pub fn new(
        policies: Vec<P>,
        optimizers: Vec<BurnOptimizer<B, P, O>>,
        config: JointTrainerConfig,
        device: B::Device,
    ) -> Result<Self> {
        if policies.is_empty() {
            return Err(anyhow!("JointMultiAgentTrainer requires at least one policy"));
        }
        if policies.len() != config.num_agents {
            return Err(anyhow!(
                "JointMultiAgentTrainer: policies.len() ({}) != config.num_agents ({})",
                policies.len(),
                config.num_agents
            ));
        }
        if optimizers.len() != policies.len() {
            return Err(anyhow!(
                "JointMultiAgentTrainer: optimizers.len() ({}) != policies.len() ({})",
                optimizers.len(),
                policies.len()
            ));
        }
        // Apply the configured gradient-norm cap on every optimizer.
        let mut optimizers = optimizers;
        for opt in optimizers.iter_mut() {
            opt.clip_grad_norm(config.max_grad_norm);
        }
        Ok(Self { policies: policies.into_iter().map(Some).collect(), optimizers, config, device })
    }

    /// Device the trainer (and all its policies) live on.
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Trainer configuration.
    pub fn config(&self) -> &JointTrainerConfig {
        &self.config
    }

    /// Borrow agent `i`'s policy. Panics if the trainer is mid-`update`.
    pub fn policy(&self, i: usize) -> &P {
        self.policies[i].as_ref().expect("policy is None mid-update")
    }

    /// Drive a [`JointEnv`] for `config.rollout_steps` and return the
    /// synchronized rollout buffer.
    ///
    /// `last_obs` is the persistent "next observation" handed in across
    /// iterations: pass agent-0's observation from the most recent
    /// `env.reset_joint()` or step. The trainer updates it in place so
    /// callers can keep the rollout stream stitched across iterations.
    pub fn collect_rollout<E: JointEnv>(
        &self,
        env: &mut E,
        last_obs: &mut Vec<f32>,
    ) -> JointRollout {
        let num_steps = self.config.rollout_steps;
        let num_agents = self.config.num_agents;
        let obs_dim = last_obs.len();
        let device = self.device.clone();

        // Probe per-dim action layout from agent 0's policy (shape-only — no
        // tensor ops touched, so the result is RNG-neutral). For this first
        // Burn-native cut we require every agent to share the same
        // num_action_dims; per-agent heterogeneous layouts can come later.
        let num_action_dims: usize = self.policies[0]
            .as_ref()
            .expect("policy 0 present at rollout time")
            .action_dims_joint()
            .len();

        let mut obs_buf = vec![0.0_f32; num_steps * obs_dim];
        let mut act_buf: Vec<Vec<i64>> =
            (0..num_agents).map(|_| vec![0_i64; num_steps * num_action_dims]).collect();
        let mut lp_buf: Vec<Vec<f32>> = (0..num_agents).map(|_| vec![0.0_f32; num_steps]).collect();
        let mut val_buf: Vec<Vec<f32>> =
            (0..num_agents).map(|_| vec![0.0_f32; num_steps]).collect();
        let mut rew_buf: Vec<Vec<f32>> =
            (0..num_agents).map(|_| vec![0.0_f32; num_steps]).collect();
        let mut done_buf = vec![0.0_f32; num_steps];

        for t in 0..num_steps {
            let start = t * obs_dim;
            obs_buf[start..start + obs_dim].copy_from_slice(last_obs);

            // Build the rollout-time observation tensor (single-row batch).
            let obs_t = Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(last_obs.clone(), [1, obs_dim]),
                &device,
            );

            let mut joint_action: Vec<Vec<i64>> = Vec::with_capacity(num_agents);
            for (i, slot) in self.policies.iter().enumerate() {
                let policy = slot.as_ref().expect("policy present at rollout time");
                let (actions_host, log_probs_host, values_host) =
                    policy.get_action_host(obs_t.clone());

                // Extract per-agent action vector (length = num_action_dims).
                let row: Vec<i64> = actions_host[..num_action_dims].to_vec();
                let off = t * num_action_dims;
                act_buf[i][off..off + num_action_dims].copy_from_slice(&row);
                joint_action.push(row);

                lp_buf[i][t] = log_probs_host.first().copied().unwrap_or(0.0);
                val_buf[i][t] = values_host.first().copied().unwrap_or(0.0);
            }

            let result = env.step_joint(&joint_action);
            for i in 0..num_agents {
                rew_buf[i][t] = result.rewards[i];
            }
            done_buf[t] = if result.done { 1.0 } else { 0.0 };

            if result.done {
                let fresh = env.reset_joint(None);
                *last_obs = fresh[0].clone();
            } else {
                *last_obs = result.observations[0].clone();
            }
        }

        JointRollout {
            observations: obs_buf,
            obs_dim,
            actions: act_buf,
            num_action_dims,
            log_probs: lp_buf,
            values: val_buf,
            rewards: rew_buf,
            dones: done_buf,
        }
    }

    /// Joint PPO update.
    ///
    /// `aux_fn` receives a slice of per-agent encoder-feature tensors for
    /// the current minibatch (one entry per policy, shape
    /// `[mb, hidden_dim]`) and returns an optional pre-scaled scalar loss
    /// (e.g. the cross-agent redundancy penalty). One `.backward()` flows
    /// through every encoder when `aux_fn` returns `Some`.
    ///
    /// # Minibatch sampling
    ///
    /// One shuffled minibatch of size `config.minibatch_size` is drawn per
    /// epoch, truncated against the rollout length. The order of indices
    /// within the minibatch is irrelevant because every loss is a `mean` /
    /// `sum` reduction over the minibatch dim and therefore
    /// permutation-invariant.
    pub fn update<F>(
        &mut self,
        rollout: &JointRollout,
        rng: &mut StdRng,
        aux_fn: F,
    ) -> Result<JointStats>
    where
        F: FnMut(&[Tensor<B, 2>]) -> Option<Tensor<B, 1>>,
    {
        let num_agents = self.config.num_agents;
        let active = vec![true; num_agents];
        self.update_with_active_agents(rollout, &active, rng, aux_fn)
    }

    /// Joint PPO update with per-agent active mask — the freeze-N-1
    /// primitive used by PSRO's best-response step.
    ///
    /// Identical to [`Self::update`] except that frozen agents
    /// (`active[i] == false`) skip the optimizer step. Their loss is
    /// still summed into the joint backward so the shared autograd
    /// graph remains balanced, but their parameters are guaranteed
    /// unchanged: we put the original policy back in its slot without
    /// calling `optimizer.step`. Per-agent stats for frozen agents are
    /// still recorded in the returned [`JointStats`] so callers can
    /// monitor the mixture's behaviour on the rollout.
    ///
    /// # Use case
    ///
    /// PSRO's outer loop trains one *best-response* policy at a time
    /// against a meta-Nash mixture over the rest of the population
    /// (see [`crate::multi_agent::psro`]). Passing
    /// `active = [false, ..., true (active idx), ..., false]` here is
    /// the canonical freeze-N-1 pattern.
    ///
    /// # Panics
    ///
    /// Returns `Err` if `active.len() != config.num_agents`.
    pub fn update_with_active_agents<F>(
        &mut self,
        rollout: &JointRollout,
        active: &[bool],
        rng: &mut StdRng,
        mut aux_fn: F,
    ) -> Result<JointStats>
    where
        F: FnMut(&[Tensor<B, 2>]) -> Option<Tensor<B, 1>>,
    {
        if active.len() != self.config.num_agents {
            return Err(anyhow!(
                "active mask length {} != config.num_agents {}",
                active.len(),
                self.config.num_agents
            ));
        }
        let device = self.device.clone();
        let num_agents = self.config.num_agents;
        let num_steps = rollout.num_steps();
        if num_steps == 0 {
            return Err(anyhow!("rollout is empty"));
        }
        if rollout.num_agents() != num_agents {
            return Err(anyhow!(
                "rollout has {} agents but trainer is configured for {}",
                rollout.num_agents(),
                num_agents
            ));
        }

        // Per-agent advantages and returns. Computed once outside the epoch
        // loop (matches the tch-era reference behaviour).
        let mut advantages_host: Vec<Vec<f32>> = Vec::with_capacity(num_agents);
        let mut returns_host: Vec<Vec<f32>> = Vec::with_capacity(num_agents);
        for i in 0..num_agents {
            let (adv, ret) = compute_gae_single_agent(
                &rollout.rewards[i],
                &rollout.values[i],
                &rollout.dones,
                self.config.gamma as f32,
                self.config.gae_lambda as f32,
            );
            let adv = if self.config.normalize_advantages {
                normalize_advantages(&adv)
            } else {
                adv
            };
            advantages_host.push(adv);
            returns_host.push(ret);
        }

        let mut stats = JointStats::zeros(num_agents);
        let mb_size = self.config.minibatch_size.min(num_steps);

        for _epoch in 0..self.config.n_epochs {
            // One shuffled minibatch per epoch. The shuffle uses the
            // caller-supplied RNG so PSRO / NFSP runs are bit-reproducible
            // under their configured seeds (see issue #109).
            let mut indices: Vec<usize> = (0..num_steps).collect();
            use rand::seq::SliceRandom;
            indices.shuffle(rng);
            indices.truncate(mb_size);

            let obs_mb = select_obs(&rollout.observations, rollout.obs_dim, &indices, &device);

            // Per-agent forward + per-agent loss accumulation.
            //
            // We collect per-agent loss tensors and feature tensors first,
            // sum them into a single `joint_loss`, then backward once. The
            // gradients of `joint_loss` w.r.t. each policy's parameters are
            // then extracted via `GradientsParams::from_grads` and applied
            // per-policy through that policy's optimizer.
            let mut per_agent_losses: Vec<Tensor<B, 1>> = Vec::with_capacity(num_agents);
            let mut features: Vec<Tensor<B, 2>> = Vec::with_capacity(num_agents);

            // Per-agent host scratch for stats.
            let mut policy_loss_hosts = vec![0.0_f64; num_agents];
            let mut value_loss_hosts = vec![0.0_f64; num_agents];
            let mut entropy_hosts = vec![0.0_f64; num_agents];
            let mut clip_frac_hosts = vec![0.0_f64; num_agents];
            let mut kl_hosts = vec![0.0_f64; num_agents];
            let mut ev_hosts = vec![0.0_f64; num_agents];

            for i in 0..num_agents {
                let policy = self.policies[i]
                    .as_ref()
                    .ok_or_else(|| anyhow!("policy {} is None mid-update", i))?;

                let actions_mb =
                    select_actions(&rollout.actions[i], rollout.num_action_dims, &indices, &device);
                let old_lp_mb = select_f32_row(&rollout.log_probs[i], &indices, &device);
                let adv_mb = select_f32_row(&advantages_host[i], &indices, &device);
                let ret_mb = select_f32_row(&returns_host[i], &indices, &device);
                let old_v_mb = select_f32_row(&rollout.values[i], &indices, &device);

                let (new_lp, entropy, values_mb) =
                    policy.evaluate_actions_joint(obs_mb.clone(), actions_mb);
                let feat = policy.encoder_features_joint(obs_mb.clone());

                let (policy_loss, clip_frac, kl) =
                    compute_policy_loss(new_lp, old_lp_mb, adv_mb, self.config.clip_range);
                let (value_loss, explained_var) =
                    compute_value_loss(values_mb, old_v_mb, ret_mb, self.config.clip_range_vf);
                let entropy_mean = entropy.mean();

                // Host-side stats. We pull scalars from each per-agent loss
                // tensor *before* moving them into the joint sum so we don't
                // need to clone twice.
                policy_loss_hosts[i] = scalar_f64(policy_loss.clone());
                value_loss_hosts[i] = scalar_f64(value_loss.clone());
                entropy_hosts[i] = scalar_f64(entropy_mean.clone());
                clip_frac_hosts[i] = clip_frac;
                kl_hosts[i] = kl;
                ev_hosts[i] = explained_var;

                let agent_loss = policy_loss
                    + value_loss.mul_scalar(self.config.vf_coef as f32)
                    + entropy_mean.neg().mul_scalar(self.config.ent_coef as f32);

                per_agent_losses.push(agent_loss);
                features.push(feat);
            }

            // Aggregate per-agent losses, then add the cross-agent aux term.
            let mut joint_loss: Option<Tensor<B, 1>> = None;
            for l in per_agent_losses {
                joint_loss = Some(match joint_loss.take() {
                    Some(acc) => acc + l,
                    None => l,
                });
            }
            let aux_opt = aux_fn(&features);
            let aux_scalar: f64 = aux_opt.as_ref().map(|t| scalar_f64(t.clone())).unwrap_or(0.0);
            stats.aux_loss += aux_scalar;
            if let Some(aux) = aux_opt {
                joint_loss = Some(match joint_loss.take() {
                    Some(acc) => acc + aux,
                    None => aux,
                });
            }
            let joint_loss = joint_loss.ok_or_else(|| anyhow!("no losses to backprop"))?;
            stats.total_loss += scalar_f64(joint_loss.clone());

            // Single backward over the joint loss; the resulting `Gradients`
            // carry grads for every policy's parameters. Sliced per-policy
            // below.
            //
            // Burn's `Gradients` container is *consumed* per param when we
            // call `from_module(&mut grads, policy_i)` — each visit removes
            // policy `i`'s param tensors from the shared container. That's
            // exactly the per-agent isolation we want: each optimizer only
            // sees grads for its own policy's parameters, and a single
            // backward feeds all of them.
            let mut grads = joint_loss.backward();

            for i in 0..num_agents {
                let policy = self.policies[i]
                    .take()
                    .ok_or_else(|| anyhow!("policy {} is None mid-step", i))?;
                // Drain gradient slice for policy `i` either way; this
                // keeps the `Gradients` container consistent across all
                // agents (Burn removes policy `i`'s params on
                // `from_module`, so we always do the drain).
                let policy_grads = GradientsParams::from_module(&mut grads, &policy);
                let updated = if active[i] {
                    let lr = self.optimizers[i].learning_rate();
                    self.optimizers[i].inner_mut().step(lr, policy, policy_grads)
                } else {
                    // Frozen agent: drop the gradients and put the policy
                    // back unchanged. This is the freeze-N-1 invariant
                    // PSRO's best-response step relies on.
                    drop(policy_grads);
                    policy
                };
                self.policies[i] = Some(updated);

                stats.policy_loss[i] += policy_loss_hosts[i];
                stats.value_loss[i] += value_loss_hosts[i];
                stats.entropy[i] += entropy_hosts[i];
                stats.clip_fraction[i] += clip_frac_hosts[i];
                stats.approx_kl[i] += kl_hosts[i];
                stats.explained_var[i] += ev_hosts[i];
            }
        }

        // Average across epochs.
        let n = self.config.n_epochs as f64;
        if n > 0.0 {
            for i in 0..num_agents {
                stats.policy_loss[i] /= n;
                stats.value_loss[i] /= n;
                stats.entropy[i] /= n;
                stats.clip_fraction[i] /= n;
                stats.approx_kl[i] /= n;
                stats.explained_var[i] /= n;
            }
            stats.aux_loss /= n;
            stats.total_loss /= n;
        }

        Ok(stats)
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Per-agent single-trajectory GAE (host-side).
///
/// Mirrors the pre-Burn `compute_gae_single_agent` helper: takes 1-D
/// rewards / values / dones host buffers and returns
/// `(advantages, returns)`. The trailing-step value is taken as zero
/// (no post-rollout bootstrap), matching the tch reference.
fn compute_gae_single_agent(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    gamma: f32,
    gae_lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let t = rewards.len();
    let mut advantages = vec![0.0_f32; t];
    let mut gae = 0.0_f32;
    for i in (0..t).rev() {
        let next_v = if i == t - 1 { 0.0 } else { values[i + 1] };
        let mask = 1.0 - dones[i];
        let delta = rewards[i] + gamma * next_v * mask - values[i];
        gae = delta + gamma * gae_lambda * mask * gae;
        advantages[i] = gae;
    }
    let returns: Vec<f32> = advantages.iter().zip(values).map(|(&a, &v)| a + v).collect();
    (advantages, returns)
}

/// Standardize a vector of advantages to zero mean / unit variance.
fn normalize_advantages(adv: &[f32]) -> Vec<f32> {
    if adv.is_empty() {
        return Vec::new();
    }
    let n = adv.len() as f64;
    let mean: f64 = adv.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var: f64 = adv.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt().max(1e-8);
    adv.iter().map(|&x| ((x as f64 - mean) / std) as f32).collect()
}

/// Build a `[mb, obs_dim]` tensor from the host observation buffer.
fn select_obs<B: AutodiffBackend>(
    obs_flat: &[f32],
    obs_dim: usize,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut out = Vec::with_capacity(indices.len() * obs_dim);
    for &i in indices {
        let start = i * obs_dim;
        out.extend_from_slice(&obs_flat[start..start + obs_dim]);
    }
    Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(out, [indices.len(), obs_dim]), device)
}

/// Build a `[mb, num_action_dims]` int tensor from the host action buffer.
fn select_actions<B: AutodiffBackend>(
    actions_flat: &[i64],
    num_action_dims: usize,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let mut out = Vec::with_capacity(indices.len() * num_action_dims);
    for &i in indices {
        let start = i * num_action_dims;
        out.extend_from_slice(&actions_flat[start..start + num_action_dims]);
    }
    Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(out, [indices.len(), num_action_dims]),
        device,
    )
}

/// Build a `[mb]` float tensor by gathering host rows.
fn select_f32_row<B: AutodiffBackend>(
    src: &[f32],
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 1> {
    let out: Vec<f32> = indices.iter().map(|&i| src[i]).collect();
    Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(out, [indices.len()]), device)
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
        optim::AdamConfig,
    };
    use rand::SeedableRng;

    use super::*;
    use crate::{
        policy::{mlp::MlpBurnPolicy, multi_discrete_mlp::MultiDiscreteMlpBurnPolicy},
        train::optimizer::BurnOptimizer,
    };

    type B = Autodiff<NdArray<f32>>;

    /// Deterministic mock env: 4-dim obs (sin/cos-ish encoding of t),
    /// scalar rewards = sum of actions, never terminates within rollout.
    struct MockEnv {
        num_agents: usize,
        obs_dim: usize,
        t: usize,
    }

    impl MockEnv {
        fn new(num_agents: usize, obs_dim: usize) -> Self {
            Self { num_agents, obs_dim, t: 0 }
        }

        fn obs_for(&self) -> Vec<f32> {
            (0..self.obs_dim)
                .map(|i| (((self.t * 7 + i * 13) % 100) as f32) / 100.0 - 0.5)
                .collect()
        }
    }

    impl JointEnv for MockEnv {
        fn reset_joint(&mut self, _seed: Option<u64>) -> Vec<Vec<f32>> {
            self.t = 0;
            let obs = self.obs_for();
            (0..self.num_agents).map(|_| obs.clone()).collect()
        }

        fn step_joint(&mut self, actions: &[Vec<i64>]) -> JointStepResult {
            self.t += 1;
            let rewards: Vec<f32> = actions
                .iter()
                .map(|a| a.iter().map(|&x| x as f32).sum::<f32>() / 10.0)
                .collect();
            let obs = self.obs_for();
            let observations = (0..self.num_agents).map(|_| obs.clone()).collect();
            JointStepResult { rewards, done: false, observations }
        }
    }

    fn make_mlp_policies(
        num_agents: usize,
        obs_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        device: &NdArrayDevice,
    ) -> Vec<MlpBurnPolicy<B>> {
        (0..num_agents)
            .map(|_| MlpBurnPolicy::<B>::new(obs_dim, action_dim, hidden_dim, device))
            .collect()
    }

    fn make_multi_discrete_policies(
        num_agents: usize,
        obs_dim: usize,
        action_dims: Vec<usize>,
        hidden_dim: usize,
        device: &NdArrayDevice,
    ) -> Vec<MultiDiscreteMlpBurnPolicy<B>> {
        (0..num_agents)
            .map(|_| {
                MultiDiscreteMlpBurnPolicy::<B>::new(
                    obs_dim,
                    action_dims.clone(),
                    hidden_dim,
                    device,
                )
            })
            .collect()
    }

    fn build_optimizers<P>(n: usize, lr: f64) -> Vec<BurnOptimizer<B, P, impl Optimizer<P, B>>>
    where
        P: AutodiffModule<B>,
    {
        (0..n)
            .map(|_| {
                let inner = AdamConfig::new().init();
                BurnOptimizer::<B, P, _>::new(inner, lr)
            })
            .collect()
    }

    #[test]
    fn test_joint_trainer_smoke() {
        // 2 tiny MlpBurnPolicy instances, scalar discrete actions, 64-step
        // rollout, one update with aux_fn returning None. Assert no panics
        // + finite stats. This is the load-bearing acceptance test for the
        // Burn-native multi_agent port (issue #100).
        let device = Default::default();
        let num_agents = 2;
        let obs_dim: usize = 4;
        let action_dim: usize = 3;
        let policies = make_mlp_policies(num_agents, obs_dim, action_dim, 16, &device);
        let optimizers = build_optimizers::<MlpBurnPolicy<B>>(num_agents, 3e-4);

        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: 64,
            n_epochs: 2,
            minibatch_size: 32,
            ..Default::default()
        };
        let mut trainer =
            JointMultiAgentTrainer::new(policies, optimizers, config, device).unwrap();

        let mut env = MockEnv::new(num_agents, obs_dim);
        let initial = env.reset_joint(None);
        let mut last_obs = initial[0].clone();

        let rollout = trainer.collect_rollout(&mut env, &mut last_obs);
        let mut rng = StdRng::seed_from_u64(0);
        let stats = trainer
            .update(&rollout, &mut rng, |_features: &[Tensor<B, 2>]| -> Option<Tensor<B, 1>> {
                None
            })
            .expect("update should not error");

        assert!(stats.total_loss.is_finite(), "total_loss must be finite");
        assert_eq!(stats.aux_loss, 0.0, "aux_loss must be 0 when aux_fn returns None");
        for i in 0..num_agents {
            assert!(stats.policy_loss[i].is_finite(), "policy_loss[{i}] finite");
            assert!(stats.value_loss[i].is_finite(), "value_loss[{i}] finite");
            assert!(stats.entropy[i].is_finite(), "entropy[{i}] finite");
            assert!(stats.clip_fraction[i].is_finite(), "clip_fraction[{i}] finite");
            assert!(stats.approx_kl[i].is_finite(), "approx_kl[{i}] finite");
            assert!(stats.explained_var[i].is_finite(), "explained_var[{i}] finite");
        }
    }

    #[test]
    fn test_joint_rollout_shapes() {
        let device = Default::default();
        let num_agents = 3;
        let obs_dim: usize = 5;
        let t: usize = 32;
        let policies = make_mlp_policies(num_agents, obs_dim, 4, 16, &device);
        let optimizers = build_optimizers::<MlpBurnPolicy<B>>(num_agents, 3e-4);

        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: t,
            n_epochs: 1,
            minibatch_size: t,
            ..Default::default()
        };
        let trainer = JointMultiAgentTrainer::new(policies, optimizers, config, device).unwrap();

        let mut env = MockEnv::new(num_agents, obs_dim);
        let initial = env.reset_joint(None);
        let mut last_obs = initial[0].clone();
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs);

        assert_eq!(rollout.num_steps(), t);
        assert_eq!(rollout.num_agents(), num_agents);
        assert_eq!(rollout.obs_dim, obs_dim);
        assert_eq!(rollout.num_action_dims, 1);
        for a in &rollout.actions {
            assert_eq!(a.len(), t);
        }
        for r in &rollout.rewards {
            assert_eq!(r.len(), t);
        }
        for lp in &rollout.log_probs {
            assert_eq!(lp.len(), t);
        }
        for v in &rollout.values {
            assert_eq!(v.len(), t);
        }
        assert_eq!(rollout.dones.len(), t);
    }

    #[test]
    fn test_aux_fn_couples_all_agents_into_stats() {
        // With aux_fn = || (features[0] - features[1]).square().sum() the
        // aux_loss stat must be strictly positive after one update because
        // the two policies' initial encoders are independently initialized
        // and so produce different features.
        let device = Default::default();
        let num_agents = 2;
        let obs_dim: usize = 4;
        let policies = make_mlp_policies(num_agents, obs_dim, 3, 16, &device);
        let optimizers = build_optimizers::<MlpBurnPolicy<B>>(num_agents, 1e-3);

        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: 32,
            n_epochs: 1,
            minibatch_size: 32,
            normalize_advantages: false,
            ..Default::default()
        };
        let mut trainer =
            JointMultiAgentTrainer::new(policies, optimizers, config, device).unwrap();

        let mut env = MockEnv::new(num_agents, obs_dim);
        let initial = env.reset_joint(None);
        let mut last_obs = initial[0].clone();
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs);

        let mut rng = StdRng::seed_from_u64(0);
        let stats = trainer
            .update(&rollout, &mut rng, |features: &[Tensor<B, 2>]| -> Option<Tensor<B, 1>> {
                Some((features[0].clone() - features[1].clone()).powf_scalar(2.0_f32).sum())
            })
            .expect("update should not error");

        assert!(
            stats.aux_loss > 0.0,
            "aux_loss must be > 0 with non-zero feature diff, got {}",
            stats.aux_loss
        );
        assert!(stats.total_loss.is_finite());
    }

    #[test]
    fn test_joint_trainer_multi_discrete() {
        // Multi-discrete repeat of the smoke test: factored [3, 2] action
        // space, exercises the `MultiDiscreteMlpBurnPolicy` path through
        // `evaluate_actions_joint` and `select_actions` (action layout
        // `[T, num_dims]`).
        let device = Default::default();
        let num_agents = 2;
        let obs_dim: usize = 4;
        let action_dims = vec![3_usize, 2];
        let policies =
            make_multi_discrete_policies(num_agents, obs_dim, action_dims.clone(), 16, &device);
        let optimizers = build_optimizers::<MultiDiscreteMlpBurnPolicy<B>>(num_agents, 3e-4);

        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: 32,
            n_epochs: 1,
            minibatch_size: 32,
            ..Default::default()
        };
        let mut trainer =
            JointMultiAgentTrainer::new(policies, optimizers, config, device).unwrap();

        let mut env = MockEnv::new(num_agents, obs_dim);
        let initial = env.reset_joint(None);
        let mut last_obs = initial[0].clone();
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs);

        assert_eq!(rollout.num_action_dims, action_dims.len());
        for a in &rollout.actions {
            assert_eq!(a.len(), 32 * action_dims.len());
        }

        let mut rng = StdRng::seed_from_u64(0);
        let stats = trainer
            .update(&rollout, &mut rng, |_features: &[Tensor<B, 2>]| -> Option<Tensor<B, 1>> {
                None
            })
            .expect("update should not error");
        assert!(stats.total_loss.is_finite());
    }
}
