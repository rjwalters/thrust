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
//! By default (and to keep the smoke test deterministic) this trainer
//! takes **one** minibatch per epoch, sampled via
//! [`crate::train::ppo::loss::generate_minibatch_indices`] and truncated to
//! `config.minibatch_size`. Setting
//! [`JointTrainerConfig::iterate_all_minibatches`] switches to the
//! conventional PPO pattern of walking *every* `minibatch_size` chunk per
//! epoch, so a large rollout is fully consumed instead of ~97% discarded
//! (issue #239). The default stays single-minibatch so existing NFSP/PSRO
//! determinism tests are bit-identical.
//!
//! # Gradient clipping
//!
//! `JointTrainerConfig::max_grad_norm` is applied as a **global** L2-norm
//! clip on each policy's gradient slice before the optimizer step (issue
//! #239): if the joint gradient norm exceeds the cap, every gradient is
//! scaled down by `max_norm / ‖g‖`, preserving direction. This stops the
//! raw-scale value-loss gradient from swamping the policy heads on the
//! shared actor-critic trunk.

use anyhow::{Result, anyhow};
use burn::{
    module::{AutodiffModule, Module, ModuleVisitor, Param},
    optim::{GradientsParams, Optimizer},
    tensor::{Int, Tensor, backend::AutodiffBackend},
};
use rand::rngs::StdRng;

use crate::train::{
    optimizer::{BackendOptimizer, BurnOptimizer},
    ppo::loss::{
        compute_policy_loss, compute_value_loss, generate_minibatch_indices_with_rng, scalar_f64,
    },
};

// -----------------------------------------------------------------------
// Trait: what a policy must support to participate.
// -----------------------------------------------------------------------

/// Capabilities a policy must expose to participate in
/// [`JointMultiAgentTrainer`].
///
/// The trait pins exactly the surface the trainer needs:
///
/// - **`get_action_host_seeded`** — rollout-time sampling. Returns
///   `(actions_per_dim, log_probs, values)` on the host so the trainer can
///   build the rollout buffer without tying it to a particular backend tensor.
///   Takes the trainer-owned `StdRng` so `PsroConfig::seed` /
///   `NfspConfig::seed` produce bit-identical rollouts (issue #114).
/// - **`evaluate_actions`** — re-evaluate the current policy on previously
///   sampled actions to compute updated log-probs / entropy / value for the PPO
///   loss; this is the only place autograd-bearing tensors are produced.
/// - **`encoder_features`** — shared-trunk activations for the auxiliary loss.
/// - **`action_dims`** — per-dim action cardinalities, used to size action
///   buffers without invoking the policy.
pub trait JointPolicy<B: AutodiffBackend>: AutodiffModule<B> + Clone {
    /// Sample actions for a single rollout step using a caller-supplied
    /// seeded RNG.
    ///
    /// `obs` carries one row per environment in the rollout batch. Returns
    /// host-side `(actions, log_probs, values)` where:
    ///
    /// - `actions` is laid out flat per-row: `actions[row * num_dims + d]` is
    ///   the action sampled for dim `d` of row `row`. Length is
    ///   `obs.dims()\[0\] * num_dims`.
    /// - `log_probs[row]` is the joint log-probability summed across dims.
    /// - `values[row]` is the value estimate.
    ///
    /// Bit-exactness: two calls with the same `obs`, same policy state,
    /// and same-seeded `rng` produce element-wise identical outputs —
    /// the load-bearing guarantee that `PsroConfig::seed` /
    /// `NfspConfig::seed` rely on after issue #114 completed plumbing
    /// the trainer-owned `StdRng` through the rollout-time action
    /// sampler.
    fn get_action_host_seeded(
        &self,
        obs: Tensor<B, 2>,
        rng: &mut StdRng,
    ) -> (Vec<i64>, Vec<f32>, Vec<f32>);

    /// Batched seeded sampler: `obs` carries `N` rows scored through **this
    /// one policy** in a single forward, returning `N` actions.
    ///
    /// # When this helps (and when it does not)
    ///
    /// This eliminates per-call batch-1 forward overhead on the NdArray
    /// backend wherever many observations are scored through the **same**
    /// model in one step — e.g. the constant-obs marginal probe
    /// ([`crate::multi_agent::nfsp::NfspTrainer::action_marginal_for`]),
    /// or a future parallel-env rollout where one frozen policy scores
    /// many env rows.
    ///
    /// It does **not** apply across the per-agent forwards of NFSP's (or
    /// PSRO's) rollout: those `num_agents` forwards each run a *distinct*
    /// per-agent policy module, and a single batched forward applies one
    /// weight set to every row — so stacking the agents' observations
    /// into one `[N, obs_dim]` tensor would be semantically wrong. NFSP
    /// runs a single shared joint episode (`env.step_joint`), so there is
    /// no within-model batch dimension to collapse in the rollout itself;
    /// the per-step forward count there is irreducible. See issue #235.
    ///
    /// # Determinism
    ///
    /// The default implementation simply calls
    /// [`Self::get_action_host_seeded`] on the full `[N, obs_dim]` tensor,
    /// which already draws RNG in row-major order. Implementors that
    /// override this (the concrete MLP policies do) must keep the per-row
    /// RNG draw order identical so the sampled action stream is
    /// bit-for-bit reproducible under `PsroConfig::seed` / `NfspConfig::seed`
    /// (issue #114).
    fn get_actions_host_seeded_batched(
        &self,
        obs: Tensor<B, 2>,
        rng: &mut StdRng,
    ) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
        self.get_action_host_seeded(obs, rng)
    }

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
    fn get_action_host_seeded(
        &self,
        obs: Tensor<B, 2>,
        rng: &mut StdRng,
    ) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
        let (actions, log_probs, values) = self.get_action_host_seeded(obs, rng);
        // Scalar discrete: actions is already 1-per-row.
        (actions, log_probs, values)
    }

    fn get_actions_host_seeded_batched(
        &self,
        obs: Tensor<B, 2>,
        rng: &mut StdRng,
    ) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
        // One forward over all rows (same model) instead of per-row
        // batch-1 forwards; RNG draw order unchanged (issue #235).
        self.get_actions_host_seeded_batched(obs, rng)
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
    fn get_action_host_seeded(
        &self,
        obs: Tensor<B, 2>,
        rng: &mut StdRng,
    ) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
        self.get_action_host_seeded(obs, rng)
    }

    fn get_actions_host_seeded_batched(
        &self,
        obs: Tensor<B, 2>,
        rng: &mut StdRng,
    ) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
        // One forward over all rows (same model) instead of per-row
        // batch-1 forwards; RNG draw order unchanged (issue #235).
        self.get_actions_host_seeded_batched(obs, rng)
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
    /// Iterate over **all** minibatches per epoch instead of a single
    /// truncated `minibatch_size` draw.
    ///
    /// The historical Burn-native cut took one shuffled minibatch per
    /// epoch (issue #100 simplification), which discards ~97% of a large
    /// rollout and starves the best-response (issue #239). When `true`,
    /// each epoch shuffles the rollout once and walks every
    /// `minibatch_size` chunk, the conventional PPO pattern.
    ///
    /// Defaults to `false` to keep existing NFSP/PSRO determinism tests
    /// bit-identical; callers that want the full-rollout update (e.g. the
    /// bucket-brigade examples) opt in explicitly.
    pub iterate_all_minibatches: bool,
    /// Optional **separate critic learning rate** (issue #239, ranked fix #4).
    ///
    /// The shared actor-critic trunk is trained with one optimizer and one
    /// combined `policy + vf_coef·value − ent_coef·entropy` backward. On the
    /// bucket-brigade cells the critic never fits (explained-variance pinned
    /// at ~0 — see #239), so the normalized advantages are noise and the
    /// policy gradient stays ~0 (entropy stuck at the uniform-max floor).
    /// PR #240's grad-clip / all-minibatch / `vf_coef` knobs did not move
    /// `ev` off 0, and dropping `BR_REWARD_SCALE` to 0.001 collapsed value
    /// loss to ~8 yet `ev` *still* stayed flat — i.e. the critic is not
    /// fitting even tiny well-scaled targets at the shared 3e-4 LR.
    ///
    /// When `Some(lr)`, the joint update splits the combined backward into
    /// two passes per minibatch:
    /// 1. **actor pass** — `policy − ent_coef·entropy` stepped through the
    ///    usual per-agent optimizer at its construction LR;
    /// 2. **critic pass** — `value_loss` alone stepped through a dedicated
    ///    per-agent **critic optimizer** at `critic_lr`.
    ///
    /// Both passes update the full module (shared trunk + their respective
    /// heads), but the critic gets its own Adam moment state and (typically
    /// higher) LR, so it can fit the value target without the policy LR
    /// dragging it. The critic optimizers are supplied via
    /// [`JointMultiAgentTrainer::with_critic_optimizers`]; if `critic_lr` is
    /// `Some` but no critic optimizers were supplied the trainer falls back
    /// to the single combined backward (logged once at construction).
    ///
    /// Defaults to `None` (single combined backward — the historical
    /// behaviour), so existing NFSP/PSRO runs and determinism tests are
    /// bit-identical unless a caller opts in.
    pub critic_lr: Option<f64>,
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
            iterate_all_minibatches: false,
            critic_lr: None,
        }
    }
}

/// Synchronized rollout buffer (host-side).
///
/// Per-agent observations are stored independently so environments with
/// distinct per-agent views (e.g. partial observability, asymmetric
/// information) work without any pre-concatenation. Each agent `i`
/// records its own observation stream into
/// `observations_per_agent[i]` (a flat `[T * obs_dim]` buffer).
/// Per-agent actions / log-probs / values / rewards are stored as
/// parallel host buffers and materialized into Burn tensors lazily
/// inside [`JointMultiAgentTrainer::update`].
#[derive(Debug, Clone)]
pub struct JointRollout {
    /// Per-agent observations: `Vec<N>[T * obs_dim]`. Each inner buffer
    /// holds the observation stream for one agent across the rollout.
    pub observations_per_agent: Vec<Vec<f32>>,
    /// Observation dimensionality (uniform across agents).
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
    /// Optional dedicated **critic** optimizer per policy (issue #239 fix
    /// #4). Present only when [`JointTrainerConfig::critic_lr`] is `Some`
    /// and the caller supplied them via
    /// [`JointMultiAgentTrainer::with_critic_optimizers`]. When present, the
    /// joint update splits actor / critic into two backward passes so the
    /// critic can step at its own LR (see `critic_lr` docs). Empty otherwise.
    critic_optimizers: Vec<BurnOptimizer<B, P, O>>,
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
        Ok(Self {
            policies: policies.into_iter().map(Some).collect(),
            optimizers,
            critic_optimizers: Vec::new(),
            config,
            device,
        })
    }

    /// Construct a trainer with a **dedicated per-agent critic optimizer**
    /// (issue #239, ranked fix #4).
    ///
    /// `critic_optimizers[i]` is paired with `policies[i]` and steps only the
    /// value-loss gradient (a second backward over `value_loss` alone) at its
    /// own learning rate. This decouples the critic's effective LR from the
    /// actor's so the critic can fit the bucket-brigade value target (whose
    /// explained-variance was pinned at ~0 under the single shared optimizer —
    /// see [`JointTrainerConfig::critic_lr`]).
    ///
    /// Requires [`JointTrainerConfig::critic_lr`] to be `Some`; the value is
    /// the LR the supplied critic optimizers were constructed with (recorded
    /// for diagnostics — the actual step uses each critic optimizer's own
    /// construction LR). The same `max_grad_norm` cap is staged on the critic
    /// optimizers too.
    pub fn with_critic_optimizers(
        policies: Vec<P>,
        optimizers: Vec<BurnOptimizer<B, P, O>>,
        critic_optimizers: Vec<BurnOptimizer<B, P, O>>,
        config: JointTrainerConfig,
        device: B::Device,
    ) -> Result<Self> {
        if critic_optimizers.len() != policies.len() {
            return Err(anyhow!(
                "JointMultiAgentTrainer: critic_optimizers.len() ({}) != policies.len() ({})",
                critic_optimizers.len(),
                policies.len()
            ));
        }
        let mut trainer = Self::new(policies, optimizers, config, device)?;
        let mut critic_optimizers = critic_optimizers;
        for opt in critic_optimizers.iter_mut() {
            opt.clip_grad_norm(trainer.config.max_grad_norm);
        }
        trainer.critic_optimizers = critic_optimizers;
        Ok(trainer)
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
    /// `last_obs` is the persistent "next observation per agent" handed
    /// in across iterations: pass the per-agent observations from the
    /// most recent `env.reset_joint()` or step (i.e. the env's full
    /// `Vec<Vec<f32>>` shape, length = `num_agents`). The trainer
    /// updates it in place so callers can keep the rollout stream
    /// stitched across iterations.
    ///
    /// `rng` is consumed by each per-step
    /// [`JointPolicy::get_action_host_seeded`] call. Pass the
    /// trainer-owned `StdRng` (e.g. `PsroTrainer::self.rng`) for
    /// `PsroConfig::seed` / `NfspConfig::seed` to produce bit-identical
    /// rollouts (issue #114).
    pub fn collect_rollout<E: JointEnv>(
        &self,
        env: &mut E,
        last_obs: &mut [Vec<f32>],
        rng: &mut StdRng,
    ) -> JointRollout {
        let num_steps = self.config.rollout_steps;
        let num_agents = self.config.num_agents;
        assert_eq!(
            last_obs.len(),
            num_agents,
            "collect_rollout: last_obs length ({}) must equal num_agents ({})",
            last_obs.len(),
            num_agents,
        );
        let obs_dim = last_obs[0].len();
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

        let mut obs_buf_per_agent: Vec<Vec<f32>> =
            (0..num_agents).map(|_| vec![0.0_f32; num_steps * obs_dim]).collect();
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

            let mut joint_action: Vec<Vec<i64>> = Vec::with_capacity(num_agents);
            for (i, slot) in self.policies.iter().enumerate() {
                let policy = slot.as_ref().expect("policy present at rollout time");

                // Per-agent observation: record into the agent-i buffer and
                // build a single-row obs tensor for the agent-i policy.
                let agent_obs = &last_obs[i];
                obs_buf_per_agent[i][start..start + obs_dim].copy_from_slice(agent_obs);
                let obs_t = Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(agent_obs.clone(), [1, obs_dim]),
                    &device,
                );

                let (actions_host, log_probs_host, values_host) =
                    policy.get_action_host_seeded(obs_t, rng);

                // Extract per-agent action vector (length = num_action_dims).
                let row: Vec<i64> = actions_host[..num_action_dims].to_vec();
                let off = t * num_action_dims;
                act_buf[i][off..off + num_action_dims].copy_from_slice(&row);
                joint_action.push(row);

                lp_buf[i][t] = log_probs_host.first().copied().unwrap_or(0.0);
                val_buf[i][t] = values_host.first().copied().unwrap_or(0.0);
            }

            let result = env.step_joint(&joint_action);
            for (i, rew) in rew_buf.iter_mut().enumerate().take(num_agents) {
                rew[t] = result.rewards[i];
            }
            done_buf[t] = if result.done { 1.0 } else { 0.0 };

            if result.done {
                let fresh = env.reset_joint(None);
                last_obs[..num_agents].clone_from_slice(&fresh[..num_agents]);
            } else {
                last_obs[..num_agents].clone_from_slice(&result.observations[..num_agents]);
            }
        }

        JointRollout {
            observations_per_agent: obs_buf_per_agent,
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
        // Number of minibatch gradient steps actually taken across the
        // whole update; used to average the accumulated per-step stats.
        let mut num_mb_steps: usize = 0;

        for _epoch in 0..self.config.n_epochs {
            // Build this epoch's minibatch index-sets.
            //
            // * `iterate_all_minibatches == false` (default): one shuffled `minibatch_size`
            //   draw per epoch — the historical #100 behaviour, kept bit-identical so
            //   existing NFSP/PSRO determinism tests are unaffected.
            // * `iterate_all_minibatches == true`: shuffle once and walk every
            //   `minibatch_size` chunk so the BR consumes the full rollout instead of
            //   discarding ~97% of it (issue #239).
            //
            // Both branches draw their shuffle randomness from the
            // caller-supplied RNG so PSRO / NFSP runs stay reproducible
            // under their configured seeds (see issues #109 / #114).
            use rand::seq::SliceRandom;
            let minibatches: Vec<Vec<usize>> = if self.config.iterate_all_minibatches {
                generate_minibatch_indices_with_rng(num_steps, mb_size, rng)
            } else {
                let mut indices: Vec<usize> = (0..num_steps).collect();
                indices.shuffle(rng);
                indices.truncate(mb_size);
                vec![indices]
            };

            for indices in minibatches {
                num_mb_steps += 1;
                // Per-agent obs minibatches: agent `i` reads from its own
                // observation buffer at `rollout.observations_per_agent[i]`.
                let obs_mb_per_agent: Vec<Tensor<B, 2>> = (0..num_agents)
                    .map(|i| {
                        select_obs(
                            &rollout.observations_per_agent[i],
                            rollout.obs_dim,
                            &indices,
                            &device,
                        )
                    })
                    .collect();

                // Per-agent forward + per-agent loss accumulation.
                //
                // We collect per-agent loss tensors and feature tensors first,
                // sum them into a single `joint_loss`, then backward once. The
                // gradients of `joint_loss` w.r.t. each policy's parameters are
                // then extracted via `GradientsParams::from_grads` and applied
                // per-policy through that policy's optimizer.
                let mut per_agent_losses: Vec<Tensor<B, 1>> = Vec::with_capacity(num_agents);
                let mut features: Vec<Tensor<B, 2>> = Vec::with_capacity(num_agents);
                // Issue #239 fix #4: when a dedicated critic optimizer is
                // present, the value-loss term is held out of the actor's
                // joint loss and stepped separately at `critic_lr`. We keep
                // each per-agent value-loss tensor (graph still alive) for a
                // second backward below.
                let split_critic = !self.critic_optimizers.is_empty();
                let mut per_agent_value_losses: Vec<Tensor<B, 1>> = if split_critic {
                    Vec::with_capacity(num_agents)
                } else {
                    Vec::new()
                };

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

                    let obs_mb_i = obs_mb_per_agent[i].clone();
                    let actions_mb = select_actions(
                        &rollout.actions[i],
                        rollout.num_action_dims,
                        &indices,
                        &device,
                    );
                    let old_lp_mb = select_f32_row(&rollout.log_probs[i], &indices, &device);
                    let adv_mb = select_f32_row(&advantages_host[i], &indices, &device);
                    let ret_mb = select_f32_row(&returns_host[i], &indices, &device);
                    let old_v_mb = select_f32_row(&rollout.values[i], &indices, &device);

                    let (new_lp, entropy, values_mb) =
                        policy.evaluate_actions_joint(obs_mb_i.clone(), actions_mb);
                    let feat = policy.encoder_features_joint(obs_mb_i);

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

                    let entropy_term = entropy_mean.neg().mul_scalar(self.config.ent_coef as f32);
                    let agent_loss = if split_critic {
                        // Critic stepped separately at `critic_lr`; keep the
                        // value-loss tensor (its autodiff graph is still live)
                        // for the second backward, and exclude it from the
                        // actor's joint loss.
                        per_agent_value_losses.push(value_loss);
                        policy_loss + entropy_term
                    } else {
                        policy_loss
                            + value_loss.mul_scalar(self.config.vf_coef as f32)
                            + entropy_term
                    };

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
                let aux_scalar: f64 =
                    aux_opt.as_ref().map(|t| scalar_f64(t.clone())).unwrap_or(0.0);
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
                        // Global gradient-norm clipping (issue #239). The cap
                        // configured via `JointTrainerConfig::max_grad_norm` is
                        // staged on every optimizer in `new`; apply it here to
                        // each policy's gradient slice before the move-through
                        // step so the (potentially large) value-loss gradient
                        // cannot swamp the shared actor-critic trunk. A `None`
                        // cap leaves the gradients untouched.
                        let policy_grads = match self.optimizers[i].grad_clip_norm() {
                            Some(max_norm) if max_norm > 0.0 => clip_grads_by_global_norm::<B, P>(
                                &policy,
                                policy_grads,
                                max_norm as f32,
                            ),
                            _ => policy_grads,
                        };
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

                // ---- Critic pass (issue #239 fix #4) -----------------------
                // When a dedicated critic optimizer is present, run a second
                // backward over the value loss alone and step it at the
                // critic LR. This gives the critic its own Adam moment state
                // and (typically higher) learning rate, decoupled from the
                // actor, so it can fit the bucket-brigade value target whose
                // explained-variance was otherwise pinned at ~0.
                if split_critic {
                    // Sum the per-agent value losses (weighted by vf_coef, so
                    // the scalar magnitude matches the historical combined
                    // term and the existing grad-clip cap stays calibrated).
                    let mut critic_joint: Option<Tensor<B, 1>> = None;
                    for vl in per_agent_value_losses {
                        let term = vl.mul_scalar(self.config.vf_coef as f32);
                        critic_joint = Some(match critic_joint.take() {
                            Some(acc) => acc + term,
                            None => term,
                        });
                    }
                    let critic_joint =
                        critic_joint.ok_or_else(|| anyhow!("no value losses to backprop"))?;
                    let mut critic_grads = critic_joint.backward();

                    // Indexing `self.policies` / `self.critic_optimizers` by `i`
                    // is load-bearing (`.take()` mutates the `Option` slot), so
                    // the range loop cannot be rewritten as an iterator over a
                    // single slice; mirrors the actor step loop above.
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..num_agents {
                        let policy = self.policies[i]
                            .take()
                            .ok_or_else(|| anyhow!("policy {} is None mid-critic-step", i))?;
                        let value_grads = GradientsParams::from_module(&mut critic_grads, &policy);
                        let updated = if active[i] {
                            let lr = self.critic_optimizers[i].learning_rate();
                            let value_grads = match self.critic_optimizers[i].grad_clip_norm() {
                                Some(max_norm) if max_norm > 0.0 => {
                                    clip_grads_by_global_norm::<B, P>(
                                        &policy,
                                        value_grads,
                                        max_norm as f32,
                                    )
                                }
                                _ => value_grads,
                            };
                            self.critic_optimizers[i].inner_mut().step(lr, policy, value_grads)
                        } else {
                            drop(value_grads);
                            policy
                        };
                        self.policies[i] = Some(updated);
                    }
                }
            }
        }

        // Average across all minibatch gradient steps taken (one per
        // minibatch per epoch). With the default single-minibatch path this
        // is exactly `n_epochs`, preserving the historical averaging.
        let n = num_mb_steps as f64;
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

/// Clip a policy's gradients to a global L2 norm (issue #239).
///
/// This is the standard PPO/A2C "global gradient norm clip": compute the
/// L2 norm of the **concatenation** of every parameter's gradient, and if
/// it exceeds `max_norm`, scale *all* gradients by
/// `max_norm / (total_norm + eps)`. Unlike Burn's built-in
/// `GradientClipping::Norm` (which clips each parameter tensor
/// independently), this preserves the *direction* of the joint gradient —
/// which is exactly what we need so the large shared-trunk value-loss
/// gradient is tamed without distorting the relative scale of the policy
/// and value contributions.
///
/// Implemented with two module visitor passes:
/// 1. accumulate `Σ ‖g_p‖²` across all parameters `p`,
/// 2. if the resulting norm exceeds `max_norm`, multiply every gradient by the
///    clip coefficient and re-register it.
///
/// A non-finite or non-positive total norm leaves the gradients untouched.
fn clip_grads_by_global_norm<B, M>(
    module: &M,
    grads: GradientsParams,
    max_norm: f32,
) -> GradientsParams
where
    B: AutodiffBackend,
    M: Module<B>,
{
    // Burn stores gradient tensors on the *inner* (non-autodiff) backend,
    // so we fetch / re-register them as `Tensor<B::InnerBackend, D>`.
    type Inner<B> = <B as AutodiffBackend>::InnerBackend;

    // Pass 1: sum of squared gradient norms.
    struct NormAccumulator<'a, B: AutodiffBackend> {
        grads: &'a GradientsParams,
        sum_sq: f64,
        _marker: core::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for NormAccumulator<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(g) = self.grads.get::<Inner<B>, D>(param.id) {
                let sq = g.powf_scalar(2.0).sum();
                self.sum_sq += scalar_f64(sq);
            }
        }
    }
    let mut acc =
        NormAccumulator::<B> { grads: &grads, sum_sq: 0.0, _marker: core::marker::PhantomData };
    module.visit(&mut acc);

    let total_norm = acc.sum_sq.sqrt();
    if !total_norm.is_finite() || total_norm <= max_norm as f64 || max_norm <= 0.0 {
        // Already within the cap (or degenerate) — nothing to scale.
        return grads;
    }
    let clip_coef = (max_norm as f64 / (total_norm + 1e-6)) as f32;

    // Pass 2: scale every gradient by the clip coefficient.
    struct Scaler<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        coef: f32,
        _marker: core::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for Scaler<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(g) = self.grads.remove::<Inner<B>, D>(param.id) {
                self.grads.register::<Inner<B>, D>(param.id, g.mul_scalar(self.coef));
            }
        }
    }
    let mut grads = grads;
    let mut scaler =
        Scaler::<B> { grads: &mut grads, coef: clip_coef, _marker: core::marker::PhantomData };
    module.visit(&mut scaler);
    grads
}

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

    /// Hand-computed GAE/returns reference on a tiny rollout (issue #243).
    ///
    /// This pins the [`compute_gae_single_agent`] recursion against values
    /// computed by hand, independent of any env. The rollout deliberately
    /// contains **both** boundary kinds the joint trainer can encounter:
    ///
    /// * a **terminal** boundary at interior step `t == 1` (`done == 1.0`):
    ///   both the value bootstrap `γ·V(s_{t+1})` *and* the GAE carry
    ///   `γ·λ·A_{t+1}` must be masked to zero, so the advantage at the terminal
    ///   step is exactly its own one-step TD error and the GAE chain does not
    ///   leak across the episode boundary.
    /// * a **truncation** boundary at the final step `t == T-1`: the rollout
    ///   ends here with `done == 0.0`, and this implementation uses a zero
    ///   trailing bootstrap (`next_v == 0.0`, no post-rollout `V(s_T)`), as
    ///   documented on `compute_gae_single_agent`.
    ///
    /// Reference values (γ = 0.9, λ = 0.8), computed step-by-step in reverse:
    ///
    /// ```text
    /// rewards = [1.0, 2.0, 3.0, 4.0]
    /// values  = [0.5, 1.0, 1.5, 2.0]
    /// dones   = [0.0, 1.0, 0.0, 0.0]
    ///
    /// t=3 (truncation/rollout-end): next_v=0,  mask=1
    ///     delta = 4 + 0.9*0*1 - 2.0           = 2.00
    ///     A_3   = 2.00 + 0.72*1*0             = 2.00
    /// t=2:                          next_v=2.0, mask=1
    ///     delta = 3 + 0.9*2.0*1 - 1.5         = 3.30
    ///     A_2   = 3.30 + 0.72*1*2.00          = 4.74
    /// t=1 (terminal, done=1):       next_v=1.5, mask=0
    ///     delta = 2 + 0.9*1.5*0 - 1.0         = 1.00   (bootstrap masked)
    ///     A_1   = 1.00 + 0.72*0*4.74          = 1.00   (GAE carry masked)
    /// t=0:                          next_v=1.0, mask=1
    ///     delta = 1 + 0.9*1.0*1 - 0.5         = 1.40
    ///     A_0   = 1.40 + 0.72*1*1.00          = 2.12
    ///
    /// advantages = [2.12, 1.00, 4.74, 2.00]
    /// returns    = advantages + values = [2.62, 2.00, 6.24, 4.00]
    /// ```
    #[test]
    fn gae_matches_hand_computed_reference_with_terminal_and_truncation() {
        let rewards = [1.0_f32, 2.0, 3.0, 4.0];
        let values = [0.5_f32, 1.0, 1.5, 2.0];
        let dones = [0.0_f32, 1.0, 0.0, 0.0];
        let gamma = 0.9_f32;
        let lambda = 0.8_f32;

        let (adv, ret) = compute_gae_single_agent(&rewards, &values, &dones, gamma, lambda);

        let expected_adv = [2.12_f32, 1.00, 4.74, 2.00];
        let expected_ret = [2.62_f32, 2.00, 6.24, 4.00];

        assert_eq!(adv.len(), 4);
        assert_eq!(ret.len(), 4);
        for (i, (&a, &e)) in adv.iter().zip(&expected_adv).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "advantage[{i}] = {a}, expected {e} (terminal at t=1, truncation at t=3)"
            );
        }
        for (i, (&r, &e)) in ret.iter().zip(&expected_ret).enumerate() {
            assert!((r - e).abs() < 1e-5, "return[{i}] = {r}, expected {e}");
        }

        // Explicit invariant check on the terminal boundary: A_1 must equal
        // exactly its own TD error (r_1 - V(s_1) = 1.0) with no leakage of
        // the γλ carry from A_2 across the episode boundary.
        assert!(
            (adv[1] - (rewards[1] - values[1])).abs() < 1e-6,
            "terminal-step advantage must equal its own TD error, no GAE carry-through"
        );
    }

    /// A pure-terminal rollout (every step `done == 1`) must produce
    /// advantages equal to each step's own one-step TD error with the
    /// trailing bootstrap zeroed — there is no cross-step propagation at
    /// all. This isolates the `(1 - done)` masking from the recursion.
    #[test]
    fn gae_all_terminal_collapses_to_td_errors() {
        let rewards = [1.0_f32, -2.0, 0.5];
        let values = [0.25_f32, 0.5, -0.5];
        let dones = [1.0_f32, 1.0, 1.0];
        let gamma = 0.99_f32;
        let lambda = 0.95_f32;

        let (adv, ret) = compute_gae_single_agent(&rewards, &values, &dones, gamma, lambda);

        for i in 0..rewards.len() {
            let td = rewards[i] - values[i]; // bootstrap and carry both masked
            assert!((adv[i] - td).abs() < 1e-6, "adv[{i}] should be pure TD error");
            assert!((ret[i] - (td + values[i])).abs() < 1e-6, "ret[{i}] = adv + value");
        }
    }

    /// `normalize_advantages` must standardize to zero mean / unit variance
    /// (population std) and preserve relative ordering / sign structure.
    #[test]
    fn normalize_advantages_zero_mean_unit_std() {
        let adv = [2.12_f32, 1.00, 4.74, 2.00];
        let out = normalize_advantages(&adv);

        let n = out.len() as f64;
        let mean: f64 = out.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var: f64 = out.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 1e-5, "normalized mean should be ~0, got {mean}");
        assert!(
            (var.sqrt() - 1.0).abs() < 1e-4,
            "normalized std should be ~1, got {}",
            var.sqrt()
        );

        // Ordering preserved: the largest raw advantage stays the largest.
        let argmax_raw = 2usize; // 4.74
        let argmax_norm =
            out.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert_eq!(argmax_raw, argmax_norm, "normalization must preserve ordering");
    }

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
        let mut last_obs = env.reset_joint(None);

        let mut rng = StdRng::seed_from_u64(0);
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs, &mut rng);
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
        let mut last_obs = env.reset_joint(None);
        let mut rng = StdRng::seed_from_u64(0);
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs, &mut rng);

        assert_eq!(rollout.num_steps(), t);
        assert_eq!(rollout.num_agents(), num_agents);
        assert_eq!(rollout.obs_dim, obs_dim);
        assert_eq!(rollout.num_action_dims, 1);
        assert_eq!(rollout.observations_per_agent.len(), num_agents);
        for buf in &rollout.observations_per_agent {
            assert_eq!(buf.len(), t * obs_dim);
        }
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
        let mut last_obs = env.reset_joint(None);
        let mut rng = StdRng::seed_from_u64(0);
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs, &mut rng);

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
        let mut last_obs = env.reset_joint(None);
        let mut rng = StdRng::seed_from_u64(0);
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs, &mut rng);

        assert_eq!(rollout.num_action_dims, action_dims.len());
        for a in &rollout.actions {
            assert_eq!(a.len(), 32 * action_dims.len());
        }

        let stats = trainer
            .update(&rollout, &mut rng, |_features: &[Tensor<B, 2>]| -> Option<Tensor<B, 1>> {
                None
            })
            .expect("update should not error");
        assert!(stats.total_loss.is_finite());
    }

    /// Env that returns *distinct* per-agent observations every step:
    /// agent `i` always sees a one-hot vector with the `i`-th slot set.
    /// Used as the load-bearing regression assertion that the trainer
    /// reads agent `i`'s observation for agent `i` (and not agent 0's
    /// view for everyone) after the per-agent obs refactor.
    struct PerAgentObsMockEnv {
        num_agents: usize,
        obs_dim: usize,
    }

    impl PerAgentObsMockEnv {
        fn new(num_agents: usize, obs_dim: usize) -> Self {
            assert!(obs_dim >= num_agents, "obs_dim must be >= num_agents for one-hot encoding");
            Self { num_agents, obs_dim }
        }

        fn per_agent_obs(&self) -> Vec<Vec<f32>> {
            (0..self.num_agents)
                .map(|i| {
                    let mut v = vec![0.0_f32; self.obs_dim];
                    v[i] = 1.0;
                    v
                })
                .collect()
        }
    }

    impl JointEnv for PerAgentObsMockEnv {
        fn reset_joint(&mut self, _seed: Option<u64>) -> Vec<Vec<f32>> {
            self.per_agent_obs()
        }

        fn step_joint(&mut self, _actions: &[Vec<i64>]) -> JointStepResult {
            JointStepResult {
                rewards: vec![0.0_f32; self.num_agents],
                done: false,
                observations: self.per_agent_obs(),
            }
        }
    }

    #[test]
    fn test_collect_rollout_reads_per_agent_observations() {
        // Regression guard for the per-agent observation refactor (PR
        // #118). The trainer must read agent `i`'s observation for
        // agent `i` — *not* agent 0's view for everyone. We construct
        // an env whose `step_joint` returns distinct one-hot
        // observations per agent and assert that
        // `rollout.observations_per_agent[i]` at every timestep
        // contains agent `i`'s one-hot, not agent 0's.
        let device = Default::default();
        let num_agents = 3;
        let obs_dim: usize = 4; // >= num_agents for one-hot encoding
        let t: usize = 16;
        let policies = make_mlp_policies(num_agents, obs_dim, 2, 8, &device);
        let optimizers = build_optimizers::<MlpBurnPolicy<B>>(num_agents, 3e-4);

        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: t,
            n_epochs: 1,
            minibatch_size: t,
            ..Default::default()
        };
        let trainer = JointMultiAgentTrainer::new(policies, optimizers, config, device).unwrap();

        let mut env = PerAgentObsMockEnv::new(num_agents, obs_dim);
        let mut last_obs = env.reset_joint(None);
        let mut rng = StdRng::seed_from_u64(0);
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs, &mut rng);

        assert_eq!(rollout.observations_per_agent.len(), num_agents);
        for (i, buf) in rollout.observations_per_agent.iter().enumerate() {
            assert_eq!(buf.len(), t * obs_dim, "obs buffer for agent {i} has wrong length");
            // Expected per-step view for agent i: one-hot with slot `i` = 1.0.
            let mut expected = vec![0.0_f32; obs_dim];
            expected[i] = 1.0;
            for step in 0..t {
                let start = step * obs_dim;
                let slice = &buf[start..start + obs_dim];
                assert_eq!(
                    slice,
                    expected.as_slice(),
                    "agent {i} step {step}: observation slice {:?} does not match agent {i}'s view {:?}",
                    slice,
                    expected,
                );
            }
        }
    }

    /// Issue #239: the global-norm clip must (a) leave an already-small
    /// gradient untouched and (b) scale an oversized gradient down so its
    /// global L2 norm equals the cap.
    #[test]
    fn test_clip_grads_by_global_norm() {
        let device: NdArrayDevice = Default::default();
        let policy = MlpBurnPolicy::<B>::new(4, 3, 16, &device);

        // Build a real gradient via a forward/backward so the param ids in
        // `GradientsParams` line up with the module's params.
        let obs = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.1_f32; 4 * 8], [8, 4]),
            &device,
        );
        let logits = policy.encoder_features_joint(obs);
        let loss = logits.powf_scalar(2.0).sum();
        let grads = GradientsParams::from_grads(loss.backward(), &policy);

        // Helper: global L2 norm of a GradientsParams over `policy`'s params.
        fn global_norm(policy: &MlpBurnPolicy<B>, grads: &GradientsParams) -> f64 {
            struct Acc<'a> {
                grads: &'a GradientsParams,
                sum_sq: f64,
            }
            impl ModuleVisitor<B> for Acc<'_> {
                fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                    if let Some(g) =
                        self.grads.get::<<B as AutodiffBackend>::InnerBackend, D>(param.id)
                    {
                        self.sum_sq += scalar_f64(g.powf_scalar(2.0).sum());
                    }
                }
            }
            let mut acc = Acc { grads, sum_sq: 0.0 };
            policy.visit(&mut acc);
            acc.sum_sq.sqrt()
        }

        let norm_before = global_norm(&policy, &grads);
        assert!(norm_before.is_finite() && norm_before > 0.0);

        // (a) A cap well above the gradient norm is a no-op.
        let big_cap = norm_before * 10.0;
        let unclipped =
            clip_grads_by_global_norm::<B, MlpBurnPolicy<B>>(&policy, grads, big_cap as f32);
        let norm_unclipped = global_norm(&policy, &unclipped);
        assert!(
            (norm_unclipped - norm_before).abs() < 1e-4,
            "cap above norm must not change gradients: {norm_before} -> {norm_unclipped}"
        );

        // (b) A small cap scales the global norm down to (approximately) the
        // cap.
        let small_cap = (norm_before / 4.0) as f32;
        let clipped =
            clip_grads_by_global_norm::<B, MlpBurnPolicy<B>>(&policy, unclipped, small_cap);
        let norm_clipped = global_norm(&policy, &clipped);
        assert!(
            (norm_clipped - small_cap as f64).abs() < 1e-3 * small_cap as f64 + 1e-4,
            "clipped global norm {norm_clipped} should equal cap {small_cap}"
        );
    }

    /// Issue #239: with `iterate_all_minibatches = true` the update walks
    /// every minibatch per epoch (more gradient steps than the single-draw
    /// default) and still produces finite stats.
    #[test]
    fn test_iterate_all_minibatches_runs() {
        let device = Default::default();
        let num_agents = 2;
        let obs_dim: usize = 4;
        let action_dim: usize = 3;
        let policies = make_mlp_policies(num_agents, obs_dim, action_dim, 16, &device);
        let optimizers = build_optimizers::<MlpBurnPolicy<B>>(num_agents, 3e-4);

        // 128-step rollout, minibatch 32 => 4 minibatches per epoch.
        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: 128,
            n_epochs: 2,
            minibatch_size: 32,
            iterate_all_minibatches: true,
            ..Default::default()
        };
        let mut trainer =
            JointMultiAgentTrainer::new(policies, optimizers, config, device).unwrap();

        let mut env = MockEnv::new(num_agents, obs_dim);
        let mut last_obs = env.reset_joint(None);
        let mut rng = StdRng::seed_from_u64(0);
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs, &mut rng);
        let stats = trainer
            .update(&rollout, &mut rng, |_features: &[Tensor<B, 2>]| -> Option<Tensor<B, 1>> {
                None
            })
            .expect("update should not error");

        for i in 0..num_agents {
            assert!(stats.policy_loss[i].is_finite(), "policy_loss[{i}] finite");
            assert!(stats.value_loss[i].is_finite(), "value_loss[{i}] finite");
            assert!(stats.entropy[i].is_finite(), "entropy[{i}] finite");
        }
        assert!(stats.total_loss.is_finite());
    }

    /// Issue #239 fix #4: with a dedicated critic optimizer
    /// (`with_critic_optimizers`) the update runs the actor and critic in two
    /// backward passes, steps both, and still produces finite stats. Mirrors
    /// `test_iterate_all_minibatches_runs` but exercises the split-critic path.
    #[test]
    fn test_split_critic_optimizer_runs() {
        let device = Default::default();
        let num_agents = 2;
        let obs_dim: usize = 4;
        let action_dim: usize = 3;
        let policies = make_mlp_policies(num_agents, obs_dim, action_dim, 16, &device);
        let optimizers = build_optimizers::<MlpBurnPolicy<B>>(num_agents, 3e-4);
        // Dedicated critic optimizers at a higher LR (the #239 fix #4 pattern).
        let critic_optimizers = build_optimizers::<MlpBurnPolicy<B>>(num_agents, 1e-3);

        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: 128,
            n_epochs: 2,
            minibatch_size: 32,
            iterate_all_minibatches: true,
            critic_lr: Some(1e-3),
            ..Default::default()
        };
        let mut trainer = JointMultiAgentTrainer::with_critic_optimizers(
            policies,
            optimizers,
            critic_optimizers,
            config,
            device,
        )
        .unwrap();

        let mut env = MockEnv::new(num_agents, obs_dim);
        let mut last_obs = env.reset_joint(None);
        let mut rng = StdRng::seed_from_u64(0);
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs, &mut rng);
        let stats = trainer
            .update(&rollout, &mut rng, |_features: &[Tensor<B, 2>]| -> Option<Tensor<B, 1>> {
                None
            })
            .expect("split-critic update should not error");

        for i in 0..num_agents {
            assert!(stats.policy_loss[i].is_finite(), "policy_loss[{i}] finite");
            assert!(stats.value_loss[i].is_finite(), "value_loss[{i}] finite");
            assert!(stats.entropy[i].is_finite(), "entropy[{i}] finite");
            assert!(stats.explained_var[i].is_finite(), "explained_var[{i}] finite");
        }
        assert!(stats.total_loss.is_finite());
    }

    /// Issue #239 fix #4: supplying a critic-optimizer count that mismatches
    /// the policy count is rejected at construction.
    #[test]
    fn test_with_critic_optimizers_length_mismatch_errors() {
        let device = Default::default();
        let num_agents = 2;
        let policies = make_mlp_policies(num_agents, 4, 3, 8, &device);
        let optimizers = build_optimizers::<MlpBurnPolicy<B>>(num_agents, 3e-4);
        // One too few critic optimizers.
        let critic_optimizers = build_optimizers::<MlpBurnPolicy<B>>(num_agents - 1, 1e-3);
        let config = JointTrainerConfig { num_agents, ..Default::default() };
        let result = JointMultiAgentTrainer::with_critic_optimizers(
            policies,
            optimizers,
            critic_optimizers,
            config,
            device,
        );
        assert!(result.is_err(), "mismatched critic_optimizers length must be rejected");
    }

    /// Issue #239 fix #4 regression guard: the default trainer (no critic
    /// optimizers) takes the single combined backward and is unaffected by
    /// the split-critic plumbing — `critic_lr` defaults to `None`.
    #[test]
    fn test_default_path_has_no_critic_optimizers() {
        let device = Default::default();
        let num_agents = 2;
        let policies = make_mlp_policies(num_agents, 4, 3, 8, &device);
        let optimizers = build_optimizers::<MlpBurnPolicy<B>>(num_agents, 3e-4);
        let config = JointTrainerConfig { num_agents, ..Default::default() };
        assert!(
            JointTrainerConfig::default().critic_lr.is_none(),
            "default critic_lr must be None"
        );
        let trainer = JointMultiAgentTrainer::new(policies, optimizers, config, device).unwrap();
        assert!(
            trainer.critic_optimizers.is_empty(),
            "default trainer must carry no critic optimizers (single combined backward)"
        );
    }
}
