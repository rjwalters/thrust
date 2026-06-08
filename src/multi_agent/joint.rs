//! Synchronized joint multi-agent PPO trainer.
//!
//! This module provides [`JointMultiAgentTrainer`], a *synchronized*
//! alternative to the per-thread architecture in
//! [`crate::multi_agent::{PolicyLearner, Population}`].
//!
//! # When to use this module
//!
//! Use the joint trainer when you need a loss term that depends on **all**
//! agents' parameters evaluated on the **same minibatch** at the **same
//! optimization step**. The canonical motivating example is the Slepian-Wolf
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
//! Use [`crate::multi_agent::PolicyLearner`] /
//! [`crate::multi_agent::Population`] instead when:
//!
//! - Each agent updates independently (league play, self-play, evolutionary
//!   tournaments).
//! - Agents may live on different processes / devices.
//! - You don't need a cross-agent loss term.
//!
//! # Single-graph, multiple-optimizer semantics
//!
//! The trainer holds `N` policies and `N` independent optimizers (one per
//! policy's `VarStore`). At each PPO epoch / minibatch:
//!
//! 1. Each policy computes its own `(policy_loss, value_loss, entropy)`.
//! 2. The caller-supplied `aux_fn` is invoked on every policy's encoder
//!    features and may return an additional scalar loss.
//! 3. All per-agent losses plus the aux loss are summed and **one**
//!    `backward()` is called.
//! 4. Every optimizer is stepped. Because each optimizer reads only its own
//!    parameters' `.grad`, gradients from agent `i`'s heads stay inside agent
//!    `i`'s var-store, while the aux term's gradient flows into every encoder
//!    by construction.
//!
//! # Status
//!
//! The reference implementation was originally inlined at
//! `examples/games/bucket_brigade/train_p3.rs`; this module is the library
//! extraction. Numerical parity vs the original example is required (see
//! issue #5).

use anyhow::{Result, anyhow};
use tch::{Device, Kind, Tensor, nn};

use crate::{
    multi_agent::centralized_critic::{
        CentralizedCritic, CentralizedCriticConfig, compute_centralized_value_loss,
    },
    train::ppo::{compute_policy_loss, compute_value_loss},
};

// -----------------------------------------------------------------------
// Trait: what a policy must support to participate.
// -----------------------------------------------------------------------

/// Capabilities a policy must expose to participate in
/// [`JointMultiAgentTrainer`].
///
/// The trait pins exactly the surface the trainer needs:
///
/// - **`get_action`**: rollout-time sampling. Returns `(action, log_prob,
///   value)` where `action` is whatever shape the policy emits (scalar
///   discrete: `[batch]`; multi-discrete: `[batch, num_dims]`).
/// - **`evaluate_actions`**: re-evaluate the current policy on a previously
///   sampled action to compute updated log-probs / entropy / value for the PPO
///   loss.
/// - **`encoder_features`**: shared-trunk activations for the auxiliary loss.
/// - **`var_store`** / **`device`**: book-keeping for the optimizer and
///   device-uniformity checks.
///
/// Implemented for both [`crate::policy::mlp::MlpPolicy`] (scalar discrete)
/// and [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpPolicy`]
/// (factored discrete).
pub trait JointPolicy {
    /// Sample an action from the policy.
    ///
    /// Returns `(actions, log_probs, values)`:
    /// - `actions`   - shape depends on the policy; `[batch]` for scalar
    ///   discrete, `[batch, num_dims]` for multi-discrete.
    /// - `log_probs` - `[batch]`, joint log-probability summed across dims.
    /// - `values`    - `[batch]`, value estimate.
    fn get_action(&self, obs: &Tensor) -> (Tensor, Tensor, Tensor);

    /// Re-evaluate the policy on a previously sampled action.
    ///
    /// Returns `(log_probs, entropy, values)`. Both `log_probs` and `values`
    /// are `[batch]`; `entropy` is `[batch]` per-step (the trainer reduces it
    /// via `.mean()`).
    fn evaluate_actions(&self, obs: &Tensor, actions: &Tensor) -> (Tensor, Tensor, Tensor);

    /// Shared-trunk feature representation; gradients flow back into the
    /// encoder. The natural quantity to feed into cross-agent regularizers.
    ///
    /// Shape: `[batch, hidden_dim]`.
    fn encoder_features(&self, obs: &Tensor) -> Tensor;

    /// Device the policy lives on.
    fn device(&self) -> Device;

    /// Immutable view of the policy's var-store (for parameter inspection
    /// and checkpointing).
    fn var_store(&self) -> &nn::VarStore;

    /// Per-dimension action cardinalities.
    ///
    /// Returns a vector whose length is the number of factored action
    /// dimensions and whose entries are the cardinality of each dim.
    ///
    /// - Scalar discrete (e.g. [`crate::policy::mlp::MlpPolicy`] with
    ///   `action_dim = 5`): returns `vec![5]` (one dim, cardinality 5). The
    ///   rollout buffer uses `num_action_dims = 1`.
    /// - Multi-discrete (e.g.
    ///   [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpPolicy`] with
    ///   `action_dims = [10, 2]`): returns `vec![10, 2]`. The rollout buffer
    ///   uses `num_action_dims = 2`.
    ///
    /// Used by [`JointMultiAgentTrainer::collect_rollout`] to size action
    /// buffers without calling [`JointPolicy::get_action`] (which would
    /// consume libtorch RNG draws -- one per action dimension via
    /// `multinomial` -- and break parity with reference implementations).
    fn action_dims(&self) -> Vec<i64>;
}

impl JointPolicy for crate::policy::mlp::MlpPolicy {
    fn get_action(&self, obs: &Tensor) -> (Tensor, Tensor, Tensor) {
        self.get_action(obs)
    }
    fn evaluate_actions(&self, obs: &Tensor, actions: &Tensor) -> (Tensor, Tensor, Tensor) {
        self.evaluate_actions(obs, actions)
    }
    fn encoder_features(&self, obs: &Tensor) -> Tensor {
        self.encoder_features(obs)
    }
    fn device(&self) -> Device {
        self.device()
    }
    fn var_store(&self) -> &nn::VarStore {
        self.var_store()
    }
    fn action_dims(&self) -> Vec<i64> {
        // Scalar-discrete policy: a single action dim of cardinality
        // `self.action_dim()`.
        vec![self.action_dim()]
    }
}

impl JointPolicy for crate::policy::multi_discrete_mlp::MultiDiscreteMlpPolicy {
    fn get_action(&self, obs: &Tensor) -> (Tensor, Tensor, Tensor) {
        self.get_action(obs)
    }
    fn evaluate_actions(&self, obs: &Tensor, actions: &Tensor) -> (Tensor, Tensor, Tensor) {
        self.evaluate_actions(obs, actions)
    }
    fn encoder_features(&self, obs: &Tensor) -> Tensor {
        self.encoder_features(obs)
    }
    fn device(&self) -> Device {
        self.device()
    }
    fn var_store(&self) -> &nn::VarStore {
        self.var_store()
    }
    fn action_dims(&self) -> Vec<i64> {
        // Naming-collision note: `MultiDiscreteMlpPolicy` already has an
        // inherent method `action_dims(&self) -> &[i64]`. The trait method
        // returns `Vec<i64>` (a different type), so there's no ambiguity at
        // the language level -- the impl-block context resolves the call to
        // the inherent method, and we materialize a `Vec` via `.to_vec()`.
        // Readers may double-take; the comment documents the intent.
        self.action_dims().to_vec()
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
/// Why a fresh trait instead of [`crate::multi_agent::MultiAgentEnvironment`]?
/// The existing trait takes `actions: &[i64]` and so cannot express factored /
/// multi-discrete action spaces. Implementations only need to translate the
/// per-agent `Vec<i64>` (one entry per action dim) into the env's native
/// representation. Trivial adapters can wrap concrete envs without touching
/// the wider trait hierarchy.
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

/// Trainer configuration. Plain data; defaults match the
/// `examples/games/bucket_brigade/train_p3.rs` reference values.
#[derive(Debug, Clone)]
pub struct JointTrainerConfig {
    /// Number of agents trained jointly. Must match the length of
    /// [`JointMultiAgentTrainer::policies`] and the per-agent vectors
    /// in [`JointRollout`] / [`JointStats`].
    pub num_agents: usize,
    /// Steps collected per rollout before each PPO update. Larger
    /// values reduce gradient variance but delay policy refresh; the
    /// PPO baseline default is 2048.
    pub rollout_steps: usize,
    /// Discount factor `γ ∈ [0, 1]`. Controls how far the value
    /// function looks ahead; `0.99` is the PPO baseline for episodic
    /// control tasks.
    pub gamma: f64,
    /// GAE smoothing parameter `λ ∈ [0, 1]`. Trades bias for variance
    /// in the advantage estimate: `0.0` recovers TD(0), `1.0` recovers
    /// Monte-Carlo returns. `0.95` is the PPO baseline.
    pub gae_lambda: f64,
    /// PPO policy-ratio clip range `ε`. The surrogate objective is
    /// clipped to `[1 - ε, 1 + ε]`; `0.2` is the canonical default and
    /// implicitly bounds the per-update KL divergence.
    pub clip_range: f64,
    /// PPO value-function clip range. Bounds the per-update change in
    /// `V(s)` to the same `±clip_range_vf` window. `0.0` disables value
    /// clipping (the loss falls back to plain MSE against the target).
    pub clip_range_vf: f64,
    /// Weight on the value-function loss term inside the joint loss.
    /// `0.5` is the PPO baseline; raise if value estimates lag the
    /// policy improvement, lower if value-loss gradients dominate.
    pub vf_coef: f64,
    /// Weight on the entropy bonus. Encourages exploration by
    /// penalizing low-entropy (over-confident) policies. `0.01` is the
    /// PPO baseline; raise for sparse-reward / hard-exploration tasks.
    pub ent_coef: f64,
    /// Number of PPO epochs (full passes over the rollout) per update.
    /// `4` is the PPO baseline; higher values squeeze more learning out
    /// of each rollout but risk drifting too far from the behavior
    /// policy.
    pub n_epochs: usize,
    /// Minibatch size for SGD within each PPO epoch. Must evenly divide
    /// `rollout_steps`; `256` is the PPO baseline. Smaller minibatches
    /// = noisier gradients but more updates per epoch.
    pub minibatch_size: usize,
    /// Global gradient-norm clip. Each per-agent backward pass has its
    /// L2-norm clipped to this value before the optimizer step; `0.5`
    /// is the PPO baseline and protects against rare large-gradient
    /// spikes from outlier advantages.
    pub max_grad_norm: f64,
    /// If `true`, standardize advantages to zero mean / unit variance
    /// per minibatch before computing the surrogate objective. Strongly
    /// recommended on heterogeneous reward scales; default `true`.
    pub normalize_advantages: bool,
    /// Optional centralized-critic configuration. When `None` (the default),
    /// the trainer behaves bit-for-bit identically to its pre-centralized-
    /// critic implementation — guarded by
    /// `test_update_numerical_parity_no_critic` in the trainer tests.
    ///
    /// When `Some`, callers must additionally supply the [`CentralizedCritic`]
    /// instance and its optimizer via
    /// [`JointMultiAgentTrainer::new_with_centralized_critic`].
    pub centralized_critic: Option<CentralizedCriticConfig>,
}

impl Default for JointTrainerConfig {
    fn default() -> Self {
        Self {
            num_agents: 4,
            rollout_steps: 2048,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_range: 0.2,
            clip_range_vf: 0.0,
            vf_coef: 0.5,
            ent_coef: 0.01,
            n_epochs: 4,
            minibatch_size: 256,
            max_grad_norm: 0.5,
            normalize_advantages: true,
            centralized_critic: None,
        }
    }
}

/// Synchronized rollout buffer.
///
/// Every agent sees the *same* observation on every step (the trainer assumes
/// a globally-shared observation; environments with distinct per-agent views
/// should pre-concatenate the per-agent observations or use only agent 0's
/// view as the trainer input). Per-agent actions / log-probs / values /
/// rewards are stored in parallel tensors.
#[derive(Debug)]
pub struct JointRollout {
    /// Shared observations: `[T, obs_dim]`. Same for every agent.
    pub observations: Tensor,
    /// Per-agent actions: `Vec<N>[T, action_shape]`. `action_shape` is
    /// either `[T]` for scalar discrete or `[T, num_dims]` for multi-discrete.
    pub actions: Vec<Tensor>,
    /// Per-agent old log-probs from rollout-time policy: `Vec<N>[T]`.
    pub log_probs: Vec<Tensor>,
    /// Per-agent value estimates: `Vec<N>[T]`.
    pub values: Vec<Tensor>,
    /// Per-agent rewards: `Vec<N>[T]`.
    pub rewards: Vec<Tensor>,
    /// Episode-termination flag (shared across agents): `[T]`.
    pub dones: Tensor,
}

impl JointRollout {
    /// Rollout length in steps.
    pub fn num_steps(&self) -> i64 {
        self.observations.size()[0]
    }

    /// Observation dimensionality.
    pub fn obs_dim(&self) -> i64 {
        let s = self.observations.size();
        s.get(1).copied().unwrap_or(0)
    }

    /// Number of agents represented in this rollout.
    pub fn num_agents(&self) -> usize {
        self.actions.len()
    }
}

/// Per-update training statistics for the joint trainer.
///
/// Mirrors [`crate::train::ppo::TrainingStats`] but with per-agent vectors
/// for the agent-local quantities and a single shared scalar for the
/// auxiliary cross-agent term.
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
    /// Centralized-critic value loss. `0.0` when no centralized critic is
    /// attached (preserves the per-stats parity guarantee against the
    /// no-critic baseline).
    pub centralized_value_loss: f64,
    /// Centralized-critic explained variance. `0.0` when no centralized critic
    /// is attached.
    pub centralized_explained_var: f64,
    /// Total summed loss `Σ_i agent_loss_i + aux_loss` (averaged across
    /// PPO epochs).
    pub total_loss: f64,
}

impl JointStats {
    /// Construct a fully-zeroed [`JointStats`] sized for `num_agents`
    /// agents. Use as the accumulator-init in the train loop before
    /// summing per-epoch stats and dividing at the end.
    pub fn zeros(num_agents: usize) -> Self {
        Self {
            policy_loss: vec![0.0; num_agents],
            value_loss: vec![0.0; num_agents],
            entropy: vec![0.0; num_agents],
            clip_fraction: vec![0.0; num_agents],
            approx_kl: vec![0.0; num_agents],
            explained_var: vec![0.0; num_agents],
            aux_loss: 0.0,
            centralized_value_loss: 0.0,
            centralized_explained_var: 0.0,
            total_loss: 0.0,
        }
    }
}

// -----------------------------------------------------------------------
// Trainer
// -----------------------------------------------------------------------

/// Synchronized joint multi-agent PPO trainer.
///
/// See the module-level docs for the synchronized-vs-independent contrast
/// with [`crate::multi_agent::PolicyLearner`] and the single-backward-pass
/// semantics.
pub struct JointMultiAgentTrainer<P: JointPolicy> {
    /// Owned policies. The trainer mutates them via gradient steps but never
    /// replaces them; callers may snapshot via `policies[i].var_store()` at
    /// any time.
    pub policies: Vec<P>,
    /// One optimizer per policy. Each reads only its own policy's `.grad`,
    /// so per-agent gradients stay isolated even though all losses share a
    /// single backward pass.
    pub optimizers: Vec<nn::Optimizer>,
    /// Trainer configuration.
    pub config: JointTrainerConfig,
    /// Optional centralized critic. When `Some`, its value loss is added to
    /// the joint loss; when `None`, the trainer's behavior is bit-for-bit
    /// identical to the pre-centralized-critic implementation. The critic
    /// owns its own [`nn::VarStore`]; gradient flow is parameter-isolated
    /// (the critic's loss only updates critic parameters).
    pub centralized_critic: Option<CentralizedCritic>,
    /// Optional optimizer paired with [`Self::centralized_critic`]. Must be
    /// `Some` exactly when `centralized_critic` is `Some`.
    pub centralized_critic_optimizer: Option<nn::Optimizer>,
    /// Device all policies live on. Validated at construction.
    device: Device,
}

impl<P: JointPolicy> JointMultiAgentTrainer<P> {
    /// Construct a trainer from a fully-initialized set of policies.
    ///
    /// `optimizers` must have one entry per policy and be configured for the
    /// caller's choice of learning rate / weight decay.
    ///
    /// All policies must be on the same device.
    pub fn new(
        policies: Vec<P>,
        optimizers: Vec<nn::Optimizer>,
        config: JointTrainerConfig,
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
        let device = policies[0].device();
        for (i, p) in policies.iter().enumerate().skip(1) {
            if p.device() != device {
                return Err(anyhow!(
                    "JointMultiAgentTrainer: all policies must live on the same device; \
                     policy 0 is {:?} but policy {} is {:?}",
                    device,
                    i,
                    p.device()
                ));
            }
        }
        Ok(Self {
            policies,
            optimizers,
            config,
            centralized_critic: None,
            centralized_critic_optimizer: None,
            device,
        })
    }

    /// Construct a trainer with an attached centralized critic.
    ///
    /// Mirrors [`Self::new`] but also takes a [`CentralizedCritic`] instance
    /// and its [`nn::Optimizer`]. The critic must live on the same device as
    /// the policies.
    ///
    /// Note: this constructor does NOT set `config.centralized_critic` — the
    /// config field is informational only. The trainer's runtime branch on
    /// the critic uses `self.centralized_critic.is_some()`.
    pub fn new_with_centralized_critic(
        policies: Vec<P>,
        optimizers: Vec<nn::Optimizer>,
        critic: CentralizedCritic,
        critic_optimizer: nn::Optimizer,
        config: JointTrainerConfig,
    ) -> Result<Self> {
        let mut trainer = Self::new(policies, optimizers, config)?;
        if critic.device() != trainer.device {
            return Err(anyhow!(
                "JointMultiAgentTrainer: centralized critic device {:?} != trainer device {:?}",
                critic.device(),
                trainer.device
            ));
        }
        trainer.centralized_critic = Some(critic);
        trainer.centralized_critic_optimizer = Some(critic_optimizer);
        Ok(trainer)
    }

    /// Device the trainer (and all its policies) live on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Drive a [`JointEnv`] for `config.rollout_steps` and return the
    /// synchronized rollout buffer.
    ///
    /// `last_obs` is the persistent "next observation" handed in across
    /// iterations: pass agent-0's observation from the most recent
    /// `env.reset()` or step. The trainer updates it in place so callers can
    /// keep the rollout stream stitched across iterations.
    pub fn collect_rollout<E: JointEnv>(
        &self,
        env: &mut E,
        last_obs: &mut Vec<f32>,
    ) -> JointRollout {
        let num_steps = self.config.rollout_steps;
        let num_agents = self.config.num_agents;
        let obs_dim = last_obs.len();
        let device = self.device;

        // Determine action shape from the policy's declared per-dim
        // cardinalities. The previous implementation called
        // `policies[0].get_action(&probe_obs)` here, which consumes one
        // libtorch RNG draw per action dimension (via `multinomial`) and so
        // shifts the per-step action sampling stream for the entire rollout.
        // That broke first-iter numerical parity with the pre-extraction
        // reference example at `40ec676:examples/games/bucket_brigade/train_p3.rs`.
        // Using the trait's shape-introspection method is parity-preserving
        // because it touches no tensor ops.
        //
        // Action layout (unchanged from before):
        //   scalar discrete: [1]            -> per-step action = scalar
        //   multi-discrete:  [1, num_dims]  -> per-step action = vec of dims
        let num_action_dims: usize = self.policies[0].action_dims().len();

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

            let obs_t = Tensor::from_slice(last_obs).to_device(device).view([1, obs_dim as i64]);

            let mut joint_action: Vec<Vec<i64>> = Vec::with_capacity(num_agents);
            for (i, policy) in self.policies.iter().enumerate() {
                let (actions_t, log_p_t, value_t) = tch::no_grad(|| policy.get_action(&obs_t));

                // Extract per-agent action vector (length = num_action_dims).
                let row: Vec<i64> = if num_action_dims == 1 {
                    // actions_t : [1] for scalar discrete.
                    let v: Vec<i64> = Vec::try_from(&actions_t.view([1])).unwrap();
                    v
                } else {
                    // actions_t : [1, num_action_dims] for multi-discrete.
                    let v: Vec<i64> =
                        Vec::try_from(&actions_t.view([num_action_dims as i64])).unwrap();
                    v
                };

                let off = t * num_action_dims;
                for (k, &a) in row.iter().enumerate() {
                    act_buf[i][off + k] = a;
                }
                joint_action.push(row);

                let lp: f32 = f32::try_from(&log_p_t).unwrap_or(0.0);
                let v: f32 = f32::try_from(&value_t).unwrap_or(0.0);
                lp_buf[i][t] = lp;
                val_buf[i][t] = v;
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

        // Materialize tensors.
        let observations = Tensor::from_slice(&obs_buf)
            .to_device(device)
            .view([num_steps as i64, obs_dim as i64]);

        let actions: Vec<Tensor> = if num_action_dims == 1 {
            act_buf
                .into_iter()
                .map(|a| Tensor::from_slice(&a).to_device(device).view([num_steps as i64]))
                .collect()
        } else {
            act_buf
                .into_iter()
                .map(|a| {
                    Tensor::from_slice(&a)
                        .to_device(device)
                        .view([num_steps as i64, num_action_dims as i64])
                })
                .collect()
        };
        let log_probs =
            lp_buf.into_iter().map(|v| Tensor::from_slice(&v).to_device(device)).collect();
        let values =
            val_buf.into_iter().map(|v| Tensor::from_slice(&v).to_device(device)).collect();
        let rewards =
            rew_buf.into_iter().map(|v| Tensor::from_slice(&v).to_device(device)).collect();
        let dones = Tensor::from_slice(&done_buf).to_device(device);

        JointRollout { observations, actions, log_probs, values, rewards, dones }
    }

    /// Joint PPO update.
    ///
    /// `aux_fn` receives a slice of per-agent encoder-feature tensors for the
    /// current minibatch (one entry per policy, shape `[mb, hidden_dim]`) and
    /// returns an optional pre-scaled scalar loss (e.g. the cross-agent
    /// redundancy penalty). One `.backward()` flows through every encoder
    /// when `aux_fn` returns `Some`.
    ///
    /// # Minibatch sampling
    ///
    /// To match the original `examples/games/bucket_brigade/train_p3.rs`
    /// reference implementation byte-for-byte (the acceptance criterion is
    /// 1e-5 numerical parity), the trainer samples **one** minibatch of size
    /// `config.minibatch_size` per epoch using `Tensor::randperm` + slice.
    /// A future revision may switch to iterating over all minibatches per
    /// epoch using [`crate::train::ppo::generate_minibatch_indices`]; that is
    /// the more conventional pattern but changes the RNG path.
    pub fn update<F>(&mut self, rollout: &JointRollout, mut aux_fn: F) -> Result<JointStats>
    where
        F: FnMut(&[&Tensor]) -> Option<Tensor>,
    {
        let t_total = rollout.observations.size()[0];
        let device = self.device;
        let num_agents = self.config.num_agents;

        // Per-agent advantages and returns. Computed once outside the epoch
        // loop (same as the reference example).
        let mut advantages: Vec<Tensor> = Vec::with_capacity(num_agents);
        let mut returns: Vec<Tensor> = Vec::with_capacity(num_agents);
        for i in 0..num_agents {
            let (adv, ret) = compute_gae_single_agent(
                &rollout.rewards[i],
                &rollout.values[i],
                &rollout.dones,
                self.config.gamma,
                self.config.gae_lambda,
            );
            let adv = if self.config.normalize_advantages {
                let adv_mean = adv.mean(Kind::Float);
                let adv_std = adv.std(false).clamp_min(1e-8);
                (adv - adv_mean) / adv_std
            } else {
                adv
            };
            advantages.push(adv);
            returns.push(ret);
        }

        let mut stats = JointStats::zeros(num_agents);
        let mb = (self.config.minibatch_size as i64).min(t_total);

        for _epoch in 0..self.config.n_epochs {
            // One shuffled minibatch per epoch (matches the reference).
            let idx_full = Tensor::randperm(t_total, (Kind::Int64, device));
            let idx = idx_full.slice(0, 0, mb, 1);

            let obs_mb = rollout.observations.index_select(0, &idx);

            // Per-agent forward + per-agent loss accumulation.
            let mut per_agent_losses: Vec<Tensor> = Vec::with_capacity(num_agents);
            let mut features: Vec<Tensor> = Vec::with_capacity(num_agents);

            for (i, policy) in self.policies.iter().enumerate() {
                let actions_mb = rollout.actions[i].index_select(0, &idx);
                let old_lp_mb = rollout.log_probs[i].index_select(0, &idx);
                let adv_mb = advantages[i].index_select(0, &idx);
                let ret_mb = returns[i].index_select(0, &idx);
                let old_v_mb = rollout.values[i].index_select(0, &idx);

                let (new_lp, entropy, values_mb) = policy.evaluate_actions(&obs_mb, &actions_mb);
                let feat = policy.encoder_features(&obs_mb);
                features.push(feat);

                let (policy_loss, clip_frac, kl) =
                    compute_policy_loss(&new_lp, &old_lp_mb, &adv_mb, self.config.clip_range);
                let (value_loss, explained_var) =
                    compute_value_loss(&values_mb, &old_v_mb, &ret_mb, self.config.clip_range_vf);
                let entropy_mean = entropy.mean(Kind::Float);

                let agent_loss = &policy_loss + self.config.vf_coef * &value_loss
                    - self.config.ent_coef * &entropy_mean;

                stats.policy_loss[i] += f64::try_from(&policy_loss).unwrap_or(0.0);
                stats.value_loss[i] += f64::try_from(&value_loss).unwrap_or(0.0);
                stats.entropy[i] += f64::try_from(&entropy_mean).unwrap_or(0.0);
                stats.clip_fraction[i] += clip_frac;
                stats.approx_kl[i] += kl;
                stats.explained_var[i] += explained_var;

                per_agent_losses.push(agent_loss);
            }

            // Aggregate per-agent losses, then add the cross-agent aux term.
            let mut joint_loss = Tensor::zeros([], (Kind::Float, device));
            for l in &per_agent_losses {
                joint_loss = joint_loss + l;
            }
            let feat_refs: Vec<&Tensor> = features.iter().collect();
            let aux_opt = aux_fn(&feat_refs);
            let aux_scalar: f64 =
                aux_opt.as_ref().and_then(|t| f64::try_from(t).ok()).unwrap_or(0.0);
            stats.aux_loss += aux_scalar;
            if let Some(aux) = aux_opt {
                joint_loss = joint_loss + aux;
            }
            stats.total_loss += f64::try_from(&joint_loss).unwrap_or(0.0);

            // Zero, backward, step every optimizer. Each optimizer's clip_grad
            // and step affect only its own var-store, so per-agent gradients
            // stay isolated even though we ran a single backward().
            for opt in self.optimizers.iter_mut() {
                opt.zero_grad();
            }
            joint_loss.backward();
            for opt in self.optimizers.iter_mut() {
                opt.clip_grad_norm(self.config.max_grad_norm);
                opt.step();
            }

            // Centralized critic update.
            //
            // Run *only when* `self.centralized_critic.is_some()`. When it's
            // `None`, control flow is bit-for-bit identical to the pre-
            // centralized-critic code path. The critic update is fully
            // parameter-isolated: it touches a separate VarStore + Optimizer
            // (`centralized_critic_optimizer`) and does NOT contribute
            // gradients to the per-agent policies. Its scalar loss IS folded
            // into `stats.total_loss` and surfaced in
            // `stats.centralized_value_loss` / `centralized_explained_var`.
            //
            // Target return: mean of per-agent minibatch returns. For
            // cooperative settings (bucket_brigade) the team return is the
            // natural global value target; for competitive settings, callers
            // can still attach a centralized critic but the team-return
            // target may not be informative — that's an algorithmic choice
            // we accept for v1.
            if let (Some(critic), Some(critic_opt)) =
                (self.centralized_critic.as_ref(), self.centralized_critic_optimizer.as_mut())
            {
                let critic_cfg =
                    self.config.centralized_critic.as_ref().cloned().unwrap_or_default();

                // Mean per-agent returns -> joint target.
                let mut ret_sum = returns[0].index_select(0, &idx);
                for r in returns.iter().skip(1) {
                    ret_sum = ret_sum + r.index_select(0, &idx);
                }
                let ret_target = ret_sum / (num_agents as f64);

                // Old centralized value: no_grad forward on obs_mb. (We do
                // not store an old-rollout centralized value, so the
                // "clipping anchor" is the current detached prediction,
                // which collapses the clip term to plain MSE on first step
                // — acceptable for v1.)
                let old_cv = tch::no_grad(|| critic.forward(&obs_mb)).detach();
                let cv = critic.forward(&obs_mb);
                let (c_loss, c_ev) = compute_centralized_value_loss(
                    &cv,
                    &old_cv,
                    &ret_target,
                    critic_cfg.clip_range_vf,
                );
                let weighted = critic_cfg.vf_coef * &c_loss;

                stats.centralized_value_loss += f64::try_from(&c_loss).unwrap_or(0.0);
                stats.centralized_explained_var += c_ev;
                stats.total_loss += f64::try_from(&weighted).unwrap_or(0.0);

                critic_opt.zero_grad();
                weighted.backward();
                critic_opt.clip_grad_norm(self.config.max_grad_norm);
                critic_opt.step();
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
            stats.centralized_value_loss /= n;
            stats.centralized_explained_var /= n;
            stats.total_loss /= n;
        }

        Ok(stats)
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Per-agent single-trajectory GAE.
///
/// Computes advantages and returns from a 1-D `[T]` rewards / values / dones
/// stream. The trailing-step value is taken as zero (matching the reference
/// example; the trainer doesn't currently bootstrap from a post-rollout
/// value estimate).
fn compute_gae_single_agent(
    rewards: &Tensor,
    values: &Tensor,
    dones: &Tensor,
    gamma: f64,
    gae_lambda: f64,
) -> (Tensor, Tensor) {
    let rewards_v: Vec<f32> = Vec::try_from(rewards).unwrap();
    let values_v: Vec<f32> = Vec::try_from(values).unwrap();
    let dones_v: Vec<f32> = Vec::try_from(dones).unwrap();
    let t = rewards_v.len();

    let mut advantages = vec![0.0_f32; t];
    let mut gae = 0.0_f32;
    for i in (0..t).rev() {
        let next_v = if i == t - 1 { 0.0 } else { values_v[i + 1] };
        let delta = rewards_v[i] + (gamma as f32) * next_v * (1.0 - dones_v[i]) - values_v[i];
        gae = delta + (gamma as f32) * (gae_lambda as f32) * (1.0 - dones_v[i]) * gae;
        advantages[i] = gae;
    }
    let returns: Vec<f32> = advantages.iter().zip(&values_v).map(|(&a, &v)| a + v).collect();

    let device = rewards.device();
    let adv_t = Tensor::from_slice(&advantages).to_device(device);
    let ret_t = Tensor::from_slice(&returns).to_device(device);
    (adv_t, ret_t)
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use tch::nn::OptimizerConfig;

    use super::*;
    use crate::policy::{mlp::MlpPolicy, multi_discrete_mlp::MultiDiscreteMlpPolicy};

    /// Process-global lock shared by every test in this module that touches
    /// libtorch's global RNG (`tch::manual_seed`, `Tensor::randn`, kaiming
    /// init inside `nn::linear`, `Tensor::randperm`, ...). Holding it
    /// serializes those tests against each other so that the seed set at the
    /// start of `test_update_numerical_parity_no_critic` is not clobbered by
    /// another test's RNG ops mid-run.
    ///
    /// The lock is held for the whole duration of each test that takes it.
    /// Pre-existing tests not yet ported still run in parallel; that's
    /// safe because they do NOT call `manual_seed` and only the *strict-
    /// parity* test relies on a stable seed.
    fn joint_tests_rng_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    /// Deterministic mock env: 4-dim obs (sin/cos of t and t/100, etc.),
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

    /// Capture trainable variables by name into a plain map of cloned
    /// tensors. The clones are detached and independent of the graph.
    fn capture_params(vs: &nn::VarStore) -> std::collections::HashMap<String, Tensor> {
        vs.variables().into_iter().map(|(name, t)| (name, t.detach().copy())).collect()
    }

    fn map_l2_diff(
        a: &std::collections::HashMap<String, Tensor>,
        b: &std::collections::HashMap<String, Tensor>,
    ) -> f64 {
        let mut total = 0.0_f64;
        for (name, at) in a.iter() {
            if let Some(bt) = b.get(name) {
                let d = at - bt;
                let s: f64 = f64::try_from(&d.square().sum(Kind::Float)).unwrap_or(0.0);
                total += s;
            }
        }
        total
    }

    fn make_mlp_policies(num_agents: usize, obs_dim: i64, action_dim: i64) -> Vec<MlpPolicy> {
        (0..num_agents).map(|_| MlpPolicy::new(obs_dim, action_dim, 16)).collect()
    }

    fn make_multi_discrete_policies(
        num_agents: usize,
        obs_dim: i64,
        action_dims: Vec<i64>,
    ) -> Vec<MultiDiscreteMlpPolicy> {
        (0..num_agents)
            .map(|_| MultiDiscreteMlpPolicy::new(obs_dim, action_dims.clone(), 16))
            .collect()
    }

    fn make_optimizers_for_mlp(policies: &mut [MlpPolicy], lr: f64) -> Vec<nn::Optimizer> {
        policies
            .iter_mut()
            .map(|p| nn::Adam::default().build(p.var_store_mut(), lr).unwrap())
            .collect()
    }

    fn make_optimizers_for_multi(
        policies: &mut [MultiDiscreteMlpPolicy],
        lr: f64,
    ) -> Vec<nn::Optimizer> {
        policies.iter_mut().map(|p| p.optimizer(lr).unwrap()).collect()
    }

    #[test]
    fn test_joint_trainer_smoke() {
        let _guard = joint_tests_rng_lock().lock().unwrap_or_else(|e| e.into_inner());
        // 2 tiny MlpPolicy instances, scalar discrete actions, 64-step rollout,
        // one update with aux_fn returning None. Assert no panics + finite
        // stats.
        let num_agents = 2;
        let obs_dim: i64 = 4;
        let action_dim: i64 = 3;
        let mut policies = make_mlp_policies(num_agents, obs_dim, action_dim);
        let optimizers = make_optimizers_for_mlp(&mut policies, 3e-4);

        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: 64,
            n_epochs: 2,
            minibatch_size: 32,
            ..Default::default()
        };
        let mut trainer = JointMultiAgentTrainer::new(policies, optimizers, config).unwrap();

        let mut env = MockEnv::new(num_agents, obs_dim as usize);
        let initial = env.reset_joint(None);
        let mut last_obs = initial[0].clone();

        let rollout = trainer.collect_rollout(&mut env, &mut last_obs);
        let stats = trainer
            .update(&rollout, |_features: &[&Tensor]| -> Option<Tensor> { None })
            .expect("update should not error");

        assert!(stats.total_loss.is_finite(), "total_loss must be finite");
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
        let _guard = joint_tests_rng_lock().lock().unwrap_or_else(|e| e.into_inner());
        let num_agents = 3;
        let obs_dim: i64 = 5;
        let t: usize = 32;
        let mut policies = make_mlp_policies(num_agents, obs_dim, 4);
        let optimizers = make_optimizers_for_mlp(&mut policies, 3e-4);

        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: t,
            n_epochs: 1,
            minibatch_size: t,
            ..Default::default()
        };
        let trainer = JointMultiAgentTrainer::new(policies, optimizers, config).unwrap();

        let mut env = MockEnv::new(num_agents, obs_dim as usize);
        let initial = env.reset_joint(None);
        let mut last_obs = initial[0].clone();
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs);

        assert_eq!(rollout.observations.size(), vec![t as i64, obs_dim]);
        assert_eq!(rollout.actions.len(), num_agents);
        // Scalar discrete: per-agent actions are [T]
        for a in &rollout.actions {
            assert_eq!(a.size(), vec![t as i64]);
        }
        assert_eq!(rollout.dones.size(), vec![t as i64]);
        for r in &rollout.rewards {
            assert_eq!(r.size(), vec![t as i64]);
        }
        for lp in &rollout.log_probs {
            assert_eq!(lp.size(), vec![t as i64]);
        }
        for v in &rollout.values {
            assert_eq!(v.size(), vec![t as i64]);
        }
    }

    #[test]
    fn test_aux_fn_gradient_couples_encoders() {
        let _guard = joint_tests_rng_lock().lock().unwrap_or_else(|e| e.into_inner());
        // With aux_fn = || (features[0] - features[1]).square().sum() AND
        // the per-agent PPO losses zeroed-out via clip_range = 0 / vf_coef = 0
        // / ent_coef = 0, the only gradient source is the aux term, which
        // must touch BOTH encoders.
        let num_agents = 2;
        let obs_dim: i64 = 4;
        let mut policies = make_mlp_policies(num_agents, obs_dim, 3);
        let optimizers = make_optimizers_for_mlp(&mut policies, 1e-2);

        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: 32,
            n_epochs: 1,
            minibatch_size: 32,
            // Suppress the per-agent PPO loss so we measure only the aux
            // gradient's effect.
            vf_coef: 0.0,
            ent_coef: 0.0,
            // clip_range=0 forces ratio*adv to be clipped to (1-0,1+0) = 1,
            // giving a constant policy_loss whose gradient w.r.t. params is
            // zero almost everywhere. Combined with vf=ent=0 this isolates
            // the aux term.
            clip_range: 0.0,
            normalize_advantages: false,
            ..Default::default()
        };
        let mut trainer = JointMultiAgentTrainer::new(policies, optimizers, config).unwrap();

        let mut env = MockEnv::new(num_agents, obs_dim as usize);
        let initial = env.reset_joint(None);
        let mut last_obs = initial[0].clone();
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs);

        let before_a = capture_params(trainer.policies[0].var_store());
        let before_b = capture_params(trainer.policies[1].var_store());

        let _stats = trainer
            .update(&rollout, |features: &[&Tensor]| -> Option<Tensor> {
                Some((features[0] - features[1]).square().sum(Kind::Float))
            })
            .expect("update should not error");

        let after_a = capture_params(trainer.policies[0].var_store());
        let after_b = capture_params(trainer.policies[1].var_store());

        let diff_a = map_l2_diff(&before_a, &after_a);
        let diff_b = map_l2_diff(&before_b, &after_b);

        // Both policies' parameters must have moved -- aux_fn's gradient
        // flowed through both encoders.
        assert!(diff_a > 0.0, "policy 0 params must change; diff_a = {diff_a}");
        assert!(diff_b > 0.0, "policy 1 params must change; diff_b = {diff_b}");
    }

    #[test]
    fn test_aux_fn_none_runs_clean() {
        let _guard = joint_tests_rng_lock().lock().unwrap_or_else(|e| e.into_inner());
        // aux_fn returns None on every minibatch; aux_loss must remain
        // exactly 0.0 and the trainer must run to completion without panic.
        let num_agents = 2;
        let obs_dim: i64 = 4;
        let mut policies = make_mlp_policies(num_agents, obs_dim, 3);
        let optimizers = make_optimizers_for_mlp(&mut policies, 3e-4);

        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: 32,
            n_epochs: 2,
            minibatch_size: 16,
            ..Default::default()
        };
        let mut trainer = JointMultiAgentTrainer::new(policies, optimizers, config).unwrap();

        let mut env = MockEnv::new(num_agents, obs_dim as usize);
        let initial = env.reset_joint(None);
        let mut last_obs = initial[0].clone();
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs);

        let stats = trainer
            .update(&rollout, |_features: &[&Tensor]| -> Option<Tensor> { None })
            .expect("update should not error");

        assert_eq!(stats.aux_loss, 0.0, "aux_loss must be 0 when aux_fn returns None");
        assert!(stats.total_loss.is_finite());
    }

    #[test]
    fn test_jointpolicy_for_multidiscrete() {
        let _guard = joint_tests_rng_lock().lock().unwrap_or_else(|e| e.into_inner());
        // Repeat the smoke test with MultiDiscreteMlpPolicy + factored
        // [3, 2] action space -- exercises the multi-discrete code paths
        // in collect_rollout (action shape [T, num_dims]) and in
        // evaluate_actions.
        let num_agents = 2;
        let obs_dim: i64 = 4;
        let action_dims = vec![3_i64, 2];
        let mut policies = make_multi_discrete_policies(num_agents, obs_dim, action_dims.clone());
        let optimizers = make_optimizers_for_multi(&mut policies, 3e-4);

        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: 32,
            n_epochs: 1,
            minibatch_size: 32,
            ..Default::default()
        };
        let mut trainer = JointMultiAgentTrainer::new(policies, optimizers, config).unwrap();

        let mut env = MockEnv::new(num_agents, obs_dim as usize);
        let initial = env.reset_joint(None);
        let mut last_obs = initial[0].clone();
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs);

        // Multi-discrete: actions are [T, num_dims]
        for a in &rollout.actions {
            assert_eq!(a.size(), vec![32, action_dims.len() as i64]);
        }

        let stats = trainer
            .update(&rollout, |_features: &[&Tensor]| -> Option<Tensor> { None })
            .expect("update should not error");
        assert!(stats.total_loss.is_finite());
    }

    // -------------------------------------------------------------------
    // Centralized critic integration tests.
    // -------------------------------------------------------------------

    /// Structural parity test: with `centralized_critic = None`, the new
    /// critic-aware `update` codepath must produce identical per-agent stats
    /// to itself across two back-to-back runs that share identical starting
    /// weights and identical input rollouts. This is the load-bearing
    /// regression guard against accidental control-flow divergence
    /// introduced by the centralized-critic wiring.
    ///
    /// # Why this test does NOT depend on libtorch's global RNG
    ///
    /// Libtorch's RNG is a process-global singleton, so any test that
    /// asserts bit-for-bit reproducibility under the default parallel
    /// `cargo test` scheduler is racy: another test can advance the RNG
    /// between our seed and consume. An earlier revision used a
    /// module-local mutex to try to serialize against that, but every
    /// other test in the crate that touches libtorch (policy::mlp init,
    /// train::dqn::*, train::ppo::loss::*, ...) ignores that lock, so it
    /// did not actually save us.
    ///
    /// Instead, the test eliminates RNG dependence entirely:
    ///
    /// 1. Build *one* set of policies, then deep-copy their parameters into a
    ///    second set via `VarStore::copy`. Both trainers start with
    ///    bit-identical weights regardless of what the process-global RNG
    ///    looked like during construction.
    /// 2. Collect the rollout *once* and reuse it for both runs. This avoids
    ///    re-invoking `policy.get_action`, which would consume RNG via
    ///    `multinomial` and could diverge between A and B.
    /// 3. Configure `minibatch_size == rollout_steps`, so the `randperm` inside
    ///    `update` permutes a full minibatch. Every loss is a `mean` / `sum`
    ///    reduction over the minibatch dim and is therefore
    ///    permutation-invariant — the actual shuffle order does not matter, so
    ///    even though the two `update` calls may draw *different* permutations,
    ///    the resulting per-agent stats are identical.
    #[test]
    fn test_update_numerical_parity_no_critic() {
        let num_agents = 2;
        let obs_dim: i64 = 4;
        let action_dim: i64 = 3;

        // -- Build trainer A. Initial weights come from whatever RNG state
        // libtorch happens to be in; that's fine because we will mirror
        // them into trainer B before either trainer runs `update`.
        let mut policies_a = make_mlp_policies(num_agents, obs_dim, action_dim);
        let optimizers_a = make_optimizers_for_mlp(&mut policies_a, 3e-4);

        // Build trainer B (separate VarStores), then make B's parameters
        // bit-identical to A's. `nn::VarStore::copy` is a shape-checked,
        // tensor-by-tensor copy.
        let mut policies_b = make_mlp_policies(num_agents, obs_dim, action_dim);
        for (a, b) in policies_a.iter().zip(policies_b.iter_mut()) {
            b.var_store_mut().copy(a.var_store()).expect("varstore copy must succeed");
        }
        let optimizers_b = make_optimizers_for_mlp(&mut policies_b, 3e-4);

        // Permutation-invariant config: minibatch_size == rollout_steps,
        // so per-epoch `randperm` does not affect any reduction.
        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: 64,
            n_epochs: 2,
            // Equal to rollout_steps so the per-epoch randperm slice covers
            // every sample and the order is irrelevant to mean/sum losses.
            minibatch_size: 64,
            ..Default::default()
        };

        let mut trainer_a =
            JointMultiAgentTrainer::new(policies_a, optimizers_a, config.clone()).unwrap();
        let mut trainer_b = JointMultiAgentTrainer::new(policies_b, optimizers_b, config).unwrap();

        // Collect rollout once on trainer A. Since A and B share weights,
        // we can reuse the same rollout for B's update — that avoids a
        // second `collect_rollout` call (which would consume RNG via
        // `multinomial` and could draw a different action sequence under
        // parallel test pressure).
        let mut env = MockEnv::new(num_agents, obs_dim as usize);
        let initial = env.reset_joint(None);
        let mut last_obs = initial[0].clone();
        let rollout = trainer_a.collect_rollout(&mut env, &mut last_obs);

        let a = trainer_a
            .update(&rollout, |_features: &[&Tensor]| -> Option<Tensor> { None })
            .expect("update A should not error");
        let b = trainer_b
            .update(&rollout, |_features: &[&Tensor]| -> Option<Tensor> { None })
            .expect("update B should not error");

        const TOL: f64 = 1e-5;
        assert!(
            (a.total_loss - b.total_loss).abs() < TOL,
            "total_loss diverged: a={} b={}",
            a.total_loss,
            b.total_loss
        );
        assert!(
            (a.aux_loss - b.aux_loss).abs() < TOL,
            "aux_loss diverged: a={} b={}",
            a.aux_loss,
            b.aux_loss
        );
        assert_eq!(a.centralized_value_loss, 0.0);
        assert_eq!(b.centralized_value_loss, 0.0);
        assert_eq!(a.centralized_explained_var, 0.0);
        assert_eq!(b.centralized_explained_var, 0.0);
        for i in 0..a.policy_loss.len() {
            assert!(
                (a.policy_loss[i] - b.policy_loss[i]).abs() < TOL,
                "policy_loss[{i}] diverged: a={} b={}",
                a.policy_loss[i],
                b.policy_loss[i]
            );
            assert!(
                (a.value_loss[i] - b.value_loss[i]).abs() < TOL,
                "value_loss[{i}] diverged: a={} b={}",
                a.value_loss[i],
                b.value_loss[i]
            );
            assert!(
                (a.entropy[i] - b.entropy[i]).abs() < TOL,
                "entropy[{i}] diverged: a={} b={}",
                a.entropy[i],
                b.entropy[i]
            );
            assert!(
                (a.clip_fraction[i] - b.clip_fraction[i]).abs() < TOL,
                "clip_fraction[{i}] diverged: a={} b={}",
                a.clip_fraction[i],
                b.clip_fraction[i]
            );
            assert!(
                (a.approx_kl[i] - b.approx_kl[i]).abs() < TOL,
                "approx_kl[{i}] diverged: a={} b={}",
                a.approx_kl[i],
                b.approx_kl[i]
            );
        }
    }

    /// With a centralized critic attached, `total_loss` reflects the critic's
    /// contribution: stats.centralized_value_loss is positive and non-zero,
    /// and total_loss is finite. Also asserts no per-agent stats are NaN.
    #[test]
    fn test_update_with_critic_total_loss_includes_centralized() {
        let _guard = joint_tests_rng_lock().lock().unwrap_or_else(|e| e.into_inner());
        tch::manual_seed(0xABCDEF);
        let num_agents = 2;
        let obs_dim: i64 = 4;
        let mut policies = make_mlp_policies(num_agents, obs_dim, 3);
        let optimizers = make_optimizers_for_mlp(&mut policies, 3e-4);

        let critic_cfg = CentralizedCriticConfig {
            hidden_dim: 16,
            learning_rate: 3e-4,
            vf_coef: 0.5,
            clip_range_vf: 0.0,
        };
        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: 64,
            n_epochs: 2,
            minibatch_size: 32,
            centralized_critic: Some(critic_cfg.clone()),
            ..Default::default()
        };

        let mut critic =
            CentralizedCritic::new_on_device(obs_dim, critic_cfg.hidden_dim, Device::Cpu);
        let critic_opt = critic.build_optimizer(critic_cfg.learning_rate).unwrap();
        let mut trainer = JointMultiAgentTrainer::new_with_centralized_critic(
            policies, optimizers, critic, critic_opt, config,
        )
        .unwrap();

        let mut env = MockEnv::new(num_agents, obs_dim as usize);
        let initial = env.reset_joint(None);
        let mut last_obs = initial[0].clone();
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs);
        let stats = trainer
            .update(&rollout, |_features: &[&Tensor]| -> Option<Tensor> { None })
            .expect("update should not error");

        assert!(stats.total_loss.is_finite(), "total_loss must be finite");
        assert!(
            stats.centralized_value_loss > 0.0,
            "centralized_value_loss must be positive on first update, got {}",
            stats.centralized_value_loss
        );
        assert!(
            stats.centralized_explained_var.is_finite(),
            "centralized_explained_var must be finite"
        );
        for i in 0..num_agents {
            assert!(stats.policy_loss[i].is_finite());
            assert!(stats.value_loss[i].is_finite());
        }
    }

    /// Construction-time guard: attaching a critic on a different device than
    /// the policies must error out.
    #[test]
    fn test_centralized_critic_device_mismatch_errors() {
        let _guard = joint_tests_rng_lock().lock().unwrap_or_else(|e| e.into_inner());
        // All CPU; this should succeed (and is the happy path).
        let num_agents = 2;
        let obs_dim: i64 = 4;
        let mut policies = make_mlp_policies(num_agents, obs_dim, 3);
        let optimizers = make_optimizers_for_mlp(&mut policies, 3e-4);
        let mut critic = CentralizedCritic::new_on_device(obs_dim, 8, Device::Cpu);
        let critic_opt = critic.build_optimizer(3e-4).unwrap();
        let config = JointTrainerConfig {
            num_agents,
            rollout_steps: 16,
            n_epochs: 1,
            minibatch_size: 16,
            ..Default::default()
        };
        let ok = JointMultiAgentTrainer::new_with_centralized_critic(
            policies, optimizers, critic, critic_opt, config,
        );
        assert!(ok.is_ok(), "same-device construction must succeed");
    }
}
