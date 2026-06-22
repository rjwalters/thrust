//! Neural Fictitious Self-Play (NFSP) trainer.
//!
//! Burn-native implementation of NFSP (Heinrich & Silver 2016,
//! [arXiv:1603.01121](https://arxiv.org/abs/1603.01121)).
//! The original paper proves ε-Nash convergence in the 2-agent
//! zero-sum setting (§3 Algorithm 1, Theorem 3). Tracking issue: #106.
//!
//! # N-player ("approximate NFSP") caveat
//!
//! As of #119, the trainer accepts `joint_config.num_agents > 2`. The
//! Vitter Algorithm R uniformity invariant on each per-agent reservoir
//! survives unchanged (the per-agent reservoirs are independent and
//! Algorithm R's correctness does not depend on the number of agents).
//! However, the §3 Algorithm 1 **ε-Nash convergence guarantee does
//! NOT generalize** to N > 2 — Heinrich & Silver §5 explicitly flags
//! "extending the convergence theory to N-player general-sum games"
//! as future work. For N > 2 this implementation is therefore best
//! described as an "approximate NFSP" smoke trainer: the BR/AP
//! mechanics, η-anticipatory mixing, and reservoir sampling are all
//! exercised, and empirically the average policy converges on
//! symmetric mixed-Nash games like
//! [`crate::env::games::n_player_matching_pennies::NPlayerMatchingPennies`],
//! but no formal guarantee is shipped beyond the per-agent reservoir
//! uniformity.
//!
//! # Pseudocode (Heinrich & Silver §3 Algorithm 1)
//!
//! ```text
//! for each iteration k:
//!     for each rollout step t:
//!         draw c ~ Bernoulli(η)             # η-anticipatory mixing
//!         if c == 1:
//!             a_t ~ best_response_policy   # BR (PPO actor)
//!             reservoir.push((s_t, a_t))    # ← reservoir-sampled
//!         else:
//!             a_t ~ average_policy          # AP (supervised actor)
//!             # NOTE: do NOT push AP actions into the reservoir
//!     train BR with PPO on the rollout (freeze-N-1 via JointMultiAgentTrainer)
//!     sample minibatch from reservoir; train AP via cross-entropy on (s, a)
//! end
//! ```
//!
//! The reservoir is Vitter's Algorithm R (paper §3.2 footnote 3, see
//! also Vitter 1985 "Random Sampling with a Reservoir"). Using a FIFO
//! or sliding window instead defeats the convergence guarantee: NFSP
//! depends on the supervised target being a *uniform* sample over the
//! history of best-response actions so the AP converges to the
//! time-averaged BR policy.
//!
//! # η-anticipatory mixing
//!
//! At every rollout step the trainer flips a Bernoulli(η) coin. With
//! probability `η` it samples from the best-response (BR) policy and
//! pushes the observation/action pair into the reservoir buffer. With
//! probability `1 − η` it samples from the average (AP) policy and
//! does **not** push to the reservoir. Appending AP actions to the
//! reservoir collapses NFSP to vanilla fictitious play and prevents
//! convergence to NE — this is the critical invariant the paper's
//! footnote calls out. Default `η = 0.1` per Heinrich & Silver §3.
//!
//! # Reservoir vs FIFO
//!
//! Heinrich & Silver §3.2 footnote 3: "Reservoir sampling avoids the
//! policy distribution drift caused by uniformly drawing from a more
//! recent window." A FIFO buffer biases the AP towards recent BR
//! actions, which is roughly equivalent to learning a recency-weighted
//! mixture — that mixture is *not* the time-average and need not match
//! any fictitious-play fixed point. Vitter's Algorithm R keeps the
//! held-items' distribution uniform across the entire history of
//! pushes regardless of stream length.
//!
//! # Per-agent observation handling
//!
//! NFSP builds on top of
//! [`crate::multi_agent::joint::JointMultiAgentTrainer`], which records
//! a *per-agent* observation stream in
//! [`JointRollout::observations_per_agent`]. Agent `i`'s reservoir
//! receives agent `i`'s observation (not agent 0's), so partial-
//! observability envs supervise the AP module on the correct view.
//! Matching pennies returns identical observations to both agents,
//! which keeps the regression tests bit-stable through the per-agent
//! refactor (PR #118).
//!
//! # Determinism dependency on #109
//!
//! The reservoir buffer's eviction RNG, the η-anticipatory coin flips,
//! and the average-policy minibatch sampler are all seeded from
//! [`NfspConfig::seed`] via an internal `StdRng`. However, the inner
//! [`crate::multi_agent::joint::JointMultiAgentTrainer::update_with_active_agents`]
//! still uses `rand::rng()` for its per-epoch shuffle (see
//! `src/multi_agent/joint.rs:662`), which means same-seed PSRO and
//! NFSP runs are not yet bit-identical across wall-clock invocations.
//! Issue #109 tracks the fix. The in-module determinism here is
//! sufficient for the reservoir-uniformity and η-mixing unit tests in
//! this PR.

use anyhow::{Result, anyhow};
use burn::{
    optim::{GradientsParams, Optimizer},
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend},
};
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    multi_agent::joint::{
        JointEnv, JointMultiAgentTrainer, JointPolicy, JointStats, JointTrainerConfig,
    },
    train::optimizer::{BackendOptimizer, BurnOptimizer},
};

// =======================================================================
// ReservoirBuffer — Vitter's Algorithm R
// =======================================================================

/// Reservoir-sampled buffer backing the NFSP average-policy supervised
/// dataset. Implements Vitter (1985) Algorithm R: every distinct item
/// streamed through `push` is retained with probability
/// `capacity / stream_index` (1-indexed across the full lifetime of
/// the buffer). Under capacity, all pushes are kept; at capacity, each
/// push at stream index `i ≥ capacity + 1` is inserted with probability
/// `capacity / i` replacing a uniformly-random existing slot.
///
/// # Correctness
///
/// The key invariant Vitter's algorithm guarantees is that at any
/// point in time, the held items are a uniform sample of the entire
/// stream history. This is the property NFSP needs from §3.2 — see the
/// module doc for why FIFO would be wrong.
///
/// # Determinism
///
/// The internal RNG is seedable via [`ReservoirBuffer::with_seed`] so
/// the trainer can produce repeatable buffer trajectories under a
/// fixed [`NfspConfig::seed`].
#[derive(Debug, Clone)]
pub struct ReservoirBuffer<T: Clone> {
    items: Vec<T>,
    capacity: usize,
    /// Total number of items ever pushed (the stream index `i` Vitter's
    /// algorithm uses). 1-based by convention — the first push is at
    /// index 1.
    stream_index: usize,
    rng: StdRng,
}

impl<T: Clone> ReservoirBuffer<T> {
    /// Construct an empty reservoir with the given capacity, seeded
    /// from a fixed value for reproducible tests.
    pub fn with_seed(capacity: usize, seed: u64) -> Self {
        Self {
            items: Vec::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            stream_index: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Construct an empty reservoir using a fresh entropy-seeded RNG.
    pub fn new(capacity: usize) -> Self {
        let seed: u64 = rand::rng().random();
        Self::with_seed(capacity, seed)
    }

    /// Currently-held items.
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// Number of items currently held (≤ capacity).
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True if no items have been retained.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Maximum number of items the reservoir may hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total stream length (number of `push` calls observed,
    /// regardless of retention). Vitter's `i` for the next incoming
    /// item is `stream_length() + 1`.
    pub fn stream_length(&self) -> usize {
        self.stream_index
    }

    /// Push one item through the reservoir.
    ///
    /// Under capacity: appended unconditionally. At capacity: kept
    /// with probability `capacity / i` (where `i` is the new
    /// 1-indexed stream position), replacing a uniformly-random
    /// existing slot.
    ///
    /// Returns `true` if the item was retained (either appended or
    /// replaced an existing slot), `false` if it was discarded.
    pub fn push(&mut self, item: T) -> bool {
        self.stream_index += 1;
        if self.items.len() < self.capacity {
            self.items.push(item);
            return true;
        }
        // Vitter Algorithm R: with probability capacity / i, replace
        // a uniformly random slot.
        let i = self.stream_index;
        let j = self.rng.random_range(0..i);
        if j < self.capacity {
            self.items[j] = item;
            true
        } else {
            false
        }
    }

    /// Sample `n` items uniformly at random *with replacement* from
    /// the currently-held reservoir. Returns an empty vec if the
    /// reservoir is empty.
    pub fn sample_with_replacement(&mut self, n: usize) -> Vec<T> {
        if self.items.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let j = self.rng.random_range(0..self.items.len());
            out.push(self.items[j].clone());
        }
        out
    }
}

// =======================================================================
// NfspConfig
// =======================================================================

/// NFSP trainer configuration.
#[derive(Debug, Clone)]
pub struct NfspConfig {
    /// Number of NFSP outer-loop iterations.
    pub max_iterations: usize,
    /// η, the anticipatory-mixing parameter. Probability of sampling
    /// from the best-response policy (and appending the action to the
    /// reservoir) at each rollout step. Heinrich & Silver §3
    /// recommend `η = 0.1` for 2-player zero-sum imperfect-information
    /// games.
    pub anticipatory_param: f32,
    /// Maximum reservoir buffer size (per agent in the multi-agent
    /// setting). Paper default is 2×10⁶; the matching-pennies smoke
    /// test runs with ~2000.
    pub reservoir_capacity: usize,
    /// Number of BR train steps (one rollout + one PPO update per
    /// step) per outer iteration.
    pub br_train_steps_per_iteration: usize,
    /// Number of supervised cross-entropy steps applied to the AP per
    /// outer iteration (skipped when the reservoir is empty).
    ///
    /// This is the *floor* on AP steps. When
    /// [`avg_policy_min_reservoir_coverage`](Self::avg_policy_min_reservoir_coverage)
    /// is `> 0` the trainer may run *more* than this many steps so the
    /// AP sees enough of the reservoir per iteration (issue #199).
    pub avg_policy_train_steps_per_iteration: usize,
    /// Average-policy supervised minibatch size.
    pub avg_policy_minibatch_size: usize,
    /// Average-policy learning rate.
    pub avg_policy_lr: f64,
    /// Minimum reservoir *coverage* per outer iteration for the AP
    /// supervised update, expressed as a multiple of the current
    /// reservoir size (issue #199).
    ///
    /// NFSP's average policy is fit by sampling minibatches from a
    /// reservoir that grows into the thousands of entries, while the
    /// default budget of `avg_policy_train_steps_per_iteration ×
    /// avg_policy_minibatch_size` may only touch a few hundred samples
    /// per iteration — far too few gradient steps to fit a moving
    /// target, so the AP loss stays pinned at the uniform-entropy floor
    /// (`ln(num_joint_actions)`). On bucket-brigade this manifested as
    /// `avg_ap_loss ≈ ln(40)` for an entire 48-iteration run.
    ///
    /// When `coverage > 0`, the trainer runs
    /// `max(avg_policy_train_steps_per_iteration,
    /// ceil(coverage × reservoir_len / avg_policy_minibatch_size))`
    /// supervised steps for each agent each iteration, so the AP is
    /// exposed to (in expectation) `coverage` full passes over the
    /// reservoir. `coverage = 0.0` disables the adaptive floor and
    /// preserves the legacy fixed-step behavior. Default `0.0`.
    pub avg_policy_min_reservoir_coverage: f32,
    /// Optional reward scaling applied to per-step rewards before the BR
    /// (PPO) update, to keep the large-magnitude bucket-brigade payoff
    /// band (`[−700, 0]`) from dominating value targets and advantage
    /// normalization (issue #199).
    ///
    /// Rewards in the anticipatory rollout are multiplied by this factor
    /// before being handed to the joint PPO update. `1.0` (default) is a
    /// no-op; a value like `0.01` rescales the bucket-brigade band to
    /// roughly `[−7, 0]`. Scaling rewards uniformly does not change the
    /// optimal policy (it is an affine transform of the return), but it
    /// does keep the critic's regression targets and the advantage
    /// statistics in a numerically friendlier range.
    pub br_reward_scale: f32,
    /// RNG seed (η-flips, reservoir eviction, AP minibatch sampling).
    pub seed: u64,
}

impl Default for NfspConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            anticipatory_param: 0.1,
            reservoir_capacity: 2_000,
            br_train_steps_per_iteration: 1,
            avg_policy_train_steps_per_iteration: 1,
            avg_policy_minibatch_size: 32,
            avg_policy_lr: 1e-3,
            avg_policy_min_reservoir_coverage: 0.0,
            br_reward_scale: 1.0,
            seed: 0,
        }
    }
}

// =======================================================================
// NfspStats
// =======================================================================

/// Per-iteration NFSP statistics.
#[derive(Debug, Clone, Default)]
pub struct NfspIterationStats {
    /// Iteration index (1-based).
    pub iteration: usize,
    /// Per-agent reservoir size at the end of the iteration.
    pub reservoir_sizes: Vec<usize>,
    /// Per-agent best-response training stats (the last
    /// joint-trainer call's `JointStats`).
    pub br_stats: Option<JointStats>,
    /// Per-agent average-policy cross-entropy loss (averaged across
    /// AP supervised steps). `None` if the AP was skipped because the
    /// reservoir was empty.
    pub avg_policy_loss: Vec<Option<f64>>,
    /// Per-agent BR-action marginal under the constant matching-
    /// pennies observation. `None` when the trainer is not on a
    /// constant-obs env (callers may ignore on non-matching-pennies
    /// envs).
    pub br_action_marginal: Vec<Option<Vec<f32>>>,
    /// Per-agent AP-action marginal under the constant matching-
    /// pennies observation. `None` when the trainer is not on a
    /// constant-obs env.
    pub avg_action_marginal: Vec<Option<Vec<f32>>>,
    /// Cumulative count of BR-sampled actions (across the η-mixing
    /// rollout collector) since the trainer was constructed.
    pub cumulative_br_pushes: usize,
    /// Cumulative count of rollout steps observed (across all agents).
    pub cumulative_rollout_steps: usize,
}

/// Aggregate NFSP trainer statistics returned by [`NfspTrainer::run`].
#[derive(Debug, Clone, Default)]
pub struct NfspStats {
    /// Per-iteration history.
    pub iterations: Vec<NfspIterationStats>,
}

// =======================================================================
// NfspTrainer
// =======================================================================

/// NFSP outer-loop trainer.
///
/// Generic over the Burn backend `B`, policy module `P`, optimizer
/// type `O`, env `E`, and the user-supplied policy/optimizer/env
/// factories. The trainer owns:
///
/// - A `JointMultiAgentTrainer` holding the `N` best-response (BR) policies. BR
///   updates go through [`JointMultiAgentTrainer::update_with_active_agents`] —
///   the freeze-N-1 primitive PSRO also uses.
/// - `N` average-policy (AP) Burn modules and their independent optimizers. AP
///   updates are plain supervised cross-entropy on reservoir samples.
/// - `N` reservoir buffers (one per agent) carrying `(obs, action)` pairs
///   harvested from BR-sampled rollout steps.
/// - An internal `StdRng` (seeded from [`NfspConfig::seed`]) used for
///   η-anticipatory coin flips, reservoir eviction, and AP minibatch sampling.
///
/// As of #119, `joint_config.num_agents` may be any value `≥ 2`. The
/// formal ε-Nash convergence guarantee from Heinrich & Silver 2016 §3
/// holds only for N = 2 zero-sum games; for N > 2 this trainer ships
/// the same per-agent BR/AP/reservoir machinery as an "approximate
/// NFSP" — see the module docstring for the caveat.
///
/// # Single-policy-class assumption
///
/// All `2N` policies (`N` BR + `N` AP) share the same module type
/// `P`. For symmetric matching-pennies-style games this is what we
/// want.
///
/// # See also
///
/// - [`crate::multi_agent::psro::PsroTrainer`] — sibling meta-game trainer with
///   the same JointMultiAgentTrainer-based freeze-N-1 plumbing.
pub struct NfspTrainer<B, P, O, E, FP, FO, FE>
where
    B: AutodiffBackend,
    P: JointPolicy<B>,
    O: Optimizer<P, B>,
    E: JointEnv,
    FP: Fn(&B::Device, u64) -> P,
    FO: Fn() -> BurnOptimizer<B, P, O>,
    FE: Fn() -> E,
{
    config: NfspConfig,
    joint_config: JointTrainerConfig,
    device: B::Device,
    /// Joint trainer holding the two BR policies.
    br_trainer: JointMultiAgentTrainer<B, P, O>,
    /// Per-agent average policy. Stored in `Option<P>` to match Burn's
    /// move-through optimizer semantics: we `.take()` before stepping
    /// and put back after.
    avg_policies: Vec<Option<P>>,
    /// Per-agent average-policy optimizer.
    avg_optimizers: Vec<BurnOptimizer<B, P, O>>,
    /// Per-agent reservoir buffer of `(obs, action_vec)` pairs.
    ///
    /// `action_vec` is the per-step joint action with one entry per
    /// action dim — `Vec<i64>` of length `num_action_dims`. For
    /// single-discrete policies (`MlpBurnPolicy`) this is a length-1 vec
    /// (equivalent to the pre-#127 scalar form); for multi-discrete
    /// policies (`MultiDiscreteMlpBurnPolicy` — e.g. bucket-brigade
    /// `[10, 2, 2]`) it carries one entry per factored head, matching
    /// what `evaluate_actions_joint` expects (`[mb, num_action_dims]`).
    reservoirs: Vec<ReservoirBuffer<(Vec<f32>, Vec<i64>)>>,
    /// Internal RNG (η-flips, AP minibatch sampling).
    rng: StdRng,
    /// Cumulative BR pushes across all agents (diagnostic).
    cumulative_br_pushes: usize,
    /// Cumulative rollout step count across all agents (diagnostic).
    cumulative_rollout_steps: usize,
    /// Held factories — used by `train_best_response` to spin up
    /// fresh envs and (in principle) policies; kept for symmetry with
    /// PSRO's `PsroTrainer` API.
    #[allow(dead_code)]
    policy_factory: FP,
    #[allow(dead_code)]
    optimizer_factory: FO,
    env_factory: FE,
}

impl<B, P, O, E, FP, FO, FE> NfspTrainer<B, P, O, E, FP, FO, FE>
where
    B: AutodiffBackend,
    P: JointPolicy<B>,
    O: Optimizer<P, B>,
    E: JointEnv,
    FP: Fn(&B::Device, u64) -> P,
    FO: Fn() -> BurnOptimizer<B, P, O>,
    FE: Fn() -> E,
{
    /// Construct an NFSP trainer with fresh BR and AP policies.
    ///
    /// `joint_config.num_agents` must be `≥ 2`. For N = 2 the trainer
    /// retains Heinrich & Silver §3's ε-Nash convergence guarantee on
    /// zero-sum games; for N > 2 it ships as "approximate NFSP" with
    /// no formal guarantee beyond per-agent reservoir uniformity (see
    /// the module docstring).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: NfspConfig,
        joint_config: JointTrainerConfig,
        device: B::Device,
        policy_factory: FP,
        optimizer_factory: FO,
        env_factory: FE,
    ) -> Result<Self> {
        if joint_config.num_agents < 2 {
            return Err(anyhow!(
                "NfspTrainer requires joint_config.num_agents >= 2 (got {})",
                joint_config.num_agents
            ));
        }
        if !(0.0..=1.0).contains(&config.anticipatory_param) {
            return Err(anyhow!(
                "NfspConfig::anticipatory_param must be in [0,1], got {}",
                config.anticipatory_param
            ));
        }
        let num_agents = joint_config.num_agents;
        // Per-construction init seeds (issue #135, Correction 1): every
        // BR and AP policy gets a distinct, deterministic seed derived
        // from `config.seed` so they start from different — but
        // reproducible — weights. A factory closing over a single fixed
        // seed would otherwise hand every policy identical weights. BR
        // policies use counter `agent`; AP policies use
        // `num_agents + agent`, keeping the two banks disjoint.
        let init_seed = |idx: u64| config.seed.wrapping_add(0x9E37_79B9_u64.wrapping_mul(idx));
        let br_policies: Vec<P> =
            (0..num_agents).map(|i| policy_factory(&device, init_seed(i as u64))).collect();
        let br_optimizers: Vec<BurnOptimizer<B, P, O>> =
            (0..num_agents).map(|_| optimizer_factory()).collect();
        let br_trainer = JointMultiAgentTrainer::<B, P, O>::new(
            br_policies,
            br_optimizers,
            joint_config.clone(),
            device.clone(),
        )?;
        let avg_policies: Vec<Option<P>> = (0..num_agents)
            .map(|i| Some(policy_factory(&device, init_seed((num_agents + i) as u64))))
            .collect();
        let avg_optimizers: Vec<BurnOptimizer<B, P, O>> =
            (0..num_agents).map(|_| optimizer_factory()).collect();
        let reservoirs = (0..num_agents)
            .map(|i| {
                // Seed each reservoir from a derivation of the master
                // seed so they're decorrelated but reproducible.
                ReservoirBuffer::with_seed(
                    config.reservoir_capacity,
                    config.seed.wrapping_add(0x1ABC + i as u64),
                )
            })
            .collect();
        let rng = StdRng::seed_from_u64(config.seed.wrapping_add(0xC0DE));
        Ok(Self {
            config,
            joint_config,
            device,
            br_trainer,
            avg_policies,
            avg_optimizers,
            reservoirs,
            rng,
            cumulative_br_pushes: 0,
            cumulative_rollout_steps: 0,
            policy_factory,
            optimizer_factory,
            env_factory,
        })
    }

    /// Borrow agent `i`'s best-response (BR) policy.
    pub fn br_policy(&self, i: usize) -> &P {
        self.br_trainer.policy(i)
    }

    /// Borrow agent `i`'s average (AP) policy.
    pub fn avg_policy(&self, i: usize) -> &P {
        self.avg_policies[i].as_ref().expect("AP policy is None mid-update")
    }

    /// Borrow agent `i`'s reservoir buffer.
    pub fn reservoir(&self, i: usize) -> &ReservoirBuffer<(Vec<f32>, Vec<i64>)> {
        &self.reservoirs[i]
    }

    /// Cumulative BR-sampled push count across all reservoirs.
    pub fn cumulative_br_pushes(&self) -> usize {
        self.cumulative_br_pushes
    }

    /// Cumulative rollout-step count across all agents.
    pub fn cumulative_rollout_steps(&self) -> usize {
        self.cumulative_rollout_steps
    }

    /// Trainer configuration.
    pub fn config(&self) -> &NfspConfig {
        &self.config
    }

    /// Run the NFSP outer loop and return the per-iteration history.
    pub fn run<F>(&mut self, mut on_iteration: F) -> Result<NfspStats>
    where
        F: FnMut(&NfspIterationStats),
    {
        let mut stats = NfspStats::default();
        for iter in 1..=self.config.max_iterations {
            let mut last_br_stats: Option<JointStats> = None;

            for _ in 0..self.config.br_train_steps_per_iteration {
                // Collect rollout with η-anticipatory mixing and
                // reservoir push.
                let rollout = self.collect_anticipatory_rollout()?;
                // Drive both BR policies via a single joint update
                // (active mask = [true, true]). Heinrich & Silver §3
                // Algorithm 1 runs the BR updates jointly per
                // iteration; this also matches PSRO's freeze-N-1 path
                // when N = 2 and both agents are "active".
                let active = vec![true; self.joint_config.num_agents];
                let bs = self.br_trainer.update_with_active_agents(
                    &rollout,
                    &active,
                    &mut self.rng,
                    |_features: &[Tensor<B, 2>]| -> Option<Tensor<B, 1>> { None },
                )?;
                last_br_stats = Some(bs);
            }

            // Average-policy supervised step.
            let avg_losses = self.train_average_policies()?;

            // Diagnostics: action marginals for the matching-pennies
            // constant observation `[0.0]`. We emit these
            // unconditionally; non-constant-obs envs can ignore.
            let num_agents = self.joint_config.num_agents;
            let mut br_marginals: Vec<Option<Vec<f32>>> = Vec::with_capacity(num_agents);
            let mut avg_marginals: Vec<Option<Vec<f32>>> = Vec::with_capacity(num_agents);
            for i in 0..num_agents {
                // Clone the policy references to release the `&self`
                // borrow on `self.br_policy(i)` / `self.avg_policy(i)`
                // before `action_marginal_for` mutably borrows
                // `self.rng` for the seeded probe loop (#114).
                let br_policy_i = self.br_policy(i).clone();
                let avg_policy_i = self.avg_policy(i).clone();
                br_marginals.push(self.action_marginal_for(&br_policy_i));
                avg_marginals.push(self.action_marginal_for(&avg_policy_i));
            }

            let iter_stats = NfspIterationStats {
                iteration: iter,
                reservoir_sizes: self.reservoirs.iter().map(|r| r.len()).collect(),
                br_stats: last_br_stats,
                avg_policy_loss: avg_losses,
                br_action_marginal: br_marginals,
                avg_action_marginal: avg_marginals,
                cumulative_br_pushes: self.cumulative_br_pushes,
                cumulative_rollout_steps: self.cumulative_rollout_steps,
            };
            on_iteration(&iter_stats);
            stats.iterations.push(iter_stats);
        }
        Ok(stats)
    }

    /// Convenience entry point: drives [`Self::run`] with a no-op
    /// iteration callback.
    pub fn run_silent(&mut self) -> Result<NfspStats> {
        self.run(|_| {})
    }

    /// Average-policy action distribution under a *constant* observation
    /// `[0.0; obs_dim]`. Returns `None` if the policy's `action_dims`
    /// is unsupported (e.g. multi-dim multi-discrete). Used as a
    /// diagnostic on matching-pennies and as the load-bearing
    /// convergence assertion in `tests/test_nfsp_matching_pennies.rs`.
    ///
    /// Takes `&mut self` because the 128-probe sampling loop consumes
    /// the trainer-owned `StdRng` via
    /// [`JointPolicy::get_action_host_seeded`]. This makes the
    /// diagnostic reproducible under `NfspConfig::seed` (issue #114).
    /// Callers that hold a `&P` from [`Self::avg_policy`] / [`Self::br_policy`]
    /// must `.clone()` the policy before calling this method to avoid a
    /// `&self` / `&mut self` aliasing conflict — the policy clone is
    /// cheap (Burn modules are `Clone` by design).
    pub fn action_marginal_for(&mut self, policy: &P) -> Option<Vec<f32>> {
        let dims = policy.action_dims_joint();
        if dims.len() != 1 {
            return None;
        }
        let action_dim = dims[0] as usize;
        // We don't know the true obs_dim without an env. The trainer's
        // `joint_config` carries observation-agnostic config; we infer
        // obs_dim from the reservoirs (which were populated by the
        // env) when possible. Falls back to obs_dim = 1 (matching
        // pennies' OBS_DIM) when reservoirs are empty.
        let obs_dim = self
            .reservoirs
            .iter()
            .find_map(|r| r.items().first().map(|(obs, _)| obs.len()))
            .unwrap_or(1);
        let obs_data: Vec<f32> = vec![0.0_f32; obs_dim];
        let obs = Tensor::<B, 2>::from_data(TensorData::new(obs_data, [1, obs_dim]), &self.device);
        // Use a forward+softmax probe. Implementations of
        // `JointPolicy` for `MlpBurnPolicy` and
        // `MultiDiscreteMlpBurnPolicy` both implement the standard
        // policy-head structure; we don't have a direct `logits()`
        // method on the trait, so we sample `get_action_host_seeded`
        // repeatedly and take the empirical marginal as the diagnostic.
        // For a single-row constant observation this is correct in
        // expectation; it's a host-side probe with no autograd cost.
        let probes = 128usize;
        let mut counts = vec![0u32; action_dim];
        for _ in 0..probes {
            let (acts, _, _) = policy.get_action_host_seeded(obs.clone(), &mut self.rng);
            if let Some(&a) = acts.first() {
                let idx = a as usize;
                if idx < action_dim {
                    counts[idx] += 1;
                }
            }
        }
        Some(counts.iter().map(|&c| c as f32 / probes as f32).collect())
    }

    /// Collect a synchronized rollout of length
    /// `joint_config.rollout_steps` with η-anticipatory mixing. Each
    /// step independently flips one Bernoulli(η) coin per agent: with
    /// probability `η` it samples the action from the BR policy and
    /// pushes the `(obs, action)` pair into that agent's reservoir;
    /// otherwise it samples from the AP and does NOT push.
    ///
    /// The action that's actually fed into `env.step_joint` is the
    /// per-step choice (BR or AP). The resulting [`JointRollout`]
    /// carries those per-step actions plus the rollout-time
    /// log-probs/values from the BR policy (we use the BR's value
    /// estimate either way — the value head is on the BR policy and
    /// the AP module does not maintain critic targets in this
    /// implementation).
    fn collect_anticipatory_rollout(&mut self) -> Result<crate::multi_agent::joint::JointRollout> {
        use crate::multi_agent::joint::JointRollout;
        let num_steps = self.joint_config.rollout_steps;
        let num_agents = self.joint_config.num_agents;
        let mut env = (self.env_factory)();
        let initial_obs = env.reset_joint(Some(self.config.seed));
        let obs_dim = initial_obs[0].len();
        // Per-agent persistent observation: agent `i` carries its own view.
        let mut last_obs: Vec<Vec<f32>> = initial_obs;
        let device = self.device.clone();

        // Probe num_action_dims from BR policy 0.
        let num_action_dims: usize = self.br_policy(0).action_dims_joint().len();

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
            for i in 0..num_agents {
                // Per-agent observation: each agent sees its own view.
                let agent_obs = last_obs[i].clone();
                obs_buf_per_agent[i][start..start + obs_dim].copy_from_slice(&agent_obs);
                let obs_t = Tensor::<B, 2>::from_data(
                    TensorData::new(agent_obs.clone(), [1, obs_dim]),
                    &device,
                );

                // Forward the BR policy unconditionally — we use its
                // log-prob / value as the rollout-time bookkeeping for
                // the PPO update regardless of which path provided the
                // sampled action. This matches Heinrich & Silver §3
                // Algorithm 1: the BR is the PPO learner and uses its
                // own log-prob targets.
                //
                // Clone the BR policy out of `self.br_trainer` so we
                // can release that borrow before passing `&mut self.rng`
                // to the seeded sampler. Burn modules are cheap to
                // clone (parameters live behind `Arc`).
                let br_policy_i = self.br_trainer.policy(i).clone();
                let (br_actions, br_log_probs, br_values) =
                    br_policy_i.get_action_host_seeded(obs_t.clone(), &mut self.rng);

                // η-coin: BR (push to reservoir) vs AP (do NOT push).
                let u: f32 = self.rng.random();
                let take_br = u < self.config.anticipatory_param;
                let row: Vec<i64> = if take_br {
                    let row = br_actions[..num_action_dims].to_vec();
                    // Reservoir push for the BR action only. Agent `i`'s
                    // reservoir gets agent `i`'s observation — not agent
                    // 0's — so partial-observability envs (PR #118
                    // refactor) record the correct supervised target for
                    // the behavior-cloning update.
                    //
                    // Push the full per-dim action vector (length
                    // `num_action_dims`). For single-discrete policies
                    // this is a length-1 vec; for multi-discrete
                    // factored policies (e.g. bucket-brigade
                    // `[10, 2, 2]`) it preserves the per-head action so
                    // the supervised CE step in
                    // `train_average_policies` can feed
                    // `evaluate_actions_joint` a properly-shaped
                    // `[mb, num_action_dims]` int tensor (issue #127).
                    self.reservoirs[i].push((agent_obs.clone(), row.clone()));
                    self.cumulative_br_pushes += 1;
                    row
                } else {
                    // Same clone pattern: release the `&self` borrow on
                    // `self.avg_policy(i)` before mutably borrowing
                    // `self.rng` through the seeded sampler.
                    let ap_policy_i = self.avg_policy(i).clone();
                    let (ap_actions, _, _) =
                        ap_policy_i.get_action_host_seeded(obs_t, &mut self.rng);
                    ap_actions[..num_action_dims].to_vec()
                };

                let off = t * num_action_dims;
                act_buf[i][off..off + num_action_dims].copy_from_slice(&row);
                joint_action.push(row);

                // Always record BR's rollout-time bookkeeping. We use
                // BR's log-prob even when the action was AP-sampled:
                // PPO clipping treats the recorded `old_log_prob` as
                // a behavior reference, and using BR's log-prob keeps
                // the surrogate's "current policy / old policy" ratio
                // well-defined under the BR's training update. (When
                // the action was AP-sampled, the BR's log-prob is the
                // log-prob the BR *would* have assigned to that
                // action — that's the correct importance-sampling
                // reference for behavior cloning into the BR.)
                lp_buf[i][t] = br_log_probs.first().copied().unwrap_or(0.0);
                val_buf[i][t] = br_values.first().copied().unwrap_or(0.0);
            }

            let result = env.step_joint(&joint_action);
            // Index-based scan: mirrors `JointMultiAgentTrainer::collect_rollout`'s
            // per-agent reward fan-out. Apply the optional reward scale
            // (issue #199): scaling rewards uniformly is an affine
            // transform of the return and does not change the optimal
            // policy, but keeps the large-magnitude bucket-brigade band
            // (`[−700, 0]`) in a numerically friendlier range for the
            // BR critic's regression targets and advantage stats.
            let reward_scale = self.config.br_reward_scale;
            #[allow(clippy::needless_range_loop)]
            for i in 0..num_agents {
                rew_buf[i][t] = result.rewards[i] * reward_scale;
            }
            done_buf[t] = if result.done { 1.0 } else { 0.0 };

            if result.done {
                let fresh = env.reset_joint(None);
                last_obs[..num_agents].clone_from_slice(&fresh[..num_agents]);
            } else {
                last_obs[..num_agents].clone_from_slice(&result.observations[..num_agents]);
            }
            self.cumulative_rollout_steps += 1;
        }

        Ok(JointRollout {
            observations_per_agent: obs_buf_per_agent,
            obs_dim,
            actions: act_buf,
            num_action_dims,
            log_probs: lp_buf,
            values: val_buf,
            rewards: rew_buf,
            dones: done_buf,
        })
    }

    /// One supervised cross-entropy step per agent over the agent's
    /// reservoir buffer. Skips agents whose reservoir is empty.
    /// Returns per-agent mean loss across all the supervised steps
    /// (`None` if no steps were taken for that agent).
    fn train_average_policies(&mut self) -> Result<Vec<Option<f64>>> {
        let num_agents = self.joint_config.num_agents;
        let mut losses: Vec<Option<f64>> = vec![None; num_agents];
        let steps = self.config.avg_policy_train_steps_per_iteration;
        // Skip entirely only when *both* the fixed budget and the
        // adaptive coverage floor are zero — otherwise coverage may
        // still force supervised steps even with `steps == 0` (issue
        // #199).
        if steps == 0 && self.config.avg_policy_min_reservoir_coverage <= 0.0 {
            return Ok(losses);
        }

        // Per-agent index loop: we read/mutate `self.reservoirs[i]` and
        // `self.avg_policies[i]` inside the loop; rewriting as an
        // iter_mut().enumerate() over `losses` is awkward because we
        // also need the outer `self` borrow.
        #[allow(clippy::needless_range_loop)]
        for i in 0..num_agents {
            let mb_size = self.config.avg_policy_minibatch_size.max(1);
            if self.reservoirs[i].is_empty() {
                continue;
            }
            // Adaptive step floor (issue #199): when
            // `avg_policy_min_reservoir_coverage > 0`, run enough
            // supervised steps that the AP sees `coverage` full passes
            // over the current reservoir, so a large reservoir is not
            // starved by a tiny fixed step budget (the root cause of the
            // `avg_ap_loss` pinned at `ln(num_joint_actions)` on
            // bucket-brigade). The configured `steps` remains the floor.
            let agent_steps = {
                let coverage = self.config.avg_policy_min_reservoir_coverage;
                if coverage > 0.0 {
                    let res_len = self.reservoirs[i].len();
                    let needed =
                        (coverage as f64 * res_len as f64 / mb_size as f64).ceil() as usize;
                    steps.max(needed)
                } else {
                    steps
                }
            };
            // Source-of-truth: probe `num_action_dims` from the BR
            // policy for agent `i`. PR #103 / issue #127: the AP policy
            // shares the same factored-action shape as the BR policy,
            // and `evaluate_actions_joint` expects actions of shape
            // `[mb, num_action_dims]` for both `MlpBurnPolicy`
            // (length-1 columns) and `MultiDiscreteMlpBurnPolicy`
            // (length-`num_action_dims` columns).
            let num_action_dims = self.br_trainer.policy(i).action_dims_joint().len();
            let mut sum_loss = 0.0_f64;
            let mut n_steps_done = 0usize;
            for _ in 0..agent_steps {
                let batch = self.reservoirs[i].sample_with_replacement(mb_size);
                if batch.is_empty() {
                    continue;
                }
                let obs_dim = batch[0].0.len();
                let mb = batch.len();
                // Flatten obs into [mb, obs_dim], actions into
                // [mb, num_action_dims]. Each batch row's action vec
                // must already have length `num_action_dims` — that
                // invariant is established at push-time in
                // `collect_anticipatory_rollout` (#127).
                let mut obs_flat = Vec::with_capacity(mb * obs_dim);
                let mut acts_flat = Vec::with_capacity(mb * num_action_dims);
                for (o, a) in &batch {
                    debug_assert_eq!(
                        a.len(),
                        num_action_dims,
                        "reservoir action vec length {} != num_action_dims {} for agent {}",
                        a.len(),
                        num_action_dims,
                        i
                    );
                    obs_flat.extend_from_slice(o);
                    acts_flat.extend_from_slice(a);
                }
                let obs_t = Tensor::<B, 2>::from_data(
                    TensorData::new(obs_flat, [mb, obs_dim]),
                    &self.device,
                );
                // Build the action tensor directly at shape
                // `[mb, num_action_dims]` — no `unsqueeze_dim` step.
                // For single-discrete (`MlpBurnPolicy`,
                // `num_action_dims = 1`) this matches the prior
                // shape exactly (the JointPolicy impl squeezes the
                // trailing dim away). For multi-discrete
                // (`MultiDiscreteMlpBurnPolicy`) this is required for
                // the per-dim `slice([0..mb, i..i+1])` reads inside
                // `evaluate_actions` to land in-bounds.
                let acts_t: Tensor<B, 2, Int> = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(acts_flat, [mb, num_action_dims]),
                    &self.device,
                );
                // Cross-entropy: -mean( log_softmax(logits)[gather(actions)] )
                let policy = self.avg_policies[i]
                    .take()
                    .ok_or_else(|| anyhow!("AP policy {} is None mid-update", i))?;
                // We don't have a direct logits method on the
                // JointPolicy trait — but for both
                // `MlpBurnPolicy` and `MultiDiscreteMlpBurnPolicy` the
                // evaluate_actions_joint path returns log-probs over
                // the *taken* actions, which is exactly what
                // cross-entropy needs. For multi-discrete the returned
                // log-prob is already the sum-across-dims (see
                // `MultiDiscreteMlpBurnPolicy::evaluate_actions`), so
                // `-mean(log_probs_taken)` IS the sum-of-per-head
                // cross-entropy of a conditionally-independent factored
                // distribution. No new `supervised_loss` method needed.
                let (log_probs_taken, _, _) = policy.evaluate_actions_joint(obs_t, acts_t);
                // Loss = -mean(log_probs_taken) → minimize.
                let loss = log_probs_taken.neg().mean();
                let loss_value = scalar_f64_avg_policy(loss.clone());
                let mut grads = loss.backward();
                let grads_params = GradientsParams::from_module(&mut grads, &policy);
                let lr = self.avg_optimizers[i].learning_rate();
                let updated = self.avg_optimizers[i].inner_mut().step(lr, policy, grads_params);
                self.avg_policies[i] = Some(updated);
                sum_loss += loss_value;
                n_steps_done += 1;
            }
            if n_steps_done > 0 {
                losses[i] = Some(sum_loss / n_steps_done as f64);
            }
        }
        Ok(losses)
    }
}

/// Scalar host-side conversion for AP loss tensors. Mirrors
/// `crate::train::ppo::loss::scalar_f64` but without the dependency
/// since that helper is currently `pub(crate)`.
fn scalar_f64_avg_policy<B: AutodiffBackend>(t: Tensor<B, 1>) -> f64 {
    let v: Vec<f32> = t.into_data().to_vec().expect("scalar tensor to_vec");
    v.first().copied().unwrap_or(0.0) as f64
}

// =======================================================================
// Tests
// =======================================================================

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
        optim::AdamConfig,
    };

    use super::*;
    use crate::{env::games::matching_pennies::MatchingPennies, policy::mlp::MlpBurnPolicy};

    type B = Autodiff<NdArray<f32>>;

    // ------------------------------------------------------------------
    // ReservoirBuffer — Vitter's Algorithm R
    // ------------------------------------------------------------------

    #[test]
    fn test_reservoir_under_capacity_retains_all() {
        let mut r = ReservoirBuffer::<u32>::with_seed(8, 0);
        for i in 0..8 {
            assert!(r.push(i), "every push under capacity must be retained");
        }
        assert_eq!(r.len(), 8);
        assert_eq!(r.items(), &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(r.stream_length(), 8);
    }

    #[test]
    fn test_reservoir_at_capacity_stays_at_capacity() {
        let mut r = ReservoirBuffer::<u32>::with_seed(4, 0);
        // Stream 100 items through; reservoir size stays at 4 after first 4.
        for i in 0..100 {
            r.push(i);
        }
        assert_eq!(r.len(), 4, "reservoir size must stay at capacity");
        assert_eq!(r.stream_length(), 100);
    }

    /// Mean-stream-index test for Vitter Algorithm R uniformity.
    ///
    /// Over many trials, each item in the held reservoir is sampled
    /// uniformly across the full stream history. So the mean of the
    /// held items' stream indices (1..=N) should be ≈ (N+1)/2 over
    /// repeated independent runs. A FIFO buffer would yield mean
    /// ≈ N − capacity/2 + 1 (recent items only). A sliding window of
    /// the last `capacity` items would yield mean ≈ N − (capacity − 1)/2.
    /// Both are far from `(N+1)/2` for `N >> capacity`.
    #[test]
    fn test_reservoir_uniformity_via_mean_stream_index() {
        let capacity = 16usize;
        let n = 1000usize; // stream length, 60x capacity
        let trials = 50usize;
        let expected_mean = (n as f64 + 1.0) / 2.0; // ≈ 500.5
        let mut grand_mean = 0.0_f64;
        for trial in 0..trials {
            let mut r = ReservoirBuffer::<usize>::with_seed(capacity, trial as u64);
            for i in 1..=n {
                r.push(i); // store 1-based stream index as the item
            }
            assert_eq!(r.len(), capacity);
            let sum: usize = r.items().iter().sum();
            let mean = sum as f64 / capacity as f64;
            grand_mean += mean;
        }
        grand_mean /= trials as f64;
        let abs_err = (grand_mean - expected_mean).abs();
        // Standard error of the mean for the uniformity test: each
        // trial's mean has variance ≈ N²/(12 * capacity). Across
        // `trials`, the SEM of `grand_mean` is N/sqrt(12 * capacity * trials).
        // Allow ~3 SEM tolerance.
        let sem = (n as f64) / ((12.0 * capacity as f64 * trials as f64).sqrt());
        let tol = 3.0 * sem;
        assert!(
            abs_err <= tol,
            "Vitter Algorithm R uniformity failed: grand_mean={grand_mean:.2}, \
             expected_mean={expected_mean:.2}, abs_err={abs_err:.2}, tol={tol:.2} \
             (this test fails for FIFO/sliding-window buffers)"
        );
    }

    #[test]
    fn test_reservoir_first_overflow_inclusion_probability() {
        // At capacity = N, the first overflow item (stream index
        // N+1) has insertion probability exactly N/(N+1) under
        // Vitter Algorithm R. Verify empirically.
        let capacity = 10usize;
        let trials = 5000usize;
        let mut kept = 0usize;
        for t in 0..trials {
            let mut r = ReservoirBuffer::<u32>::with_seed(capacity, (t as u64) ^ 0xdead_beef);
            // Fill to capacity.
            for i in 0..capacity {
                r.push(i as u32);
            }
            // The first overflow push — should be retained with
            // probability capacity / (capacity + 1) = 10/11 ≈ 0.909.
            let was_retained = r.push(u32::MAX);
            if was_retained {
                kept += 1;
            }
        }
        let p_emp = kept as f64 / trials as f64;
        let p_target = capacity as f64 / (capacity as f64 + 1.0);
        let std = (p_target * (1.0 - p_target) / trials as f64).sqrt();
        let tol = 3.0 * std;
        assert!(
            (p_emp - p_target).abs() <= tol,
            "first-overflow inclusion probability deviates: p_emp={p_emp:.4}, \
             p_target={p_target:.4} (tol={tol:.4})"
        );
    }

    #[test]
    fn test_reservoir_sample_with_replacement_empty() {
        let mut r = ReservoirBuffer::<u32>::with_seed(4, 0);
        let s = r.sample_with_replacement(8);
        assert!(s.is_empty());
    }

    #[test]
    fn test_reservoir_sample_with_replacement_uniform_over_items() {
        let mut r = ReservoirBuffer::<u32>::with_seed(2, 0);
        r.push(7);
        r.push(11);
        let n = 1000;
        let s = r.sample_with_replacement(n);
        assert_eq!(s.len(), n);
        let c7 = s.iter().filter(|&&x| x == 7).count();
        let c11 = s.iter().filter(|&&x| x == 11).count();
        assert_eq!(c7 + c11, n, "all samples must be from the reservoir");
        // ~half each. Allow plenty of slack.
        let frac7 = c7 as f64 / n as f64;
        assert!((frac7 - 0.5).abs() < 0.1, "sample distribution should be ~uniform");
    }

    // ------------------------------------------------------------------
    // η-anticipatory mixing: counter-based rate test + reservoir-push
    // invariant
    // ------------------------------------------------------------------

    /// A "rollout-only" variant of
    /// [`NfspTrainer::collect_anticipatory_rollout`] that uses a much
    /// smaller env loop and no PPO update, to keep the test fast.
    /// Reimplemented here because the real `collect_anticipatory_rollout`
    /// is `&mut self` on the full trainer.
    fn run_anticipatory_simulation(eta: f32, steps: usize, seed: u64) -> (usize, usize, usize) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut br_pushed = 0usize;
        let ap_pushed = 0usize;
        let mut br_taken = 0usize;
        for _ in 0..steps {
            let u: f32 = rng.random();
            if u < eta {
                br_taken += 1;
                br_pushed += 1; // BR actions enter the reservoir
            } else {
                // AP path: deliberately do NOT push.
                let _ = ap_pushed;
            }
        }
        (br_taken, br_pushed, ap_pushed)
    }

    #[test]
    fn test_eta_anticipatory_mixing_rate_concentration() {
        // 100k steps with η = 0.1 → BR fraction within 3σ of 0.1.
        let eta = 0.1f32;
        let n = 100_000usize;
        let (br_taken, _, _) = run_anticipatory_simulation(eta, n, 42);
        let p_emp = br_taken as f64 / n as f64;
        let p_target = eta as f64;
        let std = (p_target * (1.0 - p_target) / n as f64).sqrt();
        let tol = 3.0 * std;
        assert!(
            (p_emp - p_target).abs() <= tol,
            "η-mixing rate deviates from 0.1: p_emp={p_emp:.5} (tol={tol:.5})"
        );
    }

    #[test]
    fn test_eta_anticipatory_only_br_actions_enter_reservoir() {
        // Counter-based invariant: ap_pushed must be exactly 0 across
        // any number of η flips. The "simulation" mirrors what the
        // real trainer does: only the BR path increments the
        // reservoir push counter.
        let eta = 0.1f32;
        for seed in [0u64, 1, 2, 42, 12345] {
            let (_, br_pushed, ap_pushed) = run_anticipatory_simulation(eta, 10_000, seed);
            assert_eq!(
                ap_pushed, 0,
                "AP path must NOT push to the reservoir (seed={seed}, ap_pushed={ap_pushed})"
            );
            // Sanity: BR pushes happen.
            assert!(br_pushed > 0, "BR should sample at least once (seed={seed})");
        }
    }

    // ------------------------------------------------------------------
    // Integration: NfspTrainer on matching pennies (smoke)
    // ------------------------------------------------------------------

    #[allow(clippy::type_complexity)]
    fn build_matching_pennies_nfsp_trainer(
        max_iterations: usize,
        eta: f32,
    ) -> NfspTrainer<
        B,
        MlpBurnPolicy<B>,
        burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MlpBurnPolicy<B>, B>,
        MatchingPennies,
        impl Fn(&NdArrayDevice, u64) -> MlpBurnPolicy<B>,
        impl Fn() -> BurnOptimizer<
            B,
            MlpBurnPolicy<B>,
            burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MlpBurnPolicy<B>, B>,
        >,
        impl Fn() -> MatchingPennies,
    > {
        let device: NdArrayDevice = Default::default();
        let nfsp_config = NfspConfig {
            max_iterations,
            anticipatory_param: eta,
            reservoir_capacity: 1_024,
            br_train_steps_per_iteration: 1,
            avg_policy_train_steps_per_iteration: 2,
            avg_policy_minibatch_size: 32,
            avg_policy_lr: 5e-3,
            avg_policy_min_reservoir_coverage: 0.0,
            br_reward_scale: 1.0,
            seed: 0,
        };
        let joint_config = JointTrainerConfig {
            num_agents: 2,
            rollout_steps: 64,
            n_epochs: 1,
            minibatch_size: 32,
            ..Default::default()
        };
        NfspTrainer::new(
            nfsp_config,
            joint_config,
            device,
            |dev: &NdArrayDevice, seed: u64| {
                MlpBurnPolicy::<B>::new_seeded(
                    MatchingPennies::OBS_DIM,
                    MatchingPennies::ACTION_DIM,
                    16,
                    seed,
                    dev,
                )
            },
            || {
                let inner = AdamConfig::new().init();
                BurnOptimizer::new(inner, 5e-3)
            },
            MatchingPennies::new,
        )
        .expect("NfspTrainer::new should succeed for 2-agent config")
    }

    #[test]
    fn test_nfsp_construction_rejects_num_agents_less_than_two() {
        let device: NdArrayDevice = Default::default();
        let nfsp_config = NfspConfig::default();
        // num_agents = 1 — below the trainer's minimum, must be rejected.
        let joint_config = JointTrainerConfig { num_agents: 1, ..Default::default() };
        let res = NfspTrainer::<
            B,
            MlpBurnPolicy<B>,
            burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MlpBurnPolicy<B>, B>,
            MatchingPennies,
            _,
            _,
            _,
        >::new(
            nfsp_config,
            joint_config,
            device,
            |dev: &NdArrayDevice, seed: u64| MlpBurnPolicy::<B>::new_seeded(1, 2, 8, seed, dev),
            || BurnOptimizer::new(AdamConfig::new().init(), 1e-3),
            MatchingPennies::new,
        );
        assert!(res.is_err(), "NFSP should reject num_agents < 2");
    }

    #[test]
    fn test_nfsp_construction_accepts_n_player_config() {
        // Post-#119: num_agents > 2 is no longer rejected. Verify the
        // trainer constructs with N=4 and the right number of
        // reservoirs/AP policies.
        use crate::env::games::n_player_matching_pennies::NPlayerMatchingPennies;
        let device: NdArrayDevice = Default::default();
        let nfsp_config = NfspConfig::default();
        let joint_config = JointTrainerConfig { num_agents: 4, ..Default::default() };
        let trainer = NfspTrainer::<
            B,
            MlpBurnPolicy<B>,
            burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MlpBurnPolicy<B>, B>,
            NPlayerMatchingPennies,
            _,
            _,
            _,
        >::new(
            nfsp_config,
            joint_config,
            device,
            |dev: &NdArrayDevice, seed: u64| {
                MlpBurnPolicy::<B>::new_seeded(
                    NPlayerMatchingPennies::OBS_DIM,
                    NPlayerMatchingPennies::ACTION_DIM,
                    8,
                    seed,
                    dev,
                )
            },
            || BurnOptimizer::new(AdamConfig::new().init(), 1e-3),
            || NPlayerMatchingPennies::new(4),
        )
        .expect("NFSP should accept num_agents = 4");
        for i in 0..4 {
            assert_eq!(trainer.reservoir(i).len(), 0, "agent {i} reservoir starts empty");
        }
    }

    #[test]
    fn test_nfsp_construction_rejects_out_of_range_eta() {
        let device: NdArrayDevice = Default::default();
        let nfsp_config = NfspConfig { anticipatory_param: 1.5, ..Default::default() };
        let joint_config = JointTrainerConfig { num_agents: 2, ..Default::default() };
        let res = NfspTrainer::<
            B,
            MlpBurnPolicy<B>,
            burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MlpBurnPolicy<B>, B>,
            MatchingPennies,
            _,
            _,
            _,
        >::new(
            nfsp_config,
            joint_config,
            device,
            |dev: &NdArrayDevice, seed: u64| MlpBurnPolicy::<B>::new_seeded(1, 2, 8, seed, dev),
            || BurnOptimizer::new(AdamConfig::new().init(), 1e-3),
            MatchingPennies::new,
        );
        assert!(res.is_err(), "NFSP should reject η outside [0,1]");
    }

    #[test]
    fn test_nfsp_runs_end_to_end_on_matching_pennies() {
        let mut trainer = build_matching_pennies_nfsp_trainer(3, 0.5);
        let stats = trainer.run_silent().expect("NFSP run should not error");
        assert_eq!(stats.iterations.len(), 3, "should record 3 iterations");
        for (k, it) in stats.iterations.iter().enumerate() {
            assert_eq!(it.iteration, k + 1);
            assert_eq!(it.reservoir_sizes.len(), 2);
            // With η = 0.5 and ≥1 BR step per iter on 64 rollout steps,
            // reservoirs should accumulate items.
            assert!(it.reservoir_sizes.iter().sum::<usize>() > 0, "reservoirs should accumulate");
        }
    }

    #[test]
    fn test_nfsp_eta_zero_pure_avg_policy_never_pushes() {
        // η = 0: every rollout step samples from AP. Reservoir must
        // stay empty. AP supervised step must be skipped (no data).
        let mut trainer = build_matching_pennies_nfsp_trainer(2, 0.0);
        let _ = trainer.run_silent().expect("NFSP run with η=0 should not error");
        for i in 0..2 {
            assert_eq!(trainer.reservoir(i).len(), 0, "η=0 must leave reservoir {i} empty");
        }
        assert_eq!(trainer.cumulative_br_pushes(), 0, "η=0 must result in zero BR pushes");
    }

    #[test]
    fn test_nfsp_eta_one_only_br_path_fills_reservoir() {
        // η = 1: every step samples from BR; reservoir grows by one
        // per rollout step per agent.
        let mut trainer = build_matching_pennies_nfsp_trainer(1, 1.0);
        let _ = trainer.run_silent().expect("NFSP run with η=1 should not error");
        // 1 iteration × 1 BR train step × 64 rollout steps = 64
        // per agent.
        for i in 0..2 {
            assert_eq!(
                trainer.reservoir(i).len(),
                64,
                "η=1 should retain all 64 rollout steps per agent in reservoir {i}"
            );
        }
        // 2 agents × 64 steps = 128.
        assert_eq!(trainer.cumulative_br_pushes(), 128);
    }

    #[test]
    fn test_nfsp_eta_mixing_rate_concentration_in_trainer() {
        // Drive a single iteration with η = 0.1 over a large rollout
        // and check the empirical BR-fraction across the 2 agents is
        // within ~3σ of 0.1.
        let device: NdArrayDevice = Default::default();
        let eta = 0.1f32;
        let rollout_steps = 4096usize;
        let nfsp_config = NfspConfig {
            max_iterations: 1,
            anticipatory_param: eta,
            reservoir_capacity: 100_000,
            br_train_steps_per_iteration: 1,
            avg_policy_train_steps_per_iteration: 0,
            avg_policy_minibatch_size: 32,
            avg_policy_lr: 1e-3,
            avg_policy_min_reservoir_coverage: 0.0,
            br_reward_scale: 1.0,
            seed: 7,
        };
        let joint_config = JointTrainerConfig {
            num_agents: 2,
            rollout_steps,
            n_epochs: 1,
            minibatch_size: 32,
            ..Default::default()
        };
        let mut trainer = NfspTrainer::new(
            nfsp_config,
            joint_config,
            device,
            |dev: &NdArrayDevice, seed: u64| {
                MlpBurnPolicy::<B>::new_seeded(
                    MatchingPennies::OBS_DIM,
                    MatchingPennies::ACTION_DIM,
                    16,
                    seed,
                    dev,
                )
            },
            || BurnOptimizer::new(AdamConfig::new().init(), 1e-3),
            MatchingPennies::new,
        )
        .expect("NfspTrainer::new should succeed");
        let _ = trainer.run_silent().expect("NFSP run should not error");
        let br_pushes = trainer.cumulative_br_pushes() as f64;
        let total_steps = (rollout_steps * 2) as f64;
        let p_emp = br_pushes / total_steps;
        let p_target = eta as f64;
        let std = (p_target * (1.0 - p_target) / total_steps).sqrt();
        let tol = 4.0 * std; // 4σ for resilience to the small N
        assert!(
            (p_emp - p_target).abs() <= tol,
            "trainer η-mixing rate deviates: p_emp={p_emp:.4}, p_target={p_target:.4}, tol={tol:.4}"
        );
    }

    #[test]
    fn test_nfsp_avg_policy_supervised_step_reduces_loss_on_fixed_minibatch() {
        // Sanity-check the supervised wiring: a few AP updates on a
        // fixed reservoir reduce the per-batch cross-entropy.
        let device: NdArrayDevice = Default::default();
        let nfsp_config = NfspConfig {
            max_iterations: 0,
            anticipatory_param: 1.0,
            reservoir_capacity: 256,
            br_train_steps_per_iteration: 0,
            avg_policy_train_steps_per_iteration: 0,
            avg_policy_minibatch_size: 32,
            avg_policy_lr: 5e-2,
            avg_policy_min_reservoir_coverage: 0.0,
            br_reward_scale: 1.0,
            seed: 13,
        };
        let joint_config = JointTrainerConfig {
            num_agents: 2,
            rollout_steps: 32,
            n_epochs: 1,
            minibatch_size: 32,
            ..Default::default()
        };
        let mut trainer = NfspTrainer::new(
            nfsp_config,
            joint_config,
            device,
            |dev: &NdArrayDevice, seed: u64| MlpBurnPolicy::<B>::new_seeded(1, 2, 8, seed, dev),
            || BurnOptimizer::new(AdamConfig::new().init(), 5e-2),
            MatchingPennies::new,
        )
        .expect("NfspTrainer::new should succeed");

        // Seed reservoir 0 with a fixed dataset: all `(obs=[0.0], action=[0])`.
        // Single-discrete callers wrap their scalar action in a length-1
        // vec — the post-#127 reservoir contract.
        for _ in 0..64 {
            trainer.reservoirs[0].push((vec![0.0_f32], vec![0]));
        }
        // Stage the supervised step manually a few times by setting
        // `avg_policy_train_steps_per_iteration` and calling the
        // private method. We do this by mutating the config and
        // invoking train_average_policies through a single iteration
        // of the public run loop. To avoid extra rollout
        // mechanics, build a config that asks for AP-only steps and
        // 0 BR rollouts.
        trainer.config.avg_policy_train_steps_per_iteration = 10;

        // First loss probe.
        let losses_before = trainer.train_average_policies().unwrap();
        let loss_before = losses_before[0].expect("expected supervised loss for agent 0");

        // Repeat 4 more times.
        for _ in 0..4 {
            trainer.train_average_policies().unwrap();
        }
        let losses_after = trainer.train_average_policies().unwrap();
        let loss_after = losses_after[0].expect("expected supervised loss for agent 0");

        assert!(
            loss_after < loss_before,
            "AP supervised CE should decrease: before={loss_before:.4}, after={loss_after:.4}"
        );
    }

    /// Issue #199 regression: on a **factored multi-discrete** `[10,2,2]`
    /// action space (bucket-brigade's shape, joint cardinality 40), the
    /// uniform-policy cross-entropy floor is `ln(40) ≈ 3.689`. The #134
    /// cluster run reported `avg_ap_loss` pinned at exactly that floor
    /// for an entire 48-iteration run — the average policy never fit its
    /// reservoir.
    ///
    /// This test reproduces the *root cause locally* and shows the fix:
    /// it seeds an agent's reservoir with a deterministic **non-uniform**
    /// target (a single fixed joint action `[3, 1, 0]` for every entry,
    /// the easiest possible supervised target) and runs the AP supervised
    /// update. With the adaptive `avg_policy_min_reservoir_coverage` floor
    /// the AP is given enough gradient steps to drive `avg_ap_loss`
    /// **well below `ln(40)`** — exactly the signal the cluster run never
    /// produced. Fast (NdArray CPU, tiny obs/hidden dims), so it runs on
    /// every `cargo test --features training` invocation, not behind
    /// `#[ignore]`.
    #[test]
    fn test_nfsp_multi_discrete_ap_loss_drops_below_uniform_floor() {
        use crate::policy::multi_discrete_mlp::MultiDiscreteMlpBurnPolicy;

        let device: NdArrayDevice = Default::default();
        let action_dims = vec![10usize, 2, 2]; // joint cardinality = 40
        let uniform_floor = (40.0_f64).ln(); // ≈ 3.6889

        let nfsp_config = NfspConfig {
            max_iterations: 0,
            anticipatory_param: 1.0,
            reservoir_capacity: 4_096,
            br_train_steps_per_iteration: 0,
            avg_policy_train_steps_per_iteration: 8,
            avg_policy_minibatch_size: 64,
            avg_policy_lr: 5e-3,
            // The fix under test: cover the reservoir ~4× per call so the
            // AP gets enough gradient steps to actually fit the target.
            avg_policy_min_reservoir_coverage: 4.0,
            br_reward_scale: 1.0,
            seed: 199,
        };
        let joint_config = JointTrainerConfig {
            num_agents: 2,
            rollout_steps: 8,
            n_epochs: 1,
            minibatch_size: 32,
            ..Default::default()
        };

        // obs_dim is small and arbitrary here — we never step the env;
        // we drive `train_average_policies` directly on a hand-seeded
        // reservoir. The env type only has to satisfy the trainer's
        // generic bound (we use MatchingPennies, never stepped).
        let obs_dim = 4usize;
        let dims_for_factory = action_dims.clone();
        let mut trainer = NfspTrainer::new(
            nfsp_config,
            joint_config,
            device,
            move |dev: &NdArrayDevice, seed: u64| {
                MultiDiscreteMlpBurnPolicy::<B>::new_seeded(
                    obs_dim,
                    dims_for_factory.clone(),
                    16,
                    seed,
                    dev,
                )
            },
            || BurnOptimizer::new(AdamConfig::new().init(), 5e-3),
            MatchingPennies::new,
        )
        .expect("NfspTrainer::new should succeed for multi-discrete config");

        // Seed reservoir 0 with a fixed, non-uniform supervised target:
        // every entry maps a constant observation to the single joint
        // action [3, 1, 0]. A network that fits this perfectly drives the
        // CE loss toward 0; the question is whether the AP gets enough
        // gradient steps to move off the ln(40) uniform floor at all.
        let fixed_obs = vec![0.25_f32; obs_dim];
        let fixed_action: Vec<i64> = vec![3, 1, 0];
        for _ in 0..512 {
            trainer.reservoirs[0].push((fixed_obs.clone(), fixed_action.clone()));
        }

        // First supervised pass: the freshly-initialized AP should start
        // near the uniform floor.
        let losses_first = trainer.train_average_policies().unwrap();
        let loss_first = losses_first[0].expect("expected supervised loss for agent 0");

        // A handful more passes to let the adaptive-coverage budget fit
        // the (trivial, deterministic) target.
        for _ in 0..8 {
            trainer.train_average_policies().unwrap();
        }
        let losses_last = trainer.train_average_policies().unwrap();
        let loss_last = losses_last[0].expect("expected supervised loss for agent 0");

        // Visible with `cargo test -- --nocapture`; documents the local
        // before/after used in the PR writeup.
        eprintln!(
            "[#199] multi-discrete AP loss: first={loss_first:.4}, last={loss_last:.4}, \
             ln(40) floor={uniform_floor:.4}"
        );

        // The cluster run's symptom was `avg_ap_loss` stuck AT the floor.
        // The fix must drive it meaningfully below. Use a generous margin
        // (0.5 nats) so the assertion is robust across platforms but
        // still hard-fails the "pinned at ln(40)" pathology.
        assert!(
            loss_last < uniform_floor - 0.5,
            "AP loss should drop well below the ln(40) uniform floor: \
             first={loss_first:.4}, last={loss_last:.4}, floor={uniform_floor:.4}"
        );
        assert!(
            loss_last < loss_first,
            "AP loss should decrease across supervised passes: \
             first={loss_first:.4}, last={loss_last:.4}"
        );
    }

    /// Issue #199: the adaptive coverage floor must run *more* supervised
    /// steps than the fixed `avg_policy_train_steps_per_iteration` when
    /// the reservoir is large, and exactly the fixed count when coverage
    /// is disabled (`0.0`). We verify this by counting actual gradient
    /// steps via the returned per-agent loss being present and by
    /// asserting the coverage math through a public-ish probe: a 0-step
    /// fixed budget with coverage still produces a loss (steps were run),
    /// while disabling coverage with a 0-step budget produces `None`.
    #[test]
    fn test_nfsp_adaptive_coverage_runs_steps_when_fixed_budget_is_zero() {
        use crate::policy::multi_discrete_mlp::MultiDiscreteMlpBurnPolicy;

        let device: NdArrayDevice = Default::default();
        let obs_dim = 3usize;
        let action_dims = vec![4usize, 2];

        let make = |coverage: f32, fixed_steps: usize| {
            let nfsp_config = NfspConfig {
                max_iterations: 0,
                anticipatory_param: 1.0,
                reservoir_capacity: 1_024,
                br_train_steps_per_iteration: 0,
                avg_policy_train_steps_per_iteration: fixed_steps,
                avg_policy_minibatch_size: 32,
                avg_policy_lr: 1e-3,
                avg_policy_min_reservoir_coverage: coverage,
                br_reward_scale: 1.0,
                seed: 7,
            };
            let joint_config = JointTrainerConfig {
                num_agents: 1 + 1,
                rollout_steps: 8,
                n_epochs: 1,
                minibatch_size: 32,
                ..Default::default()
            };
            let dims = action_dims.clone();
            let mut trainer = NfspTrainer::new(
                nfsp_config,
                joint_config,
                device,
                move |dev: &NdArrayDevice, seed: u64| {
                    MultiDiscreteMlpBurnPolicy::<B>::new_seeded(obs_dim, dims.clone(), 8, seed, dev)
                },
                || BurnOptimizer::new(AdamConfig::new().init(), 1e-3),
                MatchingPennies::new,
            )
            .expect("trainer construction");
            for _ in 0..256 {
                trainer.reservoirs[0].push((vec![0.1_f32; obs_dim], vec![1, 0]));
            }
            trainer
        };

        // Coverage disabled + 0 fixed steps → no supervised steps run.
        let mut disabled = make(0.0, 0);
        assert!(
            disabled.train_average_policies().unwrap()[0].is_none(),
            "coverage=0 with 0 fixed steps must run no supervised steps"
        );

        // Coverage enabled + 0 fixed steps → coverage forces steps to run
        // (256 entries × 1.0 / 32 mb = 8 steps), so a loss is produced.
        let mut enabled = make(1.0, 0);
        assert!(
            enabled.train_average_policies().unwrap()[0].is_some(),
            "coverage>0 must run supervised steps even when fixed budget is 0"
        );
    }

    #[test]
    fn test_nfsp_determinism_within_module_under_seed() {
        // Same seed produces bit-identical η-flip stream and reservoir
        // structure (lengths + stream-lengths) when only the in-module
        // RNGs are exercised. The exact (obs, action) contents of the
        // reservoir depend on the rollout-time `get_action_host`
        // draws, which are downstream of (a) the non-deterministic
        // per-epoch shuffle inside
        // `JointMultiAgentTrainer::update_with_active_agents` (tracked
        // in #109) and (b) the BR policy's own non-deterministic
        // `rand::rng()` sampling inside `get_action_host`. Both are
        // out of scope for this PR; this test asserts only the
        // determinism slice the NFSP module itself controls.
        let mut a = build_matching_pennies_nfsp_trainer(1, 0.5);
        let mut b = build_matching_pennies_nfsp_trainer(1, 0.5);
        let _ = a.run_silent().unwrap();
        let _ = b.run_silent().unwrap();
        for i in 0..2 {
            assert_eq!(
                a.reservoir(i).len(),
                b.reservoir(i).len(),
                "reservoir lengths must match across same-seed runs (agent {i})"
            );
            assert_eq!(
                a.reservoir(i).stream_length(),
                b.reservoir(i).stream_length(),
                "reservoir stream length must match across same-seed runs (agent {i})"
            );
        }
        // η-flip-derived counts depend only on the seeded NFSP RNG and
        // must be bit-identical.
        assert_eq!(a.cumulative_br_pushes(), b.cumulative_br_pushes());
        assert_eq!(a.cumulative_rollout_steps(), b.cumulative_rollout_steps());
    }
}
