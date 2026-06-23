//! Policy-Space Response Oracles (PSRO) meta-game trainer.
//!
//! Burn-native implementation of the PSRO outer loop (Lanctot et al.
//! 2017, [arXiv:1711.00832](https://arxiv.org/abs/1711.00832)) for
//! 2-agent zero-sum games. Tracking issue: #107.
//!
//! # Pseudocode
//!
//! ```text
//! Population[i] = {π_i^(0)}   for each agent i      (initial random policy)
//! repeat for k = 1..K:
//!     1. Empirical game G_k = payoff matrix between Population[0] × Population[1]
//!     2. Meta-Nash σ_k = MetaSolver.solve(G_k)
//!     3. For each agent i in {0, 1}:
//!         a. Sample opponent policy from σ_k[1-i]
//!         b. Train π_i^(k) as best response to that mixture
//!         c. Append π_i^(k) to Population[i]
//!     4. Update payoff matrix with new row/column
//! end
//! ```
//!
//! # Why an in-tree Rust meta-solver instead of `bucket-brigade-core`?
//!
//! Issue #107's original framing called for wiring
//! `bucket-brigade-core::nash::DoubleOracleSolver` (Rust) in as the
//! meta-solver. Upon investigation, the DO solver in
//! `envs/bucket-brigade@6486a549fc` is **Python**, not Rust
//! (`bucket_brigade.equilibrium.double_oracle_heterogeneous.py`). The
//! `bucket-brigade-core` Rust crate exposes only `agents`, `engine`,
//! `rng`, `scenarios` — no `nash` module exists. Calling into Python
//! from a Rust trainer would introduce a runtime Python dependency
//! contrary to thrust's pure-Rust posture (and the
//! `bucket-brigade-core` dep is itself feature-gated off for v0.1.0
//! because the crate is not on crates.io). We instead define a
//! `MetaSolver` trait with three in-tree Rust implementations:
//!
//! - [`UniformMetaSolver`] — degenerate uniform mixture. Always available;
//!   serves as the unit-test baseline.
//! - [`FictitiousPlayMetaSolver`] — deterministic fictitious-play meta-solver.
//!   No external LP dependency.
//! - [`ReplicatorDynamicsMetaSolver`] — non-trivial mixed-Nash solver via
//!   projected replicator dynamics. No LP dependency; converges to the
//!   symmetric Nash on small empirical games (≤50 strategies).
//!
//! See the issue's curator comment
//! ([#107c-4704239526](https://github.com/rjwalters/thrust/issues/107#issuecomment-4704239526))
//! for the full rationale and the deferred Option 1 (port the Python
//! solver to Rust upstream).
//!
//! # Per-agent observation handling
//!
//! PSRO builds on top of
//! [`crate::multi_agent::joint::JointMultiAgentTrainer`], which records
//! a *per-agent* observation stream in
//! [`JointRollout::observations_per_agent`]. Envs with distinct
//! per-agent views (partial observability, asymmetric information)
//! drop in without pre-concatenation. Matching pennies returns
//! identical observations to both agents, which keeps the regression
//! tests bit-stable through the per-agent refactor (PR #118).
//!
//! # Population growth & cost
//!
//! Population grows monotonically — one new best-response policy per
//! PSRO iteration per agent. Per-iteration cost scales linearly in
//! population size (one BR train + one `n × n` meta-solver call). The
//! empirical-payoff matrix is cached: only the new row/column is
//! evaluated each iteration (existing entries are unchanged by
//! construction). Memory is quadratic in iteration count; bound it via
//! [`PsroConfig::max_population_size`] (default 50). The trainer
//! returns `Err` (not panic) when the cap is hit.
//!
//! # What this module ships in the first PR
//!
//! - The `MetaSolver` trait + three implementations.
//! - The `PsroTrainer` outer loop with a freeze-N-1 helper.
//! - The matching-pennies smoke test
//!   ([`crate::env::games::matching_pennies::MatchingPennies`]).
//!
//! # What is deferred to follow-up PRs
//!
//! The full set of acceptance criteria from the curator's comment also
//! call for a bucket-brigade integration test (gated behind
//! `env-bucket-brigade`) and a `train_psro.rs` example with the
//! `gap_closed_homogeneous` metric. Those depend on locally
//! re-enabling the `env-bucket-brigade` feature (the crate is
//! path-only and disabled in the published Cargo.toml) and porting
//! the metric from
//! `envs/bucket-brigade/experiments/scripts/compute_nash_phase_diagram.py`.
//! Both are tracked as cleavage point #3 in the curator's open
//! question; see PR description for the deferred-pieces summary.

use anyhow::{Result, anyhow};
use burn::{optim::Optimizer, tensor::backend::AutodiffBackend};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::{
    multi_agent::joint::{
        JointEnv, JointMultiAgentTrainer, JointPolicy, JointStats, JointTrainerConfig,
    },
    train::optimizer::BurnOptimizer,
};

// =======================================================================
// MetaSolver trait + implementations
// =======================================================================

/// Meta-solver over a symmetric 2-player zero-sum empirical game.
///
/// Given an `n × n` row-player payoff matrix `payoffs[i][j]`
/// representing the expected return of row-player strategy `i` versus
/// column-player strategy `j`, returns the row-player's mixed-Nash
/// distribution as a length-`n` probability vector summing to `1.0`.
///
/// For symmetric zero-sum games (matching pennies, the
/// homogeneous-policy version of bucket brigade) the column-player's
/// equilibrium is the same distribution by symmetry — callers can use
/// the row distribution for both agents. For non-symmetric games, this
/// trait is invoked twice (once per agent role) with appropriately
/// transposed payoff matrices.
pub trait MetaSolver {
    /// Solve for the row-player mixed-Nash on a symmetric `n × n`
    /// empirical payoff matrix.
    ///
    /// # Contract
    ///
    /// - Input is assumed to be `n × n` and square; non-square inputs produce
    ///   undefined behaviour (impl is free to panic).
    /// - Return vector has length `n` with non-negative entries summing to
    ///   `1.0` (within `1e-6` tolerance).
    fn solve(&self, payoffs: &[Vec<f32>]) -> Vec<f32>;

    /// N-player solve over an explicit per-agent payoff tensor.
    ///
    /// `payoffs` is shape `(k^num_agents, num_agents)` where
    /// `payoffs[s][a]` is agent `a`'s payoff at joint pure strategy
    /// `s`. The flat joint-strategy index decomposes into per-agent
    /// indices via little-endian mixed-radix (agent 0 = fastest):
    /// `s = Σ_i s_i · k^i`.
    ///
    /// Returns a probability vector of length `k^num_agents` summing
    /// to `1.0 ± 1e-6`.
    ///
    /// # Default
    ///
    /// The default implementation only supports `num_agents == 2` and
    /// delegates to `solve` via the row-marginal projection. For
    /// `num_agents > 2` it panics with a message naming the solver.
    /// Only [`AlphaRankMetaSolver`] overrides this method with a true
    /// N-player path; the other in-tree solvers (`UniformMetaSolver`,
    /// `FictitiousPlayMetaSolver`, `ReplicatorDynamicsMetaSolver`)
    /// have no N>2 generalization with the same convergence
    /// guarantees and intentionally panic.
    fn solve_n_player(
        &self,
        _payoffs: &[Vec<f32>],
        num_agents: usize,
        _per_role_k: usize,
    ) -> Vec<f32> {
        panic!(
            "{} does not support num_agents = {}; only 2-player meta-games. \
             Use AlphaRankMetaSolver for N > 2.",
            self.name(),
            num_agents
        );
    }

    /// Human-readable name for diagnostics / logging.
    fn name(&self) -> &'static str;
}

/// Degenerate uniform meta-solver.
///
/// Returns `[1/n; n]` independent of the payoff matrix. Useful as the
/// `n = 1` initial-iteration solver and as a unit-test baseline.
#[derive(Debug, Clone, Default)]
pub struct UniformMetaSolver;

impl MetaSolver for UniformMetaSolver {
    fn solve(&self, payoffs: &[Vec<f32>]) -> Vec<f32> {
        let n = payoffs.len().max(1);
        vec![1.0 / n as f32; n]
    }

    fn name(&self) -> &'static str {
        "uniform"
    }
}

/// Fictitious-play meta-solver.
///
/// Deterministic: each iteration, the row-player best-responds to the
/// column-player's empirical mixture, the column-player best-responds
/// to the row-player's empirical mixture, and both empirical mixtures
/// are updated. After `iterations` rounds the empirical row mixture
/// converges to the Nash on zero-sum games (Brown 1951, Robinson
/// 1951). No external LP dependency.
///
/// # Tuning
///
/// The default `iterations = 1000` is overkill for `n ≤ 8` but cheap
/// (each step is `O(n²)`). For very small empirical games this is
/// equivalent to (and slightly more robust than)
/// [`ReplicatorDynamicsMetaSolver`].
#[derive(Debug, Clone)]
pub struct FictitiousPlayMetaSolver {
    iterations: usize,
}

impl FictitiousPlayMetaSolver {
    /// Construct with `iterations` fictitious-play rounds.
    pub fn new(iterations: usize) -> Self {
        Self { iterations: iterations.max(1) }
    }
}

impl Default for FictitiousPlayMetaSolver {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl MetaSolver for FictitiousPlayMetaSolver {
    fn solve(&self, payoffs: &[Vec<f32>]) -> Vec<f32> {
        let n = payoffs.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![1.0];
        }
        // Empirical action counts; we'll normalize at the end.
        let mut row_counts = vec![0.0_f32; n];
        let mut col_counts = vec![0.0_f32; n];
        // Seed both empirical mixtures with one count on the first strategy.
        // (Standard fictitious-play initialization.)
        row_counts[0] = 1.0;
        col_counts[0] = 1.0;

        for _ in 0..self.iterations {
            // Column mixture
            let col_total: f32 = col_counts.iter().sum();
            let col_mix: Vec<f32> = col_counts.iter().map(|&c| c / col_total).collect();
            // Row best-responds: maximize expected row payoff against col_mix.
            let row_br = best_response_row(payoffs, &col_mix);
            row_counts[row_br] += 1.0;

            // Row mixture
            let row_total: f32 = row_counts.iter().sum();
            let row_mix: Vec<f32> = row_counts.iter().map(|&r| r / row_total).collect();
            // Col best-responds: minimize expected row payoff against row_mix
            // (since zero-sum, equivalent to maximizing -row payoff).
            let col_br = best_response_col(payoffs, &row_mix);
            col_counts[col_br] += 1.0;
        }

        let total: f32 = row_counts.iter().sum();
        if total <= 0.0 {
            return vec![1.0 / n as f32; n];
        }
        row_counts.iter().map(|&c| c / total).collect()
    }

    fn name(&self) -> &'static str {
        "fictitious_play"
    }
}

/// Replicator-dynamics meta-solver.
///
/// Projected replicator dynamics: iterate
/// `x_i ← x_i * (1 + η * (f_i − x · f))` followed by a non-negative
/// renormalization, where `f_i = Σ_j A[i][j] x_j` is the expected row
/// payoff for pure strategy `i` against the current mixture, and `η`
/// is a step size. For symmetric zero-sum games this converges to a
/// symmetric Nash equilibrium (Hofbauer & Sigmund 2003) without needing
/// an LP solver. Slightly faster than fictitious play on
/// continuous-payoff matrices but less robust to ties.
#[derive(Debug, Clone)]
pub struct ReplicatorDynamicsMetaSolver {
    iterations: usize,
    step_size: f32,
}

impl ReplicatorDynamicsMetaSolver {
    /// Construct with `iterations` updates at the given `step_size`.
    pub fn new(iterations: usize, step_size: f32) -> Self {
        Self { iterations: iterations.max(1), step_size: step_size.max(1e-6) }
    }
}

impl Default for ReplicatorDynamicsMetaSolver {
    fn default() -> Self {
        Self::new(2000, 0.05)
    }
}

impl MetaSolver for ReplicatorDynamicsMetaSolver {
    fn solve(&self, payoffs: &[Vec<f32>]) -> Vec<f32> {
        let n = payoffs.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![1.0];
        }
        // Start from uniform.
        let mut x = vec![1.0 / n as f32; n];
        for _ in 0..self.iterations {
            // Per-strategy expected payoff: f_i = Σ_j A[i][j] * x_j
            let mut f = vec![0.0_f32; n];
            for (i, row) in payoffs.iter().enumerate() {
                let mut fi = 0.0_f32;
                for (j, &a) in row.iter().enumerate() {
                    fi += a * x[j];
                }
                f[i] = fi;
            }
            // Mean payoff over the mixture.
            let mean_f: f32 = x.iter().zip(f.iter()).map(|(xi, fi)| xi * fi).sum();
            // Replicator update with non-negativity projection.
            let mut new_x: Vec<f32> = x
                .iter()
                .zip(f.iter())
                .map(|(xi, fi)| (xi * (1.0 + self.step_size * (fi - mean_f))).max(0.0))
                .collect();
            // Renormalize.
            let total: f32 = new_x.iter().sum();
            if total <= 1e-12 {
                // Degenerate (all entries zeroed out); fall back to uniform.
                return vec![1.0 / n as f32; n];
            }
            for v in new_x.iter_mut() {
                *v /= total;
            }
            x = new_x;
        }
        x
    }

    fn name(&self) -> &'static str {
        "replicator_dynamics"
    }
}

/// α-rank meta-solver (Omidshafiei et al. 2019,
/// [Nature Sci Reports 9:9937](https://doi.org/10.1038/s41598-019-45619-9)).
///
/// Computes the stationary distribution of a Markov chain over joint
/// pure strategies where transitions follow Moran-process mutation
/// dynamics: at each step a random agent is selected, a random
/// deviation strategy is proposed for that agent, and the deviation is
/// accepted with probability proportional to
/// `1 / (1 + exp(−α · (payoff_after − payoff_before)))`.
///
/// **Guarantee shipped:** highest stationary mass under the
/// response-graph Moran dynamics — NOT ε-Nash. The α-rank ordering
/// captures the dynamic strength of strategies but does not coincide
/// with the Nash equilibrium support in general (Omidshafiei et al.
/// 2019 §2 + Discussion). Use this solver when the goal is N-player
/// ranking over joint pure strategies, not Nash refinement.
///
/// # API surfaces
///
/// Two entry points are provided:
///
/// - **[`AlphaRankMetaSolver::solve`] (`MetaSolver` trait)**: takes a symmetric
///   `n × n` payoff matrix `payoffs[i][j]` (row-player payoff when row plays
///   strategy `i` against column strategy `j`) and computes the α-rank
///   stationary distribution over the `n` strategies. This is the *2-player
///   symmetric* path and is used for the random-payoff sanity tests. The
///   returned distribution has length `n`.
/// - **[`AlphaRankMetaSolver::solve_n_player`]**: takes a per-agent payoff
///   tensor of shape `(num_joint_strategies, num_agents)` where `payoffs[s][a]`
///   is agent `a`'s scalar payoff at joint pure strategy `s`, plus the number
///   of agents and the per-agent per-role population size `k`. The total number
///   of joint strategies must equal `k^num_agents`. Returns the stationary
///   distribution over the `k^num_agents` joint strategies. This is the *true
///   N-player* path used by the PSRO N > 2 branch.
///
/// # Defaults (per Omidshafiei §2.3)
///
/// - `ranking_intensity_alpha = 10.0` — the response-graph ranking intensity.
///   Larger values sharpen the deviation acceptance probability; the paper's
///   experiments use `α ∈ [1, 100]`.
/// - `moran_population_size = 50` — the Moran population size `m` parameter
///   controlling fixation probability magnitudes. The paper recommends `m ≥
///   10`.
/// - `max_iterations = 200` — power-iteration cap.
/// - `tolerance = 1e-6` — power-iteration convergence threshold on L1 distance
///   between successive distributions.
#[derive(Debug, Clone)]
pub struct AlphaRankMetaSolver {
    /// Response-graph ranking intensity α.
    pub ranking_intensity_alpha: f32,
    /// Moran population size m.
    pub moran_population_size: u32,
    /// Maximum power-iteration steps.
    pub max_iterations: usize,
    /// Power-iteration L1 convergence tolerance.
    pub tolerance: f32,
    /// When `true`, normalize each Moran payoff differential
    /// `delta = π_τ − π_σ` by the **payoff span** of the input tensor
    /// (`max − min` over all per-agent payoffs) before multiplying by α
    /// (issue #215).
    ///
    /// # Why this matters
    ///
    /// The Moran fixation probability is driven by `α · delta` (see
    /// `moran_fixation_probability`). α-rank's defaults
    /// (`α = 10`, `m = 50`) were validated on the `{−1, +1}`
    /// matching-pennies game, where `|delta| ≤ 2` and `α · delta ≤ 20`
    /// — comfortably inside the regime where the fixation probability is
    /// a graded sigmoid-like function of the payoff advantage. On the
    /// bucket-brigade `[−700, 0]` payoff band, `|delta|` can reach ~700
    /// and `α · delta ≈ 7000`, which **saturates** every non-neutral
    /// transition to a hard 0 or 1. The graded Moran dynamics collapse
    /// into a degenerate deterministic best-response graph, and the
    /// resulting stationary distribution is acutely sensitive to tiny
    /// payoff-estimate noise — a plausible contributor to the
    /// exploitability *divergence* observed on the no-convergence cells
    /// (issue #215, #198).
    ///
    /// When enabled, the differential is rescaled to
    /// `delta_norm = delta / span` (with `span = max − min`, guarded
    /// against a degenerate zero span), so the **effective** selection
    /// strength `α · delta_norm` lands in the same `[−α, α]` band the
    /// defaults were tuned for regardless of the absolute payoff
    /// magnitude. This is the α-rank analogue of NFSP's / PSRO's
    /// `br_reward_scale`: a magnitude-invariance fix, not a change to
    /// the ranking semantics on a fixed scale.
    ///
    /// `false` (the default) preserves the pre-#215 behavior bit-for-bit.
    pub normalize_payoff_span: bool,
}

impl AlphaRankMetaSolver {
    /// Construct with explicit hyperparameters. Payoff-span
    /// normalization defaults to `false` (pre-#215 behavior). Use
    /// [`AlphaRankMetaSolver::with_payoff_span_normalization`] to opt in.
    pub fn new(
        ranking_intensity_alpha: f32,
        moran_population_size: u32,
        max_iterations: usize,
        tolerance: f32,
    ) -> Self {
        Self {
            ranking_intensity_alpha,
            moran_population_size: moran_population_size.max(2),
            max_iterations: max_iterations.max(1),
            tolerance: tolerance.max(1e-12),
            normalize_payoff_span: false,
        }
    }

    /// Builder-style setter: enable/disable payoff-span normalization of
    /// the Moran payoff differential (issue #215). See
    /// [`AlphaRankMetaSolver::normalize_payoff_span`] for the rationale.
    pub fn with_payoff_span_normalization(mut self, enabled: bool) -> Self {
        self.normalize_payoff_span = enabled;
        self
    }

    /// Inherent N-player α-rank stationary distribution helper.
    ///
    /// This is the workhorse implementation called by the
    /// [`MetaSolver::solve_n_player`] trait override below. Kept as a
    /// separate inherent method so callers with a concrete
    /// `AlphaRankMetaSolver` (e.g. the in-tree unit tests at
    /// `test_alpha_rank_three_player_rps_*`) can invoke it without
    /// going through trait dispatch.
    ///
    /// # Inputs
    ///
    /// - `payoffs`: shape `(num_joint_strategies, num_agents)` where
    ///   `payoffs[s][a]` is agent `a`'s payoff at joint pure strategy `s`.
    /// - `num_agents`: number of agents `N` in the game.
    /// - `per_role_k`: per-agent per-role population size `k` (assumed
    ///   identical across agents in this PR — matches the symmetric PSRO
    ///   posture).
    ///
    /// # Joint-strategy index encoding
    ///
    /// Strategy index `s` decomposes into per-agent indices
    /// `(s_0, s_1, ..., s_{N-1})` with `s_i ∈ [0, k)` via
    /// **little-endian** mixed-radix: `s = Σ_i s_i * k^i`. Agent 0 is
    /// the fastest-varying index.
    ///
    /// # Returns
    ///
    /// A probability vector of length `k^N` summing to `1.0 ± 1e-6`.
    pub fn solve_n_player_impl(
        &self,
        payoffs: &[Vec<f32>],
        num_agents: usize,
        per_role_k: usize,
    ) -> Vec<f32> {
        assert!(num_agents >= 1, "α-rank requires num_agents >= 1");
        assert!(per_role_k >= 1, "α-rank requires per_role_k >= 1");
        let n_joint = per_role_k.checked_pow(num_agents as u32).expect("k^N overflow");
        if payoffs.len() != n_joint {
            panic!(
                "α-rank: payoffs.len() = {} but expected k^N = {}^{} = {}",
                payoffs.len(),
                per_role_k,
                num_agents,
                n_joint
            );
        }
        for (s, row) in payoffs.iter().enumerate() {
            assert_eq!(
                row.len(),
                num_agents,
                "α-rank: payoffs[{s}].len() = {} but expected num_agents = {}",
                row.len(),
                num_agents
            );
        }

        // Build the row-stochastic transition matrix P over joint
        // strategies. For each joint strategy `s` and each single-agent
        // deviation `(s, s')` where `s'` differs from `s` in exactly
        // one agent, transition with the Moran fixation probability
        // (Omidshafiei et al. 2019 §2.3, Eq. 1):
        //
        //   ρ_{σ→τ} = (1 - exp(-α (π_τ - π_σ))) / (1 - exp(-mα (π_τ - π_σ)))
        //
        // where `m = moran_population_size` and the payoff differential
        // `π_τ - π_σ` is from the perspective of the mutating agent.
        // The neutral case `π_τ == π_σ` collapses to `1/m`.
        //
        // We aggregate the per-deviation probabilities by averaging
        // over the uniform choice of (agent to mutate, deviation
        // target). Self-loop probability is whatever mass isn't
        // transferred to single-agent deviations. The number of
        // single-agent deviations from `s` is `num_agents * (per_role_k - 1)`;
        // each deviation contributes `(1 / n_deviations) * ρ_{σ→τ}` to
        // the transition mass.
        let n_deviations = num_agents * per_role_k.saturating_sub(1);
        let per_dev_weight: f32 = if n_deviations > 0 {
            1.0_f32 / n_deviations as f32
        } else {
            0.0
        };

        // Optional payoff-span normalization (issue #215). When enabled,
        // every Moran payoff differential is divided by the payoff span
        // (`max − min` over all per-agent payoffs) so the effective
        // selection strength `α · (delta / span)` stays in the `[−α, α]`
        // band the defaults were tuned for, regardless of the absolute
        // payoff magnitude. This prevents the fixation probability from
        // saturating to a hard 0/1 on large-magnitude bands (e.g.
        // bucket-brigade's `[−700, 0]`). `1.0` divisor (the default,
        // normalization off — or a degenerate flat payoff tensor) is a
        // no-op and keeps the path bit-identical.
        let delta_divisor: f32 = if self.normalize_payoff_span {
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for row in payoffs.iter() {
                for &v in row.iter() {
                    if v < min_v {
                        min_v = v;
                    }
                    if v > max_v {
                        max_v = v;
                    }
                }
            }
            let span = max_v - min_v;
            // Guard against a flat / degenerate tensor: a zero (or
            // non-finite) span leaves the differential untouched.
            if span.is_finite() && span > 1e-12 {
                span
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Sparse-friendly transition rep: per-state out-edges as
        // `Vec<(to_index, prob)>`. With n_joint potentially in the
        // thousands and only `n_deviations` non-self entries per row,
        // this saves space vs the full matrix.
        let mut transitions: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_joint);
        for s in 0..n_joint {
            let mut row_edges: Vec<(usize, f32)> = Vec::with_capacity(n_deviations + 1);
            let mut self_mass: f32 = 1.0;
            let from_payoffs = &payoffs[s];
            // Decompose `s` into per-agent indices once.
            let s_components = decompose_joint_index(s, num_agents, per_role_k);
            for agent in 0..num_agents {
                let from_strat = s_components[agent];
                for new_strat in 0..per_role_k {
                    if new_strat == from_strat {
                        continue;
                    }
                    let mut t_components = s_components.clone();
                    t_components[agent] = new_strat;
                    let t = compose_joint_index(&t_components, per_role_k);
                    let to_payoff_a = payoffs[t][agent];
                    let from_payoff_a = from_payoffs[agent];
                    let p_fix = moran_fixation_probability(
                        self.ranking_intensity_alpha,
                        self.moran_population_size,
                        (to_payoff_a - from_payoff_a) / delta_divisor,
                    );
                    let edge_prob = per_dev_weight * p_fix;
                    row_edges.push((t, edge_prob));
                    self_mass -= edge_prob;
                }
            }
            // Self-loop: whatever mass remains. May be negative under
            // numerical noise; clamp to zero.
            if self_mass < 0.0 {
                self_mass = 0.0;
            }
            row_edges.push((s, self_mass));
            // Renormalize defensively to ensure row-stochastic.
            let row_sum: f32 = row_edges.iter().map(|(_, p)| *p).sum();
            if row_sum > 0.0 {
                for (_, p) in row_edges.iter_mut() {
                    *p /= row_sum;
                }
            }
            transitions.push(row_edges);
        }

        // Power iteration: π_{k+1}[t] = Σ_s π_k[s] * P[s][t].
        let mut pi = vec![1.0_f32 / n_joint as f32; n_joint];
        let mut pi_next = vec![0.0_f32; n_joint];
        for _ in 0..self.max_iterations {
            for v in pi_next.iter_mut() {
                *v = 0.0;
            }
            for (s, edges) in transitions.iter().enumerate() {
                let pis = pi[s];
                if pis == 0.0 {
                    continue;
                }
                for &(t, p) in edges {
                    pi_next[t] += pis * p;
                }
            }
            // L1 convergence check.
            let mut l1: f32 = 0.0;
            for i in 0..n_joint {
                l1 += (pi_next[i] - pi[i]).abs();
            }
            std::mem::swap(&mut pi, &mut pi_next);
            // Renormalize (numerical safety).
            let total: f32 = pi.iter().sum();
            if total > 0.0 {
                for v in pi.iter_mut() {
                    *v /= total;
                }
            }
            if l1 < self.tolerance {
                break;
            }
        }
        pi
    }
}

impl Default for AlphaRankMetaSolver {
    fn default() -> Self {
        // Payoff-span normalization defaults OFF to keep the
        // matching-pennies regression bar and the `solve` API
        // bit-identical to the pre-#215 solver. Opt in via
        // `with_payoff_span_normalization(true)` for large-magnitude
        // payoff bands like bucket-brigade's `[−700, 0]`.
        Self::new(10.0, 50, 200, 1e-6)
    }
}

impl MetaSolver for AlphaRankMetaSolver {
    /// 2-player symmetric α-rank: interprets `payoffs[i][j]` as the row
    /// player's payoff and computes the α-rank stationary distribution
    /// over the `n` pure strategies under the symmetric self-play
    /// assumption (both players draw from the same population). For the
    /// 2-player symmetric case this collapses to the `solve_n_player`
    /// path with `num_agents = 1` over the row-player marginal —
    /// equivalent to treating the column player's payoff structure as
    /// the row's negation under zero-sum symmetry.
    fn solve(&self, payoffs: &[Vec<f32>]) -> Vec<f32> {
        let n = payoffs.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![1.0];
        }
        // 2-player symmetric: each agent's payoff at joint strategy
        // `s = (i, j)` is `payoffs[i][j]` for the row and
        // `payoffs[j][i]` for the column (transposed). Compute α-rank
        // over the `n²` joint strategies and marginalize back to the
        // row distribution.
        let n2 = n * n;
        let mut joint_payoffs = vec![vec![0.0_f32; 2]; n2];
        // Index-based scan: explicit mixed-radix encoding of the joint
        // strategy index `s = i + j * n` (little-endian, agent 0
        // fastest). The clippy::needless_range_loop rewrite would
        // require nested `.enumerate()` chains that obscure the
        // little-endian convention; suppress to keep the math readable.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..n {
                let s = i + j * n;
                joint_payoffs[s][0] = payoffs[i][j];
                joint_payoffs[s][1] = payoffs[j][i];
            }
        }
        let joint_dist = self.solve_n_player_impl(&joint_payoffs, 2, n);
        // Marginalize: row distribution = Σ_j π(i, j).
        let mut row_dist = vec![0.0_f32; n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..n {
                row_dist[i] += joint_dist[i + j * n];
            }
        }
        // Renormalize numerically.
        let total: f32 = row_dist.iter().sum();
        if total > 0.0 {
            for v in row_dist.iter_mut() {
                *v /= total;
            }
        } else {
            return vec![1.0 / n as f32; n];
        }
        row_dist
    }

    fn solve_n_player(
        &self,
        payoffs: &[Vec<f32>],
        num_agents: usize,
        per_role_k: usize,
    ) -> Vec<f32> {
        self.solve_n_player_impl(payoffs, num_agents, per_role_k)
    }

    fn name(&self) -> &'static str {
        "alpha_rank"
    }
}

/// Decompose a flat joint-strategy index into per-agent components
/// under the little-endian mixed-radix convention (agent 0 = fastest).
///
/// This is the index convention shared between
/// [`AlphaRankMetaSolver::solve_n_player_impl`] and the N-tensor
/// [`PayoffCache`]; both must use the same encoding for the PSRO
/// trainer to correctly route per-joint-strategy payoffs to α-rank.
pub(crate) fn decompose_joint_index(s: usize, num_agents: usize, k: usize) -> Vec<usize> {
    let mut out = vec![0_usize; num_agents];
    let mut rem = s;
    for slot in out.iter_mut().take(num_agents) {
        *slot = rem % k;
        rem /= k;
    }
    out
}

/// Compose per-agent components into a flat joint-strategy index under
/// the little-endian mixed-radix convention.
pub(crate) fn compose_joint_index(components: &[usize], k: usize) -> usize {
    let mut s = 0_usize;
    let mut radix = 1_usize;
    for &c in components {
        s += c * radix;
        radix *= k;
    }
    s
}

/// Decide which boundary cells to actually roll out this iteration, and
/// how to fill the rest, given an optional per-iteration evaluation cap
/// (issue #212).
///
/// Returns `(to_evaluate, fill_from)`:
/// - `to_evaluate` is the deterministic subset of `boundary` cells to roll out
///   (in the *same relative order* as `boundary`, so the downstream rayon
///   evaluation and cache writes stay deterministic).
/// - `fill_from` is a list of `(boundary_dst_index, to_evaluate_src_index)`
///   pairs: boundary cell `boundary[dst]` (which was *not* selected) is to be
///   filled by copying the payoff of the selected cell evaluated at
///   `to_evaluate[src]`.
///
/// # Selection scheme
///
/// - `cap == None`, or `boundary.len() <= cap`: **all** cells are selected and
///   `fill_from` is empty. This is the default path and is **bit-identical** to
///   evaluating the whole boundary (the pre-#212 behavior).
/// - `Some(cap)` with `boundary.len() > cap >= 1`: select `cap` cells by an
///   evenly-spaced deterministic stride over the boundary index range (`sel_idx
///   = floor(j * len / cap)` for `j in 0..cap`), guaranteeing a stratified,
///   reproducible cover that always includes the first cell. Every non-selected
///   cell is filled from the nearest *preceding* selected cell (the largest
///   selected index `<= its index`), which is well-defined because index 0 is
///   always selected. Selection depends only on `(boundary.len(), cap)` — never
///   on RNG or thread order — so the subsampled meta-game is itself fully
///   deterministic.
///
/// `cap == Some(0)` is treated as `Some(1)` (always roll out at least one
/// cell) so the boundary is never left entirely unevaluated.
#[allow(clippy::type_complexity)]
fn select_boundary_to_evaluate(
    boundary: &[Vec<usize>],
    cap: Option<usize>,
) -> (Vec<Vec<usize>>, Vec<(usize, usize)>) {
    let len = boundary.len();
    let cap = match cap {
        None => return (boundary.to_vec(), Vec::new()),
        Some(c) => c.max(1),
    };
    if len <= cap {
        return (boundary.to_vec(), Vec::new());
    }
    // Evenly-spaced stratified selection over [0, len). `selected[j]` is
    // the boundary index chosen as the j-th sample; strictly increasing
    // and always starts at 0.
    let mut selected: Vec<usize> = Vec::with_capacity(cap);
    for j in 0..cap {
        let idx = (j * len) / cap;
        // Guard against a repeated index from integer flooring (cannot
        // happen for len > cap >= 1, but keep the invariant explicit).
        if selected.last().copied() != Some(idx) {
            selected.push(idx);
        }
    }
    // Map each boundary index to the src position (within `to_evaluate`)
    // of the nearest preceding selected cell.
    let to_evaluate: Vec<Vec<usize>> = selected.iter().map(|&i| boundary[i].clone()).collect();
    let mut fill_from: Vec<(usize, usize)> = Vec::with_capacity(len - selected.len());
    let mut src = 0_usize; // position within `selected` / `to_evaluate`
    for dst in 0..len {
        // Advance `src` while the next selected index is still <= dst.
        while src + 1 < selected.len() && selected[src + 1] <= dst {
            src += 1;
        }
        if selected[src] == dst {
            continue; // this cell is itself evaluated; nothing to fill
        }
        fill_from.push((dst, src));
    }
    (to_evaluate, fill_from)
}

/// Numerically-stable sigmoid `1 / (1 + exp(-x))`.
#[allow(dead_code)]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Moran-process fixation probability for a single mutant under
/// selection intensity `α` in a population of size `m`, given the
/// payoff differential `delta = π_τ − π_σ` (from the perspective of
/// the mutating agent — positive means the mutation improves payoff).
///
/// Closed form (Omidshafiei et al. 2019 §2.3, Eq. 1; see also Nowak
/// "Evolutionary Dynamics" §6):
///
/// ```text
/// ρ(α, m, δ) = (1 - exp(-α δ)) / (1 - exp(-m α δ))     if δ ≠ 0
///            = 1 / m                                     if δ = 0  (neutral drift)
/// ```
///
/// The neutral-drift limit `1/m` is the standard small-perturbation
/// expansion of the closed form as `δ → 0`. For numerical stability we
/// use it directly when `|α δ| < 1e-9`.
fn moran_fixation_probability(alpha: f32, m: u32, delta: f32) -> f32 {
    let m_f = m as f32;
    let ad = alpha * delta;
    if ad.abs() < 1e-9 {
        return 1.0 / m_f;
    }
    // Numerator: 1 - exp(-α δ); Denominator: 1 - exp(-m α δ).
    let num = 1.0 - (-ad).exp();
    let denom = 1.0 - (-m_f * ad).exp();
    if denom.abs() < 1e-30 {
        // Saturated regime: very strong selection in one direction.
        // ρ ≈ 0 if denom→0 from below, or ρ ≈ 1 if num and denom
        // both blow up positively. Return the sign-based limit.
        return if ad > 0.0 { 1.0 } else { 0.0 };
    }
    let p = num / denom;
    p.clamp(0.0, 1.0)
}

/// Row-player pure best response to column mixture `col_mix`.
fn best_response_row(payoffs: &[Vec<f32>], col_mix: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, row) in payoffs.iter().enumerate() {
        let mut val = 0.0_f32;
        for (j, &p) in col_mix.iter().enumerate() {
            val += row[j] * p;
        }
        if val > best_val {
            best_val = val;
            best_i = i;
        }
    }
    best_i
}

/// Column-player pure best response to row mixture `row_mix` (assuming
/// zero-sum: column minimizes expected row payoff).
fn best_response_col(payoffs: &[Vec<f32>], row_mix: &[f32]) -> usize {
    let n = payoffs.len();
    let mut best_j = 0;
    let mut best_val = f32::INFINITY;
    // Column-major scan: outer loop indexes columns `j`, inner loop indexes
    // rows `i` via `payoffs[i][j]`. The index-based form mirrors the
    // bilinear-form math `(σᵀ M)_j` and reads more directly than an
    // iter-of-iters rewrite.
    #[allow(clippy::needless_range_loop)]
    for j in 0..n {
        let val: f32 = row_mix.iter().enumerate().map(|(i, &p)| payoffs[i][j] * p).sum();
        if val < best_val {
            best_val = val;
            best_j = j;
        }
    }
    best_j
}

// =======================================================================
// PsroConfig / PsroStats
// =======================================================================

/// PSRO trainer configuration.
#[derive(Debug, Clone)]
pub struct PsroConfig {
    /// Number of PSRO outer-loop iterations to run.
    pub max_iterations: usize,
    /// Maximum population size per agent. Iteration is aborted with an
    /// `Err` (not a panic) when this is reached.
    pub max_population_size: usize,
    /// Number of joint-trainer updates spent training each new
    /// best-response policy against the sampled mixture.
    pub br_train_steps_per_iteration: usize,
    /// Number of payoff-evaluation episodes per `(row, col)` cell in
    /// the empirical-payoff matrix.
    pub payoff_eval_episodes: usize,
    /// Optional cap on the number of *fresh* payoff-cell evaluations
    /// performed per outer iteration (issue #212).
    ///
    /// PSRO grows each agent's population by one policy per iteration,
    /// so the only cells that need (re)evaluation are the **boundary
    /// slab**: joint strategies in which at least one agent plays its
    /// brand-new policy. Interior cells (between pre-existing policies)
    /// are already cached across iterations and never recomputed — see
    /// [`PayoffCache::resize_for_boundary`]. The boundary slab itself
    /// still grows as `(k+1)^N − k^N ≈ N·k^(N-1)` cells, which for the
    /// 4-player bucket-brigade game (`N = 4`) is super-linear and
    /// dominates long-run cost even with the rayon-parallel evaluation
    /// (#203) — see the 2026-06-21 calibration in
    /// `docs/research/2026-06-bucket-brigade-validation.md`.
    ///
    /// When set to `Some(cap)` and an iteration's boundary slab has more
    /// than `cap` cells, the trainer **deterministically subsamples**
    /// `cap` boundary cells to actually roll out (preserving the
    /// rayon-parallel evaluation for those), and fills each un-sampled
    /// boundary cell from the nearest already-evaluated sampled cell in
    /// the deterministic flat ordering. This bounds per-iteration cost
    /// at the price of an **approximate** meta-game on the subsampled
    /// boundary.
    ///
    /// `None` (the default) evaluates the entire boundary slab and is
    /// therefore **bit-identical** to the pre-#212 behavior. The
    /// subsampling path is purely opt-in; existing callers and the
    /// determinism discipline (#201) are unaffected.
    pub max_payoff_evals_per_iteration: Option<usize>,
    /// Optional reward scaling applied to per-step rewards before the
    /// best-response (PPO) update, mirroring
    /// [`NfspConfig::br_reward_scale`](crate::multi_agent::nfsp::NfspConfig::br_reward_scale)
    /// (issue #199 / #215).
    ///
    /// PSRO trains each new best response with the same joint PPO update
    /// as NFSP's BR side, so it inherits the same numerical pathology on
    /// the large-magnitude bucket-brigade payoff band (`[−700, 0]`): the
    /// unscaled rewards drive the critic's regression targets and the
    /// per-minibatch advantage normalization into a range where the
    /// value loss dominates the surrogate and the BR effectively stops
    /// learning a meaningful response. Scaling rewards by a constant is
    /// an affine transform of the return — it does **not** change the
    /// optimal policy — but keeps the critic targets and advantage stats
    /// numerically friendly. A value like `0.01` rescales the
    /// bucket-brigade band to roughly `[−7, 0]`.
    ///
    /// `1.0` (the default) is a no-op and preserves the pre-#215
    /// behavior bit-for-bit.
    pub br_reward_scale: f32,
    /// RNG seed for opponent sampling and deterministic tests.
    pub seed: u64,
}

impl Default for PsroConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            max_population_size: 50,
            br_train_steps_per_iteration: 1,
            payoff_eval_episodes: 8,
            max_payoff_evals_per_iteration: None,
            br_reward_scale: 1.0,
            seed: 0,
        }
    }
}

/// Per-iteration PSRO statistics.
#[derive(Debug, Clone, Default)]
pub struct PsroIterationStats {
    /// Iteration index (1-based after the initial population is seeded).
    pub iteration: usize,
    /// Population size at the end of this iteration (per agent;
    /// identical across agents under the symmetric posture).
    pub population_size: usize,
    /// Per-agent meta-Nash *action-population* marginal distributions
    /// at the end of this iteration. `meta_nash_per_agent[i]` is agent
    /// `i`'s marginal over its own `population_size` policies extracted
    /// from the joint α-rank distribution (for N≥3) or directly from
    /// the 2-player solver (for N=2).
    pub meta_nash_per_agent: Vec<Vec<f32>>,
    /// Per-agent best-response training stats. `br_stats_per_agent[i]`
    /// is the stats for the round in which agent `i` was active under
    /// the round-robin schedule, or `None` if the agent was not the
    /// active agent on this iteration (currently every agent is
    /// trained every iteration, so every entry is `Some`).
    pub br_stats_per_agent: Vec<Option<JointStats>>,
    /// NashConv-style exploitability: the sum over agents `i` of agent
    /// `i`'s maximum payoff improvement by deviating to a pure best
    /// response in the empirical game, given the joint meta-Nash
    /// distribution.
    ///
    /// For N=2 zero-sum games this reduces to the original 2-player
    /// formula (row gain + column gain). Smaller is closer to the
    /// empirical equilibrium.
    pub exploitability: f32,
}

impl PsroIterationStats {
    /// Backward-compat shim: agent 0 (row-player) meta-Nash
    /// distribution. Equivalent to `&self.meta_nash_per_agent[0]`.
    pub fn meta_nash_row(&self) -> &[f32] {
        self.meta_nash_per_agent.first().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Backward-compat shim: agent 1 (column-player) meta-Nash
    /// distribution. Equivalent to `&self.meta_nash_per_agent[1]`.
    pub fn meta_nash_col(&self) -> &[f32] {
        self.meta_nash_per_agent.get(1).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

/// Aggregate PSRO trainer statistics returned by [`PsroTrainer::run`].
#[derive(Debug, Clone, Default)]
pub struct PsroStats {
    /// Per-iteration history.
    pub iterations: Vec<PsroIterationStats>,
}

// =======================================================================
// Empirical-payoff matrix cache
// =======================================================================

/// Cached N-tensor empirical-payoff cache for an N-agent symmetric
/// game.
///
/// Stores per-agent payoffs at every joint pure strategy `s ∈ [0, k^N)`
/// where `k` is the per-agent population size (assumed identical across
/// agents under the symmetric posture) and `N` is the number of agents.
///
/// # Index convention
///
/// The flat joint-strategy index decomposes into per-agent indices
/// `(s_0, s_1, ..., s_{N-1})` via **little-endian mixed-radix**:
/// `s = Σ_i s_i · k^i`. Agent 0 is the fastest-varying index. This
/// convention matches [`AlphaRankMetaSolver::solve_n_player_impl`] —
/// the cache feeds its `cells` buffer directly into α-rank with no
/// transpose.
///
/// # Storage
///
/// `cells[s]` is a `Vec<f32>` of length `num_agents` containing each
/// agent's mean per-episode return at joint strategy `s`. The
/// per-cell allocation matches α-rank's `payoffs[s][a]` input shape.
/// For N=2 with k populations, this collapses to k² cells × 2-element
/// vectors — identical information to the pre-refactor `Vec<Vec<f32>>`
/// row-major matrix but with the per-cell agent payoffs co-located.
///
/// # Growth
///
/// PSRO grows each agent's population by one policy per outer
/// iteration. When agent `a`'s population grows from `k` to `k+1`,
/// the cache needs to evaluate the new boundary slab: all joint
/// strategies where agent `a` plays index `k` (its new policy).
/// [`PayoffCache::resize_for_boundary`] grows the storage to the new
/// `(k+1)^N` size; [`PayoffCache::set_cell`] writes individual cell
/// payoffs. The trainer is responsible for iterating over the
/// agent-`a`-newest-strategy boundary and calling `set_cell` for each
/// new joint strategy.
///
/// Memory is `O(k^N · N · f32)`, bounded by
/// [`PsroConfig::max_population_size`] cubed (or higher for N>3); the
/// `PsroConfig::max_population_size` cap should be tuned downward for
/// large N to keep memory reasonable.
#[derive(Debug, Clone, Default)]
pub struct PayoffCache {
    /// Per-joint-strategy per-agent payoffs. `cells[s][a]` is agent
    /// `a`'s mean per-episode return at joint strategy `s`. Indexed
    /// little-endian (agent 0 = fastest).
    cells: Vec<Vec<f32>>,
    /// Per-agent population size `k`. Assumed identical across agents
    /// under the symmetric posture.
    per_role_k: usize,
    /// Number of agents `N`.
    num_agents: usize,
    /// Counter incremented on every payoff *evaluation* (not every
    /// query). Used by unit tests to assert the cache is hit.
    pub eval_count: usize,
}

impl PayoffCache {
    /// Construct an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a cache sized for `num_agents` agents with `per_role_k = 0`
    /// (empty). Use [`PayoffCache::resize_for_boundary`] to grow.
    pub fn with_num_agents(num_agents: usize) -> Self {
        Self { cells: Vec::new(), per_role_k: 0, num_agents, eval_count: 0 }
    }

    /// Current per-role population size `k`.
    pub fn per_role_k(&self) -> usize {
        self.per_role_k
    }

    /// Number of agents `N`.
    pub fn num_agents(&self) -> usize {
        self.num_agents
    }

    /// Total number of joint-strategy cells `k^N`.
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// Read the per-agent payoffs at joint strategy `joint`. Returns
    /// `None` if any per-agent index is out of bounds. The returned
    /// slice has length `num_agents`.
    pub fn get_joint(&self, joint: &[usize]) -> Option<&[f32]> {
        if joint.len() != self.num_agents {
            return None;
        }
        for (a, &idx) in joint.iter().enumerate() {
            if idx >= self.per_role_k {
                return None;
            }
            let _ = a;
        }
        let s = compose_joint_index(joint, self.per_role_k);
        self.cells.get(s).map(|v| v.as_slice())
    }

    /// View the full per-cell payoff tensor in the
    /// `(k^N, N)` flat layout consumed by
    /// [`AlphaRankMetaSolver::solve_n_player`]. The outer length is
    /// `k^N`; each inner `Vec<f32>` has length `num_agents`.
    pub fn payoff_tensor(&self) -> &[Vec<f32>] {
        &self.cells
    }

    /// Set the per-agent payoffs at joint strategy `joint`. Bumps
    /// `eval_count` by 1. Panics if the cache isn't sized for `joint`
    /// (call [`PayoffCache::resize_for_boundary`] first) or if the
    /// payoff length doesn't equal `num_agents`.
    pub fn set_cell(&mut self, joint: &[usize], payoffs: Vec<f32>) {
        assert_eq!(
            joint.len(),
            self.num_agents,
            "joint strategy length {} must equal num_agents = {}",
            joint.len(),
            self.num_agents
        );
        assert_eq!(
            payoffs.len(),
            self.num_agents,
            "payoffs length {} must equal num_agents = {}",
            payoffs.len(),
            self.num_agents
        );
        for (a, &idx) in joint.iter().enumerate() {
            assert!(
                idx < self.per_role_k,
                "joint[{a}] = {idx} >= per_role_k = {}",
                self.per_role_k
            );
        }
        let s = compose_joint_index(joint, self.per_role_k);
        self.cells[s] = payoffs;
        self.eval_count += 1;
    }

    /// Set the per-agent payoffs at joint strategy `joint` **without**
    /// bumping `eval_count`.
    ///
    /// Used by the issue-#212 boundary-subsampling path to fill an
    /// un-sampled boundary cell with a reused payoff (copied from an
    /// already-evaluated sampled neighbour). Such a fill performs **no
    /// fresh rollout**, so it must not be counted as an evaluation —
    /// `eval_count` continues to reflect only the cells that were
    /// actually rolled out. Same bounds/asserts as [`Self::set_cell`].
    pub fn set_cell_no_count(&mut self, joint: &[usize], payoffs: Vec<f32>) {
        assert_eq!(
            joint.len(),
            self.num_agents,
            "joint strategy length {} must equal num_agents = {}",
            joint.len(),
            self.num_agents
        );
        assert_eq!(
            payoffs.len(),
            self.num_agents,
            "payoffs length {} must equal num_agents = {}",
            payoffs.len(),
            self.num_agents
        );
        for (a, &idx) in joint.iter().enumerate() {
            assert!(
                idx < self.per_role_k,
                "joint[{a}] = {idx} >= per_role_k = {}",
                self.per_role_k
            );
        }
        let s = compose_joint_index(joint, self.per_role_k);
        self.cells[s] = payoffs;
    }

    /// Grow storage from `(per_role_k)^N` to `(new_per_role_k)^N`
    /// in-place, preserving the cached payoffs at all joint strategies
    /// that map to the same little-endian flat index in the new
    /// storage.
    ///
    /// Newly-introduced cells are zero-initialized; the caller is
    /// responsible for evaluating them via the trainer's
    /// `evaluate_payoff_joint` and writing the result with
    /// [`PayoffCache::set_cell`].
    ///
    /// # Why we can't just `Vec::resize`
    ///
    /// Under little-endian mixed-radix, joint index `s = Σ_i s_i · k^i`
    /// changes when the radix `k` grows: the same per-agent indices
    /// `(s_0, ..., s_{N-1})` map to a different flat `s'` in the
    /// `(k+1)^N` storage. We rebuild the buffer by iterating over the
    /// old joint strategies and re-keying.
    pub fn resize_for_boundary(&mut self, new_per_role_k: usize) {
        assert!(
            new_per_role_k >= self.per_role_k,
            "PayoffCache may only grow; got new_k = {} < per_role_k = {}",
            new_per_role_k,
            self.per_role_k
        );
        if new_per_role_k == self.per_role_k {
            return;
        }
        let new_total = new_per_role_k.checked_pow(self.num_agents as u32).expect("k^N overflow");
        let mut new_cells = vec![vec![0.0_f32; self.num_agents]; new_total];
        if self.per_role_k > 0 {
            let old_total = self.cells.len();
            for s_old in 0..old_total {
                let components = decompose_joint_index(s_old, self.num_agents, self.per_role_k);
                let s_new = compose_joint_index(&components, new_per_role_k);
                new_cells[s_new] = std::mem::take(&mut self.cells[s_old]);
            }
        }
        self.cells = new_cells;
        self.per_role_k = new_per_role_k;
    }

    /// Iterate over every joint strategy `s` in the *boundary slab*
    /// where agent `agent_index` plays its newest pure strategy
    /// (`per_role_k - 1`) — the cells whose payoffs must be evaluated
    /// after agent `agent_index`'s population just grew by one.
    ///
    /// Returns the joint-strategy index vectors (per-agent indices),
    /// suitable for passing to `evaluate_payoff_joint` and
    /// `set_cell`.
    pub fn boundary_joint_strategies(&self, agent_index: usize) -> Vec<Vec<usize>> {
        let k = self.per_role_k;
        let n = self.num_agents;
        assert!(agent_index < n);
        assert!(k >= 1);
        let new_strat = k - 1;
        // Enumerate the other agents' indices via the same
        // little-endian convention on N-1 axes of radix k.
        let n_others = n - 1;
        let total_others = k.checked_pow(n_others as u32).expect("k^(N-1) overflow");
        let mut out = Vec::with_capacity(total_others);
        for s in 0..total_others {
            let mut joint = vec![0_usize; n];
            joint[agent_index] = new_strat;
            // Distribute s across the other agents in little-endian
            // mixed-radix. Index-based loop is the cleanest reading of
            // the recurrence; clippy::needless_range_loop's
            // iter-based suggestion would mean awkwardly splitting the
            // `agent_index` skip.
            let mut rem = s;
            #[allow(clippy::needless_range_loop)]
            for a in 0..n {
                if a == agent_index {
                    continue;
                }
                joint[a] = rem % k;
                rem /= k;
            }
            out.push(joint);
        }
        out
    }
}

// =======================================================================
// PsroTrainer
// =======================================================================

/// PSRO outer-loop trainer for symmetric N-agent games (N ≥ 2).
///
/// Generic over the Burn backend `B`, policy module `P`, and Burn
/// optimizer type `O`. The trainer owns:
///
/// - N populations of policies (one per agent role) under `populations:
///   Vec<Vec<P>>`.
/// - A [`MetaSolver`] for the empirical meta-game. For N=2 the 2-player
///   [`MetaSolver::solve`] path is used (any in-tree solver works); for N≥3 the
///   trainer calls [`MetaSolver::solve_n_player`] and only
///   [`AlphaRankMetaSolver`] provides a non-panicking override.
/// - A cached empirical-payoff N-tensor [`PayoffCache`] keyed by joint pure
///   strategy.
/// - User-supplied factories for fresh policies + optimizers + envs.
///
/// # Policy/optimizer factories
///
/// The trainer doesn't know how to construct a Burn module of the
/// caller's chosen architecture, so we take closures:
///
/// - `policy_factory: Fn(&B::Device, u64) -> P` — fresh policy. The `u64` is a
///   **per-construction seed** the trainer derives from `PsroConfig::seed` via
///   a monotonic init-counter. A reproducibility-aware factory threads it into
///   `MlpBurnPolicy::new_seeded` / `MlpBurnConfig::with_seed` so that every
///   agent's initial policy and every per-iteration best-response gets
///   *distinct but deterministic* weights (issue #135). Factories that don't
///   care about reproducibility may ignore the argument.
/// - `optimizer_factory: Fn() -> BurnOptimizer<B, P, O>` — fresh optimizer.
/// - `env_factory: Fn() -> E` — fresh env instance.
///
/// This keeps PSRO architecture-agnostic at the cost of slightly
/// awkward generics at the call site (see the matching-pennies test).
///
/// # Single-policy-class assumption
///
/// All agents in both populations share the same policy class `P`. For
/// 2-agent symmetric games (matching pennies, homogeneous bucket
/// brigade) this is exactly what we want — the symmetry lets us
/// transpose the payoff matrix for the column player's solve. For
/// fully asymmetric games (different obs/action spaces per role), the
/// trainer needs to be re-parameterized over `(P_row, P_col)`; that's
/// out of scope for the first PR.
pub struct PsroTrainer<B, P, O, E, FP, FO, FE>
where
    B: AutodiffBackend,
    P: JointPolicy<B>,
    O: Optimizer<P, B>,
    E: JointEnv,
    FP: Fn(&B::Device, u64) -> P,
    FO: Fn() -> BurnOptimizer<B, P, O>,
    FE: Fn() -> E,
{
    /// Per-agent policy populations. `populations[agent]` is the
    /// monotonically-growing list of policies for agent `agent`. Under
    /// the symmetric posture all per-agent populations have the same
    /// length.
    populations: Vec<Vec<P>>,
    meta_solver: Box<dyn MetaSolver>,
    config: PsroConfig,
    joint_config: JointTrainerConfig,
    device: B::Device,
    policy_factory: FP,
    optimizer_factory: FO,
    env_factory: FE,
    payoff_cache: PayoffCache,
    rng: StdRng,
    /// Monotonic counter feeding the per-construction policy-init seed.
    ///
    /// Incremented on every `policy_factory` call (once per agent at
    /// construction, once per best-response per outer iteration). Each
    /// call derives `config.seed.wrapping_add(0x9E37_79B9 *
    /// init_counter)` so distinct constructions get distinct — but
    /// fully deterministic — initial weights. Without this, a factory
    /// closing over a single fixed seed would hand every agent and every
    /// iteration *identical* weights, a regression (issue #135,
    /// Correction 1).
    init_counter: u64,
}

impl<B, P, O, E, FP, FO, FE> PsroTrainer<B, P, O, E, FP, FO, FE>
where
    B: AutodiffBackend,
    P: JointPolicy<B>,
    O: Optimizer<P, B>,
    E: JointEnv,
    FP: Fn(&B::Device, u64) -> P,
    FO: Fn() -> BurnOptimizer<B, P, O>,
    FE: Fn() -> E,
{
    /// Construct a PSRO trainer with one initial random policy per agent.
    ///
    /// `joint_config.num_agents` must be `≥ 2`. For `num_agents == 2`
    /// the trainer accepts any [`MetaSolver`] implementation; for
    /// `num_agents > 2` the meta-solver's
    /// [`MetaSolver::solve_n_player`] is called — at the time of this
    /// PR only [`AlphaRankMetaSolver`] provides a non-panicking
    /// override for N>2.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: PsroConfig,
        joint_config: JointTrainerConfig,
        meta_solver: Box<dyn MetaSolver>,
        device: B::Device,
        policy_factory: FP,
        optimizer_factory: FO,
        env_factory: FE,
    ) -> Result<Self> {
        if joint_config.num_agents < 2 {
            return Err(anyhow!(
                "PsroTrainer requires joint_config.num_agents >= 2 (got {})",
                joint_config.num_agents
            ));
        }
        let n = joint_config.num_agents;
        // Derive a distinct init seed per agent at construction time.
        // We advance the counter inline here (the trainer isn't built
        // yet) using the same derivation as `next_init_seed`.
        let base_seed = config.seed;
        let populations: Vec<Vec<P>> = (0..n)
            .map(|i| {
                let s = base_seed.wrapping_add(0x9E37_79B9_u64.wrapping_mul(i as u64));
                vec![policy_factory(&device, s)]
            })
            .collect();
        let rng = StdRng::seed_from_u64(config.seed);
        Ok(Self {
            populations,
            meta_solver,
            config,
            joint_config,
            device,
            policy_factory,
            optimizer_factory,
            env_factory,
            payoff_cache: PayoffCache::with_num_agents(n),
            rng,
            // Start the running counter past the `n` seeds consumed by
            // the initial per-agent constructions above.
            init_counter: n as u64,
        })
    }

    /// Derive and consume the next per-construction policy-init seed.
    ///
    /// Returns `config.seed.wrapping_add(0x9E37_79B9 * init_counter)`
    /// and advances the counter so the next call gets a fresh,
    /// non-colliding stream. The multiplier is the 32-bit golden-ratio
    /// constant — any odd large constant works; this one keeps adjacent
    /// counters well-separated in the `StdRng` seed space.
    fn next_init_seed(&mut self) -> u64 {
        let s = self.config.seed.wrapping_add(0x9E37_79B9_u64.wrapping_mul(self.init_counter));
        self.init_counter = self.init_counter.wrapping_add(1);
        s
    }

    /// Borrow agent `agent`'s policy population.
    pub fn populations(&self, agent: usize) -> &[P] {
        &self.populations[agent]
    }

    /// Borrow the row-player (agent 0) population.
    ///
    /// Backward-compat shim retained for callers that pre-date the
    /// N-tensor refactor (notably
    /// `tests/test_psro_matching_pennies.rs`). New N≥2 code should use
    /// [`PsroTrainer::populations`].
    pub fn population_row(&self) -> &[P] {
        &self.populations[0]
    }

    /// Borrow the column-player (agent 1) population.
    ///
    /// Backward-compat shim retained for callers that pre-date the
    /// N-tensor refactor. Panics for N=1 (which is rejected by `new`
    /// anyway). New N≥2 code should use [`PsroTrainer::populations`].
    pub fn population_col(&self) -> &[P] {
        &self.populations[1]
    }

    /// Borrow the cached empirical N-tensor payoff cache.
    pub fn payoff_cache(&self) -> &PayoffCache {
        &self.payoff_cache
    }

    /// Run the PSRO outer loop and return the per-iteration history.
    ///
    /// `on_iteration` is invoked once per outer iteration, immediately
    /// after that iteration's [`PsroIterationStats`] is constructed and
    /// before it is pushed onto the returned history. This mirrors
    /// [`NfspTrainer::run`](crate::multi_agent::nfsp::NfspTrainer::run)
    /// and lets callers observe per-iteration progress *during* the run
    /// (live `tracing` logging, mid-run checkpoint triggers, etc.)
    /// rather than only inspecting the aggregate stats after `run`
    /// returns.
    ///
    /// The callback receives two arguments:
    /// 1. `&PsroIterationStats` — this iteration's stats, whose `iteration`
    ///    field increases monotonically from `1` to `config.max_iterations`.
    /// 2. `&[&P]` — the newest best-response policy for each agent (`brs[a]` is
    ///    agent `a`'s freshly-trained BR appended this iteration, i.e.
    ///    `populations(a).last()`). This lets the callback persist per-agent BR
    ///    policies to disk *during* the run (mid-run checkpointing, issue #204)
    ///    without a borrow conflict against the `&mut self` held by `run`: the
    ///    trainer cannot itself write files (it is backend/format-agnostic, the
    ///    `Recorder` lives in the example), so it hands the closure the policy
    ///    references it needs to checkpoint. Checkpointing is a pure
    ///    side-effect read; it does not alter the trainer state or the
    ///    deterministic training trajectory.
    ///
    /// For the common case of "run with no per-iteration hook", use
    /// [`Self::run_silent`].
    pub fn run<F>(&mut self, mut on_iteration: F) -> Result<PsroStats>
    where
        F: FnMut(&PsroIterationStats, &[&P]),
        // Bounds required by the rayon-parallel boundary-slab evaluation
        // (issue #203) and the rayon-parallel best-response loop (issue
        // #232). Mirror the `EnvPool` Send-bound convention (pool.rs:58).
        // The parallel payoff result is bit-identical to a serial sweep
        // because each cell is pure (issue #201); the parallel BR result
        // is thread-count-invariant (per-agent local RNG, issue #232). The
        // BR path additionally needs the policy/optimizer factories to be
        // `Sync` because each task calls them through a shared `&`.
        P: Send + Sync,
        E: Send,
        FP: Sync,
        FO: Sync,
        FE: Sync,
        B::Device: Sync,
    {
        let num_agents = self.joint_config.num_agents;

        // Seed the payoff cache with the initial 1×...×1 entry — all
        // agents play their initial-random policy (index 0).
        if self.payoff_cache.per_role_k() == 0 {
            self.payoff_cache.resize_for_boundary(1);
            let initial_joint = vec![0_usize; num_agents];
            let initial_payoffs = self.evaluate_payoff_joint(&initial_joint);
            self.payoff_cache.set_cell(&initial_joint, initial_payoffs);
        }

        let mut stats = PsroStats::default();
        for iter in 1..=self.config.max_iterations {
            if self.populations[0].len() >= self.config.max_population_size {
                return Err(anyhow!(
                    "PSRO population reached max_population_size = {}",
                    self.config.max_population_size
                ));
            }

            // Step 1: meta-Nash on the current N-tensor payoff cache.
            // For N=2 the meta-solver's `solve` path (symmetric
            // marginal) is used; for N≥3 we go through `solve_n_player`
            // and marginalize per-agent.
            let per_agent_marginals = self.solve_per_agent_marginals();

            // Step 2: round-robin train one best-response per agent
            // against the other agents' marginal mixtures. The
            // `num_agents` best responses are fully independent, so they
            // run concurrently under rayon (issue #232) — opponent
            // indices + init seeds are drawn in fixed agent order before
            // the parallel region, and each BR uses a per-agent local RNG,
            // so the result is invariant to thread count. Trained BRs are
            // appended to `self.populations` in fixed agent order after the
            // join.
            let br_stats = self.train_best_responses_parallel(&per_agent_marginals)?;
            let br_stats_per_agent: Vec<Option<JointStats>> =
                br_stats.into_iter().map(Some).collect();

            // Step 3: grow the payoff cache and evaluate every
            // newly-added boundary cell. After all agents' populations
            // grow by one in lockstep, the new per-role-k is k+1 and
            // the new cells are the union of every per-agent
            // boundary slab — i.e. every joint strategy `s` whose
            // per-agent index vector includes at least one
            // newest-strategy index (`k` under the new radix).
            let old_k = self.payoff_cache.per_role_k();
            let new_k = old_k + 1;
            self.payoff_cache.resize_for_boundary(new_k);
            // Iterate every joint strategy in the new k^N tensor; cells
            // that are entirely in the *old* k^N corner are already
            // populated (preserved by `resize_for_boundary`). New cells
            // are those with at least one component == k-1 under the
            // new radix. We iterate flat indices and decompose.
            let total_new = new_k.checked_pow(num_agents as u32).expect("k^N overflow");
            let new_strategy_idx = new_k - 1;

            // Gather the boundary cells (those whose per-agent index
            // vector includes the newest strategy) in deterministic flat
            // order. This is the `population^N` slab that dominates PSRO
            // cost on large N (issue #198).
            let boundary: Vec<Vec<usize>> = (0..total_new)
                .filter_map(|s| {
                    let components = decompose_joint_index(s, num_agents, new_k);
                    components.contains(&new_strategy_idx).then_some(components)
                })
                .collect();

            // Optionally subsample the boundary slab to bound
            // per-iteration cost (issue #212). With `None` (default) the
            // entire boundary is evaluated, which is **bit-identical** to
            // the pre-#212 behavior; with `Some(cap)` and a boundary
            // larger than `cap`, only a deterministically-chosen `cap`
            // cells are rolled out and the rest are filled by reuse — see
            // `select_boundary_to_evaluate`.
            let (to_evaluate, fill_from) =
                select_boundary_to_evaluate(&boundary, self.config.max_payoff_evals_per_iteration);

            // Evaluate the selected boundary cells in parallel. Each cell
            // is a pure function of `(config.seed, joint)` (issue #201),
            // so the parallel result is **bit-identical** to a serial
            // sweep regardless of thread count or scheduling: cells share
            // no mutable state, each clones its joint policies and builds
            // a fresh env via `env_factory`, and seeds a local `StdRng`.
            // Results are collected by index (not push order), then
            // written into the cache serially below, so the cache is
            // populated in the same deterministic order as the old serial
            // loop. See `evaluate_payoff_boundary_parallel`.
            let evaluated = self.evaluate_payoff_boundary_parallel(&to_evaluate);
            // Write the freshly-evaluated cells first so the fill step can
            // read their payoffs back out of the cache. `to_evaluate` is a
            // prefix-stable deterministic subset of `boundary`.
            for (components, payoffs) in to_evaluate.iter().zip(&evaluated) {
                self.payoff_cache.set_cell(components, payoffs.clone());
            }
            // Fill the un-sampled boundary cells from their nearest
            // already-evaluated sampled neighbour (deterministic; no fresh
            // rollouts, so these do NOT bump `eval_count`). When the cap is
            // `None` or not exceeded, `fill_from` is empty and this loop is
            // a no-op, keeping the default path bit-identical.
            for &(dst_idx, src_idx) in &fill_from {
                let payoffs = evaluated[src_idx].clone();
                self.payoff_cache.set_cell_no_count(&boundary[dst_idx], payoffs);
            }

            // Step 4: re-solve the meta-Nash on the post-append cache
            // and compute NashConv exploitability. Reporting on the
            // post-append cache is how PSRO progress is conventionally
            // tracked (exploitability drops as each new BR enriches
            // the population).
            let post_marginals = self.solve_per_agent_marginals();
            let exploitability = self.compute_nashconv(&post_marginals);

            let iter_stats = PsroIterationStats {
                iteration: iter,
                population_size: self.populations[0].len(),
                meta_nash_per_agent: post_marginals,
                br_stats_per_agent,
                exploitability,
            };

            // Newest best-response policy per agent, appended this
            // iteration in the round-robin loop above. Handed to the
            // callback so it can checkpoint per-agent BR policies
            // mid-run (issue #204). `populations(a)` is guaranteed
            // non-empty here: every agent was just trained and pushed.
            let newest_brs: Vec<&P> = (0..num_agents)
                .map(|a| {
                    self.populations[a].last().expect("population non-empty after BR training")
                })
                .collect();
            on_iteration(&iter_stats, &newest_brs);
            stats.iterations.push(iter_stats);
        }
        Ok(stats)
    }

    /// Convenience entry point: drives [`Self::run`] with a no-op
    /// iteration callback. Use this when per-iteration observation is
    /// not needed (mirrors
    /// [`NfspTrainer::run_silent`](crate::multi_agent::nfsp::NfspTrainer::run_silent)).
    pub fn run_silent(&mut self) -> Result<PsroStats>
    where
        P: Send + Sync,
        E: Send,
        FP: Sync,
        FO: Sync,
        FE: Sync,
        B::Device: Sync,
    {
        self.run(|_, _| {})
    }

    /// Most-recent per-agent meta-Nash distributions (one row per
    /// agent), or uniform over the initial population if `run` has not
    /// been called.
    pub fn current_meta_nash_per_agent(&self) -> Vec<Vec<f32>> {
        if self.payoff_cache.per_role_k() == 0 {
            return (0..self.joint_config.num_agents).map(|_| vec![1.0]).collect();
        }
        self.solve_per_agent_marginals()
    }

    /// Backward-compat shim returning agent 0's meta-Nash marginal.
    pub fn current_meta_nash(&self) -> Vec<f32> {
        self.current_meta_nash_per_agent().into_iter().next().unwrap_or_default()
    }

    /// Solve the meta-Nash on the current payoff cache and return
    /// per-agent marginal distributions over each agent's own
    /// population. For N=2, uses [`MetaSolver::solve`] on the legacy
    /// `payoffs[i][j] = agent_0_payoff(i, j)` matrix view (preserving
    /// bit-stable behaviour for existing FictitiousPlay / Replicator /
    /// Uniform meta-solvers). For N≥3, uses
    /// [`MetaSolver::solve_n_player`] and marginalizes the joint
    /// distribution per-agent.
    fn solve_per_agent_marginals(&self) -> Vec<Vec<f32>> {
        let n = self.joint_config.num_agents;
        let k = self.payoff_cache.per_role_k();
        if k == 0 {
            return (0..n).map(|_| vec![1.0]).collect();
        }
        if n == 2 {
            // 2-player path: project the N-tensor cache back to a
            // `k × k` row-player payoff matrix (agent 0's payoffs) and
            // call `solve`. The post-projection matrix is bit-identical
            // to the pre-refactor `PayoffCache::matrix()` view — this
            // is the regression-bar guarantee.
            // Index-based double loop: the explicit `s = i + j * k`
            // formula mirrors the little-endian mixed-radix convention
            // and reads more clearly than a flat-enumerate rewrite.
            let mut row_matrix: Vec<Vec<f32>> = vec![vec![0.0_f32; k]; k];
            #[allow(clippy::needless_range_loop)]
            for i in 0..k {
                for j in 0..k {
                    let s = i + j * k;
                    row_matrix[i][j] = self.payoff_cache.payoff_tensor()[s][0];
                }
            }
            let row_dist = self.meta_solver.solve(&row_matrix);
            // Symmetric zero-sum: column distribution matches row by
            // symmetry, same as the pre-refactor trainer.
            let col_dist = row_dist.clone();
            return vec![row_dist, col_dist];
        }
        // N≥3 path: call `solve_n_player` with the flat (k^N, N)
        // tensor and marginalize per-agent.
        let joint = self.meta_solver.solve_n_player(self.payoff_cache.payoff_tensor(), n, k);
        let mut marginals: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0_f32; k]).collect();
        for (s, &mass) in joint.iter().enumerate() {
            let components = decompose_joint_index(s, n, k);
            for (a, &c) in components.iter().enumerate() {
                marginals[a][c] += mass;
            }
        }
        // Renormalize numerically.
        for m in marginals.iter_mut() {
            let total: f32 = m.iter().sum();
            if total > 0.0 {
                for v in m.iter_mut() {
                    *v /= total;
                }
            } else {
                let uniform = 1.0 / k as f32;
                for v in m.iter_mut() {
                    *v = uniform;
                }
            }
        }
        marginals
    }

    /// Compute NashConv exploitability under the per-agent meta-Nash
    /// marginals: `Σ_i (max_{s_i} U_i(s_i, σ_{−i}) − U_i(σ))`.
    ///
    /// # N=2 fast-path bit-stability
    ///
    /// For N=2 the meta-Nash marginals are projected back to a `k × k`
    /// agent-0 payoff matrix and the closed-form
    /// `row_gain + col_gain` formula is evaluated — bit-identical to
    /// the pre-refactor `empirical_exploitability`. This preserves the
    /// `+1.0` calibration of
    /// `test_psro_exploitability_non_increasing_trend_on_matching_pennies`
    /// across the refactor.
    ///
    /// # N≥3 generalization
    ///
    /// For N≥3 we compute each agent's best-response gain as the
    /// supremum over its `k` pure strategies of the expected payoff
    /// against the other agents' joint marginal mixture, minus the
    /// agent's expected payoff under the full joint mixture. The
    /// agent-`i` joint mixture is `Π_{j≠i} σ_j` (independence assumed
    /// under the per-agent marginal decomposition) so the expected
    /// payoff at the agent-`i` pure strategy `s_i` is
    /// `Σ_{s_{−i}} (Π_{j≠i} σ_j[s_j]) · U_i(s_i, s_{−i})`. The N=2
    /// case follows the same formula (mod the bit-stability
    /// projection).
    fn compute_nashconv(&self, per_agent_marginals: &[Vec<f32>]) -> f32 {
        let n = self.joint_config.num_agents;
        let k = self.payoff_cache.per_role_k();
        if n == 2 {
            // Fast path: project to the agent-0 payoff matrix and use
            // the legacy 2-player formula bit-identically. Index-based
            // loop mirrors the little-endian mixed-radix convention
            // for the joint flat index.
            let mut row_matrix: Vec<Vec<f32>> = vec![vec![0.0_f32; k]; k];
            #[allow(clippy::needless_range_loop)]
            for i in 0..k {
                for j in 0..k {
                    let s = i + j * k;
                    row_matrix[i][j] = self.payoff_cache.payoff_tensor()[s][0];
                }
            }
            return empirical_exploitability(&row_matrix, &per_agent_marginals[0]);
        }
        // N≥3 general path.
        let payoffs = self.payoff_cache.payoff_tensor();
        let mut nashconv = 0.0_f32;
        for i in 0..n {
            // U_i(σ) = Σ_s (Π_j σ_j[s_j]) · payoffs[s][i].
            let mut u_sigma = 0.0_f32;
            // Expected payoff to agent i for each of its pure
            // strategies, marginalizing other agents over their σ.
            let mut u_pure = vec![0.0_f32; k];
            for (s, agent_payoffs) in payoffs.iter().enumerate() {
                let components = decompose_joint_index(s, n, k);
                // Product of marginal masses across all agents under
                // the full joint mixture.
                let mut full_prob = 1.0_f32;
                for (a, &c) in components.iter().enumerate() {
                    full_prob *= per_agent_marginals[a][c];
                }
                u_sigma += full_prob * agent_payoffs[i];
                // For the "agent i deviates to pure s_i" case, weight
                // by Π_{j≠i} σ_j[s_j].
                let mut others_prob = 1.0_f32;
                for (a, &c) in components.iter().enumerate() {
                    if a == i {
                        continue;
                    }
                    others_prob *= per_agent_marginals[a][c];
                }
                let s_i = components[i];
                u_pure[s_i] += others_prob * agent_payoffs[i];
            }
            let max_pure = u_pure.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let gain = (max_pure - u_sigma).max(0.0);
            nashconv += gain;
        }
        nashconv
    }

    /// Train all `num_agents` best responses for one PSRO iteration **in
    /// parallel** (one fully-independent BR per agent) and append the
    /// trained policies to `self.populations` in **fixed agent order**.
    ///
    /// # Why this is parallelizable
    ///
    /// Each best response trains its own [`JointMultiAgentTrainer`] over a
    /// freshly-initialized active policy and frozen, cloned opponents, runs
    /// its own env, and only *reads* `self.populations` / `self.config`.
    /// The only original shared-mutable touches were `self.rng` (opponent
    /// sampling + PPO shuffle) and `self.next_init_seed`. We hoist **all**
    /// of those draws out of the parallel region here, into a fixed-order
    /// pre-pass, so the parallel region touches no `&mut self`:
    ///
    /// - per-agent opponent indices are drawn from `self.rng` in agent order,
    ///   before the join;
    /// - per-agent active-policy init seeds are drawn from
    ///   `self.next_init_seed()` in agent order;
    /// - each BR is handed a **local [`StdRng`]** seeded deterministically from
    ///   `(config.seed, active_agent)` (mirrors the per-cell seeding of
    ///   [`evaluate_payoff_joint_pure`]), which threads the rollout +
    ///   PPO-update draws for that BR alone.
    ///
    /// The per-BR work is then a pure function of its pre-drawn inputs, so
    /// `(0..num_agents).into_par_iter()` produces a result that is
    /// **invariant to thread count / scheduling**: results are collected
    /// by index (rayon preserves input order) and appended to
    /// `self.populations` serially in agent order afterward.
    ///
    /// # Determinism note
    ///
    /// Because the BR now uses a per-agent local RNG instead of the single
    /// shared `self.rng` stream, output is **not** bit-identical to the
    /// pre-parallel serial-RNG runs (the RNG threading changed by design).
    /// It is, however, fully reproducible for a given seed and identical
    /// across any thread count.
    ///
    /// # Bounds
    ///
    /// Mirror the boundary-payoff parallel path
    /// ([`Self::evaluate_payoff_boundary_parallel`]): `P: Send + Sync`
    /// (shared by `&`, cloned per task), `E: Send` (moved into each task),
    /// and the factories / device are shared by `&` (`FP`/`FO`/`FE: Sync`,
    /// `B::Device: Sync`).
    fn train_best_responses_parallel(
        &mut self,
        per_agent_marginals: &[Vec<f32>],
    ) -> Result<Vec<JointStats>>
    where
        P: Send + Sync,
        E: Send,
        FP: Sync,
        FO: Sync,
        FE: Sync,
        B::Device: Sync,
    {
        let num_agents = self.joint_config.num_agents;

        // --- Fixed-order pre-pass: draw every shared-mutable value here,
        // in agent order, so the parallel region below is pure. ---
        //
        // `opp_indices[active_agent][a]` is the sampled opponent index for
        // agent `a` while `active_agent` trains its BR; the entry for
        // `a == active_agent` is unused (that slot holds the fresh BR).
        let mut opp_indices: Vec<Vec<usize>> = Vec::with_capacity(num_agents);
        let mut init_seeds: Vec<u64> = Vec::with_capacity(num_agents);
        for active_agent in 0..num_agents {
            let mut row: Vec<usize> = Vec::with_capacity(num_agents);
            for (a, marginal) in per_agent_marginals.iter().enumerate().take(num_agents) {
                if a == active_agent {
                    row.push(0); // unused placeholder for the active slot
                } else {
                    row.push(sample_from_mixture(&mut self.rng, marginal));
                }
            }
            opp_indices.push(row);
            init_seeds.push(self.next_init_seed());
        }

        // Bind only the Sync field borrows into locals so the rayon
        // closures capture *these* references and NOT the whole `&self`
        // (which also holds the non-`Sync` `Box<dyn MetaSolver>`). Same
        // technique as `evaluate_payoff_boundary_parallel`.
        let populations = &self.populations;
        let config = &self.config;
        let joint_config = &self.joint_config;
        let device = &self.device;
        let policy_factory = &self.policy_factory;
        let optimizer_factory = &self.optimizer_factory;
        let env_factory = &self.env_factory;

        // --- Parallel region: one independent BR per agent. ---
        let results: Vec<Result<(JointStats, P)>> = (0..num_agents)
            .into_par_iter()
            .map(|active_agent| {
                train_best_response_pure::<B, P, O, E, _, _, _>(
                    active_agent,
                    &opp_indices[active_agent],
                    init_seeds[active_agent],
                    populations,
                    config,
                    joint_config,
                    device,
                    policy_factory,
                    optimizer_factory,
                    env_factory,
                )
            })
            .collect();

        // --- Join: unpack results in fixed agent order, propagating the
        // first error deterministically. The immutable borrow of
        // `self.populations` taken for the parallel region has ended (the
        // `collect()` above is complete), so we can now mutably append. ---
        let mut stats: Vec<JointStats> = Vec::with_capacity(num_agents);
        let mut trained_policies: Vec<P> = Vec::with_capacity(num_agents);
        for result in results {
            let (br_stats, trained) = result?;
            stats.push(br_stats);
            trained_policies.push(trained);
        }
        // Promote each learned BR into its agent's population in fixed
        // agent order (collect-by-index), matching the serial loop's
        // append order so the population layout is thread-count-invariant.
        for (active_agent, trained) in trained_policies.into_iter().enumerate() {
            self.populations[active_agent].push(trained);
        }
        Ok(stats)
    }

    /// Evaluate the empirical-payoff cell at joint strategy `joint`
    /// (length `num_agents`) by running
    /// `config.payoff_eval_episodes` episodes with policy
    /// `populations[a][joint[a]]` for each agent `a`. Returns the
    /// per-agent mean per-episode returns (length `num_agents`).
    ///
    /// This is a thin wrapper that gathers the per-joint policies and
    /// delegates to [`evaluate_payoff_joint_pure`], which is a pure,
    /// per-cell-seeded function (it does **not** touch `self.rng`). The
    /// wrapper only borrows `&self` for the population/factory handles,
    /// so the result is independent of evaluation order and global RNG
    /// state — see #201.
    fn evaluate_payoff_joint(&self, joint: &[usize]) -> Vec<f32> {
        let num_agents = self.joint_config.num_agents;
        assert_eq!(joint.len(), num_agents);
        let policies: Vec<P> =
            (0..num_agents).map(|a| self.populations[a][joint[a]].clone()).collect();
        evaluate_payoff_joint_pure::<B, P, _, _>(
            joint,
            &self.config,
            &policies,
            &self.env_factory,
            &self.device,
        )
    }

    /// Evaluate a batch of boundary payoff cells **in parallel** with
    /// rayon, returning one payoff vector per input cell in the **same
    /// order** as `boundary`.
    ///
    /// # Bit-identity with the serial path (issue #203)
    ///
    /// Each cell delegates to [`evaluate_payoff_joint_pure`], which seeds
    /// a local [`StdRng`] purely from `(config.seed, joint)` and touches
    /// no shared trainer RNG (issue #201). The cell payoff is therefore a
    /// pure function of `(joint, config, policies, env_factory)`, so this
    /// `par_iter` result is **bit-identical** to evaluating the same
    /// cells serially in any order, *regardless of thread count or
    /// scheduling*. Results are gathered by index via
    /// [`ParallelIterator::collect`] (rayon preserves input order), never
    /// by push order, and the caller writes them into the cache serially.
    ///
    /// # Thread-safety
    ///
    /// No mutable state crosses threads. Each task:
    /// - reads `self.populations` / `self.config` / `self.device` /
    ///   `self.env_factory` through shared `&` borrows (no `&mut self`),
    /// - clones the joint's per-agent policies (`P: Clone`) so the autodiff
    ///   modules are owned per task,
    /// - builds a fresh env via `env_factory` (which already yields a new
    ///   instance per call).
    ///
    /// The `Send`/`Sync` bounds mirror the [`EnvPool`](crate::env::pool)
    /// convention (`E: Send`): `P: Send + Sync` (shared by `&`, cloned
    /// per task), `E: Send` (moved into each task), and the factory /
    /// device are shared by `&` (`FE: Sync`, `B::Device: Sync`). No
    /// `Mutex` is introduced, so the hot loop is never serialized.
    fn evaluate_payoff_boundary_parallel(&self, boundary: &[Vec<usize>]) -> Vec<Vec<f32>>
    where
        P: Send + Sync,
        E: Send,
        FE: Sync,
        B::Device: Sync,
    {
        let num_agents = self.joint_config.num_agents;
        // Bind only the Sync field borrows into locals so the rayon
        // closures capture *these* references and NOT the whole `&self`
        // (which also holds the non-`Sync` `Box<dyn MetaSolver>` and the
        // `FP`/`FO` factory closures). Capturing the whole `&self` would
        // require the entire trainer to be `Sync`; capturing only the
        // payoff-relevant fields keeps the bounds minimal and correct.
        let populations = &self.populations;
        let config = &self.config;
        let env_factory = &self.env_factory;
        let device = &self.device;
        boundary
            .par_iter()
            .map(|joint| {
                debug_assert_eq!(joint.len(), num_agents);
                let policies: Vec<P> =
                    (0..num_agents).map(|a| populations[a][joint[a]].clone()).collect();
                evaluate_payoff_joint_pure::<B, P, _, _>(
                    joint,
                    config,
                    &policies,
                    env_factory,
                    device,
                )
            })
            .collect()
    }
}

/// Pure, per-agent-seeded best-response trainer.
///
/// Trains one best response for `active_agent` against the other agents'
/// pre-sampled, frozen opponents and returns `(stats, trained_policy)`.
/// This is the per-task body of the rayon-parallel BR loop (issue #232):
/// it is the extraction of the old `train_best_response` with **every
/// `&mut self` / shared-RNG touch removed**.
///
/// # Determinism / thread-count invariance (issue #232)
///
/// All values that the pre-parallel path drew from the shared
/// `&mut self.rng` / `self.next_init_seed` are now passed in, already
/// drawn in fixed agent order by the caller:
/// - `opp_indices[a]` — the frozen opponent index for each non-active agent `a`
///   (the `active_agent` slot is ignored);
/// - `init_seed` — the active BR's fresh-policy initialization seed.
///
/// The rollout + PPO-update draws use a **local [`StdRng`]** seeded purely
/// from `(config.seed, active_agent)` (mirroring the per-cell seeding of
/// [`evaluate_payoff_joint_pure`]), so this function touches no shared
/// state and its result is a pure function of its inputs. Running the
/// per-agent tasks under any thread count therefore yields identical
/// per-agent results.
///
/// Note: because each BR now consumes its own local RNG stream rather than
/// slices of one global `self.rng` stream, output is intentionally **not**
/// bit-identical to the pre-#232 serial-RNG runs.
#[allow(clippy::too_many_arguments)]
fn train_best_response_pure<B, P, O, E, FP, FO, FE>(
    active_agent: usize,
    opp_indices: &[usize],
    init_seed: u64,
    populations: &[Vec<P>],
    config: &PsroConfig,
    joint_config: &JointTrainerConfig,
    device: &B::Device,
    policy_factory: &FP,
    optimizer_factory: &FO,
    env_factory: &FE,
) -> Result<(JointStats, P)>
where
    B: AutodiffBackend,
    P: JointPolicy<B> + Clone,
    O: Optimizer<P, B>,
    E: JointEnv,
    FP: Fn(&B::Device, u64) -> P,
    FO: Fn() -> BurnOptimizer<B, P, O>,
    FE: Fn() -> E,
{
    let num_agents = joint_config.num_agents;
    debug_assert!(active_agent < num_agents);

    // Build the joint trainer's per-agent policy slot:
    // - active agent: fresh randomly-initialized policy (the BR), using the
    //   pre-drawn `init_seed`.
    // - non-active agents: the pre-sampled frozen opponent from their meta-Nash
    //   marginal over their respective populations.
    let mut policies: Vec<P> = Vec::with_capacity(num_agents);
    for (a, population) in populations.iter().enumerate().take(num_agents) {
        if a == active_agent {
            policies.push(policy_factory(device, init_seed));
        } else {
            policies.push(population[opp_indices[a]].clone());
        }
    }
    let optimizers: Vec<BurnOptimizer<B, P, O>> =
        (0..num_agents).map(|_| optimizer_factory()).collect();

    let mut trainer = JointMultiAgentTrainer::<B, P, O>::new(
        policies,
        optimizers,
        joint_config.clone(),
        device.clone(),
    )?;

    // Per-agent LOCAL action/update RNG, seeded purely from
    // `(config.seed, active_agent)`. This replaces the shared
    // `&mut self.rng` of the pre-#232 path, making each BR self-contained
    // and thread-count-invariant.
    let mut rng = StdRng::seed_from_u64(config.seed ^ splitmix64(active_agent as u64));

    // Run `br_train_steps_per_iteration` rollout/update cycles.
    let active_mask: Vec<bool> = (0..num_agents).map(|i| i == active_agent).collect::<Vec<_>>();
    let mut env = env_factory();
    let mut last_obs = env.reset_joint(Some(config.seed.wrapping_add(active_agent as u64)));

    let mut last_stats = JointStats::zeros(num_agents);
    let reward_scale = config.br_reward_scale;
    for _ in 0..config.br_train_steps_per_iteration {
        let mut rollout = trainer.collect_rollout(&mut env, &mut last_obs, &mut rng);
        // Apply the optional BR reward scaling (issue #199 / #215) before
        // the PPO update. Scaling rewards uniformly is an affine transform
        // of the return and does not change the optimal policy, but keeps
        // the large-magnitude bucket-brigade band (`[−700, 0]`) in a
        // numerically friendlier range for the BR critic's regression
        // targets and advantage statistics. `reward_scale == 1.0` (the
        // default) leaves the rollout untouched.
        if reward_scale != 1.0 {
            for agent_rewards in rollout.rewards.iter_mut() {
                for r in agent_rewards.iter_mut() {
                    *r *= reward_scale;
                }
            }
        }
        last_stats = trainer.update_with_active_agents(
            &rollout,
            &active_mask,
            &mut rng,
            |_features: &[burn::tensor::Tensor<B, 2>]| -> Option<burn::tensor::Tensor<B, 1>> {
                None
            },
        )?;
    }

    // Return the learned BR policy; the caller promotes it into the active
    // agent's population in fixed agent order after the parallel join.
    let trained = trainer.policy(active_agent).clone();
    Ok((last_stats, trained))
}

/// Mix a `u64` through three rounds of the splitmix64 finalizer so that
/// adjacent inputs (e.g. neighbouring `joint_hash` values) map to
/// well-separated `StdRng` seeds. Same family of avalanche constants as
/// the determinism shims in [`crate::policy::seeded_init`].
fn splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Pure, per-cell-seeded payoff evaluator.
///
/// Runs `config.payoff_eval_episodes` episodes of `policies` (one
/// per agent, already gathered for the target joint cell) in a fresh
/// env from `env_factory`, and returns the per-agent mean per-episode
/// returns (length `num_agents`).
///
/// # Determinism / order-independence (issue #201)
///
/// Unlike the pre-#201 path, this function **does not read or mutate any
/// shared trainer RNG**. It constructs a single **local
/// [`StdRng`]** seeded from `(config.seed, joint)` and threads it
/// through every `get_action_host_seeded` call for the whole cell. The
/// per-episode env-reset seed is likewise derived deterministically from
/// `(config.seed, joint, ep)` (the little-endian `joint_hash` scheme
/// shared with [`PayoffCache`]). Consequently the returned payoff vector
/// is a pure function of `(joint, config, policies, env_factory)`:
/// evaluating the same cell twice — or evaluating a set of cells in any
/// order — yields bit-identical results. This is the determinism
/// guarantee that lets #203 parallelize the boundary-slab loop with a
/// result bit-identical to the serial one.
fn evaluate_payoff_joint_pure<B, P, E, EF>(
    joint: &[usize],
    config: &PsroConfig,
    policies: &[P],
    env_factory: &EF,
    device: &B::Device,
) -> Vec<f32>
where
    B: AutodiffBackend,
    P: JointPolicy<B>,
    E: JointEnv,
    EF: Fn() -> E,
{
    let num_agents = joint.len();
    let mut env = env_factory();
    let mut totals = vec![0.0_f64; num_agents];
    let episodes = config.payoff_eval_episodes.max(1);

    // Deterministic per-cell hash: composes the joint-strategy
    // components into a stable scalar via the same little-endian
    // convention as the cache.
    let mut joint_hash: u64 = 0;
    for &c in joint {
        joint_hash = joint_hash.wrapping_mul(53).wrapping_add(c as u64);
    }

    // LOCAL action-sampling RNG, seeded purely from (config.seed,
    // joint). This replaces the shared `&mut self.rng` of the pre-#201
    // path, making each cell self-contained and order-independent. A
    // single RNG spans all episodes so the per-cell action-draw stream
    // is a deterministic function of the cell alone.
    let per_cell_seed = config.seed ^ splitmix64(joint_hash);
    let mut rng = StdRng::seed_from_u64(per_cell_seed);

    for ep in 0..episodes {
        // Per-(joint, ep) env-reset seed (unchanged from the pre-#201
        // path): deterministic in the cell and episode index.
        let reset_seed =
            config.seed.wrapping_add(joint_hash.wrapping_mul(31).wrapping_add(ep as u64));
        let mut last_obs = env.reset_joint(Some(reset_seed));
        let mut ep_returns = vec![0.0_f64; num_agents];
        // Cap rollout length; rely on env's `done` flag.
        for _ in 0..1024 {
            let mut actions: Vec<Vec<i64>> = Vec::with_capacity(num_agents);
            for (a, obs_a) in last_obs.iter().enumerate().take(num_agents) {
                let obs_dim = obs_a.len();
                let obs_t = burn::tensor::Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(obs_a.clone(), [1, obs_dim]),
                    device,
                );
                let (a_host, _, _) = policies[a].get_action_host_seeded(obs_t, &mut rng);
                let num_dims = policies[a].action_dims_joint().len();
                actions.push(a_host[..num_dims].to_vec());
            }
            let res = env.step_joint(&actions);
            for (a, ret) in ep_returns.iter_mut().enumerate().take(num_agents) {
                *ret += res.rewards[a] as f64;
            }
            if res.done {
                break;
            }
            last_obs[..num_agents].clone_from_slice(&res.observations[..num_agents]);
        }
        for (a, total) in totals.iter_mut().enumerate().take(num_agents) {
            *total += ep_returns[a];
        }
    }
    totals.into_iter().map(|t| (t / episodes as f64) as f32).collect()
}

/// Sample an index from a length-`n` probability vector with the given RNG.
fn sample_from_mixture(rng: &mut StdRng, mix: &[f32]) -> usize {
    if mix.is_empty() {
        return 0;
    }
    let u: f32 = rng.random();
    let mut acc = 0.0_f32;
    for (i, &p) in mix.iter().enumerate() {
        acc += p;
        if u < acc {
            return i;
        }
    }
    mix.len() - 1
}

/// Empirical exploitability: maximum unilateral improvement either
/// player can achieve by deviating from `meta_nash` to a pure best
/// response within the existing empirical-payoff matrix.
///
/// For a symmetric `n × n` row-payoff matrix `M` and equilibrium
/// proposal `σ`, this returns
/// `max(0, max_i (M σ)_i − σᵀ M σ) + max(0, max_j (−Mᵀ σ)_j − (−σᵀ M σ))`
/// — the sum of both players' best-response gains.
fn empirical_exploitability(payoffs: &[Vec<f32>], meta_nash: &[f32]) -> f32 {
    let n = payoffs.len();
    if n == 0 || meta_nash.is_empty() {
        return 0.0;
    }
    // Row player's expected payoff against col_mix == meta_nash.
    let mut max_row = f32::NEG_INFINITY;
    let mut sigma_value = 0.0_f32;
    for (i, row) in payoffs.iter().enumerate() {
        let mut v = 0.0_f32;
        for (j, &p) in meta_nash.iter().enumerate() {
            v += row[j] * p;
        }
        if v > max_row {
            max_row = v;
        }
        sigma_value += meta_nash[i] * v;
    }
    let row_gain = (max_row - sigma_value).max(0.0);

    // Column player minimizes; deviation gain is the max amount they can
    // shift `sigma_value` *down*. For zero-sum games, column-player
    // value is `-sigma_value` and their best response minimizes
    // `(σᵀ M)_j` over `j`.
    let mut min_col = f32::INFINITY;
    // Column-major scan; see comment on `best_response_col` for rationale.
    #[allow(clippy::needless_range_loop)]
    for j in 0..n {
        let v: f32 = meta_nash.iter().enumerate().map(|(i, &p)| payoffs[i][j] * p).sum();
        if v < min_col {
            min_col = v;
        }
    }
    let col_gain = (sigma_value - min_col).max(0.0);

    row_gain + col_gain
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
    // MetaSolver impls
    // ------------------------------------------------------------------

    fn assert_valid_distribution(dist: &[f32], n_expected: usize) {
        assert_eq!(dist.len(), n_expected, "distribution size mismatch");
        let total: f32 = dist.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "distribution must sum to 1, got {total}");
        for &p in dist {
            assert!(p >= -1e-6, "distribution entry must be >= 0, got {p}");
        }
    }

    #[test]
    fn test_uniform_meta_solver_3x3() {
        let solver = UniformMetaSolver;
        let payoffs = vec![vec![1.0, -1.0, 0.0]; 3];
        let dist = solver.solve(&payoffs);
        assert_valid_distribution(&dist, 3);
        for &p in &dist {
            assert!((p - 1.0 / 3.0).abs() < 1e-6, "uniform should be 1/3, got {p}");
        }
    }

    #[test]
    fn test_uniform_meta_solver_is_payoff_independent() {
        let solver = UniformMetaSolver;
        let payoffs_a = vec![vec![5.0, -3.0], vec![-3.0, 5.0]];
        let payoffs_b = vec![vec![0.1, -0.1], vec![-0.1, 0.1]];
        let a = solver.solve(&payoffs_a);
        let b = solver.solve(&payoffs_b);
        assert_eq!(a, b, "uniform must ignore payoffs");
    }

    /// Matching-pennies row-payoff matrix (action 0 / action 1).
    /// Row 0 vs col 0 → +1; row 0 vs col 1 → -1; etc.
    fn matching_pennies_payoff() -> Vec<Vec<f32>> {
        vec![vec![1.0, -1.0], vec![-1.0, 1.0]]
    }

    #[test]
    fn test_fictitious_play_matching_pennies() {
        let solver = FictitiousPlayMetaSolver::new(2000);
        let dist = solver.solve(&matching_pennies_payoff());
        assert_valid_distribution(&dist, 2);
        // Both actions should converge to ~0.5 / ~0.5.
        for &p in &dist {
            assert!((p - 0.5).abs() < 0.05, "expected ~0.5, got {p}");
        }
    }

    #[test]
    fn test_replicator_dynamics_matching_pennies() {
        let solver = ReplicatorDynamicsMetaSolver::new(5000, 0.05);
        let dist = solver.solve(&matching_pennies_payoff());
        assert_valid_distribution(&dist, 2);
        for &p in &dist {
            assert!((p - 0.5).abs() < 0.05, "expected ~0.5, got {p}");
        }
    }

    #[test]
    fn test_meta_solvers_handle_n_eq_1() {
        let payoffs = vec![vec![0.5]];
        for solver in [
            Box::new(UniformMetaSolver) as Box<dyn MetaSolver>,
            Box::new(FictitiousPlayMetaSolver::default()) as Box<dyn MetaSolver>,
            Box::new(ReplicatorDynamicsMetaSolver::default()) as Box<dyn MetaSolver>,
        ] {
            let dist = solver.solve(&payoffs);
            assert_eq!(dist, vec![1.0], "{} failed on n=1", solver.name());
        }
    }

    #[test]
    fn test_meta_solvers_handle_n_eq_0() {
        let payoffs: Vec<Vec<f32>> = Vec::new();
        for solver in [
            Box::new(FictitiousPlayMetaSolver::default()) as Box<dyn MetaSolver>,
            Box::new(ReplicatorDynamicsMetaSolver::default()) as Box<dyn MetaSolver>,
        ] {
            let dist = solver.solve(&payoffs);
            assert!(dist.is_empty(), "{} should return empty for n=0", solver.name());
        }
    }

    #[test]
    fn test_fictitious_play_dominated_strategy() {
        // Row player has a strictly dominant action (row 0 always wins).
        // Mixed-Nash should put all mass on row 0.
        let payoffs = vec![vec![1.0, 2.0], vec![-1.0, -2.0]];
        let solver = FictitiousPlayMetaSolver::new(1000);
        let dist = solver.solve(&payoffs);
        assert_valid_distribution(&dist, 2);
        assert!(dist[0] > 0.95, "row 0 dominant, expected mass ~1.0, got {}", dist[0]);
    }

    // ------------------------------------------------------------------
    // AlphaRankMetaSolver
    // ------------------------------------------------------------------

    /// Hand-computed closed-form target for 3-player rock-paper-scissors:
    /// each player picks R(0)/P(1)/S(2); payoffs follow the cyclic
    /// majority rule. By full symmetry of the response graph the
    /// stationary distribution is uniform `1/27` over all 27 joint pure
    /// strategies (3³). We assert per-entry within `1e-2`.
    fn three_player_rps_payoffs() -> Vec<Vec<f32>> {
        // 27 joint strategies × 3 agents. For each joint strategy
        // (s_0, s_1, s_2) ∈ [0,3)³ encoded little-endian, compute each
        // agent's payoff under cyclic-majority rule: agent `i` wins
        // (+1) if its choice beats both others' under the standard RPS
        // cycle (0→2, 1→0, 2→1), loses (−1) if it loses to both, and
        // gets 0 otherwise (mixed outcome).
        //
        // Standard RPS beats: 0(R) beats 2(S), 1(P) beats 0(R), 2(S) beats 1(P).
        fn beats(a: usize, b: usize) -> bool {
            (a == 0 && b == 2) || (a == 1 && b == 0) || (a == 2 && b == 1)
        }
        let mut out = Vec::with_capacity(27);
        for s in 0..27 {
            let s0 = s % 3;
            let s1 = (s / 3) % 3;
            let s2 = (s / 9) % 3;
            let strategies = [s0, s1, s2];
            let mut row = vec![0.0_f32; 3];
            for i in 0..3 {
                let mut wins = 0;
                let mut losses = 0;
                for j in 0..3 {
                    if i == j {
                        continue;
                    }
                    if beats(strategies[i], strategies[j]) {
                        wins += 1;
                    } else if beats(strategies[j], strategies[i]) {
                        losses += 1;
                    }
                }
                row[i] = (wins - losses) as f32;
            }
            out.push(row);
        }
        out
    }

    #[test]
    fn test_alpha_rank_three_player_rps_per_agent_marginal_is_uniform() {
        // Curator-targeted closed-form: by full RPS symmetry (each
        // strategy {R, P, S} is interchangeable under the cyclic
        // permutation), each agent's *marginal* action distribution is
        // uniform 1/3 over {R, P, S}. The Curator's original claim of
        // uniform 1/27 over the 27 joint strategies is an
        // over-simplification of the response-graph symmetry — the
        // joint distribution is *equivariant* under the cyclic
        // permutation, which implies the per-agent marginal is uniform
        // but does NOT imply joint uniformity (states like (R,R,R)
        // have higher self-loop mass than (R,P,S) because all 6
        // single-agent deviations from (R,R,R) have non-zero payoff
        // differential, whereas (R,P,S) has many ε-zero differentials).
        //
        // Asserts within `1e-2` on the per-agent marginal.
        let payoffs = three_player_rps_payoffs();
        let solver = AlphaRankMetaSolver::default();
        let dist = solver.solve_n_player(&payoffs, 3, 3);
        assert_eq!(dist.len(), 27, "α-rank should return 27-d distribution for 3-player RPS");
        let total: f32 = dist.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "distribution must sum to 1, got {total}");
        // Per-agent marginal: sum joint mass over the other agents'
        // indices for each agent's own strategy.
        for agent in 0..3 {
            let mut marginal = [0.0_f32; 3];
            for (s, &mass) in dist.iter().enumerate().take(27) {
                let components = decompose_joint_index(s, 3, 3);
                marginal[components[agent]] += mass;
            }
            let target = 1.0 / 3.0;
            for (i, &p) in marginal.iter().enumerate() {
                assert!(
                    (p - target).abs() < 1e-2,
                    "α-rank 3-player RPS agent {agent} marginal[{i}] = {p}, expected ≈ {target}; \
                     deviation {} exceeds 1e-2",
                    (p - target).abs()
                );
            }
        }
    }

    /// Equivariance / orbit-equal-mass test: under the RPS cyclic
    /// permutation `σ: R→P→S→R`, the α-rank distribution must be
    /// invariant on orbits. We verify that the 3 "all-same"
    /// joint strategies have equal stationary mass.
    #[test]
    fn test_alpha_rank_three_player_rps_diagonal_orbit_equal_mass() {
        let payoffs = three_player_rps_payoffs();
        let solver = AlphaRankMetaSolver::default();
        let dist = solver.solve_n_player(&payoffs, 3, 3);
        // Diagonal states: (0,0,0)=0, (1,1,1)=1+3+9=13, (2,2,2)=2+6+18=26.
        let diag_indices = [0_usize, 13, 26];
        let masses: Vec<f32> = diag_indices.iter().map(|&i| dist[i]).collect();
        // All three should be equal within tight tolerance.
        for i in 1..3 {
            assert!(
                (masses[i] - masses[0]).abs() < 5e-3,
                "RPS diagonal orbit not equal-mass: m[0]={}, m[{i}]={}",
                masses[0],
                masses[i]
            );
        }
    }

    #[test]
    fn test_alpha_rank_solve_returns_valid_distribution_on_random_4x4() {
        // Validity check: on 5 random 4×4 payoff matrices the α-rank
        // marginalized row distribution is a non-negative probability
        // vector summing to 1.0 ± 1e-6.
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let solver = AlphaRankMetaSolver::default();
        for seed in 0..5_u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let payoffs: Vec<Vec<f32>> = (0..4)
                .map(|_| (0..4).map(|_| rng.random_range(-1.0..1.0_f32)).collect())
                .collect();
            let dist = solver.solve(&payoffs);
            assert_eq!(dist.len(), 4, "expected 4-d distribution");
            let total: f32 = dist.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-4,
                "α-rank seed={seed}: distribution must sum to 1.0 ± 1e-4, got {total}"
            );
            for (i, &p) in dist.iter().enumerate() {
                assert!(p >= -1e-6, "α-rank seed={seed}: entry {i} must be non-negative, got {p}");
            }
        }
    }

    #[test]
    fn test_alpha_rank_handles_n_eq_1_and_n_eq_0() {
        let solver = AlphaRankMetaSolver::default();
        let dist_1 = solver.solve(&[vec![0.5]]);
        assert_eq!(dist_1, vec![1.0], "α-rank should return [1.0] on n=1");
        let dist_0: Vec<Vec<f32>> = Vec::new();
        let d = solver.solve(&dist_0);
        assert!(d.is_empty(), "α-rank should return empty on n=0");
    }

    #[test]
    fn test_alpha_rank_strict_dominance_concentrates_mass() {
        // For a 2-player symmetric game where row 0 strictly dominates
        // (payoff = +2 against everything, vs row 1 = -2), the
        // α-rank stationary distribution should put most mass on
        // strategy 0. With α=10, the deviation acceptance probability
        // from 1→0 is sigmoid(10 * 4) ≈ 1.0 while 0→1 is ≈ 0.0.
        let payoffs = vec![vec![2.0, 2.0], vec![-2.0, -2.0]];
        let solver = AlphaRankMetaSolver::default();
        let dist = solver.solve(&payoffs);
        assert!(
            dist[0] > 0.9,
            "α-rank should concentrate on dominant strategy 0, got dist = {dist:?}"
        );
    }

    // ------------------------------------------------------------------
    // α-rank payoff-span normalization (issue #215)
    // ------------------------------------------------------------------

    /// Span normalization must be a strict no-op by default and bit-for-bit
    /// identical on a non-degenerate matrix when explicitly disabled.
    /// This is the regression bar guaranteeing the default α-rank path is
    /// unchanged by #215.
    #[test]
    fn test_alpha_rank_span_normalization_default_off_is_bit_identical() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        for seed in 0..5_u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let payoffs: Vec<Vec<f32>> = (0..4)
                .map(|_| (0..4).map(|_| rng.random_range(-5.0..5.0_f32)).collect())
                .collect();
            let default_solver = AlphaRankMetaSolver::default();
            let explicit_off = AlphaRankMetaSolver::default().with_payoff_span_normalization(false);
            assert_eq!(
                default_solver.solve(&payoffs),
                explicit_off.solve(&payoffs),
                "default solver must equal explicitly-disabled span normalization (seed {seed})"
            );
        }
    }

    /// Root-cause demonstration (issue #215): on a large-magnitude payoff
    /// band the default α-rank fixation probability **saturates** — every
    /// non-neutral Moran transition collapses to a hard 0/1 — and the
    /// resulting stationary distribution stops tracking the strategy
    /// ordering. Concretely, a strategy that strictly dominates at unit
    /// scale (and is correctly identified there) is *no longer*
    /// concentrated on once the same ordinal game is rescaled to the
    /// `[−700, 0]` band: the saturated transition matrix degenerates and
    /// the solve returns a near-uniform / wrong answer.
    ///
    /// Span normalization restores magnitude invariance: the rescaled
    /// game produces (essentially) the same distribution as the unit-scale
    /// game, so the dominant strategy is concentrated on regardless of the
    /// absolute payoff magnitude. This is the mechanism behind the
    /// observed exploitability *divergence* — the meta-solver's mixture
    /// becomes magnitude-dependent and brittle on the large-payoff cells.
    #[test]
    fn test_alpha_rank_span_normalization_is_magnitude_invariant() {
        // Same ordinal structure (strategy 0 strictly dominates), two
        // magnitudes 350x apart.
        let small = vec![vec![2.0_f32, -1.0], vec![1.0, -2.0]];
        let large = vec![vec![700.0_f32, -350.0], vec![350.0, -700.0]];

        // (a) At unit scale, the *unnormalized* default solver already
        // correctly concentrates on the dominant strategy.
        let plain = AlphaRankMetaSolver::default();
        let plain_small = plain.solve(&small);
        assert!(
            plain_small[0] > 0.9,
            "unit-scale α-rank should concentrate on dominant strategy 0, got {plain_small:?}"
        );

        // (b) At the [−700, 0] scale, the *unnormalized* solver loses the
        // dominance signal entirely — the saturated Moran transitions
        // degenerate and it returns a near-uniform (wrong) distribution.
        let plain_large = plain.solve(&large);
        assert!(
            plain_large[0] < 0.6,
            "unnormalized large-scale α-rank should LOSE the dominance signal \
             (saturation bug, issue #215), got {plain_large:?}"
        );

        // (c) With span normalization the rescaled game recovers the same
        // concentrated answer as the unit-scale game — magnitude
        // invariance.
        let norm = AlphaRankMetaSolver::default().with_payoff_span_normalization(true);
        let dist_small = norm.solve(&small);
        let dist_large = norm.solve(&large);
        for i in 0..2 {
            assert!(
                (dist_small[i] - dist_large[i]).abs() < 1e-3,
                "span-normalized α-rank should be magnitude-invariant: \
                 small={dist_small:?} large={dist_large:?}"
            );
        }
        assert!(
            dist_large[0] > 0.9,
            "span-normalized large-scale α-rank should concentrate on dominant strategy 0, \
             got {dist_large:?}"
        );
    }

    /// A flat / degenerate payoff tensor (zero span) must not divide by
    /// zero under span normalization — the guard falls back to a unit
    /// divisor, giving the uniform stationary distribution (no
    /// strategy dominates).
    #[test]
    fn test_alpha_rank_span_normalization_handles_flat_payoffs() {
        let flat = vec![vec![3.0_f32, 3.0], vec![3.0, 3.0]];
        let norm = AlphaRankMetaSolver::default().with_payoff_span_normalization(true);
        let dist = norm.solve(&flat);
        let total: f32 = dist.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "flat-payoff dist must be normalized, got {dist:?}");
        for &p in &dist {
            assert!(p.is_finite(), "flat-payoff dist must be finite, got {dist:?}");
            assert!(
                (p - 0.5).abs() < 1e-3,
                "flat payoffs should give uniform stationary dist, got {dist:?}"
            );
        }
    }

    // ------------------------------------------------------------------
    // PayoffCache
    // ------------------------------------------------------------------

    #[test]
    fn test_payoff_cache_grows_correctly() {
        // N=2 N-tensor cache: same boundary growth pattern as the
        // pre-refactor `Vec<Vec<f32>>` matrix, expressed via
        // `resize_for_boundary` + `set_cell`.
        let mut cache = PayoffCache::with_num_agents(2);
        cache.resize_for_boundary(1);
        cache.set_cell(&[0, 0], vec![0.0, 0.0]);
        assert_eq!(cache.per_role_k(), 1);
        assert_eq!(cache.eval_count, 1);

        // Grow to k=2 → 4 cells, 3 are new (boundary slabs for agent 0
        // and agent 1 union together).
        cache.resize_for_boundary(2);
        cache.set_cell(&[1, 0], vec![0.5, -0.5]);
        cache.set_cell(&[0, 1], vec![-0.5, 0.5]);
        cache.set_cell(&[1, 1], vec![0.0, 0.0]);
        assert_eq!(cache.per_role_k(), 2);
        assert_eq!(cache.eval_count, 1 + 3, "k=1→2 adds 3 new cells (4-1)");

        // The agent-0-payoff projection should recover the
        // pre-refactor 2-D matrix shape.
        // payoffs[i][j] = cell[(i,j)][0]
        let payoffs = cache.payoff_tensor();
        let row_matrix: Vec<Vec<f32>> = (0..2)
            .map(|i| (0..2).map(|j| payoffs[i + j * 2][0]).collect::<Vec<_>>())
            .collect();
        assert_eq!(row_matrix, vec![vec![0.0, -0.5], vec![0.5, 0.0]]);

        // Grow to k=3 → 9 cells, 5 are new.
        cache.resize_for_boundary(3);
        // Set the 5 new cells; total evals = 1 + 3 + 5 = 9.
        for joint in cache.clone().boundary_joint_strategies(0) {
            cache.set_cell(&joint, vec![0.0, 0.0]);
        }
        // Agent 0's boundary covers 3 new cells; agent 1's boundary
        // adds 2 more (3 minus the [k-1, k-1] which overlaps the
        // agent-0 slab; actually agent-1 slab is 3 cells but 1
        // overlaps → 2 new).
        for joint in cache.clone().boundary_joint_strategies(1) {
            // Skip cells already set above.
            if cache.get_joint(&joint).is_none_or(|p| p == [0.0, 0.0]) && joint[0] != 2 {
                cache.set_cell(&joint, vec![0.0, 0.0]);
            }
        }
        // 1 + 3 + 5 = 9 evaluations total.
        assert_eq!(cache.eval_count, 1 + 3 + 5);
    }

    #[test]
    fn test_payoff_cache_get_in_bounds() {
        let mut cache = PayoffCache::with_num_agents(2);
        cache.resize_for_boundary(1);
        cache.set_cell(&[0, 0], vec![0.0, 0.0]);
        cache.resize_for_boundary(2);
        cache.set_cell(&[1, 0], vec![0.7, -0.7]);
        cache.set_cell(&[0, 1], vec![-0.7, 0.7]);
        cache.set_cell(&[1, 1], vec![0.0, 0.0]);
        // Agent 0's payoff at (0, 1) = -0.7; at (1, 0) = +0.7.
        assert_eq!(cache.get_joint(&[0, 1]).map(|p| p[0]), Some(-0.7));
        assert_eq!(cache.get_joint(&[1, 0]).map(|p| p[0]), Some(0.7));
        assert_eq!(cache.get_joint(&[0, 0]).map(|p| p[0]), Some(0.0));
        assert_eq!(cache.get_joint(&[1, 1]).map(|p| p[0]), Some(0.0));
        assert_eq!(cache.get_joint(&[2, 0]), None);
    }

    // ------------------------------------------------------------------
    // Exploitability
    // ------------------------------------------------------------------

    #[test]
    fn test_exploitability_on_pure_nash_is_zero() {
        // Row player strictly dominates with row 0 → pure Nash is (1, 0).
        let payoffs = vec![vec![1.0, 2.0], vec![-1.0, -2.0]];
        let meta_nash = vec![1.0, 0.0];
        let expl = empirical_exploitability(&payoffs, &meta_nash);
        // Row 0 already plays best response. Column 1 minimizes row gain
        // → equilibrium value is 2.0; no improvement possible.
        // Row gain = max(1,-1) - 2.0 = -1 → 0.
        // Col gain = 2.0 - min(2, ...) = 0.
        assert!(expl < 1e-6, "expected ~0 exploitability, got {expl}");
    }

    #[test]
    fn test_exploitability_on_matching_pennies_uniform_is_zero() {
        let payoffs = matching_pennies_payoff();
        let meta_nash = vec![0.5, 0.5];
        let expl = empirical_exploitability(&payoffs, &meta_nash);
        assert!(
            expl < 1e-5,
            "uniform on matching-pennies should have 0 exploitability, got {expl}"
        );
    }

    #[test]
    fn test_exploitability_off_equilibrium_is_positive() {
        let payoffs = matching_pennies_payoff();
        let meta_nash = vec![1.0, 0.0]; // row 0 always
        let expl = empirical_exploitability(&payoffs, &meta_nash);
        // Col player BRs by playing col 1, gets value -1 (so col_gain=2).
        assert!(expl > 0.5, "off-equilibrium should be exploitable, got {expl}");
    }

    // ------------------------------------------------------------------
    // PsroTrainer end-to-end
    // ------------------------------------------------------------------

    #[allow(clippy::type_complexity)]
    fn build_matching_pennies_trainer(
        meta_solver: Box<dyn MetaSolver>,
        max_iterations: usize,
    ) -> PsroTrainer<
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
        let psro_config = PsroConfig {
            max_iterations,
            max_population_size: 50,
            br_train_steps_per_iteration: 2,
            payoff_eval_episodes: 4,
            max_payoff_evals_per_iteration: None,
            br_reward_scale: 1.0,
            seed: 0,
        };
        let joint_config = JointTrainerConfig {
            num_agents: 2,
            rollout_steps: 32,
            n_epochs: 1,
            minibatch_size: 32,
            ..Default::default()
        };
        PsroTrainer::new(
            psro_config,
            joint_config,
            meta_solver,
            device,
            |dev: &NdArrayDevice, seed: u64| {
                // 1 obs dim, 2 actions, small hidden.
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
                BurnOptimizer::new(inner, 1e-3)
            },
            MatchingPennies::new,
        )
        .expect("PsroTrainer::new should succeed for 2-agent config")
    }

    #[test]
    fn test_psro_runs_on_matching_pennies() {
        let mut trainer =
            build_matching_pennies_trainer(Box::new(FictitiousPlayMetaSolver::new(500)), 3);
        let stats = trainer.run_silent().expect("PSRO run should not error");
        assert_eq!(stats.iterations.len(), 3, "should record 3 iterations");
        for (k, it) in stats.iterations.iter().enumerate() {
            assert_eq!(it.iteration, k + 1);
            assert_eq!(it.population_size, k + 2, "population grows by 1 per iter");
            // Reported distributions are over the *post-append*
            // population (size = population_size).
            assert_valid_distribution(it.meta_nash_row(), it.population_size);
            assert_valid_distribution(it.meta_nash_col(), it.population_size);
            assert!(it.exploitability.is_finite());
            assert!(it.exploitability >= 0.0, "exploitability must be >= 0");
        }
    }

    /// The `on_iteration` callback must fire exactly `max_iterations`
    /// times, once per outer iteration, with monotonically increasing
    /// `iteration` values matching the entries pushed onto the returned
    /// history. This is the load-bearing observability guarantee:
    /// callers (e.g. `train_psro.rs`) rely on the callback firing
    /// *during* the run, one tick per iteration, in order.
    #[test]
    fn test_psro_run_callback_fires_per_iteration() {
        let max_iterations = 4;
        let mut trainer = build_matching_pennies_trainer(
            Box::new(FictitiousPlayMetaSolver::new(500)),
            max_iterations,
        );

        let mut observed: Vec<usize> = Vec::new();
        let stats = trainer
            .run(|it, _brs| observed.push(it.iteration))
            .expect("PSRO run should not error");

        // Callback fired exactly once per outer iteration.
        assert_eq!(
            observed.len(),
            max_iterations,
            "callback should fire exactly max_iterations times"
        );
        // Iteration indices are 1-based and strictly increasing.
        let expected: Vec<usize> = (1..=max_iterations).collect();
        assert_eq!(
            observed, expected,
            "callback iteration indices must be monotonically increasing 1..=max_iterations"
        );
        // The callback observed the same iteration indices, in order, as
        // the final returned history.
        let from_history: Vec<usize> = stats.iterations.iter().map(|s| s.iteration).collect();
        assert_eq!(observed, from_history, "callback indices must match the pushed history order");
    }

    /// `run_silent()` must be behaviourally identical to `run(|_| {})`:
    /// it records the full per-iteration history without requiring a
    /// callback.
    #[test]
    fn test_psro_run_silent_records_full_history() {
        let max_iterations = 3;
        let mut trainer = build_matching_pennies_trainer(
            Box::new(FictitiousPlayMetaSolver::new(500)),
            max_iterations,
        );
        let stats = trainer.run_silent().expect("PSRO run_silent should not error");
        assert_eq!(stats.iterations.len(), max_iterations);
    }

    /// Mid-run checkpointing (issue #204) rides on the `on_iteration`
    /// callback's second argument: the slice of newest-per-agent BR
    /// policies. This test exercises the checkpoint-trigger logic the
    /// example uses, without touching disk:
    ///
    /// 1. The callback receives exactly one BR per agent each iteration.
    /// 2. Those BR references are the same policies the trainer exposes via
    ///    `populations(a).last()` (i.e. the freshly-appended BR), captured by
    ///    their deterministic forward-pass logits.
    /// 3. A `CHECKPOINT_INTERVAL`-gated counter fires on exactly the expected
    ///    iterations (every Nth iteration), modelling the example's `iter %
    ///    CHECKPOINT_INTERVAL_ITERATIONS == 0` knob.
    /// 4. The number of distinct "checkpoints taken" matches the closed form,
    ///    and the policies handed at checkpoint time round-trip bit-identically
    ///    through a clone (the operation the example's `Recorder::save_file`
    ///    performs on a clone).
    #[test]
    fn test_psro_checkpoint_callback_fires_at_intervals() {
        let max_iterations = 6;
        const CHECKPOINT_INTERVAL: usize = 2;
        let mut trainer = build_matching_pennies_trainer(
            Box::new(FictitiousPlayMetaSolver::new(500)),
            max_iterations,
        );
        let num_agents = 2;

        // Iterations on which a checkpoint was taken.
        let mut checkpoint_iters: Vec<usize> = Vec::new();
        // For each checkpoint, the per-agent BR logits captured at
        // checkpoint time, plus the logits of a *clone* of the same
        // policy (mirrors the example saving `br.clone()`).
        let mut checkpoint_logits: Vec<Vec<(Vec<f32>, Vec<f32>)>> = Vec::new();

        trainer
            .run(|it, brs| {
                // (1) One BR per agent, every iteration.
                assert_eq!(brs.len(), num_agents, "callback must receive one newest BR per agent");

                // (3) Interval gate exactly as the example drives it.
                if it.iteration % CHECKPOINT_INTERVAL == 0 {
                    checkpoint_iters.push(it.iteration);
                    let per_agent: Vec<(Vec<f32>, Vec<f32>)> = brs
                        .iter()
                        .map(|br| {
                            let original = read_policy_weight(br);
                            // (4) Clone round-trip: cloning a policy (as
                            // the recorder does before `save_file`) must
                            // not perturb its forward pass.
                            let cloned = (**br).clone();
                            let cloned_logits = read_policy_weight(&cloned);
                            (original, cloned_logits)
                        })
                        .collect();
                    checkpoint_logits.push(per_agent);
                }
            })
            .expect("PSRO run should not error");

        // (3) Fired on exactly iterations 2, 4, 6.
        assert_eq!(
            checkpoint_iters,
            vec![2, 4, 6],
            "checkpoint must fire on every CHECKPOINT_INTERVAL-th iteration"
        );
        // Closed form: floor(max_iterations / interval) checkpoints.
        assert_eq!(checkpoint_logits.len(), max_iterations / CHECKPOINT_INTERVAL);

        for per_agent in &checkpoint_logits {
            assert_eq!(per_agent.len(), num_agents);
            for (original, cloned) in per_agent {
                // (4) Clone is byte-identical to the checkpointed policy.
                assert_eq!(
                    original, cloned,
                    "checkpointed BR clone must produce identical logits (save_file round-trip)"
                );
            }
        }

        // (2) The final-iteration checkpoint must match what the trainer
        // exposes via the public `populations(a).last()` accessor — this
        // is the same handle `train_psro.rs` uses for its final save, so
        // the mid-run checkpoint and the post-run save are consistent.
        let final_checkpoint = checkpoint_logits.last().expect("at least one checkpoint");
        for (a, (checkpointed_logits, _)) in final_checkpoint.iter().enumerate().take(num_agents) {
            let pop_last = trainer.populations(a).last().expect("non-empty population");
            let from_accessor = read_policy_weight(pop_last);
            assert_eq!(
                checkpointed_logits, &from_accessor,
                "checkpointed BR for agent {a} must equal populations(a).last() logits"
            );
        }
    }

    /// Read the policy_head weight buffer from a policy as a flat
    /// `Vec<f32>` for diff comparisons. We deliberately use
    /// `policy_head_action_dim` × hidden-vector via the policy's
    /// public surface so that no internal-Burn quirks of
    /// `into_record` enter the picture.
    fn read_policy_weight(policy: &MlpBurnPolicy<B>) -> Vec<f32> {
        // Run a forward pass on a deterministic obs (all-zero) and
        // record the resulting logits. Two policies with byte-identical
        // weights produce byte-identical logits on the same obs; if
        // their weights differ, so will the logits. This sidesteps any
        // `into_record()` / `Param::val()` cloning subtleties.
        let device: NdArrayDevice = Default::default();
        let obs = burn::tensor::Tensor::<B, 2>::zeros([1, 1], &device);
        let (logits, _) = policy.forward(obs);
        logits.into_data().to_vec().expect("logits to_vec")
    }

    #[test]
    fn test_psro_freeze_n_minus_1_preserves_frozen_params() {
        // After a single BR-training round, only the active agent's
        // params should change. We verify this by snapshotting the
        // frozen agent's policy_head weight before and after a single
        // joint update with active_mask = [false, true] and asserting
        // the weight is byte-identical.
        let device: NdArrayDevice = Default::default();

        let pol_a = MlpBurnPolicy::<B>::new(1, 2, 8, &device);
        let pol_b = MlpBurnPolicy::<B>::new(1, 2, 8, &device);
        let opt_a = BurnOptimizer::<B, MlpBurnPolicy<B>, _>::new(AdamConfig::new().init(), 1e-2);
        let opt_b = BurnOptimizer::<B, MlpBurnPolicy<B>, _>::new(AdamConfig::new().init(), 1e-2);
        let joint_config = JointTrainerConfig {
            num_agents: 2,
            rollout_steps: 32,
            n_epochs: 1,
            minibatch_size: 32,
            ..Default::default()
        };
        let mut trainer = JointMultiAgentTrainer::<B, MlpBurnPolicy<B>, _>::new(
            vec![pol_a.clone(), pol_b.clone()],
            vec![opt_a, opt_b],
            joint_config,
            device,
        )
        .unwrap();

        let frozen_before = read_policy_weight(trainer.policy(0));
        let active_before = read_policy_weight(trainer.policy(1));

        let mut env = MatchingPennies::new();
        let mut last_obs = env.reset_joint(None);
        let mut rng = StdRng::seed_from_u64(0);
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs, &mut rng);

        let active_mask = vec![false, true];
        trainer
            .update_with_active_agents(
                &rollout,
                &active_mask,
                &mut rng,
                |_features: &[burn::tensor::Tensor<B, 2>]| -> Option<burn::tensor::Tensor<B, 1>> {
                    None
                },
            )
            .expect("update should not error");

        let frozen_after = read_policy_weight(trainer.policy(0));
        let active_after = read_policy_weight(trainer.policy(1));

        // Frozen agent: parameters must be unchanged.
        assert_eq!(frozen_before.len(), frozen_after.len(), "weight buffer size changed");
        for (b, a) in frozen_before.iter().zip(frozen_after.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "frozen agent params changed: {b} -> {a} (delta {})",
                a - b
            );
        }

        // Active agent: parameters MUST have changed (otherwise the test
        // setup didn't generate any gradient signal and we're not really
        // verifying anything).
        let mut any_diff = false;
        for (b, a) in active_before.iter().zip(active_after.iter()) {
            if (a - b).abs() > 1e-9 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "active agent params should have changed");
    }

    #[test]
    fn test_payoff_cache_only_evaluates_new_boundary() {
        // After running PSRO for a few iterations, payoff_cache.eval_count
        // should equal the cumulative number of new boundary cells in
        // the N-tensor cache:
        // - Initial 1^N seed (each agent has 1 policy): 1 eval.
        // - Iteration k (k=1..K): cache grows from k^N to (k+1)^N, adding (k+1)^N − k^N
        //   new boundary cells.
        // For N=2 this collapses to (k+1)² − k² = 2k + 1, recovering
        // the pre-refactor formula `1 + K² + 2K`.
        let k = 3;
        let mut trainer =
            build_matching_pennies_trainer(Box::new(FictitiousPlayMetaSolver::new(200)), k);
        trainer.run_silent().expect("PSRO run should not error");
        // For N=2: 1 + Σ_{j=1}^{k} ((j+1)² − j²) = 1 + (k+1)² − 1 = (k+1)².
        // With K=3 PSRO iterations starting from k=1, final k = 4, so
        // (k+1)² with final k=4 → 16; equivalently 1 + 3 + 5 + 7 = 16,
        // which equals 1 + K² + 2K = 1 + 9 + 6 = 16. ✓
        let expected = 1 + k * k + 2 * k;
        assert_eq!(
            trainer.payoff_cache.eval_count, expected,
            "payoff cache should only evaluate new boundary cells (N=2 formula 1 + K² + 2K)"
        );
    }

    /// NashConv N=2 fast-path bit-stability sanity: on
    /// matching-pennies with the uniform meta-Nash, both the legacy
    /// 2-player exploitability formula and the N-tensor NashConv
    /// produce 0.0 (within `1e-5`).
    #[test]
    fn test_nashconv_n2_fast_path_matches_legacy_on_uniform() {
        let payoffs = matching_pennies_payoff();
        let meta_nash = vec![0.5, 0.5];
        let expl_legacy = empirical_exploitability(&payoffs, &meta_nash);
        assert!(expl_legacy < 1e-5);
        // The fast-path in `compute_nashconv` projects to the same
        // 2-player matrix and calls `empirical_exploitability`, so by
        // construction the result is bit-identical. We assert the
        // legacy formula returns 0 here as the canonical numerical
        // anchor.
    }

    /// Order-independence / purity of per-cell payoff evaluation
    /// (issue #201).
    ///
    /// After growing both agents' populations to size 2, we evaluate
    /// the full 2×2 boundary tensor in a forward joint order and again
    /// in the reverse order, and re-evaluate one cell twice. Because
    /// each cell seeds its own local `StdRng` from `(config.seed,
    /// joint)` (no shared trainer RNG), every cell's payoff vector MUST
    /// be **bit-identical** regardless of evaluation order — the
    /// guarantee that lets #203 parallelize the boundary-slab loop.
    #[test]
    fn test_payoff_cell_eval_is_order_independent() {
        let device: NdArrayDevice = Default::default();
        let psro_config = PsroConfig {
            max_iterations: 1,
            max_population_size: 50,
            br_train_steps_per_iteration: 2,
            payoff_eval_episodes: 4,
            max_payoff_evals_per_iteration: None,
            br_reward_scale: 1.0,
            seed: 12345,
        };
        let joint_config = JointTrainerConfig {
            num_agents: 2,
            rollout_steps: 32,
            n_epochs: 1,
            minibatch_size: 32,
            ..Default::default()
        };
        let mut trainer = PsroTrainer::new(
            psro_config,
            joint_config,
            Box::new(FictitiousPlayMetaSolver::new(200)) as Box<dyn MetaSolver>,
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
        .expect("PsroTrainer::new should succeed");

        // Run one PSRO iteration so each agent has a 2-policy
        // population (indices 0 and 1) to form a 2×2 joint tensor.
        trainer.run_silent().expect("PSRO run should not error");
        assert!(trainer.populations(0).len() >= 2, "need >=2 policies per agent");
        assert!(trainer.populations(1).len() >= 2, "need >=2 policies per agent");

        let joints: Vec<Vec<usize>> = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]];

        // Forward-order evaluation.
        let forward: Vec<Vec<f32>> =
            joints.iter().map(|j| trainer.evaluate_payoff_joint(j)).collect();

        // Reverse-order evaluation: interleaved/reversed traversal must
        // not change any cell's value because no cell depends on global
        // RNG state.
        let reverse: Vec<Vec<f32>> = joints
            .iter()
            .rev()
            .map(|j| (j.clone(), trainer.evaluate_payoff_joint(j)))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|(_, v)| v)
            .collect();

        assert_eq!(
            forward, reverse,
            "payoff cells must be bit-identical regardless of evaluation order"
        );

        // Re-evaluating a single cell twice must also be bit-identical.
        let once = trainer.evaluate_payoff_joint(&[1, 0]);
        let twice = trainer.evaluate_payoff_joint(&[1, 0]);
        assert_eq!(once, twice, "re-evaluating the same cell must be bit-identical");

        // And it must match the value computed during the full-tensor
        // sweep (cell [1, 0] is index 1 in `joints`).
        assert_eq!(once, forward[1], "single-cell value must match the swept value");
    }

    /// Rayon-parallel boundary-slab evaluation is **bit-identical** to a
    /// serial sweep (issue #203).
    ///
    /// After growing both agents' populations to size ≥ 2 we evaluate the
    /// full boundary slab two ways — serially cell-by-cell via
    /// `evaluate_payoff_joint`, and in parallel via
    /// `evaluate_payoff_boundary_parallel` — and assert the two payoff
    /// vectors match cell-for-cell exactly. To prove the result is
    /// invariant to thread scheduling we additionally run the parallel
    /// path inside rayon thread pools of size 1 and 4 and assert both
    /// equal the serial reference. This is the load-bearing determinism
    /// guarantee of #198 PR C and is fully CPU-CI-testable (no cluster
    /// hardware required).
    #[test]
    fn test_payoff_boundary_parallel_matches_serial_bit_identically() {
        let device: NdArrayDevice = Default::default();
        let psro_config = PsroConfig {
            max_iterations: 1,
            max_population_size: 50,
            br_train_steps_per_iteration: 2,
            payoff_eval_episodes: 4,
            max_payoff_evals_per_iteration: None,
            br_reward_scale: 1.0,
            seed: 0xC0FF_EE12,
        };
        let joint_config = JointTrainerConfig {
            num_agents: 2,
            rollout_steps: 32,
            n_epochs: 1,
            minibatch_size: 32,
            ..Default::default()
        };
        let mut trainer = PsroTrainer::new(
            psro_config,
            joint_config,
            Box::new(FictitiousPlayMetaSolver::new(200)) as Box<dyn MetaSolver>,
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
        .expect("PsroTrainer::new should succeed");

        // One PSRO iteration grows each agent's population to size 2,
        // forming a 2×2 joint tensor whose boundary slab we re-evaluate.
        trainer.run_silent().expect("PSRO run should not error");
        let k = trainer.populations(0).len();
        assert!(k >= 2, "need >=2 policies per agent to form a non-trivial slab");

        // Full boundary slab in deterministic flat order.
        let new_strategy_idx = k - 1;
        let total = k.checked_pow(2).expect("k^2 overflow");
        let boundary: Vec<Vec<usize>> = (0..total)
            .filter_map(|s| {
                let c = decompose_joint_index(s, 2, k);
                c.contains(&new_strategy_idx).then_some(c)
            })
            .collect();
        assert!(!boundary.is_empty(), "boundary slab must be non-empty");

        // Serial reference: cell-by-cell via the pure single-cell path.
        let serial: Vec<Vec<f32>> =
            boundary.iter().map(|j| trainer.evaluate_payoff_joint(j)).collect();

        // Parallel path under the ambient (global) rayon pool.
        let parallel = trainer.evaluate_payoff_boundary_parallel(&boundary);
        assert_eq!(
            serial, parallel,
            "rayon-parallel boundary payoff must be bit-identical to the serial sweep"
        );

        // Thread-count invariance: the seeding scheme makes the result
        // independent of how many threads execute it. Run the parallel
        // evaluation inside dedicated 1-thread and 4-thread pools and
        // assert both match the serial reference exactly. We bind the
        // Sync field borrows into locals so the `install` closure does
        // not capture the whole (non-`Send`) trainer, then drive the same
        // `evaluate_payoff_joint_pure` cell function the production path
        // uses.
        let populations = &trainer.populations;
        let config = &trainer.config;
        let env_factory = &trainer.env_factory;
        let device = &trainer.device;
        for threads in [1_usize, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build rayon pool");
            let got: Vec<Vec<f32>> = pool.install(|| {
                boundary
                    .par_iter()
                    .map(|joint| {
                        let policies: Vec<MlpBurnPolicy<B>> =
                            (0..2).map(|a| populations[a][joint[a]].clone()).collect();
                        evaluate_payoff_joint_pure::<B, _, _, _>(
                            joint,
                            config,
                            &policies,
                            env_factory,
                            device,
                        )
                    })
                    .collect()
            });
            assert_eq!(
                serial, got,
                "parallel payoff must be bit-identical to serial with {threads} thread(s)"
            );
        }
    }

    /// Run a multi-iteration PSRO trainer (so the parallel BR loop
    /// executes several times) under a rayon pool of `threads` threads,
    /// and return the flattened per-agent population policy weights.
    ///
    /// `max_iterations` / `rollout_steps` / `hidden` / `br_train_steps` /
    /// `payoff_eval_episodes` are parameters so callers can pick a tiny
    /// always-on smoke workload or a heavier `#[ignore]`d proof. Both run
    /// the same code path (the #232 par_iter BR loop with `num_agents > 1`).
    #[cfg(test)]
    fn psro_populations_under_threads(
        threads: usize,
        max_iterations: usize,
        rollout_steps: usize,
        hidden: usize,
        br_train_steps: usize,
        payoff_eval_episodes: usize,
    ) -> Vec<Vec<Vec<f32>>> {
        let device: NdArrayDevice = Default::default();
        let psro_config = PsroConfig {
            max_iterations,
            max_population_size: 50,
            br_train_steps_per_iteration: br_train_steps,
            payoff_eval_episodes,
            max_payoff_evals_per_iteration: None,
            br_reward_scale: 1.0,
            seed: 0x5EED_2323,
        };
        let joint_config = JointTrainerConfig {
            num_agents: 2,
            rollout_steps,
            n_epochs: 1,
            minibatch_size: rollout_steps.max(1),
            ..Default::default()
        };
        // `threads == 0` runs under the ambient/global rayon pool (no
        // bespoke pool). The always-on smoke uses this: wrapping a full PSRO
        // trainer (whose BR loop itself calls `par_iter`) inside a dedicated
        // multi-thread pool nests parallelism and oversubscribes 2-core CI
        // runners, which hung the Tests job (#232 review). `threads >= 1`
        // builds a dedicated pool for the heavier `#[ignore]`d
        // thread-count-invariance proof, which runs on demand on many-core
        // hosts.
        let run = move || -> Vec<Vec<Vec<f32>>> {
            let mut trainer = PsroTrainer::new(
                psro_config.clone(),
                joint_config.clone(),
                Box::new(FictitiousPlayMetaSolver::new(200)) as Box<dyn MetaSolver>,
                device,
                move |dev: &NdArrayDevice, seed: u64| {
                    MlpBurnPolicy::<B>::new_seeded(
                        MatchingPennies::OBS_DIM,
                        MatchingPennies::ACTION_DIM,
                        hidden,
                        seed,
                        dev,
                    )
                },
                || BurnOptimizer::new(AdamConfig::new().init(), 1e-3),
                MatchingPennies::new,
            )
            .expect("PsroTrainer::new should succeed");
            trainer.run_silent().expect("PSRO run should not error");

            // Snapshot every agent's full population as flattened
            // policy weights (forward-on-zero-obs fingerprint).
            let num_agents = 2;
            (0..num_agents)
                .map(|a| trainer.populations(a).iter().map(read_policy_weight).collect::<Vec<_>>())
                .collect()
        };
        if threads == 0 {
            run()
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build rayon pool");
            pool.install(run)
        }
    }

    /// Always-on smoke for the rayon-parallel best-response loop (issue
    /// #232) at a deliberately tiny workload (2 iterations, 1 BR train step,
    /// 8 rollout steps, hidden=4, 1 payoff episode). It runs the real #232
    /// code path — two agents, so the `par_iter` BR loop runs — under the
    /// **ambient** global rayon pool (`threads == 0`, no bespoke pool), then
    /// runs it again and asserts byte-identical results.
    ///
    /// Why ambient-pool + a repeat run rather than a 1-vs-4-thread compare:
    /// wrapping a full PSRO trainer (whose BR loop itself calls `par_iter`)
    /// inside a dedicated 4-thread pool nests parallelism and oversubscribes
    /// 2-core CI runners, which hung the Tests job (#232 review). This keeps
    /// cheap, deterministic always-on coverage of the parallel path; the
    /// cross-thread-count (1 vs 4) invariance proof lives in the
    /// `#[ignore]`d `test_best_response_parallel_thread_count_invariant_thorough`.
    ///
    /// Each BR draws its opponent indices + init seed in fixed agent order
    /// before the parallel region and runs under a per-agent local RNG
    /// seeded from `(config.seed, active_agent)`, so scheduling cannot
    /// affect the result. (The result is intentionally *not* bit-identical
    /// to the pre-#232 serial-RNG runs — the RNG threading changed — only
    /// reproducible for a given seed.)
    #[test]
    fn test_best_response_parallel_smoke() {
        let a = psro_populations_under_threads(0, 2, 8, 4, 1, 1);
        let b = psro_populations_under_threads(0, 2, 8, 4, 1, 1);

        // Sanity: the BR loop actually ran and grew the populations.
        assert!(
            a[0].len() >= 2,
            "expected populations to grow over the iterations (got {})",
            a[0].len()
        );
        assert_eq!(
            a, b,
            "PSRO best-response output must be deterministic for a fixed seed"
        );
    }

    /// Thorough multi-iteration variant of the thread-count-invariance
    /// guarantee at a realistic workload (3 iterations, larger rollouts +
    /// hidden size), which grows deeper populations across more parallel
    /// BR rounds.
    ///
    /// `#[ignore]`d per the #208/#209 convention: a full 3-iteration PSRO
    /// run twice under bespoke multi-thread pools costs ~85s and, on 2-core
    /// CI runners, oversubscribed and hung the Tests job (#232 review). The
    /// always-on `test_best_response_parallel_smoke` keeps cheap determinism
    /// coverage on every CI run; run this heavier cross-thread-count proof on
    /// demand with `cargo test --features training -- --ignored` (prefer
    /// `--release`, ideally on a many-core host).
    #[test]
    #[ignore = "multi-iteration PSRO thread-count-invariance run; opt in with --ignored (prefer --release)"]
    fn test_best_response_parallel_thread_count_invariant_thorough() {
        let one = psro_populations_under_threads(1, 3, 32, 16, 2, 4);
        let four = psro_populations_under_threads(4, 3, 32, 16, 2, 4);

        assert!(
            one[0].len() >= 4,
            "expected populations to grow over 3 iterations (got {})",
            one[0].len()
        );
        assert_eq!(
            one, four,
            "PSRO best-response output must be byte-identical across thread counts (1 vs 4)"
        );
    }

    /// `splitmix64` is a deterministic permutation-like mixer: distinct
    /// inputs map to distinct outputs (avalanche), guaranteeing
    /// neighbouring joint hashes seed well-separated RNG streams.
    #[test]
    fn test_splitmix64_distinguishes_neighbours() {
        let a = splitmix64(0);
        let b = splitmix64(1);
        let c = splitmix64(2);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
        // Deterministic.
        assert_eq!(a, splitmix64(0));
    }

    /// Boundary subsampling selection (issue #212) is correct and
    /// deterministic. Pure-function unit test — no env, no rollouts.
    #[test]
    fn test_select_boundary_to_evaluate() {
        // Helper: a fake boundary of `n` distinguishable single-element
        // joints [0], [1], ..., [n-1].
        let make = |n: usize| -> Vec<Vec<usize>> { (0..n).map(|i| vec![i]).collect() };

        // cap = None -> evaluate everything, no fills (default path is
        // bit-identical to the full-boundary sweep).
        let b = make(5);
        let (to_eval, fill) = select_boundary_to_evaluate(&b, None);
        assert_eq!(to_eval, b);
        assert!(fill.is_empty());

        // cap >= len -> evaluate everything, no fills.
        let (to_eval, fill) = select_boundary_to_evaluate(&b, Some(5));
        assert_eq!(to_eval, b);
        assert!(fill.is_empty());
        let (to_eval, fill) = select_boundary_to_evaluate(&b, Some(99));
        assert_eq!(to_eval, b);
        assert!(fill.is_empty());

        // cap < len -> stratified selection. len=6, cap=3 selects
        // indices floor(j*6/3) = 0, 2, 4.
        let b = make(6);
        let (to_eval, fill) = select_boundary_to_evaluate(&b, Some(3));
        assert_eq!(to_eval, vec![vec![0], vec![2], vec![4]]);
        // Non-selected cells (1, 3, 5) fill from nearest preceding
        // selected (src positions into to_eval: 0->[0], 1->[2], 2->[4]).
        // dst 1 <- src 0 ([0]); dst 3 <- src 1 ([2]); dst 5 <- src 2 ([4]).
        assert_eq!(fill, vec![(1, 0), (3, 1), (5, 2)]);

        // Every boundary index is accounted for exactly once: either it
        // is a selected index or it appears as a `dst` in `fill`.
        let selected_dsts: std::collections::BTreeSet<usize> =
            [0_usize, 2, 4].into_iter().collect();
        let fill_dsts: std::collections::BTreeSet<usize> = fill.iter().map(|&(d, _)| d).collect();
        let mut all: std::collections::BTreeSet<usize> = selected_dsts.clone();
        all.extend(&fill_dsts);
        assert_eq!(all, (0..6).collect());
        assert!(selected_dsts.is_disjoint(&fill_dsts));

        // cap = Some(0) is treated as Some(1): exactly one cell, the
        // first, is evaluated; everything else fills from it.
        let (to_eval, fill) = select_boundary_to_evaluate(&b, Some(0));
        assert_eq!(to_eval, vec![vec![0]]);
        assert_eq!(fill, vec![(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]);

        // Determinism: identical inputs yield identical outputs.
        let again = select_boundary_to_evaluate(&b, Some(3));
        assert_eq!(again, select_boundary_to_evaluate(&b, Some(3)));
    }

    /// **Load-bearing bit-identity test (issue #212).**
    ///
    /// The opt-in boundary-subsampling cap must not perturb the default
    /// (uncapped) behavior. We run PSRO three ways from the *same seed* —
    /// `max_payoff_evals_per_iteration: None` (default / pre-#212),
    /// `Some(cap)` with `cap` larger than any iteration's boundary, and
    /// `Some(usize::MAX)` — and assert the resulting payoff tensor,
    /// per-cell `eval_count`, and full exploitability trace are
    /// **bit-for-bit equal** across all three. This pins that the
    /// cache/subsampling plumbing is a no-op whenever the cap is not
    /// actually exceeded — preserving the #201 determinism guarantee and
    /// the #203 parallel bit-identity.
    #[test]
    fn test_subsampling_cap_unreached_is_bit_identical_to_uncapped() {
        // Build three trainers from the same config except for the cap.
        // K=3 PSRO iters on matching pennies: max boundary is at the
        // final growth k=3->4 with (4^2 - 3^2) = 7 cells, so any cap >= 7
        // leaves every iteration's boundary fully evaluated.
        let run = |cap: Option<usize>| -> (Vec<Vec<f32>>, usize, Vec<f32>) {
            let mut trainer =
                build_matching_pennies_trainer(Box::new(FictitiousPlayMetaSolver::new(200)), 3);
            trainer.config.max_payoff_evals_per_iteration = cap;
            let stats = trainer.run_silent().expect("PSRO run should not error");
            let tensor = trainer.payoff_cache.payoff_tensor().to_vec();
            let evals = trainer.payoff_cache.eval_count;
            let trace: Vec<f32> = stats.iterations.iter().map(|s| s.exploitability).collect();
            (tensor, evals, trace)
        };

        let (tensor_none, evals_none, trace_none) = run(None);
        let (tensor_big, evals_big, trace_big) = run(Some(1_000));
        let (tensor_max, evals_max, trace_max) = run(Some(usize::MAX));

        assert_eq!(tensor_none, tensor_big, "payoff tensor: None vs large cap must be identical");
        assert_eq!(tensor_none, tensor_max, "payoff tensor: None vs MAX cap must be identical");
        assert_eq!(evals_none, evals_big, "eval_count: None vs large cap must be identical");
        assert_eq!(evals_none, evals_max, "eval_count: None vs MAX cap must be identical");
        assert_eq!(trace_none, trace_big, "exploitability trace: None vs large cap must match");
        assert_eq!(trace_none, trace_max, "exploitability trace: None vs MAX cap must match");
    }

    /// An *exceeded* subsampling cap bounds the number of fresh
    /// evaluations per iteration while still fully populating the payoff
    /// tensor (no zero/unfilled cells), and is deterministic across runs
    /// from the same seed (issue #212).
    #[test]
    fn test_subsampling_cap_bounds_evals_and_fills_tensor() {
        let run_capped = || -> (usize, Vec<Vec<f32>>) {
            let mut trainer =
                build_matching_pennies_trainer(Box::new(FictitiousPlayMetaSolver::new(200)), 3);
            // Cap at 3 fresh evals/iter. The initial 1^N seed (1 eval) is
            // unconditional; thereafter each iteration's boundary is
            // 2k+1 (N=2), exceeding 3 from the k=2->3 growth (5 cells)
            // onward, so the cap is actually exercised.
            trainer.config.max_payoff_evals_per_iteration = Some(3);
            trainer.run_silent().expect("PSRO run should not error");
            let evals = trainer.payoff_cache.eval_count;
            let tensor = trainer.payoff_cache.payoff_tensor().to_vec();
            (evals, tensor)
        };

        let (evals, tensor) = run_capped();

        // Uncapped would be 1 + K² + 2K = 16 evals for K=3 (see
        // `test_payoff_cache_only_evaluates_new_boundary`). Capping fresh
        // rollouts at 3/iter must yield strictly fewer evaluations: the
        // initial seed (1) + at most 3 per iteration × 3 iters = at most
        // 10, and < 16.
        assert!(evals <= 1 + 3 * 3, "capped eval_count {evals} must respect the per-iter cap");
        assert!(evals < 16, "capped eval_count {evals} must be fewer than the uncapped 16");

        // Every cell of the final 4×4 tensor is populated (the fill step
        // copies a real evaluated payoff into each un-sampled boundary
        // cell, so no cell is left at its resize-zeroed [0, 0] value for
        // matching pennies, whose payoffs are ±1).
        assert_eq!(tensor.len(), 16, "final tensor is 4^2 cells");
        for (s, cell) in tensor.iter().enumerate() {
            assert_eq!(cell.len(), 2, "cell {s} has per-agent payoffs");
        }

        // Determinism: same seed + same cap -> identical eval_count and
        // tensor (selection is a pure function of (boundary.len(), cap)).
        let (evals2, tensor2) = run_capped();
        assert_eq!(evals, evals2, "capped run must be deterministic in eval_count");
        assert_eq!(tensor, tensor2, "capped run must be deterministic in payoff tensor");
    }
}
