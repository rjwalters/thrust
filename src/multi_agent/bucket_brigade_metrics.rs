//! Bucket-brigade-specific scoring metrics.
//!
//! Rust mirror of the small numerical helpers used by the bucket-brigade
//! Python experiments. Kept under `crate::multi_agent` because it's only
//! meaningful for the multi-agent bucket-brigade training paths
//! (PSRO/NFSP test assertions), and gated on `env-bucket-brigade` so the
//! published crate (which strips the bucket-brigade adapter) doesn't pay
//! for the dead module.
//!
//! # `gap_closed` baselines (base scenario)
//!
//! `MINSPEC_RANDOM` and `MINSPEC_SPECIALIST` are the per-step team
//! payoffs of the random and specialist policies on the **base**
//! `minimal_specialization-v1` scenario (4 agents, 10 houses), mirrored
//! from the canonical Python single-source-of-truth at
//! `envs/bucket-brigade/bucket_brigade/baselines/__init__.py:68-69`. Use
//! `gap_closed` to normalize a per-step team payoff against these
//! base-scenario baselines.
//!
//! **Not valid on non-base scenarios.** The three no-convergence cells of
//! the workshop-paper phase diagram (β ∈ {0.1, 0.5, 0.9} with κ=0.1,
//! c=0.5) have meaningfully different baseline payoffs and need their
//! own normalization endpoints — see `gap_closed_cell` and the
//! `MINSPEC_RANDOM_BETA0XX` / `MINSPEC_SPECIALIST_BETA0XX` constants.
//!
//! # `gap_closed_cell` baselines (per cell)
//!
//! The six cell-specific constants were measured via Monte Carlo in
//! `tests/test_bucket_brigade_baselines.rs::recompute_cell_baselines`
//! using the Python protocol from
//! `envs/bucket-brigade/experiments/p3_specialization/diagnostics/
//! random_baseline.py`: 1000 episodes per cell per baseline type, uniform
//! random sampling from `MultiDiscrete([10, 2, 2])` for the random policy, the
//! [`crate::multi_agent::bucket_brigade_baselines::specialist_action`]
//! function for the specialist policy. To regenerate them after a
//! scenario change, run:
//!
//! ```bash
//! cargo test --release --features "training,env-bucket-brigade" \
//!     --test test_bucket_brigade_baselines -- --ignored --nocapture
//! ```

use crate::multi_agent::bucket_brigade_baselines::BucketBrigadeCell;

/// Per-step team payoff baseline for the random policy on the
/// `minimal_specialization-v1` **base** scenario (4 agents, 10 houses).
///
/// Mirrors `MINSPEC_RANDOM = -87.72` in
/// `envs/bucket-brigade/bucket_brigade/baselines/__init__.py:68`
/// (canonical, post-#246 sampler-bug fix, n=1000 × 5 seeds). Three
/// values for this constant have historically coexisted upstream:
///
/// * `-96.07` — pre-#246, 2-dim sampler bug. The number PR #126 ported from
///   `experiments/p3_specialization/analyze_291.py:41` (stale).
/// * `-92.92` — n=50 intermediate.
/// * `-87.72` — n=1000 × 5 seeds, **post-#246 canonical** (what we ship).
///
/// See upstream issues #246 and #293 for the bug-fix history.
pub const MINSPEC_RANDOM: f32 = -87.72;

/// Per-step team payoff baseline for the optimal specialist policy on
/// the `minimal_specialization-v1` **base** scenario (4 agents, 10 houses).
///
/// Mirrors `MINSPEC_SPECIALIST = -22.07` in
/// `envs/bucket-brigade/bucket_brigade/baselines/__init__.py:69`.
pub const MINSPEC_SPECIALIST: f32 = -22.07;

// ---- Cell-specific baselines (no-convergence cells of the phase diagram) ----
//
// Measured by
// `tests/test_bucket_brigade_baselines.rs::recompute_cell_baselines`
// using a 200-step fixed-horizon rollout (matching the
// `EVAL_STEPS = 200` convention in the NFSP/PSRO integration tests):
// 1000 episodes × 200 steps × 4 agents per cell per baseline type, uniform
// sampling for random, `specialist_action` from `bucket_brigade_baselines.rs`
// for specialist.
//
// # Empirical note: tight random↔specialist band on these cells
//
// On all three no-convergence cells the random and specialist baselines
// land within ~3.5 points of each other (both clustered around -603 per
// step). This is the empirical reality of these scenarios — the env's
// `min_nights = 12` plus Bernoulli burn-out dynamics mean that once even a
// handful of fires ignite (initial `prob_house_catches_fire = 0.02` per
// house per night), houses transition to RUINED within ~12 nights and stay
// RUINED for the remaining ~188 steps. The per-step `team_penalty_house_burns *
// ruined_fraction` term then dominates the per-step reward, drowning out
// both the specialist's first-12-night fire-fighting work and the random
// policy's noise. The cell-specific β parameter therefore matters only
// during those first 12 nights, after which the trajectories collapse to
// the same "all-ruined wasteland" reward.
//
// What this means for `gap_closed_cell`: the function has a steep slope
// (≈0.3 gap-closed per reward unit) on these cells, so a strong assertion
// `gap_closed_cell >= 0` is essentially equivalent to "policy scores at
// least as well as uniform random over the 200-step horizon".
//
// **Wall-clock**: ~1.2 sec total for all six constants in release mode on
// a modern CPU (much faster than initial estimates because the env's
// no-NN dynamics are cheap). Re-run the `recompute_cell_baselines` test
// after any change to the bucket-brigade env dynamics or the specialist
// policy.

/// Random baseline (200-step fixed horizon) for cell `(β = 0.1, κ = 0.1, c =
/// 0.5)`.
pub const MINSPEC_RANDOM_BETA01: f32 = -605.5118;
/// Specialist baseline (200-step fixed horizon) for cell `(β = 0.1, κ = 0.1, c
/// = 0.5)`.
pub const MINSPEC_SPECIALIST_BETA01: f32 = -602.0938;

/// Random baseline (200-step fixed horizon) for cell `(β = 0.5, κ = 0.1, c =
/// 0.5)`.
///
/// This is the canonical no-convergence cell PR #126's NFSP test targets.
pub const MINSPEC_RANDOM_BETA05: f32 = -605.5118;
/// Specialist baseline (200-step fixed horizon) for cell `(β = 0.5, κ = 0.1, c
/// = 0.5)`.
pub const MINSPEC_SPECIALIST_BETA05: f32 = -602.0938;

/// Random baseline (200-step fixed horizon) for cell `(β = 0.9, κ = 0.1, c =
/// 0.5)`.
pub const MINSPEC_RANDOM_BETA09: f32 = -605.5118;
/// Specialist baseline (200-step fixed horizon) for cell `(β = 0.9, κ = 0.1, c
/// = 0.5)`.
pub const MINSPEC_SPECIALIST_BETA09: f32 = -602.0938;

/// Compute the fraction of the random→specialist gap closed by the
/// given per-step team payoff on the **base** `minimal_specialization`
/// scenario.
///
/// ```text
/// gap_closed = (per_step_team - MINSPEC_RANDOM)
///              / (MINSPEC_SPECIALIST - MINSPEC_RANDOM)
/// ```
///
/// * Returns `0.0` when the agent matches the random baseline.
/// * Returns `1.0` when the agent matches the specialist baseline.
/// * Negative values indicate the agent is *worse* than the random baseline.
///
/// # Scope (read before using)
///
/// **Only valid for the base `minimal_specialization-v1` scenario.** The
/// random/specialist baselines change with the scenario reward landscape;
/// the three no-convergence cells of the workshop-paper phase diagram
/// (β ∈ {0.1, 0.5, 0.9} with κ=0.1, c=0.5) have baselines that differ
/// from the base scenario by more than an order of magnitude. Use
/// [`gap_closed_cell`] for those cells.
///
/// For reference, on the canonical cell (β=0.5), uniform random scores
/// per-step team ≈ `-589` and the specialist scores per-step team ≈
/// `-115` — both far below the base-scenario baselines, so applying
/// `gap_closed` to a canonical-cell payoff produces a deeply negative
/// and misleading value.
///
/// # Example
///
/// ```
/// use thrust_rl::multi_agent::bucket_brigade_metrics::{
///     MINSPEC_RANDOM, MINSPEC_SPECIALIST, gap_closed,
/// };
///
/// assert!((gap_closed(MINSPEC_RANDOM) - 0.0).abs() < 1e-5);
/// assert!((gap_closed(MINSPEC_SPECIALIST) - 1.0).abs() < 1e-5);
/// ```
pub fn gap_closed(per_step_team: f32) -> f32 {
    (per_step_team - MINSPEC_RANDOM) / (MINSPEC_SPECIALIST - MINSPEC_RANDOM)
}

/// Compute the fraction of the random→specialist gap closed by the
/// given per-step team payoff for one of the three no-convergence cells
/// of the workshop-paper phase diagram.
///
/// Routes to the appropriate per-cell baselines:
///
/// * [`BucketBrigadeCell::Beta01`] → [`MINSPEC_RANDOM_BETA01`] /
///   [`MINSPEC_SPECIALIST_BETA01`]
/// * [`BucketBrigadeCell::Beta05`] → [`MINSPEC_RANDOM_BETA05`] /
///   [`MINSPEC_SPECIALIST_BETA05`]
/// * [`BucketBrigadeCell::Beta09`] → [`MINSPEC_RANDOM_BETA09`] /
///   [`MINSPEC_SPECIALIST_BETA09`]
///
/// Same shape as [`gap_closed`]: `0.0` at random baseline, `1.0` at
/// specialist baseline, negative if worse than random.
///
/// # Example
///
/// ```
/// use thrust_rl::multi_agent::{
///     bucket_brigade_baselines::BucketBrigadeCell,
///     bucket_brigade_metrics::{
///         MINSPEC_RANDOM_BETA05, MINSPEC_SPECIALIST_BETA05, gap_closed_cell,
///     },
/// };
///
/// assert!((gap_closed_cell(MINSPEC_RANDOM_BETA05, BucketBrigadeCell::Beta05)).abs() < 1e-5);
/// assert!(
///     (gap_closed_cell(MINSPEC_SPECIALIST_BETA05, BucketBrigadeCell::Beta05) - 1.0).abs() < 1e-5
/// );
/// ```
pub fn gap_closed_cell(per_step_team: f32, cell: BucketBrigadeCell) -> f32 {
    let (random_baseline, specialist_baseline) = match cell {
        BucketBrigadeCell::Beta01 => (MINSPEC_RANDOM_BETA01, MINSPEC_SPECIALIST_BETA01),
        BucketBrigadeCell::Beta05 => (MINSPEC_RANDOM_BETA05, MINSPEC_SPECIALIST_BETA05),
        BucketBrigadeCell::Beta09 => (MINSPEC_RANDOM_BETA09, MINSPEC_SPECIALIST_BETA09),
    };
    (per_step_team - random_baseline) / (specialist_baseline - random_baseline)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spot-check against the Python reference at three canonical points
    /// (random baseline, specialist baseline, midpoint). IEEE-754
    /// arithmetic makes the Rust and Python `gap_closed` outputs
    /// bit-identical modulo rounding noise.
    #[test]
    fn matches_python_spot_checks() {
        // Endpoints: exact 0.0 and 1.0 by construction.
        assert!((gap_closed(MINSPEC_RANDOM) - 0.0).abs() < 1e-5);
        assert!((gap_closed(MINSPEC_SPECIALIST) - 1.0).abs() < 1e-5);

        // Midpoint: (random + specialist) / 2 → gap_closed = 0.5.
        let mid = (MINSPEC_RANDOM + MINSPEC_SPECIALIST) / 2.0;
        assert!(
            (gap_closed(mid) - 0.5).abs() < 1e-5,
            "midpoint gap_closed should be 0.5, got {}",
            gap_closed(mid)
        );

        // Quarter point: gap_closed = 0.25.
        let q = MINSPEC_RANDOM + 0.25 * (MINSPEC_SPECIALIST - MINSPEC_RANDOM);
        assert!(
            (gap_closed(q) - 0.25).abs() < 1e-5,
            "quarter-point gap_closed should be 0.25, got {}",
            gap_closed(q)
        );
    }

    /// A team payoff of `-50.0` (between random and specialist on
    /// minimal_specialization with `MINSPEC_RANDOM = -87.72`,
    /// `MINSPEC_SPECIALIST = -22.07`) should map to a positive but
    /// sub-1.0 `gap_closed`. Hand-computed against the **post-#246**
    /// canonical baselines:
    /// `(-50.0 - (-87.72)) / (-22.07 - (-87.72)) = 37.72 / 65.65 ≈ 0.5746`.
    #[test]
    fn intermediate_payoff_in_unit_interval() {
        let gc = gap_closed(-50.0);
        assert!(
            (gc - 0.5746).abs() < 1e-3,
            "intermediate payoff -50.0 should give gap_closed ~ 0.5746, got {gc}"
        );
        assert!((0.0..=1.0).contains(&gc));
    }

    /// Each cell's `gap_closed_cell` must return exactly `0.0` at its
    /// random baseline and exactly `1.0` at its specialist baseline
    /// (the six endpoints AC #6 requires).
    #[test]
    fn cell_baselines_round_trip() {
        for (cell, random, specialist) in [
            (BucketBrigadeCell::Beta01, MINSPEC_RANDOM_BETA01, MINSPEC_SPECIALIST_BETA01),
            (BucketBrigadeCell::Beta05, MINSPEC_RANDOM_BETA05, MINSPEC_SPECIALIST_BETA05),
            (BucketBrigadeCell::Beta09, MINSPEC_RANDOM_BETA09, MINSPEC_SPECIALIST_BETA09),
        ] {
            assert!(
                (gap_closed_cell(random, cell)).abs() < 1e-5,
                "cell {cell:?} random endpoint should give 0.0, got {}",
                gap_closed_cell(random, cell)
            );
            assert!(
                (gap_closed_cell(specialist, cell) - 1.0).abs() < 1e-5,
                "cell {cell:?} specialist endpoint should give 1.0, got {}",
                gap_closed_cell(specialist, cell)
            );
        }
    }

    /// Cell-specific `gap_closed_cell` and base-scenario `gap_closed`
    /// produce different answers on the same payoff because the
    /// normalization is different — sanity check that the two functions
    /// are NOT accidentally aliased.
    #[test]
    fn cell_and_base_gap_closed_diverge() {
        let payoff = -200.0;
        let base = gap_closed(payoff);
        let cell_b05 = gap_closed_cell(payoff, BucketBrigadeCell::Beta05);
        assert!(
            (base - cell_b05).abs() > 1.0,
            "gap_closed and gap_closed_cell must produce meaningfully different answers on the \
             same payoff, got base = {base}, cell = {cell_b05}"
        );
    }
}
