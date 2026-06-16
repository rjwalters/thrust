//! Bucket-brigade-specific scoring metrics.
//!
//! Rust mirror of the small numerical helpers used by the bucket-brigade
//! Python experiments. Kept under `crate::multi_agent` because it's only
//! meaningful for the multi-agent bucket-brigade training paths
//! (PSRO/NFSP test assertions), and gated on `env-bucket-brigade` so the
//! published crate (which strips the bucket-brigade adapter) doesn't pay
//! for the dead module.
//!
//! # `gap_closed` baselines
//!
//! `MINSPEC_RANDOM` and `MINSPEC_SPECIALIST` are hardcoded constants
//! ported verbatim from the upstream
//! `envs/bucket-brigade/experiments/p3_specialization/analyze_291.py`
//! (lines 41–42). They are **specific to the `minimal_specialization`
//! scenario family** and the canonical 4-agent ring; using them on any
//! other scenario base (e.g. `overcrowding`, `mixed_motivation`) yields
//! a meaningless ratio. The associated random/specialist Monte-Carlo
//! re-derivation lives in the Python experiment tree; we ship the
//! constants only.

/// Per-step team payoff baseline for the random policy on the
/// `minimal_specialization` scenario family (4 agents, 10 houses).
///
/// Mirrors `MINSPEC_RANDOM = -96.07` in
/// `envs/bucket-brigade/experiments/p3_specialization/analyze_291.py:41`.
pub const MINSPEC_RANDOM: f32 = -96.07;

/// Per-step team payoff baseline for the optimal specialist policy on
/// the `minimal_specialization` scenario family (4 agents, 10 houses).
///
/// Mirrors `MINSPEC_SPECIALIST = -22.07` in
/// `envs/bucket-brigade/experiments/p3_specialization/analyze_291.py:42`.
pub const MINSPEC_SPECIALIST: f32 = -22.07;

/// Compute the fraction of the random→specialist gap closed by the
/// given per-step team payoff.
///
/// ```text
/// gap_closed = (per_step_team - MINSPEC_RANDOM)
///              / (MINSPEC_SPECIALIST - MINSPEC_RANDOM)
/// ```
///
/// * Returns `0.0` when the agent matches the random baseline.
/// * Returns `1.0` when the agent matches the specialist baseline.
/// * Negative values indicate the agent is *worse* than the random baseline
///   (which is the typical PPO-on-canonical-no-convergence-cell regime — the
///   workshop paper reports `gap_closed = -0.049` for that case).
///
/// # Scope
///
/// **Only valid for the `minimal_specialization` scenario family.** The
/// random/specialist baselines change with the scenario reward
/// landscape; other scenarios need their own constants. See the module
/// docstring for the upstream Python source the constants are ported
/// from.
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

        // PPO workshop-paper baseline:
        //   per_step_team = MINSPEC_RANDOM + (-0.049) * (MINSPEC_SPECIALIST -
        // MINSPEC_RANDOM) round-trip through `gap_closed` should give -0.049.
        let ppo_baseline_payoff = MINSPEC_RANDOM + (-0.049) * (MINSPEC_SPECIALIST - MINSPEC_RANDOM);
        assert!(
            (gap_closed(ppo_baseline_payoff) - (-0.049)).abs() < 1e-5,
            "PPO workshop baseline round-trip mismatch: got {}",
            gap_closed(ppo_baseline_payoff)
        );
    }

    /// A team payoff of `-50.0` (between random and specialist on
    /// minimal_specialization) should map to a positive but sub-1.0
    /// `gap_closed`. Hand-computed: (-50.0 - (-96.07)) / (-22.07 -
    /// (-96.07)) = 46.07 / 74.0 ≈ 0.6226.
    #[test]
    fn intermediate_payoff_in_unit_interval() {
        let gc = gap_closed(-50.0);
        assert!(
            (gc - 0.6226).abs() < 1e-3,
            "intermediate payoff -50.0 should give gap_closed ~ 0.6226, got {gc}"
        );
        assert!((0.0..=1.0).contains(&gc));
    }
}
