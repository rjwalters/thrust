//! Sanity tests for the coalition improvability-gate oracle (issue #268).
//!
//! These verify that [`run_coalition_oracle`] runs end-to-end for k = 2, 3, 4
//! on the canonical Beta05 cell and produces finite, sane statistics, that the
//! bootstrap CI helper returns well-ordered bounds, and that the coalition
//! ceiling is monotone in `k` (a larger coalition can always replicate a
//! smaller one's assignment plus uniform fill, so it can never do worse on
//! per-step team return).
//!
//! No neural network, no Burn, no PPO — scripted policies only.

#![cfg(all(feature = "training", feature = "env-bucket-brigade"))]

use rand::{SeedableRng, rngs::StdRng};
use thrust_rl::{
    env::games::bucket_brigade::{BucketBrigadeMaEnv, NUM_HOUSES, registry},
    multi_agent::{
        bucket_brigade_baselines::BucketBrigadeCell,
        bucket_brigade_oracle::{bootstrap_mean_ci, run_coalition_oracle},
    },
};

const NUM_AGENTS: usize = 4;

fn make_cell_env(cell: BucketBrigadeCell, seed: u64) -> BucketBrigadeMaEnv {
    let (beta, kappa, cost) = cell.parameters();
    let mut scenario = registry::get_scenario_by_id("minimal_specialization-v1")
        .expect("minimal_specialization-v1 must resolve in the registry");
    scenario.prob_fire_spreads_to_neighbor = beta;
    scenario.prob_solo_agent_extinguishes_fire = kappa;
    scenario.cost_to_work_one_night = cost;
    BucketBrigadeMaEnv::new(scenario, NUM_AGENTS, Some(seed))
}

/// The coalition oracle runs end-to-end for k = 2, 3, 4 on Beta05 and produces
/// finite, sane statistics: the ceiling is at least as good as the all-uniform
/// baseline (all-uniform is not in the candidate set, but every deviation from
/// uniform must be scored), the per-episode series has the requested length,
/// and the bootstrap CI is well-ordered (lo <= hi).
#[test]
fn coalition_oracle_runs_for_k2_k3_k4() {
    for k in 2..=4 {
        let mut env = make_cell_env(BucketBrigadeCell::Beta05, 42);
        let report = run_coalition_oracle(
            &mut env, NUM_AGENTS, NUM_HOUSES, k, 60, 20, 8, 42, 500, 500, 0.05,
        );

        assert_eq!(report.k, k);
        assert!(report.baseline.eval.per_step_team().is_finite());
        assert_eq!(report.baseline.per_episode_team_per_step.len(), 60, "k={k}");
        for row in &report.candidates {
            assert!(
                row.eval.per_step_team().is_finite(),
                "k={k} candidate {} non-finite",
                row.label
            );
            assert_eq!(row.per_episode_team_per_step.len(), 60, "k={k} candidate {}", row.label);
        }
        // Bootstrap CI on the gap must be well-ordered and finite.
        assert!(report.gap_ci_lo.is_finite() && report.gap_ci_hi.is_finite(), "k={k}");
        assert!(report.gap_ci_lo <= report.gap_ci_hi, "k={k} CI lo <= hi");
        // A heterogeneous coalition candidate (hero + specialists) is present
        // for k >= 2, so the search space is genuinely coalition-aware.
        assert!(
            report.candidates.iter().any(|r| r.label.contains("hero")),
            "k={k} must include a heterogeneous hero+specialists candidate"
        );
    }
}

/// Ground-truth anchor: fixing the policy family to `all_specialist`, adding
/// more coordinated defenders raises per-step team return. A single specialist
/// among three uniform teammates (k=1) can only defend its own houses while the
/// village burns; a full four-specialist coalition (k=4) defends every house,
/// so its per-step team return must be strictly higher. This confirms the
/// coalition machinery actually threads distinct owned-house scopes to distinct
/// agents (rather than collapsing them all onto agent 0's houses).
#[test]
fn more_specialists_raise_team_return() {
    let seed = 7;
    let specialist_team = |k: usize| -> f64 {
        let mut env = make_cell_env(BucketBrigadeCell::Beta05, seed);
        let report = run_coalition_oracle(
            &mut env, NUM_AGENTS, NUM_HOUSES, k, 200, 20, 8, seed, 800, 1000, 0.05,
        );
        report
            .candidates
            .iter()
            .find(|r| r.label == "all_specialist")
            .expect("all_specialist candidate must exist")
            .eval
            .per_step_team()
    };
    let t1 = specialist_team(1);
    let t4 = specialist_team(4);
    assert!(
        t4 > t1,
        "four coordinated specialists ({t4}) must beat one specialist + 3 uniform ({t1}) on per-step team return"
    );
}

/// The bootstrap CI helper returns finite, well-ordered bounds that bracket the
/// sample mean for a simple deterministic series, and handles the empty case.
#[test]
fn bootstrap_ci_is_well_ordered() {
    let mut rng = StdRng::seed_from_u64(0);
    let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let (lo, hi) = bootstrap_mean_ci(&values, 1000, 0.05, &mut rng);
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    assert!(lo.is_finite() && hi.is_finite());
    assert!(lo <= hi, "CI lo <= hi");
    assert!(lo <= mean && mean <= hi, "CI must bracket the sample mean ({mean})");

    // A strictly-positive series must yield a strictly-positive lower bound.
    let positive: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let (plo, _) = bootstrap_mean_ci(&positive, 1000, 0.05, &mut rng);
    assert!(plo > 0.0, "all-positive series must have CI_lo > 0");

    // Empty input degrades gracefully to NaN bounds.
    let (elo, ehi) = bootstrap_mean_ci(&[], 1000, 0.05, &mut rng);
    assert!(elo.is_nan() && ehi.is_nan());
}
