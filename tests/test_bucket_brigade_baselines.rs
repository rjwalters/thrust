//! Monte Carlo baseline recomputation for the no-convergence cells.
//!
//! Issue #128. Produces the six cell-specific constants used by
//! [`thrust_rl::multi_agent::bucket_brigade_metrics::gap_closed_cell`]:
//!
//! * `MINSPEC_RANDOM_BETA01`, `MINSPEC_SPECIALIST_BETA01`
//! * `MINSPEC_RANDOM_BETA05`, `MINSPEC_SPECIALIST_BETA05`
//! * `MINSPEC_RANDOM_BETA09`, `MINSPEC_SPECIALIST_BETA09`
//!
//! # Protocol
//!
//! Mirrors `envs/bucket-brigade/experiments/p3_specialization/diagnostics/
//! random_baseline.py`:
//!
//! * **Episodes**: `N_EPISODES = 1000`, each `EPISODE_STEPS = 200` long. (The
//!   Python protocol uses 200 episodes × 5 seeds = 1000 total, which is the
//!   same `n=1000` sample size.)
//! * **Action distribution (random)**: uniform `MultiDiscrete([10, 2, 2])` per
//!   agent per step.
//! * **Action distribution (specialist)**: deterministic per
//!   [`thrust_rl::multi_agent::bucket_brigade_baselines::specialist_action`].
//! * **Metric**: per-step team reward = `sum_i r_i` averaged across all
//!   non-terminal steps in the rollout. We follow the Rust convention used by
//!   the NFSP/PSRO tests: `total_team_reward / total_steps` over a fixed
//!   `EPISODE_STEPS`-step rollout that resets on `done`. This is what the
//!   `gap_closed` metric consumes downstream.
//!
//! # How to run
//!
//! ```bash
//! cargo test --release --features "training,env-bucket-brigade" \
//!     --test test_bucket_brigade_baselines -- --ignored --nocapture
//! ```
//!
//! Wall-clock is ~8 minutes total for all six constants in release mode
//! (3 cells × 2 baselines × 1000 episodes × 200 steps × 4 agents ≈ 4.8M
//! env steps). The test prints a copy-pasteable Rust block at the end;
//! update the `MINSPEC_*_BETA0XX` constants in
//! `src/multi_agent/bucket_brigade_metrics.rs` from that output.
//!
//! # Unit test (always runs)
//!
//! `specialist_matches_python_handpicked_observation` is a fast unit-test
//! Python-parity check; it does NOT need `--ignored` and runs as part of
//! the regular `cargo test` sweep.

#![cfg(all(feature = "training", feature = "env-bucket-brigade"))]

use rand::{Rng, SeedableRng, rngs::StdRng};
use thrust_rl::{
    env::games::bucket_brigade::{BucketBrigadeMaEnv, NUM_HOUSES, registry},
    multi_agent::{
        JointEnv,
        bucket_brigade_baselines::{BucketBrigadeCell, specialist_action},
    },
};

const NUM_AGENTS: usize = 4;
const EPISODE_STEPS: usize = 200;
const N_EPISODES: usize = 1000;
const SEED_BASE: u64 = 42;

/// Build a fresh env for the given cell. Mirrors `make_cell_env` from
/// PR #129's `test_psro_bucket_brigade.rs`.
fn make_cell_env(cell: BucketBrigadeCell, seed: Option<u64>) -> BucketBrigadeMaEnv {
    let (beta, kappa, cost) = cell.parameters();
    let mut scenario = registry::get_scenario_by_id("minimal_specialization-v1")
        .expect("minimal_specialization-v1 must resolve in the registry");
    scenario.prob_fire_spreads_to_neighbor = beta;
    scenario.prob_solo_agent_extinguishes_fire = kappa;
    scenario.cost_to_work_one_night = cost;
    BucketBrigadeMaEnv::new(scenario, NUM_AGENTS, seed)
}

/// Drive a uniform-random policy for one episode and return the mean
/// per-step team reward over `EPISODE_STEPS` steps. Resets on `done`.
fn random_episode_per_step_team(cell: BucketBrigadeCell, seed: u64) -> f32 {
    let mut env = make_cell_env(cell, Some(seed));
    let _ = env.reset_joint(Some(seed));
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
    let mut total: f32 = 0.0;
    for _ in 0..EPISODE_STEPS {
        let actions: Vec<Vec<i64>> = (0..NUM_AGENTS)
            .map(|_| {
                vec![
                    rng.random_range(0..NUM_HOUSES as i64),
                    rng.random_range(0..2_i64),
                    rng.random_range(0..2_i64),
                ]
            })
            .collect();
        let res = env.step_joint(&actions);
        total += res.rewards.iter().sum::<f32>();
        if res.done {
            let _ = env.reset_joint(None);
        }
    }
    total / EPISODE_STEPS as f32
}

/// Drive the deterministic specialist policy for one episode and return
/// the mean per-step team reward over `EPISODE_STEPS` steps. Resets on
/// `done`. The specialist is deterministic given the obs, so the
/// per-episode variance comes entirely from the env's RNG (fire-spread
/// and extinguish draws).
fn specialist_episode_per_step_team(cell: BucketBrigadeCell, seed: u64) -> f32 {
    let mut env = make_cell_env(cell, Some(seed));
    let mut obs = env.reset_joint(Some(seed));
    let mut total: f32 = 0.0;
    for _ in 0..EPISODE_STEPS {
        let actions: Vec<Vec<i64>> = (0..NUM_AGENTS)
            .map(|i| {
                let a = specialist_action(&obs[i], i, NUM_AGENTS, NUM_HOUSES);
                vec![a[0], a[1], a[2]]
            })
            .collect();
        let res = env.step_joint(&actions);
        total += res.rewards.iter().sum::<f32>();
        if res.done {
            obs = env.reset_joint(None);
        } else {
            obs = res.observations;
        }
    }
    total / EPISODE_STEPS as f32
}

/// Average per-step team reward across `N_EPISODES` episodes (with
/// distinct seeds). The Python protocol uses 5 seeds × 200 episodes;
/// we use 1000 distinct seeds — same sample size, simpler bookkeeping.
fn average_random(cell: BucketBrigadeCell) -> f32 {
    let mut sum: f64 = 0.0;
    for ep in 0..N_EPISODES {
        let seed = SEED_BASE.wrapping_add(ep as u64);
        sum += random_episode_per_step_team(cell, seed) as f64;
    }
    (sum / N_EPISODES as f64) as f32
}

fn average_specialist(cell: BucketBrigadeCell) -> f32 {
    let mut sum: f64 = 0.0;
    for ep in 0..N_EPISODES {
        let seed = SEED_BASE.wrapping_add(ep as u64).wrapping_add(0xDEADBEEF);
        sum += specialist_episode_per_step_team(cell, seed) as f64;
    }
    (sum / N_EPISODES as f64) as f32
}

/// Fast Python-parity check on a hand-picked observation. AC #9 from
/// issue #128 (and a guard against regressions in the Rust specialist
/// port). Always runs (not `#[ignore]`).
#[test]
fn specialist_matches_python_handpicked_observation() {
    // From the Python reference:
    //   houses = [SAFE, BURNING, SAFE, SAFE, BURNING, SAFE, SAFE, SAFE, SAFE, SAFE]
    //   num_agents = 4, num_houses = 10
    //   round-robin: agent 0 -> [0,4,8], agent 1 -> [1,5,9], agent 2 -> [2,6],
    // agent 3 -> [3,7] Expected:
    //   agent 0 owns house 4 (burning) -> [4, WORK, WORK] = [4, 1, 1]
    //   agent 1 owns house 1 (burning) -> [1, WORK, WORK] = [1, 1, 1]
    //   agent 2 owns [2, 6] (none burning) -> [2, REST, REST] = [2, 0, 0]
    //   agent 3 owns [3, 7] (none burning) -> [3, REST, REST] = [3, 0, 0]
    let mut obs = vec![0.0f32; 1 + NUM_HOUSES + 64];
    obs[1 + 1] = 1.0; // house 1 burning
    obs[1 + 4] = 1.0; // house 4 burning

    assert_eq!(specialist_action(&obs, 0, NUM_AGENTS, NUM_HOUSES), [4, 1, 1]);
    assert_eq!(specialist_action(&obs, 1, NUM_AGENTS, NUM_HOUSES), [1, 1, 1]);
    assert_eq!(specialist_action(&obs, 2, NUM_AGENTS, NUM_HOUSES), [2, 0, 0]);
    assert_eq!(specialist_action(&obs, 3, NUM_AGENTS, NUM_HOUSES), [3, 0, 0]);
}

/// Re-derive the six cell-specific baselines via Monte Carlo. `#[ignore]`-gated
/// because wall-clock is ~8 min in release mode. Prints a copy-pasteable
/// Rust block at the end; manually update the constants in
/// `src/multi_agent/bucket_brigade_metrics.rs`.
#[test]
#[ignore = "wall-clock ~8 min; run with --ignored --nocapture to regenerate baselines"]
fn recompute_cell_baselines() {
    let t_start = std::time::Instant::now();
    println!(
        "Recomputing cell-specific baselines: {N_EPISODES} episodes × {EPISODE_STEPS} steps × \
         {NUM_AGENTS} agents per (cell, baseline). 3 cells × 2 baselines = 6 constants.\n"
    );

    let mut results: Vec<(BucketBrigadeCell, f32, f32)> = Vec::new();
    for cell in BucketBrigadeCell::all() {
        let (beta, kappa, cost) = cell.parameters();
        println!(
            "== Cell {:?} ({}): β = {}, κ = {}, c = {} ==",
            cell,
            cell.tag(),
            beta,
            kappa,
            cost
        );
        let t_cell = std::time::Instant::now();
        let r = average_random(cell);
        let t_random = t_cell.elapsed();
        let s = average_specialist(cell);
        let t_specialist = t_cell.elapsed() - t_random;
        println!("  random      = {r:.4} ({t_random:?})\n  specialist = {s:.4} ({t_specialist:?})");
        results.push((cell, r, s));
    }

    println!("\n========================================");
    println!("Copy-paste into src/multi_agent/bucket_brigade_metrics.rs:");
    println!("========================================");
    for (cell, r, s) in &results {
        let suffix = match cell {
            BucketBrigadeCell::Beta01 => "BETA01",
            BucketBrigadeCell::Beta05 => "BETA05",
            BucketBrigadeCell::Beta09 => "BETA09",
        };
        println!("pub const MINSPEC_RANDOM_{suffix}: f32 = {r:.4};");
        println!("pub const MINSPEC_SPECIALIST_{suffix}: f32 = {s:.4};");
    }
    println!("========================================");
    println!("Total wall-clock: {:?}", t_start.elapsed());
}
