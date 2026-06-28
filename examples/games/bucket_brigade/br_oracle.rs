//! Best-response **improvability oracle** for bucket-brigade (issue #259).
//!
//! Answers the issue's acceptance-criterion #1 ("improvability gate"): before
//! touching the PPO update, establish — via a *non-PPO* method — whether a
//! better-than-uniform best-response even *exists* against the frozen uniform
//! opponents `train_br_probe` trains against.
//!
//! This harness scores a battery of scripted policies (uniform baseline,
//! always-rest, specialist, deterministic firefighters, and a randomized
//! search over the firefighter family) for the single BR agent against
//! `N − 1` frozen uniform-random opponents, and reports the best achievable
//! per-step **team** return (and per-step / per-episode **BR-agent** return)
//! versus the all-uniform baseline. No neural network, no Burn, no PPO.
//!
//! See `crate::multi_agent::bucket_brigade_oracle` for the methodology and the
//! `FirefighterParams` search family.
//!
//! # Usage
//!
//! ```bash
//! # Improvability gate on the canonical cell:
//! CELL=beta05 EVAL_EPISODES=400 \
//!     cargo run --release --example br_oracle \
//!         --features "training,env-bucket-brigade"
//!
//! # All three no-convergence cells:
//! for c in beta01 beta05 beta09; do
//!   CELL=$c cargo run --release --example br_oracle \
//!       --features "training,env-bucket-brigade"
//! done
//! ```
//!
//! # Env-var knobs
//!
//! - `CELL` — one of `beta01|beta05|beta09` (default `beta05`).
//! - `EVAL_EPISODES` — episodes for the final reported numbers (default 400).
//! - `SEARCH_EPISODES` — episodes used to score each searched firefighter
//!   (default 40).
//! - `NUM_SEARCH` — number of random firefighters to sample (default 64).
//! - `SEED` — base RNG seed (default 42).
//! - `STEP_CAP` — per-episode step bound (default 1000; the env terminates on
//!   its own once all houses are safe or ruined after `min_nights`).

use anyhow::Result;
use thrust_rl::{
    env::games::bucket_brigade::{BucketBrigadeMaEnv, NUM_HOUSES, registry},
    multi_agent::{
        bucket_brigade_baselines::BucketBrigadeCell,
        bucket_brigade_oracle::{OracleReport, run_oracle},
    },
};

const NUM_AGENTS: usize = 4;
const DEFAULT_CELL: &str = "beta05";

fn cell_enum(cell: &str) -> BucketBrigadeCell {
    match cell {
        "beta01" => BucketBrigadeCell::Beta01,
        "beta05" => BucketBrigadeCell::Beta05,
        "beta09" => BucketBrigadeCell::Beta09,
        other => panic!("Unknown CELL '{other}'; expected one of beta01|beta05|beta09"),
    }
}

fn make_cell_env(cell: BucketBrigadeCell, seed: u64) -> BucketBrigadeMaEnv {
    let (beta, kappa, cost) = cell.parameters();
    let mut scenario = registry::get_scenario_by_id("minimal_specialization-v1")
        .expect("minimal_specialization-v1 must resolve in the registry");
    scenario.prob_fire_spreads_to_neighbor = beta;
    scenario.prob_solo_agent_extinguishes_fire = kappa;
    scenario.cost_to_work_one_night = cost;
    BucketBrigadeMaEnv::new(scenario, NUM_AGENTS, Some(seed))
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn print_report(cell: &str, report: &OracleReport) {
    let base = report.baseline.eval;
    tracing::info!("------------------------------------------------------------");
    tracing::info!("Improvability-gate oracle (issue #259) — cell {cell}");
    tracing::info!(
        "  baseline (all-uniform): per_step_team={:.3}  per_step_br={:.3}  per_ep_br={:.1}  ep_len={:.1}  ({} eps)",
        base.per_step_team(),
        base.per_step_br(),
        base.per_episode_br(),
        base.mean_ep_len(),
        base.episodes,
    );
    tracing::info!("  candidates (BR agent = policy, other {} agents uniform):", NUM_AGENTS - 1);
    for (i, row) in report.candidates.iter().enumerate() {
        let marker = if i == report.best_idx {
            " <== ceiling"
        } else {
            ""
        };
        let e = row.eval;
        tracing::info!(
            "    {:<34} per_step_team={:.3}  per_step_br={:.3}  per_ep_br={:.1}{}",
            row.label,
            e.per_step_team(),
            e.per_step_br(),
            e.per_episode_br(),
            marker,
        );
    }
    tracing::info!("------------------------------------------------------------");
    tracing::info!(
        "  CEILING: {}  per_step_team={:.3}  (baseline {:.3})",
        report.best().label,
        report.best().eval.per_step_team(),
        base.per_step_team(),
    );
    tracing::info!(
        "  TEAM GAP (ceiling − baseline): {:+.3} per step  ({:+.2}% of |baseline|)",
        report.team_gap_per_step(),
        100.0 * report.team_gap_fraction(),
    );
    let br_gap = report.best().eval.per_step_br() - base.per_step_br();
    tracing::info!(
        "  BR-AGENT GAP (ceiling − baseline): {:+.3} per step  ({:+.1} per episode)",
        br_gap,
        report.best().eval.per_episode_br() - base.per_episode_br(),
    );
    tracing::info!("------------------------------------------------------------");
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let cell = std::env::var("CELL").unwrap_or_else(|_| DEFAULT_CELL.to_string());
    let cell_e = cell_enum(&cell);
    let eval_episodes = env_usize("EVAL_EPISODES", 400);
    let search_episodes = env_usize("SEARCH_EPISODES", 40);
    let num_search = env_usize("NUM_SEARCH", 64);
    let seed = env_usize("SEED", 42) as u64;
    let step_cap = env_usize("STEP_CAP", 1000);

    tracing::info!("BR improvability-gate oracle (issue #259)");
    tracing::info!("  cell            = {cell} (β,κ,c = {:?})", cell_e.parameters());
    tracing::info!("  num_agents      = {NUM_AGENTS} (1 BR vs {} frozen uniform)", NUM_AGENTS - 1);
    tracing::info!("  eval_episodes   = {eval_episodes}");
    tracing::info!("  search_episodes = {search_episodes}");
    tracing::info!("  num_search      = {num_search}");
    tracing::info!("  seed            = {seed}");
    tracing::info!("  step_cap        = {step_cap}");

    let mut env = make_cell_env(cell_e, seed);
    let report = run_oracle(
        &mut env,
        NUM_AGENTS,
        NUM_HOUSES,
        eval_episodes,
        search_episodes,
        num_search,
        seed,
        step_cap,
    );
    print_report(&cell, &report);

    Ok(())
}
