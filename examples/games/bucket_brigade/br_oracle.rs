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
//! # Coalition mode (issue #268)
//!
//! Setting the `K` env var switches the harness into the **coalition
//! improvability gate**: it sweeps coalition size `k = 1..=K` over all three
//! no-convergence cells, scripting `k` coordinated deviators against `N−k`
//! frozen uniform opponents, and prints a `k × cell` table plus the per-cell
//! **k\*** verdict (smallest `k` whose per-episode team-return-gap bootstrap CI
//! lower bound is strictly positive).
//!
//! ```bash
//! # Measure the coordination threshold k* on all three cells:
//! K=4 EVAL_EPISODES=400 \
//!     cargo run --release --example br_oracle \
//!         --features "training,env-bucket-brigade"
//! ```
//!
//! # Phase-diagram grid mode (issue #269)
//!
//! Setting `PHASE=1` sweeps the coalition oracle (k = 1..=`K`, default 4)
//! across an arbitrary (β, κ, c) grid — by default the FULL 5×5×3 = 75-cell
//! paper grid from `compute_nash_phase_diagram.py` — writing one JSON record
//! per cell to `OUT_DIR/<cell_tag>.json` (tag format `b{β:.2}_k{κ:.2}_c{c:.2}`,
//! joinable with the Nash phase-diagram artifacts). Cells parallelize via
//! rayon; existing per-cell outputs are skipped, so re-runs resume.
//!
//! ```bash
//! # Full 75-cell grid, k = 1..4, per-cell JSONs under phase_out/:
//! PHASE=1 K=4 EVAL_EPISODES=400 OUT_DIR=phase_out \
//!     cargo run --release --example br_oracle \
//!         --features "training,env-bucket-brigade"
//!
//! # Explicit cell subset (distributed-launch knob):
//! PHASE=1 CELLS="0.1,0.1,0.5;0.3,0.5,2.0" ...
//! ```
//!
//! # Env-var knobs
//!
//! - `CELL` — one of `beta01|beta05|beta09` (default `beta05`). Ignored in
//!   coalition mode (all three cells are swept).
//! - `K` — if set, run the coalition sweep for `k = 1..=K` (issue #268). In
//!   `PHASE=1` mode `K` is the per-cell k_max (default 4).
//! - `PHASE` — set to `1` for the phase-diagram grid mode (issue #269).
//! - `BETA_VALUES` / `KAPPA_VALUES` / `C_VALUES` — comma-separated floats
//!   overriding the FULL grid axes in `PHASE=1` mode.
//! - `CELLS` — semicolon-separated explicit `beta,kappa,c` triples; overrides
//!   the cartesian product in `PHASE=1` mode.
//! - `OUT_DIR` — per-cell JSON output directory in `PHASE=1` mode (default
//!   `phase_out`).
//! - `EVAL_EPISODES` — episodes for the final reported numbers (default 400).
//! - `SEARCH_EPISODES` — episodes used to score each searched firefighter
//!   (default 40).
//! - `NUM_SEARCH` — number of random firefighters to sample (default 64).
//! - `N_BOOT` — bootstrap resamples for the gap CI in coalition mode (default
//!   1000).
//! - `ALPHA` — CI significance level in coalition mode (default 0.05 → 95% CI).
//! - `SEED` — base RNG seed (default 42).
//! - `STEP_CAP` — per-episode step bound (default 1000; the env terminates on
//!   its own once all houses are safe or ruined after `min_nights`).

use anyhow::Result;
use rayon::prelude::*;
use thrust_rl::{
    env::games::bucket_brigade::{BucketBrigadeMaEnv, NUM_HOUSES, registry},
    multi_agent::{
        bucket_brigade_baselines::BucketBrigadeCell,
        bucket_brigade_oracle::{
            OracleReport, PhaseCellRecord, phase_cell_tag, run_coalition_oracle, run_oracle,
            run_phase_cell,
        },
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

/// Coalition improvability-gate sweep (issue #268): for each no-convergence
/// cell, sweep `k = 1..=k_max` and print the `k × cell` team-return-gap table
/// with bootstrap CIs, then the per-cell `k*` verdict.
#[allow(clippy::too_many_arguments)]
fn run_coalition_sweep(
    k_max: usize,
    eval_episodes: usize,
    search_episodes: usize,
    num_search: usize,
    seed: u64,
    step_cap: usize,
    n_boot: usize,
    alpha: f64,
) {
    tracing::info!("Coalition improvability-gate oracle (issue #268)");
    tracing::info!("  k sweep         = 1..={k_max}");
    tracing::info!("  num_agents      = {NUM_AGENTS}");
    tracing::info!("  eval_episodes   = {eval_episodes}");
    tracing::info!("  search_episodes = {search_episodes}");
    tracing::info!("  num_search      = {num_search}");
    tracing::info!("  n_boot          = {n_boot}  alpha = {alpha}");
    tracing::info!("  seed            = {seed}  step_cap = {step_cap}");
    tracing::info!("============================================================");
    tracing::info!(
        "  gap_ep = episode-mean per-step team gap (CI is on this);  gap_agg = step-weighted"
    );
    tracing::info!(
        "{:<8} {:>2}  {:<34} {:>10} {:>10} {:>10} {:>10} {:>4}",
        "cell",
        "k",
        "ceiling",
        "gap_ep",
        "gap_agg",
        "CI_lo",
        "CI_hi",
        "k*?",
    );

    for cell_e in BucketBrigadeCell::all() {
        let cell = match cell_e {
            BucketBrigadeCell::Beta01 => "beta01",
            BucketBrigadeCell::Beta05 => "beta05",
            BucketBrigadeCell::Beta09 => "beta09",
        };
        let mut k_star: Option<usize> = None;
        for k in 1..=k_max {
            let mut env = make_cell_env(cell_e, seed);
            let report = run_coalition_oracle(
                &mut env,
                NUM_AGENTS,
                NUM_HOUSES,
                k,
                eval_episodes,
                search_episodes,
                num_search,
                seed,
                step_cap,
                n_boot,
                alpha,
            );
            let is_star = report.is_k_star();
            if is_star && k_star.is_none() {
                k_star = Some(k);
            }
            tracing::info!(
                "{:<8} {:>2}  {:<34} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>4}",
                cell,
                k,
                report.best().label,
                report.gap_mean,
                report.team_gap_per_step(),
                report.gap_ci_lo,
                report.gap_ci_hi,
                if is_star { "yes" } else { "no" },
            );
            // Per-candidate breakdown (per-step team return + gap vs baseline)
            // so the appended research-doc table can cite every coalition, not
            // just the ceiling.
            let base_ps = report.baseline.eval.per_step_team();
            for row in &report.candidates {
                tracing::info!(
                    "           |- {:<40} per_step_team={:>10.3}  gap={:>+8.3}",
                    row.label,
                    row.eval.per_step_team(),
                    row.eval.per_step_team() - base_ps,
                );
            }
        }
        match k_star {
            Some(k) => tracing::info!(
                "  => cell {cell}: k* = {k} (smallest k with CI_lo > 0; coordination threshold exists)"
            ),
            None => tracing::info!(
                "  => cell {cell}: NO k* <= {k_max} (flat landscape; CI spans zero at k={k_max} — near-degenerate cell)"
            ),
        }
        tracing::info!("------------------------------------------------------------");
    }
}

/// Parse a comma-separated float list from an env var, with a default.
fn env_f32_list(key: &str, default: &[f32]) -> Vec<f32> {
    match std::env::var(key) {
        Ok(s) => s
            .split(',')
            .map(|t| {
                t.trim()
                    .parse::<f32>()
                    .unwrap_or_else(|_| panic!("{key} entry '{t}' is not a float"))
            })
            .collect(),
        Err(_) => default.to_vec(),
    }
}

/// Full paper grid (5 × 5 × 3 = 75 cells) from
/// `envs/bucket-brigade/experiments/scripts/compute_nash_phase_diagram.py`
/// (`FULL_BETA_VALUES` / `FULL_KAPPA_VALUES` / `FULL_C_VALUES`).
const FULL_BETA_VALUES: [f32; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];
const FULL_KAPPA_VALUES: [f32; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];
const FULL_C_VALUES: [f32; 3] = [0.5, 1.0, 2.0];

/// Phase-diagram k* sweep (issue #269): run the `k = 1..=k_max` coalition
/// oracle on every requested `(β, κ, c)` cell in parallel (rayon over cells)
/// and write one JSON record per cell to `out_dir/<cell_tag>.json` so partial
/// failures lose one cell, not the run. Cells whose output file already exists
/// are skipped (resume support for distributed / re-run workflows).
#[allow(clippy::too_many_arguments)]
fn run_phase_sweep(
    cells: &[(f32, f32, f32)],
    out_dir: &str,
    k_max: usize,
    eval_episodes: usize,
    search_episodes: usize,
    num_search: usize,
    seed: u64,
    step_cap: usize,
    n_boot: usize,
    alpha: f64,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    tracing::info!("Phase-diagram k* sweep (issue #269)");
    tracing::info!("  cells           = {}", cells.len());
    tracing::info!("  k sweep         = 1..={k_max}");
    tracing::info!("  num_agents      = {NUM_AGENTS}");
    tracing::info!("  eval_episodes   = {eval_episodes}");
    tracing::info!("  search_episodes = {search_episodes}");
    tracing::info!("  num_search      = {num_search}");
    tracing::info!("  n_boot          = {n_boot}  alpha = {alpha}");
    tracing::info!("  seed            = {seed}  step_cap = {step_cap}");
    tracing::info!("  out_dir         = {out_dir}");

    let pending: Vec<(f32, f32, f32)> = cells
        .iter()
        .copied()
        .filter(|&(beta, kappa, c)| {
            let tag = phase_cell_tag(beta, kappa, c);
            let path = format!("{out_dir}/{tag}.json");
            if std::path::Path::new(&path).exists() {
                tracing::info!("  skip {tag} (output exists)");
                false
            } else {
                true
            }
        })
        .collect();
    tracing::info!("  pending cells   = {}", pending.len());

    let results: Vec<Result<PhaseCellRecord>> = pending
        .par_iter()
        .map(|&(beta, kappa, c)| {
            let record = run_phase_cell(
                beta,
                kappa,
                c,
                NUM_AGENTS,
                NUM_HOUSES,
                k_max,
                eval_episodes,
                search_episodes,
                num_search,
                seed,
                step_cap,
                n_boot,
                alpha,
            );
            let path = format!("{out_dir}/{}.json", record.cell_tag);
            std::fs::write(&path, serde_json::to_string_pretty(&record)?)?;
            tracing::info!(
                "  done {}: k* = {}",
                record.cell_tag,
                record.k_star.map_or("none".to_string(), |k| k.to_string()),
            );
            Ok(record)
        })
        .collect();

    // Summary table.
    tracing::info!("============================================================");
    tracing::info!("{:<22} {:>6}  per-k gap_mean [CI]", "cell_tag", "k*");
    let mut ok = 0usize;
    for r in &results {
        match r {
            Ok(rec) => {
                ok += 1;
                let per_k: Vec<String> = rec
                    .per_k
                    .iter()
                    .map(|p| {
                        format!(
                            "k{}={:+.2}[{:+.2},{:+.2}]",
                            p.k, p.gap_mean, p.gap_ci_lo, p.gap_ci_hi
                        )
                    })
                    .collect();
                tracing::info!(
                    "{:<22} {:>6}  {}",
                    rec.cell_tag,
                    rec.k_star.map_or("none".to_string(), |k| k.to_string()),
                    per_k.join("  "),
                );
            }
            Err(e) => tracing::warn!("cell failed: {e}"),
        }
    }
    tracing::info!("phase sweep complete: {ok}/{} pending cells written", pending.len());
    Ok(())
}

/// Resolve the phase-sweep cell list from env vars: an explicit `CELLS` list
/// (semicolon-separated `beta,kappa,c` triples — the knob a distributed
/// launcher uses to assign arbitrary cell subsets to nodes) or the cartesian
/// product of `BETA_VALUES` × `KAPPA_VALUES` × `C_VALUES` (defaulting to the
/// FULL 75-cell paper grid).
fn phase_cells_from_env() -> Vec<(f32, f32, f32)> {
    if let Ok(s) = std::env::var("CELLS") {
        return s
            .split(';')
            .filter(|t| !t.trim().is_empty())
            .map(|t| {
                let v: Vec<f32> = t
                    .split(',')
                    .map(|x| {
                        x.trim()
                            .parse()
                            .unwrap_or_else(|_| panic!("CELLS entry '{t}' is not b,k,c floats"))
                    })
                    .collect();
                assert!(v.len() == 3, "CELLS entry '{t}' must have exactly 3 floats");
                (v[0], v[1], v[2])
            })
            .collect();
    }
    let betas = env_f32_list("BETA_VALUES", &FULL_BETA_VALUES);
    let kappas = env_f32_list("KAPPA_VALUES", &FULL_KAPPA_VALUES);
    let cs = env_f32_list("C_VALUES", &FULL_C_VALUES);
    let mut out = Vec::with_capacity(betas.len() * kappas.len() * cs.len());
    for &b in &betas {
        for &k in &kappas {
            for &c in &cs {
                out.push((b, k, c));
            }
        }
    }
    out
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    // Phase-diagram grid mode (issue #269): triggered by `PHASE=1`. Takes
    // precedence over the single-grid coalition mode below (`K` is reused as
    // the k_max knob in both modes).
    if std::env::var("PHASE").map(|v| v == "1").unwrap_or(false) {
        let cells = phase_cells_from_env();
        let out_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| "phase_out".to_string());
        let k_max = env_usize("K", 4);
        assert!((1..=NUM_AGENTS).contains(&k_max), "K={k_max} out of range 1..={NUM_AGENTS}");
        return run_phase_sweep(
            &cells,
            &out_dir,
            k_max,
            env_usize("EVAL_EPISODES", 400),
            env_usize("SEARCH_EPISODES", 40),
            env_usize("NUM_SEARCH", 64),
            env_usize("SEED", 42) as u64,
            env_usize("STEP_CAP", 1000),
            env_usize("N_BOOT", 1000),
            std::env::var("ALPHA").ok().and_then(|s| s.parse().ok()).unwrap_or(0.05),
        );
    }

    // Coalition mode (issue #268): triggered by the `K` env var.
    if let Ok(k_str) = std::env::var("K") {
        let k_max: usize = k_str.parse().unwrap_or_else(|_| panic!("K must be a positive integer"));
        assert!((1..=NUM_AGENTS).contains(&k_max), "K={k_max} out of range 1..={NUM_AGENTS}");
        run_coalition_sweep(
            k_max,
            env_usize("EVAL_EPISODES", 400),
            env_usize("SEARCH_EPISODES", 40),
            env_usize("NUM_SEARCH", 64),
            env_usize("SEED", 42) as u64,
            env_usize("STEP_CAP", 1000),
            env_usize("N_BOOT", 1000),
            std::env::var("ALPHA").ok().and_then(|s| s.parse().ok()).unwrap_or(0.05),
        );
        return Ok(());
    }

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
