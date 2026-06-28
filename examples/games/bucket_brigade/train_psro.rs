//! PSRO bucket-brigade trainer on the no-convergence cells.
//!
//! Long-form training driver wired against `PsroTrainer` +
//! `AlphaRankMetaSolver` + `MultiDiscreteMlpBurnPolicy` for the
//! workshop-paper no-convergence regime. Default cell is canonical
//! `(β=0.5, κ=0.1, c=0.5)`; override via `CELL=beta01|beta05|beta09`.
//!
//! Backend selection: NdArray (CPU) by default; opt into Burn's wgpu
//! GPU backend with `--features "training,env-bucket-brigade,wgpu"`.
//!
//! Sibling of `examples/games/bucket_brigade/train_nfsp.rs`. PR 4/4
//! of issue #117's bucket-brigade integration chain (closes #115).
//!
//! # Algorithmic shape
//!
//! Drives the post-#125 N-tensor PSRO trainer with the post-#123
//! α-rank meta-solver on a factored `[house, mode, signal]` action
//! policy. Each PSRO outer iteration:
//!
//! 1. Solves the meta-Nash on the current empirical-payoff N-tensor cache
//!    (`AlphaRankMetaSolver::solve_n_player` for N>2).
//! 2. Round-robin trains one fresh best-response per agent against the other
//!    agents' marginal mixtures.
//! 3. Evaluates new boundary cells in the payoff tensor and re-solves to log
//!    the post-append exploitability.
//!
//! # Usage
//!
//! ```bash
//! # Default cell (canonical β=0.5):
//! cargo run --release --example train_psro \
//!     --features "training,env-bucket-brigade"
//!
//! # Override the cell or training budget:
//! TOTAL_ITERATIONS=20 ROLLOUT_STEPS=4096 CELL=beta01 \
//!     cargo run --release --example train_psro \
//!         --features "training,env-bucket-brigade"
//!
//! # GPU (wgpu) backend:
//! cargo run --release --example train_psro \
//!     --features "training,env-bucket-brigade,wgpu"
//! ```
//!
//! # Artifacts
//!
//! Per-agent final BR policies are saved to:
//! - `psro_<cell>_final_agent<i>.bin` (Burn binary, compact)
//! - `psro_<cell>_final_agent<i>.json` (PrettyJSON, inspectable)
//!
//! Final α-rank meta-Nash distribution per agent is written to:
//! - `psro_<cell>_final_meta_nash.json`
//!
//! Per-iteration progress is logged live via the PSRO `on_iteration`
//! callback (NFSP-parity, issue #202).
//!
//! # Mid-run checkpointing (issue #204)
//!
//! The same `on_iteration` callback also writes **intermediate**
//! per-agent BR checkpoints every `CHECKPOINT_INTERVAL_ITERATIONS`
//! iterations *during* `run()`, not only after it returns:
//! - `psro_<cell>_iter<n>_agent<i>.bin` (Burn binary, per agent)
//! - `psro_<cell>_iter<n>_meta_nash.json` (α-rank meta-Nash per agent)
//!
//! A killed multi-hour run therefore leaves a resumable / partial-result
//! safe artifact on disk rather than nothing. Set
//! `CHECKPOINT_INTERVAL_ITERATIONS` (env var) to tune the cadence; `0`
//! disables mid-run checkpointing. The callback receives the newest BR
//! policy per agent (`brs[i]`) directly from the trainer, so no
//! post-hoc accessor or borrow gymnastics are needed. Checkpointing is a
//! side-effect-only write and does not change the training trajectory.
//!
//! # `gap_closed` normalization
//!
//! The final-eval headline metric is `gap_closed_cell`, which normalizes
//! the per-step team payoff against the **cell-specific** random/specialist
//! baselines for the active `CELL` (the `MINSPEC_{RANDOM,SPECIALIST}_BETA0XX`
//! constants added in #128/#131). This is the correct normalization for the
//! no-convergence cells, whose payoff scale differs from the base
//! `minimal_specialization` scenario by more than an order of magnitude.
//!
//! The base-scenario `gap_closed` (using `MINSPEC_RANDOM`/`MINSPEC_SPECIALIST`)
//! is still logged as a secondary diagnostic for continuity with older runs,
//! but it should NOT be read as the convergence metric for a cell run.

use std::collections::HashMap;

use anyhow::Result;
use burn::{
    backend::Autodiff,
    module::Module,
    optim::AdamConfig,
    record::{BinFileRecorder, FullPrecisionSettings, PrettyJsonFileRecorder},
    tensor::{Tensor, TensorData},
};
use thrust_rl::{
    env::games::bucket_brigade::{BucketBrigadeMaEnv, NUM_HOUSES, registry},
    multi_agent::{
        AlphaRankMetaSolver, JointEnv, JointTrainerConfig, MetaSolver, PsroConfig, PsroTrainer,
        bucket_brigade_baselines::BucketBrigadeCell,
        bucket_brigade_metrics::{gap_closed, gap_closed_cell},
    },
    policy::multi_discrete_mlp::MultiDiscreteMlpBurnPolicy,
    train::optimizer::BurnOptimizer,
};

// Concrete backend stack — selected at compile time via Cargo features.
// `--features "training,env-bucket-brigade,wgpu"` swaps the CPU NdArray
// default for Burn's cross-platform GPU backend (Vulkan / Metal / DX12 /
// WebGPU). Mirrors the pattern in
// `examples/games/pong/train_pong_self_play.rs`.
#[cfg(not(feature = "wgpu"))]
type InnerBackend = burn::backend::NdArray<f32>;
#[cfg(feature = "wgpu")]
type InnerBackend = burn::backend::Wgpu<f32, i32>;
type B = Autodiff<InnerBackend>;

#[cfg(not(feature = "wgpu"))]
const BACKEND_LABEL: &str = "NdArray<f32> + Autodiff (CPU)";
#[cfg(feature = "wgpu")]
const BACKEND_LABEL: &str = "Wgpu<f32, i32> + Autodiff (GPU: Vulkan/Metal/DX12/WebGPU)";

const NUM_AGENTS: usize = 4;
const HIDDEN_DIM: usize = 64;
/// Base RNG seed. Override with the `SEED` env var (default 42) so multiple
/// machines can run independent replicates of the same config (issue #134
/// seeded validation). Behavior-preserving when `SEED` is unset.
fn base_seed() -> u64 {
    std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(42)
}
const DEFAULT_TOTAL_ITERATIONS: usize = 50;
const DEFAULT_ROLLOUT_STEPS: usize = 2048;
const DEFAULT_CELL: &str = "beta05";
/// Default mid-run checkpoint cadence: write per-agent BR + meta-Nash
/// every Nth outer iteration (issue #204). Overridable via the
/// `CHECKPOINT_INTERVAL_ITERATIONS` env var; `0` disables mid-run
/// checkpointing (only the final post-run artifacts are written).
const DEFAULT_CHECKPOINT_INTERVAL_ITERATIONS: usize = 5;
/// Length of the post-iteration deterministic eval rollout used to log
/// the running per-step team estimate.
const EVAL_STEPS: usize = 50;

fn cell_params(cell: &str) -> (f32, f32, f32) {
    cell_enum(cell).parameters()
}

/// Map the `CELL` env-var string to the corresponding [`BucketBrigadeCell`],
/// which selects the cell-specific `gap_closed_cell` baselines.
fn cell_enum(cell: &str) -> BucketBrigadeCell {
    match cell {
        "beta01" => BucketBrigadeCell::Beta01,
        "beta05" => BucketBrigadeCell::Beta05,
        "beta09" => BucketBrigadeCell::Beta09,
        other => panic!("Unknown CELL '{other}'; expected one of beta01|beta05|beta09"),
    }
}

fn make_cell_env(beta: f32, kappa: f32, cost: f32, seed: Option<u64>) -> BucketBrigadeMaEnv {
    let mut scenario = registry::get_scenario_by_id("minimal_specialization-v1")
        .expect("minimal_specialization-v1 must resolve in the registry");
    scenario.prob_fire_spreads_to_neighbor = beta;
    scenario.prob_solo_agent_extinguishes_fire = kappa;
    scenario.cost_to_work_one_night = cost;
    BucketBrigadeMaEnv::new(scenario, NUM_AGENTS, seed)
}

/// Deterministic eval rollout on a fresh env. Returns mean per-step
/// team reward over `EVAL_STEPS` steps using the supplied policies
/// (one per agent).
fn eval_per_step_team_reward(
    policies: &[MultiDiscreteMlpBurnPolicy<B>],
    device: &burn::tensor::Device<InnerBackend>,
    obs_dim: usize,
    beta: f32,
    kappa: f32,
    cost: f32,
    seed_xor: u64,
) -> f32 {
    use rand::SeedableRng;
    let mut env = make_cell_env(beta, kappa, cost, Some(base_seed() ^ seed_xor));
    let mut last_obs = env.reset_joint(Some(base_seed() ^ seed_xor));
    let mut total_team_reward: f32 = 0.0;
    let mut steps: usize = 0;
    let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed() ^ seed_xor.wrapping_add(1));
    for _ in 0..EVAL_STEPS {
        let mut joint_actions: Vec<Vec<i64>> = Vec::with_capacity(NUM_AGENTS);
        for i in 0..NUM_AGENTS {
            let obs_row = &last_obs[i];
            let obs_tensor =
                Tensor::<B, 2>::from_data(TensorData::new(obs_row.clone(), [1, obs_dim]), device);
            let (acts, _, _) = policies[i].get_action_host_seeded(obs_tensor, &mut rng);
            joint_actions.push(acts);
        }
        let result = env.step_joint(&joint_actions);
        let team: f32 = result.rewards.iter().sum();
        total_team_reward += team;
        steps += 1;
        last_obs = result.observations;
        if result.done {
            last_obs = env.reset_joint(None);
        }
    }
    total_team_reward / steps.max(1) as f32
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let total_iterations: usize = std::env::var("TOTAL_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TOTAL_ITERATIONS);
    let rollout_steps: usize = std::env::var("ROLLOUT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ROLLOUT_STEPS);
    let cell = std::env::var("CELL").unwrap_or_else(|_| DEFAULT_CELL.to_string());
    let (beta, kappa, cost) = cell_params(&cell);
    let checkpoint_interval: usize = std::env::var("CHECKPOINT_INTERVAL_ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CHECKPOINT_INTERVAL_ITERATIONS);
    // Optional per-iteration payoff-eval cap (issue #212). Unset =>
    // evaluate the full boundary slab (bit-identical to the uncapped
    // path). Set MAX_PAYOFF_EVALS_PER_ITER=<N> to deterministically
    // subsample the boundary and bound per-iteration cost on the
    // 4-player (population⁴) game.
    let max_payoff_evals_per_iteration: Option<usize> =
        std::env::var("MAX_PAYOFF_EVALS_PER_ITER").ok().and_then(|s| s.parse().ok());
    // Optional BR reward scaling (issue #199 / #215). Unset => 1.0
    // (no-op). The unscaled `[−700, 0]` bucket-brigade band drives the
    // BR critic's value targets and advantage stats into a range where
    // the value loss dominates the surrogate; `BR_REWARD_SCALE=0.01`
    // rescales it to roughly `[−7, 0]` (affine transform; same optimal
    // policy). Mirrors NFSP's `br_reward_scale` knob.
    let br_reward_scale: f32 = std::env::var("BR_REWARD_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    // Optional α-rank payoff-span normalization (issue #215). Unset =>
    // off (bit-identical to the pre-#215 solver). Set
    // `ALPHA_RANK_NORMALIZE_SPAN=1` to divide the Moran payoff
    // differential by the payoff span so `α · delta` does not saturate
    // to a hard 0/1 on the large-magnitude bucket-brigade band.
    let alpha_rank_normalize_span: bool = std::env::var("ALPHA_RANK_NORMALIZE_SPAN")
        .ok()
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    // Issue #239: BR-oracle learning knobs (see train_nfsp.rs for the full
    // root-cause note). The single-BR probe (#241) found the critic only
    // starts fitting (explained-variance rises off ~0, entropy falls below the
    // uniform 1.0 floor) with `BR_REWARD_SCALE=0.001 VF_COEF=0.5
    // BR_TRAIN_STEPS=8`. PSRO's `BR_REWARD_SCALE` default is left at 1.0
    // (no-op) to keep the operator-gated #134 PSRO re-run bit-identical; run
    // with that recipe explicitly to enable the strong-BR path. grad-clip
    // (`max_grad_norm`) is applied and the update walks all minibatches.
    let vf_coef: f64 = std::env::var("VF_COEF").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let br_train_steps: usize =
        std::env::var("BR_TRAIN_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    // Issue #251 throughput lever: cap the number of minibatch gradient steps
    // per epoch in the BR PPO update. `None` (default) keeps the #239
    // all-minibatch full-rollout coverage bit-identical; a small value (e.g.
    // 2) trades a bounded amount of BR fit for a large per-iter speedup over
    // the un-batchable (#235) bucket-brigade rollout.
    let max_minibatches_per_epoch: Option<usize> = std::env::var("BR_MAX_MINIBATCHES_PER_EPOCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|v| *v > 0);

    tracing::info!("Starting PSRO bucket-brigade training (Burn backend: {BACKEND_LABEL})");
    tracing::info!("  cell             = {cell} (β={beta}, κ={kappa}, c={cost})");
    tracing::info!("  total_iterations = {total_iterations}");
    tracing::info!(
        "  checkpoint_every = {} iters{}",
        checkpoint_interval,
        if checkpoint_interval == 0 {
            " (mid-run checkpointing disabled)"
        } else {
            ""
        }
    );
    tracing::info!("  rollout_steps    = {rollout_steps}");
    tracing::info!("  num_agents       = {NUM_AGENTS}");
    tracing::info!("  hidden_dim       = {HIDDEN_DIM}");
    tracing::info!(
        "  max_payoff_evals = {}",
        match max_payoff_evals_per_iteration {
            Some(c) => format!("{c}/iter (boundary subsampling on)"),
            None => "unbounded (full boundary; bit-identical)".to_string(),
        }
    );
    tracing::info!(
        "  br_reward_scale  = {br_reward_scale} {}",
        if br_reward_scale == 1.0 {
            "(no-op)"
        } else {
            "(BR rewards rescaled)"
        }
    );
    tracing::info!(
        "  α-rank span-norm = {}",
        if alpha_rank_normalize_span {
            "on (Moran delta / payoff-span)"
        } else {
            "off (bit-identical to pre-#215)"
        }
    );

    let device: burn::tensor::Device<InnerBackend> = Default::default();

    // Probe the env to get obs_dim.
    let probe = make_cell_env(beta, kappa, cost, Some(base_seed()));
    let obs_dim = probe.obs_dim();
    drop(probe);
    tracing::info!("  obs_dim          = {obs_dim}");
    tracing::info!("  action_dims      = [{NUM_HOUSES}, 2, 2]");

    tracing::info!("  vf_coef          = {vf_coef} (issue #239 value-loss weight)");
    tracing::info!("  br_train_steps   = {br_train_steps} (issue #239 BR updates/iter)");
    match max_minibatches_per_epoch {
        Some(cap) => tracing::info!(
            "  br_max_mb/epoch  = {cap} (issue #251 throughput lever: capped minibatch subsample)"
        ),
        None => {
            tracing::info!("  br_max_mb/epoch  = (unset — full #239 all-minibatch coverage)")
        }
    }
    let psro_config = PsroConfig {
        max_iterations: total_iterations,
        max_population_size: 50,
        br_train_steps_per_iteration: br_train_steps,
        payoff_eval_episodes: 1,
        max_payoff_evals_per_iteration,
        br_reward_scale,
        seed: base_seed(),
    };
    let joint_config = JointTrainerConfig {
        num_agents: NUM_AGENTS,
        rollout_steps,
        n_epochs: 4,
        minibatch_size: 256,
        vf_coef,
        // Issue #239: consume the full rollout instead of one 256-sample draw.
        iterate_all_minibatches: true,
        // Issue #251: optional cap on minibatch steps/epoch (None => full).
        max_minibatches_per_epoch,
        ..Default::default()
    };
    let meta_solver: Box<dyn MetaSolver> = Box::new(
        AlphaRankMetaSolver::default().with_payoff_span_normalization(alpha_rank_normalize_span),
    );

    let policy_factory = move |dev: &burn::tensor::Device<InnerBackend>, seed: u64| {
        MultiDiscreteMlpBurnPolicy::<B>::new_seeded(
            obs_dim,
            vec![NUM_HOUSES, 2, 2],
            HIDDEN_DIM,
            seed,
            dev,
        )
    };
    let optimizer_factory = || {
        let inner = AdamConfig::new().init();
        BurnOptimizer::new(inner, 3e-4)
    };
    let env_factory = move || make_cell_env(beta, kappa, cost, Some(base_seed()));

    let mut trainer = PsroTrainer::<
        B,
        MultiDiscreteMlpBurnPolicy<B>,
        burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MultiDiscreteMlpBurnPolicy<B>, B>,
        BucketBrigadeMaEnv,
        _,
        _,
        _,
    >::new(
        psro_config,
        joint_config,
        meta_solver,
        device,
        policy_factory,
        optimizer_factory,
        env_factory,
    )?;

    let bin_recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let json_recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();

    tracing::info!("------------------------------------------------------------");
    let training_start = std::time::Instant::now();

    // Live per-iteration progress + mid-run checkpointing via the PSRO
    // `on_iteration` callback (NFSP-parity, issue #202 + checkpointing,
    // issue #204). The callback fires once per outer iteration with that
    // iteration's `PsroIterationStats` AND the newest BR policy per
    // agent (`brs[i]`), so progress is observable and partial results are
    // persistable *during* multi-hour runs rather than only after
    // `run()` returns.
    //
    // Every `checkpoint_interval` iterations we write
    // `psro_<cell>_iter<n>_agent<i>.bin` (one per agent) plus a
    // `psro_<cell>_iter<n>_meta_nash.json` snapshot. A killed run thus
    // leaves a resumable / partial-result-safe artifact. Writing is a
    // pure side effect on a *clone* of each policy and does not perturb
    // the deterministic training trajectory. If any checkpoint write
    // fails we log a warning and continue — a transient IO error must not
    // abort an expensive multi-hour run.
    let stats = trainer.run(|iter_stats, brs| {
        tracing::info!(
            "iter {:>3}/{}  pop_size={:>3}  exploitability={:>9.4}",
            iter_stats.iteration,
            total_iterations,
            iter_stats.population_size,
            iter_stats.exploitability,
        );

        if checkpoint_interval == 0 || iter_stats.iteration % checkpoint_interval != 0 {
            return;
        }
        let n = iter_stats.iteration;
        for (i, br) in brs.iter().enumerate() {
            let bin_path = format!("psro_{cell}_iter{n}_agent{i}");
            if let Err(e) = (*br).clone().save_file(&bin_path, &bin_recorder) {
                tracing::warn!("checkpoint BIN save failed (iter {n}, agent {i}): {e}");
            }
        }
        // Meta-Nash snapshot (per-agent marginals) alongside the BRs.
        let mut as_map: HashMap<String, Vec<f32>> = HashMap::new();
        for (i, dist) in iter_stats.meta_nash_per_agent.iter().enumerate() {
            as_map.insert(format!("agent{i}"), dist.clone());
        }
        match serde_json::to_string_pretty(&as_map) {
            Ok(json) => {
                let meta_path = format!("psro_{cell}_iter{n}_meta_nash.json");
                if let Err(e) = std::fs::write(&meta_path, json) {
                    tracing::warn!("checkpoint meta-Nash write failed (iter {n}): {e}");
                }
            }
            Err(e) => tracing::warn!("checkpoint meta-Nash serialize failed (iter {n}): {e}"),
        }
        tracing::info!(
            "  checkpoint written: psro_{cell}_iter{n}_agent{{0..{}}}.bin (+ meta_nash.json)",
            NUM_AGENTS - 1
        );
    })?;

    tracing::info!("------------------------------------------------------------");
    let final_population_size = stats.iterations.last().map(|s| s.population_size).unwrap_or(0);
    let final_expl = stats.iterations.last().map(|s| s.exploitability).unwrap_or(f32::NAN);
    tracing::info!(
        "Training complete.  iterations={}  final_population_size={}  final_exploitability={:.4}  \
         time={:.1}s",
        stats.iterations.len(),
        final_population_size,
        final_expl,
        training_start.elapsed().as_secs_f64()
    );

    // --- Final per-agent BR save ------------------------------------
    let final_brs: Vec<MultiDiscreteMlpBurnPolicy<B>> = (0..NUM_AGENTS)
        .map(|i| {
            let pop = trainer.populations(i);
            pop.last().expect("population must be non-empty post-run").clone()
        })
        .collect();

    for (i, br) in final_brs.iter().enumerate() {
        let bin_path = format!("psro_{cell}_final_agent{i}");
        br.clone()
            .save_file(&bin_path, &bin_recorder)
            .map_err(|e| anyhow::anyhow!("final BIN save failed: {e}"))?;
        let json_path = format!("psro_{cell}_final_agent{i}");
        br.clone()
            .save_file(&json_path, &json_recorder)
            .map_err(|e| anyhow::anyhow!("final JSON save failed: {e}"))?;
    }
    tracing::info!(
        "Saved final per-agent BR policies: psro_{cell}_final_agent{{0..{}}}.bin (+ .json)",
        NUM_AGENTS - 1
    );

    // --- Final meta-Nash distribution save --------------------------
    if let Some(final_iter) = stats.iterations.last() {
        let meta_nash_path = format!("psro_{cell}_final_meta_nash.json");
        let mut as_map: HashMap<String, Vec<f32>> = HashMap::new();
        for (i, dist) in final_iter.meta_nash_per_agent.iter().enumerate() {
            as_map.insert(format!("agent{i}"), dist.clone());
        }
        let json = serde_json::to_string_pretty(&as_map)
            .map_err(|e| anyhow::anyhow!("meta-Nash JSON serialize failed: {e}"))?;
        std::fs::write(&meta_nash_path, json)
            .map_err(|e| anyhow::anyhow!("meta-Nash JSON write failed: {e}"))?;
        tracing::info!("Saved final meta-Nash: {meta_nash_path}");
    }

    // --- Final convergence eval -------------------------------------
    let final_per_step_team =
        eval_per_step_team_reward(&final_brs, &device, obs_dim, beta, kappa, cost, 0xEE1 ^ 0xFFFF);
    let final_gc_cell = gap_closed_cell(final_per_step_team, cell_enum(&cell));
    let final_gc_base = gap_closed(final_per_step_team);
    tracing::info!(
        "Final eval (deterministic, K={EVAL_STEPS}): per_step_team = {final_per_step_team:.4}, \
         gap_closed_cell = {final_gc_cell:.4} (cell {}; PPO workshop paper = -0.049)",
        cell_enum(&cell).tag()
    );
    tracing::info!(
        "  (secondary diagnostic) base-scenario gap_closed = {final_gc_base:.4} — not the \
         convergence metric for a cell run"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use thrust_rl::multi_agent::bucket_brigade_metrics::{
        MINSPEC_RANDOM_BETA01, MINSPEC_RANDOM_BETA05, MINSPEC_RANDOM_BETA09,
        MINSPEC_SPECIALIST_BETA01, MINSPEC_SPECIALIST_BETA05, MINSPEC_SPECIALIST_BETA09,
    };

    use super::*;

    /// Each `CELL` env-var string must select its matching
    /// [`BucketBrigadeCell`], which is what `gap_closed_cell` keys off to
    /// pick the cell-specific baselines.
    #[test]
    fn cell_enum_selects_correct_cell() {
        assert_eq!(cell_enum("beta01"), BucketBrigadeCell::Beta01);
        assert_eq!(cell_enum("beta05"), BucketBrigadeCell::Beta05);
        assert_eq!(cell_enum("beta09"), BucketBrigadeCell::Beta09);
    }

    /// `cell_params` and `cell_enum` agree: the (β, κ, c) triple the env is
    /// built from is the same one the selected cell encodes.
    #[test]
    fn cell_params_match_cell_enum_parameters() {
        for cell in ["beta01", "beta05", "beta09"] {
            assert_eq!(cell_params(cell), cell_enum(cell).parameters());
        }
    }

    /// Final-eval normalization uses the cell-specific baselines: feeding a
    /// cell's specialist baseline through `gap_closed_cell(_, cell_enum(CELL))`
    /// must return ~1.0 (and its random baseline ~0.0) for the active cell.
    #[test]
    fn gap_closed_cell_uses_active_cell_baselines() {
        for (cell, random, specialist) in [
            ("beta01", MINSPEC_RANDOM_BETA01, MINSPEC_SPECIALIST_BETA01),
            ("beta05", MINSPEC_RANDOM_BETA05, MINSPEC_SPECIALIST_BETA05),
            ("beta09", MINSPEC_RANDOM_BETA09, MINSPEC_SPECIALIST_BETA09),
        ] {
            let active = cell_enum(cell);
            assert!(
                gap_closed_cell(random, active).abs() < 1e-5,
                "cell {cell} random endpoint should map to 0.0"
            );
            assert!(
                (gap_closed_cell(specialist, active) - 1.0).abs() < 1e-5,
                "cell {cell} specialist endpoint should map to 1.0"
            );
        }
    }
}
