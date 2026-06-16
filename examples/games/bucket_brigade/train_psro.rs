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
//! Intermediate checkpoints every `CHECKPOINT_INTERVAL_ITERATIONS`
//! outer iterations land at:
//! - `psro_<cell>_iter<n>_agent<i>.bin`
//!
//! # `gap_closed` baseline caveat
//!
//! The `MINSPEC_RANDOM = -96.07` and `MINSPEC_SPECIALIST = -22.07`
//! baselines used by `gap_closed` are computed on the *base*
//! `minimal_specialization` scenario, NOT the per-cell payoff scale of
//! the no-convergence regime. Per-cell baselines are tracked in #128.
//! Until that lands, the printed `gap_closed` values are informational
//! diagnostics, not a hard convergence bar.

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
        bucket_brigade_metrics::gap_closed,
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
const SEED: u64 = 42;
const DEFAULT_TOTAL_ITERATIONS: usize = 50;
const DEFAULT_ROLLOUT_STEPS: usize = 2048;
const DEFAULT_CELL: &str = "beta05";
/// Outer iterations between intermediate per-agent BR checkpoints.
const CHECKPOINT_INTERVAL_ITERATIONS: usize = 10;
/// Length of the post-iteration deterministic eval rollout used to log
/// the running per-step team estimate.
const EVAL_STEPS: usize = 50;

fn cell_params(cell: &str) -> (f32, f32, f32) {
    match cell {
        "beta01" => (0.1, 0.1, 0.5),
        "beta05" => (0.5, 0.1, 0.5),
        "beta09" => (0.9, 0.1, 0.5),
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
    let mut env = make_cell_env(beta, kappa, cost, Some(SEED ^ seed_xor));
    let mut last_obs = env.reset_joint(Some(SEED ^ seed_xor));
    let mut total_team_reward: f32 = 0.0;
    let mut steps: usize = 0;
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED ^ seed_xor.wrapping_add(1));
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

    tracing::info!("Starting PSRO bucket-brigade training (Burn backend: {BACKEND_LABEL})");
    tracing::info!("  cell             = {cell} (β={beta}, κ={kappa}, c={cost})");
    tracing::info!("  total_iterations = {total_iterations}");
    tracing::info!("  rollout_steps    = {rollout_steps}");
    tracing::info!("  num_agents       = {NUM_AGENTS}");
    tracing::info!("  hidden_dim       = {HIDDEN_DIM}");

    let device: burn::tensor::Device<InnerBackend> = Default::default();

    // Probe the env to get obs_dim.
    let probe = make_cell_env(beta, kappa, cost, Some(SEED));
    let obs_dim = probe.obs_dim();
    drop(probe);
    tracing::info!("  obs_dim          = {obs_dim}");
    tracing::info!("  action_dims      = [{NUM_HOUSES}, 2, 2]");

    let psro_config = PsroConfig {
        max_iterations: total_iterations,
        max_population_size: 50,
        br_train_steps_per_iteration: 1,
        payoff_eval_episodes: 1,
        seed: SEED,
    };
    let joint_config = JointTrainerConfig {
        num_agents: NUM_AGENTS,
        rollout_steps,
        n_epochs: 4,
        minibatch_size: 256,
        ..Default::default()
    };
    let meta_solver: Box<dyn MetaSolver> = Box::new(AlphaRankMetaSolver::default());

    let policy_factory = move |dev: &burn::tensor::Device<InnerBackend>| {
        MultiDiscreteMlpBurnPolicy::<B>::new(obs_dim, vec![NUM_HOUSES, 2, 2], HIDDEN_DIM, dev)
    };
    let optimizer_factory = || {
        let inner = AdamConfig::new().init();
        BurnOptimizer::new(inner, 3e-4)
    };
    let env_factory = move || make_cell_env(beta, kappa, cost, Some(SEED));

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
        device.clone(),
        policy_factory,
        optimizer_factory,
        env_factory,
    )?;

    tracing::info!("------------------------------------------------------------");
    let training_start = std::time::Instant::now();

    // PSRO's `run()` is opaque — we cannot easily inject a per-iter
    // callback the way NFSP supports. Run the full outer loop, then
    // walk the stats history to log per-iteration progress and write
    // intermediate checkpoints from the final populations. (The
    // populations are monotonically-growing, so `populations[i][k]` is
    // the BR trained on PSRO iteration `k+1`.)
    let stats = trainer.run()?;

    // --- Per-iteration progress + intermediate checkpoints ----------
    let bin_recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let json_recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
    for (iter_idx, iter_stats) in stats.iterations.iter().enumerate() {
        let iter = iter_idx + 1;
        // Each agent's `populations[i][iter_idx]` is the freshly-added
        // BR from this outer iteration (the initial random policy is
        // index 0 in the slice but iteration 1 grew the population by
        // 1, so iter_idx==0 corresponds to populations[i][1]). We
        // sample the most-recent BR per agent for the eval rollout.
        let snapshot_brs: Vec<MultiDiscreteMlpBurnPolicy<B>> = (0..NUM_AGENTS)
            .map(|i| {
                let pop = trainer.populations(i);
                // The newly-added BR is the second-to-last entry on
                // intermediate iterations; the *very* last entry was
                // added on the final iteration. For per-iter logging
                // we read the BR that corresponds to this iteration's
                // index, capped at population length.
                let target_idx = (iter_idx + 1).min(pop.len() - 1);
                pop[target_idx].clone()
            })
            .collect();
        let recent_eval = eval_per_step_team_reward(
            &snapshot_brs,
            &device,
            obs_dim,
            beta,
            kappa,
            cost,
            0xEE1 ^ iter as u64,
        );
        let recent_gc = gap_closed(recent_eval);
        tracing::info!(
            "iter {:>3}/{}  pop_size={:>3}  exploitability={:>9.4}  per_step_team≈{:>8.3}  \
             gap_closed≈{:>7.4}",
            iter,
            stats.iterations.len(),
            iter_stats.population_size,
            iter_stats.exploitability,
            recent_eval,
            recent_gc
        );

        if iter % CHECKPOINT_INTERVAL_ITERATIONS == 0 && iter < stats.iterations.len() {
            for (i, br) in snapshot_brs.iter().enumerate() {
                let path = format!("psro_{cell}_iter{iter}_agent{i}");
                br.clone()
                    .save_file(&path, &bin_recorder)
                    .map_err(|e| anyhow::anyhow!("intermediate checkpoint save failed: {e}"))?;
            }
            tracing::info!(
                "  intermediate checkpoint written: psro_{cell}_iter{iter}_agent{{0..{}}}.bin",
                NUM_AGENTS - 1
            );
        }
    }

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
    let final_gc = gap_closed(final_per_step_team);
    tracing::info!(
        "Final eval (deterministic, K={EVAL_STEPS}): per_step_team = {final_per_step_team:.4}, \
         gap_closed = {final_gc:.4} (PPO workshop paper = -0.049)"
    );
    tracing::info!(
        "Caveat: `gap_closed` baselines are base-scenario, not cell-specific (#128). Treat as \
         diagnostic."
    );

    Ok(())
}
