//! NFSP bucket-brigade trainer on the no-convergence cells.
//!
//! Long-form training driver wired against `NfspTrainer` +
//! `MultiDiscreteMlpBurnPolicy` for the workshop-paper
//! no-convergence regime. Default cell is canonical
//! `(β=0.5, κ=0.1, c=0.5)`; override via `CELL=beta01|beta05|beta09`.
//!
//! Backend selection: NdArray (CPU) by default; opt into Burn's wgpu
//! GPU backend with `--features "training,env-bucket-brigade,wgpu"`.
//!
//! Sibling of `examples/games/bucket_brigade/train_psro.rs`. PR 4/4
//! of issue #117's bucket-brigade integration chain (closes #115).
//!
//! # Factored multi-discrete policy (post-#127)
//!
//! Per issue #127, NFSP's per-agent reservoir now stores
//! `(Vec<f32>, Vec<i64>)` with one action entry per factored dim, and
//! the supervised AP-update step builds an `[mb, num_action_dims]` int
//! tensor before calling `policy.evaluate_actions_joint`. That lets
//! this example drive bucket-brigade's factored
//! `[house, mode, signal]` action space natively, matching the shape
//! `train_psro.rs` already uses. Pre-#127 this example went through a
//! Cartesian-product `Discrete(40)` wrapper (`= NUM_HOUSES * 2 * 2`) +
//! `MlpBurnPolicy`; that workaround is removed.
//!
//! # Usage
//!
//! ```bash
//! # Default cell (canonical β=0.5):
//! cargo run --release --example train_nfsp \
//!     --features "training,env-bucket-brigade"
//!
//! # Override the cell or training budget:
//! TOTAL_ITERATIONS=20 ROLLOUT_STEPS=4096 CELL=beta01 \
//!     cargo run --release --example train_nfsp \
//!         --features "training,env-bucket-brigade"
//!
//! # GPU (wgpu) backend:
//! cargo run --release --example train_nfsp \
//!     --features "training,env-bucket-brigade,wgpu"
//! ```
//!
//! # Artifacts
//!
//! Per-agent final BR and AP policies are saved to:
//! - `nfsp_<cell>_final_br_agent<i>.bin` (best-response, Burn binary)
//! - `nfsp_<cell>_final_br_agent<i>.json` (PrettyJSON, inspectable)
//! - `nfsp_<cell>_final_ap_agent<i>.bin` (average policy — the paper's
//!   recommended deploy artifact)
//! - `nfsp_<cell>_final_ap_agent<i>.json`
//!
//! # `gap_closed` normalization
//!
//! Same as `train_psro.rs`: the final-eval headline metric is
//! `gap_closed_cell`, which normalizes the per-step team payoff against the
//! **cell-specific** random/specialist baselines for the active `CELL`
//! (`MINSPEC_{RANDOM,SPECIALIST}_BETA0XX` from #128/#131). The base-scenario
//! `gap_closed` is logged only as a secondary diagnostic and is NOT the
//! convergence metric for a cell run.

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
        JointTrainerConfig, NfspConfig, NfspTrainer,
        bucket_brigade_baselines::BucketBrigadeCell,
        bucket_brigade_metrics::{gap_closed, gap_closed_cell},
        joint::JointEnv,
    },
    policy::multi_discrete_mlp::MultiDiscreteMlpBurnPolicy,
    train::optimizer::BurnOptimizer,
};

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
/// Length of the post-iteration deterministic eval rollout used to log
/// running per-step team estimates.
const EVAL_STEPS: usize = 200;

/// Per-agent factored action cardinalities `[house, mode, signal]`.
fn action_dims() -> Vec<usize> {
    vec![NUM_HOUSES, 2, 2]
}

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

/// Deterministic eval rollout on a fresh env using the supplied
/// policies (one per agent). Returns mean per-step team reward over
/// `EVAL_STEPS` steps.
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

    tracing::info!("Starting NFSP bucket-brigade training (Burn backend: {BACKEND_LABEL})");
    tracing::info!("  cell             = {cell} (β={beta}, κ={kappa}, c={cost})");
    tracing::info!("  total_iterations = {total_iterations}");
    tracing::info!("  rollout_steps    = {rollout_steps}");
    tracing::info!("  num_agents       = {NUM_AGENTS}");
    tracing::info!("  hidden_dim       = {HIDDEN_DIM}");
    tracing::info!(
        "  action_dims      = {:?} (factored multi-discrete, native shape)",
        action_dims()
    );

    let device: burn::tensor::Device<InnerBackend> = Default::default();

    let probe = make_cell_env(beta, kappa, cost, Some(SEED));
    let obs_dim = probe.obs_dim();
    drop(probe);
    tracing::info!("  obs_dim          = {obs_dim}");

    // Issue #199: the original run gave the AP only
    // `avg_policy_train_steps_per_iteration × avg_policy_minibatch_size`
    // ≈ 512 samples/iteration against a reservoir that grows to ~9,800
    // entries/agent, which pinned `avg_ap_loss` at the uniform-entropy
    // floor `ln(40)` for the whole run. Two knobs address that:
    //   * `avg_policy_min_reservoir_coverage` runs enough supervised steps to cover
    //     the reservoir `coverage`× per iteration (overridable via `AP_COVERAGE`),
    //     and
    //   * `br_reward_scale` rescales the `[−700, 0]` payoff band into a numerically
    //     friendlier range for the BR critic (overridable via `BR_REWARD_SCALE`).
    let ap_coverage: f32 =
        std::env::var("AP_COVERAGE").ok().and_then(|s| s.parse().ok()).unwrap_or(2.0);
    let br_reward_scale: f32 = std::env::var("BR_REWARD_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.01);
    tracing::info!("  ap_coverage      = {ap_coverage} (issue #199 adaptive AP-step floor)");
    tracing::info!("  br_reward_scale  = {br_reward_scale} (issue #199 payoff rescale)");
    let nfsp_config = NfspConfig {
        max_iterations: total_iterations,
        anticipatory_param: 0.1,
        reservoir_capacity: 16_384,
        br_train_steps_per_iteration: 1,
        avg_policy_train_steps_per_iteration: 8,
        avg_policy_minibatch_size: 64,
        avg_policy_lr: 5e-3,
        avg_policy_min_reservoir_coverage: ap_coverage,
        br_reward_scale,
        seed: SEED,
    };
    let joint_config = JointTrainerConfig {
        num_agents: NUM_AGENTS,
        rollout_steps,
        n_epochs: 4,
        minibatch_size: 256,
        ..Default::default()
    };

    let policy_factory = move |dev: &burn::tensor::Device<InnerBackend>, seed: u64| {
        MultiDiscreteMlpBurnPolicy::<B>::new_seeded(obs_dim, action_dims(), HIDDEN_DIM, seed, dev)
    };
    let optimizer_factory = || {
        let inner = AdamConfig::new().init();
        BurnOptimizer::new(inner, 3e-4)
    };
    let env_factory = move || make_cell_env(beta, kappa, cost, Some(SEED));

    let mut trainer = NfspTrainer::<
        B,
        MultiDiscreteMlpBurnPolicy<B>,
        burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MultiDiscreteMlpBurnPolicy<B>, B>,
        BucketBrigadeMaEnv,
        _,
        _,
        _,
    >::new(
        nfsp_config,
        joint_config,
        device,
        policy_factory,
        optimizer_factory,
        env_factory,
    )?;

    tracing::info!("------------------------------------------------------------");
    let training_start = std::time::Instant::now();

    // NFSP supports a per-iteration callback via `run`; use it to log
    // progress without needing a post-hoc walk like train_psro.rs does.
    let stats = trainer.run(|iter_stats| {
        // Cheap signal — full eval would re-snapshot policies each
        // iter, which we save for the final report. We log reservoir
        // sizes + AP loss per iter.
        let res_sizes = &iter_stats.reservoir_sizes;
        let ap_loss_avg: f64 = iter_stats
            .avg_policy_loss
            .iter()
            .filter_map(|opt| opt.as_ref())
            .copied()
            .sum::<f64>()
            / iter_stats.avg_policy_loss.iter().filter(|opt| opt.is_some()).count().max(1) as f64;
        // Uniform-entropy floor over the [10,2,2]=40 joint action space.
        // The whole #134 finding was that `avg_ap_loss` never moves off
        // this floor; log the signed delta so the learning curve is
        // legible at a glance (negative = AP has learned structure).
        let uniform_floor = (40.0_f64).ln();
        let ap_delta = ap_loss_avg - uniform_floor;
        // BR-side diagnostic (issue #199): a weak best-response starves
        // the AP target. Surface the BR policy/value loss + entropy each
        // iteration so it is clear whether the RL side is learning.
        let (br_pol, br_val, br_ent) = match &iter_stats.br_stats {
            Some(s) => {
                let n = s.policy_loss.len().max(1) as f64;
                (
                    s.policy_loss.iter().sum::<f64>() / n,
                    s.value_loss.iter().sum::<f64>() / n,
                    s.entropy.iter().sum::<f64>() / n,
                )
            }
            None => (f64::NAN, f64::NAN, f64::NAN),
        };
        tracing::info!(
            "iter {:>3}/{}  reservoir_sizes={:?}  avg_ap_loss={:.4} (Δfloor={:+.4})  \
             br[pol={:.4} val={:.4} ent={:.4}]  cum_br_pushes={}",
            iter_stats.iteration,
            total_iterations,
            res_sizes,
            ap_loss_avg,
            ap_delta,
            br_pol,
            br_val,
            br_ent,
            iter_stats.cumulative_br_pushes
        );
    })?;

    tracing::info!("------------------------------------------------------------");
    tracing::info!(
        "Training complete.  iterations={}  cumulative_br_pushes={}  time={:.1}s",
        stats.iterations.len(),
        trainer.cumulative_br_pushes(),
        training_start.elapsed().as_secs_f64()
    );

    // --- Final per-agent BR + AP save -------------------------------
    let bin_recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let json_recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();

    let final_brs: Vec<MultiDiscreteMlpBurnPolicy<B>> =
        (0..NUM_AGENTS).map(|i| trainer.br_policy(i).clone()).collect();
    let final_aps: Vec<MultiDiscreteMlpBurnPolicy<B>> =
        (0..NUM_AGENTS).map(|i| trainer.avg_policy(i).clone()).collect();

    for (i, br) in final_brs.iter().enumerate() {
        let stem = format!("nfsp_{cell}_final_br_agent{i}");
        br.clone()
            .save_file(&stem, &bin_recorder)
            .map_err(|e| anyhow::anyhow!("final BR BIN save failed: {e}"))?;
        br.clone()
            .save_file(&stem, &json_recorder)
            .map_err(|e| anyhow::anyhow!("final BR JSON save failed: {e}"))?;
    }
    for (i, ap) in final_aps.iter().enumerate() {
        let stem = format!("nfsp_{cell}_final_ap_agent{i}");
        ap.clone()
            .save_file(&stem, &bin_recorder)
            .map_err(|e| anyhow::anyhow!("final AP BIN save failed: {e}"))?;
        ap.clone()
            .save_file(&stem, &json_recorder)
            .map_err(|e| anyhow::anyhow!("final AP JSON save failed: {e}"))?;
    }
    tracing::info!(
        "Saved final per-agent policies: nfsp_{cell}_final_{{br,ap}}_agent{{0..{}}}.bin (+ .json)",
        NUM_AGENTS - 1
    );

    // --- Final convergence eval (use AP — paper's recommended deploy artifact)
    let final_ap_per_step_team =
        eval_per_step_team_reward(&final_aps, &device, obs_dim, beta, kappa, cost, 0xEE1 ^ 0xFFFF);
    let active_cell = cell_enum(&cell);
    let final_ap_gc_cell = gap_closed_cell(final_ap_per_step_team, active_cell);
    let final_ap_gc_base = gap_closed(final_ap_per_step_team);
    let final_br_per_step_team =
        eval_per_step_team_reward(&final_brs, &device, obs_dim, beta, kappa, cost, 0xEE1 ^ 0xFEFE);
    let final_br_gc_cell = gap_closed_cell(final_br_per_step_team, active_cell);
    let final_br_gc_base = gap_closed(final_br_per_step_team);
    tracing::info!("Final eval (deterministic, K={EVAL_STEPS}, cell {}):", active_cell.tag());
    tracing::info!(
        "  AP (avg policy, paper-deploy): per_step_team = {final_ap_per_step_team:.4}, \
         gap_closed_cell = {final_ap_gc_cell:.4}"
    );
    tracing::info!(
        "  BR (best response):            per_step_team = {final_br_per_step_team:.4}, \
         gap_closed_cell = {final_br_gc_cell:.4}"
    );
    tracing::info!("  Reference: PPO workshop paper = -0.049 gap_closed");
    tracing::info!(
        "  (secondary diagnostic) base-scenario gap_closed: AP = {final_ap_gc_base:.4}, BR = \
         {final_br_gc_base:.4} — not the convergence metric for a cell run"
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
