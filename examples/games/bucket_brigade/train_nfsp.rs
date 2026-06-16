//! NFSP bucket-brigade trainer on the no-convergence cells.
//!
//! Long-form training driver wired against `NfspTrainer` +
//! `MlpBurnPolicy` (single-discrete) for the workshop-paper
//! no-convergence regime. Default cell is canonical
//! `(β=0.5, κ=0.1, c=0.5)`; override via `CELL=beta01|beta05|beta09`.
//!
//! Backend selection: NdArray (CPU) by default; opt into Burn's wgpu
//! GPU backend with `--features "training,env-bucket-brigade,wgpu"`.
//!
//! Sibling of `examples/games/bucket_brigade/train_psro.rs`. PR 4/4
//! of issue #117's bucket-brigade integration chain (closes #115).
//!
//! # Why a Cartesian-product single-discrete adapter
//!
//! `MultiDiscreteMlpBurnPolicy` is the natural fit for bucket-brigade's
//! factored `[house, mode, signal]` action space — and PSRO uses that
//! shape directly (see `train_psro.rs`). However, the post-#106 NFSP
//! trainer's per-agent reservoir stores `(Vec<f32>, i64)` — a single
//! scalar action — and its supervised AP-update path reshapes that
//! scalar as a `[mb, 1]` int tensor before calling
//! `policy.evaluate_actions_joint`. For a multi-discrete policy with
//! `action_dims = [10, 2, 2]`, that shape causes a Burn `Squeeze` panic
//! at the second per-dim slice.
//!
//! Until #127 (multi-discrete reservoirs) lands, this example
//! Cartesian-product-flattens the action space into `Discrete(40)`
//! (`= NUM_HOUSES * 2 * 2`) via the `SingleDiscreteBucketBrigade`
//! wrapper — exactly the same shape `tests/test_nfsp_bucket_brigade.rs`
//! (PR #126) uses. Once #127 ships, this example should switch back to
//! the factored policy, matching the `train_psro.rs` shape.
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
//! # `gap_closed` baseline caveat
//!
//! Same as `train_psro.rs`: the `MINSPEC_RANDOM = -96.07` /
//! `MINSPEC_SPECIALIST = -22.07` baselines are computed on the *base*
//! `minimal_specialization` scenario, NOT the per-cell payoff scale of
//! the no-convergence regime (tracked in #128).

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
        JointEnv, JointStepResult, JointTrainerConfig, NfspConfig, NfspTrainer,
        bucket_brigade_metrics::gap_closed,
    },
    policy::mlp::MlpBurnPolicy,
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
/// Cartesian-product cardinality `NUM_HOUSES * 2 (mode) * 2 (signal)`.
/// Wrapping into a single-discrete dim lets us use `MlpBurnPolicy`
/// instead of `MultiDiscreteMlpBurnPolicy`, sidestepping the NFSP
/// multi-discrete reservoir gap (#127).
const FLAT_ACTION_DIM: usize = NUM_HOUSES * 2 * 2;
/// Length of the post-iteration deterministic eval rollout used to log
/// running per-step team estimates.
const EVAL_STEPS: usize = 200;

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

/// Cartesian-product single-discrete adapter over [`BucketBrigadeMaEnv`].
///
/// Each per-agent action is a single scalar in `0..FLAT_ACTION_DIM`
/// (`= NUM_HOUSES * 2 * 2 = 40`); the adapter decodes it as
/// `house = a / 4`, `mode = (a / 2) % 2`, `signal = a % 2` and forwards
/// the factored `[house, mode, signal]` triple to
/// [`BucketBrigadeMaEnv::step_joint`]. Mirrors the wrapper inlined in
/// `tests/test_nfsp_bucket_brigade.rs` (PR #126); kept inline here per
/// the Curator's "examples self-contained at slight cost of duplication"
/// guidance.
struct SingleDiscreteBucketBrigade {
    inner: BucketBrigadeMaEnv,
}

impl SingleDiscreteBucketBrigade {
    fn new(inner: BucketBrigadeMaEnv) -> Self {
        Self { inner }
    }
}

impl JointEnv for SingleDiscreteBucketBrigade {
    fn reset_joint(&mut self, seed: Option<u64>) -> Vec<Vec<f32>> {
        self.inner.reset_joint(seed)
    }

    fn step_joint(&mut self, actions: &[Vec<i64>]) -> JointStepResult {
        let factored: Vec<Vec<i64>> = actions
            .iter()
            .map(|a| {
                assert_eq!(
                    a.len(),
                    1,
                    "single-discrete adapter expects length-1 actions, got {}",
                    a.len()
                );
                let v = a[0];
                let signal = v % 2;
                let mode = (v / 2) % 2;
                let house = (v / 4) % NUM_HOUSES as i64;
                vec![house, mode, signal]
            })
            .collect();
        self.inner.step_joint(&factored)
    }
}

/// Deterministic eval rollout on a fresh env using the supplied
/// policies (one per agent). Returns mean per-step team reward over
/// `EVAL_STEPS` steps.
fn eval_per_step_team_reward(
    policies: &[MlpBurnPolicy<B>],
    device: &burn::tensor::Device<InnerBackend>,
    obs_dim: usize,
    beta: f32,
    kappa: f32,
    cost: f32,
    seed_xor: u64,
) -> f32 {
    use rand::SeedableRng;
    let mut env =
        SingleDiscreteBucketBrigade::new(make_cell_env(beta, kappa, cost, Some(SEED ^ seed_xor)));
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
    tracing::info!("  flat_action_dim  = {FLAT_ACTION_DIM} (Cartesian-product wrapper; #127)");

    let device: burn::tensor::Device<InnerBackend> = Default::default();

    let probe = make_cell_env(beta, kappa, cost, Some(SEED));
    let obs_dim = probe.obs_dim();
    drop(probe);
    tracing::info!("  obs_dim          = {obs_dim}");

    let nfsp_config = NfspConfig {
        max_iterations: total_iterations,
        anticipatory_param: 0.1,
        reservoir_capacity: 16_384,
        br_train_steps_per_iteration: 1,
        avg_policy_train_steps_per_iteration: 8,
        avg_policy_minibatch_size: 64,
        avg_policy_lr: 5e-3,
        seed: SEED,
    };
    let joint_config = JointTrainerConfig {
        num_agents: NUM_AGENTS,
        rollout_steps,
        n_epochs: 4,
        minibatch_size: 256,
        ..Default::default()
    };

    let policy_factory = move |dev: &burn::tensor::Device<InnerBackend>| {
        MlpBurnPolicy::<B>::new(obs_dim, FLAT_ACTION_DIM, HIDDEN_DIM, dev)
    };
    let optimizer_factory = || {
        let inner = AdamConfig::new().init();
        BurnOptimizer::new(inner, 3e-4)
    };
    let env_factory =
        move || SingleDiscreteBucketBrigade::new(make_cell_env(beta, kappa, cost, Some(SEED)));

    let mut trainer = NfspTrainer::<
        B,
        MlpBurnPolicy<B>,
        burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MlpBurnPolicy<B>, B>,
        SingleDiscreteBucketBrigade,
        _,
        _,
        _,
    >::new(
        nfsp_config,
        joint_config,
        device.clone(),
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
        tracing::info!(
            "iter {:>3}/{}  reservoir_sizes={:?}  avg_ap_loss={:.4}  cum_br_pushes={}",
            iter_stats.iteration,
            total_iterations,
            res_sizes,
            ap_loss_avg,
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

    let final_brs: Vec<MlpBurnPolicy<B>> =
        (0..NUM_AGENTS).map(|i| trainer.br_policy(i).clone()).collect();
    let final_aps: Vec<MlpBurnPolicy<B>> =
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
    let final_ap_gc = gap_closed(final_ap_per_step_team);
    let final_br_per_step_team =
        eval_per_step_team_reward(&final_brs, &device, obs_dim, beta, kappa, cost, 0xEE1 ^ 0xFEFE);
    let final_br_gc = gap_closed(final_br_per_step_team);
    tracing::info!("Final eval (deterministic, K={EVAL_STEPS}):");
    tracing::info!(
        "  AP (avg policy, paper-deploy): per_step_team = {final_ap_per_step_team:.4}, \
         gap_closed = {final_ap_gc:.4}"
    );
    tracing::info!(
        "  BR (best response):            per_step_team = {final_br_per_step_team:.4}, \
         gap_closed = {final_br_gc:.4}"
    );
    tracing::info!("  Reference: PPO workshop paper = -0.049 gap_closed");
    tracing::info!(
        "Caveat: `gap_closed` baselines are base-scenario, not cell-specific (#128). Treat as \
         diagnostic."
    );

    Ok(())
}
