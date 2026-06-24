//! Standalone single-best-response (BR) A/B probe for bucket-brigade.
//!
//! Issue #241. Investigating why the PPO best-response does not learn on
//! bucket-brigade (#239) is slow when the only signal is the per-iteration
//! `br[...]` line from a full NFSP/PSRO run. This harness isolates ONE
//! best-response:
//!
//! - freezes `N−1` opponents (uniform-random policies) for one cell,
//! - trains a single PPO best-response agent for `ITERATIONS` iterations
//!   against those frozen opponents,
//! - logs per iteration: value loss, **explained variance**, policy entropy,
//!   and mean episode return.
//!
//! The explained variance (`ev = 1 − Var(returns − values) / Var(returns)`)
//! is computed inside the joint PPO update (`src/multi_agent/joint.rs`) and
//! surfaced here. `ev` near 0 or negative = critic not fitting; `ev → 1` =
//! critic fits. That single metric is the most informative signal for the
//! #239 "BR does not learn" investigation, and this harness lets you A/B a
//! single BR in ~1 minute instead of via a full NFSP/PSRO run.
//!
//! The "opponents" are freshly-initialized policies (uniform-ish random at
//! init) held frozen via the joint trainer's freeze-N−1 primitive
//! (`update_with_active_agents` with an active mask selecting only the BR
//! agent). They are NOT updated — only the single BR agent's optimizer steps.
//!
//! # Usage
//!
//! ```bash
//! # Default A/B probe (cell β=0.5, 20 iters, ndarray CPU backend):
//! cargo run --release --example train_br_probe \
//!     --features "training,env-bucket-brigade"
//!
//! # A/B the #240 knobs on one cell in ~1 min:
//! CELL=beta05 ITERATIONS=20 ROLLOUT_STEPS=2048 \
//!     VF_COEF=0.05 BR_TRAIN_STEPS=4 BR_REWARD_SCALE=0.01 \
//!     cargo run --release --example train_br_probe \
//!         --features "training,env-bucket-brigade"
//! ```
//!
//! # Env-var knobs
//!
//! - `CELL` — one of `beta01|beta05|beta09` (default `beta05`).
//! - `ITERATIONS` — number of BR rollout/update iterations (default 20).
//! - `ROLLOUT_STEPS` — env steps collected per iteration (default 2048).
//! - `VF_COEF` — value-loss weight in the joint loss (#240, default 0.5).
//! - `BR_TRAIN_STEPS` — PPO updates per iteration (#240, default 8).
//! - `BR_REWARD_SCALE` — affine reward rescale before the update (#240/#199,
//!   default 0.001). Does not change the optimal policy; keeps the `[−700, 0]`
//!   payoff band in a critic-friendly range.
//! - `CRITIC_LR` — optional dedicated critic learning rate (#239 fix #4). Unset
//!   (default) => single shared actor+critic optimizer. When set, the joint
//!   update splits actor/critic into two backward passes and steps the critic
//!   at this LR. In the #239 A/B this split did NOT improve critic fit over the
//!   single optimizer (it can hurt); kept gated/off as a knob.
//!
//! The defaults above are the empirically-winning #239 settings: with them the
//! critic explained-variance rises off ~0 (to ~0.3 by iter 16 on beta05) and
//! the policy entropy falls below the uniform 1.0 floor — the directional win
//! the #239 acceptance criteria ask for.
//!
//! All of these reuse the exact same plumbing the NFSP/PSRO trainers use;
//! this example introduces no new training knobs.

use anyhow::Result;
use burn::{backend::Autodiff, optim::AdamConfig};
use rand::SeedableRng;
use thrust_rl::{
    env::games::bucket_brigade::{BucketBrigadeMaEnv, NUM_HOUSES, registry},
    multi_agent::{
        JointTrainerConfig,
        bucket_brigade_baselines::BucketBrigadeCell,
        joint::{JointEnv, JointMultiAgentTrainer},
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
const DEFAULT_ITERATIONS: usize = 20;
const DEFAULT_ROLLOUT_STEPS: usize = 2048;
const DEFAULT_CELL: &str = "beta05";
/// The single agent we train a best-response for. The other
/// `NUM_AGENTS − 1` agents are frozen opponents.
const BR_AGENT: usize = 0;

/// Per-agent factored action cardinalities `[house, mode, signal]`.
fn action_dims() -> Vec<usize> {
    vec![NUM_HOUSES, 2, 2]
}

fn cell_params(cell: &str) -> (f32, f32, f32) {
    cell_enum(cell).parameters()
}

/// Map the `CELL` env-var string to the corresponding [`BucketBrigadeCell`].
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

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let iterations: usize = std::env::var("ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ITERATIONS);
    let rollout_steps: usize = std::env::var("ROLLOUT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ROLLOUT_STEPS);
    let cell = std::env::var("CELL").unwrap_or_else(|_| DEFAULT_CELL.to_string());
    let (beta, kappa, cost) = cell_params(&cell);

    // #240 knobs — identical semantics and defaults to train_nfsp.rs so the
    // probe and the full trainer can be A/B'd against each other directly.
    // Defaults updated to the #239 empirical win: at these settings the critic
    // explained-variance rises off ~0 and the policy entropy falls below the
    // uniform 1.0 floor within ~8 iters on cell beta05 (vs. flat ev=0 /
    // entropy≈1.22 at the earlier 0.05 / 1 / 0.01 defaults).
    let vf_coef: f64 = std::env::var("VF_COEF").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let br_train_steps: usize =
        std::env::var("BR_TRAIN_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let br_reward_scale: f32 = std::env::var("BR_REWARD_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.001);
    // #239 fix #4: optional dedicated critic LR. When set, the joint update
    // splits actor/critic into two backward passes and steps the critic at
    // this LR (vs. the actor's 3e-4), so the critic can fit the value target.
    let critic_lr: Option<f64> = std::env::var("CRITIC_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|v| *v > 0.0);
    // Issue #251 throughput lever: cap the number of minibatch gradient steps
    // per epoch in the BR PPO update. `None` (default) keeps the #239
    // all-minibatch full-rollout coverage; a small value (e.g. 2) trades a
    // bounded amount of BR fit for a per-iter speedup. Mirrored here so the
    // fast single-BR A/B stays representative of the NFSP/PSRO BR cost.
    let max_minibatches_per_epoch: Option<usize> = std::env::var("BR_MAX_MINIBATCHES_PER_EPOCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|v| *v > 0);

    tracing::info!("Single-BR probe (issue #241) — Burn backend: {BACKEND_LABEL}");
    tracing::info!("  cell             = {cell} (β={beta}, κ={kappa}, c={cost})");
    tracing::info!("  br_agent         = {BR_AGENT} ({} opponents frozen)", NUM_AGENTS - 1);
    tracing::info!("  iterations       = {iterations}");
    tracing::info!("  rollout_steps    = {rollout_steps}");
    tracing::info!("  vf_coef          = {vf_coef} (#240 value-loss weight)");
    tracing::info!("  br_train_steps   = {br_train_steps} (#240 PPO updates/iter)");
    tracing::info!("  br_reward_scale  = {br_reward_scale} (#240/#199 payoff rescale)");
    match critic_lr {
        Some(lr) => {
            tracing::info!("  critic_lr        = {lr} (#239 fix #4: separate critic optimizer)")
        }
        None => {
            tracing::info!("  critic_lr        = (unset — single shared actor+critic optimizer)")
        }
    }
    match max_minibatches_per_epoch {
        Some(cap) => tracing::info!(
            "  br_max_mb/epoch  = {cap} (issue #251 throughput lever: capped minibatch subsample)"
        ),
        None => {
            tracing::info!("  br_max_mb/epoch  = (unset — full #239 all-minibatch coverage)")
        }
    }

    let device: burn::tensor::Device<InnerBackend> = Default::default();

    let probe = make_cell_env(beta, kappa, cost, Some(SEED));
    let obs_dim = probe.obs_dim();
    drop(probe);
    tracing::info!("  obs_dim          = {obs_dim}");
    tracing::info!("  action_dims      = {:?} (factored multi-discrete)", action_dims());

    // Build N policies: the BR agent is freshly initialized and trained;
    // the rest are freshly-initialized frozen opponents (random at init).
    let policies: Vec<MultiDiscreteMlpBurnPolicy<B>> = (0..NUM_AGENTS)
        .map(|i| {
            MultiDiscreteMlpBurnPolicy::<B>::new_seeded(
                obs_dim,
                action_dims(),
                HIDDEN_DIM,
                SEED ^ (i as u64).wrapping_add(1),
                &device,
            )
        })
        .collect();
    let optimizers: Vec<_> = (0..NUM_AGENTS)
        .map(|_| {
            let inner = AdamConfig::new().init();
            BurnOptimizer::new(inner, 3e-4)
        })
        .collect();

    let joint_config = JointTrainerConfig {
        num_agents: NUM_AGENTS,
        rollout_steps,
        n_epochs: 4,
        minibatch_size: 256,
        vf_coef,
        // #240: consume the full rollout instead of one 256-sample draw.
        iterate_all_minibatches: true,
        // #239 fix #4: dedicated critic LR (None => single shared optimizer).
        critic_lr,
        // #251: optional cap on minibatch steps/epoch (None => full coverage).
        max_minibatches_per_epoch,
        ..Default::default()
    };

    let mut trainer = match critic_lr {
        Some(lr) => {
            // Dedicated per-agent critic optimizer at `critic_lr`. The actor
            // optimizers keep the construction LR (3e-4); the critic steps the
            // value-loss gradient separately at `lr`.
            let critic_optimizers: Vec<_> = (0..NUM_AGENTS)
                .map(|_| {
                    let inner = AdamConfig::new().init();
                    BurnOptimizer::new(inner, lr)
                })
                .collect();
            JointMultiAgentTrainer::<B, _, _>::with_critic_optimizers(
                policies,
                optimizers,
                critic_optimizers,
                joint_config,
                device,
            )?
        }
        None => JointMultiAgentTrainer::<B, _, _>::new(policies, optimizers, joint_config, device)?,
    };

    // Freeze-N−1: only BR_AGENT's optimizer steps; opponents stay fixed.
    let active_mask: Vec<bool> = (0..NUM_AGENTS).map(|i| i == BR_AGENT).collect();

    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let mut env = make_cell_env(beta, kappa, cost, Some(SEED));
    let mut last_obs = env.reset_joint(Some(SEED));

    tracing::info!("------------------------------------------------------------");
    let training_start = std::time::Instant::now();
    // Issue #251 AC#1 profile: coarse wall-clock attribution of the BR phase.
    // The single-BR probe isolates exactly the NFSP/PSRO inner work (rollout
    // collection vs. PPO update) over the un-batchable bucket-brigade rollout;
    // meta-solve is PSRO-only and separate. We accumulate per-phase time and
    // report the split at the end.
    let mut rollout_time = std::time::Duration::ZERO;
    let mut update_time = std::time::Duration::ZERO;

    for iter in 1..=iterations {
        let mut last_stats = None;
        // Track the BR agent's mean per-episode return in ORIGINAL (unscaled)
        // reward units, accumulated across this iteration's update steps.
        let mut ep_return_sum = 0.0_f64;
        let mut ep_count = 0_usize;

        for _ in 0..br_train_steps {
            let rollout_start = std::time::Instant::now();
            let mut rollout = trainer.collect_rollout(&mut env, &mut last_obs, &mut rng);
            rollout_time += rollout_start.elapsed();

            // Mean episode return for the BR agent, computed from the raw
            // (pre-scale) rollout rewards so the logged value is in the
            // game's native payoff units regardless of BR_REWARD_SCALE.
            let br_rewards = &rollout.rewards[BR_AGENT];
            let mut running = 0.0_f64;
            for (t, &r) in br_rewards.iter().enumerate() {
                running += r as f64;
                if rollout.dones[t] != 0.0 {
                    ep_return_sum += running;
                    ep_count += 1;
                    running = 0.0;
                }
            }
            // Partial trailing episode: count it so short runs still report.
            if running != 0.0 && ep_count == 0 {
                ep_return_sum += running;
                ep_count += 1;
            }

            // Affine reward rescale (#240/#199) before the PPO update. Mirrors
            // the PSRO/NFSP path: uniform scaling is return-invariant but keeps
            // the large-magnitude band numerically friendly for the critic.
            if br_reward_scale != 1.0 {
                for agent_rewards in rollout.rewards.iter_mut() {
                    for r in agent_rewards.iter_mut() {
                        *r *= br_reward_scale;
                    }
                }
            }

            let update_start = std::time::Instant::now();
            let stats = trainer.update_with_active_agents(
                &rollout,
                &active_mask,
                &mut rng,
                |_features: &[burn::tensor::Tensor<B, 2>]| -> Option<burn::tensor::Tensor<B, 1>> {
                    None
                },
            )?;
            update_time += update_start.elapsed();
            last_stats = Some(stats);
        }

        let stats = last_stats.expect("br_train_steps >= 1");
        let val_loss = stats.value_loss[BR_AGENT];
        let ev = stats.explained_var[BR_AGENT];
        let entropy = stats.entropy[BR_AGENT];
        let mean_return = if ep_count > 0 {
            ep_return_sum / ep_count as f64
        } else {
            f64::NAN
        };

        tracing::info!(
            "iter {:>3}/{}  br[val={:.4} ev={:.4} ent={:.4}]  mean_ep_return={:.2} ({} eps)",
            iter,
            iterations,
            val_loss,
            ev,
            entropy,
            mean_return,
            ep_count
        );
    }

    tracing::info!("------------------------------------------------------------");
    let total = training_start.elapsed().as_secs_f64();
    let rollout_s = rollout_time.as_secs_f64();
    let update_s = update_time.as_secs_f64();
    let measured = rollout_s + update_s;
    let pct = |x: f64| {
        if measured > 0.0 {
            100.0 * x / measured
        } else {
            0.0
        }
    };
    tracing::info!("Probe complete.  iterations={}  time={:.1}s", iterations, total);
    // Issue #251 AC#1 profile breakdown (BR phase: rollout vs PPO update).
    tracing::info!(
        "Profile (issue #251): rollout={:.1}s ({:.0}%)  br_update={:.1}s ({:.0}%)  [of {:.1}s measured]",
        rollout_s,
        pct(rollout_s),
        update_s,
        pct(update_s),
        measured
    );
    tracing::info!(
        "Read the per-iter `ev` column: near 0/negative = critic not fitting; → 1 = critic fits."
    );

    Ok(())
}
