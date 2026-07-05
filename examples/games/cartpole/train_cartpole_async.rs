//! Asynchronous actor-learner CartPole PPO training (Burn backend).
//!
//! Phase 2 of the distributed-training epic (#265): N inference-only
//! actor threads collect rollouts in parallel on their own CartPole
//! envs and stream [`thrust_rl::multi_agent::Experience`] messages over
//! a `crossbeam-channel` MPSC channel to a single learner running the
//! unchanged [`thrust_rl::train::ppo::PPOTrainerBurn`]. After every
//! update the learner broadcasts refreshed policy weights back to the
//! actors as [`thrust_rl::multi_agent::PolicyBroadcast`] bytes.
//!
//! Same PPO recipe as `train_cartpole_modern.rs` (the synchronous
//! baseline): 2-layer 128-hidden ReLU MLP with orthogonal init,
//! `256`-step rollouts, `n_epochs = 10`, `batch_size = 128`, 200k env
//! steps total. The rollout batch here is `[256, 4]` (4 actors) instead
//! of the baseline's `[256, 16]` (16 pooled envs), so the async run does
//! ~4x more (smaller) PPO updates for the same step budget.
//!
//! **Staleness note**: actor trajectories are stale relative to the
//! learner. By default they are passed to PPO uncorrected; with
//! `broadcast_every = 1` and 4 actors this is empirically fine on
//! CartPole (expect mean episode reward ≥ 400 within the 200k budget,
//! comparable to the synchronous baseline). Set `USE_VTRACE=1` to enable
//! V-trace off-policy correction (issue #280), which re-weights the
//! advantages by the importance ratio between the current learner policy
//! and each actor's (possibly stale) behavior policy — the correction
//! that keeps convergence stable when staleness is elevated.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_cartpole_async --features training --release
//! ```
//!
//! Override the step budget / actor count / staleness via env vars:
//!
//! ```bash
//! TOTAL_TIMESTEPS=50000 NUM_ACTORS=2 cargo run --example train_cartpole_async \
//!     --features training --release
//!
//! # Staleness experiment: elevate the actor lead budget and compare the
//! # GAE and V-trace arms on final reward.
//! MAX_LEAD_STEPS=25600 USE_VTRACE=0 cargo run --example train_cartpole_async \
//!     --features training --release   # GAE arm
//! MAX_LEAD_STEPS=25600 USE_VTRACE=1 cargo run --example train_cartpole_async \
//!     --features training --release   # V-trace arm
//! ```
//!
//! Recognized env vars: `TOTAL_TIMESTEPS`, `NUM_ACTORS`, `BROADCAST_EVERY`,
//! `MAX_LEAD_STEPS`, `USE_VTRACE` (`1`/`true` to enable), `GAE_LAMBDA`,
//! `SEED`.

use anyhow::Result;
use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    optim::AdamConfig,
    tensor::{Tensor, TensorData},
};
use crossbeam_channel::unbounded;
use rand::rngs::StdRng;
use thrust_rl::{
    env::{Environment, cartpole::CartPole},
    policy::mlp::{BurnActivation, MlpBurnConfig, MlpBurnPolicy},
    train::{
        optimizer::BurnOptimizer,
        ppo::{
            AsyncActorLearnerConfig, PPOConfig, PPOTrainerBurn, actor_learner::ActorHandle,
            learner_loop, spawn_actor,
        },
    },
};

// The async actor-learner ships CPU-first: policy bytes are
// backend-agnostic, but actor threads each run their own inference
// module, which is the natural fit for the NdArray CPU backend.
type InnerBackend = NdArray<f32>;
type Backend = Autodiff<InnerBackend>;

const DEFAULT_NUM_ACTORS: usize = 4;
const NUM_STEPS: usize = 256;
const DEFAULT_TIMESTEPS: usize = 200_000;
const LEARNING_RATE: f64 = 3e-4;
const HIDDEN_DIM: usize = 128;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;
const SEED: u64 = 0;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let total_timesteps: usize = std::env::var("TOTAL_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TIMESTEPS);
    let num_actors: usize = std::env::var("NUM_ACTORS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_ACTORS);
    let broadcast_every: usize =
        std::env::var("BROADCAST_EVERY").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let max_lead_steps: usize =
        std::env::var("MAX_LEAD_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let use_vtrace: bool = std::env::var("USE_VTRACE")
        .ok()
        .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
        .unwrap_or(false);
    let seed: u64 = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(SEED);
    let gae_lambda: f32 = std::env::var("GAE_LAMBDA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(GAE_LAMBDA);

    tracing::info!("Starting Async Actor-Learner CartPole PPO (NdArray<f32> CPU)");

    let training_start = std::time::Instant::now();

    // Probe environment dimensions.
    let probe = CartPole::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n,
        _ => panic!("Expected discrete action space"),
    };

    let config = AsyncActorLearnerConfig {
        num_actors,
        num_steps: NUM_STEPS,
        total_env_steps: total_timesteps,
        broadcast_every,
        max_lead_steps, // 0 = auto: 2 * broadcast_every * num_steps of actor lead
        gamma: GAMMA,
        gae_lambda,
        use_vtrace,
        vtrace_rho_bar: 1.0,
        vtrace_c_bar: 1.0,
        seed,
    };
    config.validate()?;

    tracing::info!("Environment: CartPole-v1");
    tracing::info!("  obs_dim     = {}", obs_dim);
    tracing::info!("  action_dim  = {}", action_dim);
    tracing::info!("  num_actors  = {}", config.num_actors);
    tracing::info!("  num_steps   = {}", config.num_steps);
    tracing::info!("  total_timesteps = {}", config.total_env_steps);
    tracing::info!("  broadcast_every = {}", config.broadcast_every);
    tracing::info!(
        "  max_lead_steps  = {} (effective {})",
        config.max_lead_steps,
        config.effective_max_lead_steps()
    );
    tracing::info!("  advantage       = {}", if config.use_vtrace { "V-trace" } else { "GAE" });
    tracing::info!("  planned PPO updates = {}", config.num_updates());
    tracing::info!("------------------------------------------------------------");

    let device = Default::default();

    // Learner policy: same architecture + seed as the sync baseline.
    let policy_config = MlpBurnConfig {
        num_layers: 2,
        hidden_dim: HIDDEN_DIM,
        use_orthogonal_init: true,
        activation: BurnActivation::ReLU,
        seed: Some(seed),
    };
    let policy = MlpBurnPolicy::<Backend>::with_config(obs_dim, action_dim, policy_config, &device);

    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<Backend, MlpBurnPolicy<Backend>, _> =
        BurnOptimizer::new(inner_opt, LEARNING_RATE);

    let ppo_config = PPOConfig::new()
        .learning_rate(LEARNING_RATE)
        .n_epochs(10)
        .batch_size(128)
        .gamma(GAMMA as f64)
        .gae_lambda(gae_lambda as f64)
        .clip_range(0.2)
        .clip_range_vf(0.2)
        .vf_coef(0.5)
        .ent_coef(0.01)
        .max_grad_norm(0.5)
        .target_kl(1.0);

    let trainer = PPOTrainerBurn::new(ppo_config, policy, burn_opt)?;

    // --- Spawn actors: each owns one env + a copy of the initial policy ---
    let (experience_tx, experience_rx) = unbounded();
    let actors: Vec<ActorHandle> = (0..config.num_actors)
        .map(|actor_id| {
            let act_device = device;
            spawn_actor::<InnerBackend, _, _, _>(
                actor_id,
                CartPole::new(),
                trainer.policy().valid(),
                experience_tx.clone(),
                act_device,
                seed + 1 + actor_id as u64,
                config.actor_throttle(),
                move |policy: &MlpBurnPolicy<InnerBackend>, obs: &[f32], rng: &mut StdRng| {
                    let obs_t = Tensor::<InnerBackend, 2>::from_data(
                        TensorData::new(obs.to_vec(), [1, obs.len()]),
                        &act_device,
                    );
                    let (actions, log_probs, values) = policy.get_action_host_seeded(obs_t, rng);
                    (actions[0], log_probs[0], values[0])
                },
            )
        })
        .collect();
    drop(experience_tx); // learner holds the only receiver; actors hold senders

    // --- Learner: fills the rollout buffer, trains, broadcasts ---
    let (trainer, report) = learner_loop(
        &config,
        trainer,
        obs_dim,
        &device,
        &experience_rx,
        &actors,
        |p: &MlpBurnPolicy<Backend>, o, a| p.evaluate_actions(o, a),
        |p: &MlpBurnPolicy<Backend>, o| p.forward(o).1.into_data().to_vec().unwrap_or_default(),
    )?;

    // --- Join actors and report ---
    let training_duration = training_start.elapsed();
    tracing::info!("------------------------------------------------------------");
    for handle in actors {
        let stats = handle.join()?;
        tracing::info!(
            "actor {}: steps_sent={} episodes={} policy_updates_received={} last_version={}",
            stats.actor_id,
            stats.steps_sent,
            stats.episodes_completed,
            stats.policy_updates_received,
            stats.last_policy_version,
        );
    }

    tracing::info!("Training complete.");
    tracing::info!("  updates          : {}", report.updates_completed);
    tracing::info!("  env steps trained: {}", report.env_steps_consumed);
    tracing::info!("  episodes         : {}", report.episodes_completed);
    tracing::info!(
        "  broadcasts sent  : {} (last version {})",
        report.broadcasts_sent,
        report.last_policy_version
    );
    tracing::info!(
        "  final mean episode reward (last≤100): {:.2}",
        report.mean_recent_episode_reward(100)
    );
    tracing::info!("  training time    : {:.1}s", training_duration.as_secs_f64());
    tracing::info!(
        "  steps/sec        : {:.0}",
        report.env_steps_consumed as f64 / training_duration.as_secs_f64()
    );
    let _ = trainer; // trainer ownership returns to the caller, as with the sync loop

    Ok(())
}
