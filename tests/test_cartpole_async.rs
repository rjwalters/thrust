//! Asynchronous actor-learner PPO tests on [`CartPole`] (issue #279,
//! Phase 2 of the distributed-training epic #265).
//!
//! Two tiers, mirroring the repo's convergence-test convention
//! (`tests/test_a2c_cartpole.rs`, `tests/test_sac_pendulum.rs`):
//!
//! 1. **`cartpole_async_wiring_smoke`** (always runs) — fast end-to-end wiring
//!    check: 2 actor threads stream real CartPole transitions over the
//!    crossbeam channel, the learner runs a couple of PPO updates, every actor
//!    receives at least one policy broadcast, and all reported stats are
//!    finite. No learning bar.
//!
//! 2. **`cartpole_async_reaches_learning_bar`** — the curated acceptance bar
//!    for #279: a 2-actor run reaches **mean episode reward ≥ 50** (random
//!    baseline ≈ 22) within a 5k env-step budget, confirming that learning
//!    happens through the async path (stale-but-fresh trajectories,
//!    `broadcast_every = 1`, no V-trace — see issue #280 for the staleness
//!    correction). Run with `--release` to keep the wall clock reasonable:
//!
//!    ```text
//!    cargo test --release --features training --test test_cartpole_async
//!    ```

#![cfg(feature = "training")]

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
            AsyncActorLearnerConfig, PPOConfig, PPOTrainerBurn,
            actor_learner::{ActorHandle, ActorStats, LearnerReport},
            learner_loop, spawn_actor,
        },
    },
};

type InnerBackend = NdArray<f32>;
type Backend = Autodiff<InnerBackend>;

const HIDDEN_DIM: usize = 64;
const SEED: u64 = 0;

/// Wire up trainer + actors, run the learner to completion, join the
/// actors, and return everything the assertions need.
fn run_async_cartpole(
    config: &AsyncActorLearnerConfig,
    learning_rate: f64,
    n_epochs: usize,
    batch_size: usize,
) -> (LearnerReport, Vec<ActorStats>) {
    let device = Default::default();

    let probe = CartPole::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n,
        _ => panic!("CartPole is discrete"),
    };

    let policy_config = MlpBurnConfig {
        num_layers: 2,
        hidden_dim: HIDDEN_DIM,
        use_orthogonal_init: true,
        activation: BurnActivation::ReLU,
        seed: Some(config.seed),
    };
    let policy = MlpBurnPolicy::<Backend>::with_config(obs_dim, action_dim, policy_config, &device);

    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<Backend, MlpBurnPolicy<Backend>, _> =
        BurnOptimizer::new(inner_opt, learning_rate);

    let ppo_config = PPOConfig::new()
        .learning_rate(learning_rate)
        .n_epochs(n_epochs)
        .batch_size(batch_size)
        .gamma(config.gamma as f64)
        .gae_lambda(config.gae_lambda as f64)
        .clip_range(0.2)
        .clip_range_vf(0.2)
        .vf_coef(0.5)
        .ent_coef(0.01)
        .max_grad_norm(0.5)
        .target_kl(1.0);

    let trainer = PPOTrainerBurn::new(ppo_config, policy, burn_opt).unwrap();

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
                config.seed + 1 + actor_id as u64,
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
    drop(experience_tx);

    let (_trainer, report) = learner_loop(
        config,
        trainer,
        obs_dim,
        &device,
        &experience_rx,
        &actors,
        |p: &MlpBurnPolicy<Backend>, o, a| p.evaluate_actions(o, a),
        |p: &MlpBurnPolicy<Backend>, o| p.forward(o).1.into_data().to_vec().unwrap_or_default(),
    )
    .expect("learner_loop failed");

    let actor_stats: Vec<ActorStats> =
        actors.into_iter().map(|h| h.join().expect("actor failed")).collect();

    (report, actor_stats)
}

/// Tier 1 (always runs): the async pipeline wires together end-to-end.
/// 2 actors × 32-step rollouts × 2 updates = 128 env steps; finishes in
/// well under a second even unoptimized. Asserts bookkeeping and that
/// every actor received at least one policy broadcast — no learning bar.
#[test]
fn cartpole_async_wiring_smoke() {
    let config = AsyncActorLearnerConfig {
        num_actors: 2,
        num_steps: 32,
        total_env_steps: 32 * 2 * 2, // 2 updates
        broadcast_every: 1,
        max_lead_steps: 0,
        gamma: 0.99,
        gae_lambda: 0.95,
        seed: SEED,
    };

    let (report, actor_stats) = run_async_cartpole(&config, 3e-4, 1, 32);

    assert_eq!(report.updates_completed, 2);
    assert_eq!(report.env_steps_consumed, 128);
    assert_eq!(report.broadcasts_sent, 2);
    assert_eq!(report.last_policy_version, 2);

    let stats = report.final_stats.expect("updates ran");
    assert!(stats.policy_loss.is_finite());
    assert!(stats.value_loss.is_finite());
    assert!(stats.entropy.is_finite());

    assert_eq!(actor_stats.len(), 2);
    for (i, stats) in actor_stats.iter().enumerate() {
        assert_eq!(stats.actor_id, i);
        // Each actor contributed 32 steps per update; it may have sent a
        // few more that stayed queued past the final update.
        assert!(stats.steps_sent >= 32 * 2, "actor {i} sent only {} steps", stats.steps_sent);
        assert!(
            stats.policy_updates_received >= 1,
            "actor {i} never received a policy broadcast"
        );
        assert!(stats.last_policy_version >= 1);
    }
}

/// Tier 2: the #279 acceptance bar. 2 actors, 5k env steps, mean episode
/// reward over the last ≤100 completed episodes must reach ≥ 50 — well
/// clear of the ~22-step random baseline, confirming the learner's PPO
/// updates propagate back to the actors and improve the data they
/// collect. Uses the same PPO recipe as the sync baseline example.
#[test]
fn cartpole_async_reaches_learning_bar() {
    let config = AsyncActorLearnerConfig {
        num_actors: 2,
        num_steps: 64,
        total_env_steps: 10_240, // 80 updates of 128 transitions
        broadcast_every: 1,
        max_lead_steps: 0,
        gamma: 0.99,
        gae_lambda: 0.95,
        seed: SEED,
    };

    let (report, actor_stats) = run_async_cartpole(&config, 1e-3, 10, 32);

    assert_eq!(report.updates_completed, 80);
    assert_eq!(report.env_steps_consumed, 10_240);

    // Policy broadcast is confirmed end-to-end: the learner versioned
    // every update and each actor loaded at least one refresh.
    assert_eq!(report.last_policy_version, 80);
    for stats in &actor_stats {
        assert!(
            stats.policy_updates_received >= 1,
            "actor {} never received a policy broadcast",
            stats.actor_id
        );
    }

    let mean_reward = report.mean_recent_episode_reward(30);
    assert!(report.episodes_completed > 0, "no episodes completed within the 5k-step budget");
    assert!(
        mean_reward >= 50.0,
        "async actor-learner failed the learning bar: mean episode reward {mean_reward:.1} < 50 \
         after {} env steps ({} episodes)",
        report.env_steps_consumed,
        report.episodes_completed,
    );
}
