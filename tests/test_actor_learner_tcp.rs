//! Networked actor-learner PPO loopback verification (issue #281, Phase 4 of
//! the distributed-training epic #265).
//!
//! Mirrors `tests/test_cartpole_async.rs` (the in-process crossbeam Phase 2
//! test) but wires the actors and learner together over real TCP sockets on
//! `127.0.0.1` instead of local channels, using
//! [`thrust_rl::train::ppo::transport`]'s `connect_actor_transport` /
//! `accept_actors_over_tcp`. Two remote "actor" threads dial in to a learner
//! listening on an OS-assigned loopback port, train CartPole PPO, and the
//! test asserts the loopback-verification acceptance bar from #281: reward
//! improves and shutdown is clean (every thread joins, no hang, no panic).
//!
//! Real multi-host runs (distinct `alc-*` hosts) are documented as a runbook
//! in `docs/DISTRIBUTED_TRAINING_DESIGN.md` rather than automated here.

#![cfg(feature = "training")]

use std::{net::TcpListener, thread};

use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    module::AutodiffModule,
    optim::AdamConfig,
    tensor::{Tensor, TensorData},
};
use rand::rngs::StdRng;
use thrust_rl::{
    env::{Environment, cartpole::CartPole},
    policy::mlp::{BurnActivation, MlpBurnConfig, MlpBurnPolicy},
    train::{
        optimizer::BurnOptimizer,
        ppo::{
            ActorStats, AsyncActorLearnerConfig, LearnerReport, PPOConfig, PPOTrainerBurn,
            actor_thread, learner_loop,
            transport::{accept_actors_over_tcp, connect_actor_transport},
        },
    },
};

type InnerBackend = NdArray<f32>;
type Backend = Autodiff<InnerBackend>;

const HIDDEN_DIM: usize = 64;
const SEED: u64 = 0;

/// Run one remote actor: dial `addr`, then run the *unmodified*
/// [`actor_thread`] against the resulting TCP-backed [`ActorChannels`].
/// Returns the actor's own [`ActorStats`] once it shuts down, after joining
/// the connection's demux thread.
fn run_remote_actor(
    actor_id: usize,
    addr: std::net::SocketAddr,
    policy: MlpBurnPolicy<InnerBackend>,
    device: NdArrayDevice,
    seed: u64,
    throttle: thrust_rl::train::ppo::actor_learner::ActorThrottle,
) -> ActorStats {
    let (channels, demux) =
        connect_actor_transport(addr).expect("remote actor failed to connect to learner");

    let act_device = device;
    let stats = actor_thread::<InnerBackend, _, _, _, _>(
        actor_id,
        CartPole::new(),
        policy,
        channels,
        device,
        seed,
        throttle,
        move |policy: &MlpBurnPolicy<InnerBackend>, obs: &[f32], rng: &mut StdRng| {
            let obs_t = Tensor::<InnerBackend, 2>::from_data(
                TensorData::new(obs.to_vec(), [1, obs.len()]),
                &act_device,
            );
            let (actions, log_probs, values) = policy.get_action_host_seeded(obs_t, rng);
            (actions[0], log_probs[0], values[0])
        },
    )
    .expect("actor_thread failed over tcp transport");

    demux.join().expect("actor tcp demux thread panicked");
    stats
}

/// Bind a loopback listener, spawn `config.num_actors` remote actor threads
/// dialing in, run the learner over the accepted TCP connections, and
/// return the learner's report plus each actor's own stats (from the actor
/// side, not the learner's reader-thread view).
fn run_tcp_cartpole(
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

    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind loopback listener");
    let addr = listener.local_addr().expect("listener has no local addr");

    // Spawn the remote actors first: TCP connections queue in the listen
    // backlog until `accept_actors_over_tcp` below calls `accept()`, so
    // ordering here is safe either way.
    let throttle = config.actor_throttle();
    let actor_threads: Vec<thread::JoinHandle<ActorStats>> = (0..config.num_actors)
        .map(|actor_id| {
            let policy = trainer.policy().valid();
            let seed = config.seed + 1 + actor_id as u64;
            thread::Builder::new()
                .name(format!("remote-actor-{actor_id}"))
                .spawn(move || run_remote_actor(actor_id, addr, policy, device, seed, throttle))
                .expect("failed to spawn remote actor thread")
        })
        .collect();

    let (experience_rx, actor_handles) = accept_actors_over_tcp(&listener, config.num_actors)
        .expect("learner failed to accept remote actors");

    let (_trainer, report) = learner_loop(
        config,
        trainer,
        obs_dim,
        &device,
        &experience_rx,
        &actor_handles,
        |p: &MlpBurnPolicy<Backend>, o, a| p.evaluate_actions(o, a),
        |p: &MlpBurnPolicy<Backend>, o| p.forward(o).1.into_data().to_vec().unwrap_or_default(),
    )
    .expect("learner_loop failed over tcp transport");

    // Clean shutdown: every learner-side reader thread and every remote
    // actor thread must join without hanging or panicking.
    for handle in actor_handles {
        handle.join().expect("learner-side tcp reader thread failed");
    }
    let actor_stats: Vec<ActorStats> = actor_threads
        .into_iter()
        .map(|h| h.join().expect("remote actor thread panicked"))
        .collect();

    (report, actor_stats)
}

/// Fast wiring smoke test: 2 remote actors dial in over real TCP sockets,
/// stream a couple of updates' worth of real CartPole transitions, and the
/// learner broadcasts policy updates back over the wire. No learning bar —
/// proves the transport plumbing (wire encoding, demux, shutdown) end to
/// end quickly.
#[test]
fn tcp_actor_learner_wiring_smoke() {
    let config = AsyncActorLearnerConfig {
        num_actors: 2,
        num_steps: 32,
        total_env_steps: 32 * 2 * 2, // 2 updates
        broadcast_every: 1,
        max_lead_steps: 0,
        gamma: 0.99,
        gae_lambda: 0.95,
        use_vtrace: false,
        vtrace_rho_bar: 1.0,
        vtrace_c_bar: 1.0,
        seed: SEED,
    };

    let (report, actor_stats) = run_tcp_cartpole(&config, 3e-4, 1, 32);

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
        assert!(stats.steps_sent >= 32 * 2, "actor {i} sent only {} steps", stats.steps_sent);
        assert!(
            stats.policy_updates_received >= 1,
            "actor {i} never loaded a policy broadcast delivered over tcp"
        );
        assert!(stats.last_policy_version >= 1);
    }
}

/// Loopback-verification acceptance bar (issue #281): remote actors connect
/// over TCP, the learner trains CartPole PPO, and reward improves over the
/// ~22-step random baseline within a modest env-step budget (kept smaller
/// than the in-process #279 bar test to bound wall-clock: this test pays
/// per-step socket + bincode framing overhead the in-process path does
/// not). Shutdown must be clean: every actor thread and every learner-side
/// reader thread joins without hanging or panicking.
#[test]
fn tcp_actor_learner_reward_improves_and_shuts_down_cleanly() {
    let config = AsyncActorLearnerConfig {
        num_actors: 2,
        num_steps: 64,
        total_env_steps: 64 * 2 * 20, // 20 updates
        broadcast_every: 1,
        max_lead_steps: 0,
        gamma: 0.99,
        gae_lambda: 0.95,
        use_vtrace: false,
        vtrace_rho_bar: 1.0,
        vtrace_c_bar: 1.0,
        seed: SEED,
    };

    let (report, actor_stats) = run_tcp_cartpole(&config, 1e-3, 10, 32);

    assert_eq!(report.updates_completed, 20);
    assert_eq!(report.last_policy_version, 20);
    assert_eq!(actor_stats.len(), 2);
    for stats in &actor_stats {
        assert!(
            stats.policy_updates_received >= 1,
            "actor {} never loaded a policy broadcast delivered over tcp",
            stats.actor_id
        );
    }

    // Random-policy CartPole averages ~22 steps/episode; a working PPO
    // update pipeline (including the network hop) should clear that
    // comfortably within 20 updates.
    let mean_reward = report.mean_recent_episode_reward(20);
    assert!(
        report.episodes_completed > 0,
        "no episodes completed within the tcp test budget"
    );
    assert!(
        mean_reward > 30.0,
        "networked actor-learner did not show learning: mean episode reward {mean_reward:.1} \
         after {} env steps ({} episodes) over tcp",
        report.env_steps_consumed,
        report.episodes_completed,
    );
}
