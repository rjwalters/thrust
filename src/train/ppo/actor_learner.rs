//! Single-host asynchronous actor-learner PPO (IMPALA-style topology).
//!
//! Phase 2 of the distributed-training epic (#265) — see
//! `docs/DISTRIBUTED_TRAINING_DESIGN.md`. N inference-only actor threads
//! collect rollouts in parallel; one learner thread runs the existing
//! [`crate::train::ppo::trainer::PPOTrainerBurn`] update unchanged.
//! No gradient synchronization is required.
//!
//! # Data flow
//!
//! ```text
//! [actor 0]  env_0 → inference (inner module) → Experience → sender ─┐
//! [actor 1]  env_1 → inference (inner module) → Experience → sender ─┤→ learner_rx
//! [actor N]  env_N → inference (inner module) → Experience → sender ─┘
//!                                                                     │
//!                                     ┌───────────────────────────────┘
//!                                     ▼
//!                          [learner thread]
//!                          RolloutBuffer.add(experience)      (column = actor)
//!                          when every actor column has num_steps rows:
//!                              advantages (GAE or V-trace) → PPOTrainerBurn::train_step
//!                              serialize policy → PolicyBroadcast to every actor
//! ```
//!
//! **Channel topology.** One `crossbeam_channel::unbounded` MPSC channel
//! carries [`crate::multi_agent::Experience`] messages from all actors to
//! the learner, plus one unbounded SPSC broadcast channel per actor for
//! [`crate::multi_agent::PolicyBroadcast`] (the learner sends a clone to
//! each actor's sender). Actors poll their broadcast receiver with a
//! non-blocking `try_recv` on every env step; there is no `select!`.
//!
//! # Staleness correction (V-trace) and the staleness valve
//!
//! By default (`use_vtrace = false`) trajectories collected under a
//! stale policy are passed to PPO **without importance-weighting
//! correction**. For low staleness (`broadcast_every = 1`,
//! `num_actors <= 4`) this is empirically acceptable on simple envs such
//! as CartPole. Correctness under higher staleness is provided by
//! V-trace (Espeholt et al. 2018 — Phase 3 of the epic, issue #280):
//! set [`AsyncActorLearnerConfig::use_vtrace`] to `true` and the learner
//! re-evaluates each stored `(obs, action)` under its *current* policy,
//! using the importance ratio against the actor's stored behavior
//! log-probs to correct the advantages before the PPO update. On fresh
//! on-policy data this reduces to GAE(λ=1); under staleness it clips the
//! ratio at `vtrace_rho_bar` / `vtrace_c_bar`. As a soft
//! staleness valve (orthogonal to V-trace, and useful with or without
//! it), each actor pauses collection once it has produced
//! more than `max_lead_steps` env steps beyond what the learner has
//! provably consumed (inferred from the newest received policy
//! version), and waits for the next broadcast (see
//! [`AsyncActorLearnerConfig::max_lead_steps`]); this bounds both the
//! learner's queue depth and the worst-case policy lag without any
//! off-policy correction, while still letting collection overlap with
//! the learner's gradient steps.
//!
//! # Policy transfer
//!
//! Policy weights travel learner → actor as serialized bytes
//! ([`burn::record::BinBytesRecorder`] with
//! [`burn::record::FullPrecisionSettings`]), wrapped in
//! [`crate::multi_agent::PolicyBroadcast::Bytes`]. Bytes are always
//! `Send` regardless of the tensor backend (raw module clones are only
//! `Send` when the backend's tensor primitives are, which e.g. wgpu's
//! are not), and the serialized record is backend-agnostic, so the
//! learner can train on `Autodiff<NdArray>` while an actor runs plain
//! `NdArray` — or any other backend pair with matching element types.
//!
//! # Failure model
//!
//! Actor death is fatal for the run: the learner's fill loop starves
//! waiting for the dead actor's column and errors out after a timeout.
//! Graceful actor restart is out of scope for Phase 2.
// TODO(#265 Phase 4): graceful recovery on actor thread panic instead of
// timing out the learner fill loop.

use std::{collections::VecDeque, sync::Arc, thread, time::Duration};

use anyhow::{Result, anyhow};
use burn::{
    module::{AutodiffModule, Module},
    optim::Optimizer,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::{
        Int, Tensor, TensorData,
        backend::{AutodiffBackend, Backend},
    },
};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, unbounded};
use rand::{SeedableRng, rngs::StdRng};

use super::{stats::TrainingStats, trainer::PPOTrainerBurn};
use crate::{
    buffer::rollout::{RolloutBatch, RolloutBuffer, compute_advantages},
    env::Environment,
    multi_agent::{AgentId, ControlMessage, Experience, PolicyBroadcast},
};

/// How long the learner waits on the experience channel before deciding
/// an actor has died. Generous on purpose: a healthy actor produces an
/// experience every few microseconds, so hitting this means something is
/// genuinely wrong (actor panic, deadlocked env), not that we are slow.
const LEARNER_RECV_TIMEOUT: Duration = Duration::from_secs(60);

/// How long a throttled actor blocks on its broadcast receiver before
/// re-checking the control channel for shutdown.
const ACTOR_BROADCAST_WAIT: Duration = Duration::from_millis(50);

/// Configuration for the single-host asynchronous actor-learner runner.
///
/// See the [module docs](crate::train::ppo::actor_learner) for the
/// architecture and the staleness caveat.
#[derive(Debug, Clone)]
pub struct AsyncActorLearnerConfig {
    /// Number of actor threads (each owns one environment instance).
    pub num_actors: usize,

    /// Per-actor env steps consumed per PPO update. Each update trains on
    /// a `[num_steps, num_actors]` rollout buffer, i.e.
    /// `num_steps * num_actors` transitions.
    pub num_steps: usize,

    /// Total env-step budget. The learner runs
    /// `total_env_steps / (num_steps * num_actors)` PPO updates.
    pub total_env_steps: usize,

    /// Broadcast updated policy weights to actors after every
    /// `broadcast_every` learner updates. `1` (the default) keeps actors
    /// as fresh as possible, which is what makes the uncorrected
    /// (no V-trace) PPO update acceptable on simple envs.
    pub broadcast_every: usize,

    /// Soft staleness valve: the maximum number of env steps an actor
    /// may run **ahead of the learner's consumption**. Policy version
    /// `v` proves the learner has consumed `v * broadcast_every *
    /// num_steps` transitions from each actor, so an actor pauses (and
    /// waits for the next [`crate::multi_agent::PolicyBroadcast`]) once
    /// its total sent steps reach that mark plus `max_lead_steps`. This
    /// bounds the learner's queue depth and the worst-case policy lag
    /// at `max_lead_steps` — a *cumulative* budget, unlike a
    /// per-version allowance, which would let production permanently
    /// outpace consumption and grow the backlog (and therefore the
    /// staleness) linearly over the run.
    ///
    /// `0` selects the default of `2 * broadcast_every * num_steps`:
    /// one broadcast cycle of lead for pipeline overlap (actors collect
    /// while the learner trains) plus one cycle in flight. Must be at
    /// least `broadcast_every * num_steps` or the learner starves. See
    /// [`AsyncActorLearnerConfig::effective_max_lead_steps`].
    pub max_lead_steps: usize,

    /// Discount factor for GAE.
    pub gamma: f32,

    /// GAE lambda. Ignored when [`Self::use_vtrace`] is `true` (V-trace
    /// has no lambda parameter), but still validated to lie in `(0, 1]`.
    pub gae_lambda: f32,

    /// If true, replace GAE with V-trace (Espeholt et al. 2018) for
    /// off-policy correction of stale actor trajectories. The learner
    /// re-evaluates each stored `(obs, action)` under the *current*
    /// policy to obtain target log-probs `log π(a_t|s_t)`, and the
    /// importance ratio against the actor's stored behavior log-probs
    /// `log μ(a_t|s_t)` corrects for policy lag before PPO sees the
    /// advantages. Defaults to `false`, preserving the GAE path and all
    /// existing behavior.
    pub use_vtrace: bool,

    /// V-trace rho clipping threshold (Espeholt 2018, eq. 1). Ignored
    /// when [`Self::use_vtrace`] is `false`. Typical value: `1.0` (the
    /// IMPALA paper default).
    pub vtrace_rho_bar: f32,

    /// V-trace c clipping threshold (Espeholt 2018, eq. 2). Ignored when
    /// [`Self::use_vtrace`] is `false`. Typical value: `1.0`.
    pub vtrace_c_bar: f32,

    /// Base seed. Actor `i` samples actions with
    /// `StdRng::seed_from_u64(seed + 1 + i)` when wired through
    /// [`spawn_actor`] by the caller.
    pub seed: u64,
}

impl Default for AsyncActorLearnerConfig {
    fn default() -> Self {
        Self {
            num_actors: 4,
            num_steps: 256,
            total_env_steps: 200_000,
            broadcast_every: 1,
            max_lead_steps: 0,
            gamma: 0.99,
            gae_lambda: 0.95,
            use_vtrace: false,
            vtrace_rho_bar: 1.0,
            vtrace_c_bar: 1.0,
            seed: 0,
        }
    }
}

impl AsyncActorLearnerConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns an error when any count is zero, when the step budget is
    /// smaller than a single update's worth of transitions, when
    /// `gamma` / `gae_lambda` fall outside `(0, 1]`, or when the
    /// staleness valve is too tight for the broadcast cadence (which
    /// would deadlock the learner's fill loop).
    pub fn validate(&self) -> Result<()> {
        if self.num_actors == 0 {
            return Err(anyhow!("num_actors must be >= 1"));
        }
        if self.num_steps == 0 {
            return Err(anyhow!("num_steps must be >= 1"));
        }
        if self.broadcast_every == 0 {
            return Err(anyhow!("broadcast_every must be >= 1"));
        }
        if self.total_env_steps < self.num_steps * self.num_actors {
            return Err(anyhow!(
                "total_env_steps ({}) must cover at least one update ({} = num_steps * num_actors)",
                self.total_env_steps,
                self.num_steps * self.num_actors
            ));
        }
        if !(0.0..=1.0).contains(&self.gamma) || self.gamma == 0.0 {
            return Err(anyhow!("gamma must be in (0, 1]"));
        }
        if !(0.0..=1.0).contains(&self.gae_lambda) || self.gae_lambda == 0.0 {
            return Err(anyhow!("gae_lambda must be in (0, 1]"));
        }
        // V-trace clip thresholds are only consumed on the V-trace path,
        // but validated unconditionally so a misconfigured value can never
        // reach `compute_vtrace_advantages`.
        if self.vtrace_rho_bar <= 0.0 {
            return Err(anyhow!("vtrace_rho_bar must be positive, got {}", self.vtrace_rho_bar));
        }
        if self.vtrace_c_bar <= 0.0 {
            return Err(anyhow!("vtrace_c_bar must be positive, got {}", self.vtrace_c_bar));
        }
        // The learner consumes broadcast_every * num_steps transitions
        // per actor between broadcasts; a lead budget tighter than that
        // starves the learner and deadlocks the run.
        if self.max_lead_steps != 0 && self.max_lead_steps < self.broadcast_every * self.num_steps {
            return Err(anyhow!(
                "max_lead_steps ({}) must be >= broadcast_every * num_steps ({}) \
                 or the learner's fill loop deadlocks",
                self.max_lead_steps,
                self.broadcast_every * self.num_steps
            ));
        }
        Ok(())
    }

    /// Number of PPO updates the learner will run for this budget.
    pub fn num_updates(&self) -> usize {
        (self.total_env_steps / (self.num_steps * self.num_actors)).max(1)
    }

    /// Resolve the staleness valve: `max_lead_steps` when nonzero,
    /// otherwise `2 * broadcast_every * num_steps`.
    pub fn effective_max_lead_steps(&self) -> usize {
        if self.max_lead_steps != 0 {
            self.max_lead_steps
        } else {
            2 * self.broadcast_every * self.num_steps
        }
    }

    /// Build the per-actor [`ActorThrottle`] for this configuration.
    pub fn actor_throttle(&self) -> ActorThrottle {
        ActorThrottle {
            steps_per_broadcast: self.broadcast_every * self.num_steps,
            max_lead_steps: self.effective_max_lead_steps(),
        }
    }

    /// Enable or disable V-trace off-policy correction (builder style).
    pub fn use_vtrace(mut self, enabled: bool) -> Self {
        self.use_vtrace = enabled;
        self
    }

    /// Set the V-trace rho clip threshold (builder style).
    pub fn vtrace_rho_bar(mut self, rho_bar: f32) -> Self {
        self.vtrace_rho_bar = rho_bar;
        self
    }

    /// Set the V-trace c clip threshold (builder style).
    pub fn vtrace_c_bar(mut self, c_bar: f32) -> Self {
        self.vtrace_c_bar = c_bar;
        self
    }
}

/// Per-actor staleness throttle derived from
/// [`AsyncActorLearnerConfig::actor_throttle`].
///
/// Policy version `v` proves the learner has consumed
/// `v * steps_per_broadcast` transitions from this actor, so the actor
/// pauses once `steps_sent >= v * steps_per_broadcast + max_lead_steps`
/// and waits for the next broadcast. See
/// [`AsyncActorLearnerConfig::max_lead_steps`] for why the budget is
/// cumulative.
#[derive(Debug, Clone, Copy)]
pub struct ActorThrottle {
    /// Env steps per actor the learner consumes between successive
    /// policy broadcasts (`broadcast_every * num_steps`).
    pub steps_per_broadcast: usize,
    /// Maximum steps the actor may run ahead of proven consumption.
    /// `0` disables the throttle entirely (unbounded lead — only safe
    /// when the caller guarantees consumption keeps up).
    pub max_lead_steps: usize,
}

impl ActorThrottle {
    /// A disabled throttle (unbounded actor lead).
    pub fn disabled() -> Self {
        Self { steps_per_broadcast: 0, max_lead_steps: 0 }
    }

    /// Whether an actor that has sent `steps_sent` experiences and last
    /// loaded policy `version` should pause and wait for a broadcast.
    pub fn should_pause(&self, steps_sent: usize, version: u64) -> bool {
        self.max_lead_steps != 0
            && steps_sent >= (version as usize) * self.steps_per_broadcast + self.max_lead_steps
    }
}

/// Channel endpoints owned by one actor thread.
///
/// Constructed by [`spawn_actor`]; exposed publicly so callers with
/// bespoke threading can wire [`actor_thread`] themselves.
pub struct ActorChannels {
    /// Cloned sender half of the shared actors→learner MPSC channel.
    pub experience_tx: Sender<Experience>,
    /// Receive half of this actor's learner→actor broadcast channel.
    pub broadcast_rx: Receiver<PolicyBroadcast>,
    /// Receive half of this actor's control channel (shutdown etc.).
    pub control_rx: Receiver<ControlMessage>,
}

/// Per-actor counters returned by [`actor_thread`] on exit.
#[derive(Debug, Clone, Default)]
pub struct ActorStats {
    /// Which actor produced these stats.
    pub actor_id: AgentId,
    /// Experiences sent to the learner.
    pub steps_sent: usize,
    /// Episodes this actor completed (terminated or truncated).
    pub episodes_completed: usize,
    /// Policy broadcasts received *and loaded* by this actor.
    pub policy_updates_received: usize,
    /// Version number of the newest policy this actor loaded
    /// (`0` = still on the initial policy).
    pub last_policy_version: u64,
}

/// Handle to a spawned actor thread.
///
/// The learner uses [`ActorHandle::send_broadcast`] /
/// [`ActorHandle::send_shutdown`]; the orchestrating caller consumes the
/// handle with [`ActorHandle::join`] after [`learner_loop`] returns.
pub struct ActorHandle {
    /// Which actor this handle controls.
    pub actor_id: AgentId,
    join: thread::JoinHandle<Result<ActorStats>>,
    broadcast_tx: Sender<PolicyBroadcast>,
    control_tx: Sender<ControlMessage>,
}

impl ActorHandle {
    /// Send a policy broadcast to this actor. Returns `false` when the
    /// actor has already exited (its receiver is disconnected).
    pub fn send_broadcast(&self, broadcast: PolicyBroadcast) -> bool {
        self.broadcast_tx.send(broadcast).is_ok()
    }

    /// Ask this actor to shut down (best-effort; a dead actor is fine).
    pub fn send_shutdown(&self) {
        let _ = self.control_tx.send(ControlMessage::Shutdown);
    }

    /// Wait for the actor thread to exit and return its stats.
    ///
    /// # Errors
    /// Returns an error when the actor thread panicked or itself
    /// returned an error (e.g. a malformed policy broadcast).
    pub fn join(self) -> Result<ActorStats> {
        self.join.join().map_err(|_| anyhow!("actor thread panicked"))?
    }
}

/// Deserialize a [`crate::multi_agent::PolicyBroadcast`] into `module`.
///
/// The inverse of [`serialize_policy`]: decodes the
/// [`burn::record::BinBytesRecorder`] blob and loads it into the given
/// module (consuming and returning it, per Burn's move-through record
/// API). Backend-agnostic — the blob may have been produced on a
/// different backend with matching element types.
///
/// # Errors
/// Returns an error when the byte blob fails to decode as `M`'s record.
pub fn load_policy_from_broadcast<B2, M>(
    module: M,
    broadcast: &PolicyBroadcast,
    device: &B2::Device,
) -> Result<M>
where
    B2: Backend,
    M: Module<B2>,
{
    match broadcast {
        PolicyBroadcast::Bytes { bytes, version } => {
            let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
            let record =
                <BinBytesRecorder<FullPrecisionSettings> as Recorder<B2>>::load::<M::Record>(
                    &recorder,
                    bytes.as_ref().clone(),
                    device,
                )
                .map_err(|e| anyhow!("failed to load policy broadcast v{version}: {e}"))?;
            Ok(module.load_record(record))
        }
    }
}

/// Serialize the learner policy's autodiff-stripped (`valid()`) view to
/// bytes for broadcasting.
///
/// Uses [`burn::record::BinBytesRecorder`] with
/// [`burn::record::FullPrecisionSettings`], so the blob loads on any
/// backend with matching element types via
/// [`load_policy_from_broadcast`].
///
/// # Errors
/// Returns an error when the recorder fails to encode the record.
pub fn serialize_policy<B, P>(policy: &P) -> Result<Vec<u8>>
where
    B: AutodiffBackend,
    P: AutodiffModule<B>,
{
    let record = policy.valid().into_record();
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    <BinBytesRecorder<FullPrecisionSettings> as Recorder<B::InnerBackend>>::record(
        &recorder,
        record,
        (),
    )
    .map_err(|e| anyhow!("failed to serialize policy: {e}"))
}

/// Body of one inference-only actor thread.
///
/// Steps `env` with the local policy copy, sending one
/// [`crate::multi_agent::Experience`] per step over
/// `channels.experience_tx`, until either a
/// [`crate::multi_agent::ControlMessage::Shutdown`] arrives or the
/// learner drops the experience receiver. Polls
/// `channels.broadcast_rx` non-blocking on every iteration and hot-swaps
/// the local policy when a newer
/// [`crate::multi_agent::PolicyBroadcast`] is available (keeping only
/// the newest when several are queued).
///
/// `act_fn(&policy, observation, rng)` must return the sampled
/// `(action, log_prob, value)` for a single observation — the host-side
/// counterpart of one batch-1 policy forward. `throttle` is the
/// staleness valve described on
/// [`AsyncActorLearnerConfig::max_lead_steps`]
/// ([`ActorThrottle::disabled`] turns it off).
///
/// Most callers should use [`spawn_actor`] instead of calling this
/// directly.
///
/// # Errors
/// Returns an error when a policy broadcast fails to decode.
#[allow(clippy::too_many_arguments)] // mirror of spawn_actor; bundling hides the wiring
pub fn actor_thread<B2, M, E, F>(
    actor_id: AgentId,
    mut env: E,
    mut policy: M,
    channels: ActorChannels,
    device: B2::Device,
    seed: u64,
    throttle: ActorThrottle,
    mut act_fn: F,
) -> Result<ActorStats>
where
    B2: Backend,
    M: Module<B2>,
    E: Environment<Action = i64>,
    F: FnMut(&M, &[f32], &mut StdRng) -> (i64, f32, f32),
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut stats = ActorStats { actor_id, ..Default::default() };

    env.reset();
    loop {
        // 1. Shutdown requested? (Other control messages are best-effort hints the
        //    actor does not act on.)
        match channels.control_rx.try_recv() {
            Ok(ControlMessage::Shutdown) | Err(TryRecvError::Disconnected) => break,
            Ok(_) | Err(TryRecvError::Empty) => {}
        }

        // 2. Non-blocking poll for policy broadcasts; keep only the newest when several
        //    are queued.
        let mut latest: Option<PolicyBroadcast> = None;
        while let Ok(broadcast) = channels.broadcast_rx.try_recv() {
            latest = Some(broadcast);
        }

        // Staleness valve: pause collection until the learner catches up
        // (see ActorThrottle — the budget is cumulative, so an actor can
        // never run more than max_lead_steps ahead of consumption).
        if latest.is_none() && throttle.should_pause(stats.steps_sent, stats.last_policy_version) {
            match channels.broadcast_rx.recv_timeout(ACTOR_BROADCAST_WAIT) {
                Ok(broadcast) => latest = Some(broadcast),
                Err(RecvTimeoutError::Timeout) => continue, // re-check shutdown
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }

        if let Some(broadcast) = latest {
            let version = broadcast.version();
            policy = load_policy_from_broadcast(policy, &broadcast, &device)?;
            stats.policy_updates_received += 1;
            stats.last_policy_version = version;
            tracing::debug!(actor_id, version, "actor loaded policy broadcast");
        }

        // 3. One env step under the local (possibly stale) policy.
        let observation = env.get_observation();
        let (action, log_prob, value) = act_fn(&policy, &observation, &mut rng);
        let result = env.step(action);
        let done = result.terminated || result.truncated;

        let experience = Experience::new(
            actor_id,
            observation,
            vec![action],
            result.reward,
            result.observation,
            result.terminated,
            result.truncated,
            value,
            log_prob,
        );
        if channels.experience_tx.send(experience).is_err() {
            break; // learner is gone; nothing left to do
        }
        stats.steps_sent += 1;

        if done {
            stats.episodes_completed += 1;
            env.reset();
        }
    }

    Ok(stats)
}

/// Spawn one actor thread and return its [`ActorHandle`].
///
/// Creates the per-actor broadcast and control channels internally; the
/// caller supplies a clone of the shared actors→learner experience
/// sender. See [`actor_thread`] for the loop semantics and the
/// `act_fn` contract.
#[allow(clippy::too_many_arguments)] // one slot per channel/knob; a struct would just rename them
pub fn spawn_actor<B2, M, E, F>(
    actor_id: AgentId,
    env: E,
    policy: M,
    experience_tx: Sender<Experience>,
    device: B2::Device,
    seed: u64,
    throttle: ActorThrottle,
    act_fn: F,
) -> ActorHandle
where
    B2: Backend,
    B2::Device: Send + 'static,
    M: Module<B2> + Send + 'static,
    E: Environment<Action = i64> + Send + 'static,
    F: FnMut(&M, &[f32], &mut StdRng) -> (i64, f32, f32) + Send + 'static,
{
    let (broadcast_tx, broadcast_rx) = unbounded();
    let (control_tx, control_rx) = unbounded();
    let channels = ActorChannels { experience_tx, broadcast_rx, control_rx };
    let join = thread::Builder::new()
        .name(format!("thrust-actor-{actor_id}"))
        .spawn(move || {
            actor_thread::<B2, M, E, F>(
                actor_id, env, policy, channels, device, seed, throttle, act_fn,
            )
        })
        .expect("failed to spawn actor thread");
    ActorHandle { actor_id, join, broadcast_tx, control_tx }
}

/// Summary returned by [`learner_loop`] alongside the trained
/// [`crate::train::ppo::trainer::PPOTrainerBurn`].
#[derive(Debug, Clone, Default)]
pub struct LearnerReport {
    /// PPO updates completed.
    pub updates_completed: usize,
    /// Env steps consumed into updates (`num_steps * num_actors` each).
    pub env_steps_consumed: usize,
    /// Episodes completed within the consumed steps.
    pub episodes_completed: usize,
    /// Policy broadcasts sent (one per `broadcast_every` updates).
    pub broadcasts_sent: usize,
    /// Version number of the newest broadcast policy.
    pub last_policy_version: u64,
    /// Total reward of every completed episode, in consumption order.
    pub episode_rewards: Vec<f32>,
    /// Stats from the final PPO update, if any ran.
    pub final_stats: Option<TrainingStats>,
}

impl LearnerReport {
    /// Mean total reward over the most recent `n` completed episodes
    /// (or all of them when fewer have completed). Returns `0.0` before
    /// the first episode completes.
    pub fn mean_recent_episode_reward(&self, n: usize) -> f32 {
        if self.episode_rewards.is_empty() {
            return 0.0;
        }
        let start = self.episode_rewards.len().saturating_sub(n);
        let recent = &self.episode_rewards[start..];
        recent.iter().sum::<f32>() / recent.len() as f32
    }
}

/// Run the learner side of the asynchronous actor-learner loop.
///
/// Blocks on `experience_rx`, filling a `[num_steps, num_actors]`
/// [`crate::buffer::rollout::RolloutBuffer`] (buffer column =
/// `experience.agent_id`). When every actor column holds `num_steps`
/// transitions, it computes advantages — GAE
/// ([`crate::buffer::rollout::compute_advantages`]) by default, or
/// V-trace ([`crate::buffer::rollout::compute_vtrace_advantages`]) when
/// [`AsyncActorLearnerConfig::use_vtrace`] is set — runs one
/// [`crate::train::ppo::trainer::PPOTrainerBurn::train_step`], and — every
/// [`AsyncActorLearnerConfig::broadcast_every`] updates — serializes the
/// refreshed policy and sends a
/// [`crate::multi_agent::PolicyBroadcast`] to every actor. Experiences
/// arriving beyond an update's quota stay queued for the next update, so
/// nothing an actor sends is dropped.
///
/// On completion the learner sends
/// [`crate::multi_agent::ControlMessage::Shutdown`] to every actor and
/// returns the trainer (unchanged ownership model: the caller gets it
/// back) plus a [`LearnerReport`].
///
/// The two closures mirror
/// [`crate::train::ppo::trainer::PPOTrainerBurn::train_step`]'s
/// `evaluate_fn` pattern so the loop stays generic over the policy
/// module:
/// - `evaluate_fn(&policy, obs, actions) -> (log_probs, entropy, values)`
/// - `value_fn(&policy, obs) -> host values` (bootstrap `V(s_T)` for GAE)
///
/// # Errors
/// Returns an error when the experience channel starves for 60 seconds
/// (actor death is fatal in Phase 2 — see the module docs), when an
/// experience carries an out-of-range `agent_id`, or when a
/// `train_step` / policy serialization fails.
#[allow(clippy::too_many_arguments)] // trainer + channels + two closures; same shape as train_step
pub fn learner_loop<B, P, O, F, G>(
    config: &AsyncActorLearnerConfig,
    mut trainer: PPOTrainerBurn<B, P, O>,
    obs_dim: usize,
    device: &B::Device,
    experience_rx: &Receiver<Experience>,
    actors: &[ActorHandle],
    mut evaluate_fn: F,
    mut value_fn: G,
) -> Result<(PPOTrainerBurn<B, P, O>, LearnerReport)>
where
    B: AutodiffBackend,
    P: AutodiffModule<B> + Clone,
    O: Optimizer<P, B>,
    F: FnMut(&P, Tensor<B, 2>, Tensor<B, 1, Int>) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>),
    G: FnMut(&P, Tensor<B, 2>) -> Vec<f32>,
{
    config.validate()?;
    let num_actors = config.num_actors;
    let num_steps = config.num_steps;
    let num_updates = config.num_updates();

    let mut report = LearnerReport::default();
    // Per-actor FIFO of experiences not yet consumed into an update.
    let mut pending: Vec<VecDeque<Experience>> = vec![VecDeque::new(); num_actors];
    // Per-actor running reward of the in-flight episode.
    let mut running_reward = vec![0.0_f32; num_actors];
    let mut version: u64 = 0;

    for update in 0..num_updates {
        // --- Fill: block until every actor column has num_steps rows ---
        while pending.iter().any(|q| q.len() < num_steps) {
            let experience = experience_rx.recv_timeout(LEARNER_RECV_TIMEOUT).map_err(|e| {
                anyhow!(
                    "learner starved waiting for experiences ({e}); \
                     actor death is fatal in Phase 2 (see module docs)"
                )
            })?;
            let actor = experience.agent_id;
            if actor >= num_actors {
                return Err(anyhow!(
                    "experience agent_id {actor} out of range (num_actors = {num_actors})"
                ));
            }
            pending[actor].push_back(experience);
        }

        // --- Drain num_steps rows per actor into the rollout buffer ---
        let mut buffer = RolloutBuffer::new(num_steps, num_actors, obs_dim);
        let mut last_next_obs = vec![0.0_f32; num_actors * obs_dim];
        for actor in 0..num_actors {
            for step in 0..num_steps {
                let exp = pending[actor].pop_front().expect("fill loop guarantees num_steps");
                debug_assert_eq!(exp.agent_id, actor);
                buffer.add(
                    step,
                    actor,
                    &exp.observation,
                    exp.action[0],
                    exp.reward,
                    exp.value,
                    exp.log_prob,
                    exp.terminated,
                    exp.truncated,
                );
                running_reward[actor] += exp.reward;
                if exp.is_done() {
                    report.episode_rewards.push(running_reward[actor]);
                    report.episodes_completed += 1;
                    running_reward[actor] = 0.0;
                    trainer.increment_episodes(1);
                }
                if step == num_steps - 1 {
                    last_next_obs[actor * obs_dim..(actor + 1) * obs_dim]
                        .copy_from_slice(&exp.next_observation);
                }
            }
        }
        report.env_steps_consumed += num_steps * num_actors;

        // --- Advantage estimation (bootstrap from V(s_T) under the
        //     current policy). GAE by default; V-trace when enabled, which
        //     corrects for the actors' policy lag off-policy. ---
        let last_obs_t = Tensor::<B, 2>::from_data(
            TensorData::new(last_next_obs, [num_actors, obs_dim]),
            device,
        );
        let last_values = value_fn(trainer.policy(), last_obs_t);
        if config.use_vtrace {
            // Re-evaluate the stored (obs, action) pairs under the CURRENT
            // (fresh) learner policy to obtain target log-probs
            // `log π(a_t|s_t)`. The buffer's stored `log_probs()` are the
            // actor's behavior log-probs `log μ(a_t|s_t)`, possibly from a
            // stale policy version; the importance ratio between them is
            // what V-trace uses to correct the advantages.
            let batch = RolloutBatch::from_buffer(&buffer);
            let flat_len = num_steps * num_actors;
            let obs_t = Tensor::<B, 2>::from_data(
                TensorData::new(batch.observations.clone(), [flat_len, obs_dim]),
                device,
            );
            let actions_t = Tensor::<B, 1, Int>::from_data(
                TensorData::new(batch.actions.clone(), [flat_len]),
                device,
            );
            let (flat_lps, _, _) = evaluate_fn(trainer.policy(), obs_t, actions_t);
            let flat_target: Vec<f32> = flat_lps.into_data().to_vec().map_err(|e| {
                anyhow!("failed to read V-trace target log-probs off the device: {e:?}")
            })?;
            // RolloutBatch flattens the buffer step-major (index =
            // step * num_actors + actor); reshape back to
            // [num_steps][num_actors] for compute_vtrace_advantages.
            let target_log_probs: Vec<Vec<f32>> = (0..num_steps)
                .map(|step| flat_target[step * num_actors..(step + 1) * num_actors].to_vec())
                .collect();
            buffer.compute_vtrace_advantages(
                &target_log_probs,
                &last_values,
                config.gamma,
                config.vtrace_rho_bar,
                config.vtrace_c_bar,
            );
        } else {
            compute_advantages(&mut buffer, &last_values, config.gamma, config.gae_lambda);
        }

        // --- PPO update (trainer unchanged; actors wrap around it) ---
        let tensors = RolloutBatch::from_buffer(&buffer).to_burn_tensors::<B>(device);
        let stats = trainer.train_step(
            tensors.observations,
            tensors.actions,
            tensors.old_log_probs,
            tensors.old_values,
            tensors.advantages,
            tensors.returns,
            &mut evaluate_fn,
        )?;
        report.updates_completed += 1;

        // --- Broadcast refreshed policy to actors ---
        if (update + 1) % config.broadcast_every == 0 {
            version += 1;
            let bytes = Arc::new(serialize_policy(trainer.policy())?);
            let mut delivered = 0usize;
            for handle in actors {
                if handle
                    .send_broadcast(PolicyBroadcast::Bytes { version, bytes: Arc::clone(&bytes) })
                {
                    delivered += 1;
                }
            }
            report.broadcasts_sent += 1;
            report.last_policy_version = version;
            tracing::info!(
                version,
                delivered,
                num_actors,
                "learner broadcast policy version {version} to {delivered}/{num_actors} actors"
            );
        }

        tracing::info!(
            "update {:>3}/{}  env_steps={:>7}  episodes={:>4}  mean_ep_reward(last≤100)={:6.1}  policy_loss={:7.4}  entropy={:5.3}",
            update + 1,
            num_updates,
            report.env_steps_consumed,
            report.episodes_completed,
            report.mean_recent_episode_reward(100),
            stats.policy_loss,
            stats.entropy,
        );
        report.final_stats = Some(stats);
    }

    for handle in actors {
        handle.send_shutdown();
    }

    Ok((trainer, report))
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, NdArray},
        optim::AdamConfig,
    };

    use super::*;
    use crate::{
        env::{SpaceInfo, SpaceType, StepInfo, StepResult},
        policy::mlp::{MlpBurnConfig, MlpBurnPolicy},
        train::{optimizer::BurnOptimizer, ppo::PPOConfig},
    };

    type B = Autodiff<NdArray<f32>>;
    type Inner = NdArray<f32>;

    const OBS_DIM: usize = 2;
    const ACTION_DIM: usize = 2;

    /// Deterministic stub env: observation is `[t, t]` (step counter),
    /// reward +1/step, episode terminates every 5 steps.
    struct StubEnv {
        t: usize,
    }

    impl Environment for StubEnv {
        type Action = i64;
        type State = usize;

        fn reset(&mut self) {
            self.t = 0;
        }
        fn get_observation(&self) -> Vec<f32> {
            vec![self.t as f32; OBS_DIM]
        }
        fn step(&mut self, _action: i64) -> StepResult {
            self.t += 1;
            StepResult {
                observation: self.get_observation(),
                reward: 1.0,
                terminated: self.t.is_multiple_of(5),
                truncated: false,
                info: StepInfo::default(),
            }
        }
        fn observation_space(&self) -> SpaceInfo {
            SpaceInfo { shape: vec![OBS_DIM], space_type: SpaceType::Box }
        }
        fn action_space(&self) -> SpaceInfo {
            SpaceInfo { shape: vec![1], space_type: SpaceType::Discrete(ACTION_DIM) }
        }
        fn render(&self) -> Vec<u8> {
            Vec::new()
        }
        fn close(&mut self) {}
        fn clone_state(&self) -> usize {
            self.t
        }
        fn restore_state(&mut self, state: &usize) {
            self.t = *state;
        }
    }

    fn seeded_inner_policy(seed: u64) -> MlpBurnPolicy<Inner> {
        let device = Default::default();
        MlpBurnPolicy::<Inner>::with_config(
            OBS_DIM,
            ACTION_DIM,
            MlpBurnConfig { hidden_dim: 8, ..Default::default() }.with_seed(seed),
            &device,
        )
    }

    fn seeded_autodiff_policy(seed: u64) -> MlpBurnPolicy<B> {
        let device = Default::default();
        MlpBurnPolicy::<B>::with_config(
            OBS_DIM,
            ACTION_DIM,
            MlpBurnConfig { hidden_dim: 8, ..Default::default() }.with_seed(seed),
            &device,
        )
    }

    fn act_fn(policy: &MlpBurnPolicy<Inner>, obs: &[f32], rng: &mut StdRng) -> (i64, f32, f32) {
        let device = Default::default();
        let t =
            Tensor::<Inner, 2>::from_data(TensorData::new(obs.to_vec(), [1, obs.len()]), &device);
        let (actions, log_probs, values) = policy.get_action_host_seeded(t, rng);
        (actions[0], log_probs[0], values[0])
    }

    #[test]
    fn config_default_is_valid_and_derives_updates() {
        let config = AsyncActorLearnerConfig::default();
        config.validate().unwrap();
        assert_eq!(config.broadcast_every, 1);
        assert_eq!(config.num_updates(), 200_000 / (256 * 4));
        assert_eq!(config.effective_max_lead_steps(), 2 * 256);
        // V-trace is off by default; clip thresholds default to the
        // IMPALA values so the GAE path is entirely unchanged.
        assert!(!config.use_vtrace);
        assert_eq!(config.vtrace_rho_bar, 1.0);
        assert_eq!(config.vtrace_c_bar, 1.0);

        let throttle = config.actor_throttle();
        assert_eq!(throttle.steps_per_broadcast, 256);
        assert_eq!(throttle.max_lead_steps, 512);
        // Lead budget is cumulative: version v proves v * 256 consumed.
        assert!(!throttle.should_pause(511, 0));
        assert!(throttle.should_pause(512, 0));
        assert!(!throttle.should_pause(512, 1)); // v1 ⇒ budget now 768
        assert!(throttle.should_pause(768, 1));
        assert!(!ActorThrottle::disabled().should_pause(usize::MAX - 1, 0));
    }

    #[test]
    fn config_rejects_bad_values() {
        let base = AsyncActorLearnerConfig::default();
        assert!(AsyncActorLearnerConfig { num_actors: 0, ..base.clone() }.validate().is_err());
        assert!(AsyncActorLearnerConfig { num_steps: 0, ..base.clone() }.validate().is_err());
        assert!(
            AsyncActorLearnerConfig { broadcast_every: 0, ..base.clone() }
                .validate()
                .is_err()
        );
        assert!(
            AsyncActorLearnerConfig { total_env_steps: 10, ..base.clone() }
                .validate()
                .is_err()
        );
        assert!(AsyncActorLearnerConfig { gamma: 0.0, ..base.clone() }.validate().is_err());
        assert!(AsyncActorLearnerConfig { gae_lambda: 1.5, ..base.clone() }.validate().is_err());
        // Throttle tighter than one broadcast cycle would deadlock.
        assert!(AsyncActorLearnerConfig { max_lead_steps: 255, ..base }.validate().is_err());
    }

    #[test]
    fn config_vtrace_fields_validate() {
        let base = AsyncActorLearnerConfig::default();
        // Enabling V-trace with the default (1.0) clips validates.
        assert!(AsyncActorLearnerConfig { use_vtrace: true, ..base.clone() }.validate().is_ok());
        // Non-positive clip thresholds are rejected regardless of use_vtrace.
        assert!(
            AsyncActorLearnerConfig { vtrace_rho_bar: 0.0, ..base.clone() }
                .validate()
                .is_err()
        );
        assert!(
            AsyncActorLearnerConfig { vtrace_rho_bar: -1.0, ..base.clone() }
                .validate()
                .is_err()
        );
        assert!(
            AsyncActorLearnerConfig { vtrace_c_bar: 0.0, ..base.clone() }
                .validate()
                .is_err()
        );
        assert!(
            AsyncActorLearnerConfig { vtrace_c_bar: -1.0, ..base.clone() }
                .validate()
                .is_err()
        );
        // Builder-style setters compose and produce a valid config.
        let cfg = base.use_vtrace(true).vtrace_rho_bar(0.9).vtrace_c_bar(1.1);
        assert!(cfg.use_vtrace);
        assert_eq!(cfg.vtrace_rho_bar, 0.9);
        assert_eq!(cfg.vtrace_c_bar, 1.1);
        cfg.validate().unwrap();
    }

    /// Serialize on the autodiff backend, load on the inner backend, and
    /// confirm the loaded policy computes the same forward pass as the
    /// source policy's `valid()` view.
    #[test]
    fn policy_broadcast_bytes_roundtrip() {
        let device = Default::default();
        let source = seeded_autodiff_policy(42);
        let target = seeded_inner_policy(43); // different weights on purpose

        let bytes = serialize_policy(&source).unwrap();
        let broadcast = PolicyBroadcast::Bytes { version: 1, bytes: Arc::new(bytes) };
        let loaded = load_policy_from_broadcast(target, &broadcast, &device).unwrap();

        let obs =
            Tensor::<Inner, 2>::from_data(TensorData::new(vec![0.1, -0.2], [1, OBS_DIM]), &device);
        let (logits_src, value_src) = source.valid().forward(obs.clone());
        let (logits_loaded, value_loaded) = loaded.forward(obs);

        let src: Vec<f32> = logits_src.into_data().to_vec().unwrap();
        let got: Vec<f32> = logits_loaded.into_data().to_vec().unwrap();
        assert_eq!(src, got, "loaded policy must reproduce source logits bit-for-bit");
        let v_src: Vec<f32> = value_src.into_data().to_vec().unwrap();
        let v_got: Vec<f32> = value_loaded.into_data().to_vec().unwrap();
        assert_eq!(v_src, v_got);
    }

    /// The actor sends experiences tagged with its actor_id, in step
    /// order, and hot-swaps its policy when a broadcast arrives.
    ///
    /// Broadcast receipt is made deterministic through the throttle: the
    /// actor pauses after `max_lead_steps` (still on version 0) and can
    /// only resume by loading a broadcast, so receiving any experience
    /// beyond the pause point proves the load happened — no sleeps, no
    /// timing sensitivity under a loaded CI machine.
    #[test]
    fn actor_thread_sends_ordered_experiences_and_loads_broadcasts() {
        const PAUSE_AT: usize = 40;

        let device = burn::backend::ndarray::NdArrayDevice::default();
        let (experience_tx, experience_rx) = unbounded();

        let handle = spawn_actor::<Inner, _, _, _>(
            3,
            StubEnv { t: 0 },
            seeded_inner_policy(7),
            experience_tx,
            device,
            123,
            // Version v proves v * PAUSE_AT consumed; at version 0 the
            // actor pauses after exactly PAUSE_AT experiences.
            ActorThrottle { steps_per_broadcast: PAUSE_AT, max_lead_steps: PAUSE_AT },
            act_fn,
        );

        // Collect the pre-pause experiences; they must be ordered and tagged.
        let mut received = Vec::new();
        for _ in 0..PAUSE_AT {
            let exp = experience_rx.recv_timeout(Duration::from_secs(30)).unwrap();
            assert_eq!(exp.agent_id, 3);
            received.push(exp);
        }
        for (i, exp) in received.iter().enumerate() {
            // StubEnv observation is [t, t]; actor visits t = 0,1,2,...
            // (reset restarts the counter at 0 every 5 steps).
            let expected_t = (i % 5) as f32;
            assert_eq!(exp.observation, vec![expected_t; OBS_DIM], "step {i} out of order");
            assert_eq!(exp.action.len(), 1);
            assert_eq!(exp.terminated, (i + 1) % 5 == 0);
        }
        // The actor is now paused at its lead budget: no further
        // experience may arrive until it loads a policy broadcast.

        // Broadcast a new policy; version 7 raises the budget to
        // 7 * PAUSE_AT + PAUSE_AT, so the actor resumes iff it loads it.
        let source = seeded_autodiff_policy(99);
        let bytes = Arc::new(serialize_policy(&source).unwrap());
        assert!(handle.send_broadcast(PolicyBroadcast::Bytes { version: 7, bytes }));

        // Any post-pause experience proves the broadcast was loaded.
        experience_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("actor should resume after loading the broadcast");

        handle.send_shutdown();
        let stats = handle.join().unwrap();

        assert_eq!(stats.actor_id, 3);
        assert!(stats.steps_sent > PAUSE_AT);
        assert!(stats.episodes_completed >= PAUSE_AT / 5);
        assert!(
            stats.policy_updates_received >= 1,
            "actor must have loaded the broadcast (got {})",
            stats.policy_updates_received
        );
        assert_eq!(stats.last_policy_version, 7);
    }

    /// End-to-end wiring smoke test on the stub env: 2 actors, a few
    /// tiny updates, broadcasts delivered, report bookkeeping correct.
    #[test]
    fn learner_loop_runs_updates_and_broadcasts() {
        let device = Default::default();
        let num_actors = 2;
        let num_steps = 8;

        let config = AsyncActorLearnerConfig {
            num_actors,
            num_steps,
            total_env_steps: num_steps * num_actors * 3, // 3 updates
            broadcast_every: 1,
            max_lead_steps: 2 * num_steps,
            gamma: 0.99,
            gae_lambda: 0.95,
            use_vtrace: false,
            vtrace_rho_bar: 1.0,
            vtrace_c_bar: 1.0,
            seed: 0,
        };

        let policy = seeded_autodiff_policy(0);
        let inner_opt = AdamConfig::new().init();
        let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 1e-3);
        let ppo_config = PPOConfig::default().batch_size(8).n_epochs(1).target_kl(1.0);
        let trainer = PPOTrainerBurn::new(ppo_config, policy, burn_opt).unwrap();

        let (experience_tx, experience_rx) = unbounded();
        let actors: Vec<ActorHandle> = (0..num_actors)
            .map(|i| {
                spawn_actor::<Inner, _, _, _>(
                    i,
                    StubEnv { t: 0 },
                    trainer.policy().valid(),
                    experience_tx.clone(),
                    device,
                    100 + i as u64,
                    config.actor_throttle(),
                    act_fn,
                )
            })
            .collect();
        drop(experience_tx);

        let (trainer, report) = learner_loop(
            &config,
            trainer,
            OBS_DIM,
            &device,
            &experience_rx,
            &actors,
            |p: &MlpBurnPolicy<B>, o, a| p.evaluate_actions(o, a),
            |p: &MlpBurnPolicy<B>, o| p.forward(o).1.into_data().to_vec().unwrap(),
        )
        .unwrap();

        assert_eq!(report.updates_completed, 3);
        assert_eq!(report.env_steps_consumed, num_steps * num_actors * 3);
        assert_eq!(report.broadcasts_sent, 3);
        assert_eq!(report.last_policy_version, 3);
        // StubEnv terminates every 5 steps → 16 steps/update/actor ≥ 1 episode.
        assert!(report.episodes_completed >= num_actors);
        assert_eq!(trainer.total_episodes(), report.episodes_completed);
        let stats = report.final_stats.expect("3 updates ran");
        assert!(stats.policy_loss.is_finite());
        assert!(stats.value_loss.is_finite());

        // Every actor must have received at least one broadcast before
        // shutdown (broadcast_every = 1 and 3 updates ran).
        for handle in actors {
            let stats = handle.join().unwrap();
            assert!(
                stats.policy_updates_received >= 1,
                "actor {} never received a policy broadcast",
                stats.actor_id
            );
            assert!(stats.last_policy_version >= 1);
        }
    }

    /// Run exactly one PPO update through [`learner_loop`] and return the
    /// resulting [`TrainingStats`]. The single actor is never stale
    /// (`broadcast_every = 1`), the initial policy and RNG seeds are
    /// fixed, and the stub env is deterministic, so the first
    /// `num_steps` experiences — the only ones consumed — are identical
    /// across calls that differ solely in `use_vtrace`.
    fn run_one_update(use_vtrace: bool) -> TrainingStats {
        let device = Default::default();
        let num_actors = 1;
        let num_steps = 8;

        let config = AsyncActorLearnerConfig {
            num_actors,
            num_steps,
            total_env_steps: num_steps * num_actors, // exactly one update
            broadcast_every: 1,
            max_lead_steps: 2 * num_steps,
            gamma: 0.99,
            gae_lambda: 1.0, // V-trace on-policy reduces to GAE(λ = 1)
            use_vtrace,
            vtrace_rho_bar: 1.0,
            vtrace_c_bar: 1.0,
            seed: 0,
        };

        let policy = seeded_autodiff_policy(0);
        let inner_opt = AdamConfig::new().init();
        let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 1e-3);
        // Full-batch, single-epoch update: no minibatch shuffling, so the
        // only thing that can differ between the two runs is the advantage
        // estimator.
        let ppo_config = PPOConfig::default()
            .batch_size(num_steps * num_actors)
            .n_epochs(1)
            .target_kl(1.0);
        let trainer = PPOTrainerBurn::new(ppo_config, policy, burn_opt).unwrap();

        let (experience_tx, experience_rx) = unbounded();
        let actors: Vec<ActorHandle> = (0..num_actors)
            .map(|i| {
                spawn_actor::<Inner, _, _, _>(
                    i,
                    StubEnv { t: 0 },
                    trainer.policy().valid(),
                    experience_tx.clone(),
                    device,
                    100 + i as u64,
                    config.actor_throttle(),
                    act_fn,
                )
            })
            .collect();
        drop(experience_tx);

        let (_trainer, report) = learner_loop(
            &config,
            trainer,
            OBS_DIM,
            &device,
            &experience_rx,
            &actors,
            |p: &MlpBurnPolicy<B>, o, a| p.evaluate_actions(o, a),
            |p: &MlpBurnPolicy<B>, o| p.forward(o).1.into_data().to_vec().unwrap(),
        )
        .unwrap();
        for handle in actors {
            let _ = handle.join();
        }
        report.final_stats.expect("one update ran")
    }

    /// On-policy identity: with a single never-stale actor the V-trace
    /// path re-evaluates the very policy that produced the actions, so
    /// every importance ratio is 1 and V-trace collapses to GAE(λ = 1).
    /// One update under each path from identical initial conditions must
    /// therefore produce matching PPO losses. This lifts the buffer-level
    /// `vtrace::tests::buffer_on_policy_matches_gae_lambda_one` property
    /// up through the learner-loop wiring (mirrors PR #283).
    #[test]
    fn learner_loop_vtrace_on_policy_matches_gae() {
        let gae = run_one_update(false);
        let vtrace = run_one_update(true);

        eprintln!(
            "on-policy: gae(policy={:.9}, value={:.9}) vtrace(policy={:.9}, value={:.9})",
            gae.policy_loss, gae.value_loss, vtrace.policy_loss, vtrace.value_loss
        );

        // `value_loss = MSE(V(s), returns)` is the strong discriminator:
        // the returns come straight from the advantage estimator with no
        // normalization, so a wiring bug (wrong target log-probs) would
        // move it well beyond the on-policy match. Compare it relatively.
        let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(b.abs()).max(1e-6);
        assert!(
            rel(gae.value_loss, vtrace.value_loss) < 1e-2,
            "value_loss should match on-policy: gae={} vtrace={}",
            gae.value_loss,
            vtrace.value_loss
        );
        // `policy_loss` is ~0 by construction on a single full-batch epoch
        // (unit importance ratio × zero-mean normalized advantages), so it
        // is only a sanity floor; compare it absolutely.
        assert!(
            (gae.policy_loss - vtrace.policy_loss).abs() < 1e-3,
            "policy_loss should match on-policy: gae={} vtrace={}",
            gae.policy_loss,
            vtrace.policy_loss
        );
    }

    /// V-trace under real staleness: with 2 actors and a broadcast only
    /// every 3 updates, actors collect several rollouts under a policy the
    /// learner has already moved past. The re-evaluated target log-probs
    /// then differ from the stored behavior log-probs, exercising
    /// non-trivial importance ratios. The loop must complete without
    /// panicking and every reported statistic must be finite.
    #[test]
    fn learner_loop_vtrace_stale_completes() {
        let device = Default::default();
        let num_actors = 2;
        let num_steps = 8;

        let config = AsyncActorLearnerConfig {
            num_actors,
            num_steps,
            total_env_steps: num_steps * num_actors * 5, // 5 updates
            broadcast_every: 3,                          // actors run stale between broadcasts
            max_lead_steps: 6 * num_steps,               // room to run ahead under a stale policy
            gamma: 0.99,
            gae_lambda: 0.95,
            use_vtrace: true,
            vtrace_rho_bar: 1.0,
            vtrace_c_bar: 1.0,
            seed: 0,
        };

        let policy = seeded_autodiff_policy(0);
        let inner_opt = AdamConfig::new().init();
        let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 1e-3);
        let ppo_config = PPOConfig::default().batch_size(8).n_epochs(1).target_kl(1.0);
        let trainer = PPOTrainerBurn::new(ppo_config, policy, burn_opt).unwrap();

        let (experience_tx, experience_rx) = unbounded();
        let actors: Vec<ActorHandle> = (0..num_actors)
            .map(|i| {
                spawn_actor::<Inner, _, _, _>(
                    i,
                    StubEnv { t: 0 },
                    trainer.policy().valid(),
                    experience_tx.clone(),
                    device,
                    200 + i as u64,
                    config.actor_throttle(),
                    act_fn,
                )
            })
            .collect();
        drop(experience_tx);

        let (_trainer, report) = learner_loop(
            &config,
            trainer,
            OBS_DIM,
            &device,
            &experience_rx,
            &actors,
            |p: &MlpBurnPolicy<B>, o, a| p.evaluate_actions(o, a),
            |p: &MlpBurnPolicy<B>, o| p.forward(o).1.into_data().to_vec().unwrap(),
        )
        .unwrap();
        for handle in actors {
            let _ = handle.join();
        }

        assert_eq!(report.updates_completed, 5);
        let stats = report.final_stats.expect("5 updates ran");
        assert!(stats.policy_loss.is_finite(), "policy_loss must be finite under staleness");
        assert!(stats.value_loss.is_finite(), "value_loss must be finite under staleness");
        assert!(stats.entropy.is_finite(), "entropy must be finite under staleness");
        for r in &report.episode_rewards {
            assert!(r.is_finite(), "episode reward must be finite");
        }
    }

    /// Provenance / non-triviality of the IS correction: the log-probs
    /// the learner re-evaluates under an *updated* policy differ from the
    /// behavior log-probs an actor stored under the policy that generated
    /// the actions. If they were equal the V-trace ratio would collapse
    /// to 1 and the correction would be a no-op; this guards against a
    /// wiring bug that feeds the stored behavior log-probs back in as the
    /// target (which is exactly what `buffer.log_probs()` holds).
    #[test]
    fn vtrace_target_log_probs_differ_from_behavior_when_policy_updated() {
        let device = Default::default();
        // Behavior policy (what an actor used) vs. a different "updated"
        // policy the learner has since moved to.
        let behavior = seeded_autodiff_policy(0);
        let updated = seeded_autodiff_policy(7);

        // Two [t, t] observations and the actions an actor sampled.
        let obs = vec![0.0_f32, 0.0, 1.0, 1.0];
        let obs_t = Tensor::<B, 2>::from_data(TensorData::new(obs, [2, OBS_DIM]), &device);
        let actions_t =
            Tensor::<B, 1, Int>::from_data(TensorData::new(vec![0_i64, 1], [2]), &device);

        let (behavior_lp, _, _) = behavior.evaluate_actions(obs_t.clone(), actions_t.clone());
        let (updated_lp, _, _) = updated.evaluate_actions(obs_t, actions_t);
        let b: Vec<f32> = behavior_lp.into_data().to_vec().unwrap();
        let u: Vec<f32> = updated_lp.into_data().to_vec().unwrap();

        assert!(
            b.iter().zip(&u).any(|(x, y)| (x - y).abs() > 1e-3),
            "target log-probs under an updated policy must differ from behavior log-probs: \
             {b:?} vs {u:?}"
        );
    }
}
