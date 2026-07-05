//! Fixed-vocab agent-to-agent communication (Phase 1 of epic #266).
//!
//! This module introduces discrete, fixed-vocabulary comms as a *non-breaking,
//! reusable* capability layered on top of the existing
//! [`crate::multi_agent::environment::MultiAgentEnvironment`] surface. It
//! follows the design note
//! [`docs/COMMS_DESIGN.md`](../../../docs/COMMS_DESIGN.md), section 5 (the
//! authoritative phase breakdown).
//!
//! # The action-space-extension model
//!
//! A message is just **extra action dims on the sender** and **extra
//! observation dims on the receiver** — exactly the pattern the Bucket Brigade
//! env proves end-to-end with its 1-bit `signal` channel
//! (`src/env/games/bucket_brigade/env.rs`, read-only reference here).
//! Concretely, an agent's action vector is laid out as
//!
//! ```text
//! [ ...task action dims (agent_action_space)... , ...message dims (message_vocab)... ]
//! ```
//!
//! The env slices the message dims off inside `step_multi`
//! ([`split_action`](crate::multi_agent::comms::split_action)), applies
//! [`Delivery`], and writes the received tokens into the observation layout it
//! already controls
//! ([`place_message`](crate::multi_agent::comms::place_message)). No change to
//! [`crate::multi_agent::environment::MultiAgentResult`] is required for Phase
//! 1; a first-class `messages` field is deliberately deferred to Phase 3.
//!
//! # Non-breaking by construction
//!
//! [`CommunicatingEnvironment`] is a **supertrait** of `MultiAgentEnvironment`
//! whose methods are all defaulted to describe a zero-width channel
//! (`message_vocab -> []`, `message_obs_size -> 0`, `delivery -> Broadcast`).
//! Existing implementors (snake, matching pennies, bucket brigade, the joint
//! env) satisfy it for free without any code change.
//!
//! # Host-payload discipline
//!
//! Like [`crate::multi_agent::messages`], everything here is plain
//! `Vec<f32>` / `Vec<i64>` host data — no tensor types cross this surface, so
//! producer and consumer stay free to pick different Burn backends.

use crate::multi_agent::{environment::MultiAgentEnvironment, messages::AgentId};

/// A message emitted by one agent for delivery to others within a rollout.
///
/// Host-side payload only (no tensor types), consistent with the rest of the
/// multi-agent message surface. Discrete tokens live in
/// [`AgentMessage::tokens`]; an optional continuous payload
/// ([`AgentMessage::embedding`]) reserves space for the differentiable/learned
/// path (Phase 3) and is empty for fixed-vocab discrete comms.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentMessage {
    /// Sending agent.
    pub sender: AgentId,

    /// Discrete message tokens (multi-discrete, one entry per message dim).
    /// Cardinalities are published by
    /// [`CommunicatingEnvironment::message_vocab`].
    pub tokens: Vec<i64>,

    /// Optional continuous payload (relaxed / learned comms, Phase 3). Empty
    /// for fixed-vocab discrete comms.
    pub embedding: Vec<f32>,
}

impl AgentMessage {
    /// Construct a discrete (fixed-vocab) message from `sender` carrying
    /// `tokens`. The continuous [`AgentMessage::embedding`] is left empty.
    pub fn discrete(sender: AgentId, tokens: Vec<i64>) -> Self {
        Self { sender, tokens, embedding: Vec::new() }
    }
}

/// Delivery routing for messages emitted in a single step.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Delivery {
    /// Delivered to every *other* agent (the Bucket Brigade `signal` model).
    #[default]
    Broadcast,

    /// Delivered only to the listed recipients.
    Targeted(Vec<AgentId>),
}

impl Delivery {
    /// Whether a message routed under this policy reaches `recipient`, given
    /// the `sender`. A sender never receives its own broadcast.
    pub fn reaches(&self, sender: AgentId, recipient: AgentId) -> bool {
        match self {
            Delivery::Broadcast => sender != recipient,
            Delivery::Targeted(ids) => ids.contains(&recipient),
        }
    }
}

/// Opt-in extension of [`MultiAgentEnvironment`] for environments that expose a
/// first-class agent-to-agent message channel.
///
/// Every method is defaulted to describe a *non-communicating* environment, so
/// this is a **non-breaking** supertrait: existing implementors get a
/// zero-width channel for free and keep compiling unchanged. Comms-aware envs
/// override the methods to publish their message layout.
pub trait CommunicatingEnvironment: MultiAgentEnvironment {
    /// Per-agent message action-space layout (one bin count per message dim),
    /// analogous to
    /// [`MultiAgentEnvironment::agent_action_space`]. These dims are appended
    /// *after* the task action dims in each agent's action vector. An empty vec
    /// means the agent sends nothing.
    fn message_vocab(&self, _agent_id: AgentId) -> Vec<usize> {
        Vec::new()
    }

    /// Number of observation dims that carry *received* messages for this
    /// agent. Zero means the agent receives nothing. The env is responsible
    /// for placing received messages into the observation vector returned
    /// by `get_agent_observation` / `step_multi` (see [`place_message`]).
    fn message_obs_size(&self, _agent_id: AgentId) -> usize {
        0
    }

    /// Delivery policy for messages emitted this step. Defaults to broadcast,
    /// matching the Bucket Brigade `signal` semantics.
    fn delivery(&self) -> Delivery {
        Delivery::Broadcast
    }
}

/// Split a flat action vector into its `(task, message)` halves at `task_len`.
///
/// The action layout is `[ ...task dims... , ...message dims... ]`, so the
/// first `task_len` entries are the task action and the remainder are the
/// message tokens. This factors the Bucket-Brigade-style slicing so every
/// [`CommunicatingEnvironment`] shares one implementation.
///
/// Boundary behavior:
/// - `task_len == 0` → task slice is empty, message slice is the whole action.
/// - `task_len == action.len()` → task slice is the whole action, message slice
///   is empty (an agent with no message dims).
///
/// # Panics
///
/// Panics if `task_len > action.len()` — that indicates a mismatch between the
/// env's declared `agent_action_space` and the action vector it was handed.
///
/// # Examples
///
/// ```
/// use thrust_rl::multi_agent::comms::split_action;
///
/// // Bucket-Brigade-style [house, mode, signal] with a 1-dim message tail.
/// let action = [7, 1, 0];
/// let (task, message) = split_action(&action, 2);
/// assert_eq!(task, &[7, 1]);
/// assert_eq!(message, &[0]);
/// ```
pub fn split_action(action: &[i64], task_len: usize) -> (&[i64], &[i64]) {
    assert!(
        task_len <= action.len(),
        "split_action: task_len ({task_len}) exceeds action length ({})",
        action.len()
    );
    action.split_at(task_len)
}

/// Write received message `tokens` into an observation slice starting at
/// `offset`, one f32 per token.
///
/// This is the receiver-side counterpart to [`split_action()`]: after slicing a
/// sender's message off its action, the env lays those tokens into the
/// receiver's observation vector (the region the env reserves via
/// [`CommunicatingEnvironment::message_obs_size`]). Tokens are written verbatim
/// as `f32`, matching the flat host-vector discipline of the observation
/// surface.
///
/// # Panics
///
/// Panics if the destination region `obs[offset..offset + tokens.len()]` would
/// run past the end of `obs`, which indicates a `message_obs_size` / layout
/// mismatch.
///
/// # Examples
///
/// ```
/// use thrust_rl::multi_agent::comms::place_message;
///
/// let mut obs = vec![0.0_f32; 4];
/// place_message(&mut obs, 1, &[3, 2]);
/// assert_eq!(obs, vec![0.0, 3.0, 2.0, 0.0]);
/// ```
pub fn place_message(obs: &mut [f32], offset: usize, tokens: &[i64]) {
    let end = offset + tokens.len();
    assert!(
        end <= obs.len(),
        "place_message: region [{offset}..{end}) exceeds observation length ({})",
        obs.len()
    );
    for (slot, &tok) in obs[offset..end].iter_mut().zip(tokens) {
        *slot = tok as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult},
        multi_agent::environment::MultiAgentResult,
    };

    /// Minimal `MultiAgentEnvironment` that does **not** override any comms
    /// method — used to prove the zero-width default channel.
    struct NoCommsEnv;

    impl Environment for NoCommsEnv {
        type Action = i64;
        type State = ();
        fn reset(&mut self) {}
        fn get_observation(&self) -> Vec<f32> {
            vec![0.0]
        }
        fn step(&mut self, _action: i64) -> StepResult {
            StepResult {
                observation: vec![0.0],
                reward: 0.0,
                terminated: false,
                truncated: false,
                info: StepInfo::default(),
            }
        }
        fn observation_space(&self) -> SpaceInfo {
            SpaceInfo { shape: vec![1], space_type: SpaceType::Box }
        }
        fn action_space(&self) -> SpaceInfo {
            SpaceInfo { shape: vec![], space_type: SpaceType::Discrete(2) }
        }
        fn render(&self) -> Vec<u8> {
            Vec::new()
        }
        fn close(&mut self) {}
        fn clone_state(&self) {}
        fn restore_state(&mut self, _state: &()) {}
    }

    impl MultiAgentEnvironment for NoCommsEnv {
        fn num_agents(&self) -> usize {
            2
        }
        fn get_agent_observation(&self, _agent_id: usize) -> Vec<f32> {
            vec![0.0]
        }
        fn agent_action_space(&self, _agent_id: usize) -> Vec<usize> {
            vec![2]
        }
        fn step_multi(&mut self, _actions: &[Vec<i64>]) -> MultiAgentResult {
            MultiAgentResult::new(vec![vec![0.0]; 2], vec![0.0; 2], vec![false; 2], vec![false; 2])
        }
        fn active_agents(&self) -> Vec<bool> {
            vec![true; 2]
        }
    }

    // The whole point of the supertrait: opting in with zero overrides.
    impl CommunicatingEnvironment for NoCommsEnv {}

    #[test]
    fn default_impls_report_zero_width_channel() {
        let env = NoCommsEnv;
        for agent in 0..env.num_agents() {
            assert_eq!(env.message_vocab(agent), Vec::<usize>::new());
            assert_eq!(env.message_obs_size(agent), 0);
        }
        assert_eq!(env.delivery(), Delivery::Broadcast);
    }

    #[test]
    fn split_action_at_interior_boundary() {
        let action = [7, 1, 0];
        let (task, message) = split_action(&action, 2);
        assert_eq!(task, &[7, 1]);
        assert_eq!(message, &[0]);
    }

    #[test]
    fn split_action_zero_message_dims() {
        // task_len == action.len(): the whole thing is task, message is empty.
        let action = [4, 2];
        let (task, message) = split_action(&action, action.len());
        assert_eq!(task, &[4, 2]);
        assert_eq!(message, &[] as &[i64]);
    }

    #[test]
    fn split_action_all_message_dims() {
        // task_len == 0: a pure sender with no task action.
        let action = [5];
        let (task, message) = split_action(&action, 0);
        assert_eq!(task, &[] as &[i64]);
        assert_eq!(message, &[5]);
    }

    #[test]
    fn split_action_empty_action() {
        let action: [i64; 0] = [];
        let (task, message) = split_action(&action, 0);
        assert!(task.is_empty());
        assert!(message.is_empty());
    }

    #[test]
    #[should_panic(expected = "exceeds action length")]
    fn split_action_task_len_overflow_panics() {
        let action = [1, 2];
        let _ = split_action(&action, 3);
    }

    #[test]
    fn place_message_writes_tokens_at_offset() {
        let mut obs = vec![0.0_f32; 4];
        place_message(&mut obs, 1, &[3, 2]);
        assert_eq!(obs, vec![0.0, 3.0, 2.0, 0.0]);
    }

    #[test]
    fn place_message_empty_tokens_is_noop() {
        let mut obs = vec![9.0_f32; 3];
        place_message(&mut obs, 2, &[]);
        assert_eq!(obs, vec![9.0, 9.0, 9.0]);
    }

    #[test]
    #[should_panic(expected = "exceeds observation length")]
    fn place_message_out_of_range_panics() {
        let mut obs = vec![0.0_f32; 2];
        place_message(&mut obs, 1, &[1, 2]);
    }

    #[test]
    fn delivery_broadcast_skips_sender() {
        let d = Delivery::Broadcast;
        assert!(!d.reaches(0, 0));
        assert!(d.reaches(0, 1));
    }

    #[test]
    fn delivery_targeted_routes_to_listed() {
        let d = Delivery::Targeted(vec![2, 3]);
        assert!(d.reaches(0, 2));
        assert!(d.reaches(0, 3));
        assert!(!d.reaches(0, 1));
    }

    #[test]
    fn agent_message_discrete_has_empty_embedding() {
        let m = AgentMessage::discrete(1, vec![4]);
        assert_eq!(m.sender, 1);
        assert_eq!(m.tokens, vec![4]);
        assert!(m.embedding.is_empty());
    }
}
