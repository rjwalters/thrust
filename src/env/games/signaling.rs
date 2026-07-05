//! Referential signaling game — reference `CommunicatingEnvironment` (issue #274).
//!
//! The smallest environment that exercises the fixed-vocab comms surface from
//! [`crate::multi_agent::comms`] end-to-end. It is the Phase 1 reference impl
//! called out by [`docs/COMMS_DESIGN.md`](../../../../docs/COMMS_DESIGN.md)
//! section 5: "one small reference env implementing `CommunicatingEnvironment`
//! (e.g. a 2-agent referential signaling game)".
//!
//! # The game
//!
//! Two agents, a fixed vocabulary of `V` tokens:
//!
//! - **Agent 0 (speaker)** observes a hidden token `t ∈ 0..V` (one-hot) and has
//!   *no task action* — its only action dim is a **message token** it emits.
//! - **Agent 1 (listener)** sends nothing. Its observation is the message token
//!   it received, and its *task action* is a guess `g ∈ 0..V`.
//!
//! Both agents earn `+1` when the listener's guess matches the hidden token and
//! `0` otherwise, so the speaker is incentivized to transmit `t` faithfully. This
//! is the canonical Lewis signaling game.
//!
//! # Comms wiring (the point of this env)
//!
//! | agent | `agent_action_space` | `message_vocab` | full action | `message_obs_size` |
//! |-------|----------------------|-----------------|-------------|--------------------|
//! | 0 speaker | `[]` (no task action) | `[V]` | `[token]` | `0` |
//! | 1 listener | `[V]` (the guess) | `[]` | `[guess]` | `1` (received token) |
//!
//! Inside [`SignalingGame::step_multi`] the speaker's message dim is sliced off
//! with [`crate::multi_agent::comms::split_action`], routed by broadcast, and
//! written into the listener's observation with
//! [`crate::multi_agent::comms::place_message`]. No new `MultiAgentResult`
//! machinery is involved — the message rides inline in the action / observation
//! vectors, exactly as Phase 1 prescribes.
//!
//! # Determinism
//!
//! The env consumes no internal RNG: the hidden token is fixed at construction
//! (or by [`SignalingGame::set_hidden`]) and unchanged by stepping, so
//! [`Environment::clone_state`] / [`Environment::restore_state`] reproduce every
//! subsequent step exactly.

use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};
use crate::multi_agent::comms::{place_message, split_action, CommunicatingEnvironment, Delivery};
use crate::multi_agent::environment::{MultiAgentEnvironment, MultiAgentResult};

/// Agent id of the speaker (observes the hidden token, emits a message).
pub const SPEAKER: usize = 0;
/// Agent id of the listener (receives the message, guesses the token).
pub const LISTENER: usize = 1;

/// 2-agent referential signaling game over a fixed vocabulary.
///
/// See the [module docs](self) for the full description and the comms layout
/// table.
#[derive(Debug, Clone)]
pub struct SignalingGame {
    /// Vocabulary size `V`: both the hidden-token cardinality and the message /
    /// guess cardinality.
    vocab: usize,
    /// Hidden token the speaker must transmit (`0..vocab`).
    hidden: i64,
    /// Message token the listener received on the most recent step (`-1` before
    /// the first step of an episode).
    received: i64,
}

impl SignalingGame {
    /// Number of agents (always 2: speaker + listener).
    pub const NUM_AGENTS: usize = 2;

    /// Construct a signaling game over a `vocab`-token vocabulary with hidden
    /// token `0`.
    ///
    /// # Panics
    ///
    /// Panics if `vocab == 0` — a signaling game needs at least one token.
    pub fn new(vocab: usize) -> Self {
        assert!(vocab > 0, "SignalingGame requires vocab >= 1");
        Self { vocab, hidden: 0, received: -1 }
    }

    /// Construct a signaling game with an explicit hidden token (for tests /
    /// deterministic rollouts).
    ///
    /// # Panics
    ///
    /// Panics if `vocab == 0` or `hidden` is outside `0..vocab`.
    pub fn with_hidden(vocab: usize, hidden: i64) -> Self {
        assert!(vocab > 0, "SignalingGame requires vocab >= 1");
        assert!(
            hidden >= 0 && (hidden as usize) < vocab,
            "hidden token {hidden} out of range 0..{vocab}"
        );
        Self { vocab, hidden, received: -1 }
    }

    /// Vocabulary size `V`.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// The hidden token the speaker must transmit this episode.
    pub fn hidden(&self) -> i64 {
        self.hidden
    }

    /// The message token the listener received on the most recent step, or `-1`
    /// if no step has occurred since the last reset.
    pub fn received(&self) -> i64 {
        self.received
    }

    /// Set the hidden token (`0..vocab`) and clear the received message.
    ///
    /// # Panics
    ///
    /// Panics if `hidden` is outside `0..vocab`.
    pub fn set_hidden(&mut self, hidden: i64) {
        assert!(
            hidden >= 0 && (hidden as usize) < self.vocab,
            "hidden token {hidden} out of range 0..{}",
            self.vocab
        );
        self.hidden = hidden;
        self.received = -1;
    }

    /// One-hot encoding of the hidden token — the speaker's observation.
    fn speaker_obs(&self) -> Vec<f32> {
        let mut obs = vec![0.0_f32; self.vocab];
        obs[self.hidden as usize] = 1.0;
        obs
    }

    /// The listener's observation: a single dim carrying the received token
    /// (`-1` before the first step), laid out via
    /// [`crate::multi_agent::comms::place_message`].
    fn listener_obs(&self) -> Vec<f32> {
        let mut obs = vec![-1.0_f32; self.message_obs_size(LISTENER)];
        place_message(&mut obs, 0, &[self.received]);
        obs
    }
}

impl Default for SignalingGame {
    /// A 2-token signaling game (the smallest non-trivial vocabulary).
    fn default() -> Self {
        Self::new(2)
    }
}

impl Environment for SignalingGame {
    type Action = i64;
    /// Full snapshot: the env consumes no RNG, so restoring reproduces every
    /// subsequent step exactly.
    type State = (i64, i64);

    fn reset(&mut self) {
        self.received = -1;
    }

    /// Single-agent surface returns the speaker's observation. Multi-agent
    /// consumers should use [`MultiAgentEnvironment::get_agent_observation`].
    fn get_observation(&self) -> Vec<f32> {
        self.speaker_obs()
    }

    /// Degenerate single-agent step: interpret `action` as the speaker's emitted
    /// message and broadcast it. Multi-agent consumers should use
    /// [`MultiAgentEnvironment::step_multi`].
    fn step(&mut self, action: i64) -> StepResult {
        let r = self.step_multi(&[vec![action], vec![action]]);
        StepResult {
            observation: r.observations.into_iter().next().unwrap_or_default(),
            reward: r.rewards.first().copied().unwrap_or(0.0),
            terminated: r.terminated.first().copied().unwrap_or(false),
            truncated: false,
            info: StepInfo::default(),
        }
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo { shape: vec![self.vocab], space_type: SpaceType::Box }
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo { shape: vec![], space_type: SpaceType::Discrete(self.vocab) }
    }

    fn render(&self) -> Vec<u8> {
        Vec::new()
    }

    fn close(&mut self) {}

    fn clone_state(&self) -> (i64, i64) {
        (self.hidden, self.received)
    }

    fn restore_state(&mut self, state: &(i64, i64)) {
        self.hidden = state.0;
        self.received = state.1;
    }
}

impl MultiAgentEnvironment for SignalingGame {
    fn num_agents(&self) -> usize {
        Self::NUM_AGENTS
    }

    fn get_agent_observation(&self, agent_id: usize) -> Vec<f32> {
        match agent_id {
            SPEAKER => self.speaker_obs(),
            LISTENER => self.listener_obs(),
            other => panic!("SignalingGame has 2 agents; no agent {other}"),
        }
    }

    fn agent_action_space(&self, agent_id: usize) -> Vec<usize> {
        match agent_id {
            // Speaker has no task action — only the message dim (see `message_vocab`).
            SPEAKER => Vec::new(),
            // Listener's task action is its guess over the vocabulary.
            LISTENER => vec![self.vocab],
            other => panic!("SignalingGame has 2 agents; no agent {other}"),
        }
    }

    fn step_multi(&mut self, actions: &[Vec<i64>]) -> MultiAgentResult {
        assert_eq!(actions.len(), Self::NUM_AGENTS, "signaling game requires 2 action vectors");

        // Peel the message dims off each agent's action using the shared helper.
        // Speaker: task_len 0 -> the whole action is its message token.
        let (_speaker_task, speaker_msg) =
            split_action(&actions[SPEAKER], self.agent_action_space(SPEAKER).len());
        // Listener: task_len == vocab dims -> its guess; it sends no message.
        let (listener_task, _listener_msg) =
            split_action(&actions[LISTENER], self.agent_action_space(LISTENER).len());

        // Broadcast delivery: the speaker's message reaches the listener.
        debug_assert!(self.delivery().reaches(SPEAKER, LISTENER));
        self.received = speaker_msg.first().copied().unwrap_or(-1);

        // Reward: both agents win iff the listener's guess matches the hidden token.
        let guess = listener_task.first().copied().unwrap_or(-1);
        let correct = guess == self.hidden;
        let reward = if correct { 1.0_f32 } else { 0.0 };

        MultiAgentResult::new(
            vec![self.speaker_obs(), self.listener_obs()],
            vec![reward, reward],
            // Single-shot game: the round terminates every step.
            vec![true, true],
            vec![false, false],
        )
    }

    fn active_agents(&self) -> Vec<bool> {
        vec![true; Self::NUM_AGENTS]
    }
}

impl CommunicatingEnvironment for SignalingGame {
    fn message_vocab(&self, agent_id: usize) -> Vec<usize> {
        match agent_id {
            // Speaker emits one token over the full vocabulary.
            SPEAKER => vec![self.vocab],
            // Listener sends nothing.
            _ => Vec::new(),
        }
    }

    fn message_obs_size(&self, agent_id: usize) -> usize {
        match agent_id {
            // Listener receives the speaker's single message token.
            LISTENER => 1,
            // Speaker receives nothing.
            _ => 0,
        }
    }

    fn delivery(&self) -> Delivery {
        Delivery::Broadcast
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comms_layout_matches_design() {
        let env = SignalingGame::new(4);
        // Speaker: no task action, one 4-way message token, receives nothing.
        assert_eq!(env.agent_action_space(SPEAKER), Vec::<usize>::new());
        assert_eq!(env.message_vocab(SPEAKER), vec![4]);
        assert_eq!(env.message_obs_size(SPEAKER), 0);
        // Listener: a 4-way guess, sends nothing, receives one token dim.
        assert_eq!(env.agent_action_space(LISTENER), vec![4]);
        assert_eq!(env.message_vocab(LISTENER), Vec::<usize>::new());
        assert_eq!(env.message_obs_size(LISTENER), 1);
        assert_eq!(env.delivery(), Delivery::Broadcast);
    }

    #[test]
    fn speaker_observes_hidden_token_one_hot() {
        let env = SignalingGame::with_hidden(5, 3);
        let obs = env.get_agent_observation(SPEAKER);
        assert_eq!(obs, vec![0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn message_round_trips_speaker_to_listener() {
        // Core acceptance criterion: agent 0 emits token T; after `step_multi`
        // agent 1 receives T in its observation vector.
        let mut env = SignalingGame::with_hidden(4, 2);
        let token = 2_i64;
        // Speaker action = [token]; listener guesses (value irrelevant here).
        let result = env.step_multi(&[vec![token], vec![0]]);

        // Listener's post-step observation carries the emitted token.
        let listener_obs = &result.observations[LISTENER];
        assert_eq!(listener_obs.len(), env.message_obs_size(LISTENER));
        assert_eq!(listener_obs[0], token as f32);
        // And it is reflected by the standalone observation accessor too.
        assert_eq!(env.get_agent_observation(LISTENER)[0], token as f32);
        assert_eq!(env.received(), token);
    }

    #[test]
    fn reward_is_one_when_listener_guesses_hidden_token() {
        let mut env = SignalingGame::with_hidden(3, 1);
        // Listener guesses correctly (1 == hidden).
        let correct = env.step_multi(&[vec![1], vec![1]]);
        assert_eq!(correct.rewards, vec![1.0, 1.0]);

        // Listener guesses incorrectly.
        let mut env = SignalingGame::with_hidden(3, 1);
        let wrong = env.step_multi(&[vec![1], vec![0]]);
        assert_eq!(wrong.rewards, vec![0.0, 0.0]);
    }

    #[test]
    fn listener_obs_before_step_is_sentinel() {
        let env = SignalingGame::new(3);
        // No step yet -> received == -1 sentinel.
        assert_eq!(env.get_agent_observation(LISTENER), vec![-1.0]);
    }

    #[test]
    fn reset_clears_received_message() {
        let mut env = SignalingGame::with_hidden(4, 0);
        env.step_multi(&[vec![3], vec![0]]);
        assert_eq!(env.received(), 3);
        env.reset();
        assert_eq!(env.received(), -1);
        assert_eq!(env.get_agent_observation(LISTENER), vec![-1.0]);
    }

    #[test]
    fn snapshot_restore_round_trips() {
        let mut env = SignalingGame::with_hidden(4, 2);
        env.step_multi(&[vec![3], vec![0]]);
        let snap = env.clone_state();
        env.step_multi(&[vec![1], vec![0]]);
        assert_eq!(env.received(), 1);
        env.restore_state(&snap);
        assert_eq!(env.hidden(), 2);
        assert_eq!(env.received(), 3);
    }

    #[test]
    fn num_agents_and_active() {
        let env = SignalingGame::default();
        assert_eq!(env.num_agents(), 2);
        assert_eq!(env.active_agents(), vec![true, true]);
    }
}
