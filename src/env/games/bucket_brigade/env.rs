//! Bucket Brigade multi-agent environment adapter.
//!
//! Wraps [`bucket_brigade_core::BucketBrigade`] with a flat-observation,
//! multi-discrete-action interface that the thrust joint trainer can consume
//! directly.
//!
//! # Observation layout
//!
//! Each agent observation is a single flat `Vec<f32>` laid out as:
//!
//! ```text
//! [agent_id_norm(1), houses(NUM_HOUSES), signals(N), locations(N),
//!  last_actions(3*N), round1_signals(N), scenario_info(12)]
//! ```
//!
//! where `N = num_agents` and `NUM_HOUSES = 10`. The leading
//! `agent_id_norm` scalar is `agent_id as f32 / (num_agents - 1).max(1) as
//! f32`, which makes each agent's flat observation distinguishable from
//! the others (necessary for trainers like NFSP that need to learn
//! agent-specialized policies on this shared-observation env). The `3` in
//! `last_actions` reflects the post-#235 action shape `[house, mode,
//! signal]` (see [`bucket_brigade_core::Action`]). `round1_signals` is the
//! two-phase commitment channel from issue #252; it is zero-filled when
//! the underlying scenario runs in the default `simultaneous` commitment
//! mode.
//!
//! This layout intentionally diverges from the pre-#235 flat layout: the
//! signal action dim and the round-1 commitment channel are both first-class
//! observation features now, so we surface them rather than hiding them.
//!
//! # Action layout
//!
//! Each agent emits a length-3 action vector `[house_index, mode, signal]`:
//!
//! * `house_index` — 0..[`NUM_HOUSES`]
//! * `mode` — 0 (REST) or 1 (WORK)
//! * `signal` — 0 (REST) or 1 (WORK)
//!
//! The trainer is expected to use a multi-discrete policy with
//! `action_dims = vec![NUM_HOUSES, 2, 2]`.
//!
//! # Two-phase commitment mode
//!
//! Scenarios with `commitment_mode == "two_phase"` (issue #252) require
//! calling the engine's `step_two_phase(round1_signals, round2_actions)`
//! API instead of `step(actions)`. The current adapter only supports
//! simultaneous (single-phase) scenarios; calling [`BucketBrigadeMaEnv::step`]
//! on a two-phase scenario will panic via the underlying engine.

use bucket_brigade_core::{Action, AgentObservation, BucketBrigade, Scenario};

use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

/// Number of houses on the Bucket Brigade ring (used for the default
/// `num_houses = 10` topology).
///
/// The underlying [`Scenario`] now carries a `num_houses` field (issue
/// #254) which may be smaller (e.g. 2 for `v2_minimal`). Callers that
/// support variable-size topologies should read [`Scenario::num_houses`]
/// at runtime; this constant is provided as the legacy/default value for
/// callers wired to the 10-house ring.
pub const NUM_HOUSES: usize = 10;

/// Number of action dimensions per agent (`[house_index, mode, signal]`).
pub const ACTION_DIMS: usize = 3;

/// Number of static scenario features in the observation footer (mirrors
/// the Python flattener and the engine's
/// [`bucket_brigade_core::AgentObservation::scenario_info`] field).
pub const SCENARIO_INFO_LEN: usize = 12;

/// Multi-agent Bucket Brigade environment for thrust trainers.
///
/// One instance owns one game; the trainer is responsible for collecting
/// rollouts and resetting on done.
pub struct BucketBrigadeMaEnv {
    inner: BucketBrigade,
    scenario: Scenario,
    num_agents: usize,
    /// Cached `num_houses` from the scenario, captured at construction so
    /// downstream rollout code doesn't need to round-trip through the
    /// scenario clone every step.
    num_houses: usize,
}

/// Per-step result returned by [`BucketBrigadeMaEnv::step`].
#[derive(Debug, Clone)]
pub struct MaStepResult {
    /// Per-agent rewards for this night.
    pub rewards: Vec<f32>,
    /// Whether the episode has terminated.
    pub done: bool,
    /// Per-agent flattened observations *after* this step.
    pub observations: Vec<Vec<f32>>,
}

impl BucketBrigadeMaEnv {
    /// Create a new env with the given scenario and number of agents.
    ///
    /// `seed` is consumed by the underlying engine's deterministic RNG; pass
    /// `None` for a non-deterministic episode stream.
    pub fn new(scenario: Scenario, num_agents: usize, seed: Option<u64>) -> Self {
        let num_houses = scenario.num_houses as usize;
        let inner = BucketBrigade::new(scenario.clone(), num_agents, seed);
        Self { inner, scenario, num_agents, num_houses }
    }

    /// Construct a [`BucketBrigadeMaEnv`] from a versioned scenario ID
    /// (e.g. `"minimal_specialization-v1"`).
    ///
    /// `num_agents` defaults to the value frozen into the ID (see
    /// [`super::registry::DEFAULT_NUM_AGENTS`]) if `None`. Overriding it
    /// produces an env that is not covered by the frozen ID's
    /// reproducibility guarantee — record the override in experiment
    /// metadata for traceability.
    ///
    /// This is the Rust mirror of `bucket_brigade.make(scenario_id, ...)`.
    pub fn from_scenario_id(
        scenario_id: &str,
        num_agents: Option<usize>,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        let scenario = super::registry::get_scenario_by_id(scenario_id)?;
        let n = num_agents
            .or_else(|| super::registry::default_num_agents_for(scenario_id))
            .unwrap_or(super::registry::DEFAULT_NUM_AGENTS);
        Ok(Self::new(scenario, n, seed))
    }

    /// Number of agents this env was constructed with.
    pub fn num_agents(&self) -> usize {
        self.num_agents
    }

    /// Number of houses on the ring for this scenario.
    pub fn num_houses(&self) -> usize {
        self.num_houses
    }

    /// Action-space layout: `[num_houses, 2, 2]` for `[house, mode, signal]`.
    ///
    /// Returns a `Vec` rather than a fixed-size array so callers can drive
    /// scenarios with `num_houses != 10` (e.g. `v2_minimal` has 2 houses).
    pub fn action_dims(&self) -> Vec<i64> {
        vec![self.num_houses as i64, 2, 2]
    }

    /// Flattened observation dimensionality (same for every agent).
    ///
    /// Layout:
    /// `agent_id_norm(1) + houses(num_houses) + signals(N) + locations(N) +
    ///  last_actions(3*N) + round1_signals(N) + scenario_info(12)`
    ///
    /// The leading `1` is the normalized `agent_id` scalar that
    /// distinguishes agents on this shared-observation env (see the
    /// module docstring for the rationale).
    pub fn obs_dim(&self) -> usize {
        1                                  // normalized agent_id
            + self.num_houses
            + self.num_agents              // signals
            + self.num_agents              // locations
            + ACTION_DIMS * self.num_agents // last_actions ([h, m, s] flattened)
            + self.num_agents              // round1_signals
            + SCENARIO_INFO_LEN
    }

    /// Read-only access to the underlying scenario.
    pub fn scenario(&self) -> &Scenario {
        &self.scenario
    }

    /// Read-only access to the underlying engine.
    ///
    /// Exposed so that sibling submodules (e.g. the
    /// `MultiAgentEnvironment` impl, gated behind the `training` feature)
    /// can call engine methods without duplicating the env's caching
    /// logic. The `allow(dead_code)` is needed because, without the
    /// `training` feature, no sibling actually consumes this accessor.
    #[allow(dead_code)]
    pub(super) fn inner(&self) -> &BucketBrigade {
        &self.inner
    }

    /// Reset the env in-place and return per-agent initial observations.
    ///
    /// `seed` re-seeds the engine RNG (matching the Python env's contract).
    ///
    /// Note: the underlying engine doesn't currently expose a re-seed hook
    /// on its existing instance; when `seed` is `Some`, we rebuild the
    /// engine. Otherwise we just call the engine's [`BucketBrigade::reset`].
    pub fn reset(&mut self, seed: Option<u64>) -> Vec<Vec<f32>> {
        if let Some(seed) = seed {
            self.inner = BucketBrigade::new(self.scenario.clone(), self.num_agents, Some(seed));
        } else {
            self.inner.reset();
        }
        self.collect_observations()
    }

    /// Step the env with one action per agent.
    ///
    /// `actions[i] = [house_index, mode, signal]` — a length-3 slice with
    /// `house_index in 0..num_houses`, `mode in 0..2`, `signal in 0..2`.
    pub fn step(&mut self, actions: &[Action]) -> MaStepResult {
        assert_eq!(
            actions.len(),
            self.num_agents,
            "BucketBrigadeMaEnv::step: expected {} actions, got {}",
            self.num_agents,
            actions.len()
        );

        let rust_actions: Vec<Action> = actions.to_vec();
        let result = self.inner.step(&rust_actions);

        let observations = self.collect_observations();

        MaStepResult { rewards: result.rewards, done: result.done, observations }
    }

    fn collect_observations(&self) -> Vec<Vec<f32>> {
        (0..self.num_agents)
            .map(|i| {
                flatten_observation(
                    &self.inner.get_observation(i),
                    self.num_houses,
                    i,
                    self.num_agents,
                )
            })
            .collect()
    }
}

// -------- Single-agent `Environment` adapter --------
//
// Bucket Brigade is fundamentally multi-agent, so this single-agent surface
// exists primarily to (a) implement the base `Environment` trait that the
// rest of Thrust assumes for snapshot/restore/pool plumbing, and (b) act
// as a base for the `MultiAgentEnvironment` impl (which is gated behind
// the `training` feature in `mod.rs`).
//
// `step()` decodes a single scalar action as a Cartesian-product index over
// `[house, mode, signal]` and broadcasts it to every agent. Real
// multi-agent consumers should drive the env through
// `MultiAgentEnvironment` (or `BucketBrigadeMaEnv::step` directly).
impl Environment for BucketBrigadeMaEnv {
    type Action = i64;

    /// Snapshot type for the single-agent `Environment` adapter.
    ///
    /// **Stub**: `bucket_brigade_core::BucketBrigade` does not expose a
    /// snapshot/restore API, so this implementation is currently a no-op.
    /// `clone_state` returns `()` and `restore_state` does nothing. Wiring up
    /// real snapshots requires upstream support in `bucket_brigade_core`.
    type State = ();

    fn reset(&mut self) {
        let _ = BucketBrigadeMaEnv::reset(self, None);
    }

    fn get_observation(&self) -> Vec<f32> {
        flatten_observation(&self.inner.get_observation(0), self.num_houses, 0, self.num_agents)
    }

    fn step(&mut self, action: i64) -> StepResult {
        // Single-agent surface: decode the scalar Cartesian-product index
        // `house * (2 * 2) + mode * 2 + signal` and broadcast it to every
        // agent. This is a degenerate path; multi-agent consumers should
        // use `MultiAgentEnvironment::step_multi` instead.
        let signal = (action % 2) as u8;
        let mode = ((action / 2) % 2) as u8;
        let house = ((action / 4) % self.num_houses as i64) as u8;
        let actions: Vec<Action> = (0..self.num_agents).map(|_| [house, mode, signal]).collect();
        let r = BucketBrigadeMaEnv::step(self, &actions);
        StepResult {
            observation: r.observations.into_iter().next().unwrap_or_default(),
            reward: r.rewards.first().copied().unwrap_or(0.0),
            terminated: r.done,
            truncated: false,
            info: StepInfo::default(),
        }
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo { shape: vec![self.obs_dim()], space_type: SpaceType::Box }
    }

    fn action_space(&self) -> SpaceInfo {
        // Cartesian-product flattening for the single-agent surface only.
        // `num_houses * 2 (mode) * 2 (signal)` per agent; the same scalar is
        // broadcast to every agent in `step()`.
        SpaceInfo { shape: vec![], space_type: SpaceType::Discrete(self.num_houses * 2 * 2) }
    }

    fn render(&self) -> Vec<u8> {
        Vec::new()
    }

    fn close(&mut self) {}

    /// Stub: returns `()`. The underlying `bucket_brigade_core::BucketBrigade`
    /// engine does not currently expose snapshot/restore; calling this is a
    /// no-op for now and the returned snapshot cannot meaningfully be used
    /// with `restore_state`.
    fn clone_state(&self) {}

    /// Stub: does nothing. See [`Self::clone_state`] for the limitation.
    fn restore_state(&mut self, _state: &()) {}
}

// -------- `JointEnv` adapter (gated on `training`) --------
//
// The [`crate::multi_agent::JointEnv`] trait lives behind the `training`
// feature, so the impl is gated to keep `env-bucket-brigade` usable on
// its own (e.g. for the standalone `tests/test_bucket_brigade_env.rs`
// surface). Both features together are required for the
// `tests/test_nfsp_bucket_brigade.rs` integration test and for any
// PSRO/NFSP trainer wiring that drives this env.
//
// Patterned after `impl JointEnv for NPlayerMatchingPennies`
// (`src/env/games/n_player_matching_pennies.rs`) and
// `impl JointEnv for MatchingPennies` (`src/env/games/matching_pennies.rs`).
#[cfg(feature = "training")]
impl crate::multi_agent::JointEnv for BucketBrigadeMaEnv {
    fn reset_joint(&mut self, seed: Option<u64>) -> Vec<Vec<f32>> {
        BucketBrigadeMaEnv::reset(self, seed)
    }

    fn step_joint(&mut self, actions: &[Vec<i64>]) -> crate::multi_agent::JointStepResult {
        debug_assert_eq!(
            actions.len(),
            self.num_agents,
            "BucketBrigadeMaEnv::step_joint: expected {} agent actions, got {}",
            self.num_agents,
            actions.len()
        );
        // Convert each per-agent `Vec<i64>` (length 3, `[house, mode,
        // signal]`) into the engine's `Action = [u8; 3]` shape.
        let acts: Vec<Action> = actions
            .iter()
            .map(|a| {
                debug_assert_eq!(
                    a.len(),
                    ACTION_DIMS,
                    "per-agent action must be length {} ([house, mode, signal]), got {}",
                    ACTION_DIMS,
                    a.len()
                );
                [a[0] as u8, a[1] as u8, a[2] as u8]
            })
            .collect();
        let res = BucketBrigadeMaEnv::step(self, &acts);
        crate::multi_agent::JointStepResult {
            rewards: res.rewards,
            done: res.done,
            observations: res.observations,
        }
    }
}

/// Flatten an [`AgentObservation`] into a `Vec<f32>` matching the layout
/// documented at the top of this module:
///
/// ```text
/// [agent_id_norm(1), houses(num_houses), signals(N), locations(N),
///  last_actions(3*N), round1_signals(N), scenario_info(12)]
/// ```
///
/// `num_houses` is taken from the env (cached at construction time) so this
/// helper is independent of the global [`NUM_HOUSES`] constant — necessary
/// because newer scenarios (e.g. `v2_minimal`) override the default ring
/// size.
///
/// The leading scalar is `agent_id as f32 / (num_agents - 1).max(1) as f32`,
/// so agent 0 sees `0.0`, the last agent sees `1.0`, and intermediate
/// agents see linearly interpolated values. This is the same one-dim
/// agent-identity encoding used by
/// [`crate::env::games::n_player_matching_pennies::NPlayerMatchingPennies::per_agent_obs`].
/// Without this scalar, every per-agent observation from
/// [`BucketBrigadeMaEnv::collect_observations`] would be bit-identical
/// (the underlying [`AgentObservation`] fields surfaced here are all
/// global state), which would prevent NFSP / PSRO from learning
/// agent-specialized policies.
pub(crate) fn flatten_observation(
    obs: &AgentObservation,
    num_houses: usize,
    agent_id: usize,
    num_agents: usize,
) -> Vec<f32> {
    let n = 1                                     // normalized agent_id
        + num_houses
        + obs.signals.len()
        + obs.locations.len()
        + obs.last_actions.len() * ACTION_DIMS
        + obs.round1_signals.len()
        + obs.scenario_info.len();
    let mut out = Vec::with_capacity(n);
    // Leading agent-identity scalar; mirrors `NPlayerMatchingPennies`.
    let denom = (num_agents.saturating_sub(1)).max(1) as f32;
    out.push(agent_id as f32 / denom);
    // The engine's `houses` length already matches `num_houses`; we don't
    // re-truncate or pad. Diverging here would indicate an engine bug.
    debug_assert_eq!(obs.houses.len(), num_houses);
    out.extend(obs.houses.iter().map(|&v| v as f32));
    out.extend(obs.signals.iter().map(|&v| v as f32));
    out.extend(obs.locations.iter().map(|&v| v as f32));
    for [house, mode, signal] in &obs.last_actions {
        out.push(*house as f32);
        out.push(*mode as f32);
        out.push(*signal as f32);
    }
    out.extend(obs.round1_signals.iter().map(|&v| v as f32));
    out.extend(obs.scenario_info.iter().copied());
    out
}
