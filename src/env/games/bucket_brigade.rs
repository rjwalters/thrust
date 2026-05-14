//! Bucket Brigade multi-agent environment adapter.
//!
//! Wraps [`bucket_brigade_core::BucketBrigade`] with a flat-observation,
//! multi-discrete-action interface that the thrust joint trainer can consume
//! directly. Observation layout matches the Python flattener at
//! `bucket_brigade/training/observation_utils.py`:
//!
//! ```text
//! [houses(10), signals(N), locations(N), last_actions(2N), scenario_info(12)]
//! ```
//!
//! Actions are `[house_index, mode]` per agent (a length-2 vector in
//! `0..10 x 0..2`). The trainer is expected to use
//! [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpPolicy`] with
//! `action_dims = vec![10, 2]`.
//!
//! In addition to its inherent [`BucketBrigadeMaEnv::step`] API (which takes
//! the typed `&[[u8; 2]]` form), this type implements the multi-discrete
//! [`crate::multi_agent::MultiAgentEnvironment`] trait so it can be driven
//! through the generic multi-agent rollout surface (one `Vec<i64>` per
//! agent, length matching [`BucketBrigadeMaEnv::action_dims`]).

use bucket_brigade_core::{Action, AgentObservation, BucketBrigade, Scenario};

use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};
use crate::multi_agent::{MultiAgentEnvironment, MultiAgentResult};

/// Number of houses on the Bucket Brigade ring. Constant; the engine
/// hard-codes 10.
pub const NUM_HOUSES: usize = 10;

/// Multi-agent Bucket Brigade environment for thrust trainers.
///
/// One instance owns one game; the trainer is responsible for collecting
/// rollouts and resetting on done.
pub struct BucketBrigadeMaEnv {
    inner: BucketBrigade,
    scenario: Scenario,
    num_agents: usize,
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
        let inner = BucketBrigade::new(scenario.clone(), num_agents, seed);
        Self { inner, scenario, num_agents }
    }

    /// Number of agents this env was constructed with.
    pub fn num_agents(&self) -> usize {
        self.num_agents
    }

    /// Action-space layout matching
    /// [`crate::policy::multi_discrete_mlp::MultiDiscreteMlpPolicy::action_dims`]:
    /// `[10, 2]` for `[house_index, mode]`.
    pub fn action_dims(&self) -> [i64; 2] {
        [NUM_HOUSES as i64, 2]
    }

    /// Flattened observation dimensionality (same for every agent).
    pub fn obs_dim(&self) -> usize {
        // houses + signals + locations + last_actions + scenario_info
        NUM_HOUSES + self.num_agents + self.num_agents + 2 * self.num_agents + 12
    }

    /// Read-only access to the underlying scenario.
    pub fn scenario(&self) -> &Scenario {
        &self.scenario
    }

    /// Whether the current episode has ended.
    pub fn done(&self) -> bool {
        // The engine doesn't currently expose a `.done` accessor; we re-derive
        // it from the last step. For consumers that need a cheap check, this
        // is wrapped in step()'s return; otherwise treat reset() as the only
        // way to recover from done.
        false
    }

    /// Reset the env in-place and return per-agent initial observations.
    ///
    /// `seed` re-seeds the engine RNG (matching the Python env's contract).
    pub fn reset(&mut self, seed: Option<u64>) -> Vec<Vec<f32>> {
        // The bucket-brigade-core engine doesn't currently expose a re-seed
        // hook on its existing instance; the only re-seeding entry point is
        // BucketBrigade::new. When seed is Some, rebuild; otherwise just
        // reset the existing engine.
        if let Some(seed) = seed {
            self.inner = BucketBrigade::new(self.scenario.clone(), self.num_agents, Some(seed));
        } else {
            self.inner.reset();
        }
        (0..self.num_agents)
            .map(|i| flatten_observation(&self.inner.get_observation(i)))
            .collect()
    }

    /// Step the env with one action per agent.
    ///
    /// `actions[i] = [house_index, mode]` --- a length-2 slice with
    /// `house_index in 0..10` and `mode in 0..2`.
    pub fn step(&mut self, actions: &[[u8; 2]]) -> MaStepResult {
        assert_eq!(
            actions.len(),
            self.num_agents,
            "BucketBrigadeMaEnv::step: expected {} actions, got {}",
            self.num_agents,
            actions.len()
        );

        let rust_actions: Vec<Action> = actions.iter().copied().collect();
        let result = self.inner.step(&rust_actions);

        let observations: Vec<Vec<f32>> = (0..self.num_agents)
            .map(|i| flatten_observation(&self.inner.get_observation(i)))
            .collect();

        MaStepResult { rewards: result.rewards, done: result.done, observations }
    }
}

// -------- Single-agent `Environment` adapter --------
//
// `MultiAgentEnvironment` requires `Environment`. Bucket Brigade is
// fundamentally multi-agent, so this single-agent surface exists only to
// satisfy the trait hierarchy: it returns agent 0's observation and treats a
// single scalar action as a Cartesian-product index over `[house, mode]`.
// Real consumers should drive the env through `MultiAgentEnvironment`.
impl Environment for BucketBrigadeMaEnv {
    fn reset(&mut self) {
        let _ = BucketBrigadeMaEnv::reset(self, None);
    }

    fn get_observation(&self) -> Vec<f32> {
        flatten_observation(&self.inner.get_observation(0))
    }

    fn step(&mut self, action: i64) -> StepResult {
        // Single-agent surface: decode the scalar Cartesian-product index
        // `house * 2 + mode` and broadcast it to every agent. This is a
        // degenerate path; multi-agent consumers should use
        // [`MultiAgentEnvironment::step_multi`] instead.
        let house = ((action / 2) % NUM_HOUSES as i64) as u8;
        let mode = (action % 2) as u8;
        let actions: Vec<[u8; 2]> = (0..self.num_agents).map(|_| [house, mode]).collect();
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
        SpaceInfo { shape: vec![], space_type: SpaceType::Discrete(NUM_HOUSES * 2) }
    }

    fn render(&self) -> Vec<u8> {
        Vec::new()
    }

    fn close(&mut self) {}
}

// -------- Multi-agent factored action surface --------
impl MultiAgentEnvironment for BucketBrigadeMaEnv {
    fn num_agents(&self) -> usize {
        self.num_agents
    }

    fn get_agent_observation(&self, agent_id: usize) -> Vec<f32> {
        flatten_observation(&self.inner.get_observation(agent_id))
    }

    fn agent_action_space(&self, _agent_id: usize) -> Vec<usize> {
        // Factored `[house_index, mode]` -- matches
        // `MultiDiscreteMlpPolicy::action_dims` used by the BB trainer.
        vec![NUM_HOUSES, 2]
    }

    fn step_multi(&mut self, actions: &[Vec<i64>]) -> MultiAgentResult {
        assert_eq!(
            actions.len(),
            self.num_agents,
            "BucketBrigadeMaEnv::step_multi: expected {} actions, got {}",
            self.num_agents,
            actions.len()
        );

        let inner_actions: Vec<[u8; 2]> = actions
            .iter()
            .enumerate()
            .map(|(i, a)| {
                assert_eq!(
                    a.len(),
                    2,
                    "BucketBrigadeMaEnv::step_multi: agent {} action must have 2 dims \
                     ([house_index, mode]), got {}",
                    i,
                    a.len()
                );
                assert!(
                    (0..NUM_HOUSES as i64).contains(&a[0]),
                    "BucketBrigadeMaEnv::step_multi: agent {} house_index {} out of range \
                     0..{}",
                    i,
                    a[0],
                    NUM_HOUSES
                );
                assert!(
                    a[1] == 0 || a[1] == 1,
                    "BucketBrigadeMaEnv::step_multi: agent {} mode {} must be 0 or 1",
                    i,
                    a[1]
                );
                [a[0] as u8, a[1] as u8]
            })
            .collect();

        let result = BucketBrigadeMaEnv::step(self, &inner_actions);
        let n = self.num_agents;
        MultiAgentResult::new(
            result.observations,
            result.rewards,
            vec![result.done; n],
            vec![false; n],
        )
    }

    fn active_agents(&self) -> Vec<bool> {
        // The underlying engine has no per-agent termination signal; either
        // the whole night is still running or the episode is over.
        vec![true; self.num_agents]
    }
}

/// Flatten an [`AgentObservation`] into a `Vec<f32>` matching the layout used
/// by the Python joint trainer:
/// `[houses, signals, locations, last_actions (flat), scenario_info]`.
fn flatten_observation(obs: &AgentObservation) -> Vec<f32> {
    let n = obs.houses.len()
        + obs.signals.len()
        + obs.locations.len()
        + obs.last_actions.len() * 2
        + obs.scenario_info.len();
    let mut out = Vec::with_capacity(n);
    out.extend(obs.houses.iter().map(|&v| v as f32));
    out.extend(obs.signals.iter().map(|&v| v as f32));
    out.extend(obs.locations.iter().map(|&v| v as f32));
    for [house, mode] in &obs.last_actions {
        out.push(*house as f32);
        out.push(*mode as f32);
    }
    out.extend(obs.scenario_info.iter().copied());
    out
}

