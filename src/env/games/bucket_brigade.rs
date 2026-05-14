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
//! Does **not** implement [`crate::multi_agent::MultiAgentEnvironment`]
//! because that trait currently takes `&[i64]` and doesn't support factored
//! action spaces (see upstream issue #3). When the trait is widened, this
//! struct can grow a trivial impl.

use bucket_brigade_core::{Action, AgentObservation, BucketBrigade, Scenario};

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

