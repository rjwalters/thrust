//! Simple Contextual Bandit Environment
//!
//! This is a trivial environment for testing PPO implementation correctness:
//! - State: Single binary value (0 or 1)
//! - Actions: Two choices (0 or 1)
//! - Optimal policy: Always choose action = state
//! - Reward: +1.0 if action == state, 0.0 otherwise
//! - Episodes: Fixed length of 100 steps
//!
//! This environment should reach 100% success rate (reward=1.0 every step)
//! if PPO is implemented correctly. If it doesn't converge reliably, there's a
//! bug.

use rand::{Rng, SeedableRng};

use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

/// Snapshot of SimpleBandit's simulation state.
///
/// Captures the visible state (current context and step counter) but **not**
/// the internal RNG. After `restore_state`, the next `step` will compute the
/// reward using the restored `state`, and `terminated` matches the original
/// trajectory. However, the *next sampled state* (drawn from the bandit's
/// internal RNG inside `step`) is **not** reproduced: see the per-env
/// snapshot semantics in [`SimpleBandit`].
#[derive(Debug, Clone)]
pub struct SimpleBanditState {
    /// Current context (0.0 or 1.0)
    pub state: f32,
    /// Step counter
    pub steps: usize,
}

/// Simple contextual bandit for testing PPO correctness.
///
/// # Snapshot semantics
///
/// `clone_state` / `restore_state` capture the current context and step
/// counter, so the next `step(action)` produces the same `reward` and
/// `terminated` flag it would have at snapshot time. The bandit's internal
/// `StdRng` is **not** captured: the *next* state sampled inside `step` will
/// differ between the snapshotted run and the restored run, so the
/// `observation` returned by `step` is not reproduced bit-for-bit. The reward
/// for the action taken at the snapshot point is still deterministic.
#[derive(Debug)]
pub struct SimpleBandit {
    state: f32,
    steps: usize,
    max_steps: usize,
    rng: rand::rngs::StdRng,
}

impl SimpleBandit {
    /// Create a new simple bandit environment
    pub fn new() -> Self {
        Self { state: 0.0, steps: 0, max_steps: 100, rng: rand::rngs::StdRng::from_os_rng() }
    }
}

impl Default for SimpleBandit {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for SimpleBandit {
    type Action = i64;
    type State = SimpleBanditState;

    fn reset(&mut self) {
        // Random start state (0 or 1)
        self.state = self.rng.random_range(0..2) as f32;
        self.steps = 0;
    }

    fn get_observation(&self) -> Vec<f32> {
        vec![self.state]
    }

    fn step(&mut self, action: i64) -> StepResult {
        // Reward +1 if action matches state, 0 otherwise
        let reward = if action == self.state as i64 {
            1.0
        } else {
            0.0
        };

        self.steps += 1;
        let terminated = self.steps >= self.max_steps;

        // Random next state
        self.state = self.rng.random_range(0..2) as f32;

        StepResult {
            observation: self.get_observation(),
            reward,
            terminated,
            truncated: false,
            info: StepInfo::default(),
        }
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo {
            shape: vec![1], // Single value: 0 or 1
            space_type: SpaceType::Box,
        }
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo {
            shape: vec![],
            space_type: SpaceType::Discrete(2), // Two actions: 0 or 1
        }
    }

    fn render(&self) -> Vec<u8> {
        Vec::new()
    }

    fn close(&mut self) {
        // Nothing to clean up
    }

    fn clone_state(&self) -> SimpleBanditState {
        SimpleBanditState { state: self.state, steps: self.steps }
    }

    fn restore_state(&mut self, state: &SimpleBanditState) {
        self.state = state.state;
        self.steps = state.steps;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_bandit_rewards() {
        let mut env = SimpleBandit::new();
        env.reset();

        // Test correct action gives reward 1.0
        let initial_state = env.state as i64;
        let result = env.step(initial_state);
        assert_eq!(result.reward, 1.0);

        // Test incorrect action gives reward 0.0
        env.state = 0.0;
        let result = env.step(1);
        assert_eq!(result.reward, 0.0);

        env.state = 1.0;
        let result = env.step(0);
        assert_eq!(result.reward, 0.0);
    }

    #[test]
    fn clone_restore_round_trips() {
        let mut env = SimpleBandit::new();
        env.reset();

        // Step a few times to a non-initial state.
        for _ in 0..5 {
            env.step(0);
        }
        let snap = env.clone_state();
        let pre_state = env.state;
        let pre_steps = env.steps;

        // Take an experimental step.
        let r1 = env.step(pre_state as i64);

        // Restore and take the same step again.
        env.restore_state(&snap);
        assert_eq!(env.state, pre_state, "state must round-trip");
        assert_eq!(env.steps, pre_steps, "steps must round-trip");

        let r2 = env.step(pre_state as i64);

        // Reward depends only on (state, action) and is fully deterministic.
        assert_eq!(r1.reward, r2.reward);
        assert_eq!(r1.terminated, r2.terminated);
        // observation reflects the *next* random state, which is not
        // reproduced by restore_state (RNG is not snapshotted).
    }

    #[test]
    fn test_episode_length() {
        let mut env = SimpleBandit::new();
        env.reset();

        // Episode should terminate after 100 steps
        for i in 0..99 {
            let result = env.step(0);
            assert!(!result.terminated, "Episode terminated early at step {}", i);
        }

        let result = env.step(0);
        assert!(result.terminated, "Episode should terminate at step 100");
    }
}
