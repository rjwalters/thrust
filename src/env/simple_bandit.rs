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

use super::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

/// Simple contextual bandit for testing PPO correctness
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
        Self { state: 0.0, steps: 0, max_steps: 100, rng: rand::rngs::StdRng::from_entropy() }
    }
}

impl Default for SimpleBandit {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for SimpleBandit {
    fn reset(&mut self) {
        // Random start state (0 or 1)
        self.state = self.rng.gen_range(0..2) as f32;
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
        self.state = self.rng.gen_range(0..2) as f32;

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
