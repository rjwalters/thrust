//! Vectorized environment pool for parallel execution
//!
//! This module provides high-performance parallel environment execution using
//! Rayon. Inspired by EnvPool (<https://arxiv.org/abs/2206.10558>), it achieves significant
//! speedups by executing multiple environments in parallel.
//!
//! # Example
//!
//! ```rust
//! use thrust_rl::env::{cartpole::CartPole, pool::EnvPool};
//!
//! // Create pool with 4 parallel environments
//! let mut pool = EnvPool::new(|| CartPole::new(), 4);
//!
//! // Reset all environments in parallel
//! let observations = pool.reset();
//!
//! // Step all environments in parallel
//! let actions = vec![0, 1, 0, 1]; // One action per environment
//! let results = pool.step(&actions);
//! ```

use rayon::prelude::*;

use crate::env::{Environment, StepResult};

/// A pool of environments for parallel execution
///
/// EnvPool manages multiple environment instances and executes operations
/// across them in parallel using Rayon's thread pool. This provides
/// significant performance improvements over sequential execution.
///
/// # Performance
///
/// For N environments with average step time T:
/// - Sequential: O(N * T)
/// - Parallel: O(max(T)) ≈ O(T) when N ≤ num_cores
///
/// This can provide 10-100x speedups depending on environment complexity
/// and number of CPU cores available.
pub struct EnvPool<E: Environment> {
    /// Vector of environment instances
    envs: Vec<E>,

    /// Number of environments
    num_envs: usize,
}

impl<E: Environment + Send> EnvPool<E> {
    /// Create a new environment pool
    ///
    /// # Arguments
    ///
    /// * `env_fn` - Factory function to create environment instances
    /// * `num_envs` - Number of parallel environments
    ///
    /// # Example
    ///
    /// ```rust
    /// use thrust_rl::env::{cartpole::CartPole, pool::EnvPool};
    ///
    /// let pool = EnvPool::new(|| CartPole::new(), 8);
    /// ```
    pub fn new<F>(env_fn: F, num_envs: usize) -> Self
    where
        F: Fn() -> E,
    {
        let envs = (0..num_envs).map(|_| env_fn()).collect();
        Self { envs, num_envs }
    }

    /// Reset all environments in parallel
    ///
    /// Returns a vector of initial observations, one per environment.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use thrust_rl::env::pool::EnvPool;
    /// # use thrust_rl::env::cartpole::CartPole;
    /// # let mut pool = EnvPool::new(|| CartPole::new(), 4);
    /// let observations = pool.reset();
    /// assert_eq!(observations.len(), 4);
    /// ```
    pub fn reset(&mut self) -> Vec<Vec<f32>>
    where
        E: Send,
    {
        use rayon::iter::ParallelIterator;
        self.envs
            .par_iter_mut()
            .map(|env| {
                env.reset();
                env.get_observation()
            })
            .collect()
    }

    /// Step all environments in parallel with given actions
    ///
    /// # Arguments
    ///
    /// * `actions` - Slice of actions, one per environment
    ///
    /// # Panics
    ///
    /// Panics if the number of actions doesn't match the number of
    /// environments.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use thrust_rl::env::pool::EnvPool;
    /// # use thrust_rl::env::cartpole::CartPole;
    /// # let mut pool = EnvPool::new(|| CartPole::new(), 4);
    /// # pool.reset();
    /// let actions = vec![0, 1, 0, 1];
    /// let results = pool.step(&actions);
    /// assert_eq!(results.len(), 4);
    /// ```
    pub fn step(&mut self, actions: &[i64]) -> Vec<StepResult>
    where
        E: Send,
    {
        use rayon::iter::ParallelIterator;
        assert_eq!(
            actions.len(),
            self.num_envs,
            "Number of actions must match number of environments"
        );

        self.envs
            .par_iter_mut()
            .zip(actions.par_iter())
            .map(|(env, &action)| env.step(action))
            .collect()
    }

    /// Get the number of environments in the pool
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get observation space information from first environment
    pub fn observation_space(&self) -> crate::env::SpaceInfo {
        self.envs[0].observation_space()
    }

    /// Get action space information from first environment
    pub fn action_space(&self) -> crate::env::SpaceInfo {
        self.envs[0].action_space()
    }

    /// Reset a specific environment by index
    ///
    /// # Arguments
    ///
    /// * `env_id` - Index of environment to reset
    ///
    /// # Returns
    ///
    /// Initial observation from the reset environment
    pub fn reset_env(&mut self, env_id: usize) -> anyhow::Result<Vec<f32>> {
        self.envs[env_id].reset();
        Ok(self.envs[env_id].get_observation())
    }
}

/// Result of stepping an environment pool
///
/// Contains observations, rewards, and done flags for all environments.
#[derive(Debug, Clone)]
pub struct PoolStepResult<O> {
    /// Observations for each environment
    pub observations: Vec<O>,

    /// Rewards for each environment
    pub rewards: Vec<f32>,

    /// Termination flags for each environment
    pub terminated: Vec<bool>,

    /// Truncation flags for each environment
    pub truncated: Vec<bool>,
}

impl<E: Environment + Send> EnvPool<E> {
    /// Step all environments and return structured result
    ///
    /// This is a convenience method that unpacks individual StepResults
    /// into a single PoolStepResult with parallel vectors.
    pub fn step_structured(&mut self, actions: &[i64]) -> PoolStepResult<Vec<f32>> {
        let results = self.step(actions);

        let mut observations = Vec::with_capacity(self.num_envs);
        let mut rewards = Vec::with_capacity(self.num_envs);
        let mut terminated = Vec::with_capacity(self.num_envs);
        let mut truncated = Vec::with_capacity(self.num_envs);

        for result in results {
            observations.push(result.observation);
            rewards.push(result.reward);
            terminated.push(result.terminated);
            truncated.push(result.truncated);
        }

        PoolStepResult { observations, rewards, terminated, truncated }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::cartpole::CartPole;

    #[test]
    fn test_pool_creation() {
        let pool = EnvPool::new(CartPole::new, 4);
        assert_eq!(pool.num_envs(), 4);
    }

    #[test]
    fn test_pool_reset() {
        let mut pool = EnvPool::new(CartPole::new, 4);
        let observations = pool.reset();

        assert_eq!(observations.len(), 4);
        for obs in observations {
            assert_eq!(obs.len(), 4); // CartPole has 4D observations
        }
    }

    #[test]
    fn test_pool_step() {
        let mut pool = EnvPool::new(CartPole::new, 4);
        pool.reset();

        let actions = vec![0, 1, 0, 1];
        let results = pool.step(&actions);

        assert_eq!(results.len(), 4);
        for result in results {
            assert_eq!(result.observation.len(), 4);
            assert!(result.reward == 0.0 || result.reward == 1.0);
        }
    }

    #[test]
    fn test_pool_step_structured() {
        let mut pool = EnvPool::new(CartPole::new, 4);
        pool.reset();

        let actions = vec![0, 1, 0, 1];
        let result = pool.step_structured(&actions);

        assert_eq!(result.observations.len(), 4);
        assert_eq!(result.rewards.len(), 4);
        assert_eq!(result.terminated.len(), 4);
        assert_eq!(result.truncated.len(), 4);
    }

    #[test]
    #[should_panic(expected = "Number of actions must match number of environments")]
    fn test_pool_step_wrong_action_count() {
        let mut pool = EnvPool::new(CartPole::new, 4);
        pool.reset();

        let actions = vec![0, 1]; // Wrong number of actions
        pool.step(&actions);
    }

    #[test]
    fn test_pool_multiple_steps() {
        let mut pool = EnvPool::new(CartPole::new, 4);
        pool.reset();

        // Run multiple steps
        for _ in 0..10 {
            let actions = vec![0, 1, 0, 1];
            let results = pool.step(&actions);
            assert_eq!(results.len(), 4);
        }
    }

    #[test]
    fn test_pool_observation_space() {
        let pool = EnvPool::new(CartPole::new, 4);
        let obs_space = pool.observation_space();
        assert_eq!(obs_space.shape, vec![4]);
    }

    #[test]
    fn test_pool_action_space() {
        let pool = EnvPool::new(CartPole::new, 4);
        let action_space = pool.action_space();
        assert_eq!(action_space.shape, Vec::<usize>::new());
    }

    #[test]
    fn test_pool_large_batch() {
        // Test with larger batch to verify parallelism
        let mut pool = EnvPool::new(CartPole::new, 16);
        let observations = pool.reset();
        assert_eq!(observations.len(), 16);

        let actions = vec![0; 16];
        let results = pool.step(&actions);
        assert_eq!(results.len(), 16);
    }

    #[test]
    fn test_pool_alternating_actions() {
        let mut pool = EnvPool::new(CartPole::new, 8);
        pool.reset();

        // Test alternating left/right actions
        for i in 0..5 {
            let actions: Vec<i64> = (0..8).map(|j| ((i + j) % 2) as i64).collect();
            let results = pool.step(&actions);

            for result in results {
                // Should not immediately terminate with alternating actions
                if i == 0 {
                    assert!(!result.terminated);
                }
            }
        }
    }
}
