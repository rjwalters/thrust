//! Generalized Advantage Estimation (GAE) computation
//!
//! This module implements GAE for computing advantages from trajectories.
//! GAE helps reduce variance in policy gradient methods while maintaining
//! sufficient bias for learning.

/// Compute Generalized Advantage Estimation (GAE)
///
/// GAE computes advantages using a weighted sum of n-step returns,
/// providing a balance between bias and variance.
///
/// # Arguments
/// * `buffer` - Rollout buffer to compute advantages for
/// * `last_values` - Value estimates for the final states [num_envs]
/// * `gamma` - Discount factor (0 < gamma <= 1)
/// * `gae_lambda` - GAE lambda parameter (0 < lambda <= 1)
///
/// # Mathematical Formula
/// ```text
/// δ_t = r_t + γ * V_{t+1} - V_t
/// A_t = δ_t + γ * λ * A_{t+1}
/// ```
///
/// Where:
/// - δ_t is the temporal difference error
/// - A_t is the advantage estimate
/// - r_t is the reward at time t
/// - V_t is the value estimate at time t
/// - γ is the discount factor
/// - λ is the GAE parameter
pub fn compute_advantages(
    buffer: &mut super::storage::RolloutBuffer,
    last_values: &[f32],
    gamma: f32,
    gae_lambda: f32,
) {
    let (num_steps, num_envs) = (buffer.shape().0, buffer.shape().1);

    debug_assert_eq!(last_values.len(), num_envs, "last_values length mismatch");

    // Collect all immutable data first to avoid borrow checker issues
    let rewards: Vec<Vec<f32>> = buffer.rewards().iter()
        .map(|step| step.to_vec())
        .collect();
    let values: Vec<Vec<f32>> = buffer.values().iter()
        .map(|step| step.to_vec())
        .collect();
    let terminated: Vec<Vec<bool>> = buffer.terminated().iter()
        .map(|step| step.to_vec())
        .collect();

    // Now we can get mutable access to both advantages and returns
    let (advantages, returns) = buffer.advantages_and_returns_mut();

    // Compute advantages and returns for each environment
    for env_id in 0..num_envs {
        let env_rewards: Vec<f32> = rewards.iter().map(|step| step[env_id]).collect();
        let env_values: Vec<f32> = values.iter().map(|step| step[env_id]).collect();
        let env_terminated: Vec<bool> = terminated.iter().map(|step| step[env_id]).collect();

        let mut env_advantages: Vec<f32> = vec![0.0; num_steps];
        let mut env_returns: Vec<f32> = vec![0.0; num_steps];

        compute_gae_single_env(
            &env_rewards,
            &env_values,
            &env_terminated,
            last_values[env_id],
            gamma,
            gae_lambda,
            &mut env_advantages,
            &mut env_returns,
        );

        // Copy results back
        for step in 0..num_steps {
            advantages[step][env_id] = env_advantages[step];
            returns[step][env_id] = env_returns[step];
        }
    }
}

/// Compute GAE for a single environment
///
/// This is an optimized implementation that processes one environment
/// at a time to improve cache locality.
fn compute_gae_single_env(
    rewards: &[f32],
    values: &[f32],
    terminated: &[bool],
    last_value: f32,
    gamma: f32,
    gae_lambda: f32,
    advantages: &mut [f32],
    returns: &mut [f32],
) {
    let num_steps = rewards.len();
    debug_assert_eq!(values.len(), num_steps);
    debug_assert_eq!(terminated.len(), num_steps);
    debug_assert_eq!(advantages.len(), num_steps);
    debug_assert_eq!(returns.len(), num_steps);

    // Compute advantages backwards (GAE algorithm)
    let mut gae = 0.0;

    for t in (0..num_steps).rev() {
        // Reset GAE when crossing episode boundary (backwards iteration)
        // If step t is terminal, reset GAE before computing its advantage
        // This prevents accumulating GAE from future episodes
        if terminated[t] {
            gae = 0.0;
        }

        // Bootstrap from next value if not terminated
        let next_value = if t == num_steps - 1 {
            // Last step: bootstrap from final value estimate
            last_value
        } else if terminated[t] {
            // Episode ended: no bootstrap
            0.0
        } else {
            // Continue episode: bootstrap from next value
            values[t + 1]
        };

        // Compute temporal difference error
        let delta = rewards[t] + gamma * next_value - values[t];

        // Compute GAE advantage
        gae = delta + gamma * gae_lambda * gae;

        // Store results
        advantages[t] = gae;
        returns[t] = values[t] + gae;
    }
}

/// Compute n-step returns (simpler alternative to GAE)
///
/// This computes un-discounted n-step returns without advantage normalization.
/// Useful for debugging or when GAE is not needed.
///
/// # Arguments
/// * `buffer` - Rollout buffer to compute returns for
/// * `last_values` - Value estimates for final states [num_envs]
/// * `gamma` - Discount factor
pub fn compute_nstep_returns(
    buffer: &mut super::storage::RolloutBuffer,
    last_values: &[f32],
    gamma: f32,
) {
    let (num_steps, num_envs) = (buffer.shape().0, buffer.shape().1);

    debug_assert_eq!(last_values.len(), num_envs, "last_values length mismatch");

    // Collect immutable data first
    let rewards: Vec<Vec<f32>> = buffer.rewards().iter()
        .map(|step| step.to_vec())
        .collect();
    let terminated: Vec<Vec<bool>> = buffer.terminated().iter()
        .map(|step| step.to_vec())
        .collect();

    // Now get mutable access
    let returns = buffer.returns_mut();

    for env_id in 0..num_envs {
        let mut discounted_return = last_values[env_id];

        // Compute returns backwards
        for step in (0..num_steps).rev() {
            if terminated[step][env_id] {
                discounted_return = 0.0;
            }

            discounted_return = rewards[step][env_id] + gamma * discounted_return;
            returns[step][env_id] = discounted_return;
        }
    }
}

/// Compute Monte Carlo returns (full episode returns)
///
/// Computes returns using the full trajectory, which provides unbiased
/// but high-variance estimates. Only works for complete episodes.
///
/// # Arguments
/// * `buffer` - Rollout buffer to compute returns for
pub fn compute_mc_returns(buffer: &mut super::storage::RolloutBuffer) {
    let (num_steps, num_envs) = (buffer.shape().0, buffer.shape().1);

    // Collect immutable data first
    let rewards: Vec<Vec<f32>> = buffer.rewards().iter()
        .map(|step| step.to_vec())
        .collect();
    let terminated: Vec<Vec<bool>> = buffer.terminated().iter()
        .map(|step| step.to_vec())
        .collect();

    // Now get mutable access
    let returns = buffer.returns_mut();

    for env_id in 0..num_envs {
        let mut episode_return = 0.0;

        // Find episode boundaries (where terminated is true)
        let mut episode_start = 0;

        for step in 0..num_steps {
            episode_return += rewards[step][env_id];

            if terminated[step][env_id] || step == num_steps - 1 {
                // Episode ended - assign return to all steps in episode
                for s in episode_start..=step {
                    returns[s][env_id] = episode_return;
                }

                // Start new episode
                episode_return = 0.0;
                episode_start = step + 1;
            }
        }
    }
}

/// Normalize advantages across the entire buffer
///
/// This performs advantage normalization as described in the PPO paper,
/// which helps with training stability.
///
/// # Arguments
/// * `buffer` - Rollout buffer with computed advantages
pub fn normalize_advantages(buffer: &mut super::storage::RolloutBuffer) {
    let (num_steps, num_envs) = (buffer.shape().0, buffer.shape().1);

    // Collect all advantages into a single vector
    let mut all_advantages = Vec::with_capacity(num_steps * num_envs);
    for step in 0..num_steps {
        for env in 0..num_envs {
            all_advantages.push(buffer.advantages()[step][env]);
        }
    }

    // Compute mean and std
    let mean: f32 = all_advantages.iter().sum::<f32>() / all_advantages.len() as f32;
    let variance: f32 = all_advantages.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / all_advantages.len() as f32;
    let std = variance.sqrt().max(1e-8); // Avoid division by zero

    // Normalize advantages
    let advantages = buffer.advantages_mut();
    for step in 0..num_steps {
        for env in 0..num_envs {
            advantages[step][env] = (advantages[step][env] - mean) / std;
        }
    }
}
