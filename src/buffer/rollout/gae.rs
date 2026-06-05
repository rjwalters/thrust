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
/// * `last_values` - Value estimates for the final states `[num_envs]`
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
    let num_steps = buffer.shape().0;
    compute_advantages_partial(buffer, num_steps, last_values, gamma, gae_lambda);
}

/// Compute Generalized Advantage Estimation (GAE) over the first
/// `valid_steps` rows of the buffer.
///
/// This is the partial-fill counterpart to [`compute_advantages`]. The
/// backward GAE iteration is restricted to `(0..valid_steps).rev()`, so
/// zero-padded tail rows (rows in `valid_steps..num_steps`) do not
/// contaminate the advantage/return signal on the real rows.
///
/// `last_values[env_id]` is the bootstrap `V(s_{T+1})` for the state
/// immediately after row `valid_steps - 1` — exactly the same semantics
/// as for the full-capacity variant, just anchored to the end of the
/// filled prefix rather than the end of capacity.
///
/// # Arguments
/// * `buffer` - Rollout buffer to compute advantages for
/// * `valid_steps` - Number of filled rows at the start of the buffer. Must be
///   `<= buffer.shape().0`.
/// * `last_values` - Value estimates for the final states `[num_envs]`
/// * `gamma` - Discount factor (0 < gamma <= 1)
/// * `gae_lambda` - GAE lambda parameter (0 < lambda <= 1)
///
/// # Panics
/// Panics if `valid_steps > buffer.shape().0`.
pub fn compute_advantages_partial(
    buffer: &mut super::storage::RolloutBuffer,
    valid_steps: usize,
    last_values: &[f32],
    gamma: f32,
    gae_lambda: f32,
) {
    let (num_steps, num_envs) = (buffer.shape().0, buffer.shape().1);

    assert!(
        valid_steps <= num_steps,
        "valid_steps ({}) must not exceed buffer.num_steps ({})",
        valid_steps,
        num_steps
    );
    debug_assert_eq!(last_values.len(), num_envs, "last_values length mismatch");

    if valid_steps == 0 {
        return;
    }

    // Collect all immutable data first to avoid borrow checker issues
    let rewards: Vec<Vec<f32>> = buffer.rewards().iter().map(|step| step.to_vec()).collect();
    let values: Vec<Vec<f32>> = buffer.values().iter().map(|step| step.to_vec()).collect();
    let terminated: Vec<Vec<bool>> = buffer.terminated().iter().map(|step| step.to_vec()).collect();

    // Now we can get mutable access to both advantages and returns
    let (advantages, returns) = buffer.advantages_and_returns_mut();

    // Compute advantages and returns for each environment, restricted to
    // the filled prefix.
    for env_id in 0..num_envs {
        let env_rewards: Vec<f32> =
            rewards.iter().take(valid_steps).map(|step| step[env_id]).collect();
        let env_values: Vec<f32> =
            values.iter().take(valid_steps).map(|step| step[env_id]).collect();
        let env_terminated: Vec<bool> =
            terminated.iter().take(valid_steps).map(|step| step[env_id]).collect();

        let mut env_advantages: Vec<f32> = vec![0.0; valid_steps];
        let mut env_returns: Vec<f32> = vec![0.0; valid_steps];

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

        // Copy results back into the filled prefix only.
        for step in 0..valid_steps {
            advantages[step][env_id] = env_advantages[step];
            returns[step][env_id] = env_returns[step];
        }
    }
}

/// Compute GAE for a single environment
///
/// This is an optimized implementation that processes one environment
/// at a time to improve cache locality.
///
/// Visible to the parent `rollout` module so the sibling `tests` module
/// can exercise it directly.
pub(super) fn compute_gae_single_env(
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

/// Compute Generalized Advantage Estimation (GAE) over a flat,
/// interleaved multi-agent rollout buffer.
///
/// This variant is intended for ad-hoc multi-agent rollout structures
/// that cannot use the `[num_steps, num_envs]`
/// [`crate::buffer::rollout::RolloutBuffer`] layout because they also carry an
/// agent dimension. Multi-agent Snake training uses this shape: each rollout
/// step pushes one entry per (env, agent) pair, in `(env, agent)`-major order,
/// giving a flat buffer of length `num_steps * num_envs * num_agents`.
///
/// # Buffer layout
///
/// All input slices use the same layout and indexing scheme:
///
/// ```text
/// index = t * (num_envs * num_agents) + env * num_agents + agent
/// ```
///
/// where `t` ranges over `[0, num_steps)`, `env` over `[0, num_envs)`,
/// and `agent` over `[0, num_agents)`. The total length must be
/// `num_steps * num_envs * num_agents`.
///
/// `last_values` is a per-(env, agent) bootstrap of length
/// `num_envs * num_agents`, indexed as `env * num_agents + agent`. It
/// supplies `V(s_{T+1})` for the state immediately after the final
/// rollout step.
///
/// # Algorithm
///
/// Independently iterates GAE backwards in time for each (env, agent)
/// trajectory:
///
/// ```text
/// δ_t = r_t + γ * V_{t+1} * (1 - done_t) - V_t
/// A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}
/// ```
///
/// A `done` flag at step `t` zeroes both the bootstrap `V_{t+1}` and
/// the GAE accumulator, so episode boundaries do not leak across
/// trajectories. At the final step (`t == num_steps - 1`), the bootstrap
/// is taken from `last_values` when the trajectory has not terminated
/// and from `0.0` when it has.
///
/// # Arguments
/// * `rewards` - Per-(t, env, agent) immediate rewards
/// * `values` - Per-(t, env, agent) value estimates from the policy
/// * `dones` - Per-(t, env, agent) episode termination flags
/// * `last_values` - Per-(env, agent) bootstrap `V(s_{T+1})`
/// * `num_envs` - Number of parallel environments in the rollout
/// * `num_agents` - Number of agents per environment
/// * `gamma` - Discount factor (0 < gamma <= 1)
/// * `gae_lambda` - GAE lambda parameter (0 < lambda <= 1)
///
/// # Returns
/// `(advantages, returns)` as flat `Vec<f32>` matching the input layout.
///
/// # Panics
/// Panics if input slice lengths are inconsistent with
/// `num_steps * num_envs * num_agents` (computed from
/// `rewards.len()`), or if `last_values.len() != num_envs * num_agents`.
pub fn compute_advantages_multi_agent(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    last_values: &[f32],
    num_envs: usize,
    num_agents: usize,
    gamma: f32,
    gae_lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let stride = num_envs * num_agents;
    assert!(stride > 0, "num_envs * num_agents must be > 0");
    assert_eq!(
        rewards.len() % stride,
        0,
        "rewards.len() ({}) must be divisible by num_envs * num_agents ({})",
        rewards.len(),
        stride
    );
    assert_eq!(values.len(), rewards.len(), "values length mismatch");
    assert_eq!(dones.len(), rewards.len(), "dones length mismatch");
    assert_eq!(
        last_values.len(),
        stride,
        "last_values length must equal num_envs * num_agents ({})",
        stride
    );

    let num_steps = rewards.len() / stride;
    let total = rewards.len();
    let mut advantages = vec![0.0_f32; total];
    let mut returns = vec![0.0_f32; total];

    // Independently roll up GAE backwards for each (env, agent)
    // trajectory. This is the multi-agent analog of
    // `compute_gae_single_env`.
    for env in 0..num_envs {
        for agent in 0..num_agents {
            let slot = env * num_agents + agent;
            let bootstrap = last_values[slot];
            let mut gae = 0.0_f32;

            for t in (0..num_steps).rev() {
                let idx = t * stride + slot;
                let done = dones[idx];
                let next_value = if t == num_steps - 1 {
                    // Final step: bootstrap from post-rollout estimate
                    // unless the trajectory terminated here.
                    if done { 0.0 } else { bootstrap }
                } else if done {
                    // Episode ended: no bootstrap into the next step.
                    0.0
                } else {
                    values[idx + stride]
                };

                let mask = if done { 0.0_f32 } else { 1.0_f32 };
                let delta = rewards[idx] + gamma * next_value * mask - values[idx];
                // When the current step is terminal, drop the accumulated
                // GAE from the next step — otherwise it would leak across
                // an episode boundary.
                gae = delta + gamma * gae_lambda * mask * gae;

                advantages[idx] = gae;
                returns[idx] = values[idx] + gae;
            }
        }
    }

    (advantages, returns)
}

/// Compute n-step returns (simpler alternative to GAE)
///
/// This computes un-discounted n-step returns without advantage normalization.
/// Useful for debugging or when GAE is not needed.
///
/// # Arguments
/// * `buffer` - Rollout buffer to compute returns for
/// * `last_values` - Value estimates for final states `[num_envs]`
/// * `gamma` - Discount factor
pub fn compute_nstep_returns(
    buffer: &mut super::storage::RolloutBuffer,
    last_values: &[f32],
    gamma: f32,
) {
    let (num_steps, num_envs) = (buffer.shape().0, buffer.shape().1);

    debug_assert_eq!(last_values.len(), num_envs, "last_values length mismatch");

    // Collect immutable data first
    let rewards: Vec<Vec<f32>> = buffer.rewards().iter().map(|step| step.to_vec()).collect();
    let terminated: Vec<Vec<bool>> = buffer.terminated().iter().map(|step| step.to_vec()).collect();

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
    let rewards: Vec<Vec<f32>> = buffer.rewards().iter().map(|step| step.to_vec()).collect();
    let terminated: Vec<Vec<bool>> = buffer.terminated().iter().map(|step| step.to_vec()).collect();

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
    let variance: f32 = all_advantages.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
        / all_advantages.len() as f32;
    let std = variance.sqrt().max(1e-8); // Avoid division by zero

    // Normalize advantages
    let advantages = buffer.advantages_mut();
    for step in 0..num_steps {
        for env in 0..num_envs {
            advantages[step][env] = (advantages[step][env] - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::rollout::storage::RolloutBuffer;

    /// `compute_advantages_partial` over the first `valid_steps` rows must
    /// match the values that `compute_advantages` would produce on a
    /// shorter buffer containing only those rows. This locks in the
    /// invariant that the partial variant is purely a loop-bound change
    /// and not a math change.
    #[test]
    fn test_compute_advantages_partial_matches_full_on_prefix() {
        let valid_steps = 3usize;
        let total_capacity = 8usize;
        let num_envs = 1usize;
        let obs_dim = 1usize;

        // Build a full-capacity buffer where rows 0..valid_steps carry
        // real data and rows valid_steps..total_capacity are left at the
        // zero-initialized default.
        let mut partial_buffer = RolloutBuffer::new(total_capacity, num_envs, obs_dim);
        let rewards = [1.0_f32, 0.5, -0.25];
        let values = [0.4_f32, 0.6, 0.8];
        let log_probs = [-0.1_f32, -0.2, -0.3];
        for step in 0..valid_steps {
            partial_buffer.add(
                step,
                0,
                &[0.0],
                0,
                rewards[step],
                values[step],
                log_probs[step],
                false,
                false,
            );
        }

        // Build a tight buffer with only `valid_steps` rows holding the
        // same data.
        let mut tight_buffer = RolloutBuffer::new(valid_steps, num_envs, obs_dim);
        for step in 0..valid_steps {
            tight_buffer.add(
                step,
                0,
                &[0.0],
                0,
                rewards[step],
                values[step],
                log_probs[step],
                false,
                false,
            );
        }

        let last_values = vec![0.7_f32];
        let gamma = 0.99_f32;
        let gae_lambda = 0.95_f32;

        compute_advantages_partial(
            &mut partial_buffer,
            valid_steps,
            &last_values,
            gamma,
            gae_lambda,
        );
        compute_advantages(&mut tight_buffer, &last_values, gamma, gae_lambda);

        // Filled prefix: partial and tight buffers must agree exactly.
        for step in 0..valid_steps {
            let p_adv = partial_buffer.advantages()[step][0];
            let t_adv = tight_buffer.advantages()[step][0];
            let p_ret = partial_buffer.returns()[step][0];
            let t_ret = tight_buffer.returns()[step][0];
            assert!(
                (p_adv - t_adv).abs() < 1e-6_f32,
                "advantage mismatch at step {}: partial={}, tight={}",
                step,
                p_adv,
                t_adv
            );
            assert!(
                (p_ret - t_ret).abs() < 1e-6_f32,
                "return mismatch at step {}: partial={}, tight={}",
                step,
                p_ret,
                t_ret
            );
        }

        // Unwritten tail: advantages and returns must remain at the
        // zero-initialized default. If `compute_advantages_partial` had
        // accidentally written into the tail, this would catch it.
        for step in valid_steps..total_capacity {
            assert_eq!(partial_buffer.advantages()[step][0], 0.0);
            assert_eq!(partial_buffer.returns()[step][0], 0.0);
        }
    }

    /// Direct regression for the second contamination point flagged on
    /// issue #28: with the old full-capacity GAE, a partially-filled
    /// buffer with a non-zero bootstrap propagates `γ^k * last_value`
    /// backward through the zero-padded tail and into the real prefix.
    /// `compute_advantages_partial` must not exhibit that behavior.
    #[test]
    fn test_compute_advantages_partial_does_not_leak_bootstrap_through_padding() {
        let valid_steps = 2usize;
        let total_capacity = 16usize;
        let num_envs = 1usize;

        let mut buffer = RolloutBuffer::new(total_capacity, num_envs, 1);
        // Two real rows, neither terminal. With a single step left after
        // index 1, the bootstrap should hit at row 1 (index valid_steps - 1).
        buffer.add(0, 0, &[0.0], 0, 0.0, 0.0, 0.0, false, false);
        buffer.add(1, 0, &[0.0], 0, 0.0, 0.0, 0.0, false, false);

        let bootstrap = 1.0_f32;
        let last_values = vec![bootstrap];
        let gamma = 0.99_f32;
        let gae_lambda = 0.95_f32;

        compute_advantages_partial(&mut buffer, valid_steps, &last_values, gamma, gae_lambda);

        // Expected (closed form, single env, no terminations):
        //   delta_1 = 0 + γ * bootstrap - 0 = γ * bootstrap
        //   gae_1   = delta_1                = γ * bootstrap
        //   delta_0 = 0 + γ * V_1 - 0        = 0  (since V_1 = 0)
        //   gae_0   = 0 + γ * λ * gae_1      = γ^2 * λ * bootstrap
        let expected_adv_1 = gamma * bootstrap;
        let expected_adv_0 = gamma * gamma * gae_lambda * bootstrap;

        let a1 = buffer.advantages()[1][0];
        let a0 = buffer.advantages()[0][0];
        assert!(
            (a1 - expected_adv_1).abs() < 1e-6_f32,
            "advantage[1] expected {}, got {}",
            expected_adv_1,
            a1
        );
        assert!(
            (a0 - expected_adv_0).abs() < 1e-6_f32,
            "advantage[0] expected {} (γ²λ * bootstrap), got {}. \
             If this is closer to γ^15 * bootstrap, the GAE iteration is \
             still walking through padded rows.",
            expected_adv_0,
            a0
        );

        // Tail must stay zero.
        for step in valid_steps..total_capacity {
            assert_eq!(buffer.advantages()[step][0], 0.0);
        }
    }

    /// `valid_steps == 0` is a defensive edge case — the function should
    /// be a no-op (no panics, no writes to advantages/returns).
    #[test]
    fn test_compute_advantages_partial_zero_valid_is_noop() {
        let mut buffer = RolloutBuffer::new(4, 1, 1);
        buffer.add(0, 0, &[0.0], 0, 1.0, 0.5, 0.0, false, false);
        // Pre-stamp the advantage so we can detect any rogue write.
        buffer.advantages_mut()[0][0] = 99.0;

        compute_advantages_partial(&mut buffer, 0, &[0.0], 0.99, 0.95);

        assert_eq!(buffer.advantages()[0][0], 99.0);
    }

    /// Passing `valid_steps > num_steps` must panic with a clear message.
    #[test]
    #[should_panic(expected = "valid_steps")]
    fn test_compute_advantages_partial_panics_on_overflow() {
        let mut buffer = RolloutBuffer::new(4, 1, 1);
        compute_advantages_partial(&mut buffer, 5, &[0.0], 0.99, 0.95);
    }
}
