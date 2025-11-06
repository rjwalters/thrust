//! Loss computation functions for PPO
//!
//! This module contains the core loss computation functions used
//! in PPO training including policy loss, value loss, and entropy loss.

use tch::Tensor;

/// Compute PPO policy loss with clipping
///
/// Returns (policy_loss, clip_fraction, approx_kl)
///
/// # Arguments
/// * `log_probs` - Log probabilities of actions under current policy
/// * `old_log_probs` - Log probabilities of actions under old policy
/// * `advantages` - Computed advantages
/// * `clip_range` - PPO clipping parameter (epsilon)
pub fn compute_policy_loss(
    log_probs: &Tensor,
    old_log_probs: &Tensor,
    advantages: &Tensor,
    clip_range: f64,
) -> (Tensor, f64, f64) {
    // Compute probability ratio
    let ratio = (log_probs - old_log_probs).exp();

    // Compute clipped surrogate objective
    let clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range);
    let policy_loss_1 = advantages * ratio;
    let policy_loss_2 = advantages * clipped_ratio;
    let policy_loss = -policy_loss_1.minimum(&policy_loss_2).mean(0);

    // Compute fraction of clipped updates
    let clip_fraction = (ratio - 1.0).abs().greater(&clip_range).to_kind(tch::Kind::Float).mean(0);

    // Approximate KL divergence for early stopping
    let approx_kl = (old_log_probs - log_probs).mean(0);

    (
        policy_loss,
        f64::try_from(&clip_fraction).unwrap_or(0.0),
        f64::try_from(&approx_kl).unwrap_or(0.0),
    )
}

/// Compute value function loss with optional clipping
///
/// Returns (value_loss, explained_variance)
///
/// # Arguments
/// * `values` - Predicted values under current value function
/// * `old_values` - Predicted values under old value function
/// * `returns` - Computed returns (targets)
/// * `clip_range_vf` - Value function clipping parameter
pub fn compute_value_loss(
    values: &Tensor,
    old_values: &Tensor,
    returns: &Tensor,
    clip_range_vf: f64,
) -> (Tensor, f64) {
    let values_clipped = old_values + (values - old_values).clamp(-clip_range_vf, clip_range_vf);
    let vf_loss_1 = (values - returns).square();
    let vf_loss_2 = (values_clipped - returns).square();
    let value_loss = vf_loss_1.minimum(&vf_loss_2).mean(0);

    // Compute explained variance
    let var_returns = returns.var(false);
    let explained_var = if var_returns.equal(0.0) {
        1.0 // Perfect prediction if no variance in returns
    } else {
        1.0 - (returns - values).var(false) / var_returns
    };

    (
        value_loss,
        f64::try_from(&explained_var).unwrap_or(0.0),
    )
}

/// Compute entropy loss (negative entropy for maximization)
///
/// # Arguments
/// * `entropy` - Entropy tensor from policy distribution
pub fn compute_entropy_loss(entropy: &Tensor) -> Tensor {
    -entropy.mean(0)
}

/// Generate minibatch indices for PPO training
///
/// Creates shuffled minibatches from a buffer of the given size.
/// Each minibatch contains approximately `batch_size` samples.
///
/// # Arguments
/// * `buffer_size` - Total number of samples in buffer
/// * `batch_size` - Desired size of each minibatch
///
/// # Returns
/// Vector of vectors, where each inner vector contains indices for one minibatch
pub fn generate_minibatch_indices(buffer_size: usize, batch_size: usize) -> Vec<Vec<usize>> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let mut indices: Vec<usize> = (0..buffer_size).collect();
    indices.shuffle(&mut thread_rng());

    indices
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Compute Generalized Advantage Estimation (GAE)
///
/// # Arguments
/// * `rewards` - Reward tensor [num_steps, num_envs]
/// * `values` - Value estimates [num_steps, num_envs]
/// * `next_values` - Next step value estimates [num_envs]
/// * `dones` - Done flags [num_steps, num_envs]
/// * `gamma` - Discount factor
/// * `gae_lambda` - GAE lambda parameter
///
/// # Returns
/// (advantages, returns) tensors
pub fn compute_gae(
    rewards: &Tensor,
    values: &Tensor,
    next_values: &Tensor,
    dones: &Tensor,
    gamma: f64,
    gae_lambda: f64,
) -> (Tensor, Tensor) {
    let mut advantages = Vec::new();
    let mut returns = Vec::new();

    let num_steps = rewards.size()[0];
    let num_envs = rewards.size()[1];

    for env_id in 0..num_envs {
        let env_rewards = rewards.slice(1, env_id, env_id + 1, 1).squeeze_dim(1);
        let env_values = values.slice(1, env_id, env_id + 1, 1).squeeze_dim(1);
        let env_next_values = next_values.slice(0, env_id, env_id + 1, 1).squeeze_dim(0);
        let env_dones = dones.slice(1, env_id, env_id + 1, 1).squeeze_dim(1);

        let (env_advantages, env_returns) = compute_gae_single_env(
            &env_rewards,
            &env_values,
            &env_next_values,
            &env_dones,
            gamma,
            gae_lambda,
        );

        advantages.push(env_advantages.unsqueeze(1));
        returns.push(env_returns.unsqueeze(1));
    }

    let advantages = Tensor::cat(&advantages, 1);
    let returns = Tensor::cat(&returns, 1);

    (advantages, returns)
}

/// Compute GAE for a single environment
fn compute_gae_single_env(
    rewards: &Tensor,
    values: &Tensor,
    next_value: &Tensor,
    dones: &Tensor,
    gamma: f64,
    gae_lambda: f64,
) -> (Tensor, Tensor) {
    let mut advantages = Vec::new();
    let mut returns = Vec::new();
    let mut last_gae = 0.0;

    let rewards_vec: Vec<f32> = rewards.into();
    let values_vec: Vec<f32> = values.into();
    let dones_vec: Vec<f32> = dones.into();
    let next_value_scalar: f32 = next_value.into();

    // Iterate backwards through the trajectory
    for t in (0..rewards_vec.len()).rev() {
        let reward = rewards_vec[t];
        let value = values_vec[t];
        let done = dones_vec[t];

        if t == rewards_vec.len() - 1 {
            // Last step
            let next_value = if done == 1.0 { 0.0 } else { next_value_scalar };
            let delta = reward + gamma * next_value - value;
            last_gae = delta;
        } else {
            // Bootstrap from next advantage
            let next_value = if done == 1.0 { 0.0 } else { values_vec[t + 1] };
            let delta = reward + gamma * next_value - value;
            last_gae = delta + gamma * gae_lambda * last_gae;
        }

        advantages.push(last_gae);
        returns.push(value + last_gae);
    }

    advantages.reverse();
    returns.reverse();

    (
        Tensor::from_slice(&advantages),
        Tensor::from_slice(&returns),
    )
}
