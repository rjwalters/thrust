//! Train PPO on Single-Agent Snake
//!
//! This example trains a single snake agent using PPO with a CNN policy.
//! We start with single-agent to validate the environment and learning
//! before scaling to multi-agent scenarios.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_snake --release
//! ```

use anyhow::Result;
use tch::{Device, Tensor, nn, nn::OptimizerConfig};
use thrust_rl::{
    env::{Environment, snake::SnakeEnv},
    policy::snake_cnn::SnakeCNN,
};

/// Rollout buffer for storing trajectories
struct RolloutBuffer {
    observations: Vec<Vec<f32>>,
    actions: Vec<i64>,
    log_probs: Vec<f32>,
    rewards: Vec<f32>,
    values: Vec<f32>,
    dones: Vec<bool>,
}

impl RolloutBuffer {
    fn new() -> Self {
        Self {
            observations: Vec::new(),
            actions: Vec::new(),
            log_probs: Vec::new(),
            rewards: Vec::new(),
            values: Vec::new(),
            dones: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.observations.clear();
        self.actions.clear();
        self.log_probs.clear();
        self.rewards.clear();
        self.values.clear();
        self.dones.clear();
    }

    fn add(
        &mut self,
        obs: Vec<f32>,
        action: i64,
        log_prob: f32,
        reward: f32,
        value: f32,
        done: bool,
    ) {
        self.observations.push(obs);
        self.actions.push(action);
        self.log_probs.push(log_prob);
        self.rewards.push(reward);
        self.values.push(value);
        self.dones.push(done);
    }

    fn len(&self) -> usize {
        self.observations.len()
    }
}

/// Compute advantages using GAE
fn compute_advantages(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    gamma: f64,
    lambda: f64,
) -> (Vec<f32>, Vec<f32>) {
    let mut advantages = vec![0.0; rewards.len()];
    let mut returns = vec![0.0; rewards.len()];

    let mut gae = 0.0;
    for t in (0..rewards.len()).rev() {
        let next_value = if t + 1 < values.len() && !dones[t] {
            values[t + 1]
        } else {
            0.0
        };

        let delta = rewards[t] + (gamma as f32) * next_value - values[t];
        gae = delta + (gamma * lambda) as f32 * gae * if dones[t] { 0.0 } else { 1.0 };
        advantages[t] = gae;
        returns[t] = advantages[t] + values[t];
    }

    (advantages, returns)
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt().with_env_filter("info").init();

    tracing::info!("🐍 Starting Single-Agent Snake PPO Training");

    // Hyperparameters
    const NUM_ENVS: usize = 16; // Parallel environments
    const NUM_STEPS: usize = 512; // Steps per rollout
    const TOTAL_TIMESTEPS: usize = 2_000_000; // 2M timesteps
    const LEARNING_RATE: f64 = 0.0003;
    const GRID_WIDTH: i32 = 20;
    const GRID_HEIGHT: i32 = 20;
    const GAMMA: f64 = 0.99;
    const GAE_LAMBDA: f64 = 0.95;
    const CLIP_PARAM: f64 = 0.2;
    const VALUE_COEF: f64 = 0.5;
    const ENTROPY_COEF: f64 = 0.01;
    const PPO_EPOCHS: usize = 4;
    const MINIBATCH_SIZE: usize = 64;

    // Create single-agent snake environment (1 snake)
    let env = SnakeEnv::new(GRID_WIDTH, GRID_HEIGHT);

    // Grid observations are 5 channels: own body, own head, other bodies, other
    // heads, food
    let channels = 5i64;
    let height = GRID_HEIGHT as i64;
    let width = GRID_WIDTH as i64;

    tracing::info!("Environment: Snake (Single Agent)");
    tracing::info!("  Grid size: {}x{}", GRID_WIDTH, GRID_HEIGHT);
    tracing::info!("  Observation shape: [{}, {}, {}]", channels, height, width);
    tracing::info!("  Num envs: {}", NUM_ENVS);
    tracing::info!("  Steps per rollout: {}", NUM_STEPS);
    tracing::info!("  Total timesteps: {}", TOTAL_TIMESTEPS);

    // Create environment pool
    let mut envs: Vec<SnakeEnv> =
        (0..NUM_ENVS).map(|_| SnakeEnv::new(GRID_WIDTH, GRID_HEIGHT)).collect();

    // Create CNN policy
    tracing::info!("Creating CNN policy...");
    let device = Device::cuda_if_available();
    tracing::info!("  Device: {:?}", device);

    let vs = nn::VarStore::new(device);
    let policy = SnakeCNN::new(&vs.root(), GRID_WIDTH as i64, channels);
    let mut optimizer = nn::Adam::default().build(&vs, LEARNING_RATE)?;

    // Training loop setup
    let num_updates = TOTAL_TIMESTEPS / (NUM_STEPS * NUM_ENVS);
    tracing::info!("Starting training loop ({} updates)...", num_updates);

    // Initialize observations - reset() modifies in place, then get grid
    // observations
    for env in &mut envs {
        env.reset();
    }
    // Get grid observations for single-agent Snake (agent_id=0)
    let mut observations: Vec<Vec<f32>> =
        envs.iter().map(|env| env.get_grid_observation(0)).collect();

    // Verify observation size
    if observations[0].len() != (channels as usize) * (height as usize) * (width as usize) {
        panic!(
            "Observation size mismatch! Expected {} but got {}",
            (channels as usize) * (height as usize) * (width as usize),
            observations[0].len()
        );
    }
    let mut buffer = RolloutBuffer::new();
    let mut total_episodes = 0;
    let mut total_steps = 0;

    for update in 0..num_updates {
        buffer.clear();

        // Collect rollout
        for _ in 0..NUM_STEPS {
            // Prepare observations tensor
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_tensor = Tensor::from_slice(&obs_flat)
                .reshape([NUM_ENVS as i64, channels, height, width])
                .to_device(device);

            // Get actions from policy
            let (actions, log_probs, values) = tch::no_grad(|| policy.sample_action(&obs_tensor));

            let actions_vec: Vec<i64> = Vec::try_from(actions.squeeze_dim(1))?;
            let log_probs_vec: Vec<f32> = Vec::try_from(log_probs.squeeze_dim(1))?;
            let values_vec: Vec<f32> = Vec::try_from(values.squeeze_dim(1))?;

            // Step environments
            for env_id in 0..NUM_ENVS {
                let step_result = envs[env_id].step(actions_vec[env_id]);

                buffer.add(
                    observations[env_id].clone(),
                    actions_vec[env_id],
                    log_probs_vec[env_id],
                    step_result.reward,
                    values_vec[env_id],
                    step_result.terminated || step_result.truncated,
                );

                // Reset if episode ended
                if step_result.terminated || step_result.truncated {
                    total_episodes += 1;
                    envs[env_id].reset();
                }

                // Always get grid observation (step() returns simple features, but we need
                // grid)
                observations[env_id] = envs[env_id].get_grid_observation(0);
            }

            total_steps += NUM_ENVS;
        }

        // Compute advantages
        let (advantages, returns) =
            compute_advantages(&buffer.rewards, &buffer.values, &buffer.dones, GAMMA, GAE_LAMBDA);

        // Normalize advantages
        let adv_mean = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let adv_std = (advantages.iter().map(|a| (a - adv_mean).powi(2)).sum::<f32>()
            / advantages.len() as f32)
            .sqrt();
        let advantages_normalized: Vec<f32> =
            advantages.iter().map(|a| (a - adv_mean) / (adv_std + 1e-8)).collect();

        // PPO update
        let batch_size = buffer.len();
        let num_minibatches = batch_size / MINIBATCH_SIZE;

        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_entropy = 0.0;

        for _ in 0..PPO_EPOCHS {
            // Shuffle indices for minibatch sampling
            let mut indices: Vec<usize> = (0..batch_size).collect();
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());

            for mb in 0..num_minibatches {
                let mb_indices = &indices[mb * MINIBATCH_SIZE..(mb + 1) * MINIBATCH_SIZE];

                // Prepare minibatch
                let mb_obs: Vec<f32> = mb_indices
                    .iter()
                    .flat_map(|&i| buffer.observations[i].iter())
                    .copied()
                    .collect();
                let mb_actions: Vec<i64> = mb_indices.iter().map(|&i| buffer.actions[i]).collect();
                let mb_old_log_probs: Vec<f32> =
                    mb_indices.iter().map(|&i| buffer.log_probs[i]).collect();
                let mb_advantages: Vec<f32> =
                    mb_indices.iter().map(|&i| advantages_normalized[i]).collect();
                let mb_returns: Vec<f32> = mb_indices.iter().map(|&i| returns[i]).collect();

                let mb_obs_tensor = Tensor::from_slice(&mb_obs)
                    .reshape([MINIBATCH_SIZE as i64, channels, height, width])
                    .to_device(device);
                let mb_actions_tensor = Tensor::from_slice(&mb_actions).to_device(device);
                let mb_old_log_probs_tensor =
                    Tensor::from_slice(&mb_old_log_probs).to_device(device);
                let mb_advantages_tensor = Tensor::from_slice(&mb_advantages).to_device(device);
                let mb_returns_tensor = Tensor::from_slice(&mb_returns).to_device(device);

                // Forward pass
                let (logits, values) = policy.forward(&mb_obs_tensor);
                let log_probs_all = logits.log_softmax(-1, tch::Kind::Float);
                let new_log_probs =
                    log_probs_all.gather(1, &mb_actions_tensor.unsqueeze(1), false).squeeze_dim(1);

                // Policy loss (PPO clipped objective)
                let ratio = (&new_log_probs - &mb_old_log_probs_tensor).exp();
                let surr1 = &ratio * &mb_advantages_tensor;
                let surr2 = ratio.clamp(1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * &mb_advantages_tensor;
                let policy_loss = -surr1.min_other(&surr2).mean(tch::Kind::Float);

                // Value loss
                let value_loss = (&values.squeeze_dim(1) - &mb_returns_tensor)
                    .pow_tensor_scalar(2)
                    .mean(tch::Kind::Float);

                // Entropy bonus
                let probs = logits.softmax(-1, tch::Kind::Float);
                let entropy = -(probs * log_probs_all)
                    .sum_dim_intlist(-1, false, tch::Kind::Float)
                    .mean(tch::Kind::Float);

                // Total loss
                let loss = &policy_loss + &value_loss * VALUE_COEF - &entropy * ENTROPY_COEF;

                // Backward pass
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                // Track losses
                total_policy_loss += f64::try_from(policy_loss)?;
                total_value_loss += f64::try_from(value_loss)?;
                total_entropy += f64::try_from(entropy)?;
            }
        }

        // Logging
        if (update + 1) % 10 == 0 {
            let num_opt_steps = (PPO_EPOCHS * num_minibatches) as f64;
            let mean_policy_loss = total_policy_loss / num_opt_steps;
            let mean_value_loss = total_value_loss / num_opt_steps;
            let mean_entropy = total_entropy / num_opt_steps;

            let mean_reward = if total_episodes > 0 {
                total_steps as f32 / total_episodes as f32
            } else {
                0.0
            };

            tracing::info!(
                "Update {}/{} | Steps: {} | Episodes: {} | Mean Steps/Episode: {:.1} | PL: {:.3} | VL: {:.3} | Ent: {:.3}",
                update + 1,
                num_updates,
                total_steps,
                total_episodes,
                mean_reward,
                mean_policy_loss,
                mean_value_loss,
                mean_entropy,
            );
        }

        // Periodic checkpoint
        if (update + 1) % 100 == 0 {
            let model_path = format!("models/snake_single_update{}.safetensors", update + 1);
            vs.save(&model_path)?;
            tracing::info!("  Saved checkpoint: {}", model_path);
        }
    }

    tracing::info!("Training complete!");
    tracing::info!("  Total steps: {}", total_steps);
    tracing::info!("  Total episodes: {}", total_episodes);

    // Save final model
    vs.save("models/snake_single_final.safetensors")?;
    tracing::info!("  Saved final model");

    Ok(())
}
