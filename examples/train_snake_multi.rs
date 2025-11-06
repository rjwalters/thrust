//! Train a multi-agent Snake policy using PPO with self-play
//!
//! This example demonstrates training 4 snakes to compete for food using
//! a CNN policy network and PPO reinforcement learning.

use anyhow::Result;
use std::path::PathBuf;
use tch::{nn, nn::OptimizerConfig, Device, Tensor};
use thrust_rl::{
    env::snake::SnakeEnv,
    policy::snake_cnn::SnakeCNN,
};

#[derive(Debug)]
struct Args {
    num_envs: usize,
    num_agents: usize,
    grid_width: i32,
    grid_height: i32,
    steps_per_rollout: usize,
    epochs: usize,
    learning_rate: f64,
    gae_lambda: f64,
    gamma: f64,
    clip_param: f64,
    value_coef: f64,
    entropy_coef: f64,
    ppo_epochs: usize,
    minibatch_size: usize,
    output: PathBuf,
    save_interval: usize,
    cuda: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            num_envs: 16,
            num_agents: 4,
            grid_width: 20,
            grid_height: 20,
            steps_per_rollout: 512,
            epochs: 1000,
            learning_rate: 3e-4,
            gae_lambda: 0.95,
            gamma: 0.99,
            clip_param: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            ppo_epochs: 4,
            minibatch_size: 64,
            output: PathBuf::from("models/snake_policy.pt"),
            save_interval: 100,
            cuda: true,
        }
    }
}

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

    fn add(&mut self, obs: Vec<f32>, action: i64, log_prob: f32, reward: f32, value: f32, done: bool) {
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
        let next_value = if t + 1 < values.len() {
            values[t + 1]
        } else {
            0.0
        };

        let delta = rewards[t] + gamma as f32 * next_value * (!dones[t] as i32 as f32) - values[t];
        gae = delta + gamma as f32 * lambda as f32 * (!dones[t] as i32 as f32) * gae;
        advantages[t] = gae;
        returns[t] = advantages[t] + values[t];
    }

    (advantages, returns)
}

fn main() -> Result<()> {
    let args = Args::default();

    // Setup device
    let device = if args.cuda && tch::Cuda::is_available() {
        println!("Using CUDA");
        Device::Cuda(0)
    } else {
        println!("Using CPU");
        Device::Cpu
    };

    // Create variable store and policy
    let mut vs = nn::VarStore::new(device);
    let policy = SnakeCNN::new(
        &vs.root(),
        args.grid_width as i64,
        5, // 5 input channels
    );

    // Create optimizer
    let mut opt = nn::Adam::default().build(&vs, args.learning_rate)?;

    // Create environments
    let mut envs: Vec<SnakeEnv> = (0..args.num_envs)
        .map(|_| SnakeEnv::new_multi(args.grid_width, args.grid_height, args.num_agents))
        .collect();

    println!("Training configuration:");
    println!("  Environments: {}", args.num_envs);
    println!("  Agents per env: {}", args.num_agents);
    println!("  Grid size: {}x{}", args.grid_width, args.grid_height);
    println!("  Steps per rollout: {}", args.steps_per_rollout);
    println!("  Learning rate: {}", args.learning_rate);
    println!();

    let mut rollout_buffer = RolloutBuffer::new();
    let mut total_episodes = 0;
    let mut total_steps = 0;

    // Training loop
    for epoch in 0..args.epochs {
        rollout_buffer.clear();

        // Reset all environments
        for env in &mut envs {
            env.reset();
        }

        let mut episode_rewards = Vec::new();

        // Collect rollout
        for step in 0..args.steps_per_rollout {
            // Collect observations from all environments and agents
            let mut all_obs = Vec::new();
            for env in &envs {
                for agent_id in 0..args.num_agents {
                    let obs = env.get_grid_observation(agent_id);
                    all_obs.push(obs);
                }
            }

            // Convert to tensor [batch, channels, height, width]
            let batch_size = all_obs.len();
            let obs_flat: Vec<f32> = all_obs.iter().flatten().copied().collect();
            let obs_tensor = Tensor::from_slice(&obs_flat)
                .reshape([batch_size as i64, 5, args.grid_height as i64, args.grid_width as i64])
                .to_device(device);

            // Get actions and values
            let (actions, log_probs, values) = tch::no_grad(|| policy.sample_action(&obs_tensor));

            let actions_vec: Vec<i64> = actions.squeeze_dim(1).try_into().unwrap();
            let log_probs_vec: Vec<f32> = log_probs.squeeze_dim(1).try_into().unwrap();
            let values_vec: Vec<f32> = values.squeeze_dim(1).try_into().unwrap();

            // Step environments
            let mut obs_idx = 0;
            for env_idx in 0..args.num_envs {
                // Collect actions for all agents in this environment
                let env_actions: Vec<i64> = (0..args.num_agents)
                    .map(|_| {
                        let a = actions_vec[obs_idx];
                        obs_idx += 1;
                        a
                    })
                    .collect();

                // Step environment
                let result = envs[env_idx].step_multi(&env_actions);

                // Store transitions for all agents
                let reward_per_agent = result.reward / args.num_agents as f32;
                for agent_id in 0..args.num_agents {
                    let obs_idx_curr = env_idx * args.num_agents + agent_id;
                    rollout_buffer.add(
                        all_obs[obs_idx_curr].clone(),
                        actions_vec[obs_idx_curr],
                        log_probs_vec[obs_idx_curr],
                        reward_per_agent,
                        values_vec[obs_idx_curr],
                        result.terminated || result.truncated,
                    );
                }

                // Track episode stats
                if result.terminated || result.truncated {
                    episode_rewards.push(result.reward);
                    total_episodes += 1;
                }
            }

            total_steps += args.num_envs * args.num_agents;
        }

        // Compute advantages
        let (advantages, returns) = compute_advantages(
            &rollout_buffer.rewards,
            &rollout_buffer.values,
            &rollout_buffer.dones,
            args.gamma,
            args.gae_lambda,
        );

        // Normalize advantages
        let mean_adv = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let std_adv = (advantages
            .iter()
            .map(|a| (a - mean_adv).powi(2))
            .sum::<f32>()
            / advantages.len() as f32)
            .sqrt();
        let norm_advantages: Vec<f32> = advantages
            .iter()
            .map(|a| (a - mean_adv) / (std_adv + 1e-8))
            .collect();

        // PPO update
        let buffer_size = rollout_buffer.len();
        for _ in 0..args.ppo_epochs {
            // Shuffle indices
            let mut indices: Vec<usize> = (0..buffer_size).collect();
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());

            // Mini-batch updates
            for chunk in indices.chunks(args.minibatch_size) {
                // Prepare batch
                let batch_obs: Vec<Vec<f32>> = chunk
                    .iter()
                    .map(|&i| rollout_buffer.observations[i].clone())
                    .collect();
                let batch_actions: Vec<i64> = chunk
                    .iter()
                    .map(|&i| rollout_buffer.actions[i])
                    .collect();
                let batch_old_log_probs: Vec<f32> = chunk
                    .iter()
                    .map(|&i| rollout_buffer.log_probs[i])
                    .collect();
                let batch_advantages: Vec<f32> = chunk
                    .iter()
                    .map(|&i| norm_advantages[i])
                    .collect();
                let batch_returns: Vec<f32> = chunk
                    .iter()
                    .map(|&i| returns[i])
                    .collect();

                // Convert to tensors
                let obs_flat: Vec<f32> = batch_obs.iter().flatten().copied().collect();
                let obs_tensor = Tensor::from_slice(&obs_flat)
                    .reshape([chunk.len() as i64, 5, args.grid_height as i64, args.grid_width as i64])
                    .to_device(device);

                let actions_tensor = Tensor::from_slice(&batch_actions).to_device(device);
                let old_log_probs_tensor = Tensor::from_slice(&batch_old_log_probs).to_device(device);
                let advantages_tensor = Tensor::from_slice(&batch_advantages).to_device(device);
                let returns_tensor = Tensor::from_slice(&batch_returns).to_device(device);

                // Forward pass
                let (logits, values) = policy.forward(&obs_tensor);
                let log_probs_all = logits.log_softmax(-1, tch::Kind::Float);
                let new_log_probs = log_probs_all.gather(1, &actions_tensor.unsqueeze(1), false).squeeze_dim(1);

                // PPO loss
                let ratio = (&new_log_probs - &old_log_probs_tensor).exp();
                let surr1 = &ratio * &advantages_tensor;
                let surr2 = ratio.clamp(1.0 - args.clip_param, 1.0 + args.clip_param) * &advantages_tensor;
                let policy_loss = -surr1.min_other(&surr2).mean(tch::Kind::Float);

                // Value loss
                let value_loss = (&values.squeeze_dim(1) - &returns_tensor).pow_tensor_scalar(2).mean(tch::Kind::Float);

                // Entropy bonus
                let probs = logits.softmax(-1, tch::Kind::Float);
                let entropy = -(probs * log_probs_all).sum_dim_intlist(&[-1i64][..], false, tch::Kind::Float).mean(tch::Kind::Float);

                // Total loss
                let loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy;

                // Backward pass
                opt.zero_grad();
                loss.backward();
                opt.step();
            }
        }

        // Logging
        if !episode_rewards.is_empty() {
            let mean_reward = episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
            println!(
                "Epoch {}/{} | Episodes: {} | Steps: {} | Mean Reward: {:.2}",
                epoch + 1,
                args.epochs,
                total_episodes,
                total_steps,
                mean_reward
            );
        }

        // Save checkpoint
        if (epoch + 1) % args.save_interval == 0 {
            let checkpoint_path = args.output.with_extension(format!("epoch{}.pt", epoch + 1));
            vs.save(&checkpoint_path)?;
            println!("Saved checkpoint to {:?}", checkpoint_path);
        }
    }

    // Save final model
    vs.save(&args.output)?;
    println!("Training complete! Model saved to {:?}", args.output);

    Ok(())
}
