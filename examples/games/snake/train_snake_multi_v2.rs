//! Train a multi-agent Snake policy using PPO with self-play
//!
//! This example demonstrates training 4 snakes to compete for food using
//! a CNN policy network and PPO reinforcement learning.
//!
//! Supports TWO training modes:
//! - **Shared policy mode** (--mode shared): Single policy controls all agents,
//!   with individualized rewards for better credit assignment
//! - **Independent policy mode** (--mode independent): Each agent has its own
//!   policy network, enabling true multi-agent learning
//!
//! # Usage
//!
//! ```bash
//! # Train with shared policy on CPU
//! cargo run --example train_snake_multi_v2 --release -- --mode shared
//!
//! # Train with independent policies on GPU
//! cargo run --example train_snake_multi_v2 --release -- --mode independent --cuda
//! ```

use std::path::PathBuf;

use anyhow::Result;
use tch::{Device, Tensor, nn, nn::OptimizerConfig};
use thrust_rl::{
    buffer::rollout::compute_advantages_multi_agent, env::snake::SnakeEnv,
    policy::snake_cnn::SnakeCNN,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainingMode {
    Shared,      // Single policy, individualized rewards
    Independent, // Multiple policies, one per agent
}

#[derive(Debug)]
struct Args {
    mode: TrainingMode,
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
            mode: TrainingMode::Shared,
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
            entropy_coef: 0.03, // Increased from 0.01 for better exploration (sparse rewards)
            ppo_epochs: 4,
            minibatch_size: 64,
            output: PathBuf::from("models/snake_policy.safetensors"),
            save_interval: 10,
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
    agent_ids: Vec<usize>, // Track which agent each experience belongs to
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
            agent_ids: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.observations.clear();
        self.actions.clear();
        self.log_probs.clear();
        self.rewards.clear();
        self.values.clear();
        self.dones.clear();
        self.agent_ids.clear();
    }

    fn add(
        &mut self,
        obs: Vec<f32>,
        action: i64,
        log_prob: f32,
        reward: f32,
        value: f32,
        done: bool,
        agent_id: usize,
    ) {
        self.observations.push(obs);
        self.actions.push(action);
        self.log_probs.push(log_prob);
        self.rewards.push(reward);
        self.values.push(value);
        self.dones.push(done);
        self.agent_ids.push(agent_id);
    }

    fn len(&self) -> usize {
        self.observations.len()
    }

    /// Get experiences for a specific agent (used in independent mode)
    fn get_agent_indices(&self, agent_id: usize) -> Vec<usize> {
        self.agent_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &id)| if id == agent_id { Some(idx) } else { None })
            .collect()
    }
}

fn main() -> Result<()> {
    let mut args = Args::default();

    // Parse command-line arguments
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--mode" => {
                if let Some(mode_str) = iter.next() {
                    args.mode = match mode_str.as_str() {
                        "shared" => TrainingMode::Shared,
                        "independent" => TrainingMode::Independent,
                        _ => {
                            eprintln!("Invalid mode: {}. Use 'shared' or 'independent'", mode_str);
                            std::process::exit(1);
                        }
                    };
                }
            }
            "--cuda" => args.cuda = true,
            "--cpu" => args.cuda = false,
            "--epochs" => {
                if let Some(epochs_str) = iter.next() {
                    args.epochs = epochs_str.parse().unwrap_or(args.epochs);
                }
            }
            _ => eprintln!("Unknown argument: {}", arg),
        }
    }

    // Setup device
    let device = if args.cuda && tch::Cuda::is_available() {
        println!("Using CUDA");
        Device::Cuda(0)
    } else {
        println!("Using CPU");
        Device::Cpu
    };

    println!("Training configuration:");
    println!("  Mode: {:?}", args.mode);
    println!("  Environments: {}", args.num_envs);
    println!("  Agents per env: {}", args.num_agents);
    println!("  Grid size: {}x{}", args.grid_width, args.grid_height);
    println!("  Steps per rollout: {}", args.steps_per_rollout);
    println!("  Learning rate: {}", args.learning_rate);
    println!();

    match args.mode {
        TrainingMode::Shared => train_shared_policy(args, device),
        TrainingMode::Independent => train_independent_policies(args, device),
    }
}

/// Train with a single shared policy (parameter sharing)
fn train_shared_policy(args: Args, device: Device) -> Result<()> {
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
        for _step in 0..args.steps_per_rollout {
            // Collect observations from all environments and agents
            let mut all_obs = Vec::new();
            let mut agent_ids = Vec::new();
            for env in &envs {
                for agent_id in 0..args.num_agents {
                    let obs = env.get_grid_observation(agent_id);
                    all_obs.push(obs);
                    agent_ids.push(agent_id);
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

            // Step environments with per-agent rewards
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

                // Step environment with per-agent rewards
                let (agent_rewards, terminated, truncated) =
                    envs[env_idx].step_multi_agents(&env_actions);

                // Store transitions for each agent with individualized rewards
                for agent_id in 0..args.num_agents {
                    let obs_idx_curr = env_idx * args.num_agents + agent_id;
                    rollout_buffer.add(
                        all_obs[obs_idx_curr].clone(),
                        actions_vec[obs_idx_curr],
                        log_probs_vec[obs_idx_curr],
                        agent_rewards[agent_id], // Individual reward!
                        values_vec[obs_idx_curr],
                        terminated || truncated,
                        agent_id,
                    );
                }

                // Track episode stats
                if terminated || truncated {
                    let total_reward: f32 = agent_rewards.iter().sum();
                    episode_rewards.push(total_reward);
                    total_episodes += 1;
                    envs[env_idx].reset();
                }
            }

            total_steps += args.num_envs * args.num_agents;
        }

        // Compute advantages.
        //
        // Shared-mode rollout buffer layout: each step pushes
        // `num_envs * num_agents` transitions in (env, agent)-major
        // order, so the flat buffer matches the layout expected by
        // `compute_advantages_multi_agent`. We pass a zero bootstrap
        // for `V(s_{T+1})` to preserve the prior behavior (the local
        // copy this replaced did the same implicitly).
        //
        // TODO(loom): future issue could lift the env-stepping rayon
        // loop into a `MultiAgentEnvPool` (see issue #46 curator notes).
        let stride = args.num_envs * args.num_agents;
        let last_values = vec![0.0_f32; stride];
        let (advantages, returns) = compute_advantages_multi_agent(
            &rollout_buffer.rewards,
            &rollout_buffer.values,
            &rollout_buffer.dones,
            &last_values,
            args.num_envs,
            args.num_agents,
            args.gamma as f32,
            args.gae_lambda as f32,
        );

        // Normalize advantages
        let mean_adv = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let std_adv = (advantages.iter().map(|a| (a - mean_adv).powi(2)).sum::<f32>()
            / advantages.len() as f32)
            .sqrt();
        let norm_advantages: Vec<f32> =
            advantages.iter().map(|a| (a - mean_adv) / (std_adv + 1e-8)).collect();

        // Track training metrics across all PPO epochs
        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_entropy = 0.0;
        let mut num_updates = 0;

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
                let batch_obs: Vec<Vec<f32>> =
                    chunk.iter().map(|&i| rollout_buffer.observations[i].clone()).collect();
                let batch_actions: Vec<i64> =
                    chunk.iter().map(|&i| rollout_buffer.actions[i]).collect();
                let batch_old_log_probs: Vec<f32> =
                    chunk.iter().map(|&i| rollout_buffer.log_probs[i]).collect();
                let batch_advantages: Vec<f32> =
                    chunk.iter().map(|&i| norm_advantages[i]).collect();
                let batch_returns: Vec<f32> = chunk.iter().map(|&i| returns[i]).collect();

                // Convert to tensors
                let obs_flat: Vec<f32> = batch_obs.iter().flatten().copied().collect();
                let obs_tensor = Tensor::from_slice(&obs_flat)
                    .reshape([
                        chunk.len() as i64,
                        5,
                        args.grid_height as i64,
                        args.grid_width as i64,
                    ])
                    .to_device(device);

                let actions_tensor = Tensor::from_slice(&batch_actions).to_device(device);
                let old_log_probs_tensor =
                    Tensor::from_slice(&batch_old_log_probs).to_device(device);
                let advantages_tensor = Tensor::from_slice(&batch_advantages).to_device(device);
                let returns_tensor = Tensor::from_slice(&batch_returns).to_device(device);

                // Forward pass
                let (logits, values) = policy.forward(&obs_tensor);
                let log_probs_all = logits.log_softmax(-1, tch::Kind::Float);
                let new_log_probs =
                    log_probs_all.gather(1, &actions_tensor.unsqueeze(1), false).squeeze_dim(1);

                // PPO loss
                let ratio = (&new_log_probs - &old_log_probs_tensor).exp();
                let surr1 = &ratio * &advantages_tensor;
                let surr2 =
                    ratio.clamp(1.0 - args.clip_param, 1.0 + args.clip_param) * &advantages_tensor;
                let policy_loss = -surr1.min_other(&surr2).mean(tch::Kind::Float);

                // Value loss
                let value_loss = (&values.squeeze_dim(1) - &returns_tensor)
                    .pow_tensor_scalar(2)
                    .mean(tch::Kind::Float);

                // Entropy bonus
                let probs = logits.softmax(-1, tch::Kind::Float);
                let entropy = -(probs * log_probs_all)
                    .sum_dim_intlist(&[-1i64][..], false, tch::Kind::Float)
                    .mean(tch::Kind::Float);

                // Track metrics (extract values before computing loss)
                total_policy_loss += f64::try_from(&policy_loss).unwrap_or(0.0);
                total_value_loss += f64::try_from(&value_loss).unwrap_or(0.0);
                total_entropy += f64::try_from(&entropy).unwrap_or(0.0);
                num_updates += 1;

                // Total loss
                let loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy;

                // Backward pass
                opt.zero_grad();
                loss.backward();
                opt.step();
            }
        }

        // Compute average metrics
        let avg_policy_loss = total_policy_loss / num_updates as f64;
        let avg_value_loss = total_value_loss / num_updates as f64;
        let avg_entropy = total_entropy / num_updates as f64;

        // Compute explained variance: 1 - Var(returns - values) / Var(returns)
        let mean_returns = returns.iter().sum::<f32>() / returns.len() as f32;
        let var_returns =
            returns.iter().map(|r| (r - mean_returns).powi(2)).sum::<f32>() / returns.len() as f32;
        let residuals: Vec<f32> =
            returns.iter().zip(&rollout_buffer.values).map(|(r, v)| r - v).collect();
        let mean_residuals = residuals.iter().sum::<f32>() / residuals.len() as f32;
        let var_residuals = residuals.iter().map(|r| (r - mean_residuals).powi(2)).sum::<f32>()
            / residuals.len() as f32;
        let explained_var = if var_returns > 1e-8 {
            1.0 - (var_residuals / var_returns)
        } else {
            0.0
        };

        // Logging
        if !episode_rewards.is_empty() {
            let mean_reward = episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
            println!(
                "[SHARED] Epoch {}/{} | Episodes: {} | Steps: {} | Reward: {:.2} | Policy Loss: {:.4} | Value Loss: {:.4} | Entropy: {:.3} | ExpVar: {:.3}",
                epoch + 1,
                args.epochs,
                total_episodes,
                total_steps,
                mean_reward,
                avg_policy_loss,
                avg_value_loss,
                avg_entropy,
                explained_var
            );
        } else {
            // Still log metrics even if no episodes completed
            println!(
                "[SHARED] Epoch {}/{} | Steps: {} | Policy Loss: {:.4} | Value Loss: {:.4} | Entropy: {:.3} | ExpVar: {:.3}",
                epoch + 1,
                args.epochs,
                total_steps,
                avg_policy_loss,
                avg_value_loss,
                avg_entropy,
                explained_var
            );
        }

        // Save checkpoint
        if (epoch + 1) % args.save_interval == 0 {
            let checkpoint_path =
                args.output.with_extension(format!("shared_epoch{}.safetensors", epoch + 1));
            vs.save(&checkpoint_path)?;
            println!("Saved checkpoint to {:?}", checkpoint_path);
        }
    }

    // Save final model
    let final_path = args.output.with_file_name("snake_policy_shared.safetensors");
    vs.save(&final_path)?;
    println!("Training complete! Model saved to {:?}", final_path);

    Ok(())
}

/// Train with independent policies (one per agent)
fn train_independent_policies(args: Args, device: Device) -> Result<()> {
    // Create variable stores and policies for each agent
    let mut var_stores: Vec<nn::VarStore> = Vec::new();
    let mut policies: Vec<SnakeCNN> = Vec::new();
    let mut optimizers: Vec<nn::Optimizer> = Vec::new();

    for agent_id in 0..args.num_agents {
        let mut vs = nn::VarStore::new(device);
        let policy = SnakeCNN::new(
            &vs.root(),
            args.grid_width as i64,
            5, // 5 input channels
        );
        let opt = nn::Adam::default().build(&vs, args.learning_rate)?;

        var_stores.push(vs);
        policies.push(policy);
        optimizers.push(opt);
    }

    // Create environments
    let mut envs: Vec<SnakeEnv> = (0..args.num_envs)
        .map(|_| SnakeEnv::new_multi(args.grid_width, args.grid_height, args.num_agents))
        .collect();

    // One buffer per agent
    let mut rollout_buffers: Vec<RolloutBuffer> =
        (0..args.num_agents).map(|_| RolloutBuffer::new()).collect();

    let mut total_episodes = 0;
    let mut total_steps = 0;

    // Training loop
    for epoch in 0..args.epochs {
        for buffer in &mut rollout_buffers {
            buffer.clear();
        }

        // Reset all environments
        for env in &mut envs {
            env.reset();
        }

        let mut episode_rewards = Vec::new();

        // Collect rollout
        for _step in 0..args.steps_per_rollout {
            // For each environment, get actions from each agent's policy
            for env_idx in 0..args.num_envs {
                let mut env_actions = Vec::new();
                let mut env_log_probs = Vec::new();
                let mut env_values = Vec::new();
                let mut env_obs = Vec::new();

                // Each agent selects action using its own policy
                for agent_id in 0..args.num_agents {
                    let obs = envs[env_idx].get_grid_observation(agent_id);
                    let obs_tensor = Tensor::from_slice(&obs)
                        .reshape([1, 5, args.grid_height as i64, args.grid_width as i64])
                        .to_device(device);

                    let (action, log_prob, value) =
                        tch::no_grad(|| policies[agent_id].sample_action(&obs_tensor));

                    let action_val: i64 = action.int64_value(&[0, 0]);
                    let log_prob_val: f32 = log_prob.double_value(&[0, 0]) as f32;
                    let value_val: f32 = value.double_value(&[0, 0]) as f32;

                    env_actions.push(action_val);
                    env_log_probs.push(log_prob_val);
                    env_values.push(value_val);
                    env_obs.push(obs);
                }

                // Step environment with per-agent rewards
                let (agent_rewards, terminated, truncated) =
                    envs[env_idx].step_multi_agents(&env_actions);

                // Store transitions in each agent's buffer
                for agent_id in 0..args.num_agents {
                    rollout_buffers[agent_id].add(
                        env_obs[agent_id].clone(),
                        env_actions[agent_id],
                        env_log_probs[agent_id],
                        agent_rewards[agent_id], // Individual reward!
                        env_values[agent_id],
                        terminated || truncated,
                        agent_id,
                    );
                }

                // Track episode stats
                if terminated || truncated {
                    let total_reward: f32 = agent_rewards.iter().sum();
                    episode_rewards.push(total_reward);
                    total_episodes += 1;
                    envs[env_idx].reset();
                }
            }

            total_steps += args.num_envs * args.num_agents;
        }

        // Train each agent's policy independently
        for agent_id in 0..args.num_agents {
            let buffer = &rollout_buffers[agent_id];

            // Compute advantages for this agent.
            //
            // Independent-mode buffers hold one agent's transitions
            // across all envs; layout is `(step, env)`-major with
            // `num_envs` slots per step. We treat that as a
            // `num_envs` x `num_agents = 1` flat buffer to share the
            // multi-agent GAE helper. Zero bootstrap preserves the
            // prior behavior of the deleted local copy.
            let stride = args.num_envs;
            let last_values = vec![0.0_f32; stride];
            let (advantages, returns) = compute_advantages_multi_agent(
                &buffer.rewards,
                &buffer.values,
                &buffer.dones,
                &last_values,
                args.num_envs,
                // num_agents =
                1,
                args.gamma as f32,
                args.gae_lambda as f32,
            );

            // Normalize advantages
            let mean_adv = advantages.iter().sum::<f32>() / advantages.len().max(1) as f32;
            let std_adv = (advantages.iter().map(|a| (a - mean_adv).powi(2)).sum::<f32>()
                / advantages.len().max(1) as f32)
                .sqrt();
            let norm_advantages: Vec<f32> =
                advantages.iter().map(|a| (a - mean_adv) / (std_adv + 1e-8)).collect();

            // Track training metrics for this agent
            let mut total_policy_loss = 0.0;
            let mut total_value_loss = 0.0;
            let mut total_entropy = 0.0;
            let mut num_updates = 0;

            // PPO update for this agent
            let buffer_size = buffer.len();
            for _ in 0..args.ppo_epochs {
                let mut indices: Vec<usize> = (0..buffer_size).collect();
                use rand::seq::SliceRandom;
                indices.shuffle(&mut rand::thread_rng());

                for chunk in indices.chunks(args.minibatch_size.min(buffer_size)) {
                    // Prepare batch
                    let batch_obs: Vec<Vec<f32>> =
                        chunk.iter().map(|&i| buffer.observations[i].clone()).collect();
                    let batch_actions: Vec<i64> =
                        chunk.iter().map(|&i| buffer.actions[i]).collect();
                    let batch_old_log_probs: Vec<f32> =
                        chunk.iter().map(|&i| buffer.log_probs[i]).collect();
                    let batch_advantages: Vec<f32> =
                        chunk.iter().map(|&i| norm_advantages[i]).collect();
                    let batch_returns: Vec<f32> = chunk.iter().map(|&i| returns[i]).collect();

                    // Convert to tensors
                    let obs_flat: Vec<f32> = batch_obs.iter().flatten().copied().collect();
                    let obs_tensor = Tensor::from_slice(&obs_flat)
                        .reshape([
                            chunk.len() as i64,
                            5,
                            args.grid_height as i64,
                            args.grid_width as i64,
                        ])
                        .to_device(device);

                    let actions_tensor = Tensor::from_slice(&batch_actions).to_device(device);
                    let old_log_probs_tensor =
                        Tensor::from_slice(&batch_old_log_probs).to_device(device);
                    let advantages_tensor = Tensor::from_slice(&batch_advantages).to_device(device);
                    let returns_tensor = Tensor::from_slice(&batch_returns).to_device(device);

                    // Forward pass
                    let (logits, values) = policies[agent_id].forward(&obs_tensor);
                    let log_probs_all = logits.log_softmax(-1, tch::Kind::Float);
                    let new_log_probs =
                        log_probs_all.gather(1, &actions_tensor.unsqueeze(1), false).squeeze_dim(1);

                    // PPO loss
                    let ratio = (&new_log_probs - &old_log_probs_tensor).exp();
                    let surr1 = &ratio * &advantages_tensor;
                    let surr2 = ratio.clamp(1.0 - args.clip_param, 1.0 + args.clip_param)
                        * &advantages_tensor;
                    let policy_loss = -surr1.min_other(&surr2).mean(tch::Kind::Float);

                    // Value loss
                    let value_loss = (&values.squeeze_dim(1) - &returns_tensor)
                        .pow_tensor_scalar(2)
                        .mean(tch::Kind::Float);

                    // Entropy bonus
                    let probs = logits.softmax(-1, tch::Kind::Float);
                    let entropy = -(probs * log_probs_all)
                        .sum_dim_intlist(&[-1i64][..], false, tch::Kind::Float)
                        .mean(tch::Kind::Float);

                    // Track metrics (extract values before computing loss)
                    total_policy_loss += f64::try_from(&policy_loss).unwrap_or(0.0);
                    total_value_loss += f64::try_from(&value_loss).unwrap_or(0.0);
                    total_entropy += f64::try_from(&entropy).unwrap_or(0.0);
                    num_updates += 1;

                    // Total loss
                    let loss =
                        policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy;

                    // Backward pass
                    optimizers[agent_id].zero_grad();
                    loss.backward();
                    optimizers[agent_id].step();
                }
            }

            // Log metrics for this agent (could aggregate across agents if desired)
            if agent_id == 0 && epoch % 10 == 0 {
                let avg_policy_loss = total_policy_loss / num_updates.max(1) as f64;
                let avg_value_loss = total_value_loss / num_updates.max(1) as f64;
                let avg_entropy = total_entropy / num_updates.max(1) as f64;

                // Compute explained variance for agent 0
                let mean_returns = returns.iter().sum::<f32>() / returns.len().max(1) as f32;
                let var_returns = returns.iter().map(|r| (r - mean_returns).powi(2)).sum::<f32>()
                    / returns.len().max(1) as f32;
                let residuals: Vec<f32> =
                    returns.iter().zip(&buffer.values).map(|(r, v)| r - v).collect();
                let mean_residuals = residuals.iter().sum::<f32>() / residuals.len().max(1) as f32;
                let var_residuals =
                    residuals.iter().map(|r| (r - mean_residuals).powi(2)).sum::<f32>()
                        / residuals.len().max(1) as f32;
                let explained_var = if var_returns > 1e-8 {
                    1.0 - (var_residuals / var_returns)
                } else {
                    0.0
                };

                println!(
                    "[INDEPENDENT] Agent {} | Policy Loss: {:.4} | Value Loss: {:.4} | Entropy: {:.3} | ExpVar: {:.3}",
                    agent_id, avg_policy_loss, avg_value_loss, avg_entropy, explained_var
                );
            }
        }

        // Logging
        if !episode_rewards.is_empty() {
            let mean_reward = episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
            println!(
                "[INDEPENDENT] Epoch {}/{} | Episodes: {} | Steps: {} | Mean Reward: {:.2}",
                epoch + 1,
                args.epochs,
                total_episodes,
                total_steps,
                mean_reward
            );
        }

        // Save checkpoints for all agents
        if (epoch + 1) % args.save_interval == 0 {
            for agent_id in 0..args.num_agents {
                let checkpoint_path = args.output.with_file_name(format!(
                    "snake_policy_agent{}_epoch{}.safetensors",
                    agent_id,
                    epoch + 1
                ));
                var_stores[agent_id].save(&checkpoint_path)?;
            }
            println!("Saved checkpoints for epoch {}", epoch + 1);
        }
    }

    // Save final models
    for agent_id in 0..args.num_agents {
        let final_path = args
            .output
            .with_file_name(format!("snake_policy_independent_agent{}.safetensors", agent_id));
        var_stores[agent_id].save(&final_path)?;
    }
    println!("Training complete! Models saved");

    Ok(())
}
