//! Hyperparameter Optimization for Multi-Agent Snake
//!
//! This runs random search over the hyperparameter space to find
//! configurations that achieve high mean episode rewards.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example optimize_snake --release -- --trials 30
//! ```
//!
//! Results are saved to `snake_optimization_results.json` after each trial.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use tch::{nn, nn::OptimizerConfig, Device, Tensor};
use thrust_rl::{
    env::snake::SnakeEnv,
    optimize::{ParameterValue, SearchSpace, Trial, TrialResult},
    policy::snake_cnn::SnakeCNN,
};

#[derive(Debug, Serialize, Deserialize)]
struct OptimizationState {
    trials: Vec<Trial>,
    best_performance: f64,
    best_trial_id: Option<usize>,
}

impl OptimizationState {
    fn new() -> Self {
        Self {
            trials: Vec::new(),
            best_performance: f64::NEG_INFINITY,
            best_trial_id: None,
        }
    }

    fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    fn load(path: &str) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    fn add_trial(&mut self, trial: Trial) {
        if let Some(result) = &trial.result {
            if result.success {
                if let Some(perf) = result.get_metric("performance") {
                    if perf > self.best_performance {
                        self.best_performance = perf;
                        self.best_trial_id = Some(trial.id);
                    }
                }
            }
        }
        self.trials.push(trial);
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

fn run_trial(
    trial_id: usize,
    config: &HashMap<String, ParameterValue>,
    device: Device,
) -> Result<TrialResult> {
    let start_time = std::time::Instant::now();

    // Extract hyperparameters
    let num_envs = config.get("num_envs").and_then(|v| v.as_i64()).unwrap_or(16) as usize;
    let num_agents = 4; // Fixed for now
    let grid_size = config.get("grid_size").and_then(|v| v.as_i64()).unwrap_or(20) as i32;

    let learning_rate = config.get("learning_rate").map(|v| v.as_f64()).unwrap_or(3e-4);
    let steps_per_rollout = config.get("steps_per_rollout").and_then(|v| v.as_i64()).unwrap_or(512) as usize;
    let ppo_epochs = config.get("ppo_epochs").and_then(|v| v.as_i64()).unwrap_or(4) as usize;
    let minibatch_size = config.get("minibatch_size").and_then(|v| v.as_i64()).unwrap_or(64) as usize;

    let gamma = config.get("gamma").map(|v| v.as_f64()).unwrap_or(0.99);
    let gae_lambda = config.get("gae_lambda").map(|v| v.as_f64()).unwrap_or(0.95);
    let clip_param = config.get("clip_param").map(|v| v.as_f64()).unwrap_or(0.2);
    let value_coef = config.get("value_coef").map(|v| v.as_f64()).unwrap_or(0.5);
    let entropy_coef = config.get("entropy_coef").map(|v| v.as_f64()).unwrap_or(0.01);

    tracing::info!("Trial {}: envs={}, grid={}, lr={:.5}, steps={}, ppo_epochs={}, batch={}, gamma={:.3}, lambda={:.3}, clip={:.2}, vf={:.2}, ent={:.4}",
        trial_id, num_envs, grid_size, learning_rate, steps_per_rollout, ppo_epochs, minibatch_size,
        gamma, gae_lambda, clip_param, value_coef, entropy_coef);

    // Create variable store and policy
    let vs = nn::VarStore::new(device);
    let policy = SnakeCNN::new(&vs.root(), grid_size as i64, 5);
    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;

    // Create environments
    let mut envs: Vec<SnakeEnv> = (0..num_envs)
        .map(|_| SnakeEnv::new_multi(grid_size, grid_size, num_agents))
        .collect();

    let mut rollout_buffer = RolloutBuffer::new();
    let mut total_episodes = 0;
    let mut total_episode_rewards = Vec::new();
    let total_steps = 150_000; // 150k steps per trial
    let num_rollouts = total_steps / (steps_per_rollout * num_envs * num_agents);

    // Training loop
    for rollout_idx in 0..num_rollouts {
        rollout_buffer.clear();

        // Reset all environments
        for env in &mut envs {
            env.reset();
        }

        let mut episode_rewards = Vec::new();

        // Collect rollout
        for _step in 0..steps_per_rollout {
            // Collect observations from all environments and agents
            let mut all_obs = Vec::new();
            for env in &envs {
                for agent_id in 0..num_agents {
                    let obs = env.get_grid_observation(agent_id);
                    all_obs.push(obs);
                }
            }

            // Convert to tensor [batch, channels, height, width]
            let batch_size = all_obs.len();
            let obs_flat: Vec<f32> = all_obs.iter().flatten().copied().collect();
            let obs_tensor = Tensor::from_slice(&obs_flat)
                .reshape([batch_size as i64, 5, grid_size as i64, grid_size as i64])
                .to_device(device);

            // Get actions and values
            let (actions, log_probs, values) = tch::no_grad(|| policy.sample_action(&obs_tensor));

            let actions_vec: Vec<i64> = actions.squeeze_dim(1).try_into().unwrap();
            let log_probs_vec: Vec<f32> = log_probs.squeeze_dim(1).try_into().unwrap();
            let values_vec: Vec<f32> = values.squeeze_dim(1).try_into().unwrap();

            // Step environments
            let mut obs_idx = 0;
            for env_idx in 0..num_envs {
                // Collect actions for all agents in this environment
                let env_actions: Vec<i64> = (0..num_agents)
                    .map(|_| {
                        let a = actions_vec[obs_idx];
                        obs_idx += 1;
                        a
                    })
                    .collect();

                // Step environment
                let result = envs[env_idx].step_multi(&env_actions);

                // Store transitions for all agents
                for agent_id in 0..num_agents {
                    let obs_idx_curr = env_idx * num_agents + agent_id;
                    rollout_buffer.add(
                        all_obs[obs_idx_curr].clone(),
                        actions_vec[obs_idx_curr],
                        log_probs_vec[obs_idx_curr],
                        result.reward,
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
        }

        // Track episode rewards
        if !episode_rewards.is_empty() {
            total_episode_rewards.extend(episode_rewards);
        }

        // Compute advantages
        let (advantages, returns) = compute_advantages(
            &rollout_buffer.rewards,
            &rollout_buffer.values,
            &rollout_buffer.dones,
            gamma,
            gae_lambda,
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
        for _ in 0..ppo_epochs {
            // Shuffle indices
            let mut indices: Vec<usize> = (0..buffer_size).collect();
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());

            // Mini-batch updates
            for chunk in indices.chunks(minibatch_size) {
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
                    .reshape([chunk.len() as i64, 5, grid_size as i64, grid_size as i64])
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
                let surr2 = ratio.clamp(1.0 - clip_param, 1.0 + clip_param) * &advantages_tensor;
                let policy_loss = -surr1.min_other(&surr2).mean(tch::Kind::Float);

                // Value loss
                let value_loss = (&values.squeeze_dim(1) - &returns_tensor).pow_tensor_scalar(2).mean(tch::Kind::Float);

                // Entropy bonus
                let probs = logits.softmax(-1, tch::Kind::Float);
                let entropy = -(probs * log_probs_all).sum_dim_intlist(&[-1i64][..], false, tch::Kind::Float).mean(tch::Kind::Float);

                // Total loss
                let loss = policy_loss + value_coef * value_loss - entropy_coef * entropy;

                // Backward pass
                opt.zero_grad();
                loss.backward();
                opt.step();
            }
        }

        // Log progress every 10 rollouts
        if (rollout_idx + 1) % 10 == 0 {
            let recent_rewards: Vec<f32> = total_episode_rewards
                .iter()
                .rev()
                .take(20)
                .copied()
                .collect();
            if !recent_rewards.is_empty() {
                let mean_reward = recent_rewards.iter().sum::<f32>() / recent_rewards.len() as f32;
                tracing::info!("  Rollout {}/{} | Recent mean reward: {:.2}",
                    rollout_idx + 1, num_rollouts, mean_reward);
            }
        }
    }

    // Compute final performance (mean of all episode rewards)
    let final_performance = if !total_episode_rewards.is_empty() {
        total_episode_rewards.iter().sum::<f32>() as f64 / total_episode_rewards.len() as f64
    } else {
        0.0
    };

    let duration = start_time.elapsed().as_secs_f64();

    let mut metrics = HashMap::new();
    metrics.insert("performance".to_string(), final_performance);
    metrics.insert("training_time".to_string(), duration);
    metrics.insert("total_episodes".to_string(), total_episodes as f64);
    metrics.insert("num_rewards_collected".to_string(), total_episode_rewards.len() as f64);

    tracing::info!("Trial {} complete: performance={:.2}, episodes={}, time={:.1}s",
        trial_id, final_performance, total_episodes, duration);

    Ok(TrialResult::success(metrics, duration))
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    tracing::info!("🔍 Starting Snake Hyperparameter Optimization");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let num_trials = if args.len() > 2 && args[1] == "--trials" {
        args[2].parse().unwrap_or(30)
    } else {
        30
    };

    tracing::info!("Configuration:");
    tracing::info!("  Trials: {}", num_trials);
    tracing::info!("  Steps per trial: ~150,000");

    // Setup device
    let device = if tch::Cuda::is_available() {
        tracing::info!("  Device: CUDA");
        Device::Cuda(0)
    } else {
        tracing::info!("  Device: CPU");
        Device::Cpu
    };

    // Define search space
    let space = SearchSpace::new()
        .add_discrete("num_envs", vec![8, 16, 24])
        .add_discrete("grid_size", vec![15, 20, 25])
        .add_continuous("learning_rate", 1e-4, 1e-3, true) // log scale
        .add_discrete("steps_per_rollout", vec![256, 512, 1024])
        .add_discrete("ppo_epochs", vec![3, 4, 6])
        .add_discrete("minibatch_size", vec![32, 64, 128])
        .add_continuous("gamma", 0.97, 0.995, false)
        .add_continuous("gae_lambda", 0.90, 0.98, false)
        .add_continuous("clip_param", 0.1, 0.3, false)
        .add_continuous("value_coef", 0.25, 1.0, false)
        .add_continuous("entropy_coef", 0.001, 0.02, true); // log scale

    // Load or create optimization state
    let state_path = "snake_optimization_results.json";
    let mut state = OptimizationState::load(state_path).unwrap_or_else(|_| OptimizationState::new());

    let start_trial = state.trials.len();
    tracing::info!("Resuming from trial {}", start_trial);

    // Run trials
    for trial_id in start_trial..num_trials {
        tracing::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        tracing::info!("Starting Trial {}/{}", trial_id + 1, num_trials);

        // Sample configuration
        let config = space.sample();

        // Run trial
        let result = run_trial(trial_id, &config, device);

        // Create trial object
        let mut trial = Trial::new(trial_id, config);
        trial.result = Some(match result {
            Ok(r) => r,
            Err(e) => TrialResult::failure(e.to_string()),
        });

        // Log result
        if let Some(result) = &trial.result {
            if result.success {
                let perf = result.get_metric("performance").unwrap_or(0.0);
                let time = result.get_metric("training_time").unwrap_or(0.0);
                tracing::info!("✅ Trial {} complete: performance={:.2}, time={:.1}s", trial_id, perf, time);

                if perf >= 50.0 {
                    tracing::info!("🎉 HIGH PERFORMANCE! Found 50+ mean reward configuration!");
                }
            } else {
                tracing::warn!("❌ Trial {} failed: {:?}", trial_id, result.error);
            }
        }

        // Update state and save
        state.add_trial(trial);
        state.save(state_path)?;

        tracing::info!("Best so far: {:.2} (trial {})",
            state.best_performance,
            state.best_trial_id.unwrap_or(0));
    }

    tracing::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    tracing::info!("🏁 Optimization Complete!");
    tracing::info!("Best performance: {:.2} (trial {})",
        state.best_performance,
        state.best_trial_id.unwrap_or(0));

    if let Some(best_id) = state.best_trial_id {
        if let Some(best_trial) = state.trials.get(best_id) {
            tracing::info!("Best configuration:");
            for (key, value) in &best_trial.config {
                tracing::info!("  {}: {:?}", key, value);
            }
        }
    }

    Ok(())
}
