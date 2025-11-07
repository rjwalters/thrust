//! Hyperparameter Optimization for CartPole
//!
//! This runs random search over the hyperparameter space to find
//! configurations that achieve 450+ steps/episode.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example optimize_cartpole --release -- --trials 30
//! ```
//!
//! Results are saved to `cartpole_optimization_results.json` after each trial.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use thrust_rl::{
    buffer::rollout::RolloutBuffer,
    env::{cartpole::CartPole, pool::EnvPool, Environment},
    optimize::{Objective, ParameterValue, SearchSpace, Trial, TrialResult},
    policy::mlp::{Activation, MlpConfig, MlpPolicy},
    train::ppo::{PPOConfig, PPOTrainer},
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
            best_performance: 0.0,
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

fn run_trial(
    trial_id: usize,
    config: &HashMap<String, ParameterValue>,
    training_steps: usize,
) -> Result<TrialResult> {
    let start_time = std::time::Instant::now();

    // Extract hyperparameters
    let n_steps = config.get("n_steps").and_then(|v| v.as_i64()).unwrap_or(256) as usize;
    let hidden_dim = config.get("hidden_dim").and_then(|v| v.as_i64()).unwrap_or(128);
    let learning_rate = config.get("learning_rate").map(|v| v.as_f64()).unwrap_or(0.0003);
    let n_epochs = config.get("n_epochs").and_then(|v| v.as_i64()).unwrap_or(10) as usize;
    let batch_size = config.get("batch_size").and_then(|v| v.as_i64()).unwrap_or(128) as usize;
    let ent_coef = config.get("ent_coef").map(|v| v.as_f64()).unwrap_or(0.01);
    let gamma = config.get("gamma").map(|v| v.as_f64()).unwrap_or(0.99);
    let num_envs = config.get("num_envs").and_then(|v| v.as_i64()).unwrap_or(16) as usize;

    tracing::info!("Trial {}: n_steps={}, hidden={}, lr={:.5}, epochs={}, batch={}, ent={:.3}, gamma={:.3}, envs={}",
        trial_id, n_steps, hidden_dim, learning_rate, n_epochs, batch_size, ent_coef, gamma, num_envs);

    // Environment setup
    let env = CartPole::new();
    let obs_space = env.observation_space();
    let action_space = env.action_space();
    let obs_dim = obs_space.shape[0] as i64;
    let action_dim = match action_space.space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n as i64,
        _ => return Err(anyhow::anyhow!("Expected discrete action space")),
    };

    // Create policy
    let config_mlp = MlpConfig {
        num_layers: 2,
        hidden_dim,
        use_orthogonal_init: true,
        activation: Activation::ReLU,
    };
    let mut policy = MlpPolicy::with_config(obs_dim, action_dim, config_mlp);
    let device = policy.device();

    // Create PPO trainer
    let ppo_config = PPOConfig::new()
        .learning_rate(learning_rate)
        .n_epochs(n_epochs)
        .batch_size(batch_size)
        .gamma(gamma)
        .gae_lambda(0.95)
        .clip_range(0.2)
        .vf_coef(0.5)
        .ent_coef(ent_coef)
        .max_grad_norm(0.5);

    let optimizer = policy.optimizer(learning_rate);
    let dummy_policy = MlpPolicy::with_config(obs_dim, action_dim, MlpConfig::default());
    let mut trainer = PPOTrainer::new(ppo_config, dummy_policy)?;
    trainer.set_optimizer(optimizer);

    // Create environment pool and buffer
    let mut env_pool = EnvPool::new(CartPole::new, num_envs);
    let mut buffer = RolloutBuffer::new(n_steps, num_envs, obs_dim as usize);

    // Training loop
    let mut observations = env_pool.reset();
    let num_updates = training_steps / (n_steps * num_envs);

    for _update in 0..num_updates {
        buffer.reset();

        // Collect rollout
        for step in 0..n_steps {
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_tensor = tch::Tensor::from_slice(&obs_flat)
                .reshape([num_envs as i64, obs_dim])
                .to_device(device);

            let (actions, log_probs, values) = policy.get_action(&obs_tensor);
            let actions_vec: Vec<i64> = Vec::try_from(actions)?;
            let log_probs_vec: Vec<f32> = Vec::try_from(log_probs)?;
            let values_vec: Vec<f32> = Vec::try_from(values)?;

            let results = env_pool.step(&actions_vec);

            for env_id in 0..num_envs {
                buffer.add(
                    step,
                    env_id,
                    &observations[env_id],
                    actions_vec[env_id],
                    results[env_id].reward,
                    values_vec[env_id],
                    log_probs_vec[env_id],
                    results[env_id].terminated,
                    results[env_id].truncated,
                );

                observations[env_id] = results[env_id].observation.clone();

                if results[env_id].terminated || results[env_id].truncated {
                    trainer.increment_episodes(1);
                    observations[env_id] = env_pool.reset_env(env_id)?;
                }
            }

            trainer.increment_steps(num_envs);
        }

        // Compute advantages
        let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let obs_tensor = tch::Tensor::from_slice(&obs_flat)
            .reshape([num_envs as i64, obs_dim])
            .to_device(device);

        let (_, _, last_values) = policy.get_action(&obs_tensor);
        let last_values_vec: Vec<f32> = Vec::try_from(last_values)?;
        buffer.compute_advantages(&last_values_vec, gamma as f32, 0.95);

        // Get batch and train
        let batch = buffer.get_batch();
        let batch_size_actual = batch.observations.len() / obs_dim as usize;

        let obs_tensor = tch::Tensor::from_slice(&batch.observations)
            .reshape([batch_size_actual as i64, obs_dim])
            .to_device(device);
        let actions_tensor = tch::Tensor::from_slice(&batch.actions).to_device(device);
        let old_log_probs_tensor = tch::Tensor::from_slice(&batch.old_log_probs).to_device(device);
        let old_values_tensor = tch::Tensor::from_slice(&batch.old_values).to_device(device);
        let advantages_tensor = tch::Tensor::from_slice(&batch.advantages).to_device(device);
        let returns_tensor = tch::Tensor::from_slice(&batch.returns).to_device(device);

        trainer.train_step_with_policy(
            &policy,
            &obs_tensor,
            &actions_tensor,
            &old_log_probs_tensor,
            &old_values_tensor,
            &advantages_tensor,
            &returns_tensor,
            |p, obs, acts| p.evaluate_actions(obs, acts),
        )?;
    }

    // Compute final performance
    let final_avg = if trainer.total_episodes() > 0 {
        trainer.total_steps() as f64 / trainer.total_episodes() as f64
    } else {
        0.0
    };

    let duration = start_time.elapsed().as_secs_f64();

    let mut metrics = HashMap::new();
    metrics.insert("performance".to_string(), final_avg);
    metrics.insert("training_time".to_string(), duration);
    metrics.insert("total_steps".to_string(), trainer.total_steps() as f64);
    metrics.insert("total_episodes".to_string(), trainer.total_episodes() as f64);

    Ok(TrialResult::success(metrics, duration))
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    tracing::info!("🔍 Starting CartPole Hyperparameter Optimization");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let num_trials = if args.len() > 2 && args[1] == "--trials" {
        args[2].parse().unwrap_or(30)
    } else {
        30
    };
    let training_steps_per_trial = 200_000; // 200k steps per trial for speed

    tracing::info!("Configuration:");
    tracing::info!("  Trials: {}", num_trials);
    tracing::info!("  Steps per trial: {}", training_steps_per_trial);

    // Define search space
    let space = SearchSpace::new()
        .add_discrete("n_steps", vec![128, 256, 512])
        .add_discrete("hidden_dim", vec![64, 128, 256])
        .add_continuous("learning_rate", 1e-4, 1e-3, true) // log scale
        .add_discrete("n_epochs", vec![5, 10, 15, 20])
        .add_discrete("batch_size", vec![64, 128, 256])
        .add_continuous("ent_coef", 0.0, 0.02, false)
        .add_continuous("gamma", 0.97, 0.995, false)
        .add_discrete("num_envs", vec![8, 16, 32]);

    // Load or create optimization state
    let state_path = "cartpole_optimization_results.json";
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
        let result = run_trial(trial_id, &config, training_steps_per_trial);

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
                tracing::info!("✅ Trial {} complete: performance={:.1}, time={:.1}s", trial_id, perf, time);

                if perf >= 450.0 {
                    tracing::info!("🎉 BREAKTHROUGH! Found 450+ configuration!");
                }
            } else {
                tracing::warn!("❌ Trial {} failed: {:?}", trial_id, result.error);
            }
        }

        // Update state and save
        state.add_trial(trial);
        state.save(state_path)?;

        tracing::info!("Best so far: {:.1} (trial {})",
            state.best_performance,
            state.best_trial_id.unwrap_or(0));
    }

    tracing::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    tracing::info!("🏁 Optimization Complete!");
    tracing::info!("Best performance: {:.1} (trial {})",
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
