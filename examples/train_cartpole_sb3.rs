//! CartPole PPO Training - Stable-Baselines3 Configuration
//!
//! This example replicates Stable-Baselines3's hyperparameters that achieve
//! 500.0 score in just 100k timesteps.
//!
//! Key differences from our previous approach:
//! - Much shorter rollouts (32 vs 256 steps)
//! - Higher learning rate with linear decay (0.001 -> 0)
//! - More training epochs per update (20 vs 10)
//! - No entropy bonus (0.0 vs 0.01)
//! - Less discount (0.98 vs 0.99)
//!
//! Target: 450-500 average episode length
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_cartpole_sb3 --release
//! ```

use anyhow::Result;
use thrust_rl::{
    buffer::rollout::RolloutBuffer,
    env::{Environment, cartpole::CartPole, pool::EnvPool},
    policy::{
        inference::TrainingMetadata,
        mlp::{Activation, MlpConfig, MlpPolicy},
    },
    train::ppo::{PPOConfig, PPOTrainer},
};

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt().with_env_filter("info").init();

    tracing::info!("🚀 Starting CartPole PPO Training (SB3 Config)");

    // Start timing
    let training_start = std::time::Instant::now();

    // Hyperparameters - matching Stable-Baselines3 exactly
    const NUM_ENVS: usize = 8;
    const NUM_STEPS: usize = 32; // Much shorter rollouts!
    const TOTAL_TIMESTEPS: usize = 100_000; // Only 100k needed!
    const INITIAL_LEARNING_RATE: f64 = 0.001; // 4x higher than before
    const HIDDEN_DIM: i64 = 64; // Use default network size

    // Environment dimensions
    let env = CartPole::new();
    let obs_space = env.observation_space();
    let action_space = env.action_space();

    let obs_dim = obs_space.shape[0] as i64;
    let action_dim = match action_space.space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n as i64,
        _ => panic!("Expected discrete action space"),
    };

    tracing::info!("Environment: CartPole-v1 (SB3 Configuration)");
    tracing::info!("  Observation dim: {}", obs_dim);
    tracing::info!("  Action dim: {}", action_dim);
    tracing::info!("  Num envs: {}", NUM_ENVS);
    tracing::info!("  Steps per rollout: {}", NUM_STEPS);
    tracing::info!("  Total timesteps: {}", TOTAL_TIMESTEPS);

    // Create environment pool
    let mut env_pool = EnvPool::new(CartPole::new, NUM_ENVS);

    // Create policy with standard architecture
    tracing::info!("Creating MLP policy...");
    let config = MlpConfig {
        num_layers: 2,
        hidden_dim: HIDDEN_DIM,
        use_orthogonal_init: true,
        activation: Activation::Tanh, // SB3 uses Tanh
    };
    let mut policy = MlpPolicy::with_config(obs_dim, action_dim, config);
    let device = policy.device();
    tracing::info!("  Device: {:?}", device);
    tracing::info!("  Hidden dim: {}", HIDDEN_DIM);
    tracing::info!("  Activation: Tanh");
    tracing::info!("  Initialization: Orthogonal");

    // Create PPO trainer with SB3 hyperparameters
    let ppo_config = PPOConfig::new()
        .learning_rate(INITIAL_LEARNING_RATE)
        .n_epochs(20)  // More epochs per update
        .batch_size(256)  // Larger batch size
        .gamma(0.98)  // Less discount than before
        .gae_lambda(0.8)  // Less GAE smoothing
        .clip_range(0.2)
        .vf_coef(0.5)
        .ent_coef(0.0)  // NO entropy bonus!
        .max_grad_norm(0.5);

    let dummy_policy = MlpPolicy::with_config(obs_dim, action_dim, MlpConfig::default());
    let mut trainer = PPOTrainer::new(ppo_config, dummy_policy)?;

    // Create rollout buffer
    let mut buffer = RolloutBuffer::new(NUM_STEPS, NUM_ENVS, obs_dim as usize);

    // Training loop
    tracing::info!("Starting training...");
    tracing::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut observations = env_pool.reset();
    let num_updates = TOTAL_TIMESTEPS / (NUM_STEPS * NUM_ENVS);

    for update in 0..num_updates {
        // Compute current learning rate (linear decay)
        let progress = (update * NUM_STEPS * NUM_ENVS) as f64 / TOTAL_TIMESTEPS as f64;
        let current_lr = INITIAL_LEARNING_RATE * (1.0 - progress);

        // Create new optimizer with updated learning rate
        let optimizer = policy.optimizer(current_lr);
        trainer.set_optimizer(optimizer);

        // Collect rollout
        buffer.reset();

        for step in 0..NUM_STEPS {
            // Convert to tensor (no normalization)
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_tensor = tch::Tensor::from_slice(&obs_flat)
                .reshape([NUM_ENVS as i64, obs_dim])
                .to_device(device);

            // Get actions from policy
            let (actions, log_probs, values) = policy.get_action(&obs_tensor);

            let actions_vec: Vec<i64> = Vec::try_from(actions)?;
            let log_probs_vec: Vec<f32> = Vec::try_from(log_probs)?;
            let values_vec: Vec<f32> = Vec::try_from(values)?;

            // Step environments
            let results = env_pool.step(&actions_vec);

            // Store transitions
            for env_id in 0..NUM_ENVS {
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

                // Update observation
                observations[env_id] = results[env_id].observation.clone();

                // Handle episode end
                if results[env_id].terminated || results[env_id].truncated {
                    trainer.increment_episodes(1);
                    observations[env_id] = env_pool.reset_env(env_id)?;
                }
            }

            trainer.increment_steps(NUM_ENVS);
        }

        // Compute advantages
        let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let obs_tensor = tch::Tensor::from_slice(&obs_flat)
            .reshape([NUM_ENVS as i64, obs_dim])
            .to_device(device);

        let (_, _, last_values) = policy.get_action(&obs_tensor);
        let last_values_vec: Vec<f32> = Vec::try_from(last_values)?;

        buffer.compute_advantages(&last_values_vec, 0.98, 0.8);

        // Get training batch
        let batch = buffer.get_batch();
        let batch_size = batch.observations.len() / obs_dim as usize;

        let obs_tensor = tch::Tensor::from_slice(&batch.observations)
            .reshape([batch_size as i64, obs_dim])
            .to_device(device);
        let actions_tensor = tch::Tensor::from_slice(&batch.actions).to_device(device);
        let old_log_probs_tensor = tch::Tensor::from_slice(&batch.old_log_probs).to_device(device);
        let old_values_tensor = tch::Tensor::from_slice(&batch.old_values).to_device(device);
        let advantages_tensor = tch::Tensor::from_slice(&batch.advantages).to_device(device);
        let returns_tensor = tch::Tensor::from_slice(&batch.returns).to_device(device);

        // Train
        let stats = trainer.train_step_with_policy(
            &policy,
            &obs_tensor,
            &actions_tensor,
            &old_log_probs_tensor,
            &old_values_tensor,
            &advantages_tensor,
            &returns_tensor,
            |p, obs, acts| p.evaluate_actions(obs, acts),
        )?;

        // Log progress
        if update % 10 == 0 {
            let timesteps = trainer.total_steps();
            let episodes = trainer.total_episodes();
            let avg_steps_per_episode = if episodes > 0 {
                timesteps as f64 / episodes as f64
            } else {
                0.0
            };

            tracing::info!(
                "Update {:3}/{} | Steps: {:6} | Eps: {:4} | Avg Steps/Ep: {:5.1} | LR: {:.4} | Loss: {:.3} | Ent: {:.3}",
                update + 1,
                num_updates,
                timesteps,
                episodes,
                avg_steps_per_episode,
                current_lr,
                stats.total_loss,
                stats.entropy,
            );
        }
    }

    tracing::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    tracing::info!("✅ Training complete!");

    // Calculate training time
    let training_duration = training_start.elapsed();
    let training_secs = training_duration.as_secs_f64();

    let final_avg = if trainer.total_episodes() > 0 {
        trainer.total_steps() as f64 / trainer.total_episodes() as f64
    } else {
        0.0
    };

    tracing::info!("📊 Final Performance:");
    tracing::info!("  Total steps: {}", trainer.total_steps());
    tracing::info!("  Total episodes: {}", trainer.total_episodes());
    tracing::info!("  Avg steps per episode: {:.1}", final_avg);
    tracing::info!("  Training time: {:.1}s", training_secs);
    tracing::info!("  Steps per second: {:.0}", trainer.total_steps() as f64 / training_secs);
    tracing::info!("  Device: {:?}", device);

    // Save model
    let save_path = "cartpole_model_sb3.pt";
    policy.save(save_path)?;
    tracing::info!("💾 Model saved to {}", save_path);

    // Export to JSON for web with metadata
    tracing::info!("🔄 Exporting model to JSON with training metadata...");
    let mut exported_model = policy.export_for_inference();

    // Add training metadata
    let metadata = TrainingMetadata {
        total_steps: trainer.total_steps(),
        total_episodes: trainer.total_episodes(),
        final_performance: final_avg,
        training_time_secs: training_secs,
        device: format!("{:?}", device),
        environment: "CartPole-v1".to_string(),
        algorithm: "PPO (Stable-Baselines3 config)".to_string(),
        timestamp: Some(chrono::Utc::now().to_rfc3339()),
        notes: Some(format!(
            "SB3-matched training: n_steps=32, batch_size=256, n_epochs=20, lr=0.001->0 (linear decay), \
             gamma=0.98, gae_lambda=0.8, ent_coef=0.0, Tanh activation. \
             Achieved {:.1} steps/episode in {:.1}s ({:.0} steps/sec).",
            final_avg,
            training_secs,
            trainer.total_steps() as f64 / training_secs
        )),
    };
    exported_model.metadata = Some(metadata);

    let json_path = "cartpole_model_sb3.json";
    exported_model.save_json(json_path)?;
    let file_size = std::fs::metadata(json_path)?.len();
    tracing::info!("✅ Model exported to {} ({} KB)", json_path, file_size / 1024);

    if final_avg >= 450.0 {
        tracing::info!("🎉 SUCCESS! Achieved 450+ performance!");
    } else if final_avg >= 400.0 {
        tracing::info!("🎊 GREAT! Strong performance achieved!");
    } else if final_avg >= 300.0 {
        tracing::info!("👍 GOOD! Solid performance achieved!");
    } else {
        tracing::info!(
            "💡 Performance below target - may need more training or hyperparameter tuning"
        );
    }

    Ok(())
}
