//! Train PPO on CartPole-v1 (Best Quality Training)
//!
//! This example trains a high-quality agent with extended training time
//! and optimized hyperparameters for the best possible performance.
//!
//! # Usage
//!
//! ```bash
//! # On GPU machine
//! cargo run --example train_cartpole_best --release
//! ```

use anyhow::Result;
use thrust_rl::{
    buffer::rollout::RolloutBuffer,
    env::{cartpole::CartPole, pool::EnvPool, Environment},
    policy::mlp::MlpPolicy,
    train::ppo::{PPOConfig, PPOTrainer},
};

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    tracing::info!("🚀 Starting CartPole PPO Training (Best Quality)");

    // Hyperparameters - optimized for quality
    const NUM_ENVS: usize = 16;  // More environments for better sampling
    const NUM_STEPS: usize = 256;  // Longer rollouts
    const TOTAL_TIMESTEPS: usize = 2_000_000;  // Extended training
    const LEARNING_RATE: f64 = 2.5e-4;  // Lower LR for stability

    // Environment dimensions
    let env = CartPole::new();
    let obs_space = env.observation_space();
    let action_space = env.action_space();

    let obs_dim = obs_space.shape[0] as i64;
    let action_dim = match action_space.space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n as i64,
        _ => panic!("Expected discrete action space"),
    };

    tracing::info!("Environment: CartPole-v1 (Best Quality Training)");
    tracing::info!("  Observation dim: {}", obs_dim);
    tracing::info!("  Action dim: {}", action_dim);
    tracing::info!("  Num envs: {}", NUM_ENVS);
    tracing::info!("  Steps per rollout: {}", NUM_STEPS);
    tracing::info!("  Total timesteps: {}", TOTAL_TIMESTEPS);

    // Create environment pool
    let mut env_pool = EnvPool::new(CartPole::new, NUM_ENVS);

    // Create policy with larger network
    tracing::info!("Creating MLP policy...");
    let mut policy = MlpPolicy::new(obs_dim, action_dim, 128);  // Larger hidden layer
    let device = policy.device();
    tracing::info!("  Device: {:?}", device);

    // Create optimizer
    let optimizer = policy.optimizer(LEARNING_RATE);

    // Create PPO trainer with optimized hyperparameters
    let config = PPOConfig::new()
        .learning_rate(LEARNING_RATE)
        .n_epochs(10)  // More training epochs per batch
        .batch_size(128)  // Larger batch size
        .gamma(0.99)
        .gae_lambda(0.95)
        .clip_range(0.2)
        .vf_coef(0.5)
        .ent_coef(0.01)
        .max_grad_norm(0.5);

    let dummy_policy = MlpPolicy::new(obs_dim, action_dim, 128);
    let mut trainer = PPOTrainer::new(config, dummy_policy)?;
    trainer.set_optimizer(optimizer);

    // Create rollout buffer
    let mut buffer = RolloutBuffer::new(NUM_STEPS, NUM_ENVS, obs_dim as usize);

    // Training loop
    tracing::info!("Starting training for {} timesteps...", TOTAL_TIMESTEPS);

    let mut observations = env_pool.reset();
    let num_updates = TOTAL_TIMESTEPS / (NUM_STEPS * NUM_ENVS);

    for update in 0..num_updates {
        // Collect rollout
        buffer.reset();

        for step in 0..NUM_STEPS {
            // Convert observations to tensor
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

            // Store transitions and update observations
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

                // Update observation for next step
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

        buffer.compute_advantages(&last_values_vec, 0.99, 0.95);

        // Get training batch
        let batch = buffer.get_batch();

        // Convert to tensors (batch.observations is already flattened)
        let batch_size = batch.observations.len() / obs_dim as usize;

        let obs_tensor = tch::Tensor::from_slice(&batch.observations)
            .reshape([batch_size as i64, obs_dim])
            .to_device(device);
        let actions_tensor = tch::Tensor::from_slice(&batch.actions)
            .to_device(device);
        let old_log_probs_tensor = tch::Tensor::from_slice(&batch.old_log_probs)
            .to_device(device);
        let old_values_tensor = tch::Tensor::from_slice(&batch.old_values)
            .to_device(device);
        let advantages_tensor = tch::Tensor::from_slice(&batch.advantages)
            .to_device(device);
        let returns_tensor = tch::Tensor::from_slice(&batch.returns)
            .to_device(device);

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

        // Log progress every 10 updates
        if update % 10 == 0 {
            let timesteps = trainer.total_steps();
            let episodes = trainer.total_episodes();
            let avg_steps_per_episode = if episodes > 0 {
                timesteps as f64 / episodes as f64
            } else {
                0.0
            };

            tracing::info!(
                "Update {}/{} | Steps: {} | Episodes: {} | Avg Steps/Ep: {:.1} | Loss: {:.3} | Policy: {:.3} | Value: {:.3} | Entropy: {:.3}",
                update + 1,
                num_updates,
                timesteps,
                episodes,
                avg_steps_per_episode,
                stats.total_loss,
                stats.policy_loss,
                stats.value_loss,
                stats.entropy,
            );
        }
    }

    tracing::info!("✅ Training complete!");
    tracing::info!("Total steps: {}", trainer.total_steps());
    tracing::info!("Total episodes: {}", trainer.total_episodes());
    let avg_steps = trainer.total_steps() as f64 / trainer.total_episodes() as f64;
    tracing::info!("Average steps per episode: {:.1}", avg_steps);

    // Save model in PyTorch format
    let save_path = "cartpole_model_best.pt";
    policy.save(save_path)?;
    tracing::info!("💾 Model saved to {}", save_path);

    // Export model to JSON for web demo (bypasses PyTorch loading issues)
    tracing::info!("🔄 Exporting model to JSON for web...");
    let exported_model = policy.export_for_inference();
    let json_path = "cartpole_model_best.json";
    exported_model.save_json(json_path)?;
    tracing::info!("✅ Model exported to {}", json_path);
    tracing::info!("📦 File size: {} bytes", std::fs::metadata(json_path)?.len());

    Ok(())
}
