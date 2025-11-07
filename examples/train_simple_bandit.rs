//! Train PPO on SimpleBandit (Correctness Test)
//!
//! This is a sanity check for PPO implementation correctness.
//! SimpleBandit is trivial: state is 0 or 1, optimal action equals state.
//! If PPO can't reach 100% success rate here, there's a bug.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_simple_bandit --release
//! ```

use anyhow::Result;
use thrust_rl::{
    buffer::rollout::RolloutBuffer,
    env::{pool::EnvPool, simple_bandit::SimpleBandit, Environment},
    policy::mlp::MlpPolicy,
    train::ppo::{PPOConfig, PPOTrainer},
};

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    tracing::info!("🧪 SimpleBandit PPO Correctness Test");
    tracing::info!("Expected result: ~100% success rate (1.0 reward every step)");
    tracing::info!("");

    // Hyperparameters - simple configuration for easy task
    const NUM_ENVS: usize = 4;
    const NUM_STEPS: usize = 100;
    const TOTAL_TIMESTEPS: usize = 50_000;  // Should be plenty for this trivial task
    const LEARNING_RATE: f64 = 0.001;

    // Environment dimensions
    let env = SimpleBandit::new();
    let obs_space = env.observation_space();
    let action_space = env.action_space();

    let obs_dim = obs_space.shape[0] as i64;
    let action_dim = match action_space.space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n as i64,
        _ => panic!("Expected discrete action space"),
    };

    tracing::info!("Environment: SimpleBandit");
    tracing::info!("  Observation dim: {}", obs_dim);
    tracing::info!("  Action dim: {}", action_dim);
    tracing::info!("  Num envs: {}", NUM_ENVS);
    tracing::info!("  Steps per rollout: {}", NUM_STEPS);
    tracing::info!("  Total timesteps: {}", TOTAL_TIMESTEPS);

    // Create environment pool
    let mut env_pool = EnvPool::new(SimpleBandit::new, NUM_ENVS);

    // Create policy - small network is fine for this simple task
    tracing::info!("Creating MLP policy...");
    let mut policy = MlpPolicy::new(obs_dim, action_dim, 64);
    let device = policy.device();
    tracing::info!("  Device: {:?}", device);

    // Create optimizer
    let optimizer = policy.optimizer(LEARNING_RATE);

    // Create PPO trainer with config optimized for contextual bandits
    // Key difference: gamma=0.0 and gae_lambda=0.0 disable GAE bootstrapping
    // This gives simple advantages: A = r - V (no temporal dependencies)
    let config = PPOConfig::new()
        .learning_rate(LEARNING_RATE)
        .n_epochs(10)
        .batch_size(64)
        .gamma(0.0)  // No discount - SimpleBandit is a single-step contextual bandit
        .gae_lambda(0.0)  // No GAE bootstrapping - states are independent
        .clip_range(0.2)
        .vf_coef(0.5)
        .ent_coef(0.1)  // Higher entropy to encourage exploration
        .max_grad_norm(0.5);

    let dummy_policy = MlpPolicy::new(obs_dim, action_dim, 64);
    let mut trainer = PPOTrainer::new(config, dummy_policy)?;
    trainer.set_optimizer(optimizer);

    // Create rollout buffer
    let mut buffer = RolloutBuffer::new(NUM_STEPS, NUM_ENVS, obs_dim as usize);

    // Training loop
    tracing::info!("Starting training for {} timesteps...", TOTAL_TIMESTEPS);
    tracing::info!("");

    let mut observations = env_pool.reset();
    let num_updates = TOTAL_TIMESTEPS / (NUM_STEPS * NUM_ENVS);

    // Track cumulative reward for success rate
    let mut total_reward = 0.0;
    let mut total_steps = 0;

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

                // Track reward
                total_reward += results[env_id].reward as f64;
                total_steps += 1;

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

        // Use gamma=0.0, gae_lambda=0.0 for contextual bandit (no temporal dependencies)
        // This gives: A = r - V (simple advantage, no bootstrapping)
        buffer.compute_advantages(&last_values_vec, 0.0, 0.0);

        // Get training batch
        let batch = buffer.get_batch();

        // Convert to tensors
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

        // Debug: Log first few samples to understand what's happening
        if update == 0 || update == 10 || update == 50 {
            tracing::info!("=== DEBUG: Update {} ===", update);
            tracing::info!("Sample observations (first 5): {:?}", &batch.observations[..5.min(batch.observations.len())]);
            tracing::info!("Sample actions (first 5): {:?}", &batch.actions[..5.min(batch.actions.len())]);
            tracing::info!("Sample advantages (first 5): {:?}", &batch.advantages[..5.min(batch.advantages.len())]);
            tracing::info!("Sample old_log_probs (first 5): {:?}", &batch.old_log_probs[..5.min(batch.old_log_probs.len())]);

            // Check policy outputs for a few samples
            let sample_obs = tch::Tensor::from_slice(&batch.observations[..obs_dim as usize])
                .reshape([1, obs_dim])
                .to_device(device);
            let (sample_logits, sample_values) = policy.forward(&sample_obs);
            let sample_probs = sample_logits.softmax(-1, tch::Kind::Float);
            // Flatten 2D tensors before converting to Vec
            tracing::info!("Sample policy probs: {:?}", Vec::<f32>::try_from(sample_probs.flatten(0, -1)).unwrap());
            tracing::info!("Sample value: {:?}", Vec::<f32>::try_from(sample_values.flatten(0, -1)).unwrap());
        }

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
        if update % 10 == 0 || update == num_updates - 1 {
            let success_rate = total_reward / total_steps as f64;
            let timesteps = trainer.total_steps();
            let episodes = trainer.total_episodes();

            tracing::info!(
                "Update {}/{} | Steps: {} | Episodes: {} | Success Rate: {:.1}% | Loss: {:.3} | Entropy: {:.3}",
                update + 1,
                num_updates,
                timesteps,
                episodes,
                success_rate * 100.0,
                stats.total_loss,
                stats.entropy,
            );

            // Warn if success rate is not improving
            if update > 50 && success_rate < 0.8 {
                tracing::warn!("⚠️  Success rate is low after {} updates. Expected ~100% for this trivial task.", update);
            }
        }
    }

    // Final evaluation
    let final_success_rate = total_reward / total_steps as f64;

    tracing::info!("");
    tracing::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    tracing::info!("🏁 Training Complete");
    tracing::info!("Total steps: {}", trainer.total_steps());
    tracing::info!("Total episodes: {}", trainer.total_episodes());
    tracing::info!("Final success rate: {:.1}%", final_success_rate * 100.0);
    tracing::info!("");

    if final_success_rate >= 0.95 {
        tracing::info!("✅ SUCCESS: PPO correctly learned SimpleBandit!");
        tracing::info!("   This suggests PPO implementation is working correctly.");
        tracing::info!("   CartPole variance may be environment-specific or require more investigation.");
    } else if final_success_rate >= 0.8 {
        tracing::warn!("⚠️  MARGINAL: Success rate is {:.1}%, expected ~100%", final_success_rate * 100.0);
        tracing::warn!("   PPO may have implementation issues or suboptimal hyperparameters.");
    } else {
        tracing::error!("❌ FAILURE: Success rate is only {:.1}%!", final_success_rate * 100.0);
        tracing::error!("   This indicates a bug in the PPO implementation.");
        tracing::error!("   SimpleBandit is trivial - a working PPO should reach ~100%.");
    }

    Ok(())
}
