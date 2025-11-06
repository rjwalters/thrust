//! Integration test for the complete training→export→inference pipeline
//!
//! This test validates that we can:
//! 1. Train a policy on GPU/CPU
//! 2. Export weights to JSON format (bypassing PyTorch serialization)
//! 3. Load the JSON weights into our pure Rust inference engine
//! 4. Run inference and get valid action predictions
//!
//! This ensures our remote GPU training workflow works end-to-end.

use anyhow::Result;
use thrust_rl::{
    buffer::rollout::RolloutBuffer,
    env::{cartpole::CartPole, pool::EnvPool, Environment},
    policy::{mlp::MlpPolicy, inference::InferenceModel},
    train::ppo::{PPOConfig, PPOTrainer},
};

#[test]
fn test_train_export_load_inference() -> Result<()> {
    // Use small hyperparameters for fast test
    const NUM_ENVS: usize = 4;
    const NUM_STEPS: usize = 64;
    const TOTAL_TIMESTEPS: usize = 512;  // Very short training
    const LEARNING_RATE: f64 = 3e-4;

    // Environment dimensions
    let env = CartPole::new();
    let obs_space = env.observation_space();
    let action_space = env.action_space();

    let obs_dim = obs_space.shape[0] as i64;
    let action_dim = match action_space.space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n as i64,
        _ => panic!("Expected discrete action space"),
    };

    println!("=== Step 1: Training Policy ===");

    // Create environment pool
    let mut env_pool = EnvPool::new(CartPole::new, NUM_ENVS);

    // Create policy
    let mut policy = MlpPolicy::new(obs_dim, action_dim, 64);
    let device = policy.device();
    println!("Training on device: {:?}", device);

    // Create optimizer
    let optimizer = policy.optimizer(LEARNING_RATE);

    // Create PPO trainer
    let config = PPOConfig::new()
        .learning_rate(LEARNING_RATE)
        .n_epochs(4)
        .batch_size(64)
        .gamma(0.99)
        .gae_lambda(0.95)
        .clip_range(0.2)
        .vf_coef(0.5)
        .ent_coef(0.01)
        .max_grad_norm(0.5);

    let dummy_policy = MlpPolicy::new(obs_dim, action_dim, 64);
    let mut trainer = PPOTrainer::new(config, dummy_policy)?;
    trainer.set_optimizer(optimizer);

    // Create rollout buffer
    let mut buffer = RolloutBuffer::new(NUM_STEPS, NUM_ENVS, obs_dim as usize);

    // Short training loop
    let mut observations = env_pool.reset();
    let num_updates = TOTAL_TIMESTEPS / (NUM_STEPS * NUM_ENVS);

    for update in 0..num_updates {
        buffer.reset();

        for step in 0..NUM_STEPS {
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_tensor = tch::Tensor::from_slice(&obs_flat)
                .reshape([NUM_ENVS as i64, obs_dim])
                .to_device(device);

            let (actions, log_probs, values) = policy.get_action(&obs_tensor);
            let actions_vec: Vec<i64> = Vec::try_from(actions)?;
            let log_probs_vec: Vec<f32> = Vec::try_from(log_probs)?;
            let values_vec: Vec<f32> = Vec::try_from(values)?;

            let results = env_pool.step(&actions_vec);

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

                observations[env_id] = results[env_id].observation.clone();

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
        let _stats = trainer.train_step_with_policy(
            &policy,
            &obs_tensor,
            &actions_tensor,
            &old_log_probs_tensor,
            &old_values_tensor,
            &advantages_tensor,
            &returns_tensor,
            |p, obs, acts| p.evaluate_actions(obs, acts),
        )?;

        println!("Update {}/{} complete", update + 1, num_updates);
    }

    println!("✅ Training complete!");
    println!("Total episodes: {}", trainer.total_episodes());

    // === Step 2: Export to JSON ===
    println!("\n=== Step 2: Exporting to JSON ===");

    let exported_model = policy.export_for_inference();

    // Verify dimensions
    assert_eq!(exported_model.obs_dim, obs_dim as usize);
    assert_eq!(exported_model.action_dim, action_dim as usize);
    assert_eq!(exported_model.hidden_dim, 64);

    println!("✅ Export successful!");
    println!("  obs_dim: {}", exported_model.obs_dim);
    println!("  action_dim: {}", exported_model.action_dim);
    println!("  hidden_dim: {}", exported_model.hidden_dim);

    // === Step 3: Save and Load JSON ===
    println!("\n=== Step 3: Testing JSON Serialization ===");

    let json_path = "/tmp/test_model.json";
    exported_model.save_json(json_path)?;
    println!("Saved to {}", json_path);

    let loaded_model = InferenceModel::load_json(json_path)?;
    println!("✅ JSON load successful!");

    // === Step 4: Test Inference ===
    println!("\n=== Step 4: Testing Inference ===");

    // Create a test observation
    let mut test_env = CartPole::new();
    test_env.reset();
    let test_obs = test_env.get_observation();

    // Run inference with loaded model
    let (action_logits, _value) = loaded_model.forward(&test_obs);

    // Get action via softmax
    let max_val = action_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = action_logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    let action = if probs[0] > probs[1] { 0 } else { 1 };

    println!("Test observation: {:?}", test_obs);
    println!("Action logits: {:?}", action_logits);
    println!("Action probabilities: {:?}", probs);
    println!("Selected action: {}", action);

    // Verify action is valid
    assert!(action == 0 || action == 1, "Action must be 0 or 1");

    // Verify probabilities sum to ~1.0
    let prob_sum: f32 = probs.iter().sum();
    assert!((prob_sum - 1.0).abs() < 1e-5, "Probabilities must sum to 1.0");

    println!("✅ Inference successful!");

    // === Step 5: Compare PyTorch vs Pure Rust Inference ===
    println!("\n=== Step 5: Comparing PyTorch vs Pure Rust ===");

    // Get prediction from original PyTorch policy
    let obs_tensor = tch::Tensor::from_slice(&test_obs)
        .reshape([1, obs_dim])
        .to_device(device);
    let (torch_logits, torch_value) = policy.forward(&obs_tensor);
    let torch_logits_vec: Vec<f32> = Vec::try_from(torch_logits.squeeze())?;
    let torch_value_f32: f32 = f64::try_from(torch_value.squeeze())? as f32;

    println!("PyTorch logits: {:?}", torch_logits_vec);
    println!("Pure Rust logits: {:?}", action_logits);
    println!("PyTorch value: {}", torch_value_f32);
    println!("Pure Rust value: {}", _value);

    // Check that outputs are close (allow some numerical difference)
    let max_logit_diff = torch_logits_vec.iter()
        .zip(action_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Max logit difference: {}", max_logit_diff);
    assert!(max_logit_diff < 1e-4, "Logits should match between PyTorch and Pure Rust");

    let value_diff = (torch_value_f32 - _value).abs();
    println!("Value difference: {}", value_diff);
    assert!(value_diff < 1e-4, "Values should match between PyTorch and Pure Rust");

    println!("✅ PyTorch and Pure Rust outputs match!");

    // Clean up
    std::fs::remove_file(json_path).ok();

    println!("\n=== 🎉 All Tests Passed! ===");
    println!("The complete pipeline works:");
    println!("  ✓ Training");
    println!("  ✓ Export to JSON");
    println!("  ✓ JSON serialization/deserialization");
    println!("  ✓ Pure Rust inference");
    println!("  ✓ PyTorch ↔ Pure Rust equivalence");

    Ok(())
}
