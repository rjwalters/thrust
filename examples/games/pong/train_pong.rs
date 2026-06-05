//! Train PPO on Pong — agent (left paddle) vs rule-based opponent (right
//! paddle).
//!
//! # Usage
//!
//! ```bash
//! # Local (CPU)
//! cargo run --example train_pong --release --features training
//!
//! # Remote GPU
//! ssh alc-2 "cd ~/thrust && cargo run --example train_pong --release --features training"
//! ```

use std::collections::HashMap;

use anyhow::Result;
use thrust_rl::{
    buffer::rollout::RolloutBuffer,
    env::{Environment, pong::Pong, pool::EnvPool},
    policy::{inference::TrainingMetadata, mlp::MlpPolicy},
    train::ppo::{PPOConfig, PPOTrainer},
};

const NUM_ENVS: usize = 32;
const NUM_STEPS: usize = 128;
const TOTAL_TIMESTEPS: usize = 20_000_000;
const LEARNING_RATE: f64 = 0.0003;
const HIDDEN_DIM: i64 = 128;
const CHECKPOINT_INTERVAL_SECS: u64 = 300;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    tracing::info!("🏓 Starting Pong PPO Training");

    let env = Pong::new();
    let obs_dim = env.observation_space().shape[0] as i64;
    let action_dim = match env.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n as i64,
        _ => panic!("Expected discrete action space"),
    };

    tracing::info!(
        "obs_dim={obs_dim}, action_dim={action_dim}, num_envs={NUM_ENVS}, steps={NUM_STEPS}"
    );

    let mut env_pool = EnvPool::new(Pong::new, NUM_ENVS);
    let mut policy = MlpPolicy::new(obs_dim, action_dim, HIDDEN_DIM);
    let device = policy.device();
    tracing::info!("Device: {device:?}");

    let optimizer = policy.optimizer(LEARNING_RATE);

    let config = PPOConfig::new()
        .learning_rate(LEARNING_RATE)
        .n_epochs(10)
        .batch_size(512)
        .gamma(0.99)
        .gae_lambda(0.95)
        .clip_range(0.2)
        .clip_range_vf(f64::INFINITY)
        .vf_coef(0.5)
        .ent_coef(0.01)
        .max_grad_norm(0.5);

    let dummy_policy = MlpPolicy::new(obs_dim, action_dim, HIDDEN_DIM);
    let mut trainer = PPOTrainer::new(config, dummy_policy)?;
    trainer.set_optimizer(optimizer);

    let mut buffer = RolloutBuffer::new(NUM_STEPS, NUM_ENVS, obs_dim as usize);
    let mut observations = env_pool.reset();
    let num_updates = TOTAL_TIMESTEPS / (NUM_STEPS * NUM_ENVS);
    let mut last_checkpoint = std::time::Instant::now();
    let train_start = std::time::Instant::now();

    tracing::info!("Training for {num_updates} updates ({TOTAL_TIMESTEPS} total steps)...");

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

        // Bootstrap last values
        let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let obs_tensor = tch::Tensor::from_slice(&obs_flat)
            .reshape([NUM_ENVS as i64, obs_dim])
            .to_device(device);
        let (_, _, last_values) = policy.get_action(&obs_tensor);
        let last_values_vec: Vec<f32> = Vec::try_from(last_values)?;

        buffer.compute_advantages(&last_values_vec, 0.99, 0.95);

        let batch = buffer.get_batch();
        let batch_size = batch.observations.len() / obs_dim as usize;

        let obs_t = tch::Tensor::from_slice(&batch.observations)
            .reshape([batch_size as i64, obs_dim])
            .to_device(device);
        let acts_t = tch::Tensor::from_slice(&batch.actions).to_device(device);
        let old_lp_t = tch::Tensor::from_slice(&batch.old_log_probs).to_device(device);
        let old_v_t = tch::Tensor::from_slice(&batch.old_values).to_device(device);
        let adv_t = tch::Tensor::from_slice(&batch.advantages).to_device(device);
        let ret_t = tch::Tensor::from_slice(&batch.returns).to_device(device);

        let stats = trainer.train_step_with_policy(
            &policy,
            &obs_t,
            &acts_t,
            &old_lp_t,
            &old_v_t,
            &adv_t,
            &ret_t,
            |p, obs, acts| p.evaluate_actions(obs, acts),
        )?;

        if update % 20 == 0 {
            let steps = trainer.total_steps();
            let episodes = trainer.total_episodes();
            let avg = if episodes > 0 {
                steps as f64 / episodes as f64
            } else {
                0.0
            };
            tracing::info!(
                "Update {}/{} | Steps: {} | Episodes: {} | Avg: {:.1} | Loss: {:.3} | Policy: {:.3} | Value: {:.3} | Entropy: {:.3} | ExpVar: {:.3}",
                update + 1,
                num_updates,
                steps,
                episodes,
                avg,
                stats.total_loss,
                stats.policy_loss,
                stats.value_loss,
                stats.entropy,
                stats.explained_var,
            );
        }

        if last_checkpoint.elapsed().as_secs() >= CHECKPOINT_INTERVAL_SECS {
            let steps = trainer.total_steps();
            let cp = format!("pong_checkpoint_{steps}steps.pt");
            if let Err(e) = policy.save(&cp) {
                tracing::warn!("Failed to save checkpoint {cp}: {e}");
            } else {
                tracing::info!("💾 Checkpoint: {cp}");
            }
            last_checkpoint = std::time::Instant::now();
        }
    }

    tracing::info!(
        "✅ Training complete! Steps={} Episodes={}",
        trainer.total_steps(),
        trainer.total_episodes()
    );

    let elapsed = train_start.elapsed().as_secs_f64();
    let avg_steps = trainer.total_steps() as f64 / trainer.total_episodes() as f64;

    // Save PyTorch model
    policy.save("pong_model.pt")?;
    tracing::info!("💾 pong_model.pt saved");

    // Export JSON with metadata for web demo
    let mut model = policy.export_for_inference();
    let mut hparams = HashMap::new();
    hparams.insert("num_envs".to_string(), serde_json::json!(NUM_ENVS));
    hparams.insert("num_steps".to_string(), serde_json::json!(NUM_STEPS));
    hparams.insert("hidden_dim".to_string(), serde_json::json!(HIDDEN_DIM));
    hparams.insert("learning_rate".to_string(), serde_json::json!(LEARNING_RATE));
    hparams.insert("gamma".to_string(), serde_json::json!(0.99));
    hparams.insert("ent_coef".to_string(), serde_json::json!(0.01));
    hparams.insert("opponent_speed".to_string(), serde_json::json!(0.015));

    model.metadata = Some(TrainingMetadata {
        total_steps: trainer.total_steps(),
        total_episodes: trainer.total_episodes(),
        final_performance: avg_steps,
        training_time_secs: elapsed,
        device: format!("{device:?}"),
        environment: "Pong".to_string(),
        algorithm: "PPO".to_string(),
        timestamp: Some(chrono::Utc::now().to_rfc3339()),
        hyperparameters: Some(hparams),
        notes: Some("Agent (left) vs rule-based opponent at 60% speed".to_string()),
    });

    model.save_json("pong_model.json")?;
    tracing::info!("✅ pong_model.json exported");

    Ok(())
}
