//! Unit test to verify PPO can learn from synthetic data
//!
//! This test creates a simple batch where the optimal action is obvious:
//! - When obs=0, action=0 has high advantage (should increase probability)
//! - When obs=1, action=1 has high advantage (should increase probability)
//!
//! If PPO is working correctly, after a few updates:
//! - P(action=0 | obs=0) should increase
//! - P(action=1 | obs=1) should increase

use thrust_rl::{
    policy::mlp::MlpPolicy,
    train::ppo::{PPOConfig, PPOTrainer},
};
use tch::{Device, Kind, Tensor};

#[test]
fn test_ppo_learns_from_synthetic_data() {
    // Create a simple 2-action policy
    let obs_dim = 1;
    let action_dim = 2;
    let mut policy = MlpPolicy::new(obs_dim, action_dim, 64);
    let device = policy.device();

    // Create synthetic training batch
    // 8 samples: 4 with obs=0 (should learn action=0), 4 with obs=1 (should learn action=1)
    let observations = Tensor::from_slice(&[
        0.0_f32, 0.0, 0.0, 0.0,  // obs=0
        1.0, 1.0, 1.0, 1.0,      // obs=1
    ])
    .view([8, 1])
    .to_device(device);

    let actions = Tensor::from_slice(&[
        0_i64, 0, 0, 0,  // action=0 for obs=0
        1, 1, 1, 1,      // action=1 for obs=1
    ])
    .to_device(device);

    // OLD log probs (all equal - uniform random policy initially)
    let old_log_probs = Tensor::from_slice(&[
        -0.693_f32, -0.693, -0.693, -0.693,  // log(0.5)
        -0.693, -0.693, -0.693, -0.693,
    ])
    .to_device(device);

    // OLD values (doesn't matter for this test)
    let old_values = Tensor::zeros([8], (Kind::Float, device));

    // ADVANTAGES: High positive advantage for correct actions
    let advantages = Tensor::from_slice(&[
        10.0_f32, 10.0, 10.0, 10.0,  // action=0 for obs=0 is good
        10.0, 10.0, 10.0, 10.0,      // action=1 for obs=1 is good
    ])
    .to_device(device);

    // RETURNS (doesn't matter much for this test)
    let returns = Tensor::from_slice(&[
        10.0_f32, 10.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0,
    ])
    .to_device(device);

    // Measure initial policy probabilities
    let (initial_logits, _) = policy.forward(&observations);
    let initial_probs = initial_logits.softmax(-1, Kind::Float);

    // Get probabilities for obs=0
    let obs_0_probs_before: Vec<f32> = Vec::try_from(initial_probs.get(0)).unwrap();
    let obs_1_probs_before: Vec<f32> = Vec::try_from(initial_probs.get(4)).unwrap();

    println!("BEFORE training:");
    println!("  P(action=0 | obs=0) = {:.4}", obs_0_probs_before[0]);
    println!("  P(action=1 | obs=1) = {:.4}", obs_1_probs_before[1]);

    // Create PPO trainer
    let config = PPOConfig::new()
        .learning_rate(0.01)  // High learning rate for fast learning
        .n_epochs(10)
        .batch_size(8)
        .clip_range(0.2)
        .vf_coef(0.5)
        .ent_coef(0.0)  // Zero entropy coefficient to focus purely on advantage
        .max_grad_norm(0.5);

    let dummy_policy = MlpPolicy::new(obs_dim, action_dim, 64);
    let mut trainer = PPOTrainer::new(config, dummy_policy).unwrap();

    let optimizer = policy.optimizer(0.01);
    trainer.set_optimizer(optimizer);

    // Run PPO updates
    for i in 0..5 {
        // Debug: Check what evaluate_actions returns
        let (new_log_probs, entropy, new_values) = policy.evaluate_actions(&observations, &actions);
        let new_log_probs_vec: Vec<f32> = Vec::try_from(&new_log_probs).unwrap();
        let old_log_probs_vec: Vec<f32> = Vec::try_from(&old_log_probs).unwrap();

        if i == 0 {
            println!("\nDEBUG iteration {}:", i);
            println!("  old_log_probs[0:4]: {:?}", &old_log_probs_vec[0..4]);
            println!("  new_log_probs[0:4]: {:?}", &new_log_probs_vec[0..4]);

            let ratio = (&new_log_probs - &old_log_probs).exp();
            let ratio_vec: Vec<f32> = Vec::try_from(&ratio).unwrap();
            println!("  ratio[0:4]: {:?}", &ratio_vec[0..4]);

            let advantages_vec: Vec<f32> = Vec::try_from(&advantages).unwrap();
            println!("  advantages[0:4]: {:?}", &advantages_vec[0..4]);

            let policy_term = &ratio * &advantages;
            let policy_term_vec: Vec<f32> = Vec::try_from(&policy_term).unwrap();
            println!("  advantages * ratio[0:4]: {:?}", &policy_term_vec[0..4]);
        }

        let result = trainer.train_step_with_policy(
            &policy,
            &observations,
            &actions,
            &old_log_probs,
            &old_values,
            &advantages,
            &returns,
            |p, obs, acts| p.evaluate_actions(obs, acts),
        );

        match result {
            Ok(stats) => {
                println!("PPO update {}: policy_loss={:.4}, value_loss={:.4}, entropy={:.4}",
                         i, stats.policy_loss, stats.value_loss, stats.entropy);
            }
            Err(e) => {
                panic!("PPO update failed: {}", e);
            }
        }
    }

    // Measure final policy probabilities
    let (final_logits, _) = policy.forward(&observations);
    let final_probs = final_logits.softmax(-1, Kind::Float);

    let obs_0_probs_after: Vec<f32> = Vec::try_from(final_probs.get(0)).unwrap();
    let obs_1_probs_after: Vec<f32> = Vec::try_from(final_probs.get(4)).unwrap();

    println!("\nAFTER training:");
    println!("  P(action=0 | obs=0) = {:.4}", obs_0_probs_after[0]);
    println!("  P(action=1 | obs=1) = {:.4}", obs_1_probs_after[1]);

    // ASSERTIONS: Policy should have learned from advantages
    // With advantages of +10 for correct actions, probabilities should increase significantly
    let delta_0 = obs_0_probs_after[0] - obs_0_probs_before[0];
    let delta_1 = obs_1_probs_after[1] - obs_1_probs_before[1];

    println!("\nChanges:");
    println!("  ΔP(action=0 | obs=0) = {:.4}", delta_0);
    println!("  ΔP(action=1 | obs=1) = {:.4}", delta_1);

    // Policy should learn: probabilities should increase by at least 0.1
    assert!(
        delta_0 > 0.1,
        "PPO failed to learn: P(action=0 | obs=0) only increased by {:.4} (expected > 0.1)",
        delta_0
    );

    assert!(
        delta_1 > 0.1,
        "PPO failed to learn: P(action=1 | obs=1) only increased by {:.4} (expected > 0.1)",
        delta_1
    );

    println!("\n✅ PPO successfully learned from synthetic data!");
}
