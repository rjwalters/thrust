///! Debug CartPole environment and policy outputs
use anyhow::Result;
use thrust_rl::{
    env::{cartpole::CartPole, Environment},
    policy::mlp::MlpPolicy,
};
use tch::{Kind, Tensor};

fn main() -> Result<()> {
    println!("🔍 Debugging CartPole training...\n");

    // Create env and policy
    let mut env = CartPole::new();
    let policy = MlpPolicy::new(4, 2, 64);
    let device = policy.device();

    // Reset environment
    let mut obs = env.reset()?;
    println!("Initial observation: {:?}", obs);
    println!();

    // Run a few steps
    for step in 0..10 {
        // Get action from policy
        let obs_tensor = Tensor::from_slice(&obs).reshape([1, 4]).to_device(device);
        let (actions, log_probs, values) = policy.get_action(&obs_tensor);

        let action: i64 = actions.int64_value(&[0]);
        let log_prob: f64 = log_probs.double_value(&[0]);
        let value: f64 = values.double_value(&[0]);

        println!("Step {}:", step + 1);
        println!("  Observation: {:?}", obs);
        println!("  Action: {} (0=left, 1=right)", action);
        println!("  Log prob: {:.4}", log_prob);
        println!("  Value estimate: {:.4}", value);

        // Step environment
        let result = env.step(action)?;

        println!("  Reward: {}", result.reward);
        println!("  Terminated: {}", result.terminated);
        println!("  Truncated: {}", result.truncated);

        if result.terminated || result.truncated {
            println!("\n❌ Episode ended after {} steps", step + 1);
            break;
        }

        // Update observation
        obs = result.observation;
        println!();
    }

    // Check policy distribution
    println!("\n📊 Checking policy distribution...");
    let test_obs = Tensor::randn([100, 4], (Kind::Float, device));
    let (test_actions, test_log_probs, _) = policy.get_action(&test_obs);

    let actions_vec: Vec<i64> = Vec::try_from(test_actions)?;
    let log_probs_vec: Vec<f32> = Vec::try_from(test_log_probs)?;

    let left_count = actions_vec.iter().filter(|&&a| a == 0).count();
    let right_count = actions_vec.iter().filter(|&&a| a == 1).count();

    println!("Action distribution over 100 random observations:");
    println!("  Left (0): {} ({:.1}%)", left_count, left_count as f32);
    println!("  Right (1): {} ({:.1}%)", right_count, right_count as f32);

    let avg_log_prob: f32 = log_probs_vec.iter().sum::<f32>() / log_probs_vec.len() as f32;
    println!("  Average log prob: {:.4}", avg_log_prob);

    // Check entropy
    let (logits, _) = policy.forward(&test_obs);
    let log_probs_all = logits.log_softmax(-1, Kind::Float);
    let probs = log_probs_all.exp();
    let entropy = -(probs * log_probs_all).sum_dim_intlist(-1, false, Kind::Float).mean(Kind::Float);
    let entropy_val: f64 = entropy.try_into()?;

    println!("  Policy entropy: {:.4}", entropy_val);
    println!("  (max entropy for 2 actions: {:.4})", 0.693f64.ln()); // ln(2)

    if entropy_val < 0.1 {
        println!("\n⚠️  Policy is nearly deterministic from initialization!");
    } else {
        println!("\n✅ Policy has reasonable entropy");
    }

    Ok(())
}
