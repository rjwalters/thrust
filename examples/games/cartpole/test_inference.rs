//! Test CartPole inference to verify model works correctly

use anyhow::Result;
use thrust_rl::{
    env::{Environment, cartpole::CartPole},
    policy::inference::InferenceModel,
    policy::mlp::MlpPolicy,
};

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    // Load the trained model in PyTorch format
    let mut pytorch_policy = MlpPolicy::new(4, 2, 256);
    pytorch_policy.load("models/policies/cartpole/cartpole_model_best.pt")?;
    tracing::info!("✅ Loaded PyTorch model");

    // Export to inference model
    let inference_model = pytorch_policy.export_for_inference();
    tracing::info!("✅ Exported to inference model");

    // Create environment
    let mut env = CartPole::new();
    env.reset();
    let obs = env.get_state();
    tracing::info!("Initial observation: {:?}", obs);

    // Test both models on same observation
    let obs_tensor = tch::Tensor::from_slice(&obs).reshape([1, 4]).to_device(pytorch_policy.device());

    // PyTorch model
    let (pytorch_actions, _, _) = pytorch_policy.get_action(&obs_tensor);
    let pytorch_action: i64 = pytorch_actions.int64_value(&[0]);
    tracing::info!("PyTorch action: {}", pytorch_action);

    // Inference model
    let inference_action = inference_model.get_action(&obs);
    tracing::info!("Inference action: {}", inference_action);

    if pytorch_action as usize != inference_action {
        tracing::error!("❌ MISMATCH! PyTorch: {}, Inference: {}", pytorch_action, inference_action);
    } else {
        tracing::info!("✅ Actions match!");
    }

    // Run 10 episodes with each model
    tracing::info!("\n🎮 Running 10 episodes with PyTorch model...");
    let pytorch_avg = test_policy_pytorch(&mut pytorch_policy, 10)?;
    tracing::info!("PyTorch average steps: {:.1}", pytorch_avg);

    tracing::info!("\n🎮 Running 10 episodes with Inference model...");
    let inference_avg = test_policy_inference(&inference_model, 10)?;
    tracing::info!("Inference average steps: {:.1}", inference_avg);

    Ok(())
}

fn test_policy_pytorch(policy: &mut MlpPolicy, num_episodes: usize) -> Result<f64> {
    let mut total_steps = 0;

    for episode in 0..num_episodes {
        let mut env = CartPole::new();
        env.reset();
        let mut steps = 0;

        loop {
            let obs = env.get_state();
            let obs_tensor = tch::Tensor::from_slice(&obs).reshape([1, 4]).to_device(policy.device());
            let (actions, _, _) = policy.get_action(&obs_tensor);
            let action: i64 = actions.int64_value(&[0]);

            let result = env.step(action);
            steps += 1;

            if result.terminated || result.truncated {
                tracing::info!("  Episode {} finished with {} steps", episode + 1, steps);
                break;
            }
        }

        total_steps += steps;
    }

    Ok(total_steps as f64 / num_episodes as f64)
}

fn test_policy_inference(policy: &InferenceModel, num_episodes: usize) -> Result<f64> {
    let mut total_steps = 0;

    for episode in 0..num_episodes {
        let mut env = CartPole::new();
        env.reset();
        let mut steps = 0;

        loop {
            let obs = env.get_state();
            let action = policy.get_action(&obs) as i64;

            let result = env.step(action);
            steps += 1;

            if result.terminated || result.truncated {
                tracing::info!("  Episode {} finished with {} steps", episode + 1, steps);
                break;
            }
        }

        total_steps += steps;
    }

    Ok(total_steps as f64 / num_episodes as f64)
}
