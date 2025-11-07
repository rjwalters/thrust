//! Test CartPole JSON model inference

use anyhow::Result;
use thrust_rl::{
    env::{Environment, cartpole::CartPole},
    policy::inference::InferenceModel,
};

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    // Get model path from command line or use default
    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        "cartpole_model_best.json"
    };

    // Load the JSON inference model
    tracing::info!("Loading {}...", model_path);
    let model = InferenceModel::load_json(model_path)?;
    tracing::info!("✅ Model loaded successfully");
    tracing::info!("   Obs dim: {}, Action dim: {}, Hidden dim: {}",
                   model.obs_dim, model.action_dim, model.hidden_dim);
    tracing::info!("   Activation: {:?}", model.activation);

    if let Some(ref metadata) = model.metadata {
        tracing::info!("   Training metadata:");
        tracing::info!("     Total steps: {}", metadata.total_steps);
        tracing::info!("     Total episodes: {}", metadata.total_episodes);
        tracing::info!("     Final performance: {:.1} steps/episode", metadata.final_performance);
    }

    // Test on a few episodes
    tracing::info!("\n🎮 Testing model on 20 episodes...\n");

    let mut total_steps = 0;
    let mut episode_scores = Vec::new();

    for episode in 0..20 {
        let mut env = CartPole::new();
        env.reset();
        let mut steps = 0;

        loop {
            let obs = env.get_state();
            let action = model.get_action(&obs) as i64;

            let result = env.step(action);
            steps += 1;

            if result.terminated || result.truncated {
                tracing::info!("  Episode {:2} finished with {:3} steps", episode + 1, steps);
                episode_scores.push(steps);
                break;
            }
        }

        total_steps += steps;
    }

    let avg_steps = total_steps as f64 / 20.0;
    let max_steps = *episode_scores.iter().max().unwrap();
    let min_steps = *episode_scores.iter().min().unwrap();

    tracing::info!("\n📊 Results:");
    tracing::info!("  Average: {:.1} steps/episode", avg_steps);
    tracing::info!("  Min: {} steps", min_steps);
    tracing::info!("  Max: {} steps", max_steps);

    if avg_steps < 100.0 {
        tracing::error!("\n❌ Performance is POOR! Expected >450 steps/episode but got {:.1}", avg_steps);
        tracing::error!("   This indicates a bug in the model export or inference code.");
    } else if avg_steps < 400.0 {
        tracing::warn!("\n⚠️  Performance is suboptimal. Expected >450 steps/episode but got {:.1}", avg_steps);
    } else {
        tracing::info!("\n✅ Performance looks good! ({:.1} steps/episode)", avg_steps);
    }

    Ok(())
}
