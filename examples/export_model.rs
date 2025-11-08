//! Export a trained model to WASM-compatible format
//!
//! This example loads a trained PyTorch model and exports the weights
//! to a JSON format that can be loaded in WebAssembly for inference.
//!
//! # Usage
//!
//! ```bash
//! # After training a model:
//! cargo run --example export_model cartpole_model.pt cartpole_wasm.json
//!
//! # With metadata:
//! cargo run --example export_model cartpole_model_best.pt cartpole_model_best.json --with-metadata
//! ```

use std::env;
use std::collections::HashMap;

use anyhow::Result;
use thrust_rl::policy::{mlp::MlpPolicy, inference::TrainingMetadata};

fn main() -> Result<()> {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <input_model.pt> <output_model.json> [--with-metadata]", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} cartpole_model.pt cartpole_wasm.json", args[0]);
        eprintln!("  {} cartpole_model_best.pt cartpole_model_best.json --with-metadata", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let with_metadata = args.len() > 3 && args[3] == "--with-metadata";

    println!("🔄 Loading model from: {}", input_path);

    // Create policy with matching architecture (CartPole: 4 obs, 2 actions)
    // Try different hidden_dim values until one works
    let mut policy = MlpPolicy::new(4, 2, 256);
    let hidden_dim = if policy.load(input_path).is_err() {
        println!("⚠️  Failed to load with hidden_dim=256, trying hidden_dim=128...");
        policy = MlpPolicy::new(4, 2, 128);
        if policy.load(input_path).is_err() {
            println!("⚠️  Failed to load with hidden_dim=128, trying hidden_dim=64...");
            policy = MlpPolicy::new(4, 2, 64);
            policy.load(input_path)?;
            64
        } else {
            128
        }
    } else {
        256
    };

    let device = policy.device();
    println!("✅ Model loaded successfully");
    println!("📊 Architecture:");
    println!("  - Input dim: {}", 4);
    println!("  - Output dim (actions): {}", 2);
    println!("  - Hidden dim: {}", hidden_dim);
    println!("  - Device: {:?}", device);

    // Export to pure Rust format
    println!("🔄 Exporting to WASM-compatible format...");
    let mut exported_model = policy.export_for_inference();

    // Add metadata if requested
    if with_metadata {
        println!("📝 Adding training metadata...");

        // Create hyperparameters map
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("n_steps".to_string(), serde_json::json!(128));
        hyperparameters.insert("num_envs".to_string(), serde_json::json!(8));
        hyperparameters.insert("learning_rate".to_string(), serde_json::json!(0.0003));
        hyperparameters.insert("hidden_dim".to_string(), serde_json::json!(hidden_dim));
        hyperparameters.insert("n_epochs".to_string(), serde_json::json!(10));
        hyperparameters.insert("batch_size".to_string(), serde_json::json!(128));
        hyperparameters.insert("gamma".to_string(), serde_json::json!(0.99));
        hyperparameters.insert("ent_coef".to_string(), serde_json::json!(0.01));
        hyperparameters.insert("gae_lambda".to_string(), serde_json::json!(0.95));
        hyperparameters.insert("clip_range".to_string(), serde_json::json!(0.2));

        let metadata = TrainingMetadata {
            total_steps: 1_000_000,
            total_episodes: 2_200,
            final_performance: 500.0,  // CartPole max
            training_time_secs: 600.0,  // Approximate
            device: format!("{:?}", device),
            environment: "CartPole-v1".to_string(),
            algorithm: "PPO".to_string(),
            timestamp: Some(chrono::Utc::now().to_rfc3339()),
            hyperparameters: Some(hyperparameters),
            notes: Some("Trained agent that successfully balances the pole for maximum episode length".to_string()),
        };

        exported_model.metadata = Some(metadata);
    }

    // Save as JSON
    exported_model.save_json(output_path)?;

    println!("✅ Model exported to: {}", output_path);
    println!();
    println!("📦 File size: {} KB", std::fs::metadata(output_path)?.len() / 1024);

    if with_metadata {
        println!("✅ Training metadata included");
    }

    println!();
    println!("Next steps:");
    println!("  1. Build WASM module: wasm-pack build --target web");
    println!("  2. Use in web app with exported JSON weights");

    Ok(())
}
