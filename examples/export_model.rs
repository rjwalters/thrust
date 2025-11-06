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
//! ```

use anyhow::Result;
use std::env;
use thrust_rl::policy::mlp::MlpPolicy;

fn main() -> Result<()> {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <input_model.pt> <output_model.json>", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} cartpole_model.pt cartpole_wasm.json", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    println!("🔄 Loading model from: {}", input_path);

    // Create policy with matching architecture (CartPole: 4 obs, 2 actions, 128 hidden)
    let mut policy = MlpPolicy::new(4, 2, 128);

    // Load trained weights
    policy.load(input_path)?;

    println!("✅ Model loaded successfully");
    println!("📊 Architecture:");
    println!("  - Input dim: {}", 4);
    println!("  - Output dim (actions): {}", 2);
    println!("  - Hidden dim: {}", 128);

    // Export to pure Rust format
    println!("🔄 Exporting to WASM-compatible format...");
    let exported_model = policy.export_for_inference();

    // Save as JSON
    exported_model.save_json(output_path)?;

    println!("✅ Model exported to: {}", output_path);
    println!();
    println!("📦 File size: {} bytes", std::fs::metadata(output_path)?.len());
    println!();
    println!("Next steps:");
    println!("  1. Build WASM module: wasm-pack build --target web");
    println!("  2. Use in web app with exported JSON weights");

    Ok(())
}
