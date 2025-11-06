//! Export a trained Snake CNN model to WASM-compatible format
//!
//! This example loads a trained PyTorch Snake model and exports the weights
//! to a JSON format that can be loaded in WebAssembly for inference.

use anyhow::Result;
use std::env;
use tch::{nn, Device, Tensor, Kind};
use thrust_rl::policy::snake_cnn::SnakeCNN;

fn main() -> Result<()> {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <input_model.pt> <output_model.json>", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} models/snake_policy.pt web/public/snake_model.json", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let grid_size = 20; // Default grid size

    println!("🔄 Loading Snake CNN model from: {}", input_path);

    // Create variable store and load weights
    let mut vs = nn::VarStore::new(Device::Cpu);
    let _policy = SnakeCNN::new(&vs.root(), grid_size, 5);

    // Load trained weights
    vs.load(input_path)?;

    println!("✅ Model loaded successfully");
    println!("📊 Architecture:");
    println!("  - Grid size: {}x{}", grid_size, grid_size);
    println!("  - Input channels: {}", 5);
    println!("  - Output actions: {}", 4);

    // Helper function to convert a 4D tensor to Vec<Vec<Vec<Vec<f32>>>>
    fn tensor_to_4d(tensor: &Tensor) -> Vec<Vec<Vec<Vec<f32>>>> {
        let size = tensor.size();
        assert_eq!(size.len(), 4, "Expected 4D tensor");
        let d0 = size[0] as usize;
        let d1 = size[1] as usize;
        let d2 = size[2] as usize;
        let d3 = size[3] as usize;

        let cpu_tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
        let flat_tensor = cpu_tensor.view([-1]);
        let flat: Vec<f32> = Vec::try_from(&flat_tensor).unwrap();

        let mut result = Vec::with_capacity(d0);
        for i0 in 0..d0 {
            let mut layer1 = Vec::with_capacity(d1);
            for i1 in 0..d1 {
                let mut layer2 = Vec::with_capacity(d2);
                for i2 in 0..d2 {
                    let start = ((i0 * d1 + i1) * d2 + i2) * d3;
                    layer2.push(flat[start..start + d3].to_vec());
                }
                layer1.push(layer2);
            }
            result.push(layer1);
        }
        result
    }

    // Helper function to convert a 2D tensor to Vec<Vec<f32>>
    fn tensor_to_2d(tensor: &Tensor) -> Vec<Vec<f32>> {
        let size = tensor.size();
        assert_eq!(size.len(), 2, "Expected 2D tensor");
        let rows = size[0] as usize;
        let cols = size[1] as usize;

        let cpu_tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
        let flat_tensor = cpu_tensor.view([-1]);
        let flat: Vec<f32> = Vec::try_from(&flat_tensor).unwrap();

        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            result.push(flat[i * cols..(i + 1) * cols].to_vec());
        }
        result
    }

    // Helper function to convert a 1D tensor to Vec<f32>
    fn tensor_to_1d(tensor: &Tensor) -> Vec<f32> {
        let cpu_tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
        let flat_tensor = cpu_tensor.view([-1]); // Flatten to 1D
        Vec::try_from(&flat_tensor).unwrap()
    }

    println!("🔄 Extracting weights...");

    // Get the variable store's named variables
    let variables = vs.variables();

    // Extract weights
    let conv1_weight = tensor_to_4d(variables.get("conv1.weight").expect("Missing conv1.weight"));
    let conv1_bias = tensor_to_1d(variables.get("conv1.bias").expect("Missing conv1.bias"));

    let conv2_weight = tensor_to_4d(variables.get("conv2.weight").expect("Missing conv2.weight"));
    let conv2_bias = tensor_to_1d(variables.get("conv2.bias").expect("Missing conv2.bias"));

    let conv3_weight = tensor_to_4d(variables.get("conv3.weight").expect("Missing conv3.weight"));
    let conv3_bias = tensor_to_1d(variables.get("conv3.bias").expect("Missing conv3.bias"));

    let fc_common_weight = tensor_to_2d(variables.get("fc_common.weight").expect("Missing fc_common.weight"));
    let fc_common_bias = tensor_to_1d(variables.get("fc_common.bias").expect("Missing fc_common.bias"));

    let fc_policy_weight = tensor_to_2d(variables.get("policy.weight").expect("Missing policy.weight"));
    let fc_policy_bias = tensor_to_1d(variables.get("policy.bias").expect("Missing policy.bias"));

    let fc_value_weight = tensor_to_2d(variables.get("value.weight").expect("Missing value.weight"));
    let fc_value_bias = tensor_to_1d(variables.get("value.bias").expect("Missing value.bias"));

    // Create inference model
    let exported_model = thrust_rl::inference::snake::SnakeCNNInference {
        grid_width: grid_size as usize,
        grid_height: grid_size as usize,
        input_channels: 5,
        num_actions: 4,
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
        conv3_weight,
        conv3_bias,
        fc_common_weight,
        fc_common_bias,
        fc_policy_weight,
        fc_policy_bias,
        fc_value_weight,
        fc_value_bias,
    };

    // Save as JSON
    println!("🔄 Exporting to WASM-compatible format...");
    exported_model.save_json(output_path)?;

    println!("✅ Model exported to: {}", output_path);
    println!();
    println!("📦 File size: {} bytes", std::fs::metadata(output_path)?.len());
    println!();
    println!("Next steps:");
    println!("  1. Model is ready to use in WASM");
    println!("  2. Load it in the web app with the Snake environment");

    Ok(())
}
