#!/usr/bin/env python3
"""Export PyTorch model weights to JSON for WASM inference."""

import sys
import json
import torch

def export_model(input_path, output_path):
    """Export PyTorch model to JSON format."""
    print(f"🔄 Loading model from: {input_path}")

    # Try to load as state dict first, then as JIT model
    try:
        state_dict = torch.load(input_path, map_location='cpu', weights_only=False)
        # Check if it's a JIT model
        if isinstance(state_dict, torch.jit.ScriptModule):
            print("Detected TorchScript model, extracting state dict...")
            state_dict = state_dict.state_dict()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("✅ Model loaded successfully")
    print("📊 Model structure:")
    for key in state_dict.keys():
        print(f"  - {key}: {state_dict[key].shape}")

    # Extract weights and convert to lists
    def tensor_to_list(tensor):
        return tensor.cpu().numpy().tolist()

    # Handle different key formats (. or | as separator)
    def get_key(prefix, suffix):
        # Try with . first, then |
        key = f"{prefix}.{suffix}"
        if key not in state_dict:
            key = f"{prefix}|{suffix}"
        return key

    # Build the inference model structure
    model_data = {
        "obs_dim": state_dict[get_key("shared", "fc1|weight")].shape[1],
        "action_dim": state_dict[get_key("policy", "weight")].shape[0],
        "hidden_dim": state_dict[get_key("shared", "fc1|weight")].shape[0],
        "shared_fc1_weight": tensor_to_list(state_dict[get_key("shared", "fc1|weight")]),
        "shared_fc1_bias": tensor_to_list(state_dict[get_key("shared", "fc1|bias")]),
        "shared_fc2_weight": tensor_to_list(state_dict[get_key("shared", "fc2|weight")]),
        "shared_fc2_bias": tensor_to_list(state_dict[get_key("shared", "fc2|bias")]),
        "policy_weight": tensor_to_list(state_dict[get_key("policy", "weight")]),
        "policy_bias": tensor_to_list(state_dict[get_key("policy", "bias")]),
        "value_weight": tensor_to_list(state_dict[get_key("value", "weight")]),
        "value_bias": tensor_to_list(state_dict[get_key("value", "bias")]),
    }

    print(f"🔄 Exporting to WASM-compatible format...")

    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(model_data, f)

    print(f"✅ Model exported to: {output_path}")
    print(f"📦 File size: {len(json.dumps(model_data))} bytes")
    print()
    print("Next steps:")
    print("  1. Build WASM module: cd wasm && wasm-pack build --target web")
    print("  2. Use in web app with exported JSON weights")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_pytorch_model.py <input_model.pt> <output_model.json>")
        print()
        print("Example:")
        print("  python export_pytorch_model.py cartpole_model.pt cartpole_wasm.json")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    export_model(input_path, output_path)
