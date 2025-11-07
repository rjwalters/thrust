# Universal Inference System

**Status**: Complete
**File**: `src/policy/universal_inference.rs` (555 lines)
**Purpose**: Flexible, JSON-serializable neural network inference without PyTorch dependency

## Overview

The Universal Inference System is a pure Rust neural network inference engine that can execute any model architecture defined in JSON format. This enables:

1. **WASM Deployment**: Run trained models in the browser without PyTorch
2. **Flexible Architectures**: Support any network design via JSON configuration
3. **Cross-Platform**: Same model format works on native, GPU, and WASM targets
4. **Version Tracking**: Self-describing format with version metadata

## Architecture

```
Training (with PyTorch)          Production (Pure Rust)
┌──────────────────┐            ┌──────────────────┐
│  tch-rs Policy   │            │ UniversalModel   │
│  ─────────────   │            │  ─────────────   │
│  • MlpPolicy     │  export    │  • Pure Rust     │
│  • SnakeCNN      │  ──────>   │  • No PyTorch    │
│  • Custom nets   │   JSON     │  • Fast forward  │
│                  │            │  • WASM-ready    │
└──────────────────┘            └──────────────────┘
```

## Supported Layers

### Linear (Dense/Fully-Connected)
```json
{
  "type": "linear",
  "name": "fc1",
  "in_features": 4,
  "out_features": 64,
  "weight": [[...]],  // Shape: [out_features, in_features]
  "bias": [...]        // Shape: [out_features]
}
```

### Conv2d (2D Convolution)
```json
{
  "type": "conv2d",
  "name": "conv1",
  "in_channels": 5,
  "out_channels": 32,
  "kernel_size": 3,
  "padding": 1,
  "stride": 1,
  "weight": [[[[...]]]],  // Shape: [out_channels, in_channels, kernel_size, kernel_size]
  "bias": [...]            // Shape: [out_channels]
}
```

### Activations
```json
{
  "type": "activation",
  "name": "relu1",
  "activation": "relu"  // Options: relu, tanh, sigmoid, identity, gelu, swish
}
```

### Reshape
```json
{
  "type": "reshape",
  "name": "reshape1",
  "target_shape": [32, 7, 7]  // Target shape (excluding batch dimension)
}
```

### Flatten
```json
{
  "type": "flatten",
  "name": "flatten1"
}
```

### Residual Connection
```json
{
  "type": "residual",
  "name": "res1",
  "inner_layers": [...]  // Layers to apply before adding to input
}
```

## Model Format

### Complete Model Structure

```json
{
  "version": "1.0",
  "input": {
    "shape": [4],
    "dtype": "float32"
  },
  "output": {
    "num_actions": 2,
    "has_value_head": true
  },
  "metadata": {
    "environment": "CartPole-v1",
    "training_steps": 100000,
    "avg_reward": 301.6,
    "created_at": "2025-11-07T...",
    "framework_version": "thrust-0.1.0"
  },
  "shared_layers": [
    {"type": "linear", "name": "shared1", ...},
    {"type": "activation", "name": "relu1", ...}
  ],
  "policy_head": [
    {"type": "linear", "name": "policy_out", ...}
  ],
  "value_head": [
    {"type": "linear", "name": "value_out", ...}
  ]
}
```

### Input Specification

```rust
pub struct InputSpec {
    pub shape: Vec<usize>,    // Input dimensions (e.g., [4] for CartPole, [5, 7, 7] for Snake)
    pub dtype: String,         // "float32" or "float64"
}
```

### Output Specification

```rust
pub struct OutputSpec {
    pub num_actions: usize,        // Number of discrete actions
    pub has_value_head: bool,      // Whether model includes value prediction
}
```

### Metadata (Optional)

```rust
pub struct ModelMetadata {
    pub environment: String,        // Which environment this was trained on
    pub training_steps: usize,      // Total training steps
    pub avg_reward: f32,            // Average reward achieved
    pub created_at: String,         // ISO 8601 timestamp
    pub framework_version: String,  // Version of Thrust used
}
```

## Usage Examples

### 1. Training: Export from PyTorch Model

```rust
use thrust_rl::policy::mlp::MlpPolicy;
use thrust_rl::policy::universal_inference::UniversalModel;

// Train your model
let policy = MlpPolicy::new(4, 2, &[64, 64], &device)?;
// ... training ...

// Export to universal format
let universal_model = UniversalModel::from_mlp(&policy, "CartPole-v1", 100000, 301.6)?;

// Save as JSON
let json = serde_json::to_string_pretty(&universal_model)?;
std::fs::write("cartpole_policy.json", json)?;
```

### 2. Production: Load and Run Inference

```rust
use thrust_rl::policy::universal_inference::UniversalModel;

// Load model from JSON
let json = std::fs::read_to_string("cartpole_policy.json")?;
let model: UniversalModel = serde_json::from_str(&json)?;

// Run inference
let observation = vec![0.01, 0.02, 0.03, 0.04];
let action = model.get_action(&observation);
println!("Chosen action: {}", action);

// Get action probabilities
let probs = model.get_action_probs(&observation);
println!("Action probabilities: {:?}", probs);

// Get value prediction (if model has value head)
if let Some(value) = model.get_value(&observation) {
    println!("Predicted value: {}", value);
}
```

### 3. WASM: Browser Inference

```rust
use wasm_bindgen::prelude::*;
use thrust_rl::policy::universal_inference::UniversalModel;

#[wasm_bindgen]
pub struct WasmPolicy {
    model: UniversalModel,
}

#[wasm_bindgen]
impl WasmPolicy {
    #[wasm_bindgen(constructor)]
    pub fn new(json: &str) -> Result<WasmPolicy, JsValue> {
        let model = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmPolicy { model })
    }

    #[wasm_bindgen]
    pub fn get_action(&self, obs: &[f32]) -> usize {
        self.model.get_action(obs)
    }
}
```

JavaScript usage:
```javascript
import init, { WasmPolicy } from './thrust_wasm.js';

await init();
const response = await fetch('cartpole_policy.json');
const policyJson = await response.json();
const policy = new WasmPolicy(JSON.stringify(policyJson));

// Use in game loop
function gameStep(observation) {
    const action = policy.get_action(observation);
    applyAction(action);
}
```

## Implementation Details

### Tensor Representation

The inference system uses simplified tensor representations:

```rust
pub enum Tensor {
    Tensor1D(Vec<f32>),                      // Shape: [n]
    Tensor3D(Vec<Vec<Vec<f32>>>),            // Shape: [c, h, w]
}
```

This is sufficient for:
- **1D**: Fully-connected layer inputs/outputs, value predictions
- **3D**: Convolutional layer inputs/outputs (channels, height, width)

### Forward Pass Flow

```
Input (1D or 3D)
    ↓
Shared Layers (feature extraction)
    ↓
    ├─→ Policy Head → Softmax → Action probabilities
    └─→ Value Head → Scalar value (optional)
```

### Activation Functions

All activations operate element-wise:

```rust
fn relu(x: f32) -> f32 { x.max(0.0) }
fn tanh(x: f32) -> f32 { x.tanh() }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn gelu(x: f32) -> f32 { 0.5 * x * (1.0 + (x * 0.7978845608).tanh()) }
fn swish(x: f32) -> f32 { x * sigmoid(x) }
```

### Convolution Implementation

Pure Rust 2D convolution with padding and stride support:

```rust
fn conv2d(
    input: &[Vec<Vec<f32>>],      // [in_channels, height, width]
    kernel: &[Vec<Vec<f32>>],     // [kernel_h, kernel_w]
    padding: usize,
    stride: usize
) -> Vec<Vec<f32>>                // [out_height, out_width]
```

## Performance Characteristics

### Memory Usage
- **Model Size**: Proportional to weight count
  - CartPole (4→64→64→2): ~8KB JSON
  - Snake CNN (5→32→64→4): ~250KB JSON
- **Runtime Memory**: Single inference requires ~1MB stack for intermediate activations

### Speed
- **Native**: <0.1ms per forward pass (CartPole MLP)
- **WASM**: <1ms per forward pass (CartPole MLP)
- **Bottleneck**: Convolution layers (optimized with loop unrolling)

### Limitations
- **No Batch Processing**: Single input per forward pass (sufficient for real-time inference)
- **No GPU Acceleration**: Pure CPU inference (acceptable for production deployment)
- **Fixed Architecture**: Model architecture defined at export, not modifiable at runtime

## Migration from Old Inference

The universal inference system replaces the old hardcoded implementations:

### Before (Hardcoded)
```rust
// src/policy/inference.rs - Only 2-layer MLP
pub struct InferenceModel {
    shared1_weight: Vec<Vec<f32>>,
    shared1_bias: Vec<f32>,
    shared2_weight: Vec<Vec<f32>>,
    shared2_bias: Vec<f32>,
    // ... hardcoded structure
}
```

### After (Universal)
```rust
// src/policy/universal_inference.rs - Any architecture
pub struct UniversalModel {
    shared_layers: Vec<Layer>,     // Any number of layers
    policy_head: Vec<Layer>,         // Flexible head
    value_head: Option<Vec<Layer>>,  // Optional value prediction
}
```

**Backward compatibility was removed** to keep the codebase simple. All new models should use the universal format.

## Export Workflow

### Step 1: Train with PyTorch
```bash
cargo run --example train_cartpole --release
# Creates: models/cartpole_model.pt
```

### Step 2: Export to Universal Format
```bash
cargo run --example export_model --release cartpole_model.pt cartpole_model.json
```

The export tool:
1. Loads the `.pt` checkpoint
2. Extracts layer weights from tch-rs VarStore
3. Converts to UniversalModel format
4. Adds metadata (env name, training steps, etc.)
5. Serializes to JSON

### Step 3: Use in Production
```bash
# Native
cargo run --example run_policy cartpole_model.json

# WASM
wasm-pack build --target web
# Load JSON in browser
```

## Testing

The universal inference system includes comprehensive tests:

```rust
#[test]
fn test_linear_forward() { ... }      // Layer operations

#[test]
fn test_conv2d_forward() { ... }      // Convolution correctness

#[test]
fn test_activation_functions() { ... } // All activations

#[test]
fn test_model_inference() { ... }     // End-to-end inference

#[test]
fn test_json_serialization() { ... }  // Save/load roundtrip
```

Run tests:
```bash
cargo test universal_inference
```

## Future Enhancements

Potential improvements (not yet implemented):

1. **Batch Processing**: Process multiple inputs in parallel
2. **SIMD Optimization**: Vectorized operations for faster inference
3. **Quantization**: INT8 weights for smaller models
4. **Layer Fusion**: Combine consecutive operations for efficiency
5. **Dynamic Shapes**: Support variable input sizes
6. **More Layer Types**: BatchNorm, Dropout (inference mode), MaxPool, etc.

## Related Files

- **Implementation**: `src/policy/universal_inference.rs`
- **Module Export**: `src/policy/mod.rs`
- **WASM Bindings**: `src/wasm.rs` (uses UniversalModel)
- **Export Tool**: `examples/export_model.rs`
- **Training Examples**: `examples/train_cartpole.rs`, `examples/train_snake.rs`

## References

- JSON Schema: [JSON specification draft-07](https://json-schema.org/)
- Serde: [Rust serialization framework](https://serde.rs/)
- WASM Bindgen: [Rust ↔ JavaScript bindings](https://rustwasm.github.io/wasm-bindgen/)

---

**Last Updated**: 2025-11-07
**File Location**: `src/policy/universal_inference.rs:1-555`
