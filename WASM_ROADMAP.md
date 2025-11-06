# WASM Visualization Roadmap

This document outlines the plan for creating WebAssembly-based visualizations of trained RL agents.

## Vision

Compile Rust inference code to WASM and run trained policies in the browser with interactive visualizations. This allows:
- **Zero-install demos**: Share trained agents via a simple web link
- **Pure Rust stack**: Same language for training and inference
- **Fast inference**: WASM performance close to native
- **Interactive visualization**: Real-time policy rendering with Canvas API

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Training (Native Rust + PyTorch)                            │
├─────────────────────────────────────────────────────────────┤
│  • GPU training with tch-rs                                 │
│  • Full PyTorch neural network                              │
│  • Saves model weights (.pt format)                         │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ Extract weights
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Weight Export (Rust)                                        │
├─────────────────────────────────────────────────────────────┤
│  • Load .pt model                                           │
│  • Extract layer weights & biases                           │
│  • Export to JSON/bincode                                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ Portable weights
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ WASM Inference (Pure Rust, no PyTorch)                     │
├─────────────────────────────────────────────────────────────┤
│  • Pure Rust NN implementation                              │
│  • Load weights from JSON                                   │
│  • Forward pass only (no training)                          │
│  • Compiles to WASM                                         │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ wasm-bindgen
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Web Visualization (JavaScript + Canvas)                     │
├─────────────────────────────────────────────────────────────┤
│  • Load WASM module                                         │
│  • Render environment (CartPole, etc.)                      │
│  • Call policy.get_action(obs)                             │
│  • Animate agent playing                                    │
└─────────────────────────────────────────────────────────────┘
```

## Current Status

### ✅ Completed

1. **Pure Rust Inference Module** (`src/inference/`)
   - `ExportedModel`: Portable model format with `serde` support
   - `LayerWeights`: Linear layer with forward pass
   - `softmax()`: Pure Rust activation
   - Tests: Layer forward pass, softmax, model prediction

2. **Weight Export/Import** (`src/inference/weights.rs`)
   - `save_json()` / `load_json()`: JSON serialization
   - `save_bincode()` / `load_bincode()`: Binary format
   - Tests: Roundtrip serialization

3. **Training Infrastructure**
   - GPU training with scripts
   - CartPole PPO training (301 steps/episode achieved)
   - Model checkpointing (`policy.save()`)

### 🚧 In Progress

1. **Weight Extraction from tch-rs**
   - Challenge: tch-rs API for accessing layer parameters differs between versions
   - Need to implement `MlpPolicy::export_for_inference()`
   - Must handle VarStore → Vec<f32> conversion

2. **Export Example** (`examples/export_model.rs`)
   - Loads trained `.pt` model
   - Calls `export_for_inference()`
   - Saves JSON weights

### 📋 TODO

1. **Complete Weight Export**
   ```rust
   // In src/policy/mlp.rs
   pub fn export_for_inference(&self) -> ExportedModel {
       // Extract weights from self.vs (VarStore)
       // For each layer (shared, policy_head, value_head):
       //   - Get weight tensor
       //   - Convert to Vec<f32>
       //   - Store in LayerWeights
       // Return ExportedModel
   }
   ```

2. **WASM Build Setup**
   ```toml
   # Add to Cargo.toml
   [lib]
   crate-type = ["cdylib", "rlib"]

   [dependencies]
   wasm-bindgen = "0.2"
   # Note: tch must be optional for WASM target
   tch = { version = "0.22", optional = true }

   [features]
   default = ["training"]
   training = ["tch"]  # Include PyTorch for training
   wasm = []            # Pure Rust for WASM
   ```

3. **WASM Bindings** (new file: `src/wasm.rs`)
   ```rust
   use wasm_bindgen::prelude::*;
   use crate::inference::ExportedModel;

   #[wasm_bindgen]
   pub struct WasmPolicy {
       model: ExportedModel,
   }

   #[wasm_bindgen]
   impl WasmPolicy {
       #[wasm_bindgen(constructor)]
       pub fn new(json_weights: &str) -> Result<WasmPolicy, JsValue> {
           let model = serde_json::from_str(json_weights)
               .map_err(|e| JsValue::from_str(&e.to_string()))?;
           Ok(WasmPolicy { model })
       }

       #[wasm_bindgen]
       pub fn get_action(&self, obs: &[f32]) -> usize {
           self.model.get_action(obs)
       }

       #[wasm_bindgen]
       pub fn get_action_probs(&self, obs: &[f32]) -> Vec<f32> {
           self.model.get_action_probs(obs)
       }
   }
   ```

4. **CartPole Environment in WASM** (`src/env/cartpole_wasm.rs`)
   ```rust
   // Pure Rust CartPole (no PyTorch)
   pub struct CartPoleWasm {
       state: [f32; 4],  // x, x_dot, theta, theta_dot
       // ... physics simulation
   }
   ```

5. **Web Visualization** (`web/index.html`, `web/cartpole.js`)
   ```javascript
   // Load WASM
   import init, { WasmPolicy } from './thrust_rl.js';

   async function main() {
       await init();

       // Load weights
       const weights = await fetch('cartpole_weights.json').then(r => r.json());
       const policy = new WasmPolicy(JSON.stringify(weights));

       // Game loop
       function step() {
           const obs = getObservation();
           const action = policy.get_action(obs);
           applyAction(action);
           render();
           requestAnimationFrame(step);
       }
       step();
   }
   ```

6. **Canvas Rendering**
   - Draw cart (rectangle)
   - Draw pole (line from cart pivot)
   - Show score/time
   - Control buttons (reset, pause)

## Build Commands

```bash
# Training (native)
cargo build --release --features training
cargo run --example train_cartpole

# Export weights
cargo run --example export_model cartpole_model.pt cartpole_weights.json

# Build WASM
wasm-pack build --target web --features wasm --no-default-features

# Serve web demo
cd web && python -m http.server 8000
# Open http://localhost:8000
```

## Example Usage

```bash
# 1. Train on GPU
ssh gpu-machine
cd thrust && git pull
./scripts/gpu-train.sh train_cartpole_long

# 2. Export weights
cargo run --example export_model cartpole_model_long.pt web/weights.json

# 3. Build WASM
wasm-pack build --target web --features wasm --no-default-features

# 4. Deploy
cp pkg/* web/
cd web && python -m http.server
# Share link: https://your-site.com/cartpole-demo
```

## Performance Targets

- **WASM inference**: < 1ms per forward pass
- **Rendering**: 60 FPS
- **Bundle size**: < 500KB (gzipped)
- **Load time**: < 2s on 3G

## Future Enhancements

1. **More Environments**
   - Atari games with pixel rendering
   - MuJoCo humanoid (3D rendering with Three.js)
   - Custom environments

2. **Interactive Features**
   - Manual control vs AI
   - Policy visualization (attention maps)
   - Reward shaping UI

3. **Multiplayer**
   - WebRTC for multi-agent scenarios
   - Leaderboards
   - Policy tournaments

## References

- [wasm-bindgen Book](https://rustwasm.github.io/wasm-bindgen/)
- [Rust and WebAssembly](https://rustwasm.github.io/docs/book/)
- [tch-rs Documentation](https://docs.rs/tch/)
- [CartPole-v1 Specification](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
