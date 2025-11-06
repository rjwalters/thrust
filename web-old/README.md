# Thrust RL - Web Visualizer

This directory contains the WebAssembly-based visualizations for Thrust RL agents.

## Overview

The web visualizer compiles our Rust environments to WASM, allowing trained agents to run directly in the browser with no Python dependencies. This provides:

- **Single source of truth**: Same Rust code for training and visualization
- **Zero-install demos**: Share trained agents via a simple web link
- **Fast inference**: WASM performance close to native Rust
- **Interactive visualization**: Real-time rendering with Canvas API

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Training (Native Rust + PyTorch)                        │
│  • GPU training with tch-rs                             │
│  • CartPole, Snake environments                         │
│  • PPO algorithm                                        │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Export weights
                          ▼
┌─────────────────────────────────────────────────────────┐
│ WASM Module (Pure Rust, no PyTorch)                    │
│  • CartPole & Snake environments compiled to WASM       │
│  • Pure Rust neural network inference                  │
│  • wasm-bindgen JavaScript bindings                    │
└─────────────────────────────────────────────────────────┘
                          │
                          │ JavaScript calls
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Web UI (HTML + Canvas)                                  │
│  • Render environments                                  │
│  • Display statistics                                   │
│  • Control playback                                     │
└─────────────────────────────────────────────────────────┘
```

## Files

- `index.html` - Landing page with links to all demos
- `cartpole.html` - CartPole demo page
- `cartpole.js` - CartPole visualization and controls (to be implemented)
- `snake.html` - Multi-agent Snake demo page (to be implemented)
- `snake.js` - Snake visualization (to be implemented)

## Building for WASM

### Prerequisites

Install `wasm-pack`:
```bash
cargo install wasm-pack
```

### Build

```bash
# From the project root
wasm-pack build --target web --features wasm --no-default-features
```

This creates a `pkg/` directory with:
- `thrust_rl_bg.wasm` - Compiled WASM module
- `thrust_rl.js` - JavaScript bindings
- `thrust_rl.d.ts` - TypeScript definitions

### Copy to Web Directory

```bash
cp pkg/thrust_rl_bg.wasm web/
cp pkg/thrust_rl.js web/
```

## Running Locally

Serve the web directory with any HTTP server:

```bash
# Python
cd web && python3 -m http.server 8000

# Or with Node.js
cd web && npx http-server -p 8000
```

Then visit: http://localhost:8000

## Deploying to GitHub Pages

The web directory is ready to be deployed to GitHub Pages:

1. Push to the repository
2. Enable GitHub Pages in Settings → Pages
3. Select the `web/` directory as the source
4. Your demos will be live at `https://yourusername.github.io/thrust/`

## WASM API

### CartPole

```javascript
import init, { WasmCartPole } from './thrust_rl.js';

await init();

const env = new WasmCartPole();

// Reset environment
const obs = env.reset();  // Returns [x, x_dot, theta, theta_dot]

// Take step
const result = env.step(action);  // action: 0 (left) or 1 (right)
// Returns: [obs0, obs1, obs2, obs3, reward, terminated, truncated]

// Get state for rendering
const state = env.get_state();  // [x, x_dot, theta, theta_dot]

// Get statistics
const episode = env.get_episode();
const steps = env.get_steps();
const best = env.get_best_score();
```

### Snake

```javascript
import init, { WasmSnake } from './thrust_rl.js';

await init();

const env = new WasmSnake(20, 20, 4);  // width, height, num_agents

// Reset
env.reset();

// Step with actions for all agents
env.step([0, 1, 2, 3]);  // 0=Up, 1=Down, 2=Left, 3=Right

// Get observations
const obs = env.get_observation(0);  // Agent 0's observation

// Get rendering data
const snakePositions = env.get_snake_positions();
const foodPositions = env.get_food_positions();
const activeAgents = env.active_agents();
```

## Next Steps

1. **Implement CartPole JS visualization** - Canvas rendering with physics
2. **Implement Snake JS visualization** - Grid rendering with multiple agents
3. **Add trained model weights** - Export and include .json weight files
4. **Add WASM inference** - Implement pure Rust neural network forward pass
5. **Add policy controls** - Allow users to load different trained agents

## Performance

Current targets:
- WASM inference: < 1ms per forward pass
- Rendering: 60 FPS
- Bundle size: < 500KB (gzipped)
- Load time: < 2s on 3G

## References

- [wasm-bindgen Book](https://rustwasm.github.io/wasm-bindgen/)
- [Rust and WebAssembly](https://rustwasm.github.io/docs/book/)
- [CartPole Specification](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
