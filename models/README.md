# Trained Policy Models

This directory contains all trained reinforcement learning policies organized by game/environment.

## Directory Structure

```
models/
└── policies/
    ├── cartpole/      # CartPole balancing policies
    ├── snake/         # Snake game policies
    └── bandit/        # Multi-armed bandit policies
```

## Current Trained Policies

### CartPole

Located in `policies/cartpole/`:

- **cartpole_model_best.pt** - Best performing CartPole agent (467 steps/episode average)
  - Training: examples/games/cartpole/train_cartpole_best.rs
  - Performance: Consistently achieves 400+ steps

- **cartpole_model_long.pt** - Long-training session CartPole agent
  - Training: examples/games/cartpole/train_cartpole_long.rs

### Snake

Located in `policies/snake/`:

- **snake_policy_init.pt** - Initial Snake policy checkpoint
  - Grid size: 10x10
  - Training: examples/games/snake/train_snake_multi.rs

- **snake_policy.epoch200.pt** - Snake policy after 200 epochs
  - Grid size: 10x10
  - Training: examples/games/snake/train_snake_multi.rs

- **snake_single_final.safetensors** - Final single-agent Snake policy (SafeTensors format)
  - Grid size: 10x10
  - Training: examples/games/snake/train_snake_multi_v2.rs

### Bandit

Located in `policies/bandit/`:

- Currently no saved policies (bandit policies are lightweight and quick to train)

## Training Your Own Policies

Each game has dedicated training examples in `examples/games/`:

```bash
# CartPole
cargo run --example train_cartpole_best --release --features training

# Snake
cargo run --example train_snake_multi_v2 --release --features training

# Simple Bandit
cargo run --example train_simple_bandit --release --features training
```

## File Formats

- `.pt` - PyTorch format (compatible with tch-rs)
- `.safetensors` - SafeTensors format (more secure, portable)

## Policy Naming Convention

Recommended naming pattern: `{game}_{variant}_{metric}.{ext}`

Examples:
- `cartpole_ppo_467steps.pt`
- `snake_10x10_epoch500.safetensors`
- `snake_large_final.pt`

## Loading Policies

```rust
use thrust::policy::ActorCriticPolicy;
use tch::{nn, Device};

// Load a CartPole policy
let vs = nn::VarStore::new(Device::Cpu);
let mut policy = ActorCriticPolicy::new(&vs.root(), 4, 2);
vs.load("models/policies/cartpole/cartpole_model_best.pt")?;

// Load a Snake policy
let vs = nn::VarStore::new(Device::Cpu);
let mut policy = ActorCriticPolicy::new(&vs.root(), 84, 4);
vs.load("models/policies/snake/snake_single_final.safetensors")?;
```

## Web Deployment

Policies used in the web interface are loaded from this directory. See `web/src/pages/` for game-specific implementations.

## Version Control

- ✅ Committed: Lightweight policies (< 100KB)
- ⚠️ Git LFS: Medium policies (100KB - 10MB)
- ❌ Not tracked: Large experimental policies (> 10MB)

Add large files to `.gitignore` or use Git LFS for important large models.
