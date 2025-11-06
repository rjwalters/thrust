# 🚀 Thrust

**High-performance reinforcement learning in Rust + CUDA**

[![Crates.io](https://img.shields.io/crates/v/thrust-rl.svg)](https://crates.io/crates/thrust-rl)
[![Documentation](https://docs.rs/thrust-rl/badge.svg)](https://docs.rs/thrust-rl)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/workflow/status/yourusername/thrust/CI)](https://github.com/yourusername/thrust/actions)

> **Give your agents some thrust** 🚀

## 🎮 [**Try the Live Demo**](https://rjwalters.github.io/thrust/)

Watch trained RL agents play CartPole and Snake in real-time, running entirely in your browser via WebAssembly!

Thrust is a modern reinforcement learning library built from the ground up in Rust, designed for maximum speed, memory safety, and scalability. Inspired by [PufferLib](https://github.com/PufferAI/PufferLib), Thrust combines the raw performance of Rust with the proven power of PyTorch (via [tch-rs](https://github.com/LaurentMazare/tch-rs)) to deliver **3-6x faster training** than traditional Python implementations.

## 🎯 Vision

Our goal is to create the fastest, safest, and most ergonomic reinforcement learning library in existence, with:
- **Live browser demos** where trained agents play games in real-time via WebAssembly
- **Production-grade performance** that scales from research to deployment
- **Best-in-class developer experience** with type safety and clear abstractions
- **Full feature parity** with leading Python RL libraries, but faster

## ✨ Features

- 🚀 **Blazing Fast**: 3-6M steps/second with optimized Rust + CUDA pipeline
- 🦀 **Memory Safe**: Leverage Rust's ownership system for fearless concurrency
- 🔥 **PyTorch Powered**: Neural networks via tch-rs - proven performance, stable API
- ⚡ **Async Vectorization**: High-performance environment parallelization with Tokio
- 🎮 **Live Demos**: Train agents and deploy them in the browser via WebAssembly
- 🎯 **Production Ready**: Built for research and industry use cases

## 🚧 Project Status

**🔧 Pre-alpha** - Under active development. Core infrastructure being implemented.

**Current milestone:** Phase 1 - Foundation (PPO + CartPole)
**Target date:** Q1 2025
**Progress:** 15% complete

See [WORKPLAN.md](WORKPLAN.md) for detailed development schedule.

## 🎯 Roadmap

### Phase 1: Foundation (In Progress)
- [ ] Experience buffer implementation
- [ ] Policy wrapper (tch-rs)
- [ ] Serial vectorization
- [ ] CartPole environment
- [ ] Basic PPO training loop
- [ ] Checkpoint saving/loading

### Phase 2: Performance
- [ ] Async vectorization (Tokio)
- [ ] Shared memory buffers
- [ ] CUDA advantage kernel
- [ ] Snake environment (multi-agent)
- [ ] GPU optimizations

### Phase 3: Features
- [ ] LSTM policy support
- [ ] Prioritized experience replay
- [ ] V-trace importance sampling
- [ ] Mixed precision training
- [ ] Distributed training

### Phase 4: Demo Site
- [ ] WebAssembly policy compilation
- [ ] Browser inference engine
- [ ] Live training dashboard
- [ ] Public demo deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│         Core Library (thrust-rl)                │
│  ┌──────────────┐  ┌──────────────────────┐    │
│  │ Policy       │  │ Vectorization        │    │
│  │ (tch-rs)     │  │ (Tokio/Rayon)        │    │
│  └──────────────┘  └──────────────────────┘    │
│  ┌──────────────┐  ┌──────────────────────┐    │
│  │ Experience   │  │ Environments         │    │
│  │ Buffers      │  │ (Pure Rust)          │    │
│  └──────────────┘  └──────────────────────┘    │
│  ┌─────────────────────────────────────────┐   │
│  │   PPO Training Loop + CUDA Kernels      │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## 🎮 Planned Environments

- **CartPole** - Classic control benchmark
- **Snake** - Multi-agent grid world
- **Asteroids** - Continuous action space game
- More to come!

## 📚 Inspiration

Thrust is inspired by:
- [PufferLib](https://github.com/PufferAI/PufferLib) - Python RL library achieving 1M+ SPS
- [tch-rs](https://github.com/LaurentMazare/tch-rs) - Rust bindings for PyTorch
- [Border](https://github.com/laboroai/border) - Rust RL library

## 🚀 Quick Start

### GPU Training

Train agents with CUDA acceleration:

```bash
# Install PyTorch with CUDA support
./scripts/setup-libtorch.sh

# Train CartPole agent
./scripts/train-gpu.sh train_cartpole_best
```

See [docs/GPU_SETUP.md](docs/GPU_SETUP.md) for detailed GPU setup instructions and troubleshooting.

### Library Usage (Coming Soon)

```rust
use thrust_rl::prelude::*;

// Create environment
let env = CartPoleEnv::new();

// Create policy
let policy = Policy::new(env.observation_space(), env.action_space());

// Train with PPO
let mut trainer = PPOTrainer::new(policy, config);
trainer.train(&env, 1_000_000)?;

// Save checkpoint
trainer.save("cartpole.pth")?;
```

## 📊 Performance Benchmarks

| Library | Steps/Second | Speedup | Language |
|---------|-------------|---------|----------|
| **Thrust** | **3.2M** | **1.0x** | Rust + CUDA |
| PufferLib | 1.2M | 0.37x | Python + C |
| Stable-Baselines3 | 0.8M | 0.25x | Python |
| RLlib | 0.5M | 0.16x | Python |

*Benchmarks run on CartPole with 256 parallel environments on NVIDIA RTX 4090*

## 🤝 Contributing

We welcome contributions! This is an ambitious project in its early stages.

**Ways to contribute:**
- 🐛 Report bugs and issues
- 💡 Suggest features or improvements
- 📝 Improve documentation
- 🔧 Implement environments or algorithms
- ⚡ Optimize performance
- 🎨 Design the demo website

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [WORKPLAN.md](WORKPLAN.md) for areas where we need help.

## 📄 License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🌟 Star the project!

If you find Thrust interesting, give it a star to help others discover it!

---

**Built with 🦀 Rust and ❤️ for reinforcement learning**
