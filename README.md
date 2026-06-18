# 🚀 Thrust

**High-performance reinforcement learning in Rust**

[![Crates.io](https://img.shields.io/crates/v/thrust-rl.svg)](https://crates.io/crates/thrust-rl)
[![Documentation](https://docs.rs/thrust-rl/badge.svg)](https://docs.rs/thrust-rl)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

> **Give your agents some thrust** 🚀

## 🎮 [**Try the Live Demo**](https://rjwalters.github.io/thrust/)

Watch trained RL agents play CartPole and Snake in real-time, running entirely in your browser via WebAssembly!

Thrust is a modern reinforcement learning library built from the ground up in Rust, designed for maximum speed, memory safety, and scalability. Inspired by [PufferLib](https://github.com/PufferAI/PufferLib), Thrust combines the raw performance of Rust with the [Burn](https://burn.dev) tensor framework's multi-backend GPU support (CUDA / ROCm / Metal / Vulkan / WebGPU / NdArray CPU) to deliver fast training without any C++ FFI dependencies.

## 🎯 Vision

Our goal is to create the fastest, safest, and most ergonomic reinforcement learning library in existence, with:
- **Live browser demos** where trained agents play games in real-time via WebAssembly
- **Production-grade performance** that scales from research to deployment
- **Best-in-class developer experience** with type safety and clear abstractions
- **Full feature parity** with leading Python RL libraries, but faster

## ✨ Features

- 🚀 **Blazing Fast**: optimized Rust pipeline with multi-backend GPU support
- 🦀 **Memory Safe**: Leverage Rust's ownership system for fearless concurrency
- 🔥 **Burn Powered**: Neural networks via [Burn](https://burn.dev) — pure-Rust autodiff, no libtorch FFI
- ⚡ **Async Vectorization**: High-performance environment parallelization with Tokio / Rayon
- 🎮 **Live Demos**: Train agents and deploy them in the browser via WebAssembly
- 🎯 **Production Ready**: Built for research and industry use cases

## 🚧 Project Status

**🎯 Alpha** - Core training infrastructure complete. Working on production features.

**Current milestone:** Phase 2 - Multi-Agent & WASM (60% complete)
**Latest:** Universal inference system, live demos, multiple environments
**Progress:** Phase 1 Complete ✅

See [ROADMAP.md](ROADMAP.md) for detailed development schedule.

## 🎯 Roadmap

### Phase 1: Foundation (Complete ✅)
- [x] Experience buffer implementation
- [x] Policy wrapper (Burn)
- [x] EnvPool vectorization
- [x] CartPole environment (301.6 avg reward achieved)
- [x] PPO training loop with GPU support
- [x] DQN training loop (replay buffer + target network, CartPole)
- [x] Checkpoint saving/loading
- [x] Snake environment (multi-agent support)
- [x] SimpleBandit environment (contextual bandits)

### Phase 2: Multi-Agent & WASM (In Progress)
- [x] Multi-agent training infrastructure (`multi_agent::PolicyLearner` is experimental — API may change)
- [x] Population-based training design
- [x] PSRO with α-rank meta-solver (N-player)
- [x] NFSP (approximate, N-player, multi-discrete)
- [x] Bucket Brigade cooperative-MARL integration
- [x] Pure Rust inference (Burn backend or hand-rolled WASM path)
- [x] Universal inference system (JSON model format)
- [ ] Complete WASM bindings
- [ ] Browser-based demos
- [ ] Multi-agent communication channels

### Phase 3: Features
- [x] SAC for continuous control (twin critics, auto-entropy, Polyak targets)
- [x] Continuous-control environments (PendulumSwingUp, ContinuousLqr)
- [ ] A2C (Advantage Actor-Critic)
- [ ] Behavioral cloning / imitation learning
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
│  │ (Burn)       │  │ (Tokio/Rayon)        │    │
│  └──────────────┘  └──────────────────────┘    │
│  ┌──────────────┐  ┌──────────────────────┐    │
│  │ Experience   │  │ Environments         │    │
│  │ Buffers      │  │ (Pure Rust)          │    │
│  └──────────────┘  └──────────────────────┘    │
│  ┌─────────────────────────────────────────┐   │
│  │ PPO / DQN / SAC + PSRO / NFSP trainers   │   │
│  │            (Burn autodiff)               │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## 🎮 Environments

- **CartPole** ✅ - Classic control benchmark (solved: 301.6 avg reward)
- **PendulumSwingUp** ✅ - Continuous-control benchmark (Gym `Pendulum-v1`; SAC's reference env)
- **ContinuousLqr** ✅ - Linear-quadratic regulator (continuous-action trait existence proof)
- **Snake** ✅ - Multi-agent grid world with torus wrapping
- **Pong** ✅ - Two-player competitive self-play
- **SimpleBandit** ✅ - Contextual multi-armed bandits
- **Matching Pennies** ✅ - Two-player and N-player zero-sum (PSRO/NFSP smoke tests)
- **Bucket Brigade** ✅ - Cooperative multi-agent coordination (Slepian-Wolf MARL adapter)
- More coming soon!

## 🧠 Algorithms

- **PPO** ✅ - Proximal Policy Optimization (on-policy, actor-critic; single- and multi-agent)
- **DQN** ✅ - Deep Q-Network (off-policy, replay buffer + target network), including **Double-DQN** target computation and optional **Polyak (soft) target updates**
- **SAC** ✅ - Soft Actor-Critic (off-policy, continuous control; twin critics, automatic entropy tuning, Polyak target updates)
- **PSRO** ✅ - Policy-Space Response Oracles with **α-rank** meta-solver (multi-agent, N-player)
- **NFSP** ✅ - Neural Fictitious Self-Play (approximate, N-player, multi-discrete actions)
- More coming soon (A2C, behavioral cloning — see [ROADMAP.md](ROADMAP.md))

## 📚 Inspiration

Thrust is inspired by:
- [PufferLib](https://github.com/PufferAI/PufferLib) - Python RL library achieving 1M+ SPS
- [Burn](https://burn.dev) - Pure-Rust deep learning framework with multi-backend GPU support
- [Border](https://github.com/laboroai/border) - Rust RL library

## 🚀 Quick Start

### Local Development (CPU)

```bash
# Build and run the bandit training example (CPU NdArray backend)
cargo run --release --example train_simple_bandit
```

The default `training` feature builds against Burn's pure-Rust NdArray
backend; no external libraries (libtorch, CUDA, etc.) are required.

### GPU Training

Burn ships several GPU backends behind feature flags. Compose them with
the default `training` feature:

```bash
# wgpu (cross-platform: Vulkan / Metal / DX12 / WebGPU)
cargo run --release --features "training,wgpu" --example train_simple_bandit

# CUDA (Linux + NVIDIA)
cargo run --release --features "training,cuda" --example train_simple_bandit
```

### Library Usage (Coming Soon)

```rust
use thrust_rl::prelude::*;

// See examples/games/bandit/train_simple_bandit.rs for the end-to-end
// rollout/loss/update loop using the Burn backend.
```

## 📊 Performance

Benchmarks are being re-run against the Burn backend; numbers will be
published once the new baseline is stable.

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
