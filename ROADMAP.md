# Thrust Development Roadmap

**Last Updated**: 2025-11-06

## Vision

Thrust aims to be the **first production-ready RL library in pure Rust** with:
- 🚀 GPU-accelerated training (tch-rs + PyTorch)
- 🎮 Multi-agent population training (cooperative & competitive)
- 🌐 WASM inference for web demos
- ⚡ High-performance Rust environments

## Current Status

### ✅ Completed (Phase 1)

**Core Infrastructure**
- [x] Environment trait and CartPole implementation
- [x] EnvPool for parallel environments
- [x] RolloutBuffer with GAE
- [x] MlpPolicy with tch-rs
- [x] PPO trainer with clipping
- [x] GPU training pipeline (NVIDIA L4, ~1000 steps/sec)
- [x] Version management (local: tch 0.22, GPU: tch 0.15)
- [x] Training scripts for remote GPU
- [x] CartPole solved: 301.6 avg steps/episode (target: 195+)

**WASM Infrastructure (Partial)**
- [x] Pure Rust inference module (no PyTorch deps)
- [x] ExportedModel with JSON/bincode serialization
- [x] Forward pass implementation (softmax, linear layers)
- [x] 5 passing inference tests

**Multi-Agent Infrastructure (Partial)**
- [x] MultiAgentEnvironment trait
- [x] Population and Agent structs
- [x] 4 matchmaking strategies (Random, RoundRobin, Fitness, SelfPlay)
- [x] GameSimulator skeleton
- [x] PolicyLearner skeleton
- [x] 15 passing multi-agent tests

## Active Work

### 🚧 In Progress

1. **Multi-Agent Training (Phase 2 - 60% complete)**
   - Status: Core structs done, need communication layer
   - Next: Crossbeam channels, run loops
   - Files: `src/multi_agent/simulator.rs`, `src/multi_agent/learner.rs`

2. **WASM Visualization (Phase 1 - 40% complete)**
   - Status: Pure Rust inference ready, need weight export
   - Blocker: `MlpPolicy::export_for_inference()` tch-rs API
   - Files: `examples/export_model.rs`, needs `src/wasm.rs`

## Roadmap

### Q4 2025 - Foundation

#### Milestone 1: Multi-Agent Training Working ⏳
**Goal**: Train 4 agents simultaneously in CartPole

**Tasks**:
- [ ] Add crossbeam dependency to Cargo.toml
- [ ] Define Experience and PolicyUpdate message types
- [ ] Implement GameSimulator::run() with channels
- [ ] Implement PolicyLearner::train() with PPO loop
- [ ] Create examples/train_multi_cartpole.rs
- [ ] Benchmark: 4 agents @ 800+ steps/sec on GPU
- [ ] Document multi-agent API

**Deliverable**: Working multi-agent training demo

**Estimated Effort**: 2-3 days

---

#### Milestone 2: BucketBrigade Integration 🎯
**Goal**: Validate multi-agent training with cooperative game

**Tasks**:
- [ ] Clone/reference bucket-brigade repo
- [ ] Implement MultiAgentEnvironment for BucketBrigade
- [ ] Create examples/train_bucket_brigade.rs
- [ ] Train population of 8 agents
- [ ] Compare to Python baseline performance
- [ ] Analyze emergent strategies

**Deliverable**: Research-ready multi-agent training

**Estimated Effort**: 3-4 days

---

#### Milestone 3: WASM Web Demo 🌐
**Goal**: Interactive web demo of trained CartPole policy

**Tasks**:
- [ ] Solve tch-rs weight extraction (MlpPolicy::export_for_inference)
  - Option A: Manual tensor access via tch-rs API
  - Option B: Save to .pt, load with Python, extract to JSON
  - Option C: Hook into existing save/load mechanism
- [ ] Add wasm-bindgen to Cargo.toml with features
- [ ] Create src/wasm.rs with WASM bindings
- [ ] Implement CartPole environment in pure Rust (no tch)
- [ ] Create web/index.html with Canvas rendering
- [ ] Create web/cartpole.js for game loop
- [ ] Build with wasm-pack
- [ ] Deploy to GitHub Pages or similar

**Deliverable**: Live web demo at https://your-site.com/cartpole

**Estimated Effort**: 3-4 days

---

### Q1 2026 - Advanced Features

#### Milestone 4: Advanced Multi-Agent Features
- [ ] Off-policy learning with shared experience buffer
- [ ] Importance sampling for cross-agent learning
- [ ] Fitness-based matchmaking with ELO ratings
- [ ] Policy staleness tracking and mitigation
- [ ] Population diversity metrics
- [ ] Nash equilibrium computation

#### Milestone 5: More Environments
- [ ] Atari environments (via Rust ALE bindings)
- [ ] MuJoCo environments (via mujoco-rs)
- [ ] Custom 2D grid worlds
- [ ] Multi-agent competitive games (Pong, Soccer)

#### Milestone 6: Algorithm Expansion
- [ ] A2C (Advantage Actor-Critic)
- [ ] SAC (Soft Actor-Critic) for continuous control
- [ ] DQN for discrete actions
- [ ] Imitation learning (behavioral cloning)

#### Milestone 7: Performance Optimization
- [ ] Multi-GPU data parallelism
- [ ] Mixed precision training (FP16)
- [ ] JIT compilation for environments
- [ ] CUDA kernel optimization
- [ ] Benchmark suite

---

### Q2 2026 - Production Ready

#### Milestone 8: Library Polish
- [ ] Comprehensive documentation
- [ ] Tutorial series
- [ ] Example gallery
- [ ] Performance benchmarks vs Python
- [ ] CI/CD pipeline
- [ ] Publish to crates.io

#### Milestone 9: Research Features
- [ ] Curriculum learning
- [ ] Hierarchical RL
- [ ] Meta-learning (MAML)
- [ ] Evolution strategies
- [ ] Population-based training (PBT)

## Priority Queue (Next 3 Tasks)

1. **Complete Multi-Agent Simulator Run Loop** (1-2 days)
   - Add crossbeam channels
   - Implement experience routing
   - Test with CartPole 4-agent setup

2. **Complete Multi-Agent Learner Training Loop** (1 day)
   - Implement PPO updates
   - Add policy synchronization
   - Benchmark GPU utilization

3. **Create Multi-Agent CartPole Example** (0.5 days)
   - Simple 4-agent training script
   - Logging and metrics
   - Save/load population

## Dependencies & Blockers

### External Dependencies
- **tch-rs version compatibility**: Different versions on local vs GPU (acceptable)
- **PyTorch version**: Must match tch-rs (documented in VERSIONS.md)
- **WASM target**: Need pure Rust neural net (done)

### Known Blockers
1. **WASM weight export**: tch-rs API unclear for extracting layer weights
   - Impact: Blocks WASM demo
   - Workaround: Can use Python script to extract weights
   - Priority: Medium (not blocking multi-agent work)

2. **BucketBrigade integration**: Need access to Rust environment
   - Impact: Blocks validation of multi-agent design
   - Workaround: Use CartPole multi-agent first
   - Priority: Low (can validate with CartPole)

## Success Metrics

### Technical Metrics
- **Training Speed**: >1000 steps/sec on single GPU (✅ achieved: 1000 steps/sec)
- **Multi-Agent Throughput**: >800 steps/sec with 4 agents
- **GPU Utilization**: >90% during training
- **Memory Usage**: <4GB per agent
- **WASM Bundle Size**: <500KB gzipped
- **WASM Inference**: <1ms per forward pass

### Research Metrics
- **BucketBrigade Performance**: Match or exceed Python baseline
- **Population Diversity**: Multiple distinct strategies emerge
- **Nash Equilibrium**: Compute and visualize for 2-player games
- **Competitive Performance**: Agents reach superhuman level in at least 1 game

### Community Metrics (Future)
- GitHub stars: 100+
- Crates.io downloads: 1000+/month
- Tutorial completions: 50+
- Research papers using Thrust: 5+

## Long-Term Vision (2026+)

### Killer Features
1. **Multi-Agent Training**: Only pure-Rust library with population-based training
2. **WASM Deployment**: Train in Rust, deploy to web with zero Python
3. **Performance**: 10-100x faster environments than Python
4. **Type Safety**: Compile-time guarantees for RL workflows

### Research Applications
- Game theory and Nash equilibrium analysis
- Emergent multi-agent behavior
- Cooperative AI safety
- Competitive game playing
- Evolutionary strategies

### Production Use Cases
- Robotics simulation and training
- Game AI development
- Trading algorithms
- Resource allocation
- Network optimization

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

## References

- [MULTI_AGENT_DESIGN.md](MULTI_AGENT_DESIGN.md) - Multi-agent architecture
- [WASM_ROADMAP.md](WASM_ROADMAP.md) - WASM visualization plan
- [VERSIONS.md](VERSIONS.md) - Version compatibility matrix
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Detailed status updates
