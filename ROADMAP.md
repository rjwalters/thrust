# Thrust Development Roadmap

**Last Updated**: 2026-06-17

## Vision

Thrust aims to be the **first production-ready RL library in pure Rust** with:
- 🚀 GPU-accelerated training via [Burn](https://burn.dev) (CUDA / wgpu) — no libtorch FFI
- 🎮 Multi-agent population training (cooperative & competitive)
- 🌐 WASM inference for web demos
- ⚡ High-performance Rust environments

> **Backend note.** Thrust runs entirely on the [Burn](https://burn.dev) tensor
> framework (pure Rust, autodiff). The pre-v0.1.0 `tch`/libtorch path has been
> fully removed (see `docs/BURN_BACKENDS.md`). The default build uses Burn's
> NdArray CPU backend with **no external system libraries**; GPU is opt-in via
> the `wgpu` (cross-platform) or `cuda` (Linux/NVIDIA) features.

## Current Status

### ✅ Completed Foundation

**Core Infrastructure**
- [x] `Environment` trait (discrete + continuous actions) with CartPole, Pendulum, LQR, Snake, Pong, bandit, matching-pennies
- [x] EnvPool for parallel environments
- [x] RolloutBuffer with GAE (on-policy) + ReplayBuffer / ContinuousReplayBuffer (off-policy)
- [x] Burn-backed policy networks (`MlpBurnPolicy`, multi-discrete, Q-networks, SAC actor/critics)
- [x] Seeded, bit-exact-reproducible network init (`seeded_init`)
- [x] PPO trainer with clipping (single- and multi-agent)
- [x] Burn NdArray CPU backend by default; `wgpu` / `cuda` GPU backends opt-in
- [x] CartPole solved: 301.6 avg steps/episode (target: 195+)

**Algorithms** (see the Algorithms section below)
- [x] PPO, DQN (Double-DQN + Polyak), SAC (continuous control)
- [x] PSRO with α-rank meta-solver (N-player), NFSP (N-player, multi-discrete)

**WASM Infrastructure (Partial)**
- [x] Pure Rust inference module (no external deps)
- [x] ExportedModel with JSON serialization
- [x] Forward pass implementation (softmax, linear layers)

**Multi-Agent Infrastructure**
- [x] MultiAgentEnvironment trait + `JointMultiAgentTrainer` (per-agent observations)
- [x] Population and Agent structs, 4 matchmaking strategies
- [x] Bucket Brigade cooperative-MARL adapter + PSRO/NFSP integration

## Active Work

### 🚧 In Progress

1. **Research validation (operator-gated)** — long-budget PSRO/NFSP training +
   evaluation on the Bucket Brigade no-convergence cells (issue #134). Needs
   GPU host time on alc-2; the *infrastructure* is shipped, the multi-hour
   *runs* are pending an operator.

2. **WASM Visualization** — pure Rust inference is ready; remaining work is the
   `wasm-bindgen` bindings and a browser demo. Files: needs `src/wasm.rs`.

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
- [x] Weight export to JSON (`ExportedModel` / universal inference format) — the
      pre-Burn `tch` weight-extraction blocker is gone; Burn modules serialize directly
- [ ] Add wasm-bindgen to Cargo.toml with features
- [ ] Create src/wasm.rs with WASM bindings
- [ ] Implement CartPole environment in pure Rust (already env-trait native)
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
- [x] Multi-agent competitive games (Pong self-play)
- [x] Continuous-control envs (PendulumSwingUp, ContinuousLqr)
- [ ] Atari environments (via Rust ALE bindings)
- [ ] MuJoCo environments (via mujoco-rs) — now unblocked by SAC
- [ ] Custom 2D grid worlds

#### Milestone 6: Algorithm Expansion
- See [`docs/RL_TOYBOX_SURVEY.md`](docs/RL_TOYBOX_SURVEY.md) for ranked port recommendations (replay buffer, DQN, SAC, centralized critic, MCTS) and prioritized follow-up issues.
- [x] DQN for discrete actions (Double-DQN + Polyak target updates)
- [x] SAC (Soft Actor-Critic) for continuous control (twin critics, auto-entropy tuning)
- [x] PSRO with α-rank meta-solver (N-player)
- [x] NFSP (approximate, N-player, multi-discrete)
- [ ] A2C (Advantage Actor-Critic)
- [x] Imitation learning (behavioral cloning)

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

## Priority Queue (Next Tasks)

1. **Research validation on alc-2** (operator-gated, issue #134)
   - Long-budget PSRO/NFSP training on the Bucket Brigade no-convergence cells
   - Answers the open research question (do PSRO+α-rank / NFSP beat PPO there?)
   - Infrastructure is shipped; needs GPU host time + manual artifact promotion

2. **A2C trainer** (Milestone 6)
   - Reuses the existing PPO rollout/GAE path; smaller surface than SAC
   - Architect → builder decomposition

3. **Capitalize on SAC** (Milestone 5)
   - `train_sac` example + a second continuous env (LunarLanderContinuous, or
     MuJoCo via `mujoco-rs`) to take SAC from "implemented" to "benchmarked"

4. **WASM demo** — `wasm-bindgen` bindings + browser inference (pure-Rust
   inference path is already done; no weight-export blocker remains post-Burn)

## Dependencies & Blockers

### External Dependencies
- **Burn 0.21**: the sole tensor backend; NdArray (CPU) by default, `wgpu` / `cuda` opt-in
- **No libtorch / PyTorch**: the pre-v0.1.0 `tch` path was removed (see `docs/BURN_BACKENDS.md`)
- **WASM target**: pure Rust neural net (done)

### Known Blockers
1. **Research validation (#134)**: needs operator GPU time on alc-2 (multi-hour
   runs, manual artifact promotion). Labeled `loom:blocked` — not automatable.
   - Impact: the headline research question stays unanswered until run
   - Priority: High value, operator-gated

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
