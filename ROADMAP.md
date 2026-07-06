# Thrust Development Roadmap

**Last Updated**: 2026-07-06 (post-v0.3.0)

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
- [x] `Environment` trait (discrete + continuous actions) with CartPole, Pendulum, MountainCarContinuous, LQR, Snake, Pong, bandit, matching-pennies
- [x] EnvPool for parallel environments
- [x] RolloutBuffer with GAE (on-policy) + ReplayBuffer / ContinuousReplayBuffer (off-policy)
- [x] Burn-backed policy networks (`MlpBurnPolicy`, multi-discrete, Q-networks, SAC actor/critics)
- [x] Seeded, bit-exact-reproducible network init (`seeded_init`)
- [x] PPO trainer with clipping (single- and multi-agent)
- [x] Burn NdArray CPU backend by default; `wgpu` / `cuda` GPU backends opt-in
- [x] CartPole solved: 301.6 avg steps/episode (target: 195+)
- [x] Benchmark harness (`benches/trainer_throughput.rs`, criterion) — per-update + full-loop steps/sec for PPO/A2C/DQN/SAC; opt-in `CURVE_CSV` learning curves on every example

**Algorithms** (see the Algorithms section below)
- [x] PPO, A2C, DQN (Double-DQN + Polyak), SAC (continuous control)
- [x] Behavioral cloning (supervised imitation from demonstrations)
- [x] PSRO with α-rank meta-solver (N-player), NFSP (N-player, multi-discrete)
- [x] Recurrent (LSTM) PPO — `LstmBurnPolicy` + `RecurrentRolloutBuffer` +
      `RecurrentPPOTrainer`, validated on a FlickeringCartPole POMDP (v0.3.0)
- [x] V-trace (IMPALA) off-policy correction + single-host async actor-learner
      (2.1× synchronous throughput) (v0.3.0)
- [x] Multi-agent communication Phases 1–2: `CommunicatingEnvironment` +
      SignalingGame + joint-trainer comms hook (v0.3.0)

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
   *runs* are pending an operator. This is the single highest-value open item.

2. **Multi-agent communication Phase 3** — differentiable (Gumbel-softmax)
   comms channel; Phases 1–2 shipped in v0.3.0 (issue #276, deferred until
   the discrete-channel results motivate it).

3. **Multi-host distributed training** — design note shipped
   (`docs/DISTRIBUTED_TRAINING_DESIGN.md`); benchmarks and coordination
   blocked on workloads that saturate a single host (issue #281).

4. **Mixed-precision (FP16/BF16) training** — feasibility spike done
   (`docs/FP16_FEASIBILITY.md`); blocked upstream on Burn/Metal bf16 matmul
   kernels (issues #270/#272).

### ✅ Recently completed
- **v0.3.0 (2026-07-06)** — recurrent (LSTM) PPO stack validated on a
  FlickeringCartPole POMDP, V-trace, async actor-learner, comms Phases 1–2,
  k* phase-diagram research artifact, `max_grad_norm` behavior fix.
- **v0.2.0 (2026-06-30)** — PSRO/NFSP research release; the Bucket Brigade
  no-convergence question answered (negative, root-caused) in
  `docs/research/`.
- **WASM demo shipped** — live at
  [rjwalters.github.io/thrust](https://rjwalters.github.io/thrust/) with
  CartPole, Snake, Pong, bandit, and a training dashboard.

## Roadmap

### Q4 2025 - Foundation

#### Milestone 1: Multi-Agent Training Working ✅ (superseded design)
**Goal**: Train multiple agents simultaneously

Shipped via a different design than originally sketched here: per-agent
PPO learners over a shared env (`train_snake_multi_v2`), the
`JointMultiAgentTrainer`, and — in v0.3.0 — a crossbeam-channel async
actor-learner (`train_cartpole_async`). The original
GameSimulator/PolicyLearner message-passing sketch was never built and is
retained only in `docs/MULTI_AGENT_DESIGN.md` for the record.

---

#### Milestone 2: BucketBrigade Integration 🎯 (largely complete)
**Goal**: Validate multi-agent training with cooperative game

**Tasks**:
- [x] Reference bucket-brigade repo (pure-Rust `bucket-brigade-core` dep, opt-in feature)
- [x] Implement the BucketBrigade env adapter (`JointEnv` + per-agent observations)
- [x] Training examples (`examples/games/bucket_brigade/train_psro.rs`, `train_nfsp.rs`)
- [x] PSRO (α-rank) + NFSP integration with `gap_closed` metric
- [ ] Long-budget training + Python-baseline comparison + strategy analysis
      → **operator-gated, tracked in #134** (needs alc-2 GPU time)

**Deliverable**: Research-ready multi-agent training — *infrastructure shipped;
the validation run is the remaining piece (#134).*

---

#### Milestone 3: WASM Web Demo ✅
**Goal**: Interactive web demo of trained CartPole policy

**Tasks**: all done ✅ — `src/wasm.rs` bindings, the React app under `web/`,
wasm-pack build in CI, and GitHub Pages deployment.

**Deliverable**: live at
[rjwalters.github.io/thrust](https://rjwalters.github.io/thrust/) (CartPole,
Snake, Pong, bandit, training dashboard).

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
- [x] Continuous-control envs (PendulumSwingUp, ContinuousLqr, MountainCarContinuous)
- [ ] Atari environments (via Rust ALE bindings)
- [ ] MuJoCo environments (via mujoco-rs) — now unblocked by SAC
- [ ] Custom 2D grid worlds

#### Milestone 6: Algorithm Expansion ✅ COMPLETE
- See [`docs/RL_TOYBOX_SURVEY.md`](docs/RL_TOYBOX_SURVEY.md) for ranked port recommendations (replay buffer, DQN, SAC, centralized critic, MCTS) and prioritized follow-up issues.
- [x] DQN for discrete actions (Double-DQN + Polyak target updates)
- [x] SAC (Soft Actor-Critic) for continuous control (twin critics, auto-entropy tuning)
- [x] A2C (Advantage Actor-Critic)
- [x] Imitation learning (behavioral cloning)
- [x] PSRO with α-rank meta-solver (N-player)
- [x] NFSP (approximate, N-player, multi-discrete)

#### Milestone 7: Performance Optimization
- [x] Benchmark suite (criterion throughput harness for PPO/A2C/DQN/SAC; see `benches/trainer_throughput.rs`)
- [x] Async actor-learner (single-host; 2.1× synchronous throughput, v0.3.0)
- [ ] Multi-GPU data parallelism (design note: `docs/DISTRIBUTED_TRAINING_DESIGN.md`; blocked on workloads, #281)
- [ ] Mixed precision training (FP16) — blocked upstream on Burn/Metal bf16 kernels (#270/#272; spike: `docs/FP16_FEASIBILITY.md`)
- [ ] JIT compilation for environments
- [ ] CUDA kernel optimization
- [x] GPU-backend (wgpu) training benchmarks vs the CPU NdArray baseline — measured on RTX 4090; CPU wins 4.4–9.5× at current small-net sizes (see `docs/BURN_BACKENDS.md` → "Measured CPU-vs-GPU throughput")

---

### Q2 2026 - Production Ready

#### Milestone 8: Library Polish (Mostly Complete ✅)
- [x] Comprehensive documentation (docs.rs, warning-free)
- [ ] Tutorial series
- [x] Example gallery ([docs/EXAMPLES.md](docs/EXAMPLES.md), 18 examples)
- [ ] Performance benchmarks vs Python
- [x] CI/CD pipeline (tests on Linux+macOS, clippy, fmt, docs, security audit, publish dry-run, Pages deploy)
- [x] Publish to crates.io (v0.1.0 2026-06-08 → v0.3.0 2026-07-06)

#### Milestone 9: Research Features
- [ ] Curriculum learning
- [ ] Hierarchical RL
- [ ] Meta-learning (MAML)
- [ ] Evolution strategies
- [ ] Population-based training (PBT)

## Priority Queue (Next Tasks)

As of v0.3.0 the entire actionable backlog has shipped; every open item is
blocked on a scope decision, an external event, or larger workloads:

1. **Multi-agent comms Phase 3** (#276) — differentiable Gumbel-softmax
   channel; deferred until the discrete-channel (Phases 1–2) results
   motivate it.

2. **Multi-host distributed training** (#281) — design note shipped; blocked
   until single-host workloads saturate (the async actor-learner covers the
   single-host case today).

3. **Mixed precision (FP16/BF16)** (#270/#272) — blocked upstream on
   Burn/Metal bf16 matmul kernels; cheapest path is waiting for a
   Burn/cubecl release.

4. **A real memory-hard environment suite** — FlickeringCartPole proved the
   recurrent stack; T-maze or occluded-observation envs would deepen the
   POMDP story.

5. **More environments** (Milestone 5) — Atari/ALE bindings or MuJoCo via
   `mujoco-rs` for breadth beyond the in-tree envs.

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

- [MULTI_AGENT_DESIGN.md](docs/MULTI_AGENT_DESIGN.md) - Multi-agent architecture
- [WASM_ROADMAP.md](docs/WASM_ROADMAP.md) - WASM visualization plan
- [CHANGELOG.md](CHANGELOG.md) - Release history and per-version status
