# 🚀 Thrust Development Workplan

**Last Updated:** November 2024
**Status:** Phase 1 - Foundation (In Progress)

This document outlines the complete development roadmap for Thrust, from initial foundation to 1.0 release and beyond.

---

## 📅 Timeline Overview

| Phase | Duration | Target Date | Status |
|-------|----------|-------------|--------|
| **Phase 1: Foundation** | 4 weeks | Q1 2025 | 🟡 In Progress |
| **Phase 2: Performance** | 4 weeks | Q1 2025 | ⚪ Not Started |
| **Phase 3: Features** | 4 weeks | Q2 2025 | ⚪ Not Started |
| **Phase 4: Demo Site** | 4 weeks | Q2 2025 | ⚪ Not Started |
| **Phase 5: Polish & Launch** | 4 weeks | Q2 2025 | ⚪ Not Started |

**Total estimated time:** 20 weeks (~5 months)

---

## 🎯 Phase 1: Foundation (Weeks 1-4)

**Goal:** Establish core architecture and get a simple PPO agent training on CartPole

### Week 1: Core Infrastructure
- [x] Project setup and dependencies
- [x] Module structure (env, policy, buffer, train, utils)
- [x] Environment trait definition
- [x] CartPole physics implementation
- [x] Basic testing framework
- [x] CI/CD setup (Makefile with fmt, clippy, test, doc)

**Deliverables:**
- ✅ Compiling Rust project with proper structure
- ✅ Environment trait that matches PufferLib's interface
- ✅ Working CartPole environment (pure Rust, 390 lines, 11 tests)

### Week 2: Experience Buffers
- [x] Buffer trait definition
- [x] CPU-backed buffer implementation
- [x] Rollout data structures
- [x] GAE (Generalized Advantage Estimation) calculation
- [x] Unit tests for buffer operations
- [x] EnvPool for parallel environment execution

**Deliverables:**
- ✅ Experience buffer that can store trajectories (451 lines)
- ✅ Advantage computation (CPU version with GAE)
- ✅ EnvPool with Rayon parallelism (325 lines, 5 tests)
- ✅ Full test coverage with 17 buffer tests

### Week 3: Policy Wrapper
- [x] Install and configure libtorch (Homebrew PyTorch 2.9)
- [x] Enable tch-rs 0.22 in Cargo.toml (requires nightly Rust)
- [x] Policy trait definition
- [x] Simple MLP policy (feedforward, 259 lines)
- [x] Action sampling (Categorical distribution)
- [~] Integration tests (fixing tch-rs 0.22 API compatibility)

**Deliverables:**
- ✅ Working policy that can forward pass observations
- ✅ Action sampling from logits
- 🟡 Policy can train via gradient descent (optimizer created)
- ⚠️ Blocked: tch-rs 0.22 API changes (train/eval methods)

### Week 4: PPO Training Loop
- [ ] PPO config structure
- [ ] Training loop implementation
- [ ] Loss functions (policy loss, value loss, entropy)
- [ ] Minibatch sampling
- [ ] Logging and metrics
- [ ] Checkpoint saving/loading

**Deliverables:**
- ✅ End-to-end training on CartPole
- ✅ Agent solves CartPole (reward > 195)
- ✅ Training curves logged
- ✅ Saved checkpoints can be loaded and evaluated

**Phase 1 Success Criteria:**
- [x] Project compiles and has clear structure
- [x] CartPole environment implemented and tested (390 lines, 11 tests)
- [x] EnvPool for parallel execution (325 lines, 5 tests)
- [x] RolloutBuffer with GAE (451 lines, 17 tests)
- [x] MlpPolicy structure complete (259 lines)
- [~] Policy tests passing (blocked by tch-rs 0.22 API compatibility)
- [ ] PPO training loop implemented
- [ ] Agent trains and solves CartPole
- [ ] Code is well-documented (comprehensive docstrings)

---

## ⚡ Phase 2: Performance (Weeks 5-8)

**Goal:** Achieve 1M+ SPS with async vectorization and GPU optimizations

### Week 5: Serial Vectorization
- [ ] VectorEnv trait definition
- [ ] Serial vectorization (single-threaded)
- [ ] Batch environment stepping
- [ ] Shared buffer integration
- [ ] Performance profiling

**Deliverables:**
- ✅ Multiple environments running in serial
- ✅ Profiling shows where bottlenecks are

### Week 6: Async Vectorization
- [ ] Tokio-based async worker pool
- [ ] Shared memory buffers (via shared_memory crate)
- [ ] Zero-copy data transfer
- [ ] Worker process coordination
- [ ] Benchmarks vs serial

**Deliverables:**
- ✅ Async vectorization with 16+ workers
- ✅ 2-5x speedup over serial
- ✅ No data races or deadlocks

### Week 7: GPU Optimizations
- [ ] GPU-backed experience buffers
- [ ] Pinned memory for fast transfers
- [ ] Custom CUDA kernel for advantage computation
- [ ] Mixed precision training (AMP)
- [ ] GPU memory profiling

**Deliverables:**
- ✅ Buffers allocated directly on GPU
- ✅ CUDA advantage kernel (port from PufferLib)
- ✅ Faster than CPU advantage computation

### Week 8: Snake Environment
- [ ] Multi-agent Snake environment (pure Rust)
- [ ] Grid-based physics
- [ ] Observation space (local vision)
- [ ] Action space (4 directions)
- [ ] Rendering (for visualization)

**Deliverables:**
- ✅ Snake environment with 256+ agents
- ✅ Achieves 1M+ SPS
- ✅ Agents learn basic food-seeking behavior

**Phase 2 Success Criteria:**
- [ ] Achieves 1M+ SPS on CartPole
- [ ] Async vectorization works reliably
- [ ] CUDA kernels provide measurable speedup
- [ ] Snake environment trains successfully
- [ ] Performance matches or exceeds PufferLib

---

## 🎨 Phase 3: Features (Weeks 9-12)

**Goal:** Feature parity with production RL libraries

### Week 9: Recurrent Policies
- [ ] LSTM policy wrapper
- [ ] Hidden state management
- [ ] BPTT (Backpropagation Through Time)
- [ ] Truncated BPTT for long episodes
- [ ] Test on Snake (memory required)

**Deliverables:**
- ✅ LSTM policy implementation
- ✅ Agents learn temporal patterns
- ✅ Performance comparable to feedforward

### Week 10: Advanced PPO Features
- [ ] Prioritized experience replay
- [ ] V-trace importance sampling
- [ ] Clipped surrogate objective variants
- [ ] Adaptive KL penalty
- [ ] Hyperparameter tuning utilities

**Deliverables:**
- ✅ PER improves sample efficiency
- ✅ V-trace works with off-policy data
- ✅ Hyperparameter sweep functionality

### Week 11: Training Utilities
- [ ] WandB integration
- [ ] Neptune.ai integration
- [ ] TensorBoard support
- [ ] Rich console dashboard (like PufferLib)
- [ ] Automatic checkpointing

**Deliverables:**
- ✅ Training metrics logged to WandB
- ✅ Live dashboard in terminal
- ✅ Automatic checkpoint management

### Week 12: Distributed Training
- [ ] Multi-GPU support
- [ ] Data parallelism
- [ ] Gradient synchronization
- [ ] Load balancing
- [ ] Fault tolerance

**Deliverables:**
- ✅ Training scales to 2-4 GPUs
- ✅ Near-linear speedup
- ✅ Handles worker failures gracefully

**Phase 3 Success Criteria:**
- [ ] LSTM policies work correctly
- [ ] Advanced features match PufferLib
- [ ] Logging integrations work
- [ ] Distributed training scales
- [ ] Documentation is comprehensive

---

## 🌐 Phase 4: Demo Site (Weeks 13-16)

**Goal:** Live website with trained agents playing in browser

### Week 13: WebAssembly Backend
- [ ] Compile policy to WASM
- [ ] WASM-compatible inference engine
- [ ] Model serialization (SafeTensors)
- [ ] Browser-side environment rendering
- [ ] Performance optimization

**Deliverables:**
- ✅ Policy runs in browser
- ✅ Real-time inference (60 FPS)
- ✅ Model loading from checkpoint

### Week 14: Frontend Development
- [ ] Landing page design
- [ ] Live canvas rendering (WebGL)
- [ ] Game selection UI
- [ ] Training stats dashboard
- [ ] Model download functionality

**Deliverables:**
- ✅ Professional landing page
- ✅ Smooth 60 FPS rendering
- ✅ Interactive game controls

### Week 15: Backend API
- [ ] Axum REST API
- [ ] Model serving endpoint
- [ ] Training stats API
- [ ] WebSocket for live updates
- [ ] Database for analytics

**Deliverables:**
- ✅ API serves trained models
- ✅ Real-time training updates via WebSocket
- ✅ Analytics tracked

### Week 16: Deployment
- [ ] Hosting setup (Vercel/Cloudflare)
- [ ] Domain configuration (thrust-rl.com)
- [ ] CDN for model weights
- [ ] SSL certificates
- [ ] Monitoring and logging

**Deliverables:**
- ✅ Live demo at thrust-rl.com
- ✅ 3 trained agents (CartPole, Snake, Asteroids)
- ✅ Fast loading times globally

**Phase 4 Success Criteria:**
- [ ] Website is live and accessible
- [ ] Agents play smoothly in browser
- [ ] Training stats update in real-time
- [ ] Site is fast and responsive
- [ ] No crashes or errors

---

## 🎀 Phase 5: Polish & Launch (Weeks 17-20)

**Goal:** v1.0 release with documentation and marketing

### Week 17: Documentation
- [ ] API documentation (rustdoc)
- [ ] Tutorial notebooks
- [ ] Examples repository
- [ ] Architecture guide
- [ ] Performance tuning guide

**Deliverables:**
- ✅ Complete API docs
- ✅ 5+ tutorial examples
- ✅ Beginner-friendly guides

### Week 18: Testing & CI/CD
- [ ] Comprehensive unit tests
- [ ] Integration tests
- [ ] Benchmark suite
- [ ] Continuous integration
- [ ] Automated releases

**Deliverables:**
- ✅ 80%+ test coverage
- ✅ CI runs on all PRs
- ✅ Automated crate publishing

### Week 19: Community & Marketing
- [ ] Blog post: "Rewriting PufferLib in Rust"
- [ ] Hacker News launch
- [ ] Reddit posts (r/rust, r/machinelearning)
- [ ] Twitter thread with demos
- [ ] Discord community setup

**Deliverables:**
- ✅ Launch blog post
- ✅ HN/Reddit threads
- ✅ Active community channels

### Week 20: v1.0 Release
- [ ] Final bug fixes
- [ ] Version bump to 1.0.0
- [ ] crates.io publication
- [ ] GitHub release with binaries
- [ ] Changelog

**Deliverables:**
- ✅ v1.0.0 published to crates.io
- ✅ GitHub release with assets
- ✅ Stable API

**Phase 5 Success Criteria:**
- [ ] Documentation is excellent
- [ ] All tests pass
- [ ] v1.0 is published
- [ ] Community is engaged
- [ ] Project has visibility (500+ stars)

---

## 🎯 Success Metrics

### Technical Metrics
- **Performance:** 3M+ SPS on standard benchmarks
- **Test Coverage:** 80%+ code coverage
- **Documentation:** 100% public API documented
- **Reliability:** Zero known critical bugs

### Community Metrics
- **Stars:** 500+ GitHub stars by v1.0
- **Crates.io Downloads:** 1,000+ in first month
- **Contributors:** 5+ external contributors
- **Discord Members:** 100+ community members

### Demo Site Metrics
- **Uptime:** 99.9% availability
- **Performance:** <100ms page load time
- **Engagement:** 10,000+ monthly visitors

---

## 🚧 Risks & Mitigation

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| tch-rs compatibility issues | Medium | High | Keep close to stable PyTorch versions |
| CUDA kernel bugs | Medium | Medium | Extensive testing, fallback to CPU |
| Performance doesn't meet targets | Low | High | Profile early, optimize incrementally |
| WASM limitations | Medium | Medium | Test early, have backup (video demos) |

### Schedule Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Scope creep | High | Medium | Strict phase boundaries, defer features |
| Dependencies blocked | Low | High | Identify blockers early, have alternatives |
| Burnout | Medium | High | Sustainable pace, celebrate milestones |

---

## 🤝 How to Contribute

We need help with:

**Phase 1 (Current):**
- [ ] CartPole environment implementation
- [ ] Experience buffer optimizations
- [ ] PPO testing and validation

**Phase 2:**
- [ ] Async vectorization design review
- [ ] CUDA kernel development
- [ ] Snake environment artwork

**Phase 3-5:**
- [ ] WASM expertise for browser demos
- [ ] Frontend design for demo site
- [ ] Technical writing for documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started!

---

## 📞 Contact

- **GitHub Issues:** For bugs and feature requests
- **Discussions:** For questions and ideas
- **Discord:** Coming soon!

---

**Built with 🦀 Rust and ❤️ for reinforcement learning**
