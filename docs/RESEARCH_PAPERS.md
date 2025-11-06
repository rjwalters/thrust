# Research Papers & Architecture Insights for Thrust

This document catalogs key papers, systems, and insights that should inform Thrust's architecture design.

---

## Table of Contents

1. [Core RL Algorithms](#core-rl-algorithms)
2. [Systems & Performance](#systems--performance)
3. [Rust-Specific Considerations](#rust-specific-considerations)
4. [Architecture Recommendations](#architecture-recommendations)

---

## Core RL Algorithms

### 1. **Proximal Policy Optimization (PPO)**
- **Paper**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **ArXiv**: https://arxiv.org/abs/1707.06347
- **Key Insights**:
  - Trust region optimization without KL penalty
  - Clipped objective prevents large policy updates
  - Works well with continuous and discrete actions
  - Naturally parallelizable across environments

**Implementation Details** (37 Critical Details):
- **Blog**: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- **Key Details**:
  - Vectorized advantage normalization
  - Learning rate annealing
  - Value function clipping
  - Gradient clipping
  - Orthogonal initialization
  - GAE-Lambda for advantage estimation

### 2. **Generalized Advantage Estimation (GAE)**
- **Paper**: Schulman et al., "High-Dimensional Continuous Control Using GAE" (2016)
- **ArXiv**: https://arxiv.org/abs/1506.02438
- **Key Insights**:
  - Bias-variance tradeoff via λ parameter
  - Exponentially-weighted average of TD errors
  - Typically λ=0.95, γ=0.99 for good performance
  - Critical for sample efficiency

---

## Systems & Performance

### 3. **EnvPool: Highly Parallel Environment Execution**
- **Paper**: https://arxiv.org/abs/2206.10558
- **GitHub**: https://github.com/sail-sg/envpool
- **Key Insights**:
  - **1M+ FPS** on Atari, **3M+ FPS** on MuJoCo
  - C++ thread pool with async execution
  - Zero-copy environment resets
  - Batched observation/action transfers
  - 3x end-to-end speedup with CleanRL PPO (200min → 73min)

**Architecture**:
```
┌─────────────────────────────────────────┐
│  Thread Pool (C++/Rust)                 │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐           │
│  │Env1│ │Env2│ │Env3│ │Env4│  × N      │
│  └────┘ └────┘ └────┘ └────┘           │
│         Async Execution                 │
│         Zero-Copy Buffers               │
└─────────────────────────────────────────┘
            ↓ Batch Transfer
┌─────────────────────────────────────────┐
│  GPU (PyTorch/tch-rs)                   │
│  Policy Network Forward Pass            │
│  Value Network Forward Pass             │
└─────────────────────────────────────────┘
```

**Why This Matters**:
- Environment execution is often the bottleneck (not GPU)
- Async execution hides environment latency
- Perfect fit for Rust's fearless concurrency

### 4. **Sample Factory: Asynchronous RL**
- **Paper**: Petrenko et al., "Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS" (2020)
- **ArXiv**: https://arxiv.org/abs/2006.11751
- **GitHub**: https://github.com/alex-petrenko/sample-factory
- **Key Insights**:
  - Asynchronous architecture (IMPALA-style)
  - 100K+ FPS throughput
  - Batched inference on GPU
  - Separate actors and learners

**Architecture**:
```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Actor 1  │  │ Actor 2  │  │ Actor N  │
│ (CPU)    │  │ (CPU)    │  │ (CPU)    │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │              │
     └─────────────┴──────────────┘
                   ↓
         ┌─────────────────────┐
         │  Shared Queue        │
         │  (Lock-free)         │
         └─────────────────────┘
                   ↓
         ┌─────────────────────┐
         │  Learner (GPU)       │
         │  Batched Updates     │
         └─────────────────────┘
```

### 5. **CleanRL: Research-Friendly Implementations**
- **Paper**: Huang et al., "CleanRL: High-quality Single-file Implementations of Deep RL Algorithms" (2022)
- **ArXiv**: https://arxiv.org/abs/2111.08819
- **GitHub**: https://github.com/vwxyzjn/cleanrl
- **Key Insights**:
  - Single-file implementations (ppo.py = 340 lines)
  - No abstractions for debuggability
  - Each file is self-contained
  - 3-4x speedup with EnvPool integration

**Philosophy for Thrust**:
- Clear, readable implementations over abstraction
- Each algorithm in one file when possible
- Optimize hot paths without sacrificing clarity

### 6. **V-trace: Off-Policy Correction**
- **Paper**: Espeholt et al., "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures" (2018)
- **ArXiv**: https://arxiv.org/abs/1802.01561
- **Key Insights**:
  - Off-policy correction for async RL
  - Importance sampling with clipping
  - PufferLib uses this for stability
  - Critical for distributed training

---

## Rust-Specific Considerations

### 7. **Tokio vs Rayon: Async vs Parallelism**
- **Blog**: https://blog.dureuill.net/articles/dont-mix-rayon-tokio/
- **Key Insights**:
  - **Don't mix Tokio and Rayon directly** - causes deadlocks
  - Use channels for communication between async and parallel code
  - Tokio = async I/O (network, timers)
  - Rayon = data parallelism (CPU-bound work)

**For RL in Rust**:
```rust
// Good: Separate concerns
┌─────────────────────────┐
│ Rayon Thread Pool       │
│ - Environment execution │
│ - Data preprocessing    │
│ - Advantage computation │
└─────────────────────────┘
           ↓ Channel
┌─────────────────────────┐
│ Main Thread (or Tokio)  │
│ - GPU inference         │
│ - Model updates         │
│ - Logging/metrics       │
└─────────────────────────┘
```

### 8. **Border: Async RL in Rust**
- **GitHub**: https://github.com/laboroai/border
- **Key Insights**:
  - Uses Tokio for async training
  - Separate actor processes with shared replay buffer
  - tch-rs for PyTorch bindings
  - Supports DQN, SAC, IQN

**Architecture Lessons**:
- Async actors work well in Rust
- Channels for actor → learner communication
- tch-rs is mature enough for production use

### 9. **Zero-Copy and Performance**
- **Insight**: Rust's ownership model enables true zero-copy
- **Key Patterns**:
  - `Arc<Mutex<T>>` for shared mutable state
  - `crossbeam` channels for lock-free MPMC
  - Memory-mapped buffers for observations
  - GPU-pinned memory for faster transfers

---

## Architecture Recommendations

### Design Principles for Thrust

#### 1. **Separate Compute from Coordination**
```rust
// Environment execution: Rayon (CPU parallelism)
// GPU inference: tch-rs (blocking)
// Coordination: Channels (no shared memory)
```

**Why**:
- Avoid Tokio/Rayon mixing issues
- Clear ownership boundaries
- Predictable performance

#### 2. **EnvPool-Inspired Thread Pool**
```rust
pub struct EnvPool {
    envs: Vec<CartPole>,
    thread_pool: rayon::ThreadPool,
    observation_buffer: Vec<Vec<f32>>,
}

impl EnvPool {
    pub fn step_async(&mut self, actions: &[i64]) -> StepResult {
        // Parallel step across all environments
        self.thread_pool.install(|| {
            self.envs.par_iter_mut()
                .zip(actions)
                .map(|(env, &action)| env.step(action))
                .collect()
        })
    }
}
```

**Benefits**:
- 10-100x faster than sequential execution
- Scales with CPU cores
- Zero-copy observation collection

#### 3. **Batched GPU Inference**
```rust
pub struct PolicyBatch {
    observations: Tensor,  // [batch_size, obs_dim]

    pub fn forward(&self, policy: &Policy) -> (Tensor, Tensor) {
        // Single GPU call for entire batch
        policy.forward(&self.observations)
    }
}
```

**Benefits**:
- Amortize GPU kernel launch overhead
- Better GPU utilization
- 5-10x faster than per-env inference

#### 4. **Lock-Free Rollout Buffer**
```rust
pub struct RolloutBuffer {
    // Pre-allocated, no runtime allocation
    observations: Vec<Vec<f32>>,

    // Direct indexing, no locks needed
    pub fn store(&mut self, step: usize, env_id: usize, ...) {
        self.observations[step][env_id] = obs;
    }
}
```

**Benefits**:
- No allocation during training
- Cache-friendly layout
- Thread-safe with simple ownership

#### 5. **CUDA Kernels for Hot Paths (Future)**
Following PufferLib's lead:
- GAE computation → CUDA kernel (1000x speedup)
- Observation preprocessing → CUDA
- Reward normalization → CUDA

**Implementation**:
```rust
// Use cuda-sys or cudarc
mod cuda_ops {
    pub fn compute_gae_advantages(
        values: &[f32],
        rewards: &[f32],
        dones: &[bool],
        gamma: f32,
        lambda: f32,
    ) -> Vec<f32> {
        // CUDA kernel dispatch
    }
}
```

### Performance Targets

Based on literature:

| Component | Target | Reference |
|-----------|--------|-----------|
| Environment FPS | 100K-1M | EnvPool (Atari) |
| GPU Batch Size | 2048-8192 | CleanRL, Sample Factory |
| Samples/sec (CartPole) | 1M+ | Should be easy with Rust |
| Training Time (Atari) | <1 hour | EnvPool + CleanRL |

### Key Trade-offs

#### Synchronous vs Asynchronous
- **Sync (PPO)**: Simpler, more stable, on-policy
- **Async (IMPALA)**: Higher throughput, requires V-trace, off-policy
- **Recommendation**: Start with sync PPO, add async later

#### CPU vs GPU Environments
- **CPU**: Most environments, easier to parallelize
- **GPU** (IsaacGym): Massive parallelism (1000s envs), tight integration
- **Recommendation**: Focus on CPU first, GPU env support later

#### Abstraction vs Performance
- **High abstraction**: Easier to use, harder to optimize
- **Low abstraction**: CleanRL-style, clear hot paths
- **Recommendation**: Minimal abstractions, clear ownership

---

## Recommended Reading Order

1. **Start here**: [37 PPO Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
2. **System design**: [EnvPool Paper](https://arxiv.org/abs/2206.10558)
3. **Rust patterns**: [Don't Mix Tokio and Rayon](https://blog.dureuill.net/articles/dont-mix-rayon-tokio/)
4. **Reference impl**: [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
5. **Advanced**: [Sample Factory Paper](https://arxiv.org/abs/2006.11751)

---

## Next Steps for Thrust

### Phase 1: Core Synchronous PPO ✅ (In Progress)
- [x] CartPole environment
- [x] Rollout buffer with GAE
- [ ] MLP policy (tch-rs)
- [ ] PPO training loop
- [ ] Basic benchmarks

### Phase 2: Performance Optimization
- [ ] Rayon-based EnvPool
- [ ] Batched GPU inference
- [ ] Zero-copy observation buffers
- [ ] Vectorized environments
- [ ] Target: 100K+ samples/sec on CartPole

### Phase 3: Advanced Features
- [ ] CUDA kernels for GAE
- [ ] Async actors (IMPALA-style)
- [ ] Distributed training
- [ ] More environments (Atari, MuJoCo)

### Phase 4: Research Features
- [ ] Recurrent policies (LSTM)
- [ ] Multi-agent support
- [ ] Curiosity-driven exploration
- [ ] Population-based training

---

## Key Questions to Answer

### Architecture Decisions

1. **Environment Parallelism**:
   - Q: Use Rayon thread pool or manual threads?
   - A: Rayon - easier to use, well-tested, integrates with iterators

2. **GPU Strategy**:
   - Q: tch-rs, candle, or burn?
   - A: tch-rs - most mature, proven in Border, PyTorch compatibility

3. **Buffer Layout**:
   - Q: [steps, envs] or [segments, horizon]?
   - A: Start with [steps, envs] (simpler), profile and optimize later

4. **Observation Types**:
   - Q: Support images (CNNs) from the start?
   - A: Start with vectors (CartPole, MuJoCo), add images in Phase 2

5. **Communication Pattern**:
   - Q: Shared memory or message passing?
   - A: Message passing (channels) - safer, easier to reason about

---

## Performance Checklist

Before claiming "production ready":

- [ ] 100K+ samples/sec on CartPole (single machine)
- [ ] Match or beat PufferLib on same hardware
- [ ] <1 hour Atari training (40M frames)
- [ ] Zero-copy environment execution
- [ ] Batched GPU inference
- [ ] Memory efficient (no runtime allocation in hot path)
- [ ] Reproducible results (fixed seeds)
- [ ] Comprehensive benchmarks

---

## References

### Papers
- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438
- EnvPool: https://arxiv.org/abs/2206.10558
- Sample Factory: https://arxiv.org/abs/2006.11751
- CleanRL: https://arxiv.org/abs/2111.08819
- IMPALA: https://arxiv.org/abs/1802.01561

### Systems
- CleanRL: https://github.com/vwxyzjn/cleanrl
- EnvPool: https://github.com/sail-sg/envpool
- Sample Factory: https://github.com/alex-petrenko/sample-factory
- PufferLib: https://github.com/PufferAI/PufferLib
- Border (Rust): https://github.com/laboroai/border

### Resources
- 37 PPO Details: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- Tokio vs Rayon: https://blog.dureuill.net/articles/dont-mix-rayon-tokio/
- tch-rs: https://github.com/LaurentMazare/tch-rs

---

*Last Updated: 2025-11-05*
*Thrust Version: 0.1.0 (Phase 1, Week 1)*
