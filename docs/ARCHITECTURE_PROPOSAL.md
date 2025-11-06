# Thrust Architecture Proposal

## Executive Summary

Thrust aims to be the **fastest reinforcement learning library in Rust**, targeting 3-6x speedup over Python implementations like PufferLib through:

1. **Rayon-based parallel environments** (EnvPool-inspired)
2. **Zero-copy buffer management** (Rust ownership)
3. **Batched GPU inference** (tch-rs)
4. **CUDA kernels for hot paths** (GAE, preprocessing)
5. **Lock-free communication** (channels, not shared memory)

This document proposes a Rust-native architecture that leverages modern RL systems research while avoiding Python's limitations.

---

## Table of Contents

1. [PufferLib Analysis](#pufferlib-analysis)
2. [Proposed Architecture](#proposed-architecture)
3. [Component Design](#component-design)
4. [Performance Strategy](#performance-strategy)
5. [Implementation Roadmap](#implementation-roadmap)

---

## PufferLib Analysis

### Strengths ✅

1. **CUDA-Accelerated GAE**: 1000x faster than Python loops
   - C++ kernel with importance sampling
   - In-place computation
   - **Keep for Thrust** (Port to Rust/CUDA)

2. **Segmented Buffer Layout**: `[segments, horizon]`
   - Natural episode grouping
   - Efficient minibatch sampling
   - **Partially adopt** (benchmark vs traditional layout)

3. **Zero-Copy Environment Interface**:
   - Pre-allocated numpy arrays
   - Direct memory access
   - **Definitely adopt** (even better in Rust)

4. **Priority-Weighted Sampling**:
   - Weight minibatches by advantage magnitude
   - 15-30% sample efficiency improvement
   - **Add in Phase 2** (after basic PPO works)

### Weaknesses ❌

1. **Python GIL Bottleneck**:
   - Even with multiprocessing, coordination overhead
   - Rust has true parallelism
   - **Thrust advantage**: No GIL, fearless concurrency

2. **Complex Abstractions**:
   - `PufferEnv`, `VecEnv`, multiple backends
   - Hard to debug, optimize
   - **Thrust approach**: CleanRL-style simplicity

3. **Async Environment Execution**:
   - Benefits minimal for CPU envs
   - Adds complexity
   - **Thrust decision**: Sync first, async later

4. **Memory Layout Overhead**:
   - Python lists, numpy copies
   - Reference counting overhead
   - **Thrust advantage**: Contiguous allocation, no GC

5. **Tight PyTorch Coupling**:
   - Hard to optimize for other backends
   - **Thrust approach**: Use tch-rs but keep interface clean

### Key Takeaways

**What to Keep**:
- CUDA kernels for compute-heavy ops
- Zero-copy buffer philosophy
- Vectorized environment execution
- Batched GPU inference

**What to Improve**:
- Simpler, more transparent code (CleanRL philosophy)
- True parallelism (Rayon, not multiprocessing)
- Rust's ownership for zero-cost abstractions
- Lock-free communication patterns

---

## Proposed Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────┐
│                   Training Loop                      │
│                    (main thread)                     │
└──────────────┬──────────────────────┬────────────────┘
               │                      │
               ↓                      ↓
    ┌──────────────────┐   ┌──────────────────┐
    │  EnvPool         │   │  Policy/Value    │
    │  (Rayon)         │   │  (tch-rs GPU)    │
    │                  │   │                  │
    │  ┌────┬────┬────┐│   │  Forward Pass    │
    │  │Env1│Env2│EnvN││   │  Batched         │
    │  └────┴────┴────┘│   │  Inference       │
    │  Parallel Step   │   └──────────────────┘
    └──────────────────┘
               ↓
    ┌──────────────────┐
    │  RolloutBuffer   │
    │  Pre-allocated   │
    │  Zero-copy       │
    └──────────────────┘
               ↓
    ┌──────────────────┐
    │  GAE Computation │
    │  (CUDA kernel)   │
    └──────────────────┘
               ↓
    ┌──────────────────┐
    │  PPO Update      │
    │  (GPU)           │
    └──────────────────┘
```

### Design Principles

1. **Synchronous by Default**: Simpler, more stable, matches PPO paper
2. **Explicit Parallelism**: Rayon for environments, tch-rs for GPU
3. **Zero-Copy Everywhere**: Rust ownership enables true zero-copy
4. **No Shared Memory**: Use channels for inter-thread communication
5. **Profile-Guided Optimization**: Measure first, optimize hot paths

---

## Component Design

### 1. EnvPool: Vectorized Environment Execution

```rust
/// High-performance vectorized environment execution
pub struct EnvPool<E: Environment> {
    envs: Vec<E>,
    thread_pool: rayon::ThreadPool,
    num_envs: usize,
}

impl<E: Environment> EnvPool<E> {
    pub fn new(env_fn: impl Fn() -> E, num_envs: usize) -> Self {
        let envs = (0..num_envs).map(|_| env_fn()).collect();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_envs.min(num_cpus::get()))
            .build()
            .unwrap();

        Self { envs, thread_pool, num_envs }
    }

    /// Parallel step across all environments
    pub fn step(&mut self, actions: &[i64]) -> EnvPoolResult {
        self.thread_pool.install(|| {
            self.envs
                .par_iter_mut()
                .zip(actions)
                .map(|(env, &action)| env.step(action).unwrap())
                .collect()
        })
    }

    /// Parallel reset across all environments
    pub fn reset(&mut self) -> Vec<Vec<f32>> {
        self.thread_pool.install(|| {
            self.envs
                .par_iter_mut()
                .map(|env| env.reset().unwrap())
                .collect()
        })
    }
}
```

**Performance**: O(max(env_times)) instead of O(sum(env_times))

### 2. RolloutBuffer: Pre-Allocated Storage

```rust
/// Zero-allocation rollout buffer
pub struct RolloutBuffer {
    // Shape: [num_steps, num_envs, obs_dim]
    // Layout: row-major for cache efficiency
    observations: Vec<f32>,
    actions: Vec<i64>,
    rewards: Vec<f32>,
    values: Vec<f32>,
    log_probs: Vec<f32>,
    dones: Vec<bool>,

    // Computed fields
    advantages: Vec<f32>,
    returns: Vec<f32>,

    // Dimensions
    num_steps: usize,
    num_envs: usize,
    obs_dim: usize,

    // Index tracking
    current_step: usize,
}

impl RolloutBuffer {
    pub fn new(num_steps: usize, num_envs: usize, obs_dim: usize) -> Self {
        let total_size = num_steps * num_envs;
        let obs_size = total_size * obs_dim;

        Self {
            observations: vec![0.0; obs_size],
            actions: vec![0; total_size],
            rewards: vec![0.0; total_size],
            values: vec![0.0; total_size],
            log_probs: vec![0.0; total_size],
            dones: vec![false; total_size],
            advantages: vec![0.0; total_size],
            returns: vec![0.0; total_size],
            num_steps,
            num_envs,
            obs_dim,
            current_step: 0,
        }
    }

    /// Store transition (zero-copy)
    #[inline]
    pub fn store(&mut self, env_id: usize, obs: &[f32], action: i64,
                 reward: f32, value: f32, log_prob: f32, done: bool) {
        let step = self.current_step;
        let idx = step * self.num_envs + env_id;
        let obs_idx = idx * self.obs_dim;

        // Direct memory copy
        self.observations[obs_idx..obs_idx + self.obs_dim]
            .copy_from_slice(obs);
        self.actions[idx] = action;
        self.rewards[idx] = reward;
        self.values[idx] = value;
        self.log_probs[idx] = log_prob;
        self.dones[idx] = done;
    }

    /// Compute advantages using GAE
    pub fn compute_advantages(&mut self, last_values: &[f32],
                             gamma: f32, gae_lambda: f32) {
        // Option 1: Pure Rust (Phase 1)
        compute_gae_rust(
            &self.values,
            &self.rewards,
            &self.dones,
            last_values,
            gamma,
            gae_lambda,
            &mut self.advantages,
            &mut self.returns,
            self.num_steps,
            self.num_envs,
        );

        // Option 2: CUDA kernel (Phase 3)
        // cuda_ops::compute_gae_advantages(...);
    }
}
```

**Key Features**:
- Pre-allocated, no runtime allocation
- Contiguous memory layout
- Cache-friendly access patterns
- Ready for CUDA kernel integration

### 3. Policy: Batched Neural Network

```rust
use tch::{nn, Device, Kind, Tensor};

/// MLP policy for discrete actions
pub struct MlpPolicy {
    vs: nn::VarStore,
    policy_net: nn::Sequential,
    value_net: nn::Sequential,
    device: Device,
}

impl MlpPolicy {
    pub fn new(obs_dim: i64, action_dim: i64, hidden_dim: i64) -> Self {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let root = vs.root();

        // Policy network
        let policy_net = nn::seq()
            .add(nn::linear(&root / "p_fc1", obs_dim, hidden_dim, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(&root / "p_fc2", hidden_dim, hidden_dim, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(&root / "p_out", hidden_dim, action_dim, Default::default()));

        // Value network
        let value_net = nn::seq()
            .add(nn::linear(&root / "v_fc1", obs_dim, hidden_dim, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(&root / "v_fc2", hidden_dim, hidden_dim, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(&root / "v_out", hidden_dim, 1, Default::default()));

        Self { vs, policy_net, value_net, device: vs.device() }
    }

    /// Batched forward pass
    pub fn forward(&self, obs_batch: &Tensor) -> (Tensor, Tensor) {
        let logits = self.policy_net.forward(obs_batch);
        let values = self.value_net.forward(obs_batch);
        (logits, values.squeeze_dim(-1))
    }

    /// Get action probabilities and sample
    pub fn get_action(&self, obs_batch: &Tensor) -> (Tensor, Tensor, Tensor) {
        let (logits, values) = self.forward(obs_batch);
        let probs = logits.softmax(-1, Kind::Float);
        let actions = probs.multinomial(1, true).squeeze_dim(-1);
        let log_probs = probs.log().gather(-1, &actions.unsqueeze(-1), false)
            .squeeze_dim(-1);
        (actions, log_probs, values)
    }
}
```

**Performance**: Single GPU call for entire batch (no per-env overhead)

### 4. PPO Trainer: Main Training Loop

```rust
pub struct PpoTrainer<E: Environment> {
    env_pool: EnvPool<E>,
    policy: MlpPolicy,
    buffer: RolloutBuffer,
    optimizer: nn::Optimizer,

    // Hyperparameters
    num_steps: usize,
    num_envs: usize,
    gamma: f32,
    gae_lambda: f32,
    clip_epsilon: f32,
    value_coef: f32,
    entropy_coef: f32,
}

impl<E: Environment> PpoTrainer<E> {
    pub fn train_step(&mut self) -> TrainingMetrics {
        // 1. Collect rollouts (parallel environment execution)
        let obs = self.env_pool.reset();
        for step in 0..self.num_steps {
            // Convert obs to tensor (batched)
            let obs_tensor = tensor_from_obs(&obs);

            // Batched policy forward pass (GPU)
            let (actions, log_probs, values) = self.policy.get_action(&obs_tensor);

            // Parallel environment step (CPU)
            let results = self.env_pool.step(&actions_to_vec(&actions));

            // Store in buffer (zero-copy)
            for env_id in 0..self.num_envs {
                self.buffer.store(
                    env_id,
                    &obs[env_id],
                    actions_vec[env_id],
                    results[env_id].reward,
                    values_vec[env_id],
                    log_probs_vec[env_id],
                    results[env_id].terminated || results[env_id].truncated,
                );
            }

            obs = results.into_iter().map(|r| r.observation).collect();
        }

        // 2. Compute advantages (CUDA kernel in Phase 3)
        let last_values = self.compute_bootstrap_values(&obs);
        self.buffer.compute_advantages(&last_values, self.gamma, self.gae_lambda);

        // 3. PPO update (multiple epochs over buffer)
        let mut metrics = TrainingMetrics::default();
        for epoch in 0..self.ppo_epochs {
            for batch in self.buffer.iter_minibatches(self.batch_size) {
                metrics += self.ppo_update(batch);
            }
        }

        metrics
    }

    fn ppo_update(&mut self, batch: MiniBatch) -> TrainingMetrics {
        // Standard PPO loss computation (GPU)
        let obs_tensor = batch.observations_to_tensor(self.policy.device());
        let (logits, values) = self.policy.forward(&obs_tensor);

        // ... PPO loss computation ...

        self.optimizer.backward_step(&loss);
        TrainingMetrics::from_loss(&loss)
    }
}
```

**Flow**:
1. Parallel rollout collection (CPU + GPU)
2. GAE computation (CUDA kernel)
3. PPO updates (GPU)
4. Repeat

---

## Performance Strategy

### Phase 1: Correct Implementation ✅
**Goal**: Match CleanRL performance

- Synchronous PPO
- Pure Rust GAE
- Rayon environment pool
- tch-rs for neural networks
- **Target**: 10K samples/sec on CartPole

### Phase 2: Rayon Optimization
**Goal**: 3x speedup over Python

- Profile hot paths
- Optimize buffer layouts
- Zero-copy conversions
- Efficient tensor creation
- **Target**: 100K samples/sec on CartPole

### Phase 3: CUDA Acceleration
**Goal**: 10x speedup over Python

- CUDA kernel for GAE
- GPU-resident buffers
- Fused operations
- Async GPU transfers
- **Target**: 1M samples/sec on CartPole

### Phase 4: Distributed Training
**Goal**: Linear scaling

- Multi-GPU support
- Distributed actors
- Async learner
- Network-efficient communication
- **Target**: 10M+ samples/sec cluster-wide

---

## Key Architectural Decisions

### 1. Synchronous vs Asynchronous

**Decision: Start Synchronous**

Rationale:
- PPO is on-policy, sync is natural
- Simpler to implement and debug
- Async benefits minimal for fast CPU envs
- Can add async later (IMPALA-style)

Trade-off:
- Lower throughput ceiling vs async
- But better sample efficiency
- Easier to reason about

### 2. Rayon vs Tokio

**Decision: Rayon for Environments**

Rationale:
- Environments are CPU-bound, not I/O-bound
- Rayon's work-stealing perfect for this
- Avoid Tokio/Rayon mixing issues
- Clear ownership model

Trade-off:
- No async I/O benefits
- But we don't need them for local envs

### 3. Buffer Layout: [steps, envs] vs [segments, horizon]

**Decision: Start with [steps, envs]**

Rationale:
- Simpler indexing
- More intuitive
- Standard in literature
- Can switch later if benchmarks show benefit

Trade-off:
- Slightly less cache-friendly for minibatch sampling
- But difference likely negligible

### 4. GPU Library: tch-rs vs candle vs burn

**Decision: tch-rs**

Rationale:
- Most mature (5+ years)
- PyTorch compatibility (model loading)
- Used in Border (proven)
- Good performance

Trade-offs:
- Requires libtorch dependency
- Not pure Rust
- But stability and ecosystem worth it

### 5. Communication: Channels vs Shared Memory

**Decision: Channels (crossbeam)**

Rationale:
- Rust idiom (fearless concurrency)
- Explicit ownership transfers
- No lock contention
- Easier to reason about

Trade-offs:
- Slightly more copying
- But Rust's zero-cost abstractions minimize this

---

## Implementation Roadmap

### Week 1-2: Foundation ✅
- [x] CartPole environment
- [x] RolloutBuffer (current implementation)
- [ ] EnvPool (Rayon-based)
- [ ] MlpPolicy (tch-rs)
- [ ] Basic PPO trainer

### Week 3-4: Core PPO
- [ ] Complete PPO algorithm (37 details)
- [ ] Logging and metrics
- [ ] Learning rate scheduling
- [ ] Advantage normalization
- [ ] Value function clipping

### Week 5-6: Testing & Benchmarking
- [ ] Unit tests for all components
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Compare to CleanRL baseline
- [ ] Profile and identify bottlenecks

### Week 7-8: Optimization
- [ ] Zero-copy optimizations
- [ ] Buffer layout tuning
- [ ] Batch size tuning
- [ ] Memory pooling
- [ ] **Target: 100K samples/sec**

### Week 9-10: More Environments
- [ ] Snake environment
- [ ] Atari wrapper (using atari-env)
- [ ] MuJoCo wrapper (using mujoco-rs)
- [ ] Benchmark across environments

### Week 11-12: CUDA Kernels
- [ ] GAE CUDA kernel
- [ ] Observation preprocessing
- [ ] Reward normalization
- [ ] **Target: 1M samples/sec**

---

## Success Metrics

### Performance
- [ ] 100K+ samples/sec on CartPole (CPU)
- [ ] 1M+ samples/sec with CUDA
- [ ] 3-6x faster than PufferLib (same hardware)
- [ ] <1 hour Atari training (40M frames)

### Quality
- [ ] Match PPO paper results on CartPole
- [ ] Match CleanRL performance on Atari
- [ ] Reproducible with fixed seeds
- [ ] Memory efficient (<1GB for CartPole)

### Usability
- [ ] Clear, documented API
- [ ] Example scripts for each environment
- [ ] Comprehensive error messages
- [ ] Easy to extend with new envs

---

## Open Questions

1. **Observation Preprocessing**: CPU or GPU?
   - Small obs (CartPole): CPU is fine
   - Images (Atari): GPU might be faster
   - **Answer**: Profile both, make configurable

2. **Minibatch Sampling**: Random or sequential?
   - Random: Better training stability
   - Sequential: Better cache locality
   - **Answer**: Random by default, benchmark both

3. **Value Network**: Shared or separate from policy?
   - Shared: Fewer parameters, faster
   - Separate: More capacity, common in practice
   - **Answer**: Separate (matches literature)

4. **Action Distribution**: Categorical or Beta for continuous?
   - Categorical: Discrete actions (CartPole)
   - Beta: Continuous actions (MuJoCo)
   - **Answer**: Start with Categorical, add Beta later

---

## Conclusion

Thrust's architecture prioritizes:
1. **Simplicity**: CleanRL-inspired, easy to understand
2. **Performance**: Rust's zero-cost abstractions + explicit parallelism
3. **Correctness**: Match PPO paper implementation exactly
4. **Extensibility**: Clean interfaces for new envs and algorithms

By leveraging Rayon, tch-rs, and future CUDA kernels, we can achieve 3-6x speedup over Python while maintaining code clarity.

Next step: Implement `EnvPool` and `MlpPolicy` to complete the basic training loop.

---

*Last Updated: 2025-11-05*
*Status: Architecture Proposal (Pending Implementation)*
