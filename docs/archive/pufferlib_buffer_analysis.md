# PufferLib Experience Buffer and Rollout Storage Analysis

## Overview
PufferLib is a high-performance reinforcement learning library that uses a sophisticated vectorized buffer system for efficient experience storage and PPO training. The implementation uses mixed CPU/GPU tensor storage with optimized advantage computation via C++/CUDA kernels.

## Key Files

### Core Implementation Files:
1. `/Users/rwalters/GitHub/PufferLib/pufferlib/pufferl.py` - Main PPO trainer (1219 lines)
2. `/Users/rwalters/GitHub/PufferLib/pufferlib/vector.py` - Vectorization backends (926 lines)
3. `/Users/rwalters/GitHub/PufferLib/pufferlib/pufferlib.py` - Core buffer setup (459 lines)
4. `/Users/rwalters/GitHub/PufferLib/pufferlib/extensions/pufferlib.cpp` - C++ advantage kernel
5. `/Users/rwalters/GitHub/PufferLib/pufferlib/cleanrl_ppo_atari.py` - Example PPO implementation

---

## 1. BUFFER STORAGE STRUCTURE

### Data Organization
PufferLib organizes experience buffers into **segments** (episodes/trajectories) with a fixed **horizon** (sequence length):

```python
# From pufferl.py lines 90-106
segments = batch_size // horizon
total_agents = vecenv.num_agents

# Main storage tensors (shape: [segments, horizon])
self.observations = torch.zeros(segments, horizon, *obs_space.shape)
self.actions = torch.zeros(segments, horizon, *atn_space.shape)
self.values = torch.zeros(segments, horizon)
self.logprobs = torch.zeros(segments, horizon)
self.rewards = torch.zeros(segments, horizon)
self.terminals = torch.zeros(segments, horizon)      # Done flags
self.truncations = torch.zeros(segments, horizon)    # Truncation flags
self.ratio = torch.ones(segments, horizon)           # Importance weights for PPO
self.importance = torch.ones(segments, horizon)      # V-trace importance
```

### Key Characteristics:
- **2D Tensor Layout**: `[segments, horizon]` for most quantities
- **Segments** = Total agents that fit in batch (segments = batch_size / horizon)
- **Horizon** = Rollout length (BPTT horizon)
- **Device Placement**: Can be CPU offloaded or GPU-resident depending on config
- **Pin Memory**: Used for efficient CPU-GPU transfers when cpu_offload=True

### Environment Episode Tracking
```python
# Tracking which agent belongs to which segment
self.ep_lengths = torch.zeros(total_agents, dtype=torch.int32)
self.ep_indices = torch.arange(total_agents, dtype=torch.int32)
self.free_idx = total_agents
```

---

## 2. TRAJECTORY STORAGE COMPONENTS

### A. Observations
- **Shape**: `(segments, horizon, *obs_shape)`
- **Type**: Matches environment observation space dtype
- **Layout**: Stored sequentially during rollout collection phase
- **CPU Offloading**: Can be pinned in CPU memory for memory efficiency

### B. Actions & Log Probabilities
```python
self.actions = torch.zeros(segments, horizon, *atn_space.shape)
self.logprobs = torch.zeros(segments, horizon)
```
- Stored during action sampling in evaluate() phase
- Used for PPO policy loss: `pg_loss = -advantages * (new_logprob / old_logprob)`

### C. Rewards
```python
self.rewards = torch.zeros(segments, horizon)
```
- Clipped to [-1, 1] for stability: `r = torch.clamp(r, -1, 1)`
- Raw environment rewards before bootstrapping

### D. Terminal/Truncation Flags
```python
self.terminals = torch.zeros(segments, horizon)    # Episode done (terminal)
self.truncations = torch.zeros(segments, horizon)  # Episode truncated (time limit)
```
- Used in GAE computation to reset advantage accumulation
- Converted to float for computation

### E. Value Estimates
```python
self.values = torch.zeros(segments, horizon)
```
- Baseline values from critic network
- Updated during training with clipped value loss
- Used for advantage computation and returns

### F. Importance Weights
```python
self.ratio = torch.ones(segments, horizon)         # PPO ratio (new/old log prob)
self.importance = torch.ones(segments, horizon)    # V-trace importance weights
```
- `ratio` = exp(new_logprob - old_logprob)
- Used for PPO clipping and advantage weighting
- Tracks policy change across updates

---

## 3. GAE (GENERALIZED ADVANTAGE ESTIMATION) COMPUTATION

### Overview
PufferLib implements a hybrid GAE + V-trace advantage computation for improved stability:

### Implementation Details
From `pufferl.py` lines 639-659:

```python
def compute_puff_advantage(values, rewards, terminals, ratio, advantages, 
                          gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
    '''CUDA kernel for puffer advantage with automatic CPU fallback'''
    
    device = values.device
    if not ADVANTAGE_CUDA:
        # CPU fallback
        values = values.cpu()
        rewards = rewards.cpu()
        terminals = terminals.cpu()
        ratio = ratio.cpu()
        advantages = advantages.cpu()
    
    # Call C++/CUDA kernel
    torch.ops.pufferlib.compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip)
    
    if not ADVANTAGE_CUDA:
        return advantages.to(device)
    
    return advantages
```

### Algorithm (from pufferlib.cpp)
The kernel implements modified V-trace with GAE:

```cpp
void puff_advantage_row(float* values, float* rewards, float* dones,
                       float* importance, float* advantages, 
                       float gamma, float lambda,
                       float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0;
    
    // Backward iteration through trajectory
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        
        // V-trace importance clipping
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        
        // TD error
        float delta = rho_t * (rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t]);
        
        // GAE accumulation with importance weighting
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}
```

### Key Features:
1. **In-place Computation**: Advantage tensor modified in-place for efficiency
2. **Backward Iteration**: Processes trajectory from end to start (time t-1 to 0)
3. **V-trace Clipping**: 
   - `rho_clip`: Clips importance weights for TD error calculation
   - `c_clip`: Clips importance weights for GAE accumulation
4. **Terminal State Handling**: Sets `nextnonterminal = 0` when episode ends, breaking advantage propagation
5. **Dual Kernel Support**: C++ CPU implementation with optional CUDA acceleration

### Formula Breakdown:
```
For each timestep t (backward from T-1 to 0):
  nextnonterminal = 0 if done[t+1] else 1
  rho_t = min(importance[t], rho_clip)        # V-trace importance clipping
  c_t = min(importance[t], c_clip)            # GAE clipping
  
  delta_t = rho_t * (r[t+1] + gamma*V[t+1]*nextnonterminal - V[t])  # TD error
  adv[t] = delta_t + gamma*lambda*c_t*adv[t+1]*nextnonterminal
```

### Usage in Training Loop
From `pufferl.py` lines 336-356:

```python
# Initial advantage computation (full batch)
advantages = torch.zeros(shape, device=device)
advantages = compute_puff_advantage(
    self.values, self.rewards, self.terminals, self.ratio,
    advantages, config['gamma'], config['gae_lambda'],
    config['vtrace_rho_clip'], config['vtrace_c_clip']
)

# Prioritized sampling based on advantage magnitude
adv = advantages.abs().sum(axis=1)
prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
idx = torch.multinomial(prio_probs, self.minibatch_segments)

# Recompute advantages for selected minibatch after policy update
adv = compute_puff_advantage(mb_values, mb_rewards, mb_terminals,
    ratio, adv, config['gamma'], config['gae_lambda'],
    config['vtrace_rho_clip'], config['vtrace_c_clip']
)

# Normalize advantages with importance weighting
adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)
```

---

## 4. ROLLOUT BUFFER STRUCTURE

### Phase 1: Collection (evaluate() method)
```python
# Lines 228-314 in pufferl.py
def evaluate(self):
    self.full_rows = 0
    
    while self.full_rows < self.segments:
        # Get observation from environment
        o, r, d, t, info, env_id, mask = self.vecenv.recv()
        
        # Episode length tracking
        l = self.ep_lengths[env_id.start].item()
        
        # Store step data
        self.observations[batch_rows, l] = o
        self.actions[batch_rows, l] = action
        self.logprobs[batch_rows, l] = logprob
        self.rewards[batch_rows, l] = r
        self.terminals[batch_rows, l] = d.float()
        self.values[batch_rows, l] = value.flatten()
        
        # Check if episode is full (reached horizon)
        if l+1 >= config['bptt_horizon']:
            num_full = env_id.stop - env_id.start
            self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=device)
            self.ep_lengths[env_id] = 0
            self.free_idx += num_full
            self.full_rows += num_full  # Count filled segments
```

**Key Design**:
- Collects full episodes of length `horizon` in parallel
- Segments are filled asynchronously as environments complete rollouts
- When a segment is full, it's marked with an index in `ep_indices`
- Uses `full_rows` counter to know when batch is ready for training

### Phase 2: Advantage Computation
Advantages computed on entire batch with `compute_puff_advantage()`

### Phase 3: Minibatching & Training
```python
# Lines 336-426 in pufferl.py

# Priority-weighted sampling from collected episodes
adv = advantages.abs().sum(axis=1)  # Sum advantages across time
prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
idx = torch.multinomial(prio_probs, self.minibatch_segments)

# Extract minibatch
mb_obs = self.observations[idx]
mb_actions = self.actions[idx]
mb_logprobs = self.logprobs[idx]
mb_advantages = advantages[idx]
mb_values = self.values[idx]
mb_returns = advantages[idx] + mb_values

# Reshape for policy forward pass
mb_obs = mb_obs.reshape(-1, *obs_space.shape)

# Policy update
logits, newvalue = self.policy(mb_obs, state)
actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

# Compute losses
ratio = (newlogprob - mb_logprobs).exp()
pg_loss1 = -adv * ratio
pg_loss2 = -adv * torch.clamp(ratio, 1-clip_coef, 1+clip_coef)
pg_loss = torch.max(pg_loss1, pg_loss2).mean()

# Value loss with clipping
v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
v_loss_unclipped = (newvalue - mb_returns) ** 2
v_loss_clipped = (v_clipped - mb_returns) ** 2
v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

# Entropy bonus
loss = pg_loss + vf_coef*v_loss - ent_coef*entropy_loss

# Backward pass
loss.backward()
```

---

## 5. DATA STRUCTURES AND ORGANIZATION

### Tensor Organization Pattern
```
observations: [segments, horizon, *obs_shape]
actions:      [segments, horizon, *atn_shape]
rewards:      [segments, horizon]
values:       [segments, horizon]
logprobs:     [segments, horizon]
terminals:    [segments, horizon]
truncations:  [segments, horizon]
ratio:        [segments, horizon]
importance:   [segments, horizon]
```

### Episode Tracking Data
```python
ep_lengths:  [total_agents]  # Current length of each agent's episode
ep_indices:  [total_agents]  # Which segment index each agent writes to
free_idx:    int             # Next available segment index
```

### Memory Layout Strategy
- **Contiguous**: All tensors are contiguous for efficient kernel operations
- **Device**: Can be CPU with pinned memory or GPU-resident
- **Allocation**: Pre-allocated once during initialization (no dynamic resizing)
- **Reuse**: Tensors are reused across epochs (in-place operations where possible)

### LSTM Support (for RNN policies)
```python
if config['use_rnn']:
    n = vecenv.agents_per_batch
    h = policy.hidden_size
    self.lstm_h = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}
    self.lstm_c = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}
```

---

## 6. VECTORIZATION BACKENDS

### Environment Synchronization
From `vector.py`, PufferLib supports three backends:

#### Serial Backend
- Single process, synchronous execution
- Used for debugging
- Simple sequential environment stepping

#### Multiprocessing Backend
- Multi-core CPU parallelization with shared memory
- Workers run in separate processes
- Shared memory buffers for observations/rewards/actions:
  ```python
  self.shm = dict(
      observations=RawArray(obs_ctype, num_agents * int(np.prod(obs_shape))),
      actions=RawArray(atn_ctype, num_agents * int(np.prod(atn_shape))),
      rewards=RawArray('f', num_agents),
      terminals=RawArray('b', num_agents),
      truncateds=RawArray('b', num_agents),
      masks=RawArray('b', num_agents),
      semaphores=RawArray('c', num_workers),
      notify=RawArray('b', num_workers),
  )
  ```

#### Ray Backend
- Distributed parallelization using Ray
- Suitable for clusters

### Buffer Format for Environments
From `pufferlib.py` lines 22-43:
```python
def set_buffers(env, buf=None):
    if buf is None:
        env.observations = np.zeros((env.num_agents, *obs_space.shape), dtype=obs_space.dtype)
        env.rewards = np.zeros(env.num_agents, dtype=np.float32)
        env.terminals = np.zeros(env.num_agents, dtype=bool)
        env.truncations = np.zeros(env.num_agents, dtype=bool)
        env.masks = np.ones(env.num_agents, dtype=bool)
        env.actions = np.zeros(atn_space.shape, dtype=atn_dtype)
    else:
        # Environment writes directly to provided buffers
        env.observations = buf['observations']
        env.rewards = buf['rewards']
        # ... etc
```

**Key Insight**: Environments write directly to shared memory buffers, zero-copy communication!

---

## 7. PRIORITY-WEIGHTED EXPERIENCE REPLAY

PufferLib implements prioritized experience replay based on advantage magnitude:

```python
# Compute advantage magnitude for each segment
adv = advantages.abs().sum(axis=1)  # Sum across horizon dimension

# Convert to probabilities with exponent alpha
prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)

# Sample minibatches according to priorities
idx = torch.multinomial(prio_probs, self.minibatch_segments)

# Apply importance sampling correction (annealed beta)
anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
mb_prio = (self.segments*prio_probs[idx, None])**-anneal_beta
adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)
```

**Configuration Parameters**:
- `prio_alpha`: Exponent for advantage weighting (default varies by env)
- `prio_beta0`: Initial importance sampling correction exponent
- Anneals from `beta0` to 1.0 over training

---

## 8. COMPARISON WITH CLEANRL'S SIMPLE PPO

### CleanRL (Simple Baseline)
```python
# Sequential storage: [num_steps, num_envs]
obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape)
actions = torch.zeros((args.num_steps, args.num_envs) + atn_shape)
logprobs = torch.zeros((args.num_steps, args.num_envs))
rewards = torch.zeros((args.num_steps, args.num_envs))
dones = torch.zeros((args.num_steps, args.num_envs))
values = torch.zeros((args.num_steps, args.num_envs))

# Simple GAE computation (no importance weights)
for t in reversed(range(args.num_steps)):
    if t == args.num_steps - 1:
        nextnonterminal = 1.0 - next_done
        nextvalues = next_value
    else:
        nextnonterminal = 1.0 - dones[t + 1]
        nextvalues = values[t + 1]
    
    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

returns = advantages + values
```

### PufferLib (Advanced)
1. **Segmented Storage**: `[segments, horizon]` instead of `[timesteps, envs]`
2. **Advanced GAE**: V-trace with importance clipping for stability
3. **Priority Sampling**: Samples high-advantage trajectories more often
4. **Vectorized Advantage**: C++/CUDA kernel instead of Python loop
5. **LSTM Support**: Built-in recurrent policy handling
6. **Dynamic Segment Allocation**: Fills available slots as episodes complete
7. **Importance Tracking**: Maintains policy update ratios for weighting

---

## 9. CONFIGURATION PARAMETERS

Key hyperparameters from `pufferl.py`:

```python
# Buffer dimensions
batch_size              # Total samples per epoch
bptt_horizon           # Trajectory length (BPTT horizon)
segments = batch_size / bptt_horizon

# Advantage computation
gamma                  # Discount factor
gae_lambda            # GAE exponential mean weight
vtrace_rho_clip       # V-trace importance clip for TD
vtrace_c_clip         # V-trace importance clip for GAE

# Prioritized replay
prio_alpha            # Priority exponent (0=uniform, >0=advantage-weighted)
prio_beta0            # Initial importance sampling correction

# Training
clip_coef             # PPO clip range epsilon
vf_clip_coef         # Value function clip
ent_coef             # Entropy bonus coefficient
vf_coef              # Value loss coefficient

# Optimization
learning_rate
max_grad_norm
update_epochs         # Number of passes over collected data

# Device
device               # 'cuda' or 'cpu'
cpu_offload          # Pin observations in CPU for memory efficiency
```

---

## 10. PERFORMANCE OPTIMIZATIONS

1. **C++/CUDA Advantage Kernel**: Avoids Python loop, ~1000x faster
2. **Shared Memory**: Zero-copy communication between environment workers and trainer
3. **Vectorized Operations**: All operations use PyTorch SIMD where possible
4. **Pre-allocated Buffers**: No dynamic memory allocation during training
5. **Contiguous Tensors**: Efficient cache locality and kernel execution
6. **Priority Sampling**: Focuses computation on high-impact trajectories
7. **CPU Offload**: Observations stored on CPU RAM, values on GPU for memory balance
8. **Torch Compile**: Optional graph compilation for additional speedup
9. **Gradient Accumulation**: Can accumulate minibatches before optimizer step

---

## Summary Table

| Component | Data Structure | Shape | Device | Update Freq |
|-----------|---------------|-------|--------|------------|
| Observations | Tensor | [segments, horizon, *obs_shape] | CPU/GPU | Per step |
| Actions | Tensor | [segments, horizon, *atn_shape] | GPU | Per step |
| Log Probabilities | Tensor | [segments, horizon] | GPU | Per step |
| Rewards | Tensor | [segments, horizon] | GPU | Per step |
| Values | Tensor | [segments, horizon] | GPU | Per step + Training |
| Terminals | Tensor | [segments, horizon] | GPU | Per step |
| Advantages | Tensor | [segments, horizon] | GPU | Per epoch |
| Importance Ratios | Tensor | [segments, horizon] | GPU | Per minibatch |
| Returns | Computed | [segments, horizon] | GPU | Per epoch |
| Episode Indices | Tensor | [total_agents] | GPU | Per episode |
| Episode Lengths | Tensor | [total_agents] | GPU | Per step |

