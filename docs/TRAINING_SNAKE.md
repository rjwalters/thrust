# Snake Multi-Agent Training Guide

## Overview

Snake is a competitive multi-agent reinforcement learning environment where 4 snakes compete for food on a 20x20 grid. Unlike SimpleBandit and CartPole, Snake uses **visual observations** processed by a CNN policy, making it a significantly more complex learning task.

**Key Characteristics**:
- **Sequential RL** with temporal dependencies (like CartPole)
- **Multi-agent self-play** with competitive dynamics
- **Visual observations** requiring spatial feature extraction
- **Sparse rewards** (food collection events)

## Problem Description

### Environment

- **Grid size**: 20x20 (configurable)
- **Agents**: 4 snakes per environment (multi-agent self-play)
- **State space**: 5-channel spatial observation (20×20×5 = 2,000 dimensional)
  - Channel 0: Own snake body
  - Channel 1: Own snake head
  - Channel 2: Other snakes (competitors)
  - Channel 3: Food locations
  - Channel 4: Walls/boundaries
- **Action space**: Discrete 4-way movement (up, down, left, right)
- **Reward structure**:
  - +1 for eating food (snake grows)
  - -1 for death (collision with wall, self, or others)
  - Small time penalty to encourage efficiency
- **Episode termination**: When all snakes die
- **Training goal**: Learn to collect food while avoiding collisions

### Multi-Agent vs Single-Agent

Snake uses **self-play** training:
- 4 snakes compete in each environment
- Each agent shares the same policy network
- All agents receive the same reward (cooperative credit assignment)
- Agents learn both to collect food AND compete with others
- Emergent behaviors: territorial control, food competition, collision avoidance

This differs from single-agent tasks:
- SimpleBandit: Single agent, independent states
- CartPole: Single agent, continuous control
- Snake: Multi-agent, competitive dynamics

## Model Architecture

### Network: SnakeCNN

```rust
SnakeCNN::new(
    grid_size: 20,
    input_channels: 5,
)
```

**Architecture**:
```
Input: [batch, 5, 20, 20]  (5-channel grid observation)
    ↓
Conv2D(5 → 32, kernel=3, padding=1) + ReLU
    ↓
Conv2D(32 → 64, kernel=3, padding=1) + ReLU
    ↓
Conv2D(64 → 64, kernel=3, padding=1) + ReLU
    ↓
Flatten: [batch, 64 * 20 * 20] = [batch, 25,600]
    ↓
Linear(25,600 → 256) + ReLU  (common features)
    ↓
    ├─→ Policy head: Linear(256 → 4)  (action logits)
    └─→ Value head: Linear(256 → 1)   (state value)
```

**Key design choices**:
- **Padding=1**: Preserves spatial dimensions after convolutions
- **3 conv layers**: Extract hierarchical spatial features
  - Layer 1: Basic patterns (snake segments, food)
  - Layer 2: Mid-level features (snake bodies, clusters)
  - Layer 3: High-level features (strategic positions, threats)
- **Large feature vector**: 25,600 dimensions before FC layer
- **Shared features**: Common representation for both policy and value
- **Compact action space**: Only 4 outputs (vs 25,600 input features)

### Why CNN Instead of MLP?

**Spatial structure matters**:
- Snake game has strong spatial relationships
- Snake head position relative to food matters
- Distance to walls matters
- Other snakes' positions matter

**CNN advantages**:
- **Translation invariance**: Food at (5,5) vs (10,10) uses same features
- **Local receptive fields**: Snake head sees nearby threats
- **Parameter efficiency**: ~100K parameters vs millions for MLP
- **Hierarchical features**: Low-level (edges) to high-level (strategies)

**Comparison**:
- SimpleBandit MLP: 64 hidden units, ~130 parameters
- CartPole MLP: 256 hidden units, ~1K parameters
- Snake CNN: 32→64→64 filters, ~100K parameters

The CNN is necessary because visual observations require spatial processing.

## Training Configuration

### Current Hyperparameters

```rust
Args {
    num_envs: 16,                  // Parallel environments
    num_agents: 4,                 // Agents per environment
    grid_width: 20,
    grid_height: 20,
    steps_per_rollout: 512,
    epochs: 1000,
    learning_rate: 3e-4,           // Standard PPO learning rate
    gae_lambda: 0.95,              // Standard GAE
    gamma: 0.99,                   // Standard discount factor
    clip_param: 0.2,               // Standard PPO clipping
    value_coef: 0.5,               // Standard value coefficient
    entropy_coef: 0.01,            // Standard entropy coefficient
    ppo_epochs: 4,                 // Mini-epochs per update
    minibatch_size: 64,
    save_interval: 10,
}
```

### Configuration Rationale

**Parallel training**:
- 16 environments × 4 agents = 64 parallel experiences per step
- 512 steps per rollout = 32,768 samples per update
- High sample efficiency for visual RL

**Standard PPO parameters**:
- `learning_rate=3e-4`: Conservative for visual tasks
- `gamma=0.99`: Long-term planning (episodes can be 100+ steps)
- `gae_lambda=0.95`: Smooth advantage estimates
- `clip_param=0.2`: Standard PPO trust region

**Multi-agent credit assignment**:
- All agents receive **full reward** (not divided by 4)
- Rationale: All agents contribute to the outcome
- Encourages cooperative learning despite competitive dynamics

### Why GAE Works Here

Like CartPole, Snake is **sequential RL** with temporal dependencies:

```
δ_t = r_t + γ*V_{t+1} - V_t
A_t = δ_t + γ*λ*A_{t+1}
```

**Temporal dependencies**:
- Snake head position at t+1 depends on action at t
- Food collection at t+10 results from navigation actions at t through t+9
- Death at t+5 may result from poor positioning at t

**Why GAE helps**:
- Bootstrapping from V_{t+1} provides signal about long-term consequences
- GAE smooths noisy rewards (food collection is sparse)
- Credit assignment: Which action led to food collection?

**Contrast with SimpleBandit**:
- SimpleBandit: Independent states → GAE adds noise
- Snake: Sequential states → GAE provides signal

## Training Results

### Expected Performance Progression

**Note**: Snake training is still being optimized. These are preliminary observations.

```
Training Steps    Mean Reward    Notes
--------------    -----------    -----
0-50K             -1 to 0        Random exploration, mostly deaths
50K-200K          0 to +2        Learning basic movement, occasional food
200K-1M           +2 to +5       Consistent food collection, better survival
1M+               +5 to +10      Strategic play, competitive behaviors
```

**Typical learning progression**:
1. **Random phase** (0-50K): Snake wanders randomly, dies quickly
2. **Survival phase** (50K-200K): Learns to avoid walls and self-collision
3. **Food collection phase** (200K-1M): Starts eating food consistently
4. **Strategic phase** (1M+): Develops territorial control, competition tactics

### Common Issues

#### Issue 1: Slow Learning on Visual Tasks

**Symptom**: Policy stays random for 100K+ steps

**Cause**: Visual observations are high-dimensional and complex

**Solutions**:
- Ensure CNN architecture is appropriate (3 conv layers minimum)
- Verify input normalization (pixel values in [0, 1])
- Check reward signal is reaching the policy (monitor value_loss)
- Increase batch size for more stable gradients
- Consider curriculum learning (start with easier grid sizes)

#### Issue 2: Entropy Collapse

**Symptom**: Policy becomes deterministic too early, gets stuck

**Cause**: Insufficient exploration in sparse reward environment

**Solutions**:
- Increase `entropy_coef` from 0.01 to 0.02 or higher
- Monitor entropy throughout training (should stay > 0.3)
- Consider intrinsic motivation or curiosity bonuses
- Verify advantage estimation is not too noisy

#### Issue 3: Multi-Agent Instability

**Symptom**: Training is unstable, reward variance is high

**Cause**: Self-play dynamics can be unstable

**Solutions**:
- Use larger batch sizes (more samples per update)
- Lower learning rate for more conservative updates
- Consider opponent sampling (mix old and new policies)
- Verify all agents are using the same policy weights

#### Issue 4: GPU Memory Issues

**Symptom**: Out of memory errors during training

**Cause**: CNN on 20x20 grids with 64 parallel samples is memory-intensive

**Solutions**:
- Reduce `minibatch_size` from 64 to 32
- Reduce `num_envs` from 16 to 8
- Use mixed precision training (fp16)
- Clear CUDA cache between updates

## Best Practices

### When to Use Visual RL (CNN)

**Use CNN policy** for:
- Grid-based games (Snake, Pac-Man, Breakout)
- Image observations (Atari, robotics vision)
- Spatial relationships matter
- Translation invariance desired

**Use MLP policy** for:
- Low-dimensional state spaces (CartPole)
- Non-visual observations (sensor readings)
- No spatial structure
- Smaller, faster training

### Multi-Agent Training Tips

1. **Share policy weights**: All agents use the same network
   - Enables self-play learning
   - More sample efficient
   - Emergent competitive behaviors

2. **Credit assignment**: Give full reward to all agents
   - Each agent contributes to the outcome
   - Encourages cooperative learning
   - Simpler than individual rewards

3. **Monitor diversity**: Check that agents don't all learn the same strategy
   - Track position distributions
   - Measure action entropy per agent
   - Consider diversity bonuses if needed

4. **Curriculum learning**: Start simple, increase difficulty
   - Begin with fewer agents (2 instead of 4)
   - Start with smaller grids (10x10 → 20x20)
   - Gradually reduce food spawn rate

### Hyperparameter Tuning for Snake

**Priority 1 - Network architecture**:
- Number of conv layers (2-4)
- Number of filters (32, 64, 128)
- FC layer size (128, 256, 512)

**Priority 2 - Exploration**:
- `entropy_coef` (0.01 → 0.05)
- Intrinsic motivation methods
- Curriculum learning approach

**Priority 3 - Sample efficiency**:
- Batch size (32, 64, 128)
- Rollout length (256, 512, 1024)
- Number of parallel environments (8, 16, 32)

**Priority 4 - Stability**:
- Learning rate (1e-4, 3e-4, 1e-3)
- PPO epochs (2, 4, 8)
- Gradient clipping (0.5, 1.0, 5.0)

## Usage

### Run Snake Training

```bash
# CPU training (very slow for CNN)
cargo run --example train_snake_multi --release

# GPU training (recommended)
# See docs/GPU_SETUP.md for GPU configuration
ssh your-gpu-server 'cd thrust && \
  source venv/bin/activate && \
  export LIBTORCH_USE_PYTORCH=1 && \
  cargo +nightly run --example train_snake_multi --release'
```

Expected training time:
- **CPU**: ~10+ hours for 1M steps (not recommended)
- **GPU (NVIDIA L4)**: ~2-3 hours for 1M steps

### Monitor Training Progress

```bash
# On remote GPU server
ssh your-gpu-server 'cd thrust && tail -f training_*.log'
```

Look for:
- Increasing mean reward (should reach +5 by 1M steps)
- Healthy entropy (> 0.3)
- Decreasing value loss (value function learning)
- Stable policy loss

### Save and Load Models

**Checkpoints are saved every 10 epochs** to `models/snake_policy.epochN.safetensors`

```bash
# Load checkpoint for continued training
cargo run --example train_snake_multi --release -- --load models/snake_policy.epoch100.safetensors

# Export for WASM deployment
cargo run --example export_snake --release
```

### Verify Training Quality

Snake training is progressing well if:
- Mean reward > +5 by 1M steps
- Episodes last > 50 steps on average
- Snakes consistently find and eat food
- Collision deaths decrease over time
- Entropy remains > 0.3

## Related Files

- **Environment**: `src/env/snake/environment.rs`
- **CNN Policy**: `src/policy/snake_cnn.rs`
- **Training script**: `examples/train_snake_multi.rs`
- **PPO implementation**: `src/train/ppo/trainer.rs`
- **GAE implementation**: `src/buffer/rollout/gae.rs`

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [GAE Paper](https://arxiv.org/abs/1506.02438) - Schulman et al., 2016
- [Multi-Agent RL Survey](https://arxiv.org/abs/1911.10635) - Zhang et al., 2019
- [Self-Play in Games](https://arxiv.org/abs/1710.03748) - OpenAI Five

## Lessons Learned

### Key Takeaways

1. **Visual RL is sample-intensive**
   - CNN requires 10-100x more samples than MLP
   - Parallel environments are essential (16+ environments)
   - GPU acceleration is critical for practical training times

2. **Multi-agent adds complexity**
   - Self-play dynamics can be unstable
   - Credit assignment is non-trivial
   - Emergent behaviors are exciting but unpredictable

3. **Sparse rewards require patience**
   - Food collection events are rare early in training
   - Value function takes longer to learn
   - Entropy bonuses are more critical

4. **CNN architecture matters**
   - Too shallow: Can't learn spatial features
   - Too deep: Overfits or doesn't train
   - 3 conv layers (32→64→64) is a good starting point

5. **GAE is still appropriate**
   - Snake has temporal dependencies like CartPole
   - GAE helps with sparse reward credit assignment
   - Standard `gamma=0.99, gae_lambda=0.95` works well

### Comparison to Other Tasks

| Aspect | SimpleBandit | CartPole | Snake |
|--------|-------------|----------|-------|
| **Problem type** | Contextual bandit | Sequential control | Sequential multi-agent |
| **Observation** | Binary (0/1) | Vector (4D) | Visual grid (20×20×5) |
| **Policy** | MLP (64 units) | MLP (256 units) | CNN (32→64→64) |
| **Agents** | Single | Single | Multi (4) |
| **Rewards** | Dense | Dense | Sparse |
| **GAE** | Harmful | Essential | Essential |
| **Gamma** | 0.0 | 0.9717 | 0.99 |
| **Entropy coef** | 0.1 | 0.0151 | 0.01 |
| **Training steps** | 50K | 5M | 5M+ |
| **GPU needed** | No | No | Yes |

This progression (bandit → control → visual RL) demonstrates increasing complexity in RL:
1. **SimpleBandit**: Verify algorithm correctness
2. **CartPole**: Learn sequential control
3. **Snake**: Scale to visual observations and multi-agent dynamics

## Future Work

### Planned Improvements

1. **Hyperparameter optimization**
   - Run Bayesian optimization (like CartPole)
   - Test entropy coefficients (0.005 → 0.05)
   - Optimize network architecture (filter sizes, FC layer)

2. **Curriculum learning**
   - Start with 2 agents → 4 agents
   - Start with 10×10 grid → 20×20 grid
   - Gradually increase episode length

3. **Intrinsic motivation**
   - Add curiosity-driven exploration
   - Reward visiting new states
   - Encourage diverse behaviors

4. **Advanced techniques**
   - Population-based training
   - Self-play with opponent pool
   - Asymmetric self-play (mixing old/new policies)

5. **Evaluation metrics**
   - Food collection rate
   - Survival time distribution
   - Spatial coverage (exploration)
   - Head-to-head win rates

### Open Questions

- What is the optimal CNN architecture for 20×20 grids?
- How much does multi-agent self-play help vs single-agent training?
- Can we achieve human-level play with current architecture?
- What entropy coefficient prevents premature convergence?
- How important is curriculum learning vs training from scratch?

These questions will be addressed through systematic experimentation and hyperparameter optimization.
