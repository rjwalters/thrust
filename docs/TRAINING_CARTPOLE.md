# CartPole Training Guide

## Overview

CartPole is a classic control problem that serves as a benchmark for reinforcement learning algorithms. It's a **sequential task** with temporal dependencies, making it fundamentally different from contextual bandits like SimpleBandit.

**Target Performance**: 450+ steps per episode (close to the 500-step maximum)

## Problem Description

### Environment

- **State space**: 4-dimensional continuous
  - Cart position: [-4.8, 4.8]
  - Cart velocity: [-∞, ∞]
  - Pole angle: [-24°, 24°]
  - Pole angular velocity: [-∞, ∞]
- **Action space**: Binary (push left=0, push right=1)
- **Reward**: +1 for every step the pole remains upright
- **Episode termination**:
  - Pole angle > ±12°
  - Cart position > ±2.4
  - Episode length reaches 500 steps
- **Expected performance**: 450+ steps/episode consistently

### Sequential RL vs Contextual Bandit

CartPole is **sequential RL**:
- States have **temporal dependencies** (current state depends on previous action)
- Multi-step episodes (up to 500 steps)
- Credit assignment problem (actions affect future states and rewards)
- GAE is appropriate here

This contrasts with SimpleBandit:
- SimpleBandit has **independent states** (next state is random)
- Single-step episodes
- No credit assignment needed
- GAE adds noise for bandits

## Model Architecture

### Network

```rust
MlpPolicy::new(obs_dim: 4, action_dim: 2, hidden_dim: 256)
```

- **Input**: 4-dimensional observation (cart position, velocity, pole angle, angular velocity)
- **Hidden layer**: 256 units (validated through hyperparameter optimization)
- **Output**: 2-dimensional action logits (left/right)
- **Value head**: Single scalar value estimate

### Architecture Choice

Through hyperparameter optimization, we found that **hidden_dim=256** provides the best stability:
- **64 units**: Too small, unstable training
- **128 units**: Better but still prone to entropy collapse
- **256 units**: Optimal balance of capacity and stability
- **512+ units**: Unnecessary for this problem, slower training

The larger network (256 vs 64) provides:
- Better value function approximation
- More stable training over 5M steps
- Resistance to entropy collapse
- Smoother learning curves

## Training Configuration

### PPO Hyperparameters (Validated)

```rust
PPOConfig::new()
    .learning_rate(0.000247)  // ~4x lower than default
    .n_epochs(20)              // 2x more training per update
    .batch_size(256)           // 2x larger batches
    .gamma(0.9717)             // Slightly lower discount
    .gae_lambda(0.95)          // Standard GAE parameter
    .clip_range(0.2)           // Standard PPO clipping
    .vf_coef(0.5)              // Standard value function coefficient
    .ent_coef(0.0151)          // 15x higher than default (critical!)
    .max_grad_norm(0.5)        // Standard gradient clipping
```

### Critical Configuration: Why GAE Works Here

Unlike SimpleBandit, CartPole **requires GAE**:

```
δ_t = r_t + γ*V_{t+1} - V_t
A_t = δ_t + γ*λ*A_{t+1}
```

The `V_{t+1}` term bootstraps from the **next state**, which makes sense because:
- Next state **depends on** current state and action
- Pole angle at t+1 is determined by pole angle and velocity at t
- V_{t+1} provides useful signal about long-term consequences

**Why this differs from SimpleBandit:**
- SimpleBandit: next state is random → bootstrapping adds noise
- CartPole: next state is deterministic → bootstrapping provides signal

### Hyperparameter Optimization Process

We used Bayesian optimization with 12 trials, each running for 3.5M steps:

**Search space:**
- `learning_rate`: [1e-5, 1e-2] (log scale)
- `hidden_dim`: [64, 128, 256, 512]
- `n_epochs`: [4, 10, 20]
- `batch_size`: [64, 128, 256]
- `ent_coef`: [0.0, 0.1]
- `gamma`: [0.95, 0.999]

**Optimization metric:**
- Average episode length over final 100 episodes
- Must survive to 3.5M steps (70% of training)

**Best trial results:**
- **Performance**: 445.3 steps/episode @ 3.5M steps
- **Stability**: No entropy collapse throughout training
- **Configuration**: See validated hyperparameters above

### Key Insights from Optimization

1. **Entropy coefficient is critical**
   - Default (0.001): Entropy collapse at ~60% of training
   - Optimized (0.0151): Stable throughout 3.5M+ steps
   - 15x increase prevents premature convergence

2. **Larger networks are more stable**
   - 64 units: Frequent entropy collapse
   - 256 units: Stable training
   - Hypothesis: Better value function approximation

3. **Lower learning rate with more epochs**
   - Default: lr=0.001, epochs=10
   - Optimized: lr=0.000247, epochs=20
   - More conservative updates = better stability

4. **Slightly lower gamma**
   - Default: 0.99
   - Optimized: 0.9717
   - Less emphasis on distant future = faster learning

## Training Results

### Performance Progression

```
Training Steps    Avg Episode Length    Notes
--------------    ------------------    -----
0-100K           50-150                Initial exploration
100K-500K        150-300               Learning basic control
500K-1M          300-400               Refining policy
1M-2M            400-450               Near-optimal performance
2M-3.5M          445-450               Stable plateau
```

### Common Issues and Solutions

#### Issue 1: Entropy Collapse

**Symptom**: Policy entropy drops below 0.05, training stops
**Cause**: Premature convergence to deterministic policy
**Solution**:
- Increase `ent_coef` from 0.001 to 0.0151 (15x higher)
- Use larger network (256 vs 64 hidden units)
- Lower learning rate with more epochs

**Detection**: Built-in entropy monitoring in `src/train/ppo/trainer.rs:222-255`

#### Issue 2: Training Instability

**Symptom**: Performance degrades after initial learning
**Cause**: Learning rate too high, overfitting
**Solution**:
- Lower `learning_rate` from 0.001 to 0.000247
- Increase `n_epochs` from 10 to 20
- Increase `batch_size` from 64 to 256

#### Issue 3: Slow Learning

**Symptom**: Taking >2M steps to reach 400+ performance
**Cause**: Insufficient exploration, poor credit assignment
**Solution**:
- Verify GAE is enabled (`gamma=0.9717, gae_lambda=0.95`)
- Increase entropy coefficient
- Check value function is learning (monitor `value_loss`)

## Best Practices

### When to Use These Hyperparameters

**Use validated CartPole config** for:
- Sequential control tasks
- Continuous state spaces
- Short episodes (100-500 steps)
- Dense rewards

**Modify for:**
- Sparse rewards: Increase `gamma` (0.99+)
- Long episodes: Increase `gamma` and `gae_lambda`
- Large action spaces: Increase `ent_coef` further
- Complex dynamics: Increase `hidden_dim` (512+)

### General Guidelines

1. **Start with SimpleBandit to verify correctness**
   - If SimpleBandit fails, you have a bug
   - If SimpleBandit passes, proceed to CartPole

2. **Use GAE for sequential tasks**
   - CartPole benefits from temporal credit assignment
   - GAE helps smooth advantage estimates

3. **Monitor entropy throughout training**
   - Healthy entropy: 0.3-0.7 for CartPole
   - Low entropy (<0.05): Premature convergence
   - High entropy (>0.9): Insufficient learning

4. **Run hyperparameter optimization for new tasks**
   - Don't assume default hyperparameters work
   - Optimize for both performance AND stability
   - Test over full training duration (5M steps)

5. **Test for entropy collapse**
   - Run multiple seeds
   - Check performance at 70%+ of training
   - Entropy collapse often happens late in training

## Usage

### Run CartPole Training

```bash
# CPU training (slow)
cargo run --example train_cartpole_best --release

# GPU training (fast)
# See docs/GPU_SETUP.md for GPU configuration
ssh your-gpu-server 'cd thrust && \
  source venv/bin/activate && \
  export LIBTORCH_USE_PYTORCH=1 && \
  cargo +nightly run --example train_cartpole_best --release'
```

Expected training time:
- **CPU**: ~2-3 hours for 5M steps
- **GPU (NVIDIA L4)**: ~20-30 minutes for 5M steps

### Run Hyperparameter Optimization

```bash
cargo run --example optimize_cartpole --release -- --trials 12
```

This will:
1. Run 12 Bayesian optimization trials
2. Each trial trains for 3.5M steps (70% of full training)
3. Save results to `cartpole_optimization_results.json`
4. Print best hyperparameters

### Verify Correctness

CartPole should reach **400+ steps/episode** by 1M training steps:
- If not, check hyperparameters
- Verify GAE is enabled (`gamma>0, gae_lambda>0`)
- Check entropy is healthy (0.3-0.7)
- Monitor for entropy collapse

## Related Files

- **Environment**: `src/env/cartpole.rs`
- **Training script**: `examples/train_cartpole_best.rs`
- **Optimization**: `examples/optimize_cartpole.rs`
- **PPO implementation**: `src/train/ppo/trainer.rs`
- **GAE implementation**: `src/buffer/rollout/gae.rs`

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [GAE Paper](https://arxiv.org/abs/1506.02438) - Schulman et al., 2016
- [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) - OpenAI Gym

## Lessons Learned

### Key Takeaways

1. **Entropy coefficient is more important than expected**
   - Default value (0.001) causes entropy collapse
   - 15x increase (0.0151) provides stable training
   - This is problem-specific, not a universal value

2. **Network size affects stability**
   - Larger networks (256 units) are more stable
   - Better value function approximation prevents collapse
   - Diminishing returns beyond 256 for CartPole

3. **Conservative updates work better**
   - Lower learning rate with more epochs
   - Larger batch sizes
   - More stable long-term training

4. **Test at 70%+ of training**
   - Entropy collapse often happens late
   - Early performance doesn't guarantee stability
   - Optimize for full training duration

5. **GAE is crucial for sequential tasks**
   - CartPole performance drops without GAE
   - Temporal credit assignment is essential
   - Don't disable GAE for sequential RL

### Comparison to SimpleBandit

| Aspect | SimpleBandit | CartPole |
|--------|-------------|----------|
| **Problem type** | Contextual bandit | Sequential RL |
| **State dependencies** | Independent | Temporal |
| **GAE** | Harmful (adds noise) | Essential (provides signal) |
| **Gamma** | 0.0 | 0.9717 |
| **GAE Lambda** | 0.0 | 0.95 |
| **Entropy coefficient** | 0.1 | 0.0151 |
| **Network size** | 64 (sufficient) | 256 (optimal) |
| **Training stability** | Fast convergence | Requires tuning |

This comparison highlights the importance of matching algorithm configuration to problem structure.
