# SimpleBandit Training Guide

## Overview

SimpleBandit is a trivial test environment designed to validate PPO implementation correctness. It's a **contextual bandit** problem, not a sequential reinforcement learning task.

**Key Insight**: SimpleBandit revealed that GAE (Generalized Advantage Estimation) is fundamentally inappropriate for contextual bandits, requiring us to adapt our training configuration.

## Problem Description

### Environment

- **State space**: Binary (0 or 1)
- **Action space**: Binary (0 or 1)
- **Optimal policy**: `action = state` (deterministic)
- **Reward**: `1.0` if correct, `0.0` if wrong
- **Episode length**: 1 step per episode
- **Expected performance**: 100% success rate

### Contextual Bandit vs Sequential RL

SimpleBandit is a **contextual bandit**:
- Each state is **independent** (next state is random)
- No temporal dependencies
- Single-step episodes
- No credit assignment problem

This differs from sequential RL tasks like CartPole:
- States have **temporal dependencies** (pole angle depends on previous actions)
- Multi-step episodes
- Credit assignment across time steps
- Future actions affect future rewards

## Model Architecture

### Network

```rust
MlpPolicy::new(obs_dim: 1, action_dim: 2, hidden_dim: 64)
```

- **Input**: 1-dimensional observation (state ∈ {0, 1})
- **Hidden layer**: 64 units (small network for trivial task)
- **Output**: 2-dimensional action logits
- **Value head**: Single scalar value estimate

### Architecture Choice

SimpleBandit is so trivial that network architecture doesn't matter much. We use a small MLP (64 hidden units) for:
- Fast training
- Minimal compute
- Debugging simplicity

## Training Configuration

### PPO Hyperparameters

```rust
PPOConfig::new()
    .learning_rate(0.001)
    .n_epochs(10)
    .batch_size(64)
    .gamma(0.0)           // ⚠️ CRITICAL: No discounting for contextual bandits
    .gae_lambda(0.0)      // ⚠️ CRITICAL: No GAE bootstrapping
    .clip_range(0.2)
    .vf_coef(0.5)
    .ent_coef(0.1)        // Higher entropy to encourage exploration
    .max_grad_norm(0.5)
```

### Critical Configuration: gamma=0.0, gae_lambda=0.0

**Why this matters:**

Standard PPO uses GAE with `gamma=0.99` and `gae_lambda=0.95`:
```
δ_t = r_t + γ*V_{t+1} - V_t
A_t = δ_t + γ*λ*A_{t+1}
```

The `V_{t+1}` term bootstraps from the **next state's value**. For sequential tasks, this makes sense because:
- Next state depends on current state and action
- V_{t+1} provides useful signal about long-term consequences

**But for SimpleBandit:**
- Next state is **random** (independent of current state)
- V_{t+1} is a random value
- Bootstrapping from random values adds **noise**, not signal

**Solution:** Use simple advantages
```
gamma=0.0, gae_lambda=0.0  →  A = r - V
```

This gives us clean advantage estimates without noisy bootstrapping.

## Training Results

### With GAE (gamma=0.99, gae_lambda=0.95) ❌

```
Success Rate: 50.2% (stuck at random chance)
Policy: [0.51, 0.49] (nearly uniform)
```

**Analysis:**
- GAE adds noise from random future states
- Advantages are corrupted
- Learning signal is destroyed
- Policy cannot converge

### Without GAE (gamma=0.0, gae_lambda=0.0) ✅

```
Update 0:  Success Rate: 47.8%
Update 10: Success Rate: 84.0%
Policy: [0.992, 0.008] (nearly deterministic)
```

**Analysis:**
- Simple advantages: A = r - V
- Clean learning signal
- Policy converges to near-optimal
- Entropy collapse is expected (deterministic optimal policy)

## Best Practices

### When to Use GAE

**Use GAE** (`gamma=0.99, gae_lambda=0.95`) for:
- Sequential tasks (CartPole, Atari, etc.)
- Multi-step episodes
- Temporal dependencies
- Credit assignment problems

**Skip GAE** (`gamma=0.0, gae_lambda=0.0`) for:
- Contextual bandits
- Independent states
- Single-step episodes
- No temporal structure

### General Guidelines

1. **Understand your problem type**
   - Is it sequential RL or a contextual bandit?
   - Do states have temporal dependencies?

2. **Match algorithm to problem**
   - GAE for sequential tasks
   - Simple advantages for bandits

3. **Test on trivial tasks first**
   - SimpleBandit validates PPO correctness
   - If SimpleBandit fails, you have a bug
   - If SimpleBandit passes, move to harder tasks

4. **Monitor entropy**
   - SimpleBandit should converge to low entropy (deterministic)
   - Sequential tasks should maintain some entropy for exploration

## Usage

### Run SimpleBandit Training

```bash
cargo run --example train_simple_bandit --release
```

Expected output:
```
Update 1/125 | Success Rate: 47.8% | Loss: -0.033 | Entropy: 0.690
Update 11/125 | Success Rate: 84.0% | Loss: -0.012 | Entropy: 0.026
⚠️  Low entropy detected: 0.0425 (count: 3/3)
🚨 Training stopped: Entropy collapse detected!
```

The entropy collapse is **expected** - SimpleBandit has a deterministic optimal policy.

### Verify Correctness

SimpleBandit should reach **>80% success rate** within 10-20 updates. If not:
1. Check GAE configuration (`gamma=0.0, gae_lambda=0.0`)
2. Verify advantage computation
3. Check for bugs in policy gradient implementation

## Related Files

- **Environment**: `src/env/simple_bandit.rs`
- **Training script**: `examples/train_simple_bandit.rs`
- **PPO implementation**: `src/train/ppo/trainer.rs`
- **GAE implementation**: `src/buffer/rollout/gae.rs`

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [GAE Paper](https://arxiv.org/abs/1506.02438) - Schulman et al., 2016
- Contextual bandits are covered in Sutton & Barto, Chapter 2

## Lessons Learned

**Key takeaway**: Not all PPO techniques apply to all problems. GAE is designed for sequential RL with temporal dependencies. Contextual bandits require simpler advantage estimation.

This finding highlights the importance of:
- Testing on diverse problem types
- Understanding algorithm assumptions
- Matching techniques to problem structure
