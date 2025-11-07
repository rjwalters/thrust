# PPO Best Practices and Implementation Details

A curated knowledge base of PPO hyperparameters, implementation details, and research findings.

## Quick Reference: Standard Hyperparameters

### Stable-Baselines3 Defaults
```python
vf_coef = 0.5              # Value function loss coefficient
clip_range = 0.2           # Policy clipping parameter
clip_range_vf = None       # Value clipping (disabled by default)
learning_rate = 3e-4
n_epochs = 10
batch_size = 64
gamma = 0.99
gae_lambda = 0.95
ent_coef = 0.01
max_grad_norm = 0.5
```

**Source**: [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- GitHub: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py

### CleanRL CartPole Hyperparameters
```python
vf_coef = 0.5              # Value function loss coefficient
clip_coef = 0.2            # Policy clipping parameter
clip_vloss = True          # Value clipping enabled
learning_rate = 0.00025    # Lower than SB3 default
num_envs = 4
num_steps = 128
update_epochs = 4
gamma = 0.99
gae_lambda = 0.95
ent_coef = 0.01
max_grad_norm = 0.5
total_timesteps = 500_000
```

**Source**: [CleanRL PPO Documentation](https://docs.cleanrl.dev/rl-algorithms/ppo/)

## Critical Implementation Details

### 1. Value Function Loss Clipping

**Implementation Detail #9** from "The 37 Implementation Details of Proximal Policy Optimization"

**Formula**:
```
L^V = max[(V_θt - V_target)², (clip(V_θt, V_θt-1 - ε, V_θt-1 + ε) - V_target)²]
```

**Research Findings**:
- ❌ **Engstrom et al. (2020)**: No evidence that value clipping helps performance
- ❌ **Andrychowicz et al. (2021)**: Value clipping may **hurt** performance (Decision C13, Figure 43)
- ⚠️ Implemented in some codebases for "high-fidelity reproduction" rather than optimal performance

**Recommendation**: **Disable value function clipping** (`clip_range_vf = None` or `infinity`)

**Source**: [The 37 Implementation Details of PPO (ICLR Blog Track 2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

### 2. Value Function Coefficient (vf_coef)

**What it does**: Weights the value function loss in the total loss calculation
```
total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
```

**Standard Values**:
- Stable-Baselines3: `0.5`
- CleanRL: `0.5`
- Hyperparameter search range: `0.0 - 5.0`

**Common Issues**:
- **Too high (>1.0)**: Value function overtraining
  - Symptoms: Large value loss spikes, unstable explained variance
  - Can cause value function to dominate training
  - Policy learning may be suppressed

- **Too low (<0.1)**: Poor value function learning
  - Symptoms: Explained variance stays near 0
  - Advantages become noisy
  - Slow convergence

**Recommendation**: **Start with 0.5** (standard), only adjust if you see clear value function issues

### 3. Advantage Normalization

**Implementation Detail #7** from "The 37 Implementation Details"

**What it does**: Normalize advantages to zero mean and unit variance
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Why it matters**:
- Ensures balanced positive/negative advantages
- Stabilizes gradient updates
- **Standard practice** in all major PPO implementations

**Common Bugs**:
- ❌ Conditional normalization (only normalize when std > threshold)
- ❌ Skipping mean centering
- ❌ Computing std with wrong dimension

**Recommendation**: **Always normalize** across the entire batch

## Environment-Specific Tuning

### CartPole-v1

**Target Performance**: 450+ steps/episode (near-maximum 500)

**Validated Configuration** (from our hyperparameter optimization):
```rust
learning_rate: 0.000247    // ~4x lower than default (more conservative)
n_epochs: 20               // 2x more training per update
batch_size: 256            // 2x larger batches
hidden_dim: 256            // Larger network for stability
gamma: 0.9717              // Slightly lower discount
gae_lambda: 0.95
clip_range: 0.2
vf_coef: 0.5               // Standard value (not 2.0!)
ent_coef: 0.0151           // 15x higher to prevent entropy collapse
max_grad_norm: 0.5
```

**Key Insights**:
1. **Entropy coefficient is critical**: Default (0.001) causes collapse at ~60% training
2. **Larger networks (256) are more stable** than smaller (64 units)
3. **Lower learning rate with more epochs**: More conservative updates
4. **Slightly lower gamma (0.9717)**: Less emphasis on distant future

**Expected Training Progression**:
```
500K:    ~200 steps/episode
1M:      ~300 steps/episode
2M:      ~350 steps/episode
3M:      ~380 steps/episode
5M:      ~450 steps/episode (target)
```

## Debugging Common Issues

### Issue 1: Explained Variance Unstable or Negative

**Symptoms**:
- ExpVar oscillates wildly (-3.0 to +1.0)
- ExpVar doesn't improve over training
- Large value loss spikes

**Likely Causes**:
1. `vf_coef` too high (>1.0)
2. Value function clipping with wrong parameters
3. Poor advantage computation (GAE issues)

**Solutions**:
- Reduce `vf_coef` to 0.5
- Disable value clipping
- Verify GAE computation matches reference implementations

### Issue 2: Entropy Collapse

**Symptoms**:
- Entropy drops below 0.05
- Policy becomes deterministic too early
- Training stops improving

**Likely Causes**:
1. `ent_coef` too low
2. Learning rate too high
3. Network too small

**Solutions**:
- Increase `ent_coef` (try 10-15x default)
- Reduce learning rate
- Increase network size

### Issue 3: Slow Learning or Plateau

**Symptoms**:
- Performance improves slowly
- Plateaus well below target
- Takes >2M steps to reach 400+ on CartPole

**Likely Causes**:
1. Advantage normalization bugs
2. Poor value function learning
3. Learning rate too low
4. Insufficient exploration

**Solutions**:
- Verify advantage normalization is always enabled
- Check explained variance is improving
- Increase learning rate slightly
- Increase entropy coefficient

## Monitoring Metrics

**Essential metrics to log**:
1. **Policy Loss**: Should be small and stable
2. **Value Loss**: Should decrease over time
3. **Entropy**: Should stay healthy (0.3-0.7 for CartPole)
4. **Explained Variance**: Should approach 1.0 (perfect predictions)
5. **Clip Fraction**: Shows % of updates that hit the clip boundary
6. **Approx KL**: Monitors how much policy changes per update
7. **Episode Length**: Primary performance metric

**Healthy Training Signs**:
- ✅ Explained variance increases toward 1.0
- ✅ Entropy stays in healthy range (not collapsing)
- ✅ Value loss decreases steadily
- ✅ Episode length improves consistently
- ✅ Policy loss remains small

**Warning Signs**:
- ⚠️ ExpVar highly unstable or negative
- ⚠️ Large value loss spikes
- ⚠️ Entropy dropping rapidly
- ⚠️ Performance plateau with no improvement

## References

### Papers
1. **Schulman et al. (2017)**: Proximal Policy Optimization Algorithms
   - https://arxiv.org/abs/1707.06347
   - Original PPO paper

2. **Schulman et al. (2016)**: High-Dimensional Continuous Control Using Generalized Advantage Estimation
   - https://arxiv.org/abs/1506.02438
   - GAE algorithm

3. **Engstrom et al. (2020)**: Implementation Matters in Deep RL
   - Finding: Value clipping doesn't help

4. **Andrychowicz et al. (2021)**: What Matters In On-Policy Reinforcement Learning?
   - Finding: Value clipping may hurt performance

### Implementation Guides
1. **The 37 Implementation Details of Proximal Policy Optimization** (ICLR Blog Track 2022)
   - https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
   - Comprehensive guide to PPO implementation details
   - **Must-read** for anyone implementing PPO

2. **Stable-Baselines3 Documentation**
   - https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
   - Reference implementation in PyTorch

3. **CleanRL Documentation**
   - https://docs.cleanrl.dev/rl-algorithms/ppo/
   - Single-file implementations with research-friendly features

### Code References
- Stable-Baselines3 PPO: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
- CleanRL PPO: https://github.com/vwxyzjn/cleanrl

## Lessons Learned (From Our Experience)

### What Worked
1. ✅ **Always normalize advantages**: Removed conditional logic, always apply normalization
2. ✅ **Disable value clipping**: Research shows it hurts, we confirmed this
3. ✅ **Log explained variance**: Critical for diagnosing value function issues
4. ✅ **Higher entropy coefficient**: Prevents premature convergence

### What Didn't Work
1. ❌ **vf_coef = 2.0**: Caused unstable value function, large loss spikes
2. ❌ **Conditional advantage normalization**: Destroyed learning signal when variance was low
3. ❌ **Value clipping with high vf_coef**: Compounded instability

### Optimal Configuration (For CartPole)
```rust
vf_coef: 0.5              // Standard, not 2.0
clip_range_vf: infinity   // No value clipping
ent_coef: 0.0151          // 15x higher than default
learning_rate: 0.000247   // Conservative
n_epochs: 20
batch_size: 256
hidden_dim: 256
```

**Expected Result**: 450+ steps/episode on CartPole-v1 by 5M timesteps

---

Last Updated: 2025-11-07
Contributors: Claude Code, rwalters
