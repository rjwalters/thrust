# 🤖 Agent Prompt for Thrust Development

**Repository:** Thrust - High-performance Reinforcement Learning in Rust + CUDA
**Status:** Phase 1, Week 1 - Foundation (15% complete)
**Your Mission:** Help build the fastest RL library in Rust

---

## 📋 Context

You are working on **Thrust**, a modern reinforcement learning library written in Rust that aims to be 3-6x faster than Python implementations like PufferLib. The project is inspired by PufferLib but built from scratch in Rust with CUDA acceleration.

### Project Structure
```
thrust/
├── src/
│   ├── env/           # RL environments (cartpole, snake, etc.)
│   ├── policy/        # Neural network policies (tch-rs)
│   ├── buffer/        # Experience replay buffers
│   ├── train/         # Training algorithms (PPO)
│   └── utils/         # Helper functions
├── docs/              # Additional documentation
└── .github/           # CI/CD and templates
```

### Tech Stack
- **Language:** Rust (edition 2021, MSRV 1.75)
- **Neural Networks:** tch-rs (PyTorch bindings) - currently commented out until needed
- **Async Runtime:** Tokio for environment vectorization
- **Parallelism:** Rayon for data parallelism
- **CUDA:** Custom kernels for advantage computation

---

## 🎯 Current Phase: Phase 1 - Foundation

**Goal:** Get PPO training working on CartPole environment

**Week 1 Priority Tasks:**
1. ✅ Project structure (DONE)
2. ✅ CI/CD setup (DONE)
3. 🔄 CartPole environment implementation (IN PROGRESS - HELP NEEDED)
4. ⏳ Experience buffer (NEXT)
5. ⏳ PPO training loop (NEXT)

---

## 🚀 Your First Task: Implement CartPole Environment

### What to Build

The CartPole environment is a classic RL benchmark where a pole is balanced on a cart. We need a **pure Rust implementation** of the physics simulation.

**File:** `src/env/cartpole.rs`

**Requirements:**
1. Implement the `Environment` trait from `src/env/mod.rs`
2. Physics simulation following OpenAI Gym CartPole-v1 spec:
   - State: [x, x_dot, theta, theta_dot] (4D continuous)
   - Actions: 0 (left) or 1 (right) (discrete)
   - Reward: +1 for each timestep the pole stays upright
   - Done: When pole angle > 12° or cart position > 2.4
3. Episode length: Max 500 steps
4. Success criteria: Average reward > 195 over 100 episodes

### Implementation Guide

```rust
// src/env/cartpole.rs

use anyhow::Result;
use rand::Rng;
use crate::env::{Environment, StepResult, SpaceInfo, SpaceType, StepInfo};

/// CartPole-v1 environment
///
/// A pole is attached to a cart moving along a frictionless track.
/// The goal is to balance the pole by applying forces to the cart.
pub struct CartPole {
    // State variables
    x: f32,           // Cart position
    x_dot: f32,       // Cart velocity
    theta: f32,       // Pole angle (radians)
    theta_dot: f32,   // Pole angular velocity

    // Episode tracking
    steps: usize,
    max_steps: usize,

    // Physics constants
    gravity: f32,
    mass_cart: f32,
    mass_pole: f32,
    total_mass: f32,
    length: f32,      // Half-length of pole
    pole_mass_length: f32,
    force_mag: f32,
    tau: f32,         // Time step

    // Thresholds
    theta_threshold: f32,
    x_threshold: f32,
}

impl CartPole {
    pub fn new() -> Self {
        // TODO: Initialize with default physics constants
        // Reference: OpenAI Gym CartPole-v1
        todo!()
    }

    fn reset_state(&mut self) {
        // Reset to random initial state
        // Small random perturbations around equilibrium
        todo!()
    }

    fn physics_step(&mut self, action: i64) {
        // Implement Euler integration for cart-pole dynamics
        // See: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        todo!()
    }

    fn is_done(&self) -> bool {
        // Check termination conditions
        todo!()
    }

    fn get_observation(&self) -> Vec<f32> {
        // Return [x, x_dot, theta, theta_dot]
        todo!()
    }
}

impl Environment for CartPole {
    type Observation = Vec<f32>;
    type Action = i64;

    fn reset(&mut self) -> Result<Self::Observation> {
        self.reset_state();
        self.steps = 0;
        Ok(self.get_observation())
    }

    fn step(&mut self, action: Self::Action) -> Result<StepResult<Self::Observation>> {
        // 1. Apply action (physics step)
        // 2. Update state
        // 3. Calculate reward
        // 4. Check if done
        // 5. Increment step counter
        todo!()
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo {
            shape: vec![4],
            dtype: SpaceType::Continuous,
        }
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo {
            shape: vec![],
            dtype: SpaceType::Discrete(2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartpole_init() {
        let env = CartPole::new();
        // Test initialization
    }

    #[test]
    fn test_cartpole_reset() {
        let mut env = CartPole::new();
        let obs = env.reset().unwrap();
        assert_eq!(obs.len(), 4);
    }

    #[test]
    fn test_cartpole_step() {
        let mut env = CartPole::new();
        env.reset().unwrap();
        let result = env.step(0).unwrap();
        assert_eq!(result.observation.len(), 4);
    }

    #[test]
    fn test_cartpole_termination() {
        // Test that environment terminates correctly
    }

    #[test]
    fn test_cartpole_rewards() {
        // Test reward calculation
    }
}
```

### Physics Reference

The cart-pole dynamics follow these equations:

```
Force: F = action * force_mag (10N left or right)

Acceleration:
temp = (F + pole_mass_length * theta_dot^2 * sin(theta)) / total_mass
theta_acc = (gravity * sin(theta) - cos(theta) * temp) /
            (length * (4/3 - mass_pole * cos(theta)^2 / total_mass))
x_acc = temp - pole_mass_length * theta_acc * cos(theta) / total_mass

Update (Euler integration):
x_dot += tau * x_acc
x += tau * x_dot
theta_dot += tau * theta_acc
theta += tau * theta_dot
```

**Constants (from Gym CartPole-v1):**
- `gravity = 9.8`
- `mass_cart = 1.0`
- `mass_pole = 0.1`
- `length = 0.5` (half-length of pole)
- `force_mag = 10.0`
- `tau = 0.02` (timestep)
- `theta_threshold = 12° * 2π/360 = 0.2094 rad`
- `x_threshold = 2.4`

---

## 📝 Development Guidelines

### Code Style
- Run `make fmt` before committing
- Run `make clippy` to catch warnings
- Run `make test` to verify tests pass
- Use `make ci` for full local CI check

### Commit Messages
Follow conventional commits:
```bash
feat(env): implement CartPole physics simulation
test(env): add CartPole unit tests
docs(env): document CartPole implementation
```

### Testing
- Add unit tests for each function
- Test edge cases (termination, boundaries)
- Add benchmark tests for performance
- Target: 100k+ steps/second

### Documentation
- Document all public functions
- Add examples in doc comments
- Explain physics equations in comments
- Reference OpenAI Gym for compatibility

---

## 🔍 Reference Implementation

**OpenAI Gym CartPole:**
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

**Key differences in our version:**
- Pure Rust (no Python/NumPy)
- Single-precision floats (f32) for performance
- Direct buffer writes (future: for zero-copy)
- Designed for vectorization (future: parallel envs)

---

## ✅ Success Criteria

Your CartPole implementation is complete when:

1. ✅ All unit tests pass
2. ✅ `make clippy` shows zero warnings
3. ✅ Documentation is complete
4. ✅ Physics matches Gym CartPole-v1 behavior
5. ✅ Performance: >100k steps/second (single env)
6. ✅ Can run full episodes without panicking
7. ✅ Reproducible with fixed random seed

---

## 🚀 After CartPole

Once CartPole is done, next tasks are:

1. **Experience Buffer** (`src/buffer/rollout.rs`)
   - Store trajectories
   - Compute GAE (Generalized Advantage Estimation)
   - Efficient sampling

2. **Policy Wrapper** (`src/policy/mlp.rs`)
   - Enable tch-rs in Cargo.toml
   - Create simple MLP policy
   - Forward pass + action sampling

3. **PPO Training Loop** (`src/train/ppo.rs`)
   - Collect rollouts
   - Compute losses
   - Update policy
   - Log metrics

---

## 📚 Helpful Resources

### Rust Resources
- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

### RL Resources
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [CartPole Problem](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

### Project Resources
- `WORKPLAN.md` - Full 20-week roadmap
- `CONTRIBUTING.md` - Development guidelines
- `CI_SETUP.md` - CI/CD documentation
- Reference: `/Users/rwalters/GitHub/PufferLib` - Original Python implementation

---

## 💡 Tips for Success

1. **Start Simple:** Get basic physics working first, optimize later
2. **Test Often:** Run tests after each small change
3. **Use Clippy:** It catches many common mistakes
4. **Ask Questions:** Check existing code in PufferLib if stuck
5. **Benchmark:** Use `cargo bench` to verify performance
6. **Document:** Write docs as you code, not after

---

## 🎯 Your Mission

**Immediate Goal:** Implement CartPole environment that can balance a pole

**Why It Matters:** CartPole is the foundation for testing all our training infrastructure. Once this works, we can add the policy, buffer, and training loop to create a complete RL system.

**Impact:** You're building the first environment in what will become the fastest RL library in Rust! 🚀

---

## 📞 Getting Help

- **Documentation:** Check `docs/` directory
- **Code Examples:** See `src/env/mod.rs` for trait definition
- **Reference:** PufferLib at `/Users/rwalters/GitHub/PufferLib`
- **Questions:** Open a GitHub Discussion (coming soon)

---

## 🎊 Ready to Build!

You have everything you need:
- ✅ Clear task definition
- ✅ Code template
- ✅ Physics equations
- ✅ Test cases
- ✅ Success criteria
- ✅ References

**Let's build something amazing! 🚀**

---

**Current Working Directory:** `/Users/rwalters/GitHub/thrust`

**Commands to start:**
```bash
# Create the CartPole file
touch src/env/cartpole.rs

# Add to src/env/mod.rs
echo "pub mod cartpole;" >> src/env/mod.rs

# Open in editor
code src/env/cartpole.rs

# Verify it compiles
make check

# Run tests
make test

# Full CI
make ci
```

---

*"Give your agents some thrust" 🚀*

**Built with 🦀 Rust and ❤️ for reinforcement learning**
