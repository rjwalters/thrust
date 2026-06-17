//! Pendulum swing-up continuous-control environment.
//!
//! [`PendulumSwingUp`] is the canonical SAC/TD3/DDPG smoke benchmark
//! (issue #139, part of the SAC decomposition #136). It reimplements
//! the classic Gym `Pendulum-v1` task in-tree with no external
//! dependencies and closed-form physics.
//!
//! A rigid pendulum hangs from a fixed pivot. The agent applies a
//! continuous torque at the pivot and is rewarded for swinging the
//! pendulum upright and holding it there with minimal velocity and
//! control effort.
//!
//! - **Action:** `Vec<f32>` of length 1 (torque `u`), clamped to `[-2.0, 2.0]`.
//! - **Observation (3-dim):** `[cos θ, sin θ, θ̇]`. The angle is encoded as
//!   `(cos, sin)` so the observation is continuous across the `±π` wrap.
//! - **Dynamics:** standard `Pendulum-v1` physics with `g = 10`, `m = 1`, `l =
//!   1`, `dt = 0.05`; angular velocity `θ̇` clamped to `[-8.0, 8.0]`.
//! - **Reward:** `-(θ_norm² + 0.1·θ̇² + 0.001·u²)` where `θ_norm` is the angle
//!   wrapped to `[-π, π]`. Reward is always `≤ 0` and is `0` only at the
//!   upright rest state (`θ = 0`, `θ̇ = 0`, `u = 0`).
//! - **Episode length:** 200 steps, then `truncated = true`. The env never sets
//!   `terminated`.
//!
//! # Determinism contract
//!
//! The only source of randomness is the initial angle/velocity drawn at
//! [`Environment::reset`], which is seeded (see
//! [`PendulumSwingUp::with_seed`]). Reset is reproducible: two envs with
//! the same seed produce identical episodes. The dynamics themselves are
//! fully deterministic (no RNG), so [`Environment::restore_state`]
//! followed by [`Environment::step`] reproduces every subsequent
//! [`StepResult`] bit-for-bit. The [`PendulumState`] snapshot captures
//! the simulation state (`theta`, `theta_dot`, `steps`) but not the
//! reset RNG.

use std::f32::consts::PI;

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

/// Gravitational acceleration used by the dynamics.
const GRAVITY: f32 = 10.0;

/// Pendulum mass.
const MASS: f32 = 1.0;

/// Pendulum length.
const LENGTH: f32 = 1.0;

/// Integration timestep.
const DT: f32 = 0.05;

/// Torque clamp range applied to the 1D action input.
const MAX_TORQUE: f32 = 2.0;

/// Angular-velocity clamp range.
const MAX_SPEED: f32 = 8.0;

/// Default episode length cap.
const DEFAULT_MAX_STEPS: usize = 200;

/// Default seed used by [`PendulumSwingUp::new`].
const DEFAULT_SEED: u64 = 0;

/// Snapshot of [`PendulumSwingUp`]'s simulation state.
///
/// The pendulum dynamics are fully deterministic (no RNG), so
/// [`Environment::restore_state`] followed by [`Environment::step`]
/// reproduces every subsequent [`StepResult`] bit-for-bit. The snapshot
/// does *not* capture the reset RNG; restoring then calling
/// [`Environment::reset`] will draw the next seeded initial state, not
/// the one in this snapshot.
#[derive(Debug, Clone)]
pub struct PendulumState {
    /// Pole angle in radians. `0` is upright; `±π` is hanging down.
    pub theta: f32,
    /// Angular velocity in radians per second.
    pub theta_dot: f32,
    /// Step counter for the current episode.
    pub steps: usize,
}

/// Pendulum swing-up task with a continuous torque action.
///
/// See the [module docs](self) for the full observation/reward/dynamics
/// specification and determinism contract.
#[derive(Debug, Clone)]
pub struct PendulumSwingUp {
    /// Pole angle in radians. `0` is upright; `±π` is hanging down.
    theta: f32,

    /// Angular velocity in radians per second.
    theta_dot: f32,

    /// Step counter for the current episode.
    steps: usize,

    /// Maximum number of steps before truncation.
    max_steps: usize,

    /// Seeded RNG driving the initial state at [`Environment::reset`].
    rng: StdRng,
}

impl PendulumSwingUp {
    /// Create a new env with the default seed and episode length.
    pub fn new() -> Self {
        Self::with_seed(DEFAULT_SEED)
    }

    /// Create a new env with a custom reset seed.
    ///
    /// Two envs constructed with the same seed produce identical
    /// episodes (same initial state on every reset).
    pub fn with_seed(seed: u64) -> Self {
        Self::with_seed_and_max_steps(seed, DEFAULT_MAX_STEPS)
    }

    /// Create a new env with a custom reset seed and episode length.
    pub fn with_seed_and_max_steps(seed: u64, max_steps: usize) -> Self {
        let mut env = Self {
            theta: PI,
            theta_dot: 0.0,
            steps: 0,
            max_steps,
            rng: StdRng::seed_from_u64(seed),
        };
        env.reset();
        env
    }

    /// Current pole angle in radians (for inspection / tests).
    pub fn theta(&self) -> f32 {
        self.theta
    }

    /// Current angular velocity (for inspection / tests).
    pub fn theta_dot(&self) -> f32 {
        self.theta_dot
    }

    /// Wrap an angle to the canonical `[-π, π]` range.
    fn angle_normalize(angle: f32) -> f32 {
        // `rem_euclid` maps into `[0, 2π)`; shift to `[-π, π)`.
        ((angle + PI).rem_euclid(2.0 * PI)) - PI
    }
}

impl Default for PendulumSwingUp {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for PendulumSwingUp {
    /// Continuous torque action. Length-1 `Vec<f32>`; values outside
    /// `[-MAX_TORQUE, MAX_TORQUE]` are clamped. An empty vec is treated
    /// as zero torque.
    type Action = Vec<f32>;

    /// Snapshot type. The dynamics are deterministic (no RNG), so
    /// restore + step reproduces subsequent results exactly. The reset
    /// RNG is not captured; see [`PendulumState`].
    type State = PendulumState;

    fn reset(&mut self) {
        // Gym `Pendulum-v1` draws the initial angle uniformly in
        // `[-π, π]` and the initial velocity uniformly in `[-1, 1]`.
        self.theta = self.rng.random_range(-PI..PI);
        self.theta_dot = self.rng.random_range(-1.0..1.0);
        self.steps = 0;
    }

    fn get_observation(&self) -> Vec<f32> {
        vec![self.theta.cos(), self.theta.sin(), self.theta_dot]
    }

    fn step(&mut self, action: Vec<f32>) -> StepResult {
        // Length-1 action; an empty vec is treated as zero torque
        // rather than panicking. Anything beyond index 0 is ignored.
        let raw = action.first().copied().unwrap_or(0.0);
        let torque = raw.clamp(-MAX_TORQUE, MAX_TORQUE);

        // Cost is computed against the *pre-step* state (matching Gym).
        let theta_norm = Self::angle_normalize(self.theta);
        let cost = theta_norm * theta_norm
            + 0.1 * self.theta_dot * self.theta_dot
            + 0.001 * torque * torque;

        // Standard Pendulum-v1 dynamics. `theta` here measures the
        // displacement from upright, so the gravity term uses `sin`.
        let theta_dot_dot = (3.0 * GRAVITY / (2.0 * LENGTH)) * self.theta.sin()
            + (3.0 / (MASS * LENGTH * LENGTH)) * torque;

        self.theta_dot = (self.theta_dot + theta_dot_dot * DT).clamp(-MAX_SPEED, MAX_SPEED);
        self.theta += self.theta_dot * DT;

        self.steps += 1;
        let truncated = self.steps >= self.max_steps;

        StepResult {
            observation: self.get_observation(),
            reward: -cost,
            terminated: false,
            truncated,
            info: StepInfo::default(),
        }
    }

    fn observation_space(&self) -> SpaceInfo {
        // [cos θ, sin θ, θ̇], all continuous.
        SpaceInfo { shape: vec![3], space_type: SpaceType::Box }
    }

    fn action_space(&self) -> SpaceInfo {
        // 1D continuous torque.
        SpaceInfo { shape: vec![1], space_type: SpaceType::Box }
    }

    fn render(&self) -> Vec<u8> {
        Vec::new()
    }

    fn close(&mut self) {}

    fn clone_state(&self) -> PendulumState {
        PendulumState { theta: self.theta, theta_dot: self.theta_dot, steps: self.steps }
    }

    fn restore_state(&mut self, state: &PendulumState) {
        self.theta = state.theta;
        self.theta_dot = state.theta_dot;
        self.steps = state.steps;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_is_length_three_and_unit_circle() {
        let mut env = PendulumSwingUp::new();
        env.reset();

        let obs = env.get_observation();
        assert_eq!(obs.len(), 3, "observation must be 3-dimensional");

        // cos² + sin² == 1 for any angle.
        let unit = obs[0] * obs[0] + obs[1] * obs[1];
        assert!((unit - 1.0).abs() < 1e-5, "cos^2 + sin^2 should equal 1, got {unit}");

        // Step and re-check the invariant.
        let result = env.step(vec![0.5]);
        assert_eq!(result.observation.len(), 3);
        let unit2 = result.observation[0] * result.observation[0]
            + result.observation[1] * result.observation[1];
        assert!((unit2 - 1.0).abs() < 1e-5, "cos^2 + sin^2 should equal 1, got {unit2}");
    }

    #[test]
    fn torque_is_clamped() {
        // Apply huge torque from the upright rest state. With the angle
        // at 0 the gravity term vanishes, so the velocity change is
        // purely from the clamped torque.
        let mut env = PendulumSwingUp::new();
        env.restore_state(&PendulumState { theta: 0.0, theta_dot: 0.0, steps: 0 });

        env.step(vec![1000.0]);
        let max_dot = (3.0 / (MASS * LENGTH * LENGTH)) * MAX_TORQUE * DT;
        assert!(
            env.theta_dot() <= max_dot + 1e-6,
            "velocity change should reflect clamped torque, got {}",
            env.theta_dot()
        );

        // Negative direction.
        let mut env2 = PendulumSwingUp::new();
        env2.restore_state(&PendulumState { theta: 0.0, theta_dot: 0.0, steps: 0 });
        env2.step(vec![-1000.0]);
        assert!(env2.theta_dot() >= -max_dot - 1e-6);
    }

    #[test]
    fn reward_is_nonpositive_and_zero_only_at_upright_rest() {
        let mut env = PendulumSwingUp::new();

        // From the upright rest state, a zero-torque step incurs zero
        // cost (θ = 0, θ̇ = 0, u = 0 -> reward 0).
        env.restore_state(&PendulumState { theta: 0.0, theta_dot: 0.0, steps: 0 });
        let r = env.step(vec![0.0]);
        assert!(r.reward.abs() < 1e-6, "upright rest with no torque should give reward 0");

        // Any non-trivial state or torque yields strictly negative
        // reward, and reward is never positive.
        env.restore_state(&PendulumState { theta: 1.0, theta_dot: 0.0, steps: 0 });
        let r = env.step(vec![0.0]);
        assert!(r.reward < 0.0, "off-upright should give negative reward");

        env.restore_state(&PendulumState { theta: 0.0, theta_dot: 0.0, steps: 0 });
        let r = env.step(vec![1.0]);
        assert!(r.reward < 0.0, "control effort should give negative reward");

        // Sweep a range of states; reward must always be <= 0.
        for &theta in &[-PI, -1.5, -0.3, 0.0, 0.7, 2.0, PI] {
            for &dot in &[-8.0, -1.0, 0.0, 3.0, 8.0] {
                for &u in &[-2.0, 0.0, 1.5] {
                    env.restore_state(&PendulumState { theta, theta_dot: dot, steps: 0 });
                    let r = env.step(vec![u]);
                    assert!(
                        r.reward <= 0.0,
                        "reward must be <= 0 (theta={theta}, dot={dot}, u={u})"
                    );
                }
            }
        }
    }

    #[test]
    fn truncates_after_max_steps() {
        let mut env = PendulumSwingUp::new();
        env.reset();

        for i in 0..(DEFAULT_MAX_STEPS - 1) {
            let r = env.step(vec![0.0]);
            assert!(!r.truncated, "should not truncate before max_steps (step {i})");
            assert!(!r.terminated, "pendulum never terminates");
        }

        let r = env.step(vec![0.0]);
        assert!(r.truncated, "episode should truncate after {DEFAULT_MAX_STEPS} steps");
        assert!(!r.terminated, "truncation is not termination");
    }

    #[test]
    fn clone_restore_round_trips_next_step() {
        let mut env = PendulumSwingUp::new();
        env.reset();
        // Advance a few steps so the snapshot is non-trivial.
        env.step(vec![0.3]);
        env.step(vec![-0.7]);

        let snapshot = env.clone_state();
        let result_a = env.step(vec![1.1]);

        // Restore and replay the same action; results must match exactly.
        env.restore_state(&snapshot);
        let result_b = env.step(vec![1.1]);

        assert_eq!(result_a.observation, result_b.observation, "obs must reproduce bit-for-bit");
        assert_eq!(result_a.reward, result_b.reward, "reward must reproduce bit-for-bit");
        assert_eq!(result_a.truncated, result_b.truncated);
        assert_eq!(result_a.terminated, result_b.terminated);
    }

    #[test]
    fn seeded_reset_is_reproducible() {
        let mut a = PendulumSwingUp::with_seed(42);
        let mut b = PendulumSwingUp::with_seed(42);
        a.reset();
        b.reset();
        assert_eq!(a.get_observation(), b.get_observation(), "same seed -> same initial obs");

        // Different seeds should (essentially always) differ.
        let mut c = PendulumSwingUp::with_seed(7);
        c.reset();
        assert_ne!(
            a.get_observation(),
            c.get_observation(),
            "different seeds should give different initial states"
        );

        // Determinism holds across a full rollout from a fresh reset.
        a.reset();
        b.reset();
        for _ in 0..10 {
            let ra = a.step(vec![0.5]);
            let rb = b.step(vec![0.5]);
            assert_eq!(ra.observation, rb.observation);
            assert_eq!(ra.reward, rb.reward);
        }
    }

    #[test]
    fn action_space_is_box() {
        let env = PendulumSwingUp::new();
        let space = env.action_space();
        assert_eq!(space.shape, vec![1]);
        assert!(matches!(space.space_type, SpaceType::Box));
    }

    #[test]
    fn observation_space_is_box() {
        let env = PendulumSwingUp::new();
        let space = env.observation_space();
        assert_eq!(space.shape, vec![3]);
        assert!(matches!(space.space_type, SpaceType::Box));
    }

    #[test]
    fn empty_action_treated_as_zero() {
        let mut env = PendulumSwingUp::new();
        env.restore_state(&PendulumState { theta: 0.0, theta_dot: 0.0, steps: 0 });
        // No panic on empty action; behaves like zero torque.
        let r = env.step(Vec::new());
        assert!(r.reward.abs() < 1e-6, "empty action at upright rest behaves like zero torque");
    }
}
