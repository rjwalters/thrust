//! Minimal continuous-action environment for trait-wiring proof.
//!
//! `ContinuousLqr` is a one-dimensional linear quadratic regulator.
//! It exists to demonstrate that the [`crate::env::Environment`] trait
//! accepts a non-`i64` action (see issue #61), not to be a useful
//! training target.
//!
//! State:  `[position, velocity]`.
//! Action: `Vec<f32>` of length 1 (a scalar force in `[-1.0, 1.0]`).
//! Reward: `-position^2 - 0.01 * action^2` (penalise being away from
//! the origin and using control effort).
//!
//! Episode terminates after `max_steps` steps (default 50).
//!
//! Continuous-action training algorithms (SAC, TD3, DDPG, PPO with
//! Gaussian heads) are explicitly *out of scope* for this env. Its
//! only job is to compile against the trait.

use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

/// Snapshot of [`ContinuousLqr`]'s simulation state.
///
/// `ContinuousLqr` is fully deterministic (no internal RNG), so
/// [`Environment::restore_state`] followed by [`Environment::step`] reproduces
/// every subsequent [`StepResult`] bit-for-bit.
#[derive(Debug, Clone)]
pub struct ContinuousLqrState {
    /// Position scalar.
    pub position: f32,
    /// Velocity scalar.
    pub velocity: f32,
    /// Step counter.
    pub steps: usize,
}

/// Force clamp range applied to the 1D action input.
const ACTION_CLAMP: f32 = 1.0;

/// Per-step integration constant (treated as a unit timestep).
const DT: f32 = 1.0;

/// Mild linear damping on velocity each step.
const VELOCITY_DAMPING: f32 = 0.1;

/// Default episode length cap.
const DEFAULT_MAX_STEPS: usize = 50;

/// 1D linear quadratic regulator with a continuous-force action.
#[derive(Debug, Clone)]
pub struct ContinuousLqr {
    /// Position scalar.
    position: f32,

    /// Velocity scalar.
    velocity: f32,

    /// Step counter for the current episode.
    steps: usize,

    /// Maximum number of steps before truncation.
    max_steps: usize,
}

impl ContinuousLqr {
    /// Create a new env with the default episode length.
    pub fn new() -> Self {
        Self::with_max_steps(DEFAULT_MAX_STEPS)
    }

    /// Create a new env with a custom maximum episode length.
    pub fn with_max_steps(max_steps: usize) -> Self {
        Self { position: 0.5, velocity: 0.0, steps: 0, max_steps }
    }

    /// Current position (for inspection / tests).
    pub fn position(&self) -> f32 {
        self.position
    }

    /// Current velocity (for inspection / tests).
    pub fn velocity(&self) -> f32 {
        self.velocity
    }
}

impl Default for ContinuousLqr {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for ContinuousLqr {
    /// Continuous force action. Length-1 `Vec<f32>`; values outside
    /// `[-ACTION_CLAMP, ACTION_CLAMP]` are clamped.
    type Action = Vec<f32>;

    /// Snapshot type. `ContinuousLqr` is deterministic (no RNG), so
    /// restore + step reproduces subsequent results exactly.
    type State = ContinuousLqrState;

    fn reset(&mut self) {
        self.position = 0.5;
        self.velocity = 0.0;
        self.steps = 0;
    }

    fn get_observation(&self) -> Vec<f32> {
        vec![self.position, self.velocity]
    }

    fn step(&mut self, action: Vec<f32>) -> StepResult {
        // Length-1 action; if a caller hands us an empty vec we
        // treat it as zero force rather than panicking. Anything
        // beyond index 0 is ignored.
        let raw = action.first().copied().unwrap_or(0.0);
        let force = raw.clamp(-ACTION_CLAMP, ACTION_CLAMP);

        // Integrate: v_{t+1} = v_t + (force - damping*v_t)*dt
        //            x_{t+1} = x_t + v_{t+1}*dt
        self.velocity += (force - VELOCITY_DAMPING * self.velocity) * DT;
        self.position += self.velocity * DT;

        self.steps += 1;
        let truncated = self.steps >= self.max_steps;

        let reward = -(self.position * self.position) - 0.01 * (force * force);

        StepResult {
            observation: self.get_observation(),
            reward,
            terminated: false,
            truncated,
            info: StepInfo::default(),
        }
    }

    fn observation_space(&self) -> SpaceInfo {
        // [position, velocity], both continuous.
        SpaceInfo { shape: vec![2], space_type: SpaceType::Box }
    }

    fn action_space(&self) -> SpaceInfo {
        // 1D continuous force.
        SpaceInfo { shape: vec![1], space_type: SpaceType::Box }
    }

    fn render(&self) -> Vec<u8> {
        Vec::new()
    }

    fn close(&mut self) {}

    fn clone_state(&self) -> ContinuousLqrState {
        ContinuousLqrState { position: self.position, velocity: self.velocity, steps: self.steps }
    }

    fn restore_state(&mut self, state: &ContinuousLqrState) {
        self.position = state.position;
        self.velocity = state.velocity;
        self.steps = state.steps;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_lqr_accepts_vec_f32_action() {
        let mut env = ContinuousLqr::new();
        env.reset();

        // Apply a positive force; velocity should become positive
        // and position should move forward by the new velocity.
        let result = env.step(vec![0.5]);
        assert_eq!(result.observation.len(), 2);
        assert!(env.velocity() > 0.0, "positive force should produce positive velocity");
        assert!(env.position() > 0.5, "positive velocity should advance position");

        // Reward should be negative (we are off the origin and used
        // control effort).
        assert!(result.reward < 0.0);

        // Single step does not terminate or truncate the episode.
        assert!(!result.terminated);
        assert!(!result.truncated);
    }

    #[test]
    fn continuous_lqr_terminates_after_max_steps() {
        let mut env = ContinuousLqr::with_max_steps(3);
        env.reset();

        for _ in 0..2 {
            let r = env.step(vec![0.0]);
            assert!(!r.truncated);
        }

        let r = env.step(vec![0.0]);
        assert!(r.truncated, "episode should truncate after max_steps");
    }

    #[test]
    fn continuous_lqr_clamps_extreme_actions() {
        let mut env = ContinuousLqr::new();
        env.reset();

        // Force way above clamp range: effective force is clamped
        // to ACTION_CLAMP, so velocity change is bounded.
        let _ = env.step(vec![1000.0]);
        assert!(env.velocity() <= ACTION_CLAMP);
    }

    #[test]
    fn continuous_lqr_action_space_is_box() {
        let env = ContinuousLqr::new();
        let space = env.action_space();
        assert_eq!(space.shape, vec![1]);
        assert!(matches!(space.space_type, SpaceType::Box));
    }

    #[test]
    fn continuous_lqr_empty_action_treated_as_zero() {
        let mut env = ContinuousLqr::new();
        env.reset();
        // No panic on empty action.
        let _ = env.step(Vec::new());
        assert_eq!(env.velocity(), -VELOCITY_DAMPING * 0.0);
    }
}
