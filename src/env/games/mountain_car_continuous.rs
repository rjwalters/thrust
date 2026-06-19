//! MountainCarContinuous classic-control environment.
//!
//! [`MountainCarContinuous`] reimplements the Gym
//! `MountainCarContinuous-v0` task in-tree with no external dependencies
//! and closed-form dynamics (issue #166, part of the SAC env-pool epic
//! #163). It is a second, qualitatively different continuous-control
//! benchmark alongside [`PendulumSwingUp`](super::pendulum::PendulumSwingUp):
//! a sparse / deceptive-reward problem where the agent must build up
//! momentum by rocking back and forth before it can climb the hill.
//!
//! An underpowered car sits in a valley between two hills. The agent
//! applies a continuous engine force and is rewarded for reaching the
//! flag on top of the right hill while spending as little control effort
//! as possible.
//!
//! - **Action:** `Vec<f32>` of length 1 (engine force), clamped to `[-1.0,
//!   1.0]`. An empty vec is treated as zero force.
//! - **Observation (2-dim):** `[position, velocity]`.
//! - **Dynamics:** closed-form classic-control physics with `power = 0.0015`,
//!   `gravity = 0.0025`. Velocity is updated by `velocity += force·power −
//!   cos(3·position)·gravity` and clamped to `[-0.07, 0.07]`; position is
//!   updated by `position += velocity` and clamped to `[-1.2, 0.6]`. Hitting
//!   the left wall (`position == -1.2` with negative velocity) zeroes the
//!   velocity.
//! - **Reward:** `+100` on the step that reaches the goal (`position ≥ 0.45`),
//!   minus a control-effort penalty `0.1·force²` every step.
//! - **Termination:** reaching the goal sets `terminated = true`; this is a
//!   *real* terminal state (unlike Pendulum, which only ever truncates).
//!   Episodes also `truncated = true` after 999 steps.
//!
//! # Determinism contract
//!
//! The only source of randomness is the initial position drawn at
//! [`Environment::reset`], which is seeded (see
//! [`MountainCarContinuous::with_seed`]). Reset is reproducible: two envs
//! with the same seed produce identical episodes. The dynamics themselves
//! are fully deterministic (no RNG), so [`Environment::restore_state`]
//! followed by [`Environment::step`] reproduces every subsequent
//! [`StepResult`] bit-for-bit. The [`MountainCarState`] snapshot captures
//! the simulation state (`position`, `velocity`, `steps`) but not the
//! reset RNG.

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

/// Engine power coefficient applied to the action force.
const POWER: f32 = 0.0015;

/// Gravity coefficient driving the car back toward the valley.
const GRAVITY: f32 = 0.0025;

/// Velocity clamp range.
const MAX_SPEED: f32 = 0.07;

/// Minimum (leftmost) position.
const MIN_POS: f32 = -1.2;

/// Maximum (rightmost) position.
const MAX_POS: f32 = 0.6;

/// Position at which the goal flag sits; reaching it terminates.
const GOAL_POSITION: f32 = 0.45;

/// Default episode length cap.
const DEFAULT_MAX_STEPS: usize = 999;

/// Default seed used by [`MountainCarContinuous::new`].
const DEFAULT_SEED: u64 = 0;

/// Snapshot of [`MountainCarContinuous`]'s simulation state.
///
/// The dynamics are fully deterministic (no RNG), so
/// [`Environment::restore_state`] followed by [`Environment::step`]
/// reproduces every subsequent [`StepResult`] bit-for-bit. The snapshot
/// does *not* capture the reset RNG; restoring then calling
/// [`Environment::reset`] will draw the next seeded initial state, not
/// the one in this snapshot.
#[derive(Debug, Clone)]
pub struct MountainCarState {
    /// Car position along the track.
    pub position: f32,
    /// Car velocity.
    pub velocity: f32,
    /// Step counter for the current episode.
    pub steps: usize,
}

/// MountainCarContinuous task with a continuous engine-force action.
///
/// See the [module docs](self) for the full observation/reward/dynamics
/// specification and determinism contract.
#[derive(Debug, Clone)]
pub struct MountainCarContinuous {
    /// Car position along the track.
    position: f32,

    /// Car velocity.
    velocity: f32,

    /// Step counter for the current episode.
    steps: usize,

    /// Maximum number of steps before truncation.
    max_steps: usize,

    /// Seeded RNG driving the initial state at [`Environment::reset`].
    rng: StdRng,
}

impl MountainCarContinuous {
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
            position: 0.0,
            velocity: 0.0,
            steps: 0,
            max_steps,
            rng: StdRng::seed_from_u64(seed),
        };
        env.reset();
        env
    }

    /// Current car position (for inspection / tests).
    pub fn position(&self) -> f32 {
        self.position
    }

    /// Current car velocity (for inspection / tests).
    pub fn velocity(&self) -> f32 {
        self.velocity
    }
}

impl Default for MountainCarContinuous {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for MountainCarContinuous {
    /// Continuous engine-force action. Length-1 `Vec<f32>`; values
    /// outside `[-1.0, 1.0]` are clamped. An empty vec is treated as zero
    /// force.
    type Action = Vec<f32>;

    /// Snapshot type. The dynamics are deterministic (no RNG), so
    /// restore + step reproduces subsequent results exactly. The reset
    /// RNG is not captured; see [`MountainCarState`].
    type State = MountainCarState;

    fn reset(&mut self) {
        // Gym `MountainCarContinuous-v0` draws the initial position
        // uniformly in `[-0.6, -0.4]` with zero initial velocity.
        self.position = self.rng.random_range(-0.6..-0.4);
        self.velocity = 0.0;
        self.steps = 0;
    }

    fn get_observation(&self) -> Vec<f32> {
        vec![self.position, self.velocity]
    }

    fn step(&mut self, action: Vec<f32>) -> StepResult {
        // Length-1 action; an empty vec is treated as zero force rather
        // than panicking. Anything beyond index 0 is ignored.
        let force = action.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0);

        // Closed-form classic-control dynamics.
        self.velocity = (self.velocity + force * POWER - (3.0 * self.position).cos() * GRAVITY)
            .clamp(-MAX_SPEED, MAX_SPEED);
        self.position = (self.position + self.velocity).clamp(MIN_POS, MAX_POS);

        // Gym left-wall behavior: a car pinned at the leftmost position
        // moving left has its velocity zeroed.
        if self.position == MIN_POS && self.velocity < 0.0 {
            self.velocity = 0.0;
        }

        let terminated = self.position >= GOAL_POSITION;
        let reward = (if terminated { 100.0 } else { 0.0 }) - 0.1 * force * force;

        self.steps += 1;
        let truncated = self.steps >= self.max_steps;

        StepResult {
            observation: self.get_observation(),
            reward,
            terminated,
            truncated,
            info: StepInfo::default(),
        }
    }

    fn observation_space(&self) -> SpaceInfo {
        // [position, velocity], both continuous.
        SpaceInfo { shape: vec![2], space_type: SpaceType::Box }
    }

    fn action_space(&self) -> SpaceInfo {
        // 1D continuous engine force.
        SpaceInfo { shape: vec![1], space_type: SpaceType::Box }
    }

    fn render(&self) -> Vec<u8> {
        Vec::new()
    }

    fn close(&mut self) {}

    fn clone_state(&self) -> MountainCarState {
        MountainCarState { position: self.position, velocity: self.velocity, steps: self.steps }
    }

    fn restore_state(&mut self, state: &MountainCarState) {
        self.position = state.position;
        self.velocity = state.velocity;
        self.steps = state.steps;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_is_length_two() {
        let mut env = MountainCarContinuous::new();
        env.reset();

        let obs = env.get_observation();
        assert_eq!(obs.len(), 2, "observation must be 2-dimensional");

        let result = env.step(vec![0.5]);
        assert_eq!(result.observation.len(), 2);
    }

    #[test]
    fn force_is_clamped() {
        // Apply a huge positive force from a known state. The velocity
        // delta from the force term saturates at the clamped force.
        let pos = 0.0_f32;
        let baseline_grav = (3.0 * pos).cos() * GRAVITY;

        let mut env = MountainCarContinuous::new();
        env.restore_state(&MountainCarState { position: pos, velocity: 0.0, steps: 0 });
        env.step(vec![1000.0]);
        // Expected velocity using the clamped force of +1.0.
        let expected_pos = POWER - baseline_grav;
        assert!(
            (env.velocity() - expected_pos).abs() < 1e-6,
            "velocity should reflect clamped +1 force, got {}",
            env.velocity()
        );

        // Negative direction.
        let mut env2 = MountainCarContinuous::new();
        env2.restore_state(&MountainCarState { position: pos, velocity: 0.0, steps: 0 });
        env2.step(vec![-1000.0]);
        let expected_neg = -POWER - baseline_grav;
        assert!(
            (env2.velocity() - expected_neg).abs() < 1e-6,
            "velocity should reflect clamped -1 force, got {}",
            env2.velocity()
        );
    }

    #[test]
    fn velocity_is_clamped_to_max_speed() {
        // Start already at max speed; even a maximal force step must keep
        // velocity within `[-MAX_SPEED, MAX_SPEED]`.
        let mut env = MountainCarContinuous::new();
        env.restore_state(&MountainCarState { position: -0.5, velocity: MAX_SPEED, steps: 0 });
        env.step(vec![1.0]);
        assert!(
            env.velocity() <= MAX_SPEED + 1e-9 && env.velocity() >= -MAX_SPEED - 1e-9,
            "velocity must stay within bounds, got {}",
            env.velocity()
        );

        let mut env2 = MountainCarContinuous::new();
        env2.restore_state(&MountainCarState { position: 0.5, velocity: -MAX_SPEED, steps: 0 });
        env2.step(vec![-1.0]);
        assert!(
            env2.velocity() >= -MAX_SPEED - 1e-9,
            "velocity must stay within bounds, got {}",
            env2.velocity()
        );
    }

    #[test]
    fn position_is_clamped_and_left_wall_zeroes_velocity() {
        // Drive into the left wall with strong negative velocity.
        let mut env = MountainCarContinuous::new();
        env.restore_state(&MountainCarState {
            position: MIN_POS + 0.01,
            velocity: -MAX_SPEED,
            steps: 0,
        });
        env.step(vec![-1.0]);
        assert!(env.position() >= MIN_POS - 1e-9, "position must not drop below MIN_POS");
        assert_eq!(env.position(), MIN_POS, "should be pinned at left wall");
        assert_eq!(env.velocity(), 0.0, "left-wall contact zeroes velocity");
    }

    #[test]
    fn reaching_goal_terminates_with_bonus() {
        // Place the car just below the goal with enough velocity to cross
        // it in one step.
        let mut env = MountainCarContinuous::new();
        env.restore_state(&MountainCarState {
            position: GOAL_POSITION - 0.01,
            velocity: MAX_SPEED,
            steps: 0,
        });
        let r = env.step(vec![0.0]);
        assert!(r.terminated, "crossing the goal must set terminated");
        assert!(!r.truncated, "single goal step is not a truncation");
        // Reward = +100 minus zero control effort (force == 0).
        assert!(
            (r.reward - 100.0).abs() < 1e-6,
            "goal step should yield +100 reward, got {}",
            r.reward
        );
    }

    #[test]
    fn below_goal_does_not_terminate() {
        let mut env = MountainCarContinuous::new();
        env.restore_state(&MountainCarState { position: -0.5, velocity: 0.0, steps: 0 });
        let r = env.step(vec![0.0]);
        assert!(!r.terminated, "mid-valley step must not terminate");
        // Control-effort penalty only (no bonus).
        assert!(r.reward <= 0.0, "non-goal reward should be non-positive, got {}", r.reward);
    }

    #[test]
    fn control_effort_is_penalized() {
        let mut env = MountainCarContinuous::new();
        env.restore_state(&MountainCarState { position: -0.5, velocity: 0.0, steps: 0 });
        let r = env.step(vec![1.0]);
        // Reward == -0.1 * 1.0^2 == -0.1.
        assert!((r.reward - (-0.1)).abs() < 1e-6, "force=1 should cost 0.1, got {}", r.reward);
    }

    #[test]
    fn truncates_after_max_steps() {
        // Hold the car in the valley with no force so it never reaches
        // the goal; it must run to the truncation limit.
        let mut env = MountainCarContinuous::new();
        env.reset();

        for i in 0..(DEFAULT_MAX_STEPS - 1) {
            let r = env.step(Vec::new());
            assert!(!r.truncated, "should not truncate before max_steps (step {i})");
            assert!(!r.terminated, "passive car should not reach the goal (step {i})");
        }

        let r = env.step(Vec::new());
        assert!(r.truncated, "episode should truncate after {DEFAULT_MAX_STEPS} steps");
        assert!(!r.terminated, "truncation at the limit is not termination");
    }

    #[test]
    fn clone_restore_round_trips_next_step() {
        let mut env = MountainCarContinuous::new();
        env.reset();
        // Advance a few steps so the snapshot is non-trivial.
        env.step(vec![0.3]);
        env.step(vec![-0.7]);

        let snapshot = env.clone_state();
        let result_a = env.step(vec![1.0]);

        // Restore and replay the same action; results must match exactly.
        env.restore_state(&snapshot);
        let result_b = env.step(vec![1.0]);

        assert_eq!(result_a.observation, result_b.observation, "obs must reproduce bit-for-bit");
        assert_eq!(result_a.reward, result_b.reward, "reward must reproduce bit-for-bit");
        assert_eq!(result_a.truncated, result_b.truncated);
        assert_eq!(result_a.terminated, result_b.terminated);
    }

    #[test]
    fn seeded_reset_is_reproducible() {
        let mut a = MountainCarContinuous::with_seed(42);
        let mut b = MountainCarContinuous::with_seed(42);
        a.reset();
        b.reset();
        assert_eq!(a.get_observation(), b.get_observation(), "same seed -> same initial obs");

        // Different seeds should (essentially always) differ.
        let mut c = MountainCarContinuous::with_seed(7);
        c.reset();
        assert_ne!(
            a.get_observation(),
            c.get_observation(),
            "different seeds should give different initial states"
        );

        // Determinism holds across a full rollout from a fresh reset.
        a.reset();
        b.reset();
        for _ in 0..20 {
            let ra = a.step(vec![0.5]);
            let rb = b.step(vec![0.5]);
            assert_eq!(ra.observation, rb.observation);
            assert_eq!(ra.reward, rb.reward);
        }
    }

    #[test]
    fn action_space_is_box() {
        let env = MountainCarContinuous::new();
        let space = env.action_space();
        assert_eq!(space.shape, vec![1]);
        assert!(matches!(space.space_type, SpaceType::Box));
    }

    #[test]
    fn observation_space_is_box() {
        let env = MountainCarContinuous::new();
        let space = env.observation_space();
        assert_eq!(space.shape, vec![2]);
        assert!(matches!(space.space_type, SpaceType::Box));
    }

    #[test]
    fn empty_action_behaves_like_zero_force() {
        let mut env = MountainCarContinuous::new();
        env.restore_state(&MountainCarState { position: -0.5, velocity: 0.0, steps: 0 });
        let empty = env.step(Vec::new());

        let mut env2 = MountainCarContinuous::new();
        env2.restore_state(&MountainCarState { position: -0.5, velocity: 0.0, steps: 0 });
        let zero = env2.step(vec![0.0]);

        assert_eq!(empty.observation, zero.observation, "empty action == zero force");
        assert_eq!(empty.reward, zero.reward, "empty action == zero force reward");
    }
}
