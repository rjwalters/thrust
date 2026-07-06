//! Partially-observable CartPole (velocity-masked).
//!
//! Phase 3 of the recurrent-policy epic (#262). [`MaskedCartPole`] wraps the
//! fully-simulated [`CartPole`] and drops the two velocity coordinates
//! (`x_dot`, `theta_dot`) from the observation, exposing only the positional
//! pair `[cart_position, pole_angle]`. The underlying physics, termination /
//! truncation thresholds, reward, and `max_steps = 500` are all inherited
//! from `CartPole` unchanged — only the observation projection differs.
//!
//! # Why this is a POMDP
//!
//! CartPole's 4-D state `[x, x_dot, theta, theta_dot]` is Markov: a
//! feedforward policy can act optimally from a single observation. Masking
//! the velocities leaves `[x, theta]`, which is **not** Markov — the optimal
//! action depends on how fast the pole is falling, information no single
//! frame carries. A memoryless policy must therefore plateau well below the
//! CartPole-v1 "solved" bar (500), while a recurrent policy can integrate the
//! positional stream over time to recover the hidden velocities and balance.
//! This contrast is the load-bearing learning signal for the recurrent PPO
//! stack (see `docs/RECURRENT_POLICY_DESIGN.md`).
//!
//! # Composition, not inheritance
//!
//! Rust has no inheritance, so `MaskedCartPole` **embeds** a `CartPole` and
//! delegates every [`Environment`] method to it, intercepting only
//! [`Environment::get_observation`], [`Environment::step`] (to project the
//! returned observation), and [`Environment::observation_space`] (to report a
//! 2-D shape).

use crate::env::{
    Environment, SpaceInfo, SpaceType, StepResult,
    games::cartpole::{CartPole, CartPoleState},
};

/// Velocity-masked CartPole — a partially-observable variant of
/// [`CartPole`].
///
/// The observation is projected from the full 4-D state
/// `[x, x_dot, theta, theta_dot]` down to the 2-D positional pair
/// `[x, theta]`; the cart/pole velocities are hidden. All physics,
/// termination, truncation, and reward semantics are delegated to the inner
/// [`CartPole`] unchanged.
///
/// # Snapshot semantics
///
/// [`Environment::clone_state`] / [`Environment::restore_state`] delegate to
/// the inner `CartPole`, inheriting its fully-deterministic snapshot
/// contract: restoring a snapshot and calling `step(action)` reproduces the
/// same [`StepResult`] (with the masked observation).
#[derive(Debug, Default)]
pub struct MaskedCartPole {
    inner: CartPole,
}

impl MaskedCartPole {
    /// Create a new velocity-masked CartPole with default physics.
    pub fn new() -> Self {
        Self { inner: CartPole::new() }
    }

    /// Project a full 4-D CartPole observation `[x, x_dot, theta, theta_dot]`
    /// down to the observable 2-D pair `[x, theta]`, dropping the velocity
    /// coordinates at indices 1 and 3.
    fn mask(full: &[f32]) -> Vec<f32> {
        vec![full[0], full[2]]
    }
}

impl Environment for MaskedCartPole {
    type Action = i64;
    type State = CartPoleState;

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_observation(&self) -> Vec<f32> {
        Self::mask(&Environment::get_observation(&self.inner))
    }

    fn step(&mut self, action: i64) -> StepResult {
        let mut result = self.inner.step(action);
        result.observation = Self::mask(&result.observation);
        result
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo { shape: vec![2], space_type: SpaceType::Box }
    }

    fn action_space(&self) -> SpaceInfo {
        self.inner.action_space()
    }

    fn render(&self) -> Vec<u8> {
        self.inner.render()
    }

    fn close(&mut self) {
        self.inner.close();
    }

    fn clone_state(&self) -> CartPoleState {
        self.inner.clone_state()
    }

    fn restore_state(&mut self, state: &CartPoleState) {
        self.inner.restore_state(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_is_two_dimensional() {
        let env = MaskedCartPole::new();
        let obs_space = env.observation_space();
        assert_eq!(obs_space.shape, vec![2], "masked obs must be 2-D");
        assert!(matches!(obs_space.space_type, SpaceType::Box));
    }

    #[test]
    fn test_get_observation_drops_velocities() {
        let mut env = MaskedCartPole::new();
        env.reset();
        let obs = env.get_observation();
        assert_eq!(obs.len(), 2, "masked observation should have 2 elements");
    }

    #[test]
    fn test_step_returns_masked_observation() {
        let mut env = MaskedCartPole::new();
        env.reset();
        let result = env.step(1);
        assert_eq!(result.observation.len(), 2, "stepped observation should be masked to 2-D");
        assert!(result.reward == 0.0 || result.reward == 1.0, "reward inherited from CartPole");
    }

    #[test]
    fn test_action_space_delegates() {
        let env = MaskedCartPole::new();
        let action_space = env.action_space();
        assert!(matches!(action_space.space_type, SpaceType::Discrete(2)));
    }

    #[test]
    fn test_masked_value_matches_full_positions() {
        // The two exposed coordinates must be exactly the cart position and
        // pole angle from the underlying full observation (indices 0 and 2).
        let mut env = MaskedCartPole::new();
        env.reset();
        let full = Environment::get_observation(&env.inner);
        let masked = env.get_observation();
        assert_eq!(masked[0], full[0], "index 0 = cart position");
        assert_eq!(masked[1], full[2], "index 1 = pole angle");
    }

    #[test]
    fn test_hundred_random_steps_no_panic() {
        let mut env = MaskedCartPole::new();
        env.reset();
        for i in 0..100 {
            let result = env.step((i % 2) as i64);
            assert_eq!(result.observation.len(), 2);
            if result.terminated || result.truncated {
                env.reset();
            }
        }
    }
}
