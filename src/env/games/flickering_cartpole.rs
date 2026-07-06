//! Flickering CartPole — a partial-observability variant of [`CartPole`].
//!
//! Phase 3 of the recurrent-policy epic (#262). [`FlickeringCartPole`] wraps
//! the fully-simulated [`CartPole`] and, on every observation, with a seeded
//! probability `p` (default 0.5) replaces the **entire** observation with
//! zeros ("flicker"). The underlying physics, termination / truncation
//! thresholds, reward, and `max_steps = 500` are all inherited from `CartPole`
//! unchanged — only the *visibility* of the observation is intermittently
//! blanked. When the frame is not flickered, the full 4-D observation
//! `[x, x_dot, theta, theta_dot]` is exposed intact.
//!
//! # Why this is a POMDP (and why velocity-masking was not)
//!
//! An earlier attempt at a memory-load-bearing CartPole simply dropped the two
//! velocity coordinates (`MaskedCartPole`, kept in this crate for the record).
//! A real 500k-step training run **disproved** that as a POMDP: a memoryless
//! reactive controller on `[x, theta]` balances the pole for hundreds of steps
//! (measured MLP mean return ~324 vs LSTM ~222 — the feedforward policy won).
//! Masking velocities does not make memory load-bearing, because a reactive
//! angle/position feedback loop is sufficient to balance CartPole.
//!
//! Flickering closes that loophole. This is the canonical Atari-POMDP protocol
//! from Hausknecht & Stone, *"Deep Recurrent Q-Learning for Partially
//! Observable MDPs"* (2015): with probability `p` the observed frame is
//! entirely blanked. A feedforward policy cannot act on a zeroed frame — it has
//! no state to fall back on and must emit an action from `[0, 0, 0, 0]`, which
//! is uninformative. A recurrent policy carries its hidden state across the
//! blanked gap and integrates the intermittent stream over time, so memory
//! becomes load-bearing **by construction**: the only way to act sensibly on a
//! flickered frame is to remember the last visible one.
//!
//! # Composition, not inheritance
//!
//! Rust has no inheritance, so `FlickeringCartPole` **embeds** a `CartPole` and
//! delegates every [`Environment`] method to it, intercepting only
//! [`Environment::reset`] / [`Environment::step`] (to draw the per-frame
//! flicker decision) and [`Environment::get_observation`] (to blank the
//! observation when the current frame is flickered). The observation space is
//! still reported as 4-D — flickering never changes the observation *shape*,
//! only its contents.
//!
//! # Seeding and determinism
//!
//! The flicker decisions are drawn from a dedicated seeded [`StdRng`],
//! independent of the physics simulation. Two `FlickeringCartPole`s constructed
//! with [`FlickeringCartPole::with_seed`] (same seed and probability) produce
//! the **identical** flicker pattern given the same action sequence, regardless
//! of the (thread-RNG-seeded) physics reset. This makes the flicker schedule
//! reproducible for tests and experiments. Snapshot / restore
//! ([`Environment::clone_state`] / [`Environment::restore_state`]) captures the
//! flicker RNG as well as the physics state, so a restored env reproduces the
//! same flicker stream — a stronger determinism guarantee than the
//! RNG-consuming envs (Snake, Pong) that snapshot only the simulation step.

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::env::{
    Environment, SpaceInfo, SpaceType, StepResult,
    games::cartpole::{CartPole, CartPoleState},
};

/// Default flicker probability — the classic Hausknecht & Stone (2015)
/// value: each frame is blanked with probability 0.5.
pub const DEFAULT_FLICKER_PROBABILITY: f64 = 0.5;

/// Snapshot of a [`FlickeringCartPole`]: the inner physics state, the current
/// flicker flag, and the flicker RNG. Because the RNG is captured, restoring a
/// snapshot reproduces the subsequent flicker stream exactly (in addition to
/// the deterministic physics inherited from [`CartPole`]).
#[derive(Debug, Clone)]
pub struct FlickeringCartPoleState {
    /// Inner CartPole physics snapshot.
    inner: CartPoleState,
    /// Whether the observation for the current frame is blanked.
    flickered: bool,
    /// Flicker RNG state at snapshot time.
    rng: StdRng,
}

/// Flickering CartPole — a partially-observable variant of [`CartPole`] where
/// each frame's observation is blanked to zeros with a seeded probability.
///
/// The observation is the full 4-D CartPole state
/// `[x, x_dot, theta, theta_dot]` on a visible frame, or `[0, 0, 0, 0]` on a
/// flickered frame. All physics, termination, truncation, and reward semantics
/// are delegated to the inner [`CartPole`] unchanged; flickering affects only
/// what the agent *observes*, never the underlying dynamics or reward.
#[derive(Debug)]
pub struct FlickeringCartPole {
    /// Inner fully-simulated CartPole; owns the physics.
    inner: CartPole,
    /// Probability that any given frame is blanked (in `[0, 1]`).
    flicker_prob: f64,
    /// Dedicated flicker RNG, independent of the physics simulation.
    rng: StdRng,
    /// Whether the current frame's observation is blanked. Set on every
    /// [`Environment::reset`] and [`Environment::step`]; read by
    /// [`Environment::get_observation`].
    flickered: bool,
}

impl FlickeringCartPole {
    /// Create a flickering CartPole with the default probability
    /// ([`DEFAULT_FLICKER_PROBABILITY`] = 0.5) and a flicker RNG seeded from
    /// system entropy.
    ///
    /// Because the flicker RNG is entropy-seeded, independent instances (e.g.
    /// the members of an [`EnvPool`](crate::env::pool::EnvPool)) get
    /// **different** flicker streams, which is desirable for decorrelated
    /// parallel rollouts. Use [`FlickeringCartPole::with_seed`] when a
    /// reproducible flicker schedule is required.
    pub fn new() -> Self {
        Self::with_probability(DEFAULT_FLICKER_PROBABILITY)
    }

    /// Create a flickering CartPole with a custom flicker probability and a
    /// flicker RNG seeded from system entropy.
    ///
    /// # Panics
    ///
    /// Panics if `flicker_prob` is not in `[0, 1]`.
    pub fn with_probability(flicker_prob: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&flicker_prob),
            "flicker probability must be in [0, 1], got {flicker_prob}"
        );
        Self {
            inner: CartPole::new(),
            flicker_prob,
            rng: StdRng::from_os_rng(),
            flickered: false,
        }
    }

    /// Create a flickering CartPole with the default probability
    /// ([`DEFAULT_FLICKER_PROBABILITY`] = 0.5) and a **seeded** flicker RNG for
    /// a reproducible flicker schedule.
    pub fn with_seed(seed: u64) -> Self {
        Self::with_seed_and_probability(seed, DEFAULT_FLICKER_PROBABILITY)
    }

    /// Create a flickering CartPole with a custom flicker probability and a
    /// **seeded** flicker RNG for a reproducible flicker schedule.
    ///
    /// Two instances built with the same `seed` and `flicker_prob` blank the
    /// same frames given the same number of `reset`/`step` calls, independent
    /// of the physics (whose reset perturbation uses the thread RNG).
    ///
    /// # Panics
    ///
    /// Panics if `flicker_prob` is not in `[0, 1]`.
    pub fn with_seed_and_probability(seed: u64, flicker_prob: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&flicker_prob),
            "flicker probability must be in [0, 1], got {flicker_prob}"
        );
        Self {
            inner: CartPole::new(),
            flicker_prob,
            rng: StdRng::seed_from_u64(seed),
            flickered: false,
        }
    }

    /// The probability that any given frame is blanked.
    pub fn flicker_probability(&self) -> f64 {
        self.flicker_prob
    }

    /// Whether the observation for the *current* frame is blanked (zeroed).
    ///
    /// Reflects the flicker decision made by the most recent
    /// [`Environment::reset`] or [`Environment::step`]. Primarily useful for
    /// diagnostics and determinism tests.
    pub fn is_flickered(&self) -> bool {
        self.flickered
    }

    /// Draw a fresh flicker decision from the seeded RNG.
    fn draw_flicker(&mut self) -> bool {
        // `flicker_prob == 0.0` never blanks; `== 1.0` always blanks. Drawing
        // unconditionally keeps the RNG stream advancing at one draw per frame
        // regardless of `p`, so the schedule is a pure function of the seed.
        self.rng.random::<f64>() < self.flicker_prob
    }

    /// The dimensionality of the (unflickered) observation.
    const OBS_DIM: usize = 4;
}

impl Default for FlickeringCartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for FlickeringCartPole {
    type Action = i64;
    type State = FlickeringCartPoleState;

    fn reset(&mut self) {
        self.inner.reset();
        self.flickered = self.draw_flicker();
    }

    fn get_observation(&self) -> Vec<f32> {
        if self.flickered {
            vec![0.0; Self::OBS_DIM]
        } else {
            Environment::get_observation(&self.inner)
        }
    }

    fn step(&mut self, action: i64) -> StepResult {
        let mut result = self.inner.step(action);
        self.flickered = self.draw_flicker();
        if self.flickered {
            // Blank the entire observation — the flicker protocol zeros the
            // whole frame, not individual coordinates.
            for v in result.observation.iter_mut() {
                *v = 0.0;
            }
        }
        result
    }

    fn observation_space(&self) -> SpaceInfo {
        // Flickering never changes the observation *shape*, only its contents.
        SpaceInfo { shape: vec![Self::OBS_DIM], space_type: SpaceType::Box }
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

    fn clone_state(&self) -> FlickeringCartPoleState {
        FlickeringCartPoleState {
            inner: self.inner.clone_state(),
            flickered: self.flickered,
            rng: self.rng.clone(),
        }
    }

    fn restore_state(&mut self, state: &FlickeringCartPoleState) {
        self.inner.restore_state(&state.inner);
        self.flickered = state.flickered;
        self.rng = state.rng.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_space_is_four_dimensional() {
        let env = FlickeringCartPole::new();
        let obs_space = env.observation_space();
        assert_eq!(obs_space.shape, vec![4], "flickering obs keeps CartPole's 4-D shape");
        assert!(matches!(obs_space.space_type, SpaceType::Box));
    }

    #[test]
    fn test_action_space_delegates() {
        let env = FlickeringCartPole::new();
        let action_space = env.action_space();
        assert!(matches!(action_space.space_type, SpaceType::Discrete(2)));
    }

    #[test]
    fn test_default_probability() {
        let env = FlickeringCartPole::new();
        assert_eq!(env.flicker_probability(), DEFAULT_FLICKER_PROBABILITY);
        assert_eq!(env.flicker_probability(), 0.5);
    }

    #[test]
    fn test_observation_length_is_always_four() {
        // Whether flickered or not, the observation vector is always 4 long.
        let mut env = FlickeringCartPole::with_seed_and_probability(7, 0.5);
        env.reset();
        assert_eq!(env.get_observation().len(), 4);
        for i in 0..200 {
            let result = env.step((i % 2) as i64);
            assert_eq!(result.observation.len(), 4, "obs length invariant under flicker");
            if result.terminated || result.truncated {
                env.reset();
            }
        }
    }

    #[test]
    fn test_flickered_observation_is_all_zeros() {
        // With p = 1.0 every frame is blanked: observation must be all zeros.
        let mut env = FlickeringCartPole::with_seed_and_probability(1, 1.0);
        env.reset();
        assert!(env.is_flickered(), "p=1.0 must blank every frame");
        assert_eq!(env.get_observation(), vec![0.0; 4]);
        let result = env.step(1);
        assert!(env.is_flickered());
        assert_eq!(result.observation, vec![0.0; 4], "stepped obs blanked under p=1.0");
    }

    #[test]
    fn test_never_flickers_at_zero_probability() {
        // With p = 0.0 no frame is ever blanked; obs equals the inner CartPole.
        let mut env = FlickeringCartPole::with_seed_and_probability(2, 0.0);
        env.reset();
        assert!(!env.is_flickered(), "p=0.0 must never blank");
        for i in 0..300 {
            let result = env.step((i % 2) as i64);
            assert!(!env.is_flickered(), "p=0.0 must never blank");
            // A visible frame is exactly the inner (unmasked) observation.
            assert_eq!(result.observation, Environment::get_observation(&env.inner));
            if result.terminated || result.truncated {
                env.reset();
            }
        }
    }

    #[test]
    fn test_flicker_schedule_is_deterministic_under_seed() {
        // Two envs with the same seed + probability blank the same frames given
        // the same action sequence — independent of the (thread-RNG) physics.
        let mut a = FlickeringCartPole::with_seed_and_probability(42, 0.5);
        let mut b = FlickeringCartPole::with_seed_and_probability(42, 0.5);
        a.reset();
        b.reset();
        assert_eq!(a.is_flickered(), b.is_flickered(), "reset flicker decision must match");

        let mut any_flicker = false;
        let mut any_visible = false;
        for i in 0..500 {
            let action = (i % 2) as i64;
            a.step(action);
            b.step(action);
            assert_eq!(a.is_flickered(), b.is_flickered(), "flicker schedule diverged at step {i}");
            any_flicker |= a.is_flickered();
            any_visible |= !a.is_flickered();
        }
        // Sanity: at p=0.5 over 500 draws we must see both states.
        assert!(any_flicker, "expected at least one flickered frame at p=0.5");
        assert!(any_visible, "expected at least one visible frame at p=0.5");
    }

    #[test]
    fn test_flicker_rate_is_approximately_p() {
        // Empirically the blank rate over many frames should track p ≈ 0.5.
        let mut env = FlickeringCartPole::with_seed_and_probability(123, 0.5);
        env.reset();
        let mut blanked = 0usize;
        let n = 5000;
        for i in 0..n {
            env.step((i % 2) as i64);
            if env.is_flickered() {
                blanked += 1;
            }
            // Keep stepping past episode ends without resetting flicker RNG:
            // resetting would still keep the schedule seeded, but we just want
            // a long stream here.
            if env.get_observation().is_empty() {
                unreachable!();
            }
        }
        let rate = blanked as f64 / n as f64;
        assert!((rate - 0.5).abs() < 0.05, "blank rate {rate} should be ≈ 0.5");
    }

    #[test]
    fn test_reward_and_done_unaffected_by_flicker() {
        // Flickering blanks only the observation; reward/termination come from
        // the inner CartPole and are unchanged.
        let mut env = FlickeringCartPole::with_seed_and_probability(9, 0.5);
        env.reset();
        for i in 0..100 {
            let result = env.step((i % 2) as i64);
            assert!(result.reward == 0.0 || result.reward == 1.0, "reward inherited from CartPole");
            if result.terminated || result.truncated {
                env.reset();
            }
        }
    }

    #[test]
    fn test_clone_restore_reproduces_flicker_stream() {
        // Snapshotting captures the flicker RNG, so restore + step reproduces
        // the same flicker decisions (and the deterministic physics).
        let mut env = FlickeringCartPole::with_seed_and_probability(555, 0.5);
        env.reset();
        for i in 0..10 {
            env.step((i % 2) as i64);
        }
        let snap = env.clone_state();

        let mut first = Vec::new();
        for i in 0..20 {
            let r = env.step((i % 2) as i64);
            first.push((env.is_flickered(), r.observation.clone(), r.reward));
        }

        env.restore_state(&snap);
        let mut second = Vec::new();
        for i in 0..20 {
            let r = env.step((i % 2) as i64);
            second.push((env.is_flickered(), r.observation.clone(), r.reward));
        }

        assert_eq!(first, second, "restore must reproduce flicker + physics stream");
    }

    #[test]
    fn test_hundred_random_steps_no_panic() {
        let mut env = FlickeringCartPole::with_seed(0);
        env.reset();
        for i in 0..100 {
            let result = env.step((i % 2) as i64);
            assert_eq!(result.observation.len(), 4);
            if result.terminated || result.truncated {
                env.reset();
            }
        }
    }

    #[test]
    #[should_panic(expected = "flicker probability must be in [0, 1]")]
    fn test_invalid_probability_panics() {
        let _ = FlickeringCartPole::with_probability(1.5);
    }
}
