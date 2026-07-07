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
//! # I.i.d. vs. burst-structured (correlated) dropout
//!
//! The default dropout is **i.i.d.**: each frame is blanked independently with
//! probability `p`. This is only *partially* memory-hard — at CartPole's
//! control rate a reactive controller can compensate for isolated blanked
//! frames, which is why the feedforward baseline does not collapse to chance
//! (#298).
//!
//! Issue #302 adds an opt-in **burst-structured** mode (a `burst_len`
//! parameter) that closes this reactive-compensation loophole while keeping the
//! *same* overall blank rate `p`. Instead of drawing each frame independently,
//! the visibility follows a two-state Markov chain (visible ↔ blank) whose mean
//! blank-run length is `burst_len` (default [`DEFAULT_BURST_LEN`], in the 3–5
//! range from the issue) and whose stationary blank fraction is still `p`. The
//! mean visible-run length is set to `burst_len * (1 - p) / p` so the long-run
//! blank rate matches the i.i.d. baseline exactly — the *only* difference is
//! temporal correlation. Now blanks arrive in runs of several consecutive
//! frames, which a reactive controller cannot bridge (its last real
//! observation is several steps stale) but a recurrent policy can, by
//! integrating over the gap. The apples-to-apples comparison (same `p`,
//! i.i.d. vs. burst) isolates the effect of correlation on the memory
//! advantage. Enable it with
//! [`FlickeringCartPole::with_seed_probability_and_burst`]; the default
//! constructors keep the i.i.d. behavior unchanged.
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

/// Default mean blank-burst length for the correlated-occlusion mode (issue
/// #302). Sits in the middle of the 3–5 range suggested by the issue: on
/// average a blank, once started, lasts four consecutive frames.
pub const DEFAULT_BURST_LEN: f64 = 4.0;

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
    /// Mean blank-burst length for the correlated-occlusion (Markov) mode. When
    /// `None`, dropout is i.i.d. per frame (the default / #298 behavior). When
    /// `Some(l)`, visibility follows a two-state Markov chain whose mean
    /// blank-run length is `l` and whose stationary blank fraction is
    /// `flicker_prob`.
    burst_len: Option<f64>,
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
            burst_len: None,
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
            burst_len: None,
        }
    }

    /// Create a **burst-structured** (correlated-occlusion) flickering CartPole
    /// with a seeded flicker RNG (issue #302).
    ///
    /// Visibility follows a two-state Markov chain (visible ↔ blank) with
    /// stationary blank fraction `flicker_prob` and mean blank-run length
    /// `burst_len`. The mean visible-run length is derived as
    /// `burst_len * (1 - flicker_prob) / flicker_prob`, so the long-run blank
    /// rate equals `flicker_prob` exactly — matching the i.i.d. baseline while
    /// adding temporal correlation. See the [module docs](self) for the
    /// rationale.
    ///
    /// # Panics
    ///
    /// Panics if `flicker_prob` is not in the **open** interval `(0, 1)` (a
    /// Markov burst structure is only meaningful with both states reachable) or
    /// if `burst_len < 1.0` (a burst must last at least one frame).
    pub fn with_seed_probability_and_burst(seed: u64, flicker_prob: f64, burst_len: f64) -> Self {
        assert!(
            flicker_prob > 0.0 && flicker_prob < 1.0,
            "burst mode requires flicker probability in (0, 1), got {flicker_prob}"
        );
        assert!(burst_len >= 1.0, "burst length must be >= 1.0, got {burst_len}");
        Self {
            inner: CartPole::new(),
            flicker_prob,
            rng: StdRng::seed_from_u64(seed),
            flickered: false,
            burst_len: Some(burst_len),
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

    /// The mean blank-burst length for the correlated-occlusion mode, or `None`
    /// when dropout is i.i.d. per frame (the default).
    pub fn burst_length(&self) -> Option<f64> {
        self.burst_len
    }

    /// Draw a flicker decision from the **stationary** distribution (blank with
    /// probability `flicker_prob`). Used on [`Environment::reset`] for both the
    /// i.i.d. and burst modes — in both, the stationary blank fraction is `p`,
    /// so a fresh episode starts blank with probability `p`.
    fn draw_flicker(&mut self) -> bool {
        // `flicker_prob == 0.0` never blanks; `== 1.0` always blanks. Drawing
        // unconditionally keeps the RNG stream advancing at one draw per frame
        // regardless of `p`, so the schedule is a pure function of the seed.
        self.rng.random::<f64>() < self.flicker_prob
    }

    /// Advance the flicker state by one frame.
    ///
    /// - **I.i.d. mode** (`burst_len == None`): identical to [`draw_flicker`] —
    ///   each frame is blanked independently with probability `flicker_prob`.
    ///   Consumes exactly one RNG draw, so the schedule is byte-for-byte the
    ///   #298 behavior.
    /// - **Burst mode** (`burst_len == Some(l)`): a two-state Markov transition
    ///   from the *current* `flickered` state. The per-step probability of
    ///   switching out of a state is one over that state's mean run length
    ///   (`1/l` out of blank; `1 / (l * (1 - p) / p)` out of visible), which
    ///   yields geometric run lengths with the desired means and a stationary
    ///   blank fraction of `p`.
    ///
    /// [`draw_flicker`]: Self::draw_flicker
    fn advance_flicker(&mut self) -> bool {
        match self.burst_len {
            None => self.draw_flicker(),
            Some(mean_blank_run) => {
                let u = self.rng.random::<f64>();
                if self.flickered {
                    // Currently blank: leave the blank run with prob 1/l_b.
                    let p_switch = (1.0 / mean_blank_run).clamp(0.0, 1.0);
                    // Stay blank unless we switch to visible.
                    u >= p_switch
                } else {
                    // Currently visible: enter a blank run with prob 1/l_v,
                    // where l_v = l_b * (1 - p) / p.
                    let mean_visible_run =
                        mean_blank_run * (1.0 - self.flicker_prob) / self.flicker_prob;
                    let p_switch = (1.0 / mean_visible_run).clamp(0.0, 1.0);
                    u < p_switch
                }
            }
        }
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
        self.flickered = self.advance_flicker();
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

    // ---- Burst-structured (correlated-occlusion) mode, issue #302 ----------

    /// Collect the blank/visible flicker stream over `n` frames (stepping past
    /// episode ends without resetting, to sample a long uninterrupted stream).
    fn collect_flicker_stream(env: &mut FlickeringCartPole, n: usize) -> Vec<bool> {
        env.reset();
        let mut stream = Vec::with_capacity(n);
        for i in 0..n {
            env.step((i % 2) as i64);
            stream.push(env.is_flickered());
        }
        stream
    }

    /// Mean length of consecutive `true` (blank) runs in a boolean stream.
    fn mean_blank_run_length(stream: &[bool]) -> f64 {
        let mut runs = Vec::new();
        let mut cur = 0usize;
        for &b in stream {
            if b {
                cur += 1;
            } else if cur > 0 {
                runs.push(cur);
                cur = 0;
            }
        }
        if cur > 0 {
            runs.push(cur);
        }
        if runs.is_empty() {
            0.0
        } else {
            runs.iter().sum::<usize>() as f64 / runs.len() as f64
        }
    }

    #[test]
    fn test_burst_mode_reports_burst_length() {
        let env = FlickeringCartPole::with_seed_probability_and_burst(1, 0.5, 4.0);
        assert_eq!(env.burst_length(), Some(4.0));
        // The default i.i.d. constructor reports no burst length.
        assert_eq!(FlickeringCartPole::new().burst_length(), None);
    }

    #[test]
    fn test_burst_observation_shape_invariant() {
        // Burst mode never changes the observation shape (still 4-D).
        let mut env = FlickeringCartPole::with_seed_probability_and_burst(7, 0.5, 4.0);
        assert_eq!(env.observation_space().shape, vec![4]);
        env.reset();
        assert_eq!(env.get_observation().len(), 4);
        for i in 0..200 {
            let r = env.step((i % 2) as i64);
            assert_eq!(r.observation.len(), 4);
            if r.terminated || r.truncated {
                env.reset();
            }
        }
    }

    #[test]
    fn test_burst_mode_blank_rate_matches_p() {
        // The whole point of the burst construction: the stationary blank rate
        // still equals p, so it is an apples-to-apples comparison with i.i.d.
        for &p in &[0.3_f64, 0.5, 0.7] {
            let mut env = FlickeringCartPole::with_seed_probability_and_burst(123, p, 4.0);
            let stream = collect_flicker_stream(&mut env, 20000);
            let rate = stream.iter().filter(|&&b| b).count() as f64 / stream.len() as f64;
            assert!(
                (rate - p).abs() < 0.05,
                "burst blank rate {rate} should track p={p} (same as i.i.d.)"
            );
        }
    }

    #[test]
    fn test_burst_mode_produces_longer_runs_than_iid() {
        // Burst mode must produce meaningfully longer blank runs than i.i.d. at
        // the same p — that temporal correlation is what closes the reactive-
        // compensation loophole.
        let mut burst = FlickeringCartPole::with_seed_probability_and_burst(99, 0.5, 4.0);
        let mut iid = FlickeringCartPole::with_seed_and_probability(99, 0.5);

        let burst_run = mean_blank_run_length(&collect_flicker_stream(&mut burst, 20000));
        let iid_run = mean_blank_run_length(&collect_flicker_stream(&mut iid, 20000));

        // i.i.d. at p=0.5 has mean blank-run length ~1/(1-p) = 2.
        assert!(iid_run < 2.5, "i.i.d. mean blank run {iid_run} should be ~2");
        // Burst mode targets a mean blank-run length of 4.
        assert!(
            (burst_run - 4.0).abs() < 1.0,
            "burst mean blank run {burst_run} should be ≈ 4.0"
        );
        assert!(burst_run > iid_run + 1.0, "burst runs must be longer than i.i.d. runs");
    }

    #[test]
    fn test_burst_schedule_is_deterministic_under_seed() {
        let mut a = FlickeringCartPole::with_seed_probability_and_burst(42, 0.5, 4.0);
        let mut b = FlickeringCartPole::with_seed_probability_and_burst(42, 0.5, 4.0);
        let sa = collect_flicker_stream(&mut a, 2000);
        let sb = collect_flicker_stream(&mut b, 2000);
        assert_eq!(sa, sb, "burst schedule must be identical under the same seed");
    }

    #[test]
    fn test_burst_clone_restore_reproduces_stream() {
        // The Markov state lives in `flickered` + `rng`, both captured in the
        // snapshot, so restore reproduces the correlated stream.
        let mut env = FlickeringCartPole::with_seed_probability_and_burst(555, 0.5, 4.0);
        env.reset();
        for i in 0..10 {
            env.step((i % 2) as i64);
        }
        let snap = env.clone_state();
        let mut first = Vec::new();
        for i in 0..40 {
            env.step((i % 2) as i64);
            first.push(env.is_flickered());
        }
        env.restore_state(&snap);
        let mut second = Vec::new();
        for i in 0..40 {
            env.step((i % 2) as i64);
            second.push(env.is_flickered());
        }
        assert_eq!(first, second, "restore must reproduce the burst stream");
    }

    #[test]
    #[should_panic(expected = "burst mode requires flicker probability in (0, 1)")]
    fn test_burst_invalid_probability_panics() {
        let _ = FlickeringCartPole::with_seed_probability_and_burst(0, 0.0, 4.0);
    }

    #[test]
    #[should_panic(expected = "burst length must be >= 1.0")]
    fn test_burst_invalid_length_panics() {
        let _ = FlickeringCartPole::with_seed_probability_and_burst(0, 0.5, 0.5);
    }
}
