//! T-maze — the canonical memory-hard POMDP benchmark (Bakker 2001).
//!
//! Part of the recurrent-policy epic (#262), extending the memory-hard
//! environment suite alongside [`FlickeringCartPole`](super::FlickeringCartPole)
//! (#298). Where flickering CartPole is only *approximately* memory-hard — a
//! reactive controller partially compensates at CartPole's control rate — the
//! T-maze is **provably** memory-hard: a memoryless policy is at chance (50%)
//! at the junction regardless of capacity. This is the clean qualitative
//! contrast that `MaskedCartPole` failed to provide (#287's negative result).
//!
//! # The task
//!
//! The agent traverses a straight corridor of length `N` and, at the far end,
//! reaches a T-junction where it must turn **up** or **down**. Which turn is
//! rewarded is signalled by a **cue** shown *only at the start* (step 0). To act
//! correctly at the junction the agent must carry the cue across the entire
//! corridor in memory. From Bakker, *"Reinforcement Learning with Long
//! Short-Term Memory"* (NeurIPS 2001).
//!
//! - **Corridor length `N`** (configurable, default [`DEFAULT_CORRIDOR_LENGTH`]
//!   = 10). The episode is exactly `N + 1` steps long: the agent observes the
//!   cue at step 0, walks the corridor over steps `1..N`, and makes the junction
//!   decision on the observation seen at step `N`. The memory span the policy
//!   must bridge is therefore `N` steps.
//! - **Action:** `i64`, two discrete choices: `0 = up`, `1 = down`. The action
//!   is only consequential at the junction; while in the corridor the agent
//!   auto-advances and the action is ignored (a no-op). This isolates the memory
//!   variable — corridor navigation is trivial and identical for every policy,
//!   so the *only* thing that separates a memoryful from a memoryless policy is
//!   recalling the cue.
//! - **Observation:** a fixed 3-D `Vec<f32>` `[cue, corridor, junction]`:
//!   - `cue` is `+1.0` (turn up) or `-1.0` (turn down) **only at step 0**; it is
//!     `0.0` at every subsequent step. This is the memory-load-bearing channel.
//!   - `corridor` is `1.0` while the agent is in the corridor (steps `0..N-1`)
//!     and `0.0` at the junction.
//!   - `junction` is `1.0` at the junction (step `N`) and `0.0` otherwise.
//! - **Reward:** `0.0` for every corridor step; at the junction, [`REWARD_CORRECT`]
//!   (`+1.0`) if the chosen turn matches the cue, else [`REWARD_WRONG`] (`-1.0`).
//!   Episode return is therefore exactly `+1` (correct) or `-1` (wrong), so the
//!   mean return maps directly to junction accuracy: `acc = (mean_return + 1) / 2`.
//! - **Termination:** the episode `terminated`s the moment the junction decision
//!   is taken (step `N`). There is no truncation in the default configuration —
//!   the fixed-length corridor always ends at the junction.
//!
//! # Why a memoryless policy is provably at chance
//!
//! The junction observation is `[0, 0, 1]` — **identical for both cue values**.
//! The cue appears in the observation stream exactly once, at step 0, and the
//! cue is drawn independently and uniformly (50/50) each episode. A memoryless
//! policy conditions its junction action only on the current observation
//! `[0, 0, 1]`, which is a constant independent of the cue; therefore
//! `P(correct) = P(cue = chosen turn) = 0.5` for *any* memoryless policy,
//! regardless of network capacity. This is a hard information-theoretic bound,
//! not an optimization artefact.
//!
//! **This is also a self-test on the environment.** If a feedforward baseline
//! trained on this env beats 50% junction accuracy by a statistically
//! meaningful margin, the environment has an information leak (e.g. the cue
//! bleeding into a later observation) and that is a bug in the env, not a
//! finding about the policy. The unit tests below assert the no-leak property
//! directly.
//!
//! # Seeding and determinism
//!
//! The cue is drawn from a dedicated seeded [`StdRng`]. Two `TMaze`s built with
//! [`TMaze::with_seed`] (same seed, same corridor length) produce the identical
//! cue sequence across resets, independent of the actions taken. The
//! [`TMazeState`] snapshot captures the position, cue, step counter, **and** the
//! cue RNG, so [`Environment::restore_state`] followed by further resets
//! reproduces the same cue stream — the same strong determinism guarantee as
//! [`FlickeringCartPole`](super::FlickeringCartPole).

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

/// Default corridor length. At `N = 10` the memory span is 10 steps — the
/// canonical Bakker setting where an LSTM clears >90% junction accuracy while a
/// memoryless policy is pinned at chance.
pub const DEFAULT_CORRIDOR_LENGTH: usize = 10;

/// Default seed used by [`TMaze::new`].
pub const DEFAULT_SEED: u64 = 0;

/// Number of discrete actions (`0 = up`, `1 = down`).
pub const NUM_ACTIONS: usize = 2;

/// Observation dimensionality: `[cue, corridor, junction]`.
pub const OBS_DIM: usize = 3;

/// Reward for taking the junction turn that matches the cue.
pub const REWARD_CORRECT: f32 = 1.0;

/// Reward for taking the junction turn that does not match the cue.
pub const REWARD_WRONG: f32 = -1.0;

/// The cue shown at the start of the corridor: which way to turn at the
/// junction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cue {
    /// Turn up at the junction (rewarded action is `0`).
    Up,
    /// Turn down at the junction (rewarded action is `1`).
    Down,
}

impl Cue {
    /// The action index that is rewarded for this cue (`0 = up`, `1 = down`).
    pub fn correct_action(self) -> i64 {
        match self {
            Cue::Up => 0,
            Cue::Down => 1,
        }
    }

    /// The value written into the `cue` observation channel at step 0
    /// (`+1.0` for up, `-1.0` for down).
    fn signal(self) -> f32 {
        match self {
            Cue::Up => 1.0,
            Cue::Down => -1.0,
        }
    }
}

/// Snapshot of a [`TMaze`]: position, cue, step counter, and the cue RNG.
///
/// Because the RNG is captured, restoring a snapshot and then calling
/// [`Environment::reset`] reproduces the subsequent cue stream exactly, in
/// addition to the fully deterministic corridor dynamics.
#[derive(Debug, Clone)]
pub struct TMazeState {
    /// Position along the corridor in `0..=N` (`N` is the junction).
    position: usize,
    /// The cue for the current episode.
    cue: Cue,
    /// Cue RNG state at snapshot time.
    rng: StdRng,
}

/// T-maze — the provably memory-hard POMDP benchmark (Bakker 2001).
///
/// See the [module docs](self) for the full task, observation, reward, and
/// determinism specification.
#[derive(Debug)]
pub struct TMaze {
    /// Corridor length `N`. The junction is at position `N`.
    corridor_length: usize,
    /// Current position along the corridor in `0..=N`.
    position: usize,
    /// The cue for the current episode, drawn on [`Environment::reset`].
    cue: Cue,
    /// Dedicated cue RNG for reproducible cue sequences.
    rng: StdRng,
}

impl TMaze {
    /// Create a T-maze with the default corridor length
    /// ([`DEFAULT_CORRIDOR_LENGTH`] = 10) and a **seeded** cue RNG for a
    /// reproducible cue sequence.
    pub fn new() -> Self {
        Self::with_seed_and_corridor_length(DEFAULT_SEED, DEFAULT_CORRIDOR_LENGTH)
    }

    /// Create a T-maze with a custom corridor length and the default seed.
    ///
    /// # Panics
    ///
    /// Panics if `corridor_length` is zero. A zero-length corridor would place
    /// the cue and the junction at the same step, leaking the cue into the
    /// junction observation and destroying the memory task.
    pub fn with_corridor_length(corridor_length: usize) -> Self {
        Self::with_seed_and_corridor_length(DEFAULT_SEED, corridor_length)
    }

    /// Create a T-maze with the default corridor length and a custom seed.
    pub fn with_seed(seed: u64) -> Self {
        Self::with_seed_and_corridor_length(seed, DEFAULT_CORRIDOR_LENGTH)
    }

    /// Create a T-maze with a custom seed and corridor length.
    ///
    /// Two instances built with the same `seed` and `corridor_length` draw the
    /// same cue on every reset, independent of the actions taken.
    ///
    /// # Panics
    ///
    /// Panics if `corridor_length` is zero (see [`TMaze::with_corridor_length`]).
    pub fn with_seed_and_corridor_length(seed: u64, corridor_length: usize) -> Self {
        assert!(
            corridor_length >= 1,
            "corridor_length must be >= 1 (a zero-length corridor leaks the cue into the junction observation), got {corridor_length}"
        );
        let mut env = Self {
            corridor_length,
            position: 0,
            cue: Cue::Up,
            rng: StdRng::seed_from_u64(seed),
        };
        env.reset();
        env
    }

    /// The corridor length `N` (the junction is at position `N`).
    pub fn corridor_length(&self) -> usize {
        self.corridor_length
    }

    /// The cue for the current episode (primarily for diagnostics and tests).
    pub fn cue(&self) -> Cue {
        self.cue
    }

    /// The current position along the corridor in `0..=N`.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether the agent is currently at the junction (`position == N`).
    pub fn at_junction(&self) -> bool {
        self.position == self.corridor_length
    }
}

impl Default for TMaze {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for TMaze {
    /// Discrete junction action. `0 = up`, `1 = down`; only consequential at the
    /// junction (ignored, as a no-op, while in the corridor).
    type Action = i64;

    /// Snapshot type capturing position, cue, and the cue RNG. See
    /// [`TMazeState`].
    type State = TMazeState;

    fn reset(&mut self) {
        self.position = 0;
        // Draw the cue uniformly at random from the seeded RNG (exactly one draw
        // per episode, so the cue sequence is a pure function of the seed).
        self.cue = if self.rng.random::<bool>() { Cue::Up } else { Cue::Down };
    }

    fn get_observation(&self) -> Vec<f32> {
        // [cue, corridor, junction].
        //   cue      : nonzero ONLY at step 0 (position 0) — the memory channel.
        //   corridor : 1.0 while in the corridor (positions 0..N-1).
        //   junction : 1.0 at the junction (position N); constant across cues.
        let cue = if self.position == 0 { self.cue.signal() } else { 0.0 };
        let at_junction = self.position == self.corridor_length;
        let corridor = if at_junction { 0.0 } else { 1.0 };
        let junction = if at_junction { 1.0 } else { 0.0 };
        vec![cue, corridor, junction]
    }

    fn step(&mut self, action: i64) -> StepResult {
        if self.position < self.corridor_length {
            // In the corridor: auto-advance one cell; the action is a no-op.
            // Reward is zero and the episode continues.
            self.position += 1;
            StepResult {
                observation: self.get_observation(),
                reward: 0.0,
                terminated: false,
                truncated: false,
                info: StepInfo::default(),
            }
        } else {
            // At the junction: the action is the decision. Reward +1 if it
            // matches the cue, -1 otherwise, and the episode terminates.
            let reward =
                if action == self.cue.correct_action() { REWARD_CORRECT } else { REWARD_WRONG };
            StepResult {
                observation: self.get_observation(),
                reward,
                terminated: true,
                truncated: false,
                info: StepInfo::default(),
            }
        }
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo { shape: vec![OBS_DIM], space_type: SpaceType::Box }
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo { shape: vec![NUM_ACTIONS], space_type: SpaceType::Discrete(NUM_ACTIONS) }
    }

    fn render(&self) -> Vec<u8> {
        Vec::new()
    }

    fn close(&mut self) {}

    fn clone_state(&self) -> TMazeState {
        TMazeState { position: self.position, cue: self.cue, rng: self.rng.clone() }
    }

    fn restore_state(&mut self, state: &TMazeState) {
        self.position = state.position;
        self.cue = state.cue;
        self.rng = state.rng.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UP: i64 = 0;
    const DOWN: i64 = 1;

    /// Walk the corridor from a fresh reset to the junction, returning the
    /// observation seen at each step (including the post-reset observation) and
    /// leaving the env poised at the junction.
    fn walk_to_junction(env: &mut TMaze) -> Vec<Vec<f32>> {
        env.reset();
        let mut obs = vec![env.get_observation()];
        for _ in 0..env.corridor_length() {
            let r = env.step(UP);
            obs.push(r.observation);
        }
        obs
    }

    #[test]
    fn observation_space_is_three_dimensional_box() {
        let env = TMaze::new();
        let space = env.observation_space();
        assert_eq!(space.shape, vec![OBS_DIM]);
        assert!(matches!(space.space_type, SpaceType::Box));
    }

    #[test]
    fn action_space_is_discrete_two() {
        let env = TMaze::new();
        let space = env.action_space();
        assert_eq!(space.shape, vec![NUM_ACTIONS]);
        assert!(matches!(space.space_type, SpaceType::Discrete(NUM_ACTIONS)));
    }

    #[test]
    fn cue_is_only_visible_at_step_zero() {
        // The cue channel (obs[0]) must be nonzero at step 0 and zero at every
        // subsequent step, all the way to and including the junction.
        for seed in 0..16u64 {
            let mut env = TMaze::with_seed_and_corridor_length(seed, 10);
            let obs = walk_to_junction(&mut env);
            assert_ne!(obs[0][0], 0.0, "cue must be visible at step 0 (seed {seed})");
            for (i, o) in obs.iter().enumerate().skip(1) {
                assert_eq!(o[0], 0.0, "cue leaked at step {i} (seed {seed}): {o:?}");
            }
        }
    }

    #[test]
    fn junction_observation_is_identical_across_cues() {
        // The no-leak property: the junction observation must be the SAME for
        // an up-cue episode and a down-cue episode. This is what makes a
        // memoryless policy provably at chance.
        let mut up_env = None;
        let mut down_env = None;
        // Find one seed producing each cue.
        for seed in 0..64u64 {
            let env = TMaze::with_seed_and_corridor_length(seed, 8);
            match env.cue() {
                Cue::Up if up_env.is_none() => up_env = Some(env),
                Cue::Down if down_env.is_none() => down_env = Some(env),
                _ => {}
            }
            if up_env.is_some() && down_env.is_some() {
                break;
            }
        }
        let mut up_env = up_env.expect("some seed yields an up cue");
        let mut down_env = down_env.expect("some seed yields a down cue");
        assert_eq!(up_env.cue(), Cue::Up);
        assert_eq!(down_env.cue(), Cue::Down);

        let up_obs = walk_to_junction(&mut up_env);
        let down_obs = walk_to_junction(&mut down_env);
        // Step 0 differs (the cue), every later step (incl. junction) matches.
        assert_ne!(up_obs[0], down_obs[0], "cue must differ at step 0");
        let junction_up = up_obs.last().unwrap();
        let junction_down = down_obs.last().unwrap();
        assert_eq!(
            junction_up, junction_down,
            "junction observation must be identical across cues (no leak)"
        );
        assert_eq!(junction_up, &vec![0.0, 0.0, 1.0], "junction obs is [0,0,1]");
    }

    #[test]
    fn corridor_and_junction_channels_track_position() {
        let mut env = TMaze::with_seed_and_corridor_length(3, 5);
        let obs = walk_to_junction(&mut env);
        // obs has N+1 entries (steps 0..=N). Corridor channel is 1 for the
        // first N, then 0 at the junction; junction channel is the complement.
        for (i, o) in obs.iter().enumerate() {
            if i < env.corridor_length() {
                assert_eq!(o[1], 1.0, "corridor channel should be 1 at step {i}");
                assert_eq!(o[2], 0.0, "junction channel should be 0 at step {i}");
            } else {
                assert_eq!(o[1], 0.0, "corridor channel should be 0 at junction");
                assert_eq!(o[2], 1.0, "junction channel should be 1 at junction");
            }
        }
    }

    #[test]
    fn episode_length_is_corridor_length_plus_one() {
        for n in [1usize, 5, 10, 20] {
            let mut env = TMaze::with_seed_and_corridor_length(1, n);
            env.reset();
            let mut steps = 0;
            loop {
                let r = env.step(UP);
                steps += 1;
                if r.terminated || r.truncated {
                    break;
                }
                assert!(steps <= n + 1, "episode overran for N={n}");
            }
            // The junction decision is the (N+1)-th step (positions advance N
            // times through the corridor, then one decision step).
            assert_eq!(steps, n + 1, "episode length must be N+1 for N={n}");
        }
    }

    #[test]
    fn correct_turn_rewards_plus_one_and_terminates() {
        // Drive both cue values and check the junction reward semantics.
        for seed in 0..32u64 {
            let mut env = TMaze::with_seed_and_corridor_length(seed, 6);
            env.reset();
            let cue = env.cue();
            for _ in 0..env.corridor_length() {
                let r = env.step(UP);
                assert_eq!(r.reward, 0.0, "corridor steps yield zero reward");
                assert!(!r.terminated);
            }
            // Now at the junction: the correct action rewards +1.
            let r = env.step(cue.correct_action());
            assert_eq!(r.reward, REWARD_CORRECT, "correct turn rewards +1");
            assert!(r.terminated, "junction decision terminates the episode");
            assert!(!r.truncated);
        }
    }

    #[test]
    fn wrong_turn_rewards_minus_one_and_terminates() {
        for seed in 0..32u64 {
            let mut env = TMaze::with_seed_and_corridor_length(seed, 6);
            env.reset();
            let cue = env.cue();
            for _ in 0..env.corridor_length() {
                env.step(UP);
            }
            let wrong = 1 - cue.correct_action();
            let r = env.step(wrong);
            assert_eq!(r.reward, REWARD_WRONG, "wrong turn rewards -1");
            assert!(r.terminated);
        }
    }

    #[test]
    fn corridor_actions_are_noops() {
        // The action taken in the corridor must not affect position, reward, or
        // the eventual junction outcome — only the junction action matters.
        let mut a = TMaze::with_seed_and_corridor_length(11, 7);
        let mut b = TMaze::with_seed_and_corridor_length(11, 7);
        a.reset();
        b.reset();
        assert_eq!(a.cue(), b.cue());
        let cue = a.cue();
        // a walks with UP actions, b walks with DOWN actions.
        for _ in 0..a.corridor_length() {
            let ra = a.step(UP);
            let rb = b.step(DOWN);
            assert_eq!(ra.observation, rb.observation, "corridor obs independent of action");
            assert_eq!(ra.reward, rb.reward);
            assert_eq!(a.position(), b.position());
        }
        // Both reach the junction identically; the correct turn still pays +1.
        assert!(a.at_junction() && b.at_junction());
        assert_eq!(a.step(cue.correct_action()).reward, REWARD_CORRECT);
        assert_eq!(b.step(cue.correct_action()).reward, REWARD_CORRECT);
    }

    #[test]
    fn cue_sequence_is_deterministic_under_seed() {
        // Two envs with the same seed draw the same cue on every reset.
        let mut a = TMaze::with_seed_and_corridor_length(42, 4);
        let mut b = TMaze::with_seed_and_corridor_length(42, 4);
        for _ in 0..100 {
            a.reset();
            b.reset();
            assert_eq!(a.cue(), b.cue(), "cue sequence diverged under identical seeds");
        }
    }

    #[test]
    fn cue_is_approximately_balanced() {
        // Over many resets the cue should be ~50/50 up/down (uniform draw).
        let mut env = TMaze::with_seed_and_corridor_length(123, 4);
        let mut up = 0usize;
        let n = 5000;
        for _ in 0..n {
            env.reset();
            if env.cue() == Cue::Up {
                up += 1;
            }
        }
        let rate = up as f64 / n as f64;
        assert!((rate - 0.5).abs() < 0.05, "up-cue rate {rate} should be ≈ 0.5");
    }

    #[test]
    fn clone_restore_reproduces_cue_stream() {
        // Snapshot captures the cue RNG, so restore + reset reproduces the same
        // subsequent cue sequence.
        let mut env = TMaze::with_seed_and_corridor_length(555, 5);
        for _ in 0..5 {
            env.reset();
        }
        let snap = env.clone_state();

        let mut first = Vec::new();
        for _ in 0..20 {
            env.reset();
            first.push(env.cue());
        }

        env.restore_state(&snap);
        let mut second = Vec::new();
        for _ in 0..20 {
            env.reset();
            second.push(env.cue());
        }
        assert_eq!(first, second, "restore must reproduce the cue stream");
    }

    #[test]
    #[should_panic(expected = "corridor_length must be >= 1")]
    fn zero_corridor_length_panics() {
        let _ = TMaze::with_corridor_length(0);
    }

    #[test]
    fn correct_action_mapping() {
        assert_eq!(Cue::Up.correct_action(), UP);
        assert_eq!(Cue::Down.correct_action(), DOWN);
    }
}
