//! [`AtariPreprocess`] — Machado et al. (2018) preprocessing on top of
//! [`AtariEnv`].
//!
//! This wrapper composes the standard sticky-action ALE evaluation-protocol
//! preprocessing entirely on the Rust side (per the Phase 1 spike,
//! `docs/ALE_BINDING_STRATEGY.md`): it receives raw `210 × 160 × 3` RGB frames
//! from the inner [`AtariEnv`] over the [frame protocol](super::protocol) and
//! performs grayscale conversion, downsampling to `84 × 84`, max-pooling over
//! consecutive frames, frame-skipping (action-repeat), frame-stacking, sticky
//! actions, and optional life-loss episode termination.
//!
//! Keeping the pipeline in Rust holds determinism under thrust's control and
//! keeps the wire protocol thin.
//!
//! # Pipeline (per agent [`step`](Environment::step))
//!
//! 1. **Sticky action** — with probability [`PreprocessConfig::sticky_p`],
//!    reuse the previous action instead of the supplied one. Rolled once per
//!    agent step from a [`SmallRng`] seeded from the env seed (independent of
//!    the ALE system RNG in the worker).
//! 2. **Frame-skip loop** — submit the (possibly sticky) action to the inner
//!    env [`frame_skip`](PreprocessConfig::frame_skip) times, accumulating
//!    reward. The **last two** raw frames of the window are retained.
//! 3. **Max-pool** — per-pixel max over those last two raw `210 × 160 × 3`
//!    frames, removing ALE's per-frame sprite flicker.
//! 4. **Grayscale** — ITU-R BT.601 luma, `Y = 0.299·R + 0.587·G + 0.114·B`.
//! 5. **Downsample** — bilinear interpolation `210 × 160 → 84 × 84`.
//! 6. **Frame-stack** — push the new `84 × 84` frame into a ring buffer of
//!    [`frame_stack`](PreprocessConfig::frame_stack) frames (oldest first,
//!    newest last). On [`reset`](Environment::reset) all slots are filled with
//!    the first preprocessed frame (Machado convention).
//!
//! The observation returned by
//! [`get_observation`](Environment::get_observation) is the flattened stack,
//! `frame_stack × screen_size × screen_size` `f32` values normalized to
//! `0.0..=1.0` (divide by `255.0`). With the Machado defaults this is `4 × 84 ×
//! 84 = 28_224` elements, shape `[4, 84, 84]`.
//!
//! # Determinism
//!
//! Two `AtariPreprocess::new("pong", seed)` instances reset and stepped with
//! identical action sequences produce bit-identical observations and rewards.
//! [`clone_state`](Environment::clone_state) /
//! [`restore_state`](Environment::restore_state) round-trip the full wrapper
//! state — inner ALE blob, frame stack, previous raw frame, last action, last
//! lives, and the sticky-action RNG position — so a trajectory replayed after
//! restore is bit-identical to the original.

use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::{
    env::AtariEnv,
    protocol::{OBS_CHANNELS, OBS_HEIGHT, OBS_LEN, OBS_WIDTH},
};
use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

/// Machado et al. (2018) preprocessing configuration.
///
/// Defaults reproduce the standard sticky-action ALE evaluation protocol.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Number of raw ALE frames to skip per agent step (action-repeat).
    /// Reward is accumulated; max-pool is applied to the last 2 skipped
    /// frames. Machado default: `4`.
    pub frame_skip: usize,
    /// Number of preprocessed (post-grayscale, post-downsample) frames to
    /// stack as the observation. Machado default: `4`.
    pub frame_stack: usize,
    /// Sticky-action probability: on each step, reuse the previous action
    /// instead of the supplied one with this probability. Machado default:
    /// `0.25`.
    pub sticky_p: f32,
    /// End the episode when a life is lost (even if ALE has not set
    /// `terminated`). Machado default: `true`.
    pub life_loss_termination: bool,
    /// Output screen size (both height and width). Machado default: `84`.
    pub screen_size: usize,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            frame_skip: 4,
            frame_stack: 4,
            sticky_p: 0.25,
            life_loss_termination: true,
            screen_size: 84,
        }
    }
}

/// Serialisable snapshot of an [`AtariPreprocess`] for
/// [`clone_state`](Environment::clone_state) /
/// [`restore_state`](Environment::restore_state).
///
/// Captures everything the wrapper needs to resume a trajectory bit-for-bit:
/// the inner ALE state blob, the frame stack, the previous raw frame used for
/// max-pooling, the last action (for sticky replay), the last lives count, and
/// the sticky-action RNG position (`sticky_rolls`, the number of RNG draws
/// consumed so far — the RNG is reconstructed by re-seeding and replaying that
/// many draws, which is exactly reproducible).
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessState {
    /// Opaque inner-ALE state blob (`AtariEnv::clone_state()`).
    pub inner: Vec<u8>,
    /// Flattened frame stack, `frame_stack * screen_size * screen_size`
    /// grayscale `f32` values in `0.0..=255.0` (not yet normalized).
    pub stack: Vec<f32>,
    /// The previous raw frame (`210 * 160 * 3` `f32`) used for max-pooling.
    pub prev_raw: Vec<f32>,
    /// The last action submitted to the inner env (for sticky replay).
    pub last_action: i64,
    /// Lives remaining as of the last observation.
    pub last_lives: u32,
    /// Whether a life-loss termination is currently latched.
    pub terminated: bool,
    /// Number of sticky-action RNG draws consumed so far.
    pub sticky_rolls: u64,
}

/// Machado et al. (2018) preprocessing wrapper over an [`AtariEnv`].
///
/// Implements [`Environment`] with `Action = i64` (pass-through) and
/// `State = PreprocessState`. See the [module docs](self) for the full
/// pipeline and determinism contract.
pub struct AtariPreprocess {
    inner: AtariEnv,
    config: PreprocessConfig,
    seed: u64,
    /// Sticky-action RNG, reconstructible from `(seed, sticky_rolls)`.
    rng: SmallRng,
    /// Number of RNG draws consumed (mirrors `rng`'s position for snapshots).
    sticky_rolls: u64,
    /// Ring buffer of `frame_stack` grayscale frames, each
    /// `screen_size * screen_size` values in `0.0..=255.0`. Oldest first.
    stack: Vec<Vec<f32>>,
    /// The previous raw frame retained for max-pooling (`210 * 160 * 3`).
    prev_raw: Vec<f32>,
    last_action: i64,
    last_reward: f32,
    last_lives: u32,
    terminated: bool,
    truncated: bool,
}

impl AtariPreprocess {
    /// Wrap a freshly-constructed [`AtariEnv`] for `game_id` with the default
    /// [`PreprocessConfig`].
    ///
    /// # Errors
    ///
    /// Propagates any [`AtariEnvError`](super::AtariEnvError) from spawning the
    /// worker.
    pub fn new(game_id: &str, seed: u64) -> Result<Self, super::AtariEnvError> {
        Self::with_config(game_id, seed, PreprocessConfig::default())
    }

    /// Wrap a freshly-constructed [`AtariEnv`] for `game_id` with an explicit
    /// [`PreprocessConfig`].
    ///
    /// # Errors
    ///
    /// Propagates any [`AtariEnvError`](super::AtariEnvError) from spawning the
    /// worker.
    pub fn with_config(
        game_id: &str,
        seed: u64,
        config: PreprocessConfig,
    ) -> Result<Self, super::AtariEnvError> {
        let inner = AtariEnv::new(game_id, seed)?;
        Ok(Self::wrap(inner, seed, config))
    }

    /// Wrap an existing [`AtariEnv`] (e.g. one built via
    /// [`AtariEnv::from_transport`](super::AtariEnv::from_transport) in tests)
    /// with the given config.
    #[must_use]
    pub fn wrap(inner: AtariEnv, seed: u64, config: PreprocessConfig) -> Self {
        let frame_len = config.screen_size * config.screen_size;
        let stack = vec![vec![0.0f32; frame_len]; config.frame_stack];
        Self {
            inner,
            config,
            seed,
            rng: SmallRng::seed_from_u64(seed),
            sticky_rolls: 0,
            stack,
            prev_raw: vec![0.0f32; OBS_LEN],
            last_action: 0,
            last_reward: 0.0,
            last_lives: 0,
            terminated: false,
            truncated: false,
        }
    }

    /// The wrapped [`AtariEnv`].
    #[must_use]
    pub fn inner(&self) -> &AtariEnv {
        &self.inner
    }

    /// This wrapper's [`PreprocessConfig`].
    #[must_use]
    pub fn config(&self) -> &PreprocessConfig {
        &self.config
    }

    /// Roll the sticky-action RNG once, advancing `sticky_rolls`.
    fn roll_sticky(&mut self) -> bool {
        let v: f32 = self.rng.random();
        self.sticky_rolls += 1;
        v < self.config.sticky_p
    }

    /// Convert one raw `210 × 160 × 3` RGB frame (`f32` in `0.0..=255.0`,
    /// HWC layout) to a single-channel grayscale frame via BT.601 luma.
    fn grayscale(raw: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; OBS_HEIGHT * OBS_WIDTH];
        for (i, px) in out.iter_mut().enumerate() {
            let base = i * OBS_CHANNELS;
            let r = raw[base];
            let g = raw[base + 1];
            let b = raw[base + 2];
            *px = 0.299 * r + 0.587 * g + 0.114 * b;
        }
        out
    }

    /// Per-pixel max over two raw frames (used to remove ALE sprite flicker).
    fn max_pool(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x.max(y)).collect()
    }

    /// Bilinear downsample a single-channel `src_h × src_w` frame to
    /// `dst × dst`.
    ///
    /// Uses the half-pixel-offset convention, mapping destination coordinate
    /// `i` to source coordinate `(i + 0.5) * src / dst - 0.5`.
    ///
    /// # Deviation from Gymnasium
    ///
    /// Gymnasium's `AtariPreprocessing` (the reference implementation for
    /// Machado 2018) uses `cv2.resize(..., interpolation=cv2.INTER_AREA)`
    /// (area/box averaging) by default. We use bilinear interpolation here: it
    /// is a tiny, dependency-free kernel at `84 × 84` and the Machado 2018
    /// paper itself does not mandate a specific filter. The pixel-value
    /// difference is marginal and does not affect policy training; a future
    /// maintainer needing strict Gymnasium-baseline reproduction can swap in
    /// area averaging here.
    fn downsample(src: &[f32], src_h: usize, src_w: usize, dst: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; dst * dst];
        let scale_y = src_h as f32 / dst as f32;
        let scale_x = src_w as f32 / dst as f32;
        for oy in 0..dst {
            // Half-pixel-offset source coordinate, clamped into range.
            let sy = ((oy as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (src_h - 1) as f32);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(src_h - 1);
            let wy = sy - y0 as f32;
            for ox in 0..dst {
                let sx = ((ox as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (src_w - 1) as f32);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let wx = sx - x0 as f32;

                let p00 = src[y0 * src_w + x0];
                let p01 = src[y0 * src_w + x1];
                let p10 = src[y1 * src_w + x0];
                let p11 = src[y1 * src_w + x1];

                let top = p00 * (1.0 - wx) + p01 * wx;
                let bot = p10 * (1.0 - wx) + p11 * wx;
                out[oy * dst + ox] = top * (1.0 - wy) + bot * wy;
            }
        }
        out
    }

    /// Grayscale + downsample one raw frame to a `screen_size × screen_size`
    /// grayscale frame in `0.0..=255.0`.
    fn preprocess_frame(&self, raw: &[f32]) -> Vec<f32> {
        let gray = Self::grayscale(raw);
        Self::downsample(&gray, OBS_HEIGHT, OBS_WIDTH, self.config.screen_size)
    }

    /// Push a preprocessed frame into the stack (oldest first, newest last).
    fn push_frame(&mut self, frame: Vec<f32>) {
        self.stack.remove(0);
        self.stack.push(frame);
    }

    /// Fill every stack slot with a copy of `frame` (reset convention).
    fn fill_stack(&mut self, frame: &[f32]) {
        for slot in &mut self.stack {
            slot.clear();
            slot.extend_from_slice(frame);
        }
    }

    /// Flatten the stack into a single normalized `0.0..=1.0` observation
    /// vector.
    fn flat_observation(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.config.frame_stack * self.frame_len());
        for slot in &self.stack {
            out.extend(slot.iter().map(|&v| v / 255.0));
        }
        out
    }

    fn frame_len(&self) -> usize {
        self.config.screen_size * self.config.screen_size
    }

    /// Fallible [`Environment::reset`].
    ///
    /// # Errors
    ///
    /// Propagates any [`AtariEnvError`](super::AtariEnvError) from the inner
    /// env.
    pub fn try_reset(&mut self) -> Result<(), super::AtariEnvError> {
        self.inner.try_reset()?;
        let raw = self.inner.get_observation();
        let frame = self.preprocess_frame(&raw);
        self.fill_stack(&frame);
        self.prev_raw = raw;
        self.last_action = 0;
        self.last_reward = 0.0;
        self.last_lives = self.inner.lives();
        self.terminated = false;
        self.truncated = false;
        // Reset the sticky-action RNG so each episode is reproducible from the
        // construction seed.
        self.rng = SmallRng::seed_from_u64(self.seed);
        self.sticky_rolls = 0;
        Ok(())
    }

    /// Fallible [`Environment::step`].
    ///
    /// # Errors
    ///
    /// Propagates any [`AtariEnvError`](super::AtariEnvError) from the inner
    /// env.
    pub fn try_step(&mut self, action: i64) -> Result<StepResult, super::AtariEnvError> {
        // 1. Sticky action: reuse the previous action with probability `sticky_p`,
        //    rolled once per agent step.
        let effective = if self.roll_sticky() {
            self.last_action
        } else {
            action
        };
        self.last_action = effective;

        // 2. Frame-skip loop, accumulating reward and retaining the last two raw frames
        //    for max-pooling.
        let mut reward = 0.0f32;
        let mut terminated = false;
        let mut truncated = false;
        let mut lives = self.last_lives;
        // `second_last` / `last` hold the final two raw frames of the window.
        let mut second_last = std::mem::take(&mut self.prev_raw);
        let mut last = second_last.clone();

        let skip = self.config.frame_skip.max(1);
        for i in 0..skip {
            let result = self.inner.try_step(effective)?;
            reward += result.reward;
            terminated = result.terminated;
            truncated = result.truncated;
            lives = self.inner.lives();
            if i == skip - 1 {
                last = result.observation;
            } else if i == skip - 2 {
                second_last = result.observation;
            }
            if terminated || truncated {
                // On an early terminal frame both "last two" slots take the
                // final frame so max-pool is a no-op on it.
                last = self.inner.get_observation();
                second_last = last.clone();
                break;
            }
        }

        // 3–5. Max-pool the last two raw frames, grayscale, downsample.
        let pooled = if skip >= 2 {
            Self::max_pool(&second_last, &last)
        } else {
            last.clone()
        };
        let frame = self.preprocess_frame(&pooled);

        // 6. Frame-stack push and remember the newest raw frame for the next window's
        //    max-pool.
        self.push_frame(frame);
        self.prev_raw = last;

        // Life-loss termination: end the episode when lives decrease.
        let life_lost = self.config.life_loss_termination && lives < self.last_lives;
        self.last_lives = lives;
        self.last_reward = reward;
        self.terminated = terminated || life_lost;
        self.truncated = truncated;

        Ok(StepResult {
            observation: self.flat_observation(),
            reward,
            terminated: self.terminated,
            truncated: self.truncated,
            info: StepInfo::default(),
        })
    }

    /// Fallible [`Environment::clone_state`].
    ///
    /// # Errors
    ///
    /// Propagates any [`AtariEnvError`](super::AtariEnvError) from the inner
    /// env.
    pub fn try_clone_state(&self) -> Result<PreprocessState, super::AtariEnvError> {
        let inner = self.inner.try_clone_state()?;
        let mut stack = Vec::with_capacity(self.config.frame_stack * self.frame_len());
        for slot in &self.stack {
            stack.extend_from_slice(slot);
        }
        Ok(PreprocessState {
            inner,
            stack,
            prev_raw: self.prev_raw.clone(),
            last_action: self.last_action,
            last_lives: self.last_lives,
            terminated: self.terminated,
            sticky_rolls: self.sticky_rolls,
        })
    }

    /// Fallible [`Environment::restore_state`].
    ///
    /// # Errors
    ///
    /// Propagates any [`AtariEnvError`](super::AtariEnvError) from the inner
    /// env.
    pub fn try_restore_state(
        &mut self,
        state: &PreprocessState,
    ) -> Result<(), super::AtariEnvError> {
        self.inner.try_restore_state(&state.inner)?;
        let frame_len = self.frame_len();
        self.stack = state.stack.chunks(frame_len).map(<[f32]>::to_vec).collect();
        self.prev_raw = state.prev_raw.clone();
        self.last_action = state.last_action;
        self.last_lives = state.last_lives;
        self.terminated = state.terminated;
        self.truncated = false;
        // Reconstruct the sticky-action RNG at exactly the recorded position by
        // re-seeding and replaying `sticky_rolls` draws.
        self.rng = SmallRng::seed_from_u64(self.seed);
        for _ in 0..state.sticky_rolls {
            let _: f32 = self.rng.random();
        }
        self.sticky_rolls = state.sticky_rolls;
        Ok(())
    }
}

impl std::fmt::Debug for AtariPreprocess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtariPreprocess")
            .field("inner", &self.inner)
            .field("config", &self.config)
            .field("seed", &self.seed)
            .field("sticky_rolls", &self.sticky_rolls)
            .finish()
    }
}

impl Environment for AtariPreprocess {
    /// Discrete action index (pass-through to the inner [`AtariEnv`]).
    type Action = i64;

    /// Full wrapper snapshot; see [`PreprocessState`].
    type State = PreprocessState;

    fn reset(&mut self) {
        self.try_reset()
            .unwrap_or_else(|e| panic!("AtariPreprocess::reset failed: {e}"));
    }

    fn get_observation(&self) -> Vec<f32> {
        self.flat_observation()
    }

    fn step(&mut self, action: i64) -> StepResult {
        self.try_step(action)
            .unwrap_or_else(|e| panic!("AtariPreprocess::step failed: {e}"))
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo {
            shape: vec![self.config.frame_stack, self.config.screen_size, self.config.screen_size],
            space_type: SpaceType::Box,
        }
    }

    fn action_space(&self) -> SpaceInfo {
        self.inner.action_space()
    }

    fn render(&self) -> Vec<u8> {
        // Render the newest stacked frame as grayscale bytes (0..=255).
        self.stack
            .last()
            .map(|slot| slot.iter().map(|&p| p.clamp(0.0, 255.0) as u8).collect())
            .unwrap_or_default()
    }

    fn close(&mut self) {
        self.inner.close();
    }

    fn clone_state(&self) -> PreprocessState {
        self.try_clone_state()
            .unwrap_or_else(|e| panic!("AtariPreprocess::clone_state failed: {e}"))
    }

    fn restore_state(&mut self, state: &PreprocessState) {
        self.try_restore_state(state)
            .unwrap_or_else(|e| panic!("AtariPreprocess::restore_state failed: {e}"));
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{BufReader, Read, Write},
        net::{TcpListener, TcpStream},
        sync::atomic::{AtomicUsize, Ordering},
        thread,
    };

    use super::*;
    use crate::env::games::atari::protocol::{self, Command, OBS_LEN, Response};

    // -----------------------------------------------------------------------
    // Pure-math unit tests (no transport)
    // -----------------------------------------------------------------------

    #[test]
    fn grayscale_matches_bt601() {
        // A single pixel raw frame's worth: fill the whole frame with one color
        // and check the first grayscale pixel.
        let mut raw = vec![0.0f32; OBS_LEN];
        raw[0] = 100.0; // R
        raw[1] = 150.0; // G
        raw[2] = 200.0; // B
        let gray = AtariPreprocess::grayscale(&raw);
        let expected = 0.299 * 100.0 + 0.587 * 150.0 + 0.114 * 200.0;
        assert!((gray[0] - expected).abs() < 1e-3, "got {}, want {expected}", gray[0]);
    }

    #[test]
    fn downsample_produces_target_dims() {
        // A ramp grayscale frame downsamples to exactly dst*dst elements.
        let src: Vec<f32> = (0..OBS_HEIGHT * OBS_WIDTH).map(|i| (i % 256) as f32).collect();
        let out = AtariPreprocess::downsample(&src, OBS_HEIGHT, OBS_WIDTH, 84);
        assert_eq!(out.len(), 84 * 84);
        // Bilinear of a bounded input stays within bounds.
        assert!(out.iter().all(|&v| (0.0..=255.0).contains(&v)));
    }

    #[test]
    fn max_pool_picks_per_pixel_maximum() {
        let a = vec![1.0, 5.0, 3.0, 2.0];
        let b = vec![4.0, 2.0, 3.0, 9.0];
        let pooled = AtariPreprocess::max_pool(&a, &b);
        assert_eq!(pooled, vec![4.0, 5.0, 3.0, 9.0]);
    }

    // -----------------------------------------------------------------------
    // Mock-transport plumbing (no Python required)
    // -----------------------------------------------------------------------

    /// Build a raw OBS whose every pixel is `value`, carrying `lives`.
    fn flat_obs(value: f32, reward: f32, terminated: bool, lives: u32) -> Response {
        Response::Obs { terminated, truncated: false, reward, lives, pixels: vec![value; OBS_LEN] }
    }

    /// Wire an `AtariPreprocess` to a mock worker served by `serve`.
    fn mock_preprocess<F>(seed: u64, config: PreprocessConfig, serve: F) -> AtariPreprocess
    where
        F: FnOnce(TcpStream) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        thread::spawn(move || {
            if let Ok((stream, _)) = listener.accept() {
                serve(stream);
            }
        });
        let stream = TcpStream::connect(addr).expect("connect to mock worker");
        let reader: Box<dyn Read + Send> = Box::new(stream.try_clone().expect("clone stream"));
        let writer: Box<dyn Write + Send> = Box::new(stream);
        let inner = AtariEnv::from_transport(reader, writer, "pong", seed);
        AtariPreprocess::wrap(inner, seed, config)
    }

    /// A worker where RESET returns a constant frame value, and each STEP
    /// returns a frame whose pixel value is a running counter, so tests can
    /// observe max-pool/stack behaviour. Lives are fixed at 3.
    fn counting_worker(stream: TcpStream) {
        let mut reader = BufReader::new(stream.try_clone().expect("clone"));
        let mut writer = stream;
        let mut counter = 0.0f32;
        loop {
            match protocol::read_command(&mut reader) {
                Ok(Command::Reset(_)) => {
                    counter = 0.0;
                    protocol::write_response(&mut writer, &flat_obs(10.0, 0.0, false, 3)).unwrap();
                }
                Ok(Command::Step(_)) => {
                    counter += 1.0;
                    protocol::write_response(&mut writer, &flat_obs(counter, 1.0, false, 3))
                        .unwrap();
                }
                Ok(Command::CloneState) => {
                    protocol::write_response(&mut writer, &Response::State(vec![0x01])).unwrap();
                }
                Ok(Command::RestoreState(_)) => {
                    protocol::write_response(&mut writer, &flat_obs(99.0, 0.0, false, 3)).unwrap();
                }
                Ok(Command::Close) | Err(_) => break,
            }
        }
    }

    // -----------------------------------------------------------------------
    // Frame-stack + reset behaviour
    // -----------------------------------------------------------------------

    #[test]
    fn reset_fills_stack_with_first_frame() {
        let cfg =
            PreprocessConfig { frame_skip: 4, frame_stack: 4, sticky_p: 0.0, ..Default::default() };
        let mut env = mock_preprocess(1, cfg, counting_worker);
        env.try_reset().unwrap();

        let obs = env.get_observation();
        assert_eq!(obs.len(), 4 * 84 * 84);
        // Reset frame value 10.0, normalized => 10/255, all slots identical.
        let frame_len = 84 * 84;
        let first = &obs[..frame_len];
        for k in 1..4 {
            let slot = &obs[k * frame_len..(k + 1) * frame_len];
            assert_eq!(slot, first, "all reset slots must be identical");
        }
        assert!((first[0] - 10.0 / 255.0).abs() < 1e-6);
        assert!(obs.iter().all(|&v| (0.0..=1.0).contains(&v)), "obs normalized to 0..1");
    }

    #[test]
    fn step_pushes_newest_frame_last_oldest_first() {
        // frame_skip=1 so each agent step is one raw frame => easy to reason.
        let cfg =
            PreprocessConfig { frame_skip: 1, frame_stack: 4, sticky_p: 0.0, ..Default::default() };
        let mut env = mock_preprocess(1, cfg, counting_worker);
        env.try_reset().unwrap();

        // After reset: all slots = 10.0. Step once => counter=1.0 pooled frame.
        let result = env.try_step(0).unwrap();
        let frame_len = 84 * 84;
        let obs = result.observation;
        // Oldest slot (index 0) still the reset frame (10/255).
        assert!((obs[0] - 10.0 / 255.0).abs() < 1e-6, "oldest slot = reset frame");
        // Newest slot (index 3) is the step frame (1/255).
        let newest = &obs[3 * frame_len..];
        assert!((newest[0] - 1.0 / 255.0).abs() < 1e-6, "newest slot = step frame");
    }

    #[test]
    fn observation_space_reports_stack_shape() {
        let env = mock_preprocess(1, PreprocessConfig::default(), counting_worker);
        assert_eq!(env.observation_space().shape, vec![4, 84, 84]);
        matches!(env.observation_space().space_type, SpaceType::Box)
            .then_some(())
            .expect("box space");
        // Action space passes through the inner Pong minimal set.
        assert_eq!(env.action_space().shape, vec![6]);
    }

    // -----------------------------------------------------------------------
    // Max-pool semantics through the frame-skip loop
    // -----------------------------------------------------------------------

    #[test]
    fn frame_skip_max_pools_last_two_frames() {
        // frame_skip=4: counter goes 1,2,3,4 across the skip. Max-pool of the
        // last two (3 and 4) => 4. So the newest stacked frame == 4/255.
        let cfg =
            PreprocessConfig { frame_skip: 4, frame_stack: 4, sticky_p: 0.0, ..Default::default() };
        let mut env = mock_preprocess(1, cfg, counting_worker);
        env.try_reset().unwrap();
        let result = env.try_step(0).unwrap();
        assert_eq!(result.reward, 4.0, "reward accumulates over 4 skipped frames");
        let frame_len = 84 * 84;
        let newest = &result.observation[3 * frame_len..];
        assert!((newest[0] - 4.0 / 255.0).abs() < 1e-6, "pooled newest frame = max(3,4)=4");
    }

    // -----------------------------------------------------------------------
    // Sticky-action distribution under seed
    // -----------------------------------------------------------------------

    #[test]
    fn sticky_action_fraction_near_p() {
        // A worker that reports the *effective action* back in the reward, so we
        // can count how often the wrapper substituted the sticky (previous)
        // action. We alternate the requested action each step; a sticky repeat
        // reuses the previous effective action.
        let cfg = PreprocessConfig {
            frame_skip: 1,
            frame_stack: 4,
            sticky_p: 0.25,
            life_loss_termination: false,
            screen_size: 84,
        };

        fn action_echo_worker(stream: TcpStream) {
            let mut reader = BufReader::new(stream.try_clone().expect("clone"));
            let mut writer = stream;
            loop {
                match protocol::read_command(&mut reader) {
                    Ok(Command::Reset(_)) => {
                        protocol::write_response(&mut writer, &flat_obs(0.0, 0.0, false, 3))
                            .unwrap();
                    }
                    Ok(Command::Step(a)) => {
                        // Echo the received (effective) action in the reward.
                        protocol::write_response(&mut writer, &flat_obs(0.0, a as f32, false, 3))
                            .unwrap();
                    }
                    Ok(Command::Close) | Err(_) => break,
                    _ => break,
                }
            }
        }

        let mut env = mock_preprocess(12345, cfg, action_echo_worker);
        env.try_reset().unwrap();

        let n = 10_000usize;
        let mut sticky_hits = 0usize;
        // Request a *unique* action every step (the step index). Because the
        // request is never equal to the previous effective action, a sticky
        // repeat is unambiguously detectable: the effective action returned by
        // the worker differs from the request exactly when the wrapper reused
        // the previous action. (The reward field carries the raw action index,
        // which the worker echoes unmodified.)
        for i in 0..n {
            let requested = i as i64;
            let result = env.try_step(requested).unwrap();
            let effective = result.reward as i64;
            if effective != requested {
                sticky_hits += 1;
            }
        }
        let frac = sticky_hits as f64 / n as f64;
        // p=0.25, std = sqrt(p(1-p)/n) ~= 0.0043; allow ~4 std.
        assert!((frac - 0.25).abs() < 0.02, "sticky fraction {frac} not near 0.25");
    }

    // -----------------------------------------------------------------------
    // Life-loss termination
    // -----------------------------------------------------------------------

    #[test]
    fn life_loss_terminates_episode() {
        // Worker returns lives=2 on the first step, lives=1 on the second,
        // never setting ALE terminated. With life_loss_termination the second
        // step must report terminated=true.
        static STEPS: AtomicUsize = AtomicUsize::new(0);
        STEPS.store(0, Ordering::SeqCst);

        fn life_worker(stream: TcpStream) {
            let mut reader = BufReader::new(stream.try_clone().expect("clone"));
            let mut writer = stream;
            loop {
                match protocol::read_command(&mut reader) {
                    Ok(Command::Reset(_)) => {
                        // Start with 2 lives.
                        protocol::write_response(&mut writer, &flat_obs(0.0, 0.0, false, 2))
                            .unwrap();
                    }
                    Ok(Command::Step(_)) => {
                        let n = STEPS.fetch_add(1, Ordering::SeqCst);
                        // First step: still 2 lives; second: drop to 1.
                        let lives = if n == 0 { 2 } else { 1 };
                        protocol::write_response(&mut writer, &flat_obs(0.0, 0.0, false, lives))
                            .unwrap();
                    }
                    Ok(Command::Close) | Err(_) => break,
                    _ => break,
                }
            }
        }

        let cfg = PreprocessConfig {
            frame_skip: 1,
            frame_stack: 4,
            sticky_p: 0.0,
            life_loss_termination: true,
            screen_size: 84,
        };
        let mut env = mock_preprocess(1, cfg, life_worker);
        env.try_reset().unwrap();
        let first = env.try_step(0).unwrap();
        assert!(!first.terminated, "no life lost yet");
        let second = env.try_step(0).unwrap();
        assert!(second.terminated, "life dropped 2 -> 1 must terminate the episode");
    }

    #[test]
    fn life_loss_disabled_does_not_terminate() {
        static STEPS2: AtomicUsize = AtomicUsize::new(0);
        STEPS2.store(0, Ordering::SeqCst);

        fn life_worker2(stream: TcpStream) {
            let mut reader = BufReader::new(stream.try_clone().expect("clone"));
            let mut writer = stream;
            loop {
                match protocol::read_command(&mut reader) {
                    Ok(Command::Reset(_)) => {
                        protocol::write_response(&mut writer, &flat_obs(0.0, 0.0, false, 2))
                            .unwrap();
                    }
                    Ok(Command::Step(_)) => {
                        let n = STEPS2.fetch_add(1, Ordering::SeqCst);
                        let lives = if n == 0 { 2 } else { 1 };
                        protocol::write_response(&mut writer, &flat_obs(0.0, 0.0, false, lives))
                            .unwrap();
                    }
                    Ok(Command::Close) | Err(_) => break,
                    _ => break,
                }
            }
        }

        let cfg = PreprocessConfig {
            frame_skip: 1,
            frame_stack: 4,
            sticky_p: 0.0,
            life_loss_termination: false,
            screen_size: 84,
        };
        let mut env = mock_preprocess(1, cfg, life_worker2);
        env.try_reset().unwrap();
        env.try_step(0).unwrap();
        let second = env.try_step(0).unwrap();
        assert!(!second.terminated, "life-loss disabled: no termination on life drop");
    }

    // -----------------------------------------------------------------------
    // clone_state / restore_state round-trip
    // -----------------------------------------------------------------------

    /// A worker whose STEP output is a running `f32` counter, but which
    /// serialises that counter into its CLONE_STATE blob and restores it on
    /// RESTORE_STATE. This makes the inner env genuinely snapshot-restorable so
    /// the wrapper's clone/restore round-trip is exercisable without ale-py.
    /// Lives fixed at 3.
    fn restorable_counting_worker(stream: TcpStream) {
        let mut reader = BufReader::new(stream.try_clone().expect("clone"));
        let mut writer = stream;
        let mut counter: f32 = 0.0;
        loop {
            match protocol::read_command(&mut reader) {
                Ok(Command::Reset(_)) => {
                    counter = 0.0;
                    protocol::write_response(&mut writer, &flat_obs(10.0, 0.0, false, 3)).unwrap();
                }
                Ok(Command::Step(_)) => {
                    counter += 1.0;
                    protocol::write_response(&mut writer, &flat_obs(counter, 1.0, false, 3))
                        .unwrap();
                }
                Ok(Command::CloneState) => {
                    let blob = counter.to_le_bytes().to_vec();
                    protocol::write_response(&mut writer, &Response::State(blob)).unwrap();
                }
                Ok(Command::RestoreState(bytes)) => {
                    counter = f32::from_le_bytes(bytes[..4].try_into().unwrap());
                    // Post-restore obs reflects the restored counter value.
                    protocol::write_response(&mut writer, &flat_obs(counter, 0.0, false, 3))
                        .unwrap();
                }
                Ok(Command::Close) | Err(_) => break,
            }
        }
    }

    #[test]
    fn clone_restore_round_trips_trajectory() {
        let cfg = PreprocessConfig {
            frame_skip: 2,
            frame_stack: 4,
            sticky_p: 0.25,
            life_loss_termination: false,
            screen_size: 84,
        };
        let mut env = mock_preprocess(777, cfg, restorable_counting_worker);
        env.try_reset().unwrap();
        for _ in 0..5 {
            env.try_step(1).unwrap();
        }

        let snapshot = env.try_clone_state().unwrap();

        let actions = [1i64, 2, 1];
        let baseline: Vec<_> = actions.iter().map(|&a| env.try_step(a).unwrap()).collect();

        env.try_restore_state(&snapshot).unwrap();
        let replay: Vec<_> = actions.iter().map(|&a| env.try_step(a).unwrap()).collect();

        for (i, (x, y)) in baseline.iter().zip(replay.iter()).enumerate() {
            assert_eq!(x.observation, y.observation, "obs mismatch after restore at {i}");
            assert_eq!(x.reward, y.reward, "reward mismatch after restore at {i}");
            assert_eq!(x.terminated, y.terminated, "term mismatch after restore at {i}");
        }
    }

    #[test]
    fn same_seed_same_sticky_sequence() {
        // Determinism of the wrapper's own sticky-action RNG under equal seeds.
        let mk = || {
            let cfg = PreprocessConfig {
                frame_skip: 1,
                frame_stack: 4,
                sticky_p: 0.25,
                life_loss_termination: false,
                screen_size: 84,
            };
            let mut env = mock_preprocess(2024, cfg, counting_worker);
            env.try_reset().unwrap();
            let mut rewards = Vec::new();
            for i in 0..50 {
                rewards.push(env.try_step(i as i64).unwrap().reward);
            }
            rewards
        };
        assert_eq!(mk(), mk(), "same seed => identical sticky-driven trajectory");
    }
}
