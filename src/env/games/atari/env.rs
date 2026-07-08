//! [`AtariEnv`] — the subprocess client implementing [`Environment`].
//!
//! `AtariEnv` owns a child `ale-py` process (or, in tests, any transport that
//! speaks the [frame protocol](super::protocol)) and forwards
//! reset/step/clone/restore over the pipe. See the [module docs](super) for the
//! GPL-2.0 runtime NOTICE and the ROM-resolution contract.

use std::{
    cell::RefCell,
    env,
    ffi::OsStr,
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command as ProcCommand, Stdio},
};

use super::{
    error::AtariEnvError,
    protocol::{self, Command, OBS_CHANNELS, OBS_HEIGHT, OBS_LEN, OBS_WIDTH, Response},
};
use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

/// Environment variable naming the Python interpreter to launch. Defaults to
/// `python3` when unset.
pub const ENV_PYTHON: &str = "ATARI_PYTHON";
/// Environment variable naming the `ale_worker.py` script. Defaults to
/// `envs/atari/ale_worker.py` (relative to the current working directory) when
/// unset.
pub const ENV_WORKER_SCRIPT: &str = "ATARI_WORKER_SCRIPT";
/// Default worker-script path, relative to the crate root / cwd.
pub const DEFAULT_WORKER_SCRIPT: &str = "envs/atari/ale_worker.py";

/// Number of discrete actions in Pong's minimal action set (`NOOP`, `FIRE`,
/// `RIGHT`, `LEFT`, `RIGHTFIRE`, `LEFTFIRE`).
pub const PONG_ACTION_COUNT: usize = 6;

/// The reader/writer/child triple, held behind a [`RefCell`] so the
/// `&self`-only [`Environment::clone_state`] can still drive the pipe.
struct Transport {
    reader: BufReader<Box<dyn Read + Send>>,
    writer: Box<dyn Write + Send>,
    child: Option<Child>,
}

/// A subprocess-backed Atari (ALE) environment.
///
/// `AtariEnv` implements [`Environment`] by forwarding each operation to an
/// `ale-py` worker over a length-prefixed binary protocol. The raw
/// `210 × 160 × 3` RGB frame is returned verbatim as `f32` values in
/// `0.0..=255.0`; preprocessing (grayscale, downsample, frame-stack) is a
/// downstream concern.
///
/// # Determinism
///
/// `reset()` forwards the seed captured at construction to the worker, so two
/// `AtariEnv::new("pong", seed)` instances reset to bit-identical first
/// observations. `clone_state()` / `restore_state()` round-trip the full ALE
/// system state, so a trajectory replayed after `restore_state` is
/// bit-identical to the original.
///
/// # Panics
///
/// The [`Environment`] trait methods panic (with an actionable message) if the
/// worker reports an error or the pipe fails. Use the [`AtariEnv::try_reset`],
/// [`AtariEnv::try_step`], etc. variants to handle those errors explicitly.
pub struct AtariEnv {
    io: RefCell<Transport>,
    game_id: String,
    seed: u64,
    last_obs: Vec<f32>,
    last_reward: f32,
    last_terminated: bool,
    last_truncated: bool,
    last_lives: u32,
}

impl AtariEnv {
    /// Spawn the default worker (`ATARI_PYTHON` or `python3`, running
    /// `ATARI_WORKER_SCRIPT` or [`DEFAULT_WORKER_SCRIPT`]) for `game_id`.
    ///
    /// `ALE_ROM_PATH`, if set in this process's environment, is inherited by
    /// the child and used by the worker to resolve the ROM.
    ///
    /// This only launches the process; call [`Environment::reset`] (or
    /// [`AtariEnv::try_reset`]) to drive the first frame. Worker-side failures
    /// (missing `ale-py`, unresolved ROM) surface on that first exchange.
    ///
    /// # Errors
    ///
    /// Returns [`AtariEnvError::MissingPython`] if the interpreter cannot be
    /// launched, or [`AtariEnvError::Io`] for other spawn failures.
    pub fn new(game_id: &str, seed: u64) -> Result<Self, AtariEnvError> {
        let python = env::var(ENV_PYTHON).unwrap_or_else(|_| "python3".to_string());
        let script = env::var(ENV_WORKER_SCRIPT)
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_WORKER_SCRIPT));
        Self::spawn_with(python, script, game_id, seed)
    }

    /// Spawn an explicit `python`/`script` worker for `game_id`. Prefer
    /// [`AtariEnv::new`] in application code; this constructor exists for tests
    /// and callers that resolve the interpreter and script themselves.
    ///
    /// # Errors
    ///
    /// Returns [`AtariEnvError::MissingPython`] if the interpreter cannot be
    /// launched (e.g. the path does not exist), or [`AtariEnvError::Io`] for
    /// other spawn failures.
    pub fn spawn_with(
        python: impl AsRef<OsStr>,
        script: impl AsRef<Path>,
        game_id: &str,
        seed: u64,
    ) -> Result<Self, AtariEnvError> {
        let python_ref = python.as_ref();
        let mut child = ProcCommand::new(python_ref)
            .arg(script.as_ref())
            .arg(game_id)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|source| AtariEnvError::MissingPython {
                path: python_ref.to_string_lossy().into_owned(),
                source,
            })?;

        let writer: Box<dyn Write + Send> =
            Box::new(child.stdin.take().expect("child stdin was requested as piped"));
        let reader: Box<dyn Read + Send> =
            Box::new(child.stdout.take().expect("child stdout was requested as piped"));

        Ok(Self::from_parts(reader, writer, Some(child), game_id, seed))
    }

    /// Build an `AtariEnv` over an arbitrary transport that speaks the
    /// [frame protocol](super::protocol). Used by tests to drive the env with a
    /// mock worker (no Python required); also usable to bind an already-running
    /// worker over any `Read`/`Write` pair (e.g. a socket).
    pub fn from_transport(
        reader: Box<dyn Read + Send>,
        writer: Box<dyn Write + Send>,
        game_id: &str,
        seed: u64,
    ) -> Self {
        Self::from_parts(reader, writer, None, game_id, seed)
    }

    fn from_parts(
        reader: Box<dyn Read + Send>,
        writer: Box<dyn Write + Send>,
        child: Option<Child>,
        game_id: &str,
        seed: u64,
    ) -> Self {
        AtariEnv {
            io: RefCell::new(Transport { reader: BufReader::new(reader), writer, child }),
            game_id: game_id.to_string(),
            seed,
            last_obs: Vec::new(),
            last_reward: 0.0,
            last_terminated: false,
            last_truncated: false,
            last_lives: 0,
        }
    }

    /// The game id this env was constructed with (e.g. `"pong"`).
    #[must_use]
    pub fn game_id(&self) -> &str {
        &self.game_id
    }

    /// The seed forwarded to the worker on [`Environment::reset`].
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Number of lives remaining as of the last observation (`ale.lives()`).
    ///
    /// Updated on every `reset`/`step`/`restore_state`. Used by the
    /// [`AtariPreprocess`](super::AtariPreprocess) wrapper to implement
    /// life-loss episode termination (Machado et al. 2018). Returns `0` before
    /// the first observation.
    #[must_use]
    pub fn lives(&self) -> u32 {
        self.last_lives
    }

    /// Send `cmd` and read the single response frame it elicits.
    ///
    /// Takes `&self` (via interior mutability) so `clone_state` can use it.
    fn exchange(&self, cmd: &Command) -> Result<Response, AtariEnvError> {
        let mut io = self.io.borrow_mut();
        protocol::write_command(&mut io.writer, cmd)?;
        io.writer.flush()?;
        Ok(protocol::read_response(&mut io.reader)?)
    }

    /// Apply an `Obs` response to the cached observation fields, mapping
    /// `Error`/unexpected responses onto typed errors.
    fn apply_obs(&mut self, resp: Response) -> Result<(), AtariEnvError> {
        match resp {
            Response::Obs { terminated, truncated, reward, lives, pixels } => {
                if pixels.len() != OBS_LEN {
                    return Err(AtariEnvError::Protocol(format!(
                        "expected {OBS_LEN} observation elements ({OBS_HEIGHT}×{OBS_WIDTH}×{OBS_CHANNELS}), got {}",
                        pixels.len()
                    )));
                }
                self.last_obs = pixels;
                self.last_reward = reward;
                self.last_terminated = terminated;
                self.last_truncated = truncated;
                self.last_lives = lives;
                Ok(())
            }
            Response::Error(msg) => Err(AtariEnvError::from_worker_message(msg)),
            Response::State(_) => Err(AtariEnvError::Protocol(
                "expected an OBS response but the worker sent STATE".to_string(),
            )),
        }
    }

    /// Fallible [`Environment::reset`]: forward the seed and cache the initial
    /// frame.
    ///
    /// # Errors
    ///
    /// Returns an [`AtariEnvError`] if the worker reports an error (missing
    /// `ale-py`, unresolved ROM) or the pipe fails.
    pub fn try_reset(&mut self) -> Result<(), AtariEnvError> {
        let resp = self.exchange(&Command::Reset(self.seed))?;
        self.apply_obs(resp)
    }

    /// Fallible [`Environment::step`].
    ///
    /// # Errors
    ///
    /// Returns an [`AtariEnvError`] if the worker reports an error or the pipe
    /// fails.
    pub fn try_step(&mut self, action: i64) -> Result<StepResult, AtariEnvError> {
        let resp = self.exchange(&Command::Step(action as i32))?;
        self.apply_obs(resp)?;
        Ok(StepResult {
            observation: self.last_obs.clone(),
            reward: self.last_reward,
            terminated: self.last_terminated,
            truncated: self.last_truncated,
            info: StepInfo::default(),
        })
    }

    /// Fallible [`Environment::clone_state`].
    ///
    /// # Errors
    ///
    /// Returns an [`AtariEnvError`] if the worker reports an error or the pipe
    /// fails.
    pub fn try_clone_state(&self) -> Result<Vec<u8>, AtariEnvError> {
        match self.exchange(&Command::CloneState)? {
            Response::State(bytes) => Ok(bytes),
            Response::Error(msg) => Err(AtariEnvError::from_worker_message(msg)),
            Response::Obs { .. } => Err(AtariEnvError::Protocol(
                "expected a STATE response but the worker sent OBS".to_string(),
            )),
        }
    }

    /// Fallible [`Environment::restore_state`].
    ///
    /// # Errors
    ///
    /// Returns an [`AtariEnvError`] if the worker reports an error or the pipe
    /// fails.
    pub fn try_restore_state(&mut self, state: &[u8]) -> Result<(), AtariEnvError> {
        let resp = self.exchange(&Command::RestoreState(state.to_vec()))?;
        self.apply_obs(resp)
    }

    /// Fallible [`Environment::close`]: ask the worker to exit and reap the
    /// child.
    ///
    /// # Errors
    ///
    /// Returns an [`AtariEnvError`] if writing the CLOSE frame fails.
    pub fn try_close(&mut self) -> Result<(), AtariEnvError> {
        {
            let mut io = self.io.borrow_mut();
            protocol::write_command(&mut io.writer, &Command::Close)?;
            io.writer.flush()?;
            if let Some(child) = io.child.as_mut() {
                let _ = child.wait();
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for AtariEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let has_child = self.io.borrow().child.is_some();
        f.debug_struct("AtariEnv")
            .field("game_id", &self.game_id)
            .field("seed", &self.seed)
            .field("obs_len", &self.last_obs.len())
            .field("owns_child", &has_child)
            .finish()
    }
}

impl Drop for AtariEnv {
    fn drop(&mut self) {
        // Best-effort: tell the worker to exit, then make sure the child does
        // not linger as a zombie. Ignore all errors — Drop must not panic.
        let mut io = self.io.borrow_mut();
        let _ = protocol::write_command(&mut io.writer, &Command::Close);
        let _ = io.writer.flush();
        if let Some(mut child) = io.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Environment for AtariEnv {
    /// Discrete action index into the minimal action set.
    type Action = i64;

    /// Opaque ALE system-state blob (`ale.cloneSystemState()`), forwarded
    /// verbatim to and from the worker.
    type State = Vec<u8>;

    fn reset(&mut self) {
        self.try_reset().unwrap_or_else(|e| panic!("AtariEnv::reset failed: {e}"));
    }

    fn get_observation(&self) -> Vec<f32> {
        self.last_obs.clone()
    }

    fn step(&mut self, action: i64) -> StepResult {
        self.try_step(action).unwrap_or_else(|e| panic!("AtariEnv::step failed: {e}"))
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo { shape: vec![OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS], space_type: SpaceType::Box }
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo {
            shape: vec![PONG_ACTION_COUNT],
            space_type: SpaceType::Discrete(PONG_ACTION_COUNT),
        }
    }

    fn render(&self) -> Vec<u8> {
        // Raw frame pixels are already in 0..=255; clamp defensively.
        self.last_obs.iter().map(|&p| p.clamp(0.0, 255.0) as u8).collect()
    }

    fn close(&mut self) {
        let _ = self.try_close();
    }

    fn clone_state(&self) -> Vec<u8> {
        self.try_clone_state()
            .unwrap_or_else(|e| panic!("AtariEnv::clone_state failed: {e}"))
    }

    fn restore_state(&mut self, state: &Vec<u8>) {
        self.try_restore_state(state)
            .unwrap_or_else(|e| panic!("AtariEnv::restore_state failed: {e}"));
    }
}
