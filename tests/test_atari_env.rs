//! Tests for the `env-atari` subprocess adapter (issue #325).
//!
//! Two tiers:
//!
//! * **Mock-subprocess tests** drive [`AtariEnv`] over an in-process TCP socket
//!   pair whose server speaks the frame protocol. They need no Python and no
//!   `ale-py`, so they run in the normal `--features "training,env-atari"` CI
//!   matrix and cover the happy path plus every error path.
//! * **Integration tests** spawn the real `ale-py` worker and are
//!   runtime-skipped when `ale-py` is not importable, keeping default CI green.
//!
//! The whole file is gated on `env-atari` because `AtariEnv` does not exist
//! without it.
#![cfg(feature = "env-atari")]

use std::{
    io::{BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    thread,
};

use thrust_rl::env::{
    Environment,
    games::atari::{
        AtariEnv, AtariEnvError,
        protocol::{self, Command, OBS_CHANNELS, OBS_HEIGHT, OBS_LEN, OBS_WIDTH, Response},
    },
};

// ---------------------------------------------------------------------------
// Mock worker plumbing (no Python required)
// ---------------------------------------------------------------------------

/// Build an OBS response whose pixels encode `marker` in the first element so a
/// test can assert which command produced it.
fn obs(marker: f32, reward: f32, terminated: bool) -> Response {
    let mut pixels = vec![0.0f32; OBS_LEN];
    pixels[0] = marker;
    Response::Obs { terminated, truncated: false, reward, pixels }
}

/// Spin up a mock worker on a loopback socket and return an [`AtariEnv`] wired
/// to it. `serve` is handed the accepted server-side stream and speaks the
/// protocol until the client hangs up.
fn mock_env<F>(seed: u64, serve: F) -> AtariEnv
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
    AtariEnv::from_transport(reader, writer, "pong", seed)
}

/// A full mock worker: RESET/STEP/RESTORE_STATE reply with OBS, CLONE_STATE
/// replies with a fixed opaque blob, CLOSE (or hang-up) ends the loop.
fn full_worker(stream: TcpStream) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut writer = stream;
    loop {
        match protocol::read_command(&mut reader) {
            Ok(Command::Reset(seed)) => {
                let resp = obs(seed as f32, 0.0, false);
                protocol::write_response(&mut writer, &resp).unwrap();
            }
            Ok(Command::Step(action)) => {
                let resp = obs(action as f32, action as f32, action == 9);
                protocol::write_response(&mut writer, &resp).unwrap();
            }
            Ok(Command::CloneState) => {
                let resp = Response::State(vec![0xAB, 0xCD, 0xEF, 0x01]);
                protocol::write_response(&mut writer, &resp).unwrap();
            }
            Ok(Command::RestoreState(bytes)) => {
                // Echo the first restored byte back in the marker so the caller
                // can confirm the round-trip reached the worker.
                let marker = bytes.first().copied().unwrap_or(0) as f32;
                let resp = obs(marker, 0.0, false);
                protocol::write_response(&mut writer, &resp).unwrap();
            }
            Ok(Command::Close) | Err(_) => break,
        }
    }
}

/// A worker that answers the first command with a single ERROR frame.
fn error_worker(message: &'static str) -> impl FnOnce(TcpStream) + Send + 'static {
    move |stream: TcpStream| {
        let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
        let mut writer = stream;
        if protocol::read_command(&mut reader).is_ok() {
            let _ = protocol::write_response(&mut writer, &Response::Error(message.to_string()));
        }
    }
}

// ---------------------------------------------------------------------------
// Mock-subprocess unit tests (no Python / no ale-py)
// ---------------------------------------------------------------------------

#[test]
fn mock_subprocess_reset_step() {
    let mut env = mock_env(42, full_worker);

    env.try_reset().expect("reset over mock worker");
    let obs = env.get_observation();
    assert_eq!(obs.len(), OBS_LEN, "raw obs must be 210*160*3 elements");
    assert_eq!(obs[0], 42.0, "reset marker should carry the seed");

    let space = env.observation_space();
    assert_eq!(space.shape, vec![OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS]);

    let result = env.try_step(3).expect("step over mock worker");
    assert_eq!(result.observation.len(), OBS_LEN);
    assert_eq!(result.reward, 3.0);
    assert!(!result.terminated);
    assert!(!result.truncated);
    assert_eq!(result.observation[0], 3.0, "step marker should carry the action");
}

#[test]
fn mock_subprocess_clone_restore_round_trip() {
    let mut env = mock_env(7, full_worker);
    env.try_reset().expect("reset");

    let state = env.try_clone_state().expect("clone_state");
    assert_eq!(state, vec![0xAB, 0xCD, 0xEF, 0x01]);

    env.try_restore_state(&state).expect("restore_state");
    // The worker echoes the first restored byte into the obs marker.
    assert_eq!(env.get_observation()[0], 0xAB as f32);
}

#[test]
fn error_missing_python() {
    // Point at an interpreter that cannot exist; spawning must fail with an
    // actionable MissingPython error naming the remedy.
    let err = AtariEnv::spawn_with(
        "/nonexistent/definitely-not-python-42",
        Path::new("envs/atari/ale_worker.py"),
        "pong",
        0,
    )
    .expect_err("spawning a non-existent interpreter must fail");

    assert!(matches!(err, AtariEnvError::MissingPython { .. }));
    let msg = err.to_string();
    assert!(msg.contains("ATARI_PYTHON"), "message should name the remedy: {msg}");
}

#[test]
fn error_missing_ale_py() {
    let mut env =
        mock_env(0, error_worker("ale-py could not be imported (No module named ale_py)"));
    let err = env.try_reset().expect_err("worker reported missing ale-py");
    assert!(matches!(err, AtariEnvError::MissingAlePy(_)), "got {err:?}");
    assert!(
        err.to_string().contains("pip install ale-py"),
        "message should name the remedy: {err}"
    );
}

#[test]
fn error_missing_rom() {
    let mut env = mock_env(0, error_worker("ROM 'pong' not found in ale-py's ROM registry"));
    let err = env.try_reset().expect_err("worker reported missing ROM");
    assert!(matches!(err, AtariEnvError::MissingRom(_)), "got {err:?}");
    assert!(
        err.to_string().contains("ALE_ROM_PATH"),
        "message should name the ALE_ROM_PATH remedy: {err}"
    );
}

#[test]
fn mock_action_space_is_pong_minimal_set() {
    let env = mock_env(0, full_worker);
    let space = env.action_space();
    assert_eq!(space.shape, vec![6]);
    matches!(space.space_type, thrust_rl::env::SpaceType::Discrete(6))
        .then_some(())
        .expect("Pong action space is Discrete(6)");
}

// ---------------------------------------------------------------------------
// Integration tests — real ale-py, runtime-skipped when it is absent
// ---------------------------------------------------------------------------

fn python() -> String {
    std::env::var("ATARI_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn worker_script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("envs/atari/ale_worker.py")
}

fn ale_py_available() -> bool {
    std::process::Command::new(python())
        .args(["-c", "import ale_py"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Build a real ALE env and reset it; return `None` (with a SKIP note) when
/// `ale-py` or the ROM is unavailable so CI without Python stays green.
fn real_env(seed: u64) -> Option<AtariEnv> {
    if !ale_py_available() {
        eprintln!("SKIP: ale-py not installed");
        return None;
    }
    let mut env = match AtariEnv::spawn_with(python(), worker_script(), "pong", seed) {
        Ok(env) => env,
        Err(e) => {
            eprintln!("SKIP: could not spawn ale-py worker ({e})");
            return None;
        }
    };
    match env.try_reset() {
        Ok(()) => Some(env),
        Err(e) => {
            eprintln!("SKIP: could not reset ALE Pong ({e})");
            None
        }
    }
}

#[test]
fn ale_obs_shape() {
    let Some(env) = real_env(42) else { return };
    let obs = env.get_observation();
    assert_eq!(obs.len(), OBS_LEN);
    assert_eq!(env.observation_space().shape, vec![OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS]);
    assert!(obs.iter().all(|&p| (0.0..=255.0).contains(&p)));
}

#[test]
fn ale_seeded_determinism() {
    let Some(a) = real_env(42) else { return };
    let Some(b) = real_env(42) else { return };
    assert_eq!(a.get_observation(), b.get_observation(), "same seed => same first frame");
}

#[test]
fn ale_episode_terminates() {
    let Some(mut env) = real_env(1) else { return };
    let mut terminated = false;
    for _ in 0..100_000 {
        let result = env.step(0);
        if result.terminated {
            terminated = true;
            break;
        }
    }
    assert!(terminated, "a Pong episode should end within 100k NOOP frames");
}

#[test]
fn ale_clone_restore() {
    let Some(mut env) = real_env(7) else { return };
    for _ in 0..5 {
        env.step(0);
    }
    let snapshot = env.clone_state();

    let actions = [2_i64, 3, 2];
    let baseline: Vec<_> = actions.iter().map(|&a| env.step(a)).collect();

    env.restore_state(&snapshot);
    let replay: Vec<_> = actions.iter().map(|&a| env.step(a)).collect();

    for (i, (x, y)) in baseline.iter().zip(replay.iter()).enumerate() {
        assert_eq!(x.observation, y.observation, "obs mismatch after restore at step {i}");
        assert_eq!(x.reward, y.reward, "reward mismatch after restore at step {i}");
        assert_eq!(x.terminated, y.terminated, "terminated mismatch at step {i}");
    }
}
