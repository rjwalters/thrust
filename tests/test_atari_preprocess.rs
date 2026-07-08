//! Integration tests for the Machado-preprocessing wrapper
//! ([`AtariPreprocess`], issue #326).
//!
//! These spawn the **real** `ale-py` worker and are runtime-skipped when
//! `ale-py` is not importable (mirroring `tests/test_atari_env.rs`), so default
//! CI without Python stays green. The pure-Rust unit tests for the pipeline
//! math, stacking, sticky-action distribution, life-loss termination, and
//! clone/restore round-trip live inline in
//! `src/env/games/atari/preprocess.rs` and run in the normal
//! `--features "training,env-atari"` matrix without Python.
//!
//! The whole file is gated on `env-atari` because `AtariPreprocess` does not
//! exist without it.
#![cfg(feature = "env-atari")]

use std::path::{Path, PathBuf};

use thrust_rl::env::{
    Environment,
    games::atari::{AtariEnv, AtariPreprocess, PreprocessConfig},
};

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

/// Build a real preprocessing env and reset it; return `None` (with a SKIP
/// note) when `ale-py` or the ROM is unavailable so CI without Python stays
/// green.
fn real_preprocess(seed: u64, config: PreprocessConfig) -> Option<AtariPreprocess> {
    if !ale_py_available() {
        eprintln!("SKIP: ale-py not installed");
        return None;
    }
    let inner = match AtariEnv::spawn_with(python(), worker_script(), "pong", seed) {
        Ok(env) => env,
        Err(e) => {
            eprintln!("SKIP: could not spawn ale-py worker ({e})");
            return None;
        }
    };
    let mut env = AtariPreprocess::wrap(inner, seed, config);
    match env.try_reset() {
        Ok(()) => Some(env),
        Err(e) => {
            eprintln!("SKIP: could not reset ALE Pong ({e})");
            None
        }
    }
}

#[test]
fn ale_preprocess_obs_shape_and_range() {
    let Some(env) = real_preprocess(42, PreprocessConfig::default()) else {
        return;
    };
    let obs = env.get_observation();
    assert_eq!(obs.len(), 4 * 84 * 84, "stacked obs must be 28_224 elements");
    assert_eq!(env.observation_space().shape, vec![4, 84, 84]);
    assert!(
        obs.iter().all(|&p| (0.0..=1.0).contains(&p)),
        "preprocessed obs must be normalized to 0..=1"
    );
}

#[test]
fn ale_preprocess_seeded_determinism() {
    let Some(mut a) = real_preprocess(42, PreprocessConfig::default()) else {
        return;
    };
    let Some(mut b) = real_preprocess(42, PreprocessConfig::default()) else {
        return;
    };
    assert_eq!(a.get_observation(), b.get_observation(), "same seed => same first obs");
    for i in 0..20 {
        let ra = a.step(0);
        let rb = b.step(0);
        assert_eq!(ra.observation, rb.observation, "obs mismatch at step {i}");
        assert_eq!(ra.reward, rb.reward, "reward mismatch at step {i}");
        assert_eq!(ra.terminated, rb.terminated, "terminated mismatch at step {i}");
    }
}

#[test]
fn ale_preprocess_episode_terminates() {
    // Life-loss termination is on by default, so a Pong episode should end
    // quickly (first life lost) within a small NOOP budget.
    let Some(mut env) = real_preprocess(1, PreprocessConfig::default()) else {
        return;
    };
    let mut terminated = false;
    for _ in 0..100_000 {
        if env.step(0).terminated {
            terminated = true;
            break;
        }
    }
    assert!(terminated, "a Pong episode should terminate within 100k NOOP steps");
}
