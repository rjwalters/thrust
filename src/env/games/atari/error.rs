//! Error types for the `env-atari` subprocess adapter.
//!
//! Every variant's [`Display`](std::fmt::Display) text is written to be
//! *actionable*: it names the remedy (install a package, set an env var, run
//! AutoROM) rather than surfacing a bare "file not found". The
//! [`Environment`](crate::env::Environment) trait methods (`reset`, `step`, …)
//! return no `Result`, so on failure they
//! panic with these messages; the fallible `try_*` methods on
//! [`AtariEnv`](super::AtariEnv) return them directly for callers (and tests)
//! that want to inspect the error.

use std::io;

/// Errors raised while spawning or communicating with the `ale-py` worker.
#[derive(Debug, thiserror::Error)]
pub enum AtariEnvError {
    /// The Python interpreter could not be spawned (typically it is not on
    /// `PATH`).
    #[error(
        "failed to launch the Python worker `{path}`: {source}. \
         Install Python 3 and ensure it is on PATH, or set the ATARI_PYTHON \
         environment variable to a valid interpreter."
    )]
    MissingPython {
        /// The interpreter path or name that failed to launch.
        path: String,
        /// The underlying spawn error.
        source: io::Error,
    },

    /// The worker's Python environment does not have `ale-py` importable.
    #[error(
        "the ale-py package is not available in the worker's Python \
         environment ({0}). Install it with: pip install ale-py"
    )]
    MissingAlePy(String),

    /// The requested ROM could not be resolved by the worker.
    #[error(
        "could not resolve the Atari ROM ({0}). Set ALE_ROM_PATH to a \
         directory (or file) containing the ROM, or run \
         `AutoROM --accept-license` to download ROMs into ale-py's ROM directory."
    )]
    MissingRom(String),

    /// The worker sent a message that violated the frame protocol.
    #[error("Atari worker protocol error: {0}")]
    Protocol(String),

    /// The worker reported an error that did not match a more specific variant.
    #[error("Atari worker reported an error: {0}")]
    Worker(String),

    /// An I/O error occurred while reading from or writing to the worker.
    #[error("I/O error communicating with the Atari worker: {0}")]
    Io(#[from] io::Error),
}

impl AtariEnvError {
    /// Classify a free-form error string reported by the worker (an `ERROR`
    /// frame) into the most specific variant. The worker composes a
    /// human-readable message; this maps it onto the typed surface so callers
    /// can `match` on the cause while the [`Display`](std::fmt::Display) text
    /// still names the remedy.
    #[must_use]
    pub fn from_worker_message(msg: String) -> Self {
        let lower = msg.to_lowercase();
        // Check ROM first: a "ROM" mention is the more specific signal, and a
        // ROM error message may legitimately also name ale-py (e.g. "not found
        // in ale-py's ROM registry").
        if lower.contains("rom") {
            AtariEnvError::MissingRom(msg)
        } else if lower.contains("ale-py")
            || lower.contains("ale_py")
            || lower.contains("no module named")
        {
            AtariEnvError::MissingAlePy(msg)
        } else {
            AtariEnvError::Worker(msg)
        }
    }
}
