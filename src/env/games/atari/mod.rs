//! Atari (ALE) environment adapter — feature `env-atari`.
//!
//! [`AtariEnv`] implements the [`Environment`](crate::env::Environment) trait
//! by driving Farama's [`ale-py`](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
//! as a **subprocess**: the Rust parent spawns `envs/atari/ale_worker.py` and
//! exchanges seeds, actions, observations and state snapshots over a small
//! length-prefixed binary protocol on the child's stdin/stdout. This is
//! *Option D* from [`docs/ALE_BINDING_STRATEGY.md`](https://github.com/rjwalters/thrust/blob/main/docs/ALE_BINDING_STRATEGY.md);
//! it adds **no** compile-time native dependency, so `env-atari = []` is a pure
//! feature flag and a fresh checkout still builds with no system libraries.
//!
//! # ⚠️ Runtime requirements and licensing NOTICE
//!
//! Building with `--features env-atari` compiles only the Rust client. To
//! *run* it you must, separately, install the emulator in the worker's Python
//! environment:
//!
//! ```text
//! pip install ale-py                          # the ALE emulator + Python API
//! AutoROM --accept-license                     # (if your ale-py does not bundle ROMs)
//! ```
//!
//! **ALE (`ale-py`) is licensed GPL-2.0-or-later.** thrust invokes it as a
//! *separate program* over a pipe — arm's-length runtime aggregation — so no
//! GPL code is linked into, vendored in, or distributed with the thrust crate,
//! and thrust itself stays `MIT OR Apache-2.0`. Enabling this feature is a
//! deliberate choice to depend on that external GPL program at runtime.
//! Atari ROMs are copyrighted and are **never** committed to this repository or
//! shipped in the crate tarball.
//!
//! # ROM resolution
//!
//! Set `ALE_ROM_PATH` to a directory (or file) of ROMs to override lookup; the
//! variable is inherited by the worker process. When it is unset the worker
//! falls back to `ale-py`'s own ROM resolution. An unresolved ROM produces an
//! actionable [`AtariEnvError::MissingRom`] naming both remedies.
//!
//! # Scope (v1)
//!
//! Raw `210 × 160 × 3` RGB observations (as `f32`, `0.0..=255.0`); Pong's
//! 6-action minimal set. Machado preprocessing and a CNN policy compose on top
//! in follow-up issues.
//!
//! ```no_run
//! use thrust_rl::env::{Environment, games::atari::AtariEnv};
//!
//! // Requires `ale-py` installed and a resolvable Pong ROM.
//! let mut env = AtariEnv::new("pong", 42).expect("spawn ale-py worker");
//! env.reset();
//! let obs = env.get_observation();
//! assert_eq!(obs.len(), 210 * 160 * 3);
//! ```

pub mod error;
pub mod preprocess;
pub mod protocol;

mod env;

pub use env::{AtariEnv, DEFAULT_WORKER_SCRIPT, ENV_PYTHON, ENV_WORKER_SCRIPT, PONG_ACTION_COUNT};
pub use error::AtariEnvError;
pub use preprocess::{AtariPreprocess, PreprocessConfig, PreprocessState};
