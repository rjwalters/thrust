//! Guided tutorial series: from `cargo add thrust-rl` to a trained,
//! deployed policy.
//!
//! Each tutorial is a single Markdown file under [`docs/tutorials/`] whose
//! `rust` code blocks are pulled in here with `#[doc = include_str!(...)]`.
//! Because they are rendered as rustdoc, every code block is compiled and
//! run as a doc-test by `cargo test --features training` — so the copy-paste
//! code in the prose can never rot out of sync with the library API. This is
//! the CI-enforced mechanism the tutorial series is built on.
//!
//! The Markdown files are the single source of truth: read them in
//! [`docs/tutorials/`] (rendered nicely on GitHub) or here on docs.rs. The
//! series is ordered by concept dependency — see
//! [`docs/tutorials/README.md`] for the full dependency-ordered outline.
//!
//! [`docs/tutorials/`]: https://github.com/rjwalters/thrust/tree/main/docs/tutorials
//! [`docs/tutorials/README.md`]: https://github.com/rjwalters/thrust/blob/main/docs/tutorials/README.md
//!
//! # Landed tutorials
//!
//! - [`tutorial_01_first_agent`](crate::tutorials::tutorial_01_first_agent) —
//!   Your first agent (SimpleBandit + actor-critic; the rollout → loss → update
//!   loop).
//! - [`tutorial_02_cartpole_ppo`](crate::tutorials::tutorial_02_cartpole_ppo) —
//!   Solving CartPole with PPO (`EnvPool`, GAE, the config surface, reading
//!   learning curves).
//! - [`tutorial_03_dqn`](crate::tutorials::tutorial_03_dqn) — Off-policy
//!   training with DQN (replay buffer, target network, ε-annealing, Double-DQN,
//!   Polyak soft updates; when to prefer DQN over PPO).

/// Tutorial 1 — Your first agent.
///
/// See the rendered Markdown at
/// [`docs/tutorials/01-your-first-agent.md`](https://github.com/rjwalters/thrust/blob/main/docs/tutorials/01-your-first-agent.md).
#[doc = include_str!("../docs/tutorials/01-your-first-agent.md")]
pub mod tutorial_01_first_agent {}

/// Tutorial 2 — Solving CartPole with PPO.
///
/// See the rendered Markdown at
/// [`docs/tutorials/02-cartpole-ppo.md`](https://github.com/rjwalters/thrust/blob/main/docs/tutorials/02-cartpole-ppo.md).
#[doc = include_str!("../docs/tutorials/02-cartpole-ppo.md")]
pub mod tutorial_02_cartpole_ppo {}

/// Tutorial 3 — Off-policy training with DQN.
///
/// See the rendered Markdown at
/// [`docs/tutorials/03-dqn.md`](https://github.com/rjwalters/thrust/blob/main/docs/tutorials/03-dqn.md).
#[doc = include_str!("../docs/tutorials/03-dqn.md")]
pub mod tutorial_03_dqn {}
