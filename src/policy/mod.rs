//! Policy and neural network wrappers.
//!
//! After phase 5 of the Burn migration (#82), all policy networks live on
//! the Burn backend. The pure-Rust WASM inference path (`inference` and
//! `universal_inference` modules) is independent of Burn and is
//! available without the `training` feature.

pub mod inference;
pub mod universal_inference;

/// MLP actor-critic policy used by the CartPole / Pong / SimpleBandit
/// PPO trainers.
#[cfg(feature = "training")]
pub mod mlp;

/// Multi-discrete MLP policy used by Bucket Brigade and similar
/// multi-discrete action spaces.
#[cfg(feature = "training")]
pub mod multi_discrete_mlp;

/// Seeded, host-side weight-initialization helpers that make policy
/// construction bit-exact under `PsroConfig::seed` / `NfspConfig::seed`
/// (issue #135). Burn 0.21's `Initializer` has no seed parameter, so we
/// pre-compute the trunk + head weights from an `StdRng` instead.
#[cfg(feature = "training")]
pub mod seeded_init;

/// SAC stochastic Gaussian actor (tanh-squashed) for continuous control.
#[cfg(feature = "training")]
pub mod sac_actor;

/// DQN Q-network with the same MLP backbone as `MlpPolicy` but with a
/// single Q-head.
#[cfg(feature = "training")]
pub mod q_network;

/// 3-conv + 2-fc CNN used by the Snake trainer.
#[cfg(feature = "training")]
pub mod snake_cnn;
