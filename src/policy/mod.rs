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

/// DQN Q-network with the same MLP backbone as `MlpPolicy` but with a
/// single Q-head.
#[cfg(feature = "training")]
pub mod q_network;

/// 3-conv + 2-fc CNN used by the Snake trainer.
#[cfg(feature = "training")]
pub mod snake_cnn;
