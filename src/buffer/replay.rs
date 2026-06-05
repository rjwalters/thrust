//! Replay buffer for off-policy training (DQN).
//!
//! This module implements a fixed-capacity FIFO experience replay buffer
//! used by [`crate::train::dqn`]. Transitions are stored as flat
//! CPU-side `Vec`s rather than `tch::Tensor`s so the buffer stays
//! WASM-compatible if it ever needs to be exposed there.
//!
//! # Quick example
//!
//! ```ignore
//! use thrust_rl::buffer::replay::{ReplayBuffer, sample};
//! use rand::SeedableRng;
//!
//! let mut buf = ReplayBuffer::new(/* capacity */ 50_000, /* obs_dim */ 4);
//! buf.push(&[0.0; 4], /* action */ 1, /* reward */ 1.0, &[0.1; 4], /* done */ false);
//!
//! let mut rng = rand::rngs::StdRng::seed_from_u64(0);
//! if buf.is_ready(/* min_size */ 1) {
//!     let batch = sample(&buf, 64, &mut rng);
//!     let (obs, act, rew, next_obs, done) = batch.to_tensors(tch::Device::Cpu);
//!     // ... feed (obs, act, rew, next_obs, done) into the DQN trainer ...
//! }
//! ```

pub use sampling::{ReplayBatch, sample};
pub use storage::ReplayBuffer;

mod sampling;
mod storage;
