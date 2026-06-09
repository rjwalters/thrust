//! Replay buffer for off-policy training (DQN).
//!
//! This module implements two flavors of experience replay used by
//! [`crate::train::dqn`]:
//!
//! 1. [`ReplayBuffer`] — a fixed-capacity FIFO buffer with uniform sampling.
//!    The classic DQN recipe.
//! 2. [`PrioritizedReplayBuffer`] — a sum-tree backed buffer with proportional
//!    priority sampling and importance-sampling correction weights (Schaul et
//!    al. 2015).
//!
//! Both store transitions as flat CPU-side `Vec`s rather than Burn tensors
//! so the buffer stays WASM-compatible if it ever needs to be exposed
//! there. Tensor materialization happens via `to_burn_tensors` on the
//! batch types when the trainer pushes them to a device.
//!
//! # Quick example (uniform)
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
//!     let device = Default::default();
//!     let t = batch.to_burn_tensors::<MyBackend>(&device);
//!     // ... feed t.* into the DQN trainer ...
//! }
//! ```
//!
//! # Quick example (prioritized)
//!
//! ```ignore
//! use thrust_rl::buffer::replay::PrioritizedReplayBuffer;
//! use rand::SeedableRng;
//!
//! let mut buf = PrioritizedReplayBuffer::new(50_000, 4, /* α */ 0.6, /* ε */ 1e-6);
//! buf.push(&[0.0; 4], 1, 1.0, &[0.1; 4], false);
//!
//! let mut rng = rand::rngs::StdRng::seed_from_u64(0);
//! if buf.is_ready(1) {
//!     let batch = buf.sample(64, /* β */ 0.4, &mut rng);
//!     let device = Default::default();
//!     let t = batch.to_burn_tensors::<MyBackend>(&device);
//!     // ... compute weighted Smooth-L1 loss, backprop ...
//!     // ... then write the new TD-error magnitudes back:
//!     // buf.update_priorities(&batch.indices, &new_td_errors);
//! }
//! ```

pub use prioritized::{PrioritizedBatch, PrioritizedBurnTensors, PrioritizedReplayBuffer};
pub use sampling::{ReplayBatch, ReplayBurnTensors, sample};
pub use storage::ReplayBuffer;
pub use sum_tree::SumTree;

mod prioritized;
mod sampling;
mod storage;
mod sum_tree;
