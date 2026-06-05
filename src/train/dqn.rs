//! Deep Q-Network (DQN) algorithm.
//!
//! This module implements vanilla DQN for discrete-action environments.
//! The trainer maintains an online Q-network, a target Q-network synced
//! by hard copy every `target_update_interval` env steps, and a fixed-
//! capacity FIFO replay buffer ([`crate::buffer::replay::ReplayBuffer`]).
//!
//! # Algorithm Overview
//!
//! ```text
//! Loop forever:
//!   1. Observe s, select a ~ ε-greedy(Q_online(s, ·))
//!   2. Step env → (r, s', done); push (s, a, r, s', done) to replay buffer
//!   3. If buffer.is_ready(min_buffer_size):
//!        sample minibatch B from buffer
//!        y = r + γ · (1 - done) · max_a' Q_target(s', a')
//!        loss = Huber(Q_online(s, a), y)
//!        backprop, clip-grad-norm, optimizer step
//!   4. Every target_update_interval env steps: Q_target ← Q_online
//! ```
//!
//! # Scope (v1)
//!
//! Classic DQN only. Double-Q, dueling heads, prioritized replay,
//! n-step returns, Polyak soft target updates, and CNN-based variants
//! (Snake/Pong) are explicit follow-ups, not part of v1.
//!
//! # References
//!
//! - Mnih et al., *Human-level control through deep reinforcement learning*
//!   ([Nature 2015](https://www.nature.com/articles/nature14236)).
//! - [OpenAI Spinning Up: DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html)

pub use config::DQNConfig;
pub use loss::{compute_dqn_loss, compute_loss, compute_td_target, gather_action_q};
pub use trainer::{DQNStepStats, DQNTrainer};

mod config;
mod loss;
mod trainer;
