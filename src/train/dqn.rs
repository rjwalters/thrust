//! Deep Q-Network (DQN) algorithm.
//!
//! This module implements DQN for discrete-action environments, with the
//! Double-DQN target as the default and optional Polyak (soft) target
//! updates. The trainer maintains an online Q-network, a target
//! Q-network synced either by hard copy every `target_update_interval`
//! env steps or by a per-step Polyak blend (when
//! [`DQNConfig::soft_update_tau`] is `Some`), and a fixed-capacity FIFO
//! replay buffer ([`crate::buffer::replay::ReplayBuffer`]).
//!
//! # Algorithm Overview
//!
//! ```text
//! Loop forever:
//!   1. Observe s, select a ~ ε-greedy(Q_online(s, ·))
//!   2. Step env → (r, s', done); push (s, a, r, s', done) to replay buffer
//!   3. If buffer.is_ready(min_buffer_size):
//!        sample minibatch B from buffer
//!        a* = argmax_a' Q_online(s', a')                 ← Double-DQN
//!        y  = r + γ · (1 - done) · Q_target(s', a*)
//!        loss = Huber(Q_online(s, a), y)
//!        backprop, clip-grad-norm, optimizer step
//!   4. Target sync:
//!        if soft_update_tau = Some(τ):
//!          θ_target ← τ · θ_online + (1 − τ) · θ_target  ← every step
//!        else:
//!          if env_step % target_update_interval == 0:
//!            θ_target ← θ_online                         ← hard copy
//! ```
//!
//! # Scope
//!
//! - **Included**: Double-DQN target (always on), optional Polyak / soft target
//!   updates (off by default).
//! - **Follow-ups**: dueling heads, prioritized replay, n-step returns, and
//!   CNN-based variants (Snake/Pong).
//!
//! # References
//!
//! - Mnih et al., *Human-level control through deep reinforcement learning*
//!   ([Nature 2015](https://www.nature.com/articles/nature14236)).
//! - van Hasselt, Guez, Silver, *Deep Reinforcement Learning with Double
//!   Q-learning* ([AAAI 2016](https://arxiv.org/abs/1509.06461)).
//! - Lillicrap et al., *Continuous control with deep reinforcement
//!   learning* — origin of the Polyak target-update trick for DRL
//!   ([ICLR 2016](https://arxiv.org/abs/1509.02971)).
//! - [OpenAI Spinning Up: DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html)

pub use config::DQNConfig;
pub use loss::{
    compute_dqn_loss, compute_dqn_loss_double, compute_loss, compute_td_target,
    compute_td_target_double, gather_action_q,
};
pub use trainer::{DQNStepStats, DQNTrainer};

mod config;
mod loss;
mod trainer;
