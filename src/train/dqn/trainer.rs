//! Burn-backend DQN trainer (phase 3 of the Burn migration, #80).
//!
//! Sibling to [`crate::train::dqn::DQNTrainerBurn`] (tch path). Implements
//! the same Double-DQN target / Smooth-L1 loss / gradient-step recipe
//! but holds the Q-network as a Burn module that flows through the
//! optimizer on every step.
//!
//! # Scope (phase 3)
//!
//! The Burn trainer:
//!
//! - Owns an online Q-network module `Q` and a target Q-network module of the
//!   same type. The trainer is generic over `Q: AutodiffModule<B> + Clone` so
//!   the actual network shape (CartPole MLP, Snake CNN, etc.) ships in phase 4.
//! - Provides ε-greedy action selection through a caller-supplied `greedy_fn`
//!   (the network's forward + argmax). The trainer owns the ε schedule and the
//!   env-step counter — the same as the tch trainer's `select_action` API.
//! - Performs the Smooth-L1 / Double-DQN training step inside
//!   [`DQNTrainerBurn::train_step`]. Caller-supplied closures hand off the
//!   forward pass — exactly the shape the tch trainer uses when `train_step` is
//!   called against an external loss closure.
//! - Pushes transitions into a [`crate::buffer::replay::ReplayBuffer`] and
//!   samples minibatches the same way as the tch trainer.
//! - Soft / Polyak target updates are folded into
//!   [`DQNTrainerBurn::maybe_sync_target`], which the caller invokes with a
//!   blend closure that applies the `tau` from config.
//!
//! # NOT in scope for phase 3
//!
//! - Prioritized replay is intentionally **not** ported in this phase — it
//!   would inflate the LOC delta past the 800-line budget called out on issue
//!   #80. The hooks are still there (`huber_per_sample` is exposed on the loss
//!   module) and phase 4 / 5 can add the IS-weighted path.
//! - Hard / interval-based target sync is implemented as `clone()` of the
//!   online module into the target slot when the env-step counter hits the
//!   interval; the soft path uses Burn's module-tree mutation through the
//!   trainer-provided `soft_blend_fn`.
//!
//! The acceptance test for phase 3 is that `train_cartpole_dqn` can be
//! ported to this trainer with no algorithmic divergence; the
//! prioritized-replay diagnostics live with `crate::train::dqn` until
//! phase 5 drops them.

use anyhow::{Result, anyhow};
use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    prelude::ToElement,
    tensor::{Tensor, backend::AutodiffBackend},
};
use rand::Rng;

use super::{
    config::DQNConfig,
    loss::{compute_loss, compute_td_target, compute_td_target_double, gather_action_q},
};
use crate::{
    buffer::replay::{ReplayBuffer, sample},
    train::optimizer::{BackendOptimizer, BurnOptimizer},
};

/// Per-step training statistics for the Burn DQN trainer.
///
/// Mirrors `crate::train::dqn::DQNStepStats` (tch path) field-for-field
/// where the fields make sense for the Burn / non-prioritized path.
/// Prioritized-replay fields (`beta`, `mean_abs_td_error`) are dropped
/// because the Burn trainer does not yet implement prioritized replay
/// (see module doc).
#[derive(Debug, Clone, Copy)]
pub struct DQNStepStatsBurn {
    /// Mean Smooth-L1 (Huber) loss across the minibatch.
    pub td_loss: f64,
    /// Mean of `Q(s, a)` across the minibatch.
    pub mean_q: f64,
    /// ε used to draw the most recent action.
    pub epsilon: f64,
    /// Replay buffer fill level at the time of this update.
    pub buffer_len: usize,
}

/// Burn-backend DQN trainer.
///
/// Generic over:
/// - `B: AutodiffBackend`,
/// - `Q: AutodiffModule<B> + Clone` — the Q-network module type (single
///   backbone + action-dim head; ports in phase 4),
/// - `O: Optimizer<Q, B>` — the Burn optimizer.
pub struct DQNTrainerBurn<B, Q, O>
where
    B: AutodiffBackend,
    Q: AutodiffModule<B> + Clone,
    O: Optimizer<Q, B>,
{
    config: DQNConfig,
    n_actions: i64,
    online: Option<Q>,
    target: Q,
    optimizer: BurnOptimizer<B, Q, O>,
    buffer: ReplayBuffer,
    device: B::Device,
    total_env_steps: usize,
    total_train_steps: usize,
    total_episodes: usize,
    last_epsilon: f64,
}

impl<B, Q, O> DQNTrainerBurn<B, Q, O>
where
    B: AutodiffBackend,
    Q: AutodiffModule<B> + Clone,
    O: Optimizer<Q, B>,
{
    /// Build a new Burn DQN trainer.
    ///
    /// The caller supplies the online network already initialized; the
    /// trainer clones it once to seed the target network so the two
    /// start byte-equal (matching the tch trainer's
    /// `target.copy_params_from(&online)` semantics).
    pub fn new(
        config: DQNConfig,
        online: Q,
        optimizer: BurnOptimizer<B, Q, O>,
        obs_dim: usize,
        n_actions: i64,
        device: B::Device,
    ) -> Result<Self> {
        config.validate()?;
        if config.prioritized_replay {
            return Err(anyhow!(
                "DQNTrainerBurn does not yet implement prioritized replay (phase 3 \
                 scope on issue #80). Use the tch DQNTrainer or wait for phase 5."
            ));
        }
        let target = online.clone();
        let buffer = ReplayBuffer::new(config.buffer_capacity, obs_dim);
        let last_epsilon = config.epsilon_start;
        Ok(Self {
            config,
            n_actions,
            online: Some(online),
            target,
            optimizer,
            buffer,
            device,
            total_env_steps: 0,
            total_train_steps: 0,
            total_episodes: 0,
            last_epsilon,
        })
    }

    /// Number of discrete actions in this trainer's action space.
    pub fn n_actions(&self) -> i64 {
        self.n_actions
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &DQNConfig {
        &self.config
    }

    /// Borrow the online network. Panics if mid-step.
    pub fn online(&self) -> &Q {
        self.online.as_ref().expect("online network is None mid-step")
    }

    /// Borrow the target network.
    pub fn target(&self) -> &Q {
        &self.target
    }

    /// Borrow the replay buffer.
    pub fn buffer(&self) -> &ReplayBuffer {
        &self.buffer
    }

    /// Mutably borrow the replay buffer.
    pub fn buffer_mut(&mut self) -> &mut ReplayBuffer {
        &mut self.buffer
    }

    /// Number of transitions currently in the buffer.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Caller invokes this once per environment step.
    pub fn increment_env_step(&mut self) {
        self.total_env_steps += 1;
    }

    /// Caller invokes this when an episode terminates / truncates.
    pub fn increment_episodes(&mut self, n: usize) {
        self.total_episodes += n;
    }

    /// Current env-step counter.
    pub fn total_env_steps(&self) -> usize {
        self.total_env_steps
    }

    /// Number of completed gradient updates.
    pub fn total_train_steps(&self) -> usize {
        self.total_train_steps
    }

    /// Number of completed episodes.
    pub fn total_episodes(&self) -> usize {
        self.total_episodes
    }

    /// ε used to draw the most recent action.
    pub fn last_epsilon(&self) -> f64 {
        self.last_epsilon
    }

    /// ε-greedy action selection.
    ///
    /// Computes the current ε using the config's
    /// [`DQNConfig::epsilon_at`] schedule, then either picks a uniform
    /// random action with probability ε or invokes `greedy_fn` for the
    /// argmax-Q action.
    pub fn select_action<R: Rng, F>(&mut self, obs: &[f32], rng: &mut R, greedy_fn: F) -> i64
    where
        F: FnOnce(&Q, &[f32]) -> i64,
    {
        let eps = self.config.epsilon_at(self.total_env_steps);
        self.last_epsilon = eps;
        if rng.random::<f64>() < eps {
            rng.random_range(0..self.n_actions)
        } else {
            greedy_fn(self.online(), obs)
        }
    }

    /// Sync the target network from the online network.
    ///
    /// Two modes:
    /// 1. **Hard sync** (default): clones online → target every
    ///    `target_update_interval` env steps.
    /// 2. **Soft / Polyak** (`config.soft_update_tau = Some(τ)`): the
    ///    caller-supplied `blend_fn` is invoked every step. The caller is
    ///    responsible for implementing the per-parameter blend `θ_target ← τ ·
    ///    θ_online + (1 − τ) · θ_target`. Burn 0.21 does not expose a uniform
    ///    `map_params` API, so the blend is parameterized over the concrete
    ///    module type by the caller.
    pub fn maybe_sync_target<F>(&mut self, blend_fn: F) -> bool
    where
        F: FnOnce(&Q, Q, f64) -> Q,
    {
        match self.config.soft_update_tau {
            Some(tau) => {
                let online = self.online().clone();
                let target = std::mem::replace(&mut self.target, online.clone());
                self.target = blend_fn(&online, target, tau);
                true
            }
            None => {
                if self.total_env_steps > 0
                    && self.total_env_steps.is_multiple_of(self.config.target_update_interval)
                {
                    self.target = self.online().clone();
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Sample a minibatch and run one gradient step against the
    /// Double-DQN TD target.
    ///
    /// Caller supplies two closures:
    /// - `forward_fn(&Q, obs)` — forward pass returning the `[batch,
    ///   n_actions]` Q-values, with grad bearing iff the module is the online
    ///   network.
    /// - `forward_target_fn(&Q, obs)` — forward pass returning the `[batch,
    ///   n_actions]` target-net Q-values. The trainer takes care of detaching
    ///   these for the TD target — caller doesn't need to manage `no_grad`.
    ///
    /// Returns `Ok(None)` if the buffer doesn't yet hold
    /// `min_buffer_size` transitions.
    pub fn train_step<R: Rng, FOnline, FTarget>(
        &mut self,
        rng: &mut R,
        forward_fn: FOnline,
        forward_target_fn: FTarget,
    ) -> Result<Option<DQNStepStatsBurn>>
    where
        FOnline: Fn(&Q, Tensor<B, 2>) -> Tensor<B, 2>,
        FTarget: Fn(&Q, Tensor<B, 2>) -> Tensor<B, 2>,
    {
        if !self.buffer.is_ready(self.config.min_buffer_size) {
            return Ok(None);
        }

        let batch = sample(&self.buffer, self.config.batch_size, rng);
        let buffer_len = self.buffer.len();

        // Lift the replay batch into Burn tensors via the buffer's
        // built-in `to_burn_tensors` helper (phase 2a, #79).
        let t = batch.to_burn_tensors::<B>(&self.device);

        let online = self
            .online
            .take()
            .ok_or_else(|| anyhow!("online network is None; concurrent train_step calls?"))?;

        let q_online_all = forward_fn(&online, t.observations);
        let q_taken = gather_action_q(q_online_all.clone(), t.actions);

        // Target / Double-DQN target — both forward passes use the
        // online + target networks but the resulting target tensor is
        // detached inside `compute_td_target_double`, so no gradient
        // flows through them.
        let next_q_online_all = forward_fn(&online, t.next_observations.clone());
        let next_q_target_all = forward_target_fn(&self.target, t.next_observations);
        let td_target = compute_td_target_double(
            t.rewards,
            t.dones,
            next_q_online_all,
            next_q_target_all,
            self.config.gamma,
        );

        let td_loss = compute_loss(q_taken.clone(), td_target);
        let td_loss_val: f64 = td_loss.clone().into_scalar().to_f64();
        let mean_q_val: f64 = q_taken.mean().into_scalar().to_f64();

        if !td_loss_val.is_finite() {
            return Err(anyhow!("Non-finite TD loss: {}", td_loss_val));
        }

        // Burn optimizer step.
        let grads = td_loss.backward();
        let grads = GradientsParams::from_grads(grads, &online);
        let lr = self.optimizer.learning_rate();
        let online = self.optimizer.inner_mut().step(lr, online, grads);
        self.online = Some(online);

        self.total_train_steps += 1;

        Ok(Some(DQNStepStatsBurn {
            td_loss: td_loss_val,
            mean_q: mean_q_val,
            epsilon: self.last_epsilon,
            buffer_len,
        }))
    }

    /// Vanilla-DQN train step (uses [`compute_td_target`] instead of
    /// the Double-DQN target). Exposed for completeness; the default
    /// `train_step` uses Double-DQN to match the tch trainer's default.
    pub fn train_step_vanilla<R: Rng, FOnline, FTarget>(
        &mut self,
        rng: &mut R,
        forward_fn: FOnline,
        forward_target_fn: FTarget,
    ) -> Result<Option<DQNStepStatsBurn>>
    where
        FOnline: Fn(&Q, Tensor<B, 2>) -> Tensor<B, 2>,
        FTarget: Fn(&Q, Tensor<B, 2>) -> Tensor<B, 2>,
    {
        if !self.buffer.is_ready(self.config.min_buffer_size) {
            return Ok(None);
        }

        let batch = sample(&self.buffer, self.config.batch_size, rng);
        let buffer_len = self.buffer.len();

        let t = batch.to_burn_tensors::<B>(&self.device);

        let online = self
            .online
            .take()
            .ok_or_else(|| anyhow!("online network is None; concurrent train_step calls?"))?;

        let q_online_all = forward_fn(&online, t.observations);
        let q_taken = gather_action_q(q_online_all.clone(), t.actions);
        let next_q_target_all = forward_target_fn(&self.target, t.next_observations);
        let td_target = compute_td_target(t.rewards, t.dones, next_q_target_all, self.config.gamma);

        let td_loss = compute_loss(q_taken.clone(), td_target);
        let td_loss_val: f64 = td_loss.clone().into_scalar().to_f64();
        let mean_q_val: f64 = q_taken.mean().into_scalar().to_f64();

        if !td_loss_val.is_finite() {
            return Err(anyhow!("Non-finite TD loss: {}", td_loss_val));
        }

        let grads = td_loss.backward();
        let grads = GradientsParams::from_grads(grads, &online);
        let lr = self.optimizer.learning_rate();
        let online = self.optimizer.inner_mut().step(lr, online, grads);
        self.online = Some(online);

        self.total_train_steps += 1;

        Ok(Some(DQNStepStatsBurn {
            td_loss: td_loss_val,
            mean_q: mean_q_val,
            epsilon: self.last_epsilon,
            buffer_len,
        }))
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, NdArray},
        optim::AdamConfig,
    };
    use rand::SeedableRng;

    use super::*;
    use crate::{policy::mlp::MlpBurnPolicy, train::optimizer::BurnOptimizer};

    type B = Autodiff<NdArray<f32>>;

    fn small_config() -> DQNConfig {
        DQNConfig::new()
            .buffer_capacity(128)
            .min_buffer_size(8)
            .batch_size(8)
            .target_update_interval(4)
            .epsilon_decay_steps(100)
    }

    /// Smoke test: a Burn DQN trainer constructs without error using
    /// `MlpBurnPolicy` as a Q-network stand-in (phase 4 ports the
    /// proper `QNetworkBurn`).
    #[test]
    fn dqn_trainer_burn_constructs() {
        let device = Default::default();
        let online = MlpBurnPolicy::<B>::new(4, 2, 16, &device);
        let inner_opt = AdamConfig::new().init();
        let burn_opt = BurnOptimizer::new(inner_opt, small_config().learning_rate);
        let trainer = DQNTrainerBurn::new(small_config(), online, burn_opt, 4, 2, device).unwrap();
        assert_eq!(trainer.total_env_steps(), 0);
        assert_eq!(trainer.buffer_len(), 0);
    }

    /// Prioritized replay is intentionally not yet supported on the
    /// Burn path.
    #[test]
    fn dqn_trainer_burn_rejects_prioritized_config() {
        let device = Default::default();
        let online = MlpBurnPolicy::<B>::new(4, 2, 16, &device);
        let inner_opt = AdamConfig::new().init();
        let burn_opt = BurnOptimizer::new(inner_opt, 1e-3);
        let cfg = small_config().prioritized_replay(true);
        assert!(DQNTrainerBurn::new(cfg, online, burn_opt, 4, 2, device).is_err());
    }

    /// End-to-end: pushing transitions and calling `train_step` runs
    /// the Smooth-L1 / Double-DQN gradient step without panicking.
    /// Uses `MlpBurnPolicy` as a Q-network stand-in — its `forward`
    /// returns `(logits, value)`; we use the logits as the Q-values.
    #[test]
    fn dqn_trainer_burn_train_step_runs() {
        let device = Default::default();
        let online = MlpBurnPolicy::<B>::new(4, 2, 16, &device);
        let inner_opt = AdamConfig::new().init();
        let burn_opt = BurnOptimizer::new(inner_opt, 1e-3);
        let mut trainer =
            DQNTrainerBurn::new(small_config(), online, burn_opt, 4, 2, device).unwrap();

        // Push enough transitions to clear min_buffer_size.
        for i in 0..32 {
            let phase = (i as f32) * 0.1;
            let obs = [phase.sin(), phase.cos(), phase * 0.5, phase * -0.3];
            let next_obs = [(phase + 0.1).sin(), (phase + 0.1).cos(), phase * 0.5, phase * -0.3];
            let action = (i % 2) as i64;
            let reward = if action == 0 { 1.0 } else { -1.0 };
            let done = i % 8 == 7;
            trainer.buffer_mut().push(&obs, action, reward, &next_obs, done);
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let forward_fn = |q: &MlpBurnPolicy<B>, o: Tensor<B, 2>| -> Tensor<B, 2> {
            let (logits, _) = q.forward(o);
            logits
        };
        let stats = trainer.train_step(&mut rng, forward_fn, forward_fn).unwrap();
        assert!(stats.is_some());
        let s = stats.unwrap();
        assert!(s.td_loss.is_finite());
    }
}
