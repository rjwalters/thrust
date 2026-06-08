//! DQN Trainer implementation.
//!
//! Holds the online + target Q-networks, the optimizer, the replay
//! buffer, and the counters needed to drive ε-greedy exploration and
//! periodic target-net syncs. The env-interaction loop lives in the
//! caller (typically a `train_*_dqn` example) — `DQNTrainer` only owns
//! the parts that don't depend on the environment.

use anyhow::{Result, anyhow};
use rand::Rng;
use tch::{Device, Kind, Reduction, Tensor};

use super::{config::DQNConfig, loss};
use crate::{
    buffer::replay::{PrioritizedReplayBuffer, ReplayBuffer, sample},
    policy::QNetwork,
};

/// Per-step training statistics returned by [`DQNTrainer::train_step`].
#[derive(Debug, Clone, Copy)]
pub struct DQNStepStats {
    /// Mean Smooth-L1 (Huber) loss across the minibatch. In the
    /// prioritized-replay path this is the IS-weighted mean.
    pub td_loss: f64,
    /// Mean of `Q(s, a)` across the minibatch (useful diagnostic).
    pub mean_q: f64,
    /// ε used to draw the most recent action.
    pub epsilon: f64,
    /// Replay buffer fill level at the time of this update.
    pub buffer_len: usize,
    /// β used to compute the IS weights on this step. `None` in the
    /// uniform-replay path.
    pub beta: Option<f64>,
    /// Mean of the per-sample |TD error| this step, before being
    /// written back as a new priority. `None` in the uniform-replay
    /// path.
    pub mean_abs_td_error: Option<f64>,
}

/// DQN Trainer.
///
/// Wraps an online [`QNetwork`], a target [`QNetwork`] (synced via
/// `copy_params_from`), an Adam optimizer, and either a uniform
/// [`ReplayBuffer`] or a [`PrioritizedReplayBuffer`] (selected by
/// `config.prioritized_replay`). Exactly one buffer is `Some` at a time.
pub struct DQNTrainer {
    config: DQNConfig,
    online: QNetwork,
    target: QNetwork,
    optimizer: tch::nn::Optimizer,
    buffer: Option<ReplayBuffer>,
    prioritized_buffer: Option<PrioritizedReplayBuffer>,
    device: Device,
    total_env_steps: usize,
    total_train_steps: usize,
    total_episodes: usize,
    last_epsilon: f64,
}

impl DQNTrainer {
    /// Build a new trainer.
    ///
    /// Allocates the target network as an independent `QNetwork` and
    /// performs an initial hard copy so the two networks start byte-equal.
    ///
    /// # Arguments
    /// * `config` - hyperparameters; validated up front.
    /// * `obs_dim` - observation dimension.
    /// * `n_actions` - number of discrete actions.
    /// * `hidden_dim` - hidden layer width for both networks (typically 64).
    pub fn new(config: DQNConfig, obs_dim: i64, n_actions: i64, hidden_dim: i64) -> Result<Self> {
        config.validate()?;

        let mut online = QNetwork::new(obs_dim, n_actions, hidden_dim);
        let mut target = QNetwork::new(obs_dim, n_actions, hidden_dim);
        // Start the target net byte-equal to the online net.
        target.copy_params_from(&online)?;
        // The target net never trains directly; freeze it so it's clear
        // gradients never accumulate there even if a caller passes the
        // wrong VarStore to an optimizer by mistake.
        target.freeze();

        let device = online.device();
        let optimizer = online.optimizer(config.learning_rate);
        let (buffer, prioritized_buffer) = if config.prioritized_replay {
            let pb = PrioritizedReplayBuffer::new(
                config.buffer_capacity,
                obs_dim as usize,
                config.per_alpha as f32,
                config.per_epsilon as f32,
            );
            (None, Some(pb))
        } else {
            let b = ReplayBuffer::new(config.buffer_capacity, obs_dim as usize);
            (Some(b), None)
        };
        let last_epsilon = config.epsilon_start;

        Ok(Self {
            config,
            online,
            target,
            optimizer,
            buffer,
            prioritized_buffer,
            device,
            total_env_steps: 0,
            total_train_steps: 0,
            total_episodes: 0,
            last_epsilon,
        })
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &DQNConfig {
        &self.config
    }

    /// Borrow the online Q-network.
    pub fn online(&self) -> &QNetwork {
        &self.online
    }

    /// Mutably borrow the online Q-network (e.g. for saving).
    pub fn online_mut(&mut self) -> &mut QNetwork {
        &mut self.online
    }

    /// Borrow the target Q-network.
    pub fn target(&self) -> &QNetwork {
        &self.target
    }

    /// Borrow the uniform replay buffer.
    ///
    /// # Panics
    /// Panics if the trainer is configured for prioritized replay
    /// (`config.prioritized_replay = true`). Use
    /// [`Self::prioritized_buffer`] in that case, or prefer the unified
    /// [`Self::push_transition`] helper for code that should work in
    /// both modes.
    pub fn buffer(&self) -> &ReplayBuffer {
        self.buffer.as_ref().expect(
            "DQNTrainer::buffer called but trainer is in prioritized mode; \
             use prioritized_buffer() or push_transition() instead",
        )
    }

    /// Mutably borrow the uniform replay buffer.
    ///
    /// # Panics
    /// Same as [`Self::buffer`].
    pub fn buffer_mut(&mut self) -> &mut ReplayBuffer {
        self.buffer.as_mut().expect(
            "DQNTrainer::buffer_mut called but trainer is in prioritized mode; \
             use prioritized_buffer_mut() or push_transition() instead",
        )
    }

    /// Borrow the prioritized replay buffer, if the trainer was built
    /// with `prioritized_replay = true`.
    pub fn prioritized_buffer(&self) -> Option<&PrioritizedReplayBuffer> {
        self.prioritized_buffer.as_ref()
    }

    /// Mutably borrow the prioritized replay buffer, if active.
    pub fn prioritized_buffer_mut(&mut self) -> Option<&mut PrioritizedReplayBuffer> {
        self.prioritized_buffer.as_mut()
    }

    /// Push a transition into whichever replay buffer is active.
    ///
    /// This is the recommended way to feed experience to the trainer
    /// — it works in both uniform and prioritized modes without the
    /// caller needing to branch on `config.prioritized_replay`.
    pub fn push_transition(
        &mut self,
        obs: &[f32],
        action: i64,
        reward: f32,
        next_obs: &[f32],
        done: bool,
    ) {
        if let Some(b) = self.buffer.as_mut() {
            b.push(obs, action, reward, next_obs, done);
        } else if let Some(pb) = self.prioritized_buffer.as_mut() {
            pb.push(obs, action, reward, next_obs, done);
        } else {
            unreachable!("DQNTrainer has neither uniform nor prioritized buffer");
        }
    }

    /// Number of transitions currently in whichever buffer is active.
    pub fn buffer_len(&self) -> usize {
        if let Some(b) = self.buffer.as_ref() {
            b.len()
        } else if let Some(pb) = self.prioritized_buffer.as_ref() {
            pb.len()
        } else {
            0
        }
    }

    /// `true` if the active buffer holds at least `min_size` transitions.
    pub fn buffer_is_ready(&self, min_size: usize) -> bool {
        if let Some(b) = self.buffer.as_ref() {
            b.is_ready(min_size)
        } else if let Some(pb) = self.prioritized_buffer.as_ref() {
            pb.is_ready(min_size)
        } else {
            false
        }
    }

    /// Device hosting the Q-networks.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Current env-step counter.
    pub fn total_env_steps(&self) -> usize {
        self.total_env_steps
    }

    /// Number of completed gradient updates.
    pub fn total_train_steps(&self) -> usize {
        self.total_train_steps
    }

    /// Number of completed episodes (caller increments).
    pub fn total_episodes(&self) -> usize {
        self.total_episodes
    }

    /// ε used to draw the most recently selected action.
    pub fn last_epsilon(&self) -> f64 {
        self.last_epsilon
    }

    /// Caller must invoke this exactly once per environment step. The
    /// counter is the only thing driving ε-decay and the periodic
    /// target-net sync, so failing to increment it will silently disable
    /// both schedules.
    pub fn increment_env_step(&mut self) {
        self.total_env_steps += 1;
    }

    /// Caller invokes this when an episode terminates / truncates.
    pub fn increment_episodes(&mut self, n: usize) {
        self.total_episodes += n;
    }

    /// ε-greedy action selection.
    ///
    /// Computes the current ε via [`DQNConfig::epsilon_at`] using
    /// `total_env_steps`, then either picks a uniform random action with
    /// probability ε or argmax-Q greedy action otherwise.
    ///
    /// Recorded in `last_epsilon` for diagnostics.
    pub fn select_action<R: Rng>(&mut self, obs: &[f32], rng: &mut R) -> i64 {
        let eps = self.config.epsilon_at(self.total_env_steps);
        self.last_epsilon = eps;
        let n_actions = self.online.n_actions();
        if rng.random::<f64>() < eps {
            rng.random_range(0..n_actions)
        } else {
            self.greedy_action(obs)
        }
    }

    /// Pure greedy action (no exploration). Useful for evaluation rollouts.
    pub fn greedy_action(&self, obs: &[f32]) -> i64 {
        tch::no_grad(|| {
            let obs_tensor = Tensor::from_slice(obs)
                .reshape([1, obs.len() as i64])
                .to_kind(Kind::Float)
                .to_device(self.device);
            let q_values = self.online.forward(&obs_tensor); // [1, n_actions]
            let action = q_values.argmax(-1, false);
            i64::try_from(action).unwrap_or(0)
        })
    }

    /// Sync the target network from the online network.
    ///
    /// Two modes, selected by [`DQNConfig::soft_update_tau`]:
    ///
    /// 1. **Hard sync (default, `soft_update_tau = None`)**: copies `θ_target ←
    ///    θ_online` once every `target_update_interval` env steps. Returns
    ///    `true` exactly on those steps.
    ///
    /// 2. **Soft / Polyak update (`soft_update_tau = Some(τ)`)**: applies
    ///    `θ_target ← τ · θ_online + (1 − τ) · θ_target` to every parameter,
    ///    *every* call. The caller is still expected to invoke
    ///    [`Self::maybe_sync_target`] once per env step;
    ///    `target_update_interval` is ignored in this mode. Returns `true` on
    ///    every call so existing callers that gate logging or stats on the
    ///    return value still behave sensibly.
    ///
    /// In both modes the blend / copy runs inside `tch::no_grad` so the
    /// target net never lands in the autograd graph.
    pub fn maybe_sync_target(&mut self) -> Result<bool> {
        match self.config.soft_update_tau {
            Some(tau) => {
                self.soft_update_target(tau)?;
                Ok(true)
            }
            None => {
                if self.total_env_steps > 0
                    && self.total_env_steps % self.config.target_update_interval == 0
                {
                    self.target.copy_params_from(&self.online)?;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Apply a Polyak / soft update to every parameter of the target net.
    ///
    /// `θ_target ← τ · θ_online + (1 − τ) · θ_target`
    ///
    /// Pairs are matched by variable name across the two `VarStore`s.
    /// Any mismatched name is a programmer error — both networks are
    /// constructed by `QNetwork::new` with identical paths, so a missing
    /// match indicates the trainer is wired up incorrectly.
    ///
    /// We snapshot the target's `variables()` HashMap, which gives us
    /// owned `Tensor` shallow-clones pointing at the same C-side
    /// storage. Calling `f_copy_` through these clones updates the live
    /// VarStore in place — no reallocation, and the optimizer (which
    /// only ever sees the online net's vars) is unaffected.
    fn soft_update_target(&mut self, tau: f64) -> Result<()> {
        let online_vars = self.online.var_store().variables();
        let mut target_vars = self.target.var_store().variables();

        tch::no_grad(|| -> Result<()> {
            for (name, online_t) in &online_vars {
                let target_t = target_vars.get_mut(name).ok_or_else(|| {
                    anyhow!("soft_update_target: variable {} missing from target VarStore", name)
                })?;
                // target ← τ · online + (1 − τ) · target  (in-place on target)
                // `f_copy_` writes the RHS into target_t's storage without
                // breaking the optimizer's reference to it.
                let blended = online_t * tau + &*target_t * (1.0 - tau);
                target_t
                    .f_copy_(&blended)
                    .map_err(|e| anyhow!("soft_update_target: copy into {} failed: {}", name, e))?;
            }
            Ok(())
        })
    }

    /// Sample a batch from the active replay buffer and perform one
    /// gradient update against the (Double-)DQN TD target.
    ///
    /// Routes by `config.prioritized_replay`:
    ///
    /// - **Uniform**: classic recipe — sample uniformly, mean Smooth-L1 loss,
    ///   backprop.
    /// - **Prioritized**: stratified proportional sampling, per-sample loss
    ///   weighted by IS weights `wᵢ`, reduced to a scalar via mean. After the
    ///   update, the per-sample `|TD error|` is written back to the buffer's
    ///   priorities so subsequent draws upweight the transitions the network
    ///   still struggles with.
    ///
    /// Returns `Ok(None)` if the buffer doesn't yet hold
    /// `min_buffer_size` transitions; in that case the caller should
    /// keep collecting experience.
    pub fn train_step<R: Rng>(&mut self, rng: &mut R) -> Result<Option<DQNStepStats>> {
        if !self.buffer_is_ready(self.config.min_buffer_size) {
            return Ok(None);
        }

        if self.config.prioritized_replay {
            self.train_step_prioritized(rng)
        } else {
            self.train_step_uniform(rng)
        }
    }

    /// Uniform-replay training step. Equivalent to the pre-PER
    /// implementation.
    fn train_step_uniform<R: Rng>(&mut self, rng: &mut R) -> Result<Option<DQNStepStats>> {
        let buffer = self.buffer.as_ref().expect("uniform path requires uniform buffer");
        let batch = sample(buffer, self.config.batch_size, rng);
        let buffer_len = buffer.len();
        let (obs, actions, rewards, next_obs, dones) = batch.to_tensors(self.device);

        // Online: Q(s, a) per action, then gather along the action dim.
        let q_online_all = self.online.forward(&obs);
        let q_taken = loss::gather_action_q(&q_online_all, &actions);

        // Double-DQN target.
        let next_q_online_all = tch::no_grad(|| self.online.forward(&next_obs));
        let next_q_target_all = tch::no_grad(|| self.target.forward(&next_obs));
        let td_target = loss::compute_td_target_double(
            &rewards,
            &dones,
            &next_q_online_all,
            &next_q_target_all,
            self.config.gamma,
        );

        let td_loss = loss::compute_loss(&q_taken, &td_target);
        let td_loss_val: f64 = (&td_loss).try_into().unwrap_or(f64::NAN);
        let mean_q_val: f64 = (&q_taken.mean(Kind::Float)).try_into().unwrap_or(0.0);

        self.optimizer.zero_grad();
        td_loss.backward();
        self.optimizer.clip_grad_norm(self.config.max_grad_norm);
        self.optimizer.step();

        self.total_train_steps += 1;

        if !td_loss_val.is_finite() {
            return Err(anyhow!("Non-finite TD loss: {}", td_loss_val));
        }

        Ok(Some(DQNStepStats {
            td_loss: td_loss_val,
            mean_q: mean_q_val,
            epsilon: self.last_epsilon,
            buffer_len,
            beta: None,
            mean_abs_td_error: None,
        }))
    }

    /// Prioritized-replay training step.
    fn train_step_prioritized<R: Rng>(&mut self, rng: &mut R) -> Result<Option<DQNStepStats>> {
        let beta = self.config.beta_at(self.total_env_steps);
        let (batch, buffer_len) = {
            let pb = self
                .prioritized_buffer
                .as_ref()
                .expect("prioritized path requires prioritized buffer");
            let b = pb.sample(self.config.batch_size, beta as f32, rng);
            let len = pb.len();
            (b, len)
        };

        let indices = batch.indices.clone();
        let (obs, actions, rewards, next_obs, dones, is_weights) = batch.to_tensors(self.device);

        // Online: Q(s, a) per action.
        let q_online_all = self.online.forward(&obs);
        let q_taken = loss::gather_action_q(&q_online_all, &actions);

        // Double-DQN target.
        let next_q_online_all = tch::no_grad(|| self.online.forward(&next_obs));
        let next_q_target_all = tch::no_grad(|| self.target.forward(&next_obs));
        let td_target = loss::compute_td_target_double(
            &rewards,
            &dones,
            &next_q_online_all,
            &next_q_target_all,
            self.config.gamma,
        );

        // Per-sample Smooth-L1 loss, then multiply by IS weights and
        // average. Using `Reduction::None` here gives us a `[batch]`
        // tensor of unweighted per-sample losses; the `* is_weights`
        // reweights each one before the final `.mean()`.
        let per_sample_loss = q_taken.smooth_l1_loss(&td_target, Reduction::None, 1.0);
        let weighted_loss = (&per_sample_loss * &is_weights).mean(Kind::Float);
        let td_loss_val: f64 = (&weighted_loss).try_into().unwrap_or(f64::NAN);
        let mean_q_val: f64 = (&q_taken.mean(Kind::Float)).try_into().unwrap_or(0.0);

        // Per-sample |TD error| for priority writeback. Computed under
        // `no_grad` so it doesn't pollute the autograd graph.
        let abs_td_errors_cpu: Vec<f32> = tch::no_grad(|| {
            let per_sample_abs = (&q_taken - &td_target).abs();
            let cpu_t = per_sample_abs.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
            Vec::try_from(cpu_t).unwrap_or_default()
        });
        let mean_abs_td: f64 = if abs_td_errors_cpu.is_empty() {
            0.0
        } else {
            abs_td_errors_cpu.iter().map(|&v| v as f64).sum::<f64>()
                / abs_td_errors_cpu.len() as f64
        };

        self.optimizer.zero_grad();
        weighted_loss.backward();
        self.optimizer.clip_grad_norm(self.config.max_grad_norm);
        self.optimizer.step();

        // Write the new priorities back. Doing this after the
        // optimizer step (not before) is consistent with the Schaul
        // paper — the priority reflects the residual on the network
        // that produced this batch's gradient.
        if let Some(pb) = self.prioritized_buffer.as_mut() {
            pb.update_priorities(&indices, &abs_td_errors_cpu);
        }

        self.total_train_steps += 1;

        if !td_loss_val.is_finite() {
            return Err(anyhow!("Non-finite TD loss: {}", td_loss_val));
        }

        Ok(Some(DQNStepStats {
            td_loss: td_loss_val,
            mean_q: mean_q_val,
            epsilon: self.last_epsilon,
            buffer_len,
            beta: Some(beta),
            mean_abs_td_error: Some(mean_abs_td),
        }))
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};
    use tch::Kind;

    use super::*;

    fn small_config() -> DQNConfig {
        DQNConfig::new()
            .buffer_capacity(128)
            .min_buffer_size(8)
            .batch_size(8)
            .target_update_interval(4)
            .epsilon_decay_steps(100)
    }

    #[test]
    fn test_new_constructs_byte_equal_target() {
        let trainer = DQNTrainer::new(small_config(), 4, 2, 16).unwrap();
        // Forward both nets on the same input; outputs must agree exactly.
        let obs = Tensor::randn([2, 4], (Kind::Float, trainer.device()));
        let q_online = trainer.online().forward(&obs);
        let q_target = trainer.target().forward(&obs);
        let diff = (&q_online - &q_target).abs().sum(Kind::Float);
        let v: f64 = diff.try_into().unwrap();
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_select_action_in_range() {
        let mut trainer = DQNTrainer::new(small_config(), 4, 3, 16).unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..20 {
            let a = trainer.select_action(&[0.0, 0.1, 0.2, 0.3], &mut rng);
            assert!(a >= 0 && a < 3);
        }
    }

    #[test]
    fn test_train_step_returns_none_until_ready() {
        let mut trainer = DQNTrainer::new(small_config(), 4, 2, 16).unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        // Buffer empty → train_step is a no-op.
        assert!(trainer.train_step(&mut rng).unwrap().is_none());

        // Push a handful but not min_buffer_size yet.
        for i in 0..7 {
            trainer.buffer_mut().push(
                &[i as f32; 4],
                (i % 2) as i64,
                i as f32,
                &[(i + 1) as f32; 4],
                false,
            );
        }
        assert!(trainer.train_step(&mut rng).unwrap().is_none());
    }

    #[test]
    fn test_train_step_changes_weights() {
        let mut trainer = DQNTrainer::new(small_config(), 4, 2, 16).unwrap();
        let mut rng = StdRng::seed_from_u64(2);

        // Push enough synthetic transitions to clear min_buffer_size.
        for i in 0..32 {
            let phase = (i as f32) * 0.1;
            let obs = [phase.sin(), phase.cos(), phase * 0.5, phase * -0.3];
            let next_obs = [(phase + 0.1).sin(), (phase + 0.1).cos(), phase * 0.5, phase * -0.3];
            let action = (i % 2) as i64;
            let reward = if action == 0 { 1.0 } else { -1.0 };
            let done = i % 8 == 7;
            trainer.buffer_mut().push(&obs, action, reward, &next_obs, done);
        }

        // Snapshot one weight tensor before the gradient step.
        let var_name = "backbone.fc1.weight".to_string();
        let before: Vec<f32> = {
            let vars = trainer.online().var_store().variables();
            let t = vars.get(&var_name).expect("missing fc1 weight");
            let cpu_t = t.to_device(tch::Device::Cpu).to_kind(Kind::Float).contiguous();
            Vec::try_from(cpu_t.view([-1])).unwrap()
        };

        // Run several gradient updates so accumulated change is detectable
        // even with a tiny LR and a small batch.
        let mut last_loss = None;
        for _ in 0..10 {
            let stats = trainer.train_step(&mut rng).unwrap();
            assert!(stats.is_some(), "train_step should run once buffer is ready");
            let s = stats.unwrap();
            assert!(s.td_loss.is_finite(), "TD loss must be finite, got {}", s.td_loss);
            last_loss = Some(s.td_loss);
        }
        assert!(last_loss.is_some());

        let after: Vec<f32> = {
            let vars = trainer.online().var_store().variables();
            let t = vars.get(&var_name).expect("missing fc1 weight");
            let cpu_t = t.to_device(tch::Device::Cpu).to_kind(Kind::Float).contiguous();
            Vec::try_from(cpu_t.view([-1])).unwrap()
        };

        assert_eq!(before.len(), after.len());
        let mut total_abs_change = 0.0f32;
        for (a, b) in before.iter().zip(after.iter()) {
            total_abs_change += (a - b).abs();
        }
        assert!(
            total_abs_change > 0.0,
            "Expected fc1 weights to change after {} gradient updates; total abs delta = {}",
            10,
            total_abs_change
        );
    }

    #[test]
    fn test_maybe_sync_target_triggers_on_interval() {
        // Use interval = 3 so the math is easy to inspect.
        let cfg = small_config().target_update_interval(3);
        let mut trainer = DQNTrainer::new(cfg, 4, 2, 16).unwrap();

        // Step 0: not yet incremented → no sync.
        assert!(!trainer.maybe_sync_target().unwrap());

        trainer.increment_env_step(); // total = 1
        assert!(!trainer.maybe_sync_target().unwrap());
        trainer.increment_env_step(); // total = 2
        assert!(!trainer.maybe_sync_target().unwrap());
        trainer.increment_env_step(); // total = 3
        assert!(trainer.maybe_sync_target().unwrap());
        trainer.increment_env_step(); // total = 4
        assert!(!trainer.maybe_sync_target().unwrap());

        // Two more bumps → 6 → another sync.
        trainer.increment_env_step();
        trainer.increment_env_step();
        assert!(trainer.maybe_sync_target().unwrap());
    }

    #[test]
    fn test_greedy_action_in_range() {
        let trainer = DQNTrainer::new(small_config(), 4, 5, 16).unwrap();
        let a = trainer.greedy_action(&[0.5, -0.5, 0.0, 1.0]);
        assert!(a >= 0 && a < 5);
    }

    /// Pull a flat parameter snapshot out of a network for comparison.
    fn snapshot_params(net: &QNetwork) -> Vec<(String, Vec<f32>)> {
        let vars = net.var_store().variables();
        let mut out = Vec::with_capacity(vars.len());
        for (name, t) in vars {
            let cpu_t = t.to_device(tch::Device::Cpu).to_kind(Kind::Float).contiguous();
            let flat: Vec<f32> = Vec::try_from(cpu_t.view([-1])).unwrap();
            out.push((name, flat));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    fn l1_distance(a: &[(String, Vec<f32>)], b: &[(String, Vec<f32>)]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut sum = 0.0f32;
        for ((na, va), (nb, vb)) in a.iter().zip(b.iter()) {
            assert_eq!(na, nb, "param name mismatch in snapshot comparison");
            assert_eq!(va.len(), vb.len());
            for (x, y) in va.iter().zip(vb.iter()) {
                sum += (x - y).abs();
            }
        }
        sum
    }

    #[test]
    fn test_soft_update_moves_target_toward_online() {
        // Configure soft updates with τ = 0.005.
        let cfg = small_config().soft_update_tau(0.005);
        let mut trainer = DQNTrainer::new(cfg, 4, 2, 16).unwrap();

        // After construction the target is byte-equal to online (initial
        // hard copy in `DQNTrainer::new`). Perturb the online net so the
        // two diverge in a known direction, then verify Polyak pulls
        // target toward the updated online.
        //
        // We do this by running a single optimizer step on a synthetic
        // batch — this shifts online's weights without touching target.
        let mut rng = StdRng::seed_from_u64(123);
        for i in 0..32 {
            let phase = (i as f32) * 0.1;
            let obs = [phase.sin(), phase.cos(), phase * 0.5, phase * -0.3];
            let next_obs = [(phase + 0.1).sin(), (phase + 0.1).cos(), phase * 0.5, phase * -0.3];
            trainer.buffer_mut().push(&obs, (i % 2) as i64, 1.0, &next_obs, i % 8 == 7);
        }
        let _ = trainer.train_step(&mut rng).unwrap();

        // Snapshot online and target *before* the soft sync.
        let online_after_train = snapshot_params(trainer.online());
        let target_before = snapshot_params(trainer.target());

        // The training step should have changed online without touching
        // target — confirm they now differ.
        let pre_drift = l1_distance(&online_after_train, &target_before);
        assert!(
            pre_drift > 0.0,
            "expected online to diverge from target after a gradient step (l1 = {})",
            pre_drift,
        );

        // Run the Polyak update once. Caller does it every env step, but
        // for the test a single call is enough.
        let synced = trainer.maybe_sync_target().unwrap();
        assert!(synced, "maybe_sync_target should return true in soft-update mode");

        let target_after = snapshot_params(trainer.target());

        // (a) target_after must differ from target_before for at least one
        //     parameter (the update actually moved something).
        let target_motion = l1_distance(&target_before, &target_after);
        assert!(
            target_motion > 0.0,
            "soft update did not move target params (Δ = {})",
            target_motion,
        );

        // (b) target_after must be strictly closer to online than
        //     target_before was (Polyak moves toward, not away).
        let dist_before = l1_distance(&target_before, &online_after_train);
        let dist_after = l1_distance(&target_after, &online_after_train);
        assert!(
            dist_after < dist_before,
            "Polyak update should reduce |target − online|; before={} after={}",
            dist_before,
            dist_after,
        );

        // (c) target_after must NOT equal online (tau = 0.005, so only a
        //     small step toward online — not a hard copy).
        assert!(
            dist_after > 0.0,
            "soft update with τ = 0.005 should NOT make target equal online; dist={}",
            dist_after,
        );

        // (d) The expected fraction of the gap closed is exactly τ for a
        //     linear blend. Check ratio is approximately (1 − τ) per
        //     parameter, allowing slack for fp rounding across many vars.
        let expected_ratio = 1.0 - 0.005f32;
        let observed_ratio = dist_after / dist_before;
        assert!(
            (observed_ratio - expected_ratio).abs() < 1e-3,
            "Polyak distance ratio off: expected ≈{:.4}, got {:.4}",
            expected_ratio,
            observed_ratio,
        );
    }

    #[test]
    fn test_soft_update_ignores_interval() {
        // Soft mode is supposed to fire on every call, regardless of where
        // total_env_steps sits relative to target_update_interval.
        let cfg = small_config().target_update_interval(1000).soft_update_tau(0.1);
        let mut trainer = DQNTrainer::new(cfg, 4, 2, 16).unwrap();

        // total_env_steps = 0 → hard-mode would refuse, soft mode must fire.
        assert!(trainer.maybe_sync_target().unwrap());

        // After a few env-step bumps, well shy of the 1000-step interval,
        // hard mode would still refuse — soft mode keeps firing.
        for _ in 0..5 {
            trainer.increment_env_step();
            assert!(trainer.maybe_sync_target().unwrap());
        }
    }

    #[test]
    fn test_prioritized_mode_constructs_prioritized_buffer() {
        let cfg = small_config().prioritized_replay(true);
        let trainer = DQNTrainer::new(cfg, 4, 2, 16).unwrap();
        assert!(trainer.prioritized_buffer().is_some());
        // `buffer()` must panic in this mode — verify via catching the
        // panic so we don't leave a leaked handle.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| trainer.buffer()));
        assert!(result.is_err(), "buffer() should panic in prioritized mode");
    }

    #[test]
    fn test_push_transition_works_in_both_modes() {
        // Uniform mode.
        let mut trainer = DQNTrainer::new(small_config(), 4, 2, 16).unwrap();
        trainer.push_transition(&[0.0, 0.1, 0.2, 0.3], 0, 1.0, &[0.1, 0.2, 0.3, 0.4], false);
        assert_eq!(trainer.buffer_len(), 1);

        // Prioritized mode.
        let mut trainer_p =
            DQNTrainer::new(small_config().prioritized_replay(true), 4, 2, 16).unwrap();
        trainer_p.push_transition(&[0.0, 0.1, 0.2, 0.3], 0, 1.0, &[0.1, 0.2, 0.3, 0.4], false);
        assert_eq!(trainer_p.buffer_len(), 1);
        assert!(trainer_p.prioritized_buffer().is_some());
    }

    #[test]
    fn test_prioritized_train_step_updates_priorities() {
        // Push 4 transitions, all with the same default max priority
        // (1.0). Run train_step once; the priorities of the sampled
        // indices should change to (|td_err| + ε)^α, which will differ
        // from 1.0 once the network has produced any non-trivial TD
        // error.
        let cfg = small_config()
            .prioritized_replay(true)
            .min_buffer_size(4)
            .batch_size(4)
            .per_alpha(0.6)
            .per_epsilon(1e-6);
        let mut trainer = DQNTrainer::new(cfg, 4, 2, 16).unwrap();
        let mut rng = StdRng::seed_from_u64(7);

        // Push diverse transitions so the network output and the TD
        // target differ.
        for i in 0..4 {
            let phase = i as f32 * 0.5;
            let obs = [phase.sin(), phase.cos(), phase * 0.3, phase * -0.7];
            let next_obs = [(phase + 0.1).sin(), (phase + 0.1).cos(), phase * 0.3, phase * -0.7];
            trainer.push_transition(&obs, (i % 2) as i64, i as f32, &next_obs, i == 3);
        }

        // Snapshot the priorities before training.
        let before: Vec<f32> = (0..4)
            .map(|i| trainer.prioritized_buffer().unwrap().tree().leaf_priority(i))
            .collect();
        // All four leaves should sit at the initial max_priority (1.0).
        for (i, &p) in before.iter().enumerate() {
            assert!((p - 1.0).abs() < 1e-6, "leaf {} initial priority was {} not 1.0", i, p);
        }

        let stats = trainer.train_step(&mut rng).unwrap();
        assert!(stats.is_some(), "train_step should run once buffer is ready");
        let s = stats.unwrap();
        assert!(s.beta.is_some(), "prioritized stats should expose β");
        assert!(s.mean_abs_td_error.is_some(), "prioritized stats should expose mean |TD error|");
        assert!(s.td_loss.is_finite(), "TD loss must be finite");

        // After train_step, at least one leaf priority should have
        // changed (since at least one transition was sampled and given
        // a fresh priority based on its TD error).
        let after: Vec<f32> = (0..4)
            .map(|i| trainer.prioritized_buffer().unwrap().tree().leaf_priority(i))
            .collect();
        let mut total_change = 0.0f32;
        for (b, a) in before.iter().zip(after.iter()) {
            total_change += (a - b).abs();
        }
        assert!(
            total_change > 0.0,
            "prioritized train_step should change at least one leaf priority; Δ = {}",
            total_change
        );
    }

    #[test]
    fn test_hard_mode_unchanged_when_tau_is_none() {
        // soft_update_tau = None → behavior must match the original hard
        // sync semantics tested in test_maybe_sync_target_triggers_on_interval.
        let cfg = small_config().target_update_interval(3);
        let mut trainer = DQNTrainer::new(cfg, 4, 2, 16).unwrap();
        assert!(trainer.config().soft_update_tau.is_none());

        assert!(!trainer.maybe_sync_target().unwrap());
        trainer.increment_env_step();
        trainer.increment_env_step();
        trainer.increment_env_step();
        assert!(trainer.maybe_sync_target().unwrap());
    }
}
