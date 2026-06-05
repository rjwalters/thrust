//! DQN Trainer implementation.
//!
//! Holds the online + target Q-networks, the optimizer, the replay
//! buffer, and the counters needed to drive ε-greedy exploration and
//! periodic target-net syncs. The env-interaction loop lives in the
//! caller (typically a `train_*_dqn` example) — `DQNTrainer` only owns
//! the parts that don't depend on the environment.

use anyhow::{Result, anyhow};
use rand::Rng;
use tch::{Device, Kind, Tensor};

use super::{config::DQNConfig, loss};
use crate::{
    buffer::replay::{ReplayBuffer, sample},
    policy::QNetwork,
};

/// Per-step training statistics returned by [`DQNTrainer::train_step`].
#[derive(Debug, Clone, Copy)]
pub struct DQNStepStats {
    /// Mean Smooth-L1 (Huber) loss across the minibatch.
    pub td_loss: f64,
    /// Mean of `Q(s, a)` across the minibatch (useful diagnostic).
    pub mean_q: f64,
    /// ε used to draw the most recent action.
    pub epsilon: f64,
    /// Replay buffer fill level at the time of this update.
    pub buffer_len: usize,
    /// Whether the target network was synced this step.
    pub target_synced: bool,
}

/// DQN Trainer.
///
/// Wraps an online [`QNetwork`], a target [`QNetwork`] (synced via
/// `copy_params_from`), an Adam optimizer, and a [`ReplayBuffer`].
pub struct DQNTrainer {
    config: DQNConfig,
    online: QNetwork,
    target: QNetwork,
    optimizer: tch::nn::Optimizer,
    buffer: ReplayBuffer,
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
        let buffer = ReplayBuffer::new(config.buffer_capacity, obs_dim as usize);
        let last_epsilon = config.epsilon_start;

        Ok(Self {
            config,
            online,
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

    /// Borrow the replay buffer.
    pub fn buffer(&self) -> &ReplayBuffer {
        &self.buffer
    }

    /// Mutably borrow the replay buffer.
    pub fn buffer_mut(&mut self) -> &mut ReplayBuffer {
        &mut self.buffer
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
        if rng.r#gen::<f64>() < eps {
            rng.gen_range(0..n_actions)
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

    /// Hard target sync if `total_env_steps` is a positive multiple of
    /// `target_update_interval`.
    ///
    /// Returns `true` if a sync happened.
    pub fn maybe_sync_target(&mut self) -> Result<bool> {
        if self.total_env_steps > 0
            && self.total_env_steps % self.config.target_update_interval == 0
        {
            self.target.copy_params_from(&self.online)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Sample a batch from the replay buffer and perform one gradient
    /// update against the TD target.
    ///
    /// Returns `Ok(None)` if the buffer doesn't yet hold
    /// `min_buffer_size` transitions; in that case the caller should
    /// keep collecting experience.
    pub fn train_step<R: Rng>(&mut self, rng: &mut R) -> Result<Option<DQNStepStats>> {
        if !self.buffer.is_ready(self.config.min_buffer_size) {
            return Ok(None);
        }

        let batch = sample(&self.buffer, self.config.batch_size, rng);
        let (obs, actions, rewards, next_obs, dones) = batch.to_tensors(self.device);

        // Online: Q(s, a) per action, then gather along the action dim.
        let q_online_all = self.online.forward(&obs);
        let q_taken = loss::gather_action_q(&q_online_all, &actions);

        // Target: max_a' Q_target(s', a'), no_grad inside compute_td_target.
        let next_q_target_all = tch::no_grad(|| self.target.forward(&next_obs));
        let td_target =
            loss::compute_td_target(&rewards, &dones, &next_q_target_all, self.config.gamma);

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
            buffer_len: self.buffer.len(),
            target_synced: false, // set by caller via maybe_sync_target
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
}
