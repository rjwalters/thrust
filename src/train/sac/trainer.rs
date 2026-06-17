//! Soft Actor-Critic (SAC) trainer for continuous control.
//!
//! [`SacTrainer`] is the integration piece of the SAC decomposition
//! (#136, PR E): it wires the four building blocks —
//! [`SacActor`](crate::policy::sac_actor::SacActor) (#140),
//! [`ContinuousQNetwork`](crate::policy::continuous_q::ContinuousQNetwork)
//! twin critics + targets (#141),
//! [`ContinuousReplayBuffer`](crate::buffer::replay::ContinuousReplayBuffer)
//! (#138), and the
//! [`PendulumSwingUp`](crate::env::games::pendulum::PendulumSwingUp) env (#139)
//! — into the Soft Actor-Critic algorithm of Haarnoja et al. 2018 v2 (arXiv:
//! 1812.05905).
//!
//! # Owned state
//!
//! - A stochastic [`SacActor`](crate::policy::sac_actor::SacActor).
//! - Two online critics `q1`, `q2` and their targets `q1_target`, `q2_target`.
//! - A [`LogAlpha`] module holding `log_alpha` (the log entropy temperature),
//!   tuned only when [`SacConfig::auto_alpha`] is set.
//! - **Three independent Adam optimizers** (actor, critics, alpha). Burn's
//!   optimizer is move-through (it consumes its module by value per step), so
//!   the five networks cannot live in one fused module; they are owned as
//!   separate fields and stepped independently. The two online critics share a
//!   single optimizer that steps them in sequence.
//! - A [`ContinuousReplayBuffer`](crate::buffer::replay::ContinuousReplayBuffer)
//!   and a seeded [`StdRng`](rand::rngs::StdRng).
//!
//! # Update equations
//!
//! For a minibatch `(s, a, r, s', d)`:
//!
//! - **Critic target** (`a' ~ actor.sample(s')` from the *current* actor): `y =
//!   r + gamma * (1 - d) * (min(Q1_t, Q2_t)(s', a') - alpha *
//!   log_prob(a'|s'))`. Both online critics regress to `y` with an MSE loss;
//!   the target is detached.
//! - **Actor** (reparameterized `a_pi ~ actor.sample(s)`): `loss = mean(alpha *
//!   log_prob(a_pi|s) - min(Q1, Q2)(s, a_pi))`. The critics are frozen
//!   (detached) for this loss so only the actor moves.
//! - **Alpha** (when auto-tuning): `loss = -mean(log_alpha * (log_prob +
//!   target_entropy))` with the `(log_prob + target_entropy)` factor detached;
//!   `alpha = exp(log_alpha)`.
//! - **Targets**: Polyak soft update every gradient step, `theta_target <- tau
//!   * theta_online + (1 - tau) * theta_target`.

use anyhow::{Result, anyhow};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::{Module, Param},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::ToElement,
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::config::SacConfig;
use crate::{
    buffer::replay::{ContinuousReplayBuffer, sample_continuous},
    policy::{
        continuous_q::{ContinuousQNetwork, ContinuousQNetworkConfig},
        mlp::BurnActivation,
        sac_actor::{SacActor, SacActorConfig},
    },
};

/// Tiny single-parameter [`Module`] holding the log of the entropy
/// temperature `alpha`.
///
/// SAC's automatic temperature tuning optimizes `log_alpha` (so the
/// recovered `alpha = exp(log_alpha)` stays strictly positive). It is a
/// standalone module so a dedicated Adam optimizer can step it through
/// Burn's move-through interface, exactly like the actor and critics.
#[derive(Module, Debug)]
pub struct LogAlpha<B: Backend> {
    value: Param<Tensor<B, 1>>,
}

impl<B: Backend> LogAlpha<B> {
    /// Build a `log_alpha` module initialized to `ln(init_alpha)`.
    pub fn new(init_alpha: f32, device: &B::Device) -> Self {
        let data = burn::tensor::TensorData::new(vec![init_alpha.ln()], [1]);
        let tensor = Tensor::<B, 1>::from_data(data, device);
        Self { value: Param::from_tensor(tensor) }
    }

    /// The current `log_alpha` tensor (shape `[1]`), grad-bearing.
    pub fn value(&self) -> Tensor<B, 1> {
        self.value.val()
    }

    /// The current `alpha = exp(log_alpha)` as a host scalar.
    pub fn alpha_scalar(&self) -> f64 {
        self.value.val().exp().into_scalar().to_f64()
    }
}

/// Per-step training statistics for the SAC trainer.
///
/// Analogous to [`crate::train::dqn::DQNStepStatsBurn`]. Returned by
/// [`SacTrainer::train_step`] once a gradient update has actually run.
#[derive(Debug, Clone, Copy)]
pub struct SacStepStats {
    /// Mean MSE critic loss across the two online critics for this batch.
    pub critic_loss: f64,
    /// Actor (policy) loss for this batch.
    pub actor_loss: f64,
    /// Entropy-temperature loss (0.0 when `auto_alpha` is disabled).
    pub alpha_loss: f64,
    /// Current entropy temperature `alpha`.
    pub alpha: f64,
    /// Mean of `min(Q1, Q2)(s, a)` over the batch on the stored actions.
    pub mean_q: f64,
    /// Replay buffer fill level at the time of this update.
    pub buffer_len: usize,
}

/// Concrete Adam optimizer type for a SAC sub-network of module type `M`.
type SacAdam<B, M> = OptimizerAdaptor<Adam, M, B>;

/// Burn-backend Soft Actor-Critic trainer.
///
/// Generic only over the autodiff backend `B`; the network shapes are
/// fixed (a [`SacActor`](crate::policy::sac_actor::SacActor) plus four
/// [`ContinuousQNetwork`](crate::policy::continuous_q::ContinuousQNetwork)
/// critics) so the trainer can own the three concrete Adam optimizers
/// directly.
pub struct SacTrainer<B: AutodiffBackend> {
    config: SacConfig,
    obs_dim: usize,
    action_dim: usize,
    target_entropy: f32,

    actor: Option<SacActor<B>>,
    q1: Option<ContinuousQNetwork<B>>,
    q2: Option<ContinuousQNetwork<B>>,
    q1_target: ContinuousQNetwork<B>,
    q2_target: ContinuousQNetwork<B>,
    log_alpha: Option<LogAlpha<B>>,

    actor_opt: SacAdam<B, SacActor<B>>,
    critic_opt: SacAdam<B, ContinuousQNetwork<B>>,
    alpha_opt: SacAdam<B, LogAlpha<B>>,

    buffer: ContinuousReplayBuffer,
    rng: StdRng,
    device: B::Device,
    total_env_steps: usize,
    total_train_steps: usize,
    total_episodes: usize,
}

impl<B: AutodiffBackend> SacTrainer<B> {
    /// Build a new SAC trainer for an `obs_dim`-dimensional observation
    /// and `action_dim`-dimensional continuous action space.
    ///
    /// All five networks are seeded off [`SacConfig::seed`] (the actor,
    /// each critic, and the targets get distinct derived seeds so the
    /// twin critics are decorrelated), and the targets are hard-copied
    /// from their online counterparts so they start byte-equal.
    pub fn new(
        config: SacConfig,
        obs_dim: usize,
        action_dim: usize,
        device: B::Device,
    ) -> Result<Self> {
        config.validate()?;
        if obs_dim == 0 {
            return Err(anyhow!("obs_dim must be positive"));
        }
        if action_dim == 0 {
            return Err(anyhow!("action_dim must be positive"));
        }

        let target_entropy = config.resolved_target_entropy(action_dim);

        let actor_cfg = SacActorConfig {
            num_layers: config.num_hidden_layers,
            hidden_dim: config.hidden_dim,
            use_orthogonal_init: true,
            activation: BurnActivation::ReLU,
            seed: Some(config.seed),
        };
        let actor = SacActor::<B>::with_config(obs_dim, action_dim, actor_cfg, &device);

        // Distinct seeds keep the twin critics decorrelated.
        let q1_cfg = ContinuousQNetworkConfig {
            num_layers: config.num_hidden_layers,
            hidden_dim: config.hidden_dim,
            use_orthogonal_init: true,
            activation: BurnActivation::ReLU,
            seed: Some(config.seed.wrapping_add(1)),
        };
        let q2_cfg = ContinuousQNetworkConfig { seed: Some(config.seed.wrapping_add(2)), ..q1_cfg };
        let q1 = ContinuousQNetwork::<B>::with_config(obs_dim, action_dim, q1_cfg, &device);
        let q2 = ContinuousQNetwork::<B>::with_config(obs_dim, action_dim, q2_cfg, &device);

        // Targets start as exact copies of the online critics.
        let q1_target = ContinuousQNetwork::<B>::with_config(obs_dim, action_dim, q1_cfg, &device)
            .copy_params_from(&q1);
        let q2_target = ContinuousQNetwork::<B>::with_config(obs_dim, action_dim, q2_cfg, &device)
            .copy_params_from(&q2);

        let log_alpha = LogAlpha::<B>::new(config.init_alpha, &device);

        let actor_opt = build_adam::<B, SacActor<B>>(config.max_grad_norm);
        let critic_opt = build_adam::<B, ContinuousQNetwork<B>>(config.max_grad_norm);
        let alpha_opt = build_adam::<B, LogAlpha<B>>(config.max_grad_norm);

        let buffer = ContinuousReplayBuffer::new(config.buffer_capacity, obs_dim, action_dim);
        let rng = StdRng::seed_from_u64(config.seed);

        Ok(Self {
            config,
            obs_dim,
            action_dim,
            target_entropy,
            actor: Some(actor),
            q1: Some(q1),
            q2: Some(q2),
            q1_target,
            q2_target,
            log_alpha: Some(log_alpha),
            actor_opt,
            critic_opt,
            alpha_opt,
            buffer,
            rng,
            device,
            total_env_steps: 0,
            total_train_steps: 0,
            total_episodes: 0,
        })
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &SacConfig {
        &self.config
    }

    /// Observation dimension.
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Action dimension.
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Effective target entropy (after the `-action_dim` heuristic).
    pub fn target_entropy(&self) -> f32 {
        self.target_entropy
    }

    /// Borrow the actor. Panics if called mid-step.
    pub fn actor(&self) -> &SacActor<B> {
        self.actor.as_ref().expect("actor is None mid-step")
    }

    /// Borrow the first online critic. Panics if called mid-step.
    pub fn q1(&self) -> &ContinuousQNetwork<B> {
        self.q1.as_ref().expect("q1 is None mid-step")
    }

    /// Borrow the second online critic. Panics if called mid-step.
    pub fn q2(&self) -> &ContinuousQNetwork<B> {
        self.q2.as_ref().expect("q2 is None mid-step")
    }

    /// Borrow the replay buffer.
    pub fn buffer(&self) -> &ContinuousReplayBuffer {
        &self.buffer
    }

    /// Mutably borrow the replay buffer.
    pub fn buffer_mut(&mut self) -> &mut ContinuousReplayBuffer {
        &mut self.buffer
    }

    /// Number of transitions currently in the buffer.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Current entropy temperature `alpha = exp(log_alpha)`.
    pub fn alpha(&self) -> f64 {
        self.log_alpha.as_ref().expect("log_alpha is None mid-step").alpha_scalar()
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

    /// Caller invokes this once per environment step.
    pub fn increment_env_step(&mut self) {
        self.total_env_steps += 1;
    }

    /// Caller invokes this when an episode terminates / truncates.
    pub fn increment_episodes(&mut self, n: usize) {
        self.total_episodes += n;
    }

    /// `true` while the trainer is still in the random-action warmup
    /// window (env steps `< learning_starts`).
    pub fn in_warmup(&self) -> bool {
        self.total_env_steps < self.config.learning_starts
    }

    /// Select an action for `obs` (shape `[obs_dim]`).
    ///
    /// During the warmup window ([`Self::in_warmup`]) the action is drawn
    /// uniformly in `(-1, 1)` per dimension; afterwards it is a stochastic
    /// sample from the actor. The returned action is in the tanh-squashed
    /// `(-1, 1)` box — the env is responsible for rescaling to its own
    /// action range.
    pub fn select_action(&mut self, obs: &[f32]) -> Vec<f32> {
        if self.in_warmup() {
            (0..self.action_dim).map(|_| self.rng.random_range(-1.0..1.0)).collect()
        } else {
            let obs_t = Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(obs.to_vec(), [1, self.obs_dim]),
                &self.device,
            );
            let actor = self.actor.as_ref().expect("actor is None mid-step");
            let (action, _log_prob) = actor.sample(obs_t, &mut self.rng);
            action.into_data().to_vec().expect("action tensor to host vec")
        }
    }

    /// Deterministic evaluation action `tanh(mu)` for `obs` (shape
    /// `[obs_dim]`), bypassing exploration noise.
    pub fn eval_action(&self, obs: &[f32]) -> Vec<f32> {
        let obs_t = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(obs.to_vec(), [1, self.obs_dim]),
            &self.device,
        );
        let actor = self.actor.as_ref().expect("actor is None mid-step");
        actor
            .mean_action(obs_t)
            .into_data()
            .to_vec()
            .expect("action tensor to host vec")
    }

    /// Run `gradient_steps_per_env_step` SAC gradient updates if the
    /// buffer is ready.
    ///
    /// Returns `Ok(None)` until the buffer holds `min_buffer_size`
    /// transitions; afterwards returns the stats from the **last** of the
    /// performed updates.
    pub fn train(&mut self) -> Result<Option<SacStepStats>> {
        if !self.buffer.is_ready(self.config.min_buffer_size) {
            return Ok(None);
        }
        let mut last = None;
        for _ in 0..self.config.gradient_steps_per_env_step {
            last = Some(self.train_step()?);
        }
        Ok(last)
    }

    /// Run exactly one SAC gradient update (critics, actor, optional
    /// alpha, then a Polyak soft update of both targets).
    ///
    /// Returns an error if any loss is non-finite. Assumes the buffer
    /// holds at least one transition; callers typically gate this on
    /// [`ContinuousReplayBuffer::is_ready`] via [`Self::train`].
    pub fn train_step(&mut self) -> Result<SacStepStats> {
        let batch = sample_continuous(&self.buffer, self.config.batch_size, &mut self.rng);
        let buffer_len = self.buffer.len();
        let t = batch.to_burn_tensors::<B>(&self.device);

        let gamma = self.config.gamma as f32;

        let mut actor = self.actor.take().ok_or_else(|| anyhow!("actor None; reentrant step?"))?;
        let mut q1 = self.q1.take().ok_or_else(|| anyhow!("q1 None; reentrant step?"))?;
        let mut q2 = self.q2.take().ok_or_else(|| anyhow!("q2 None; reentrant step?"))?;
        let mut log_alpha = self
            .log_alpha
            .take()
            .ok_or_else(|| anyhow!("log_alpha None; reentrant step?"))?;

        let alpha = log_alpha.value().exp().detach().into_scalar().to_f32();

        // ----- Critic update -----
        // Build the TD target with a' sampled from the *current* actor at
        // s'. The whole target is detached so no gradient flows into the
        // actor or target critics here.
        let (next_action, next_log_prob) = actor.sample(t.next_observations.clone(), &mut self.rng);
        let next_action = next_action.detach();
        let next_log_prob = next_log_prob.detach();

        let q1_t = self.q1_target.forward(t.next_observations.clone(), next_action.clone());
        let q2_t = self.q2_target.forward(t.next_observations.clone(), next_action);
        let min_q_next = min_pair(q1_t, q2_t);
        let soft_value = min_q_next - next_log_prob.mul_scalar(alpha);

        // y = r + gamma * (1 - done) * soft_value
        let not_done = -t.dones.clone() + 1.0;
        let td_target = (t.rewards.clone() + soft_value.mul_scalar(gamma) * not_done).detach();

        let q1_pred = q1.forward(t.observations.clone(), t.actions.clone());
        let q2_pred = q2.forward(t.observations.clone(), t.actions.clone());

        // mean of min(Q1, Q2) over the stored batch actions, for stats —
        // detached so it does not entangle the loss graph we back-prop.
        let mean_q_val = min_pair(q1_pred.clone(), q2_pred.clone())
            .mean()
            .detach()
            .into_scalar()
            .to_f64();

        let critic1_loss = mse(q1_pred, td_target.clone());
        let critic2_loss = mse(q2_pred, td_target);
        // Sum the two critic losses and back-prop once. q1 and q2 are
        // independent modules, so a single `backward()` lets us extract a
        // disjoint `GradientsParams` for each from the shared gradient
        // container — this avoids back-propagating twice over graph nodes
        // (the shared `t.observations` / `t.actions` leaves) that Burn's
        // graph cleaner would otherwise prune after the first pass.
        let critic_loss = critic1_loss + critic2_loss;
        let critic_loss_val = critic_loss.clone().detach().into_scalar().to_f64() / 2.0;
        if !critic_loss_val.is_finite() {
            return Err(anyhow!("Non-finite critic loss: {}", critic_loss_val));
        }
        let mut critic_grads = critic_loss.backward();
        let grads1 = GradientsParams::from_module(&mut critic_grads, &q1);
        let grads2 = GradientsParams::from_module(&mut critic_grads, &q2);
        q1 = self.critic_opt.step(self.config.critic_lr, q1, grads1);
        q2 = self.critic_opt.step(self.config.critic_lr, q2, grads2);

        // ----- Actor update -----
        // Reparameterized action from the current actor; the critics are
        // frozen (we clone the just-updated online nets and detach their
        // outputs' graph from the critic params by not collecting their
        // grads).
        let (pi_action, pi_log_prob) = actor.sample(t.observations.clone(), &mut self.rng);
        let q1_pi = q1.forward(t.observations.clone(), pi_action.clone());
        let q2_pi = q2.forward(t.observations.clone(), pi_action);
        let min_q_pi = min_pair(q1_pi, q2_pi);
        // The entropy gap `(log_prob + target_entropy)` is needed (detached)
        // by the alpha update; snapshot it before the actor backward consumes
        // the policy graph.
        let entropy_gap = pi_log_prob.clone().add_scalar(self.target_entropy).detach();
        // loss = mean(alpha * log_prob - min_q_pi)
        let actor_loss = (pi_log_prob.mul_scalar(alpha) - min_q_pi).mean();
        let actor_loss_val = actor_loss.clone().detach().into_scalar().to_f64();
        if !actor_loss_val.is_finite() {
            return Err(anyhow!("Non-finite actor loss: {}", actor_loss_val));
        }
        let actor_grads = GradientsParams::from_grads(actor_loss.backward(), &actor);
        actor = self.actor_opt.step(self.config.actor_lr, actor, actor_grads);

        // ----- Alpha (entropy temperature) update -----
        let mut alpha_loss_val = 0.0;
        if self.config.auto_alpha {
            // loss = -mean(log_alpha * (log_prob + target_entropy).detach())
            let log_alpha_t = log_alpha.value();
            // Broadcast log_alpha (shape [1]) across the batch entropy gap.
            let alpha_loss = -(log_alpha_t * entropy_gap).mean();
            alpha_loss_val = alpha_loss.clone().detach().into_scalar().to_f64();
            if !alpha_loss_val.is_finite() {
                return Err(anyhow!("Non-finite alpha loss: {}", alpha_loss_val));
            }
            let alpha_grads = GradientsParams::from_grads(alpha_loss.backward(), &log_alpha);
            log_alpha = self.alpha_opt.step(self.config.alpha_lr, log_alpha, alpha_grads);
        }

        // ----- Polyak soft target update -----
        self.q1_target.soft_update_from(&q1, self.config.tau);
        self.q2_target.soft_update_from(&q2, self.config.tau);

        let alpha_after = log_alpha.alpha_scalar();

        self.actor = Some(actor);
        self.q1 = Some(q1);
        self.q2 = Some(q2);
        self.log_alpha = Some(log_alpha);
        self.total_train_steps += 1;

        Ok(SacStepStats {
            critic_loss: critic_loss_val,
            actor_loss: actor_loss_val,
            alpha_loss: alpha_loss_val,
            alpha: alpha_after,
            mean_q: mean_q_val,
            buffer_len,
        })
    }
}

/// Build an Adam optimizer for module type `M`, applying the optional
/// global gradient-norm clip from the config.
fn build_adam<B, M>(max_grad_norm: Option<f64>) -> SacAdam<B, M>
where
    B: AutodiffBackend,
    M: burn::module::AutodiffModule<B>,
{
    let mut cfg = AdamConfig::new();
    if let Some(norm) = max_grad_norm {
        cfg = cfg.with_grad_clipping(Some(GradientClippingConfig::Norm(norm as f32)));
    }
    cfg.init()
}

/// Elementwise minimum of two `[batch]` Q-value tensors (clipped
/// double-Q): `min(a, b) = b + min(a - b, 0)`.
fn min_pair<B: AutodiffBackend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    let diff = a - b.clone();
    b + diff.clamp_max(0.0)
}

/// Mean-squared-error between a prediction and a (detached) target, both
/// shape `[batch]`.
fn mse<B: AutodiffBackend>(pred: Tensor<B, 1>, target: Tensor<B, 1>) -> Tensor<B, 1> {
    let diff = pred - target;
    (diff.clone() * diff).mean()
}

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};

    use super::*;

    type B = Autodiff<NdArray<f32>>;

    fn tiny_config() -> SacConfig {
        SacConfig::new()
            .buffer_capacity(256)
            .min_buffer_size(8)
            .batch_size(8)
            .learning_starts(4)
            .hidden_dim(16)
            .seed(7)
    }

    fn fill_buffer(trainer: &mut SacTrainer<B>, n: usize) {
        for i in 0..n {
            let phase = i as f32 * 0.1;
            let obs = [phase.cos(), phase.sin(), phase * 0.2];
            let next_obs = [(phase + 0.1).cos(), (phase + 0.1).sin(), phase * 0.2];
            let action = [(phase.sin()).clamp(-0.99, 0.99)];
            let reward = -(phase * phase);
            let done = i % 5 == 4;
            trainer.buffer_mut().push(&obs, &action, reward, &next_obs, done);
        }
    }

    #[test]
    fn trainer_constructs_and_copies_targets() {
        let device = Default::default();
        let trainer = SacTrainer::<B>::new(tiny_config(), 3, 1, device).unwrap();
        assert_eq!(trainer.total_env_steps(), 0);
        assert_eq!(trainer.buffer_len(), 0);
        assert_eq!(trainer.target_entropy(), -1.0);

        // Targets start byte-equal to their online critics.
        let obs = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.1, 0.2, 0.3], [1, 3]),
            &Default::default(),
        );
        let act = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.4], [1, 1]),
            &Default::default(),
        );
        let on: f32 = trainer.q1().forward(obs.clone(), act.clone()).into_scalar().to_f32();
        let tg: f32 = trainer.q1_target.forward(obs, act).into_scalar().to_f32();
        assert!((on - tg).abs() < 1e-6, "target must start equal to online critic");
    }

    #[test]
    fn rejects_invalid_config() {
        let device = Default::default();
        let bad = SacConfig::new().gamma(2.0);
        assert!(SacTrainer::<B>::new(bad, 3, 1, device).is_err());
    }

    #[test]
    fn rejects_zero_dims() {
        let device = Default::default();
        assert!(SacTrainer::<B>::new(tiny_config(), 0, 1, device).is_err());
        let device2 = Default::default();
        assert!(SacTrainer::<B>::new(tiny_config(), 3, 0, device2).is_err());
    }

    #[test]
    fn train_returns_none_until_ready() {
        let device = Default::default();
        let mut trainer = SacTrainer::<B>::new(tiny_config(), 3, 1, device).unwrap();
        fill_buffer(&mut trainer, 4); // below min_buffer_size (8)
        assert!(trainer.train().unwrap().is_none());
    }

    #[test]
    fn one_train_step_runs_with_finite_losses() {
        let device = Default::default();
        let mut trainer = SacTrainer::<B>::new(tiny_config(), 3, 1, device).unwrap();
        fill_buffer(&mut trainer, 32);
        let stats = trainer.train().unwrap().expect("should train once buffer ready");
        assert!(stats.critic_loss.is_finite(), "critic loss finite");
        assert!(stats.actor_loss.is_finite(), "actor loss finite");
        assert!(stats.alpha_loss.is_finite(), "alpha loss finite");
        assert!(stats.alpha.is_finite() && stats.alpha > 0.0, "alpha positive finite");
        assert!(stats.mean_q.is_finite(), "mean_q finite");
        assert_eq!(stats.buffer_len, 32);
        assert_eq!(trainer.total_train_steps(), 1);
    }

    #[test]
    fn fixed_alpha_keeps_alpha_constant() {
        let device = Default::default();
        let cfg = tiny_config().auto_alpha(false).init_alpha(0.3);
        let mut trainer = SacTrainer::<B>::new(cfg, 3, 1, device).unwrap();
        fill_buffer(&mut trainer, 32);
        let before = trainer.alpha();
        for _ in 0..5 {
            trainer.train().unwrap();
        }
        let after = trainer.alpha();
        assert!((before - after).abs() < 1e-9, "fixed alpha must not move: {before} -> {after}");
        assert!((after - 0.3).abs() < 1e-6);
    }

    #[test]
    fn target_moves_after_step() {
        let device = Default::default();
        let mut trainer = SacTrainer::<B>::new(tiny_config(), 3, 1, device).unwrap();
        fill_buffer(&mut trainer, 32);

        let obs = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.1, 0.2, 0.3], [1, 3]),
            &Default::default(),
        );
        let act = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.4], [1, 1]),
            &Default::default(),
        );
        let tg_before: f32 =
            trainer.q1_target.forward(obs.clone(), act.clone()).into_scalar().to_f32();
        for _ in 0..5 {
            trainer.train().unwrap();
        }
        let tg_after: f32 = trainer.q1_target.forward(obs, act).into_scalar().to_f32();
        assert!(
            (tg_before - tg_after).abs() > 1e-7,
            "soft update should move target: {tg_before} -> {tg_after}"
        );
    }

    #[test]
    fn select_action_warmup_then_policy_in_range() {
        let device = Default::default();
        let mut trainer = SacTrainer::<B>::new(tiny_config(), 3, 1, device).unwrap();
        // Warmup: random in (-1, 1).
        let a = trainer.select_action(&[0.1, 0.2, 0.3]);
        assert_eq!(a.len(), 1);
        assert!(a[0] > -1.0 && a[0] < 1.0);
        // Advance past learning_starts and sample from the policy.
        for _ in 0..trainer.config().learning_starts {
            trainer.increment_env_step();
        }
        assert!(!trainer.in_warmup());
        let a = trainer.select_action(&[0.1, 0.2, 0.3]);
        assert_eq!(a.len(), 1);
        assert!(a[0] > -1.0 && a[0] < 1.0);
        // Deterministic eval action also in range.
        let e = trainer.eval_action(&[0.1, 0.2, 0.3]);
        assert!(e[0] > -1.0 && e[0] < 1.0);
    }
}
