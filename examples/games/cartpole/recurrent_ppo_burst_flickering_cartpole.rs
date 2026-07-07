//! Recurrent PPO on **burst-flickering** CartPole (correlated-occlusion POMDP).
//!
//! Part of the recurrent-policy epic (#262), issue #302. This example is the
//! correlated-dropout counterpart of `recurrent_ppo_flickering_cartpole`
//! (#298). It trains an
//! [`LstmBurnPolicy`](thrust_rl::policy::lstm::LstmBurnPolicy)
//! and a feedforward [`MlpBurnPolicy`](thrust_rl::policy::mlp::MlpBurnPolicy)
//! baseline on [`FlickeringCartPole`](thrust_rl::env::FlickeringCartPole),
//! under **two** dropout regimes at the *same* blank rate `p`:
//!
//! 1. **i.i.d.** — each frame blanked independently with probability `p` (the
//!    #298 protocol).
//! 2. **burst** — blanks arrive in correlated runs via a two-state Markov chain
//!    with mean blank-run length `burst_len` (default 4) and the *same*
//!    stationary blank fraction `p`.
//!
//! # Why the burst regime
//!
//! i.i.d. dropout is only *partially* memory-hard: a reactive controller
//! compensates for isolated blanked frames at CartPole's control rate, so the
//! MLP baseline does not collapse (#298 measured LSTM ~484 vs MLP ~176). Burst
//! dropout keeps the same overall blank rate but forces the blanks into
//! multi-step runs the reactive controller cannot bridge — its last real
//! observation is several steps stale. The recurrent policy integrates over the
//! gap. The hypothesis is that correlation *widens* the LSTM-vs-MLP gap
//! relative to i.i.d. This example reports both gaps side by side so the effect
//! is measured directly (honestly, either direction).
//!
//! # Measured results (issue #302, 200k steps/arm, p=0.5, burst_len=4, seed 0)
//!
//! ```text
//! regime   LSTM (final/best)   MLP (final/best)   gap (best)
//! iid        420.0 / 422.0       144.5 / 158.7       263.4
//! burst      178.9 / 203.1        89.0 / 100.4       102.7
//! ```
//!
//! **Honest negative on the absolute-gap hypothesis**: correlated bursts made
//! the task harder for BOTH policy classes, and the absolute LSTM−MLP gap
//! *narrowed* (263 → 103) rather than widened. The relative picture is more
//! nuanced: the burst LSTM still clears the 195 solved bar (peak 203) while the
//! burst MLP drops to ~100 — roughly half its i.i.d. ceiling — so memory
//! remains decisively load-bearing; the LSTM simply loses more absolute return
//! to the harder observation stream than the MLP has left to lose. Consistent
//! with the #298 i.i.d. reference (LSTM 484 vs MLP 176 at 500k; here 200k).
//!
//! # Usage
//!
//! ```bash
//! # Run all four arms (iid LSTM/MLP, burst LSTM/MLP) and print both gaps:
//! cargo run --example recurrent_ppo_burst_flickering_cartpole --features training --release
//! ```
//!
//! Env overrides: `TOTAL_TIMESTEPS` (per-arm budget, default 200k),
//! `FLICKER_PROB` (blank rate, default 0.5), `BURST_LEN` (mean blank-run
//! length, default 4).

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use burn::{
    backend::Autodiff,
    nn::LstmState,
    optim::AdamConfig,
    tensor::{Int, Tensor, TensorData, activation},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use thrust_rl::{
    buffer::rollout::RecurrentRolloutBuffer,
    env::{Environment, SpaceType, flickering_cartpole::FlickeringCartPole, pool::EnvPool},
    policy::{
        lstm::{LstmBurnConfig, LstmBurnPolicy},
        mlp::{BurnActivation, MlpBurnConfig, MlpBurnPolicy},
    },
    train::{
        optimizer::BurnOptimizer,
        ppo::{PPOConfig, PPOTrainerBurn, RecurrentPPOTrainer},
    },
};

type InnerBackend = burn::backend::NdArray<f32>;
type Backend = Autodiff<InnerBackend>;

const NUM_ENVS: usize = 16;
const NUM_STEPS: usize = 128;
const HIDDEN_DIM: usize = 64;
const DEFAULT_TIMESTEPS: usize = 200_000;
const DEFAULT_FLICKER_PROB: f64 = 0.5;
const DEFAULT_BURST_LEN: f64 = 4.0;
const LSTM_LR: f64 = 1.5e-3;
const MLP_LR: f64 = 3e-4;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;
const N_EPOCHS: usize = 4;
const ENVS_PER_MINIBATCH: usize = 4;
const SEED: u64 = 0;
/// Train-time reward scale (both arms, symmetric); see #298 for the rationale.
const REWARD_SCALE: f32 = 0.02;
const SOLVED_THRESHOLD: f32 = 195.0;

/// Which dropout regime an arm uses.
#[derive(Clone, Copy, PartialEq)]
enum Regime {
    /// i.i.d. per-frame dropout (the #298 protocol).
    Iid,
    /// Correlated bursts (Markov) with mean blank-run length `burst_len`.
    Burst(f64),
}

impl Regime {
    fn label(self) -> &'static str {
        match self {
            Regime::Iid => "iid",
            Regime::Burst(_) => "burst",
        }
    }
}

#[derive(Clone, Copy)]
struct ArmResult {
    final_mean: f32,
    best_mean: f32,
}

/// Build a pool of flickering CartPoles under the given dropout regime, with
/// per-env seeded streams (env `i` gets seed `base_seed*1000 + i`). Both policy
/// arms in a regime call this with the same `base_seed`, so they see the
/// identical dropout stream.
fn make_pool(base_seed: u64, flicker_prob: f64, regime: Regime) -> EnvPool<FlickeringCartPole> {
    let ctr = AtomicU64::new(base_seed.wrapping_mul(1000));
    EnvPool::new(
        move || {
            let s = ctr.fetch_add(1, Ordering::Relaxed);
            match regime {
                Regime::Iid => FlickeringCartPole::with_seed_and_probability(s, flicker_prob),
                Regime::Burst(l) => {
                    FlickeringCartPole::with_seed_probability_and_burst(s, flicker_prob, l)
                }
            }
        },
        NUM_ENVS,
    )
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let total_timesteps: usize = std::env::var("TOTAL_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TIMESTEPS);
    let flicker_prob: f64 = std::env::var("FLICKER_PROB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_FLICKER_PROB);
    let burst_len: f64 = std::env::var("BURST_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BURST_LEN);

    tracing::info!("Recurrent PPO on burst-flickering CartPole (correlated-occlusion POMDP)");
    tracing::info!("  num_envs={}  num_steps={}  hidden_dim={}", NUM_ENVS, NUM_STEPS, HIDDEN_DIM);
    tracing::info!("  flicker_prob={}  burst_len={}", flicker_prob, burst_len);
    tracing::info!("  total_timesteps={} per arm (4 arms)", total_timesteps);
    tracing::info!("------------------------------------------------------------");

    let regimes = [Regime::Iid, Regime::Burst(burst_len)];
    // (regime, lstm, mlp)
    let mut results: Vec<(Regime, ArmResult, ArmResult)> = Vec::new();

    for &regime in &regimes {
        tracing::info!("########## Regime: {} ##########", regime.label());
        tracing::info!(">>> [{}] Training LSTM (recurrent)", regime.label());
        let lstm = run_lstm(total_timesteps, flicker_prob, regime)?;
        tracing::info!(">>> [{}] Training feedforward (MLP) baseline", regime.label());
        let mlp = run_feedforward(total_timesteps, flicker_prob, regime)?;
        results.push((regime, lstm, mlp));
    }

    tracing::info!("============================================================");
    tracing::info!("Results (mean return of last ≤100 episodes, RAW units):");
    let mut iid_gap = None;
    let mut burst_gap = None;
    for (regime, lstm, mlp) in &results {
        let gap = lstm.best_mean - mlp.best_mean;
        tracing::info!(
            "  [{:<5}] LSTM final {:.1} / best {:.1}   MLP final {:.1} / best {:.1}   gap(best) {:.1}",
            regime.label(),
            lstm.final_mean,
            lstm.best_mean,
            mlp.final_mean,
            mlp.best_mean,
            gap,
        );
        tracing::info!(
            "          LSTM {} solved bar {} (peak {:.1})",
            if lstm.best_mean > SOLVED_THRESHOLD {
                "CLEARED"
            } else {
                "MISSED"
            },
            SOLVED_THRESHOLD,
            lstm.best_mean,
        );
        match regime {
            Regime::Iid => iid_gap = Some(gap),
            Regime::Burst(_) => burst_gap = Some(gap),
        }
    }

    if let (Some(i), Some(b)) = (iid_gap, burst_gap) {
        tracing::info!("------------------------------------------------------------");
        tracing::info!("  i.i.d.  memory gap (LSTM-MLP, best): {:.1}", i);
        tracing::info!("  burst   memory gap (LSTM-MLP, best): {:.1}", b);
        let delta = b - i;
        tracing::info!(
            "  Correlation {} the memory advantage by {:.1} ({}).",
            if delta > 0.0 {
                "WIDENED"
            } else {
                "did NOT widen"
            },
            delta.abs(),
            if delta > 0.0 {
                "burst dropout is harder for the reactive baseline, as hypothesized"
            } else {
                "reported honestly against the hypothesis"
            }
        );
    }

    Ok(())
}

fn run_lstm(total_timesteps: usize, flicker_prob: f64, regime: Regime) -> Result<ArmResult> {
    let start = std::time::Instant::now();
    let device = Default::default();

    let probe = FlickeringCartPole::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        SpaceType::Discrete(n) => n,
        _ => panic!("FlickeringCartPole must be discrete"),
    };

    let mut env_pool = make_pool(SEED, flicker_prob, regime);

    let policy_config =
        LstmBurnConfig { hidden_dim: HIDDEN_DIM, ..Default::default() }.with_seed(SEED);
    let policy =
        LstmBurnPolicy::<Backend>::with_config(obs_dim, action_dim, policy_config, &device);

    let lr = LSTM_LR;
    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<Backend, LstmBurnPolicy<Backend>, _> =
        BurnOptimizer::new(inner_opt, lr);

    let ppo_config = PPOConfig::new()
        .learning_rate(lr)
        .n_epochs(N_EPOCHS)
        .gamma(GAMMA as f64)
        .gae_lambda(GAE_LAMBDA as f64)
        .clip_range(0.2)
        .clip_range_vf(0.2)
        .vf_coef(0.5)
        .ent_coef(0.01)
        .max_grad_norm(0.5)
        .target_kl(1.0);

    let mut trainer = RecurrentPPOTrainer::new(ppo_config, policy, burn_opt, device)?;
    let num_updates = total_timesteps / (NUM_STEPS * NUM_ENVS);

    let mut norm = ObsNormalizer::new(obs_dim);
    let mut buffer = RecurrentRolloutBuffer::new(NUM_STEPS, NUM_ENVS, obs_dim, HIDDEN_DIM);
    let raw0 = env_pool.reset();
    let mut observations: Vec<Vec<f32>> = raw0.iter().map(|o| norm.normalize(o)).collect();
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut episode_returns = [0.0_f32; NUM_ENVS];
    let mut completed: Vec<f32> = Vec::new();
    let mut lstm_state: Option<LstmState<Backend, 2>> = None;
    let mut last_h = vec![0.0_f32; NUM_ENVS * HIDDEN_DIM];
    let mut last_c = vec![0.0_f32; NUM_ENVS * HIDDEN_DIM];
    let mut total_env_steps = 0usize;
    let mut last_mean = 0.0_f32;
    let mut best_mean = 0.0_f32;

    for update in 0..num_updates {
        buffer.reset();

        for step in 0..NUM_STEPS {
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_t = Tensor::<Backend, 2>::from_data(
                TensorData::new(obs_flat, [NUM_ENVS, obs_dim]),
                &device,
            );

            let (logits, values_t, new_state) =
                trainer.policy().forward_step(obs_t, lstm_state.take());
            let (actions, log_probs, values_host) = sample_actions(&logits, &values_t, &mut rng);

            let results = env_pool.step(&actions);

            let mut h_host: Vec<f32> = new_state.hidden.into_data().to_vec().unwrap();
            let mut c_host: Vec<f32> = new_state.cell.into_data().to_vec().unwrap();

            for env in 0..NUM_ENVS {
                let r = &results[env];
                buffer.add(
                    step,
                    env,
                    &observations[env],
                    actions[env],
                    r.reward * REWARD_SCALE,
                    values_host[env],
                    log_probs[env],
                    r.terminated,
                    r.truncated,
                );
                episode_returns[env] += r.reward;
                observations[env] = norm.normalize(&r.observation);

                if r.terminated || r.truncated {
                    completed.push(episode_returns[env]);
                    episode_returns[env] = 0.0;
                    trainer.increment_episodes(1);
                    for k in 0..HIDDEN_DIM {
                        h_host[env * HIDDEN_DIM + k] = 0.0;
                        c_host[env * HIDDEN_DIM + k] = 0.0;
                    }
                    let raw = env_pool.reset_env(env)?;
                    observations[env] = norm.normalize(&raw);
                }
            }

            last_h.copy_from_slice(&h_host);
            last_c.copy_from_slice(&c_host);
            let hidden_t = Tensor::<Backend, 2>::from_data(
                TensorData::new(h_host, [NUM_ENVS, HIDDEN_DIM]),
                &device,
            );
            let cell_t = Tensor::<Backend, 2>::from_data(
                TensorData::new(c_host, [NUM_ENVS, HIDDEN_DIM]),
                &device,
            );
            lstm_state = Some(LstmState::new(cell_t, hidden_t));
            total_env_steps += NUM_ENVS;
        }

        let last_obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let last_obs_t = Tensor::<Backend, 2>::from_data(
            TensorData::new(last_obs_flat, [NUM_ENVS, obs_dim]),
            &device,
        );
        let boot_hidden = Tensor::<Backend, 2>::from_data(
            TensorData::new(last_h.clone(), [NUM_ENVS, HIDDEN_DIM]),
            &device,
        );
        let boot_cell = Tensor::<Backend, 2>::from_data(
            TensorData::new(last_c.clone(), [NUM_ENVS, HIDDEN_DIM]),
            &device,
        );
        let (_, last_values_t, _) = trainer
            .policy()
            .forward_step(last_obs_t, Some(LstmState::new(boot_cell, boot_hidden)));
        let last_values: Vec<f32> = last_values_t.into_data().to_vec().unwrap();

        buffer.compute_advantages(&last_values, GAMMA, GAE_LAMBDA);

        let frac = 1.0 - (update as f64) / (num_updates.max(1) as f64);
        trainer.set_learning_rate(lr * frac.max(0.05));

        let stats =
            trainer.train_step(&buffer, ENVS_PER_MINIBATCH, |p, obs_seq, actions, starts| {
                p.evaluate_sequences(obs_seq, actions, None, starts)
            })?;

        let final_hidden: Vec<Vec<f32>> = (0..NUM_ENVS)
            .map(|e| last_h[e * HIDDEN_DIM..(e + 1) * HIDDEN_DIM].to_vec())
            .collect();
        let final_cell: Vec<Vec<f32>> = (0..NUM_ENVS)
            .map(|e| last_c[e * HIDDEN_DIM..(e + 1) * HIDDEN_DIM].to_vec())
            .collect();
        buffer.seed_warm_start(NUM_STEPS - 1, &final_hidden, &final_cell);

        lstm_state = None;

        if !completed.is_empty() {
            let n = completed.len();
            let recent = &completed[n.saturating_sub(100)..];
            last_mean = recent.iter().sum::<f32>() / recent.len() as f32;
            best_mean = best_mean.max(last_mean);
        }

        if update % 10 == 0 || update == num_updates - 1 {
            tracing::info!(
                "  [lstm {:<5}] update {:>3}/{}  env_steps={:>7}  episodes={:>5}  mean_return(last≤100)={:6.1}  entropy={:5.3}  explained_var={:5.3}",
                regime.label(),
                update + 1,
                num_updates,
                total_env_steps,
                trainer.total_episodes(),
                last_mean,
                stats.entropy,
                stats.explained_var,
            );
        }
    }

    tracing::info!(
        "  [lstm {}] done in {:.1}s — final {:.1} (best {:.1})",
        regime.label(),
        start.elapsed().as_secs_f64(),
        last_mean,
        best_mean
    );
    Ok(ArmResult { final_mean: last_mean, best_mean })
}

fn run_feedforward(total_timesteps: usize, flicker_prob: f64, regime: Regime) -> Result<ArmResult> {
    let start = std::time::Instant::now();
    let device = Default::default();

    let probe = FlickeringCartPole::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        SpaceType::Discrete(n) => n,
        _ => panic!("FlickeringCartPole must be discrete"),
    };

    let mut env_pool = make_pool(SEED, flicker_prob, regime);

    let policy_config = MlpBurnConfig {
        num_layers: 2,
        hidden_dim: 128,
        use_orthogonal_init: true,
        activation: BurnActivation::ReLU,
        seed: Some(SEED),
    };
    let policy = MlpBurnPolicy::<Backend>::with_config(obs_dim, action_dim, policy_config, &device);

    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<Backend, MlpBurnPolicy<Backend>, _> =
        BurnOptimizer::new(inner_opt, MLP_LR);

    let ppo_config = PPOConfig::new()
        .learning_rate(MLP_LR)
        .n_epochs(10)
        .batch_size(128)
        .gamma(GAMMA as f64)
        .gae_lambda(GAE_LAMBDA as f64)
        .clip_range(0.2)
        .clip_range_vf(0.2)
        .vf_coef(0.5)
        .ent_coef(0.01)
        .max_grad_norm(0.5)
        .target_kl(1.0);

    let mut trainer = PPOTrainerBurn::new(ppo_config, policy, burn_opt)?;
    let num_updates = total_timesteps / (NUM_STEPS * NUM_ENVS);

    let cap = NUM_STEPS * NUM_ENVS;
    let mut buf_obs: Vec<f32> = Vec::with_capacity(cap * obs_dim);
    let mut buf_actions: Vec<i64> = Vec::with_capacity(cap);
    let mut buf_log_probs: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_values: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_rewards: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_dones: Vec<f32> = Vec::with_capacity(cap);

    let mut norm = ObsNormalizer::new(obs_dim);
    let mut observations: Vec<Vec<f32>> =
        env_pool.reset().iter().map(|o| norm.normalize(o)).collect();
    let mut episode_returns = [0.0_f32; NUM_ENVS];
    let mut completed: Vec<f32> = Vec::new();
    let mut total_env_steps = 0usize;
    let mut last_mean = 0.0_f32;
    let mut best_mean = 0.0_f32;

    for update in 0..num_updates {
        buf_obs.clear();
        buf_actions.clear();
        buf_log_probs.clear();
        buf_values.clear();
        buf_rewards.clear();
        buf_dones.clear();

        for _step in 0..NUM_STEPS {
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_t = Tensor::<Backend, 2>::from_data(
                TensorData::new(obs_flat, [NUM_ENVS, obs_dim]),
                &device,
            );
            let (actions, log_probs, values) = trainer.policy().get_action_host(obs_t);
            let results = env_pool.step(&actions);

            for env_id in 0..NUM_ENVS {
                buf_obs.extend_from_slice(&observations[env_id]);
                buf_actions.push(actions[env_id]);
                buf_log_probs.push(log_probs[env_id]);
                buf_values.push(values[env_id]);
                buf_rewards.push(results[env_id].reward * REWARD_SCALE);
                let done = results[env_id].terminated || results[env_id].truncated;
                buf_dones.push(if done { 1.0 } else { 0.0 });

                episode_returns[env_id] += results[env_id].reward;
                observations[env_id] = norm.normalize(&results[env_id].observation);
                if done {
                    completed.push(episode_returns[env_id]);
                    episode_returns[env_id] = 0.0;
                    trainer.increment_episodes(1);
                    let raw = env_pool.reset_env(env_id)?;
                    observations[env_id] = norm.normalize(&raw);
                }
            }
            total_env_steps += NUM_ENVS;
        }

        let last_obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let last_obs_t = Tensor::<Backend, 2>::from_data(
            TensorData::new(last_obs_flat, [NUM_ENVS, obs_dim]),
            &device,
        );
        let (_, _, last_values_host) = trainer.policy().get_action_host(last_obs_t);

        let (advantages_host, returns_host) = compute_gae(
            &buf_rewards,
            &buf_values,
            &buf_dones,
            &last_values_host,
            GAMMA,
            GAE_LAMBDA,
            NUM_STEPS,
            NUM_ENVS,
        );

        let batch = NUM_STEPS * NUM_ENVS;
        let obs_b = Tensor::<Backend, 2>::from_data(
            TensorData::new(buf_obs.clone(), [batch, obs_dim]),
            &device,
        );
        let actions_b = Tensor::<Backend, 1, Int>::from_data(
            TensorData::new(buf_actions.clone(), [batch]),
            &device,
        );
        let old_log_probs_b = Tensor::<Backend, 1>::from_data(
            TensorData::new(buf_log_probs.clone(), [batch]),
            &device,
        );
        let old_values_b =
            Tensor::<Backend, 1>::from_data(TensorData::new(buf_values.clone(), [batch]), &device);
        let advantages_b =
            Tensor::<Backend, 1>::from_data(TensorData::new(advantages_host, [batch]), &device);
        let returns_b =
            Tensor::<Backend, 1>::from_data(TensorData::new(returns_host, [batch]), &device);

        let stats = trainer.train_step(
            obs_b,
            actions_b,
            old_log_probs_b,
            old_values_b,
            advantages_b,
            returns_b,
            |p, o, a| p.evaluate_actions(o, a),
        )?;

        if !completed.is_empty() {
            let n = completed.len();
            let recent = &completed[n.saturating_sub(100)..];
            last_mean = recent.iter().sum::<f32>() / recent.len() as f32;
            best_mean = best_mean.max(last_mean);
        }

        if update % 10 == 0 || update == num_updates - 1 {
            tracing::info!(
                "  [mlp  {:<5}] update {:>3}/{}  env_steps={:>7}  episodes={:>4}  mean_return(last≤100)={:6.1}  entropy={:5.3}",
                regime.label(),
                update + 1,
                num_updates,
                total_env_steps,
                trainer.total_episodes(),
                last_mean,
                stats.entropy,
            );
        }
    }

    tracing::info!(
        "  [mlp  {}] done in {:.1}s — final {:.1} (best {:.1})",
        regime.label(),
        start.elapsed().as_secs_f64(),
        last_mean,
        best_mean
    );
    Ok(ArmResult { final_mean: last_mean, best_mean })
}

/// Running per-dimension observation standardizer with flicker-aware
/// pass-through (identical to `recurrent_ppo_flickering_cartpole`). Blanked
/// (all-zero) frames pass through unchanged and are excluded from the running
/// statistics.
struct ObsNormalizer {
    mean: Vec<f64>,
    m2: Vec<f64>,
    count: f64,
}

impl ObsNormalizer {
    fn new(dim: usize) -> Self {
        Self { mean: vec![0.0; dim], m2: vec![0.0; dim], count: 0.0 }
    }

    fn normalize(&mut self, obs: &[f32]) -> Vec<f32> {
        if obs.iter().all(|&x| x == 0.0) {
            return obs.to_vec();
        }
        self.count += 1.0;
        for (i, &xf) in obs.iter().enumerate() {
            let x = xf as f64;
            let delta = x - self.mean[i];
            self.mean[i] += delta / self.count;
            self.m2[i] += delta * (x - self.mean[i]);
        }
        obs.iter()
            .enumerate()
            .map(|(i, &x)| {
                let std = (self.m2[i] / self.count).sqrt().max(1e-4);
                (((x as f64 - self.mean[i]) / std).clamp(-10.0, 10.0)) as f32
            })
            .collect()
    }
}

/// Host-side categorical sampling from LSTM `forward_step` outputs.
fn sample_actions(
    logits: &Tensor<Backend, 2>,
    values: &Tensor<Backend, 1>,
    rng: &mut StdRng,
) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
    let [n, a] = logits.dims();
    let probs = activation::softmax(logits.clone(), 1);
    let log_probs = activation::log_softmax(logits.clone(), 1);
    let probs_host: Vec<f32> = probs.into_data().to_vec().unwrap();
    let log_probs_host: Vec<f32> = log_probs.into_data().to_vec().unwrap();
    let values_host: Vec<f32> = values.clone().into_data().to_vec().unwrap();

    let mut actions = Vec::with_capacity(n);
    let mut chosen_log_probs = Vec::with_capacity(n);
    for i in 0..n {
        let u: f32 = rng.random();
        let mut cum = 0.0_f32;
        let mut chosen = a - 1;
        for j in 0..a {
            cum += probs_host[i * a + j];
            if u <= cum {
                chosen = j;
                break;
            }
        }
        actions.push(chosen as i64);
        chosen_log_probs.push(log_probs_host[i * a + chosen]);
    }
    (actions, chosen_log_probs, values_host)
}

/// Per-env host-side GAE for the feedforward baseline (step-major `[T * N]`).
#[allow(clippy::too_many_arguments)]
fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    last_values: &[f32],
    gamma: f32,
    gae_lambda: f32,
    num_steps: usize,
    num_envs: usize,
) -> (Vec<f32>, Vec<f32>) {
    let cap = num_steps * num_envs;
    let mut advantages = vec![0.0_f32; cap];
    let mut returns = vec![0.0_f32; cap];

    let mut last_gae = vec![0.0_f32; num_envs];
    for t in (0..num_steps).rev() {
        for n in 0..num_envs {
            let idx = t * num_envs + n;
            let next_value = if t == num_steps - 1 {
                last_values[n]
            } else {
                values[(t + 1) * num_envs + n]
            };
            let next_nonterminal = 1.0 - dones[idx];
            let delta = rewards[idx] + gamma * next_value * next_nonterminal - values[idx];
            last_gae[n] = delta + gamma * gae_lambda * next_nonterminal * last_gae[n];
            advantages[idx] = last_gae[n];
            returns[idx] = advantages[idx] + values[idx];
        }
    }

    (advantages, returns)
}
