//! Recurrent PPO on the T-maze (a *provably* memory-hard POMDP).
//!
//! Part of the recurrent-policy epic (#262), completing the memory-hard
//! environment suite alongside the flickering-CartPole contrast (#298). This
//! example trains an
//! [`LstmBurnPolicy`](thrust_rl::policy::lstm::LstmBurnPolicy) with
//! [`RecurrentPPOTrainer`](thrust_rl::train::ppo::RecurrentPPOTrainer) on the
//! [`TMaze`](thrust_rl::env::TMaze) (Bakker 2001) and contrasts it against a
//! feedforward [`MlpBurnPolicy`](thrust_rl::policy::mlp::MlpBurnPolicy) baseline
//! on the *same* task.
//!
//! # Why the T-maze (and not just flickering)
//!
//! FlickeringCartPole (#298) is only *approximately* memory-hard: a reactive
//! controller partially compensates for i.i.d. blanked frames at CartPole's
//! control rate, so the feedforward baseline does not collapse to chance. The
//! T-maze removes that ambiguity. A cue is shown **only at step 0**; the agent
//! must recall it `N` steps later to turn the right way at the junction, where
//! the observation is identical for both cues. A memoryless policy is therefore
//! *provably* at chance (50%) — no amount of capacity helps, because the cue is
//! information-theoretically absent from the junction observation it acts on.
//!
//! This doubles as a **self-test on the environment**: if the feedforward
//! baseline beats chance by a meaningful margin, the env has an information leak
//! (a bug), not the policy a talent. The expected qualitative result is a clean
//! separation — LSTM well above 90% junction accuracy at `N = 10`, MLP pinned at
//! ~50%.
//!
//! # Corridor-length sweep
//!
//! The corridor length `N` is the memory horizon the policy must bridge. The
//! example sweeps `N ∈ {5, 10, 20}` (override with `TMAZE_SWEEP="5,10,20"`) and
//! reports junction accuracy for both arms at each `N`, documenting where the
//! LSTM's effective memory horizon begins to degrade.
//!
//! # Usage
//!
//! ```bash
//! # Full sweep, both arms:
//! cargo run --example recurrent_ppo_t_maze --features training --release
//!
//! # One corridor length, custom budget:
//! TMAZE_SWEEP=10 TOTAL_TIMESTEPS=200000 \
//!   cargo run --example recurrent_ppo_t_maze --features training --release
//! ```

use anyhow::Result;
use burn::{
    backend::Autodiff,
    nn::LstmState,
    optim::AdamConfig,
    tensor::{Int, Tensor, TensorData, activation},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::atomic::{AtomicU64, Ordering};
use thrust_rl::{
    buffer::rollout::RecurrentRolloutBuffer,
    env::{Environment, SpaceType, pool::EnvPool, t_maze::TMaze},
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
const DEFAULT_SWEEP: &[usize] = &[5, 10, 20];
/// LSTM learning rate (annealed linearly toward a small floor over the run).
const LSTM_LR: f64 = 1e-3;
/// Feedforward baseline learning rate.
const MLP_LR: f64 = 3e-4;
/// Discount. The reward is a single ±1 at the junction, so credit must
/// propagate back across the whole corridor; a high gamma keeps the step-0 cue
/// action in the loop even at `N = 20` (`0.99^20 ≈ 0.82`).
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;
const N_EPOCHS: usize = 4;
const ENVS_PER_MINIBATCH: usize = 4;
const SEED: u64 = 0;

/// Junction accuracy that counts as "solved" for the LSTM (acceptance criterion
/// 1 at `N = 10` is >90%).
const SOLVED_ACC: f32 = 0.90;

/// Outcome of one training arm: junction accuracy over the last ≤100 episodes.
///
/// `final_acc` is the accuracy at the end of training; `best_acc` is the peak of
/// the moving accuracy observed at any update — reported together so end-of-run
/// oscillation hides nothing.
#[derive(Clone, Copy)]
struct ArmResult {
    final_acc: f32,
    best_acc: f32,
}

/// Build an [`EnvPool`] of [`TMaze`]s with per-env seeded cue streams (env `i`
/// gets seed `base_seed*1000 + i`), reproducible yet decorrelated across the
/// pool. Both arms call this with the same `base_seed`, so they see the
/// identical cue stream — the only difference between the runs is the policy
/// class (memory vs. no memory).
fn make_pool(base_seed: u64, corridor_length: usize) -> EnvPool<TMaze> {
    let ctr = AtomicU64::new(base_seed.wrapping_mul(1000));
    EnvPool::new(
        || {
            let s = ctr.fetch_add(1, Ordering::Relaxed);
            TMaze::with_seed_and_corridor_length(s, corridor_length)
        },
        NUM_ENVS,
    )
}

/// Junction accuracy over the last ≤100 completed episodes. Each episode return
/// is exactly `+1` (correct turn) or `-1` (wrong), so accuracy is the fraction
/// of recent returns that are positive.
fn recent_accuracy(completed: &[f32]) -> f32 {
    if completed.is_empty() {
        return 0.0;
    }
    let n = completed.len();
    let recent = &completed[n.saturating_sub(100)..];
    let correct = recent.iter().filter(|&&r| r > 0.0).count();
    correct as f32 / recent.len() as f32
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let total_timesteps: usize = std::env::var("TOTAL_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TIMESTEPS);
    let sweep: Vec<usize> = std::env::var("TMAZE_SWEEP")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_SWEEP.to_vec());
    let mode = std::env::args().nth(1).unwrap_or_default();

    tracing::info!("Recurrent PPO on the T-maze (Bakker 2001, provably memory-hard POMDP)");
    tracing::info!("  backend         = NdArray<f32> + Autodiff (CPU)");
    tracing::info!("  num_envs        = {}", NUM_ENVS);
    tracing::info!("  num_steps       = {}", NUM_STEPS);
    tracing::info!("  hidden_dim      = {}", HIDDEN_DIM);
    tracing::info!("  corridor sweep  = {:?}", sweep);
    tracing::info!("  total_timesteps = {} per arm", total_timesteps);
    tracing::info!("------------------------------------------------------------");

    let mut rows: Vec<(usize, Option<ArmResult>, Option<ArmResult>)> = Vec::new();

    for &n in &sweep {
        tracing::info!("############### Corridor length N = {} ###############", n);
        let lstm = if mode != "baseline" {
            tracing::info!(">>> [N={}] Training LSTM (recurrent) policy", n);
            Some(run_lstm(total_timesteps, n)?)
        } else {
            None
        };
        let mlp = if mode != "lstm" {
            tracing::info!(">>> [N={}] Training feedforward (MLP) baseline", n);
            Some(run_feedforward(total_timesteps, n)?)
        } else {
            None
        };
        rows.push((n, lstm, mlp));
    }

    tracing::info!("============================================================");
    tracing::info!("T-maze junction accuracy (last ≤100 episodes) by corridor length:");
    tracing::info!("  N      LSTM(final/best)     MLP(final/best)    verdict");
    for (n, lstm, mlp) in &rows {
        let ls = lstm
            .map(|r| format!("{:.0}%/{:.0}%", r.final_acc * 100.0, r.best_acc * 100.0))
            .unwrap_or_else(|| "   -   ".into());
        let ms = mlp
            .map(|r| format!("{:.0}%/{:.0}%", r.final_acc * 100.0, r.best_acc * 100.0))
            .unwrap_or_else(|| "   -   ".into());
        let verdict = match (lstm, mlp) {
            (Some(l), Some(m)) => {
                let lstm_solved = l.best_acc > SOLVED_ACC;
                // A memoryless policy is provably at chance; flag any material
                // over-performance as a possible env information leak.
                let mlp_leak = m.best_acc > 0.65;
                format!(
                    "LSTM {} 90%; MLP {}",
                    if lstm_solved { "CLEARS" } else { "below" },
                    if mlp_leak {
                        "ABOVE CHANCE (investigate leak!)"
                    } else {
                        "~chance (as expected)"
                    }
                )
            }
            _ => String::new(),
        };
        tracing::info!("  N={:<4}  {:<18}  {:<16}  {}", n, ls, ms, verdict);
    }
    tracing::info!("------------------------------------------------------------");
    tracing::info!(
        "Expectation: LSTM accuracy stays high at small N and degrades as N grows"
    );
    tracing::info!(
        "past its effective memory horizon; MLP stays ~50% for all N (provable)."
    );

    Ok(())
}

/// Train the recurrent LSTM policy on a T-maze of the given corridor length.
fn run_lstm(total_timesteps: usize, corridor_length: usize) -> Result<ArmResult> {
    let start = std::time::Instant::now();
    let device = Default::default();

    let probe = TMaze::with_corridor_length(corridor_length);
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        SpaceType::Discrete(n) => n,
        _ => panic!("TMaze must be discrete"),
    };

    let mut env_pool = make_pool(SEED, corridor_length);

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
        .ent_coef(0.02)
        .max_grad_norm(0.5)
        .target_kl(1.0);

    let mut trainer = RecurrentPPOTrainer::new(ppo_config, policy, burn_opt, device)?;

    let num_updates = total_timesteps / (NUM_STEPS * NUM_ENVS);

    // The T-maze observation is already O(1) ([±1, 0/1, 0/1]); no normalization
    // is applied. Normalizing per-dimension would rescale the one-shot cue
    // channel (nonzero only at step 0) by its tiny across-stream std, which
    // would distort the memory signal. Raw observations are fed to both arms.
    let mut buffer = RecurrentRolloutBuffer::new(NUM_STEPS, NUM_ENVS, obs_dim, HIDDEN_DIM);
    let mut observations: Vec<Vec<f32>> = env_pool.reset();
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut episode_returns = [0.0_f32; NUM_ENVS];
    let mut completed: Vec<f32> = Vec::new();
    let mut lstm_state: Option<LstmState<Backend, 2>> = None;
    let mut last_h = vec![0.0_f32; NUM_ENVS * HIDDEN_DIM];
    let mut last_c = vec![0.0_f32; NUM_ENVS * HIDDEN_DIM];
    let mut total_env_steps = 0usize;
    let mut last_acc = 0.0_f32;
    let mut best_acc = 0.0_f32;

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
                    r.reward,
                    values_host[env],
                    log_probs[env],
                    r.terminated,
                    r.truncated,
                );
                episode_returns[env] += r.reward;
                observations[env] = r.observation.clone();

                if r.terminated || r.truncated {
                    completed.push(episode_returns[env]);
                    episode_returns[env] = 0.0;
                    trainer.increment_episodes(1);
                    for k in 0..HIDDEN_DIM {
                        h_host[env * HIDDEN_DIM + k] = 0.0;
                        c_host[env * HIDDEN_DIM + k] = 0.0;
                    }
                    observations[env] = env_pool.reset_env(env)?;
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

        last_acc = recent_accuracy(&completed);
        best_acc = best_acc.max(last_acc);

        if update % 10 == 0 || update == num_updates - 1 {
            tracing::info!(
                "  [lstm N={:>2}] update {:>3}/{}  env_steps={:>7}  episodes={:>5}  junction_acc(last≤100)={:5.1}%  entropy={:5.3}  explained_var={:5.3}",
                corridor_length,
                update + 1,
                num_updates,
                total_env_steps,
                trainer.total_episodes(),
                last_acc * 100.0,
                stats.entropy,
                stats.explained_var,
            );
        }
    }

    tracing::info!(
        "  [lstm N={}] done in {:.1}s — final acc {:.1}% (best {:.1}%)",
        corridor_length,
        start.elapsed().as_secs_f64(),
        last_acc * 100.0,
        best_acc * 100.0
    );
    Ok(ArmResult { final_acc: last_acc, best_acc })
}

/// Train the feedforward MLP baseline on the SAME T-maze. Expected to sit at
/// ~50% junction accuracy for every `N` — it cannot recall the step-0 cue.
fn run_feedforward(total_timesteps: usize, corridor_length: usize) -> Result<ArmResult> {
    let start = std::time::Instant::now();
    let device = Default::default();

    let probe = TMaze::with_corridor_length(corridor_length);
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        SpaceType::Discrete(n) => n,
        _ => panic!("TMaze must be discrete"),
    };

    let mut env_pool = make_pool(SEED, corridor_length);

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
        .ent_coef(0.02)
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

    let mut observations: Vec<Vec<f32>> = env_pool.reset();
    let mut episode_returns = [0.0_f32; NUM_ENVS];
    let mut completed: Vec<f32> = Vec::new();
    let mut total_env_steps = 0usize;
    let mut last_acc = 0.0_f32;
    let mut best_acc = 0.0_f32;

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
                buf_rewards.push(results[env_id].reward);
                let done = results[env_id].terminated || results[env_id].truncated;
                buf_dones.push(if done { 1.0 } else { 0.0 });

                episode_returns[env_id] += results[env_id].reward;
                observations[env_id] = results[env_id].observation.clone();
                if done {
                    completed.push(episode_returns[env_id]);
                    episode_returns[env_id] = 0.0;
                    trainer.increment_episodes(1);
                    observations[env_id] = env_pool.reset_env(env_id)?;
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

        last_acc = recent_accuracy(&completed);
        best_acc = best_acc.max(last_acc);

        if update % 10 == 0 || update == num_updates - 1 {
            tracing::info!(
                "  [mlp  N={:>2}] update {:>3}/{}  env_steps={:>7}  episodes={:>5}  junction_acc(last≤100)={:5.1}%  entropy={:5.3}",
                corridor_length,
                update + 1,
                num_updates,
                total_env_steps,
                trainer.total_episodes(),
                last_acc * 100.0,
                stats.entropy,
            );
        }
    }

    tracing::info!(
        "  [mlp  N={}] done in {:.1}s — final acc {:.1}% (best {:.1}%)",
        corridor_length,
        start.elapsed().as_secs_f64(),
        last_acc * 100.0,
        best_acc * 100.0
    );
    Ok(ArmResult { final_acc: last_acc, best_acc })
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
