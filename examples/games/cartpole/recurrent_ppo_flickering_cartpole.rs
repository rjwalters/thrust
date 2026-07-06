//! Recurrent PPO on flickering CartPole (a POMDP by construction).
//!
//! Phase 3 of the recurrent-policy epic (#262), re-scoped per issue #287. This
//! end-to-end example is the learning-signal payoff for the whole recurrent
//! stack: it trains an
//! [`LstmBurnPolicy`](thrust_rl::policy::lstm::LstmBurnPolicy) with
//! [`RecurrentPPOTrainer`](thrust_rl::train::ppo::RecurrentPPOTrainer) on
//! [`FlickeringCartPole`](thrust_rl::env::FlickeringCartPole) — CartPole whose
//! 4-D observation is blanked to zeros with probability `p` (default 0.5) — and
//! contrasts it against a feedforward
//! [`MlpBurnPolicy`](thrust_rl::policy::mlp::MlpBurnPolicy) baseline on the
//! *same* flickering observation stream.
//!
//! # Why flickering (and not velocity-masking)
//!
//! An earlier version of this example masked CartPole's two velocity
//! coordinates. A real 500k-step run **disproved** that as a POMDP: a
//! memoryless reactive controller on `[x, theta]` solved it and actually *beat*
//! the LSTM (MLP mean return ~324 vs LSTM ~222). Masking velocities does not
//! make memory load-bearing, because a reactive angle/position feedback loop
//! balances CartPole on its own. `MaskedCartPole` is retained in the crate as a
//! documented negative result.
//!
//! Flickering closes that loophole. This is the canonical Atari-POMDP protocol
//! from Hausknecht & Stone (2015): with probability `p` the entire observed
//! frame is blanked to zeros. A feedforward policy cannot act on a zeroed frame
//! — it has no state to fall back on. A recurrent policy carries its hidden
//! state across the blanked gap and integrates the intermittent stream over
//! time. Memory is load-bearing **by construction**.
//!
//! # Usage
//!
//! ```bash
//! # Run both (LSTM, then feedforward baseline) and print the contrast:
//! cargo run --example recurrent_ppo_flickering_cartpole --features training --release
//!
//! # Run just one arm:
//! cargo run --example recurrent_ppo_flickering_cartpole --features training --release -- lstm
//! cargo run --example recurrent_ppo_flickering_cartpole --features training --release -- baseline
//! ```
//!
//! Override the per-arm step budget with `TOTAL_TIMESTEPS` (default 500k) and
//! the flicker probability with `FLICKER_PROB` (default 0.5).

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
const DEFAULT_TIMESTEPS: usize = 500_000;
const DEFAULT_FLICKER_PROB: f64 = 0.5;
/// LSTM base learning rate; annealed linearly toward a small floor over the
/// run (the POMDP overshoots late, so annealing stabilizes the final policy
/// above the solved bar).
const LSTM_LR: f64 = 1.5e-3;
/// Feedforward baseline learning rate (standard CartPole PPO value).
const MLP_LR: f64 = 3e-4;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;
/// PPO epochs per rollout for the LSTM arm. Kept low (4): reusing each
/// recurrent rollout many times amplifies off-policy drift through the hidden
/// state and drives the late-stage oscillation seen at higher epoch counts.
const N_EPOCHS: usize = 4;
const ENVS_PER_MINIBATCH: usize = 4;
const SEED: u64 = 0;

/// Train-time reward scale (both arms, symmetric). CartPole's +1/step reward
/// gives discounted returns of magnitude ~40-60; against targets that large the
/// PPO2 clipped value loss (`clip_range_vf = 0.2`) freezes the critic (it can
/// move at most ~0.2 per rollout iteration — measured explained_var ≈ 0 over an
/// entire 500k run), while *disabling* the clip lets the huge squared-error
/// gradients blow out the shared LSTM trunk. Scaling rewards to O(1) returns
/// fixes both at the source. Reported episode returns remain in RAW units.
const REWARD_SCALE: f32 = 0.02;

/// CartPole-v1 "solved" bar; the LSTM must clear this on the flickering POMDP.
const SOLVED_THRESHOLD: f32 = 195.0;

/// Outcome of one training arm.
///
/// `final_mean` is the mean return over the last ≤100 episodes at the end of
/// training (noisy at episode granularity). `best_mean` is the peak of that
/// moving average observed at any update during the run — the honest answer to
/// "did the policy *reach* the solved bar within budget?", reported alongside
/// the final value so nothing is hidden by end-of-run oscillation.
#[derive(Clone, Copy)]
struct ArmResult {
    final_mean: f32,
    best_mean: f32,
}

/// Build an [`EnvPool`] of [`FlickeringCartPole`]s with per-env seeded flicker
/// streams (env `i` gets seed `base_seed*1000 + i`), so the schedule is
/// reproducible yet decorrelated across the pool. Both training arms call this
/// with the same `base_seed`, so they see the *identical* flicker stream — the
/// only difference between the runs is the policy class (memory vs. no memory).
fn make_pool(base_seed: u64, flicker_prob: f64) -> EnvPool<FlickeringCartPole> {
    let ctr = AtomicU64::new(base_seed.wrapping_mul(1000));
    EnvPool::new(
        || {
            let s = ctr.fetch_add(1, Ordering::Relaxed);
            FlickeringCartPole::with_seed_and_probability(s, flicker_prob)
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

    let mode = std::env::args().nth(1).unwrap_or_default();

    tracing::info!("Recurrent PPO on FlickeringCartPole (Hausknecht & Stone POMDP)");
    tracing::info!("  backend         = NdArray<f32> + Autodiff (CPU)");
    tracing::info!("  num_envs        = {}", NUM_ENVS);
    tracing::info!("  num_steps       = {}", NUM_STEPS);
    tracing::info!("  hidden_dim      = {}", HIDDEN_DIM);
    tracing::info!("  flicker_prob    = {}", flicker_prob);
    tracing::info!("  total_timesteps = {} per arm", total_timesteps);
    tracing::info!("------------------------------------------------------------");

    let mut lstm_return: Option<ArmResult> = None;
    let mut ff_return: Option<ArmResult> = None;

    if mode != "baseline" {
        tracing::info!(">>> Training LSTM (recurrent) policy");
        lstm_return = Some(run_lstm(total_timesteps, flicker_prob)?);
    }
    if mode != "lstm" {
        tracing::info!(">>> Training feedforward (MLP) baseline");
        ff_return = Some(run_feedforward(total_timesteps, flicker_prob)?);
    }

    tracing::info!("============================================================");
    tracing::info!(
        "Results on FlickeringCartPole (p={}, mean return of last ≤100 episodes):",
        flicker_prob
    );
    if let Some(r) = lstm_return {
        tracing::info!(
            "  LSTM (recurrent)     : final {:.1}  best {:.1}",
            r.final_mean,
            r.best_mean
        );
    }
    if let Some(r) = ff_return {
        tracing::info!(
            "  MLP  (feedforward)   : final {:.1}  best {:.1}",
            r.final_mean,
            r.best_mean
        );
    }
    if let (Some(l), Some(f)) = (lstm_return, ff_return) {
        tracing::info!("------------------------------------------------------------");
        // Acceptance ("reaches >195 within budget") is answered by the peak of
        // the moving average; the final value is reported too for honesty.
        tracing::info!(
            "  LSTM {} solved bar ({}) within budget (peak {:.1}); final {:.1}.",
            if l.best_mean > SOLVED_THRESHOLD {
                "CLEARED"
            } else {
                "MISSED"
            },
            SOLVED_THRESHOLD,
            l.best_mean,
            l.final_mean,
        );
        // Compare the peaks: the memoryless policy's ceiling vs the LSTM's.
        let gap = l.best_mean - f.best_mean;
        let rel = if l.best_mean > 0.0 {
            100.0 * gap / l.best_mean
        } else {
            0.0
        };
        tracing::info!(
            "  Memory advantage (peak): LSTM {:.1} vs feedforward {:.1} — gap {:.1} ({:.0}% of LSTM).",
            l.best_mean,
            f.best_mean,
            gap,
            rel,
        );
    }

    Ok(())
}

/// Train the recurrent LSTM policy on FlickeringCartPole. Returns the final and
/// peak mean return over the last ≤100 completed episodes.
fn run_lstm(total_timesteps: usize, flicker_prob: f64) -> Result<ArmResult> {
    let start = std::time::Instant::now();
    let device = Default::default();

    let probe = FlickeringCartPole::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        SpaceType::Discrete(n) => n,
        _ => panic!("FlickeringCartPole must be discrete"),
    };

    let mut env_pool = make_pool(SEED, flicker_prob);

    let policy_config =
        LstmBurnConfig { hidden_dim: HIDDEN_DIM, ..Default::default() }.with_seed(SEED);
    let policy =
        LstmBurnPolicy::<Backend>::with_config(obs_dim, action_dim, policy_config, &device);

    let lr = LSTM_LR;
    // Gradient clipping comes from `PPOConfig::max_grad_norm` below (issue
    // #299): the trainer applies a global gradient-norm clip before every
    // optimizer step. Load-bearing for the recurrent trunk, where value-loss
    // spikes otherwise wipe out the policy features.
    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<Backend, LstmBurnPolicy<Backend>, _> =
        BurnOptimizer::new(inner_opt, lr);

    let ppo_config = PPOConfig::new()
        .learning_rate(lr)
        .n_epochs(N_EPOCHS)
        .gamma(GAMMA as f64)
        .gae_lambda(GAE_LAMBDA as f64)
        .clip_range(0.2)
        // With rewards scaled to O(1) returns (REWARD_SCALE) the standard 0.2
        // value clip is meaningful relative to its targets and the critic can
        // actually track them.
        .clip_range_vf(0.2)
        .vf_coef(0.5)
        .ent_coef(0.01)
        .max_grad_norm(0.5)
        // Disable KL early-stop; the collapse guard handles pathology.
        .target_kl(1.0);

    let mut trainer = RecurrentPPOTrainer::new(ppo_config, policy, burn_opt, device)?;

    let num_updates = total_timesteps / (NUM_STEPS * NUM_ENVS);
    tracing::info!("  planned PPO updates: {}", num_updates);

    // Standardize the VISIBLE observations to ~unit scale. CartPole's raw
    // observations are tiny early on (perturbations ~0.05); the LSTM's tanh
    // gates squash such small inputs to near-zero features, starving both heads
    // of gradient. A running per-dimension mean/std normalizer keeps every
    // visible coordinate at O(1). Blanked (flickered) frames — all-zeros — are
    // passed through as zeros and excluded from the running statistics, so the
    // "no observation" signal stays a clean zero for both policies.
    let mut norm = ObsNormalizer::new(obs_dim);

    let mut buffer = RecurrentRolloutBuffer::new(NUM_STEPS, NUM_ENVS, obs_dim, HIDDEN_DIM);
    let raw0 = env_pool.reset();
    let mut observations: Vec<Vec<f32>> = raw0.iter().map(|o| norm.normalize(o)).collect();
    let mut rng = StdRng::seed_from_u64(SEED);

    // Per-env running episode return (CartPole reward = +1/step, so
    // return == episode length). Recurrent state threaded across steps.
    let mut episode_returns = [0.0_f32; NUM_ENVS];
    let mut completed: Vec<f32> = Vec::new();
    // Strategy A: the training forward (`evaluate_sequences`) always
    // reconstructs the hidden state from zeros at the start of each rollout
    // window. The behavior policy must match, or `old_log_probs` (warm state)
    // become inconsistent with the recomputed `log_probs` (cold state) and the
    // PPO ratio is corrupted. So the collection state starts at zeros (`None`)
    // every window — the accepted BPTT truncation. In-window per-episode resets
    // still happen below. Reset to `None` at the end of each update.
    let mut lstm_state: Option<LstmState<Backend, 2>> = None;
    // Host copy of the recurrent state exiting the last collected step
    // (post per-env reset), reused for warm-starting the next iteration.
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

            // Pull the recurrent state exiting this step to the host so we can
            // zero the rows of envs whose episode just ended (fresh episode ⇒
            // zeroed memory).
            let mut h_host: Vec<f32> = new_state.hidden.into_data().to_vec().unwrap();
            let mut c_host: Vec<f32> = new_state.cell.into_data().to_vec().unwrap();

            for env in 0..NUM_ENVS {
                let r = &results[env];
                buffer.add(
                    step,
                    env,
                    &observations[env],
                    actions[env],
                    // Train on scaled rewards (see REWARD_SCALE); reporting
                    // below stays in raw units.
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

        // Bootstrap value for the final observations under the carried state.
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

        // Linear LR annealing (standard PPO): decay from `lr` toward a small
        // floor over the run. The POMDP learns fast early but overshoots and
        // oscillates late once the value function catches up; annealing the
        // step size stabilizes the final policy above the solved bar.
        let frac = 1.0 - (update as f64) / (num_updates.max(1) as f64);
        trainer.set_learning_rate(lr * frac.max(0.05));

        let stats =
            trainer.train_step(&buffer, ENVS_PER_MINIBATCH, |p, obs_seq, actions, starts| {
                p.evaluate_sequences(obs_seq, actions, None, starts)
            })?;

        // Seed step-0 state + carry flag for the next iteration.
        let final_hidden: Vec<Vec<f32>> = (0..NUM_ENVS)
            .map(|e| last_h[e * HIDDEN_DIM..(e + 1) * HIDDEN_DIM].to_vec())
            .collect();
        let final_cell: Vec<Vec<f32>> = (0..NUM_ENVS)
            .map(|e| last_c[e * HIDDEN_DIM..(e + 1) * HIDDEN_DIM].to_vec())
            .collect();
        buffer.seed_warm_start(NUM_STEPS - 1, &final_hidden, &final_cell);

        // BPTT truncation at the window boundary: the next window's collection
        // starts from a zeroed recurrent state to stay consistent with the
        // training forward (which recomputes from zeros).
        lstm_state = None;

        if !completed.is_empty() {
            let n = completed.len();
            let recent = &completed[n.saturating_sub(100)..];
            last_mean = recent.iter().sum::<f32>() / recent.len() as f32;
            best_mean = best_mean.max(last_mean);
        }

        if update % 10 == 0 || update == num_updates - 1 {
            tracing::info!(
                "  [lstm] update {:>3}/{}  env_steps={:>7}  episodes={:>5}  mean_return(last≤100)={:6.1}  entropy={:5.3}  explained_var={:5.3}",
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
        "  [lstm] done in {:.1}s — final mean return {:.1} (best {:.1})",
        start.elapsed().as_secs_f64(),
        last_mean,
        best_mean
    );
    Ok(ArmResult { final_mean: last_mean, best_mean })
}

/// Train the feedforward MLP baseline on the SAME flickering observation
/// stream. Returns the final and peak mean return over the last ≤100 completed
/// episodes. Expected to plateau below the LSTM — it cannot act on a blanked
/// frame and has no memory to bridge the gaps.
fn run_feedforward(total_timesteps: usize, flicker_prob: f64) -> Result<ArmResult> {
    let start = std::time::Instant::now();
    let device = Default::default();

    let probe = FlickeringCartPole::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        SpaceType::Discrete(n) => n,
        _ => panic!("FlickeringCartPole must be discrete"),
    };

    let mut env_pool = make_pool(SEED, flicker_prob);

    let policy_config = MlpBurnConfig {
        num_layers: 2,
        hidden_dim: 128,
        use_orthogonal_init: true,
        activation: BurnActivation::ReLU,
        seed: Some(SEED),
    };
    let policy = MlpBurnPolicy::<Backend>::with_config(obs_dim, action_dim, policy_config, &device);

    // Gradient clipping comes from `PPOConfig::max_grad_norm` below,
    // mirroring the LSTM arm (see run_lstm).
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

    // Same running normalizer as the LSTM arm so the ONLY difference between
    // the two runs is the policy class (memory vs. no memory). Blanked frames
    // pass through as zeros (see ObsNormalizer).
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
                // Train on scaled rewards (see REWARD_SCALE); reporting stays
                // in raw units.
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
                "  [mlp ] update {:>3}/{}  env_steps={:>7}  episodes={:>4}  mean_return(last≤100)={:6.1}  entropy={:5.3}",
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
        "  [mlp ] done in {:.1}s — final mean return {:.1} (best {:.1})",
        start.elapsed().as_secs_f64(),
        last_mean,
        best_mean
    );
    Ok(ArmResult { final_mean: last_mean, best_mean })
}

/// Running per-dimension observation standardizer (Welford mean/variance) with
/// flicker-aware pass-through.
///
/// Normalizes each *visible* observation coordinate to zero mean / unit
/// variance using online statistics accumulated over every visible observation
/// seen so far, then clips to `[-10, 10]`. This is the standard
/// `VecNormalize`-style wrapper. It is load-bearing for the recurrent policy:
/// CartPole's raw observations are tiny (perturbations ~0.05), and an LSTM's
/// tanh gates squash such small inputs to near-zero features. Standardizing to
/// O(1) restores a usable gradient signal through the recurrent trunk.
///
/// A **blanked (flickered) frame is all-zeros** — a state CartPole physics
/// never produces. Such frames are passed through unchanged (all zeros) and
/// **excluded** from the running statistics: the "no observation" signal must
/// stay a clean, constant zero for both policies, and folding zeros into the
/// mean/std would (a) pollute the statistics and (b) turn the blank into a
/// recognizable non-zero constant, both of which would muddy the memory-vs-no-
/// memory contrast.
struct ObsNormalizer {
    mean: Vec<f64>,
    /// Sum of squared deviations from the running mean (Welford's M2).
    m2: Vec<f64>,
    count: f64,
}

impl ObsNormalizer {
    fn new(dim: usize) -> Self {
        Self { mean: vec![0.0; dim], m2: vec![0.0; dim], count: 0.0 }
    }

    /// Update the running statistics with a *visible* `obs` and return the
    /// standardized, clipped observation. A blanked frame (all-zeros) is
    /// returned as-is and does not update the statistics.
    fn normalize(&mut self, obs: &[f32]) -> Vec<f32> {
        // Flickered frames are exactly all-zero; pass them through untouched.
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
///
/// `logits` is `[N_env, action_dim]`, `values` is `[N_env]`. Returns
/// `(actions, action_log_probs, values)` as host `Vec`s. The rollout does not
/// need gradient flow through the sampled action (only the eventual
/// `evaluate_sequences` call matters for the surrogate), so the draw is done
/// on the host.
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

/// Per-env host-side GAE for the feedforward baseline (step-major `[T * N]`
/// layout). Mirrors `train_cartpole_modern`.
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
