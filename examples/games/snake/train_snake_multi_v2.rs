//! Multi-agent Snake PPO training (v2) on the Burn backend.
//!
//! Resurrected from the pre-Burn deletion (PR #98) per issue #101. The
//! v2 trainer runs N independent PPO learners (one per snake) sharing a
//! single multi-agent SnakeEnv. Each snake has its own
//! [`SnakeCnnBurnPolicy`](thrust_rl::policy::snake_cnn::SnakeCnnBurnPolicy)
//! + [`PPOTrainerBurn`](thrust_rl::train::ppo::PPOTrainerBurn) instance, gets a
//!   per-agent grid observation from the shared env, and is updated
//!   independently on its own rollout buffer.
//!
//! # Why "v2"?
//!
//! The original v1 trainer trained a single policy on summed
//! multi-agent rewards. v2 (this file) trains each snake's policy
//! independently — the canonical multi-agent training recipe.
//!
//! # Architecture
//!
//! - Multi-agent SnakeEnv with `NUM_AGENTS` snakes on a grid.
//! - Per-agent `SnakeCnnBurnPolicy` (3 conv + 2 fc); see the policy's module
//!   doc for the channel layout.
//! - Single env instance shared across agents; observations are per-agent
//!   through `SnakeEnv::get_grid_observation(agent_id)`.
//! - Default budget: ~80k env steps total across all agents.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_snake_multi_v2 --features training --release
//! ```
//!
//! Override the total step budget via the `TOTAL_TIMESTEPS` env var.

use anyhow::Result;
use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    optim::AdamConfig,
    tensor::{Int, Tensor, TensorData, activation},
};
use rand::Rng;
use thrust_rl::{
    env::games::snake::SnakeEnv,
    policy::snake_cnn::SnakeCnnBurnPolicy,
    train::{
        optimizer::BurnOptimizer,
        ppo::{PPOConfig, PPOTrainerBurn},
    },
};

type B = Autodiff<NdArray<f32>>;

const NUM_AGENTS: usize = 2;
const GRID_W: i32 = 10;
const GRID_H: i32 = 10;
const NUM_CHANNELS: usize = 5;
const NUM_STEPS: usize = 128;
const DEFAULT_TIMESTEPS: usize = 80_000;
const LEARNING_RATE: f64 = 3e-4;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let total_timesteps: usize = std::env::var("TOTAL_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TIMESTEPS);

    tracing::info!("Starting Multi-Agent Snake PPO Training v2 (Burn backend)");

    let training_start = std::time::Instant::now();

    let device: NdArrayDevice = Default::default();

    let grid_size = (GRID_W * GRID_H) as usize;
    let obs_len = NUM_CHANNELS * grid_size;

    tracing::info!("Environment: SnakeEnv multi-agent");
    tracing::info!("  num_agents      = {}", NUM_AGENTS);
    tracing::info!("  grid            = {}x{}", GRID_W, GRID_H);
    tracing::info!("  obs_channels    = {}", NUM_CHANNELS);
    tracing::info!("  num_steps       = {}", NUM_STEPS);
    tracing::info!("  total_timesteps = {}", total_timesteps);

    // Per-agent policies + optimizers.
    let mut trainers: Vec<PPOTrainerBurn<B, SnakeCnnBurnPolicy<B>, _>> =
        Vec::with_capacity(NUM_AGENTS);
    for _ in 0..NUM_AGENTS {
        let policy = SnakeCnnBurnPolicy::<B>::new(GRID_W as usize, NUM_CHANNELS, &device);
        let inner_opt = AdamConfig::new().init();
        let burn_opt: BurnOptimizer<B, SnakeCnnBurnPolicy<B>, _> =
            BurnOptimizer::new(inner_opt, LEARNING_RATE);
        let ppo_config = PPOConfig::new()
            .learning_rate(LEARNING_RATE)
            .n_epochs(4)
            .batch_size(64)
            .gamma(GAMMA as f64)
            .gae_lambda(GAE_LAMBDA as f64)
            .clip_range(0.2)
            .clip_range_vf(0.2)
            .vf_coef(0.5)
            .ent_coef(0.01)
            .max_grad_norm(0.5)
            .target_kl(1.0);
        trainers.push(PPOTrainerBurn::new(ppo_config, policy, burn_opt)?);
    }

    let mut env = SnakeEnv::new_multi(GRID_W, GRID_H, NUM_AGENTS);
    env.reset();

    let num_updates = total_timesteps / (NUM_STEPS * NUM_AGENTS);
    tracing::info!("Planned PPO updates: {}", num_updates);
    tracing::info!("------------------------------------------------------------");

    // Per-agent rollout buffers.
    let cap = NUM_STEPS;
    let mut buf_obs: Vec<Vec<f32>> =
        (0..NUM_AGENTS).map(|_| Vec::with_capacity(cap * obs_len)).collect();
    let mut buf_actions: Vec<Vec<i64>> = (0..NUM_AGENTS).map(|_| Vec::with_capacity(cap)).collect();
    let mut buf_log_probs: Vec<Vec<f32>> =
        (0..NUM_AGENTS).map(|_| Vec::with_capacity(cap)).collect();
    let mut buf_values: Vec<Vec<f32>> = (0..NUM_AGENTS).map(|_| Vec::with_capacity(cap)).collect();
    let mut buf_rewards: Vec<Vec<f32>> = (0..NUM_AGENTS).map(|_| Vec::with_capacity(cap)).collect();
    let mut buf_dones: Vec<f32> = Vec::with_capacity(cap);

    let mut total_env_steps: usize = 0;
    let mut episodes_completed: usize = 0;
    let mut per_episode_returns: Vec<Vec<f32>> = vec![Vec::new(); NUM_AGENTS];
    let mut current_returns = [0.0_f32; NUM_AGENTS];

    for update in 0..num_updates {
        for buf in buf_obs.iter_mut() {
            buf.clear();
        }
        for buf in buf_actions.iter_mut() {
            buf.clear();
        }
        for buf in buf_log_probs.iter_mut() {
            buf.clear();
        }
        for buf in buf_values.iter_mut() {
            buf.clear();
        }
        for buf in buf_rewards.iter_mut() {
            buf.clear();
        }
        buf_dones.clear();

        for _step in 0..NUM_STEPS {
            // Gather per-agent observations from the shared env.
            let observations: Vec<Vec<f32>> =
                (0..NUM_AGENTS).map(|i| env.get_grid_observation(i)).collect();

            // Each agent samples its own action via its policy.
            let mut actions = vec![0_i64; NUM_AGENTS];
            for i in 0..NUM_AGENTS {
                let obs_t = obs_to_tensor4(&observations[i], &device);
                let (logits, value) = trainers[i].policy().forward(obs_t);
                // Sample action from softmax(logits).
                let probs = activation::softmax(logits, 1);
                let probs_host: Vec<f32> = probs.into_data().to_vec().unwrap_or_default();
                let log_probs_t: Vec<f32> = probs_host.iter().map(|p| (p + 1e-9).ln()).collect();
                let value_host: f32 = value.into_data().to_vec::<f32>().unwrap_or_default()[0];

                let mut rng = rand::rng();
                let u: f32 = rng.random();
                let mut cum = 0.0;
                let mut chosen: i64 = (probs_host.len() - 1) as i64;
                for (j, &p) in probs_host.iter().enumerate() {
                    cum += p;
                    if u < cum {
                        chosen = j as i64;
                        break;
                    }
                }

                buf_obs[i].extend_from_slice(&observations[i]);
                buf_actions[i].push(chosen);
                buf_log_probs[i].push(log_probs_t[chosen as usize]);
                buf_values[i].push(value_host);
                actions[i] = chosen;
            }

            // Shared env step.
            let (rewards, terminated, truncated) = env.step_multi_agents(&actions);
            let done = terminated || truncated;

            for i in 0..NUM_AGENTS {
                buf_rewards[i].push(rewards[i]);
                current_returns[i] += rewards[i];
            }
            buf_dones.push(if done { 1.0 } else { 0.0 });

            total_env_steps += NUM_AGENTS;
            if done {
                episodes_completed += 1;
                for i in 0..NUM_AGENTS {
                    per_episode_returns[i].push(current_returns[i]);
                    current_returns[i] = 0.0;
                    trainers[i].increment_episodes(1);
                }
                env.reset();
            }
        }

        // --- Update each agent's policy ----------------------------
        for i in 0..NUM_AGENTS {
            // Bootstrap last value.
            let last_obs = env.get_grid_observation(i);
            let last_obs_t = obs_to_tensor4(&last_obs, &device);
            let (_, last_v) = trainers[i].policy().forward(last_obs_t);
            let last_v_host: f32 = last_v.into_data().to_vec::<f32>().unwrap_or_default()[0];

            let (advantages, returns) = compute_gae_single(
                &buf_rewards[i],
                &buf_values[i],
                &buf_dones,
                last_v_host,
                GAMMA,
                GAE_LAMBDA,
            );

            let batch = NUM_STEPS;
            let obs_b: Tensor<B, 2> =
                Tensor::from_data(TensorData::new(buf_obs[i].clone(), [batch, obs_len]), &device);
            let actions_b: Tensor<B, 1, Int> =
                Tensor::from_data(TensorData::new(buf_actions[i].clone(), [batch]), &device);
            let old_log_probs_b: Tensor<B, 1> =
                Tensor::from_data(TensorData::new(buf_log_probs[i].clone(), [batch]), &device);
            let old_values_b: Tensor<B, 1> =
                Tensor::from_data(TensorData::new(buf_values[i].clone(), [batch]), &device);
            let advantages_b: Tensor<B, 1> =
                Tensor::from_data(TensorData::new(advantages, [batch]), &device);
            let returns_b: Tensor<B, 1> =
                Tensor::from_data(TensorData::new(returns, [batch]), &device);

            // Closure: reshape flat obs back into [batch, C, H, W] and run forward.
            let evaluate_fn = |p: &SnakeCnnBurnPolicy<B>,
                               o_flat: Tensor<B, 2>,
                               a: Tensor<B, 1, Int>|
             -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
                let b = o_flat.dims()[0];
                let o4: Tensor<B, 4> =
                    o_flat.reshape([b, NUM_CHANNELS, GRID_H as usize, GRID_W as usize]);
                let (logits, value) = p.forward(o4);
                let log_probs_all = activation::log_softmax(logits.clone(), 1);
                let probs_all = log_probs_all.clone().exp();
                let entropy = -(probs_all * log_probs_all.clone()).sum_dim(1).squeeze_dim::<1>(1);
                let action_log_probs =
                    log_probs_all.gather(1, a.unsqueeze_dim::<2>(1)).squeeze_dim::<1>(1);
                let value_1d: Tensor<B, 1> = value.squeeze_dim::<1>(1);
                (action_log_probs, entropy, value_1d)
            };

            let _stats = trainers[i].train_step(
                obs_b,
                actions_b,
                old_log_probs_b,
                old_values_b,
                advantages_b,
                returns_b,
                evaluate_fn,
            )?;
        }

        let recent_avg_per_agent: Vec<f32> = (0..NUM_AGENTS)
            .map(|i| {
                if per_episode_returns[i].is_empty() {
                    0.0
                } else {
                    let n = per_episode_returns[i].len();
                    let slice = &per_episode_returns[i][n.saturating_sub(20)..];
                    slice.iter().copied().sum::<f32>() / slice.len() as f32
                }
            })
            .collect();

        if update % 2 == 0 || update == num_updates - 1 {
            tracing::info!(
                "update {:>3}/{}  env_steps={:>7}  episodes={:>4}  avg_returns(last≤20)={:?}",
                update + 1,
                num_updates,
                total_env_steps,
                episodes_completed,
                recent_avg_per_agent.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>(),
            );
        }
    }

    let final_avg_per_agent: Vec<f32> = (0..NUM_AGENTS)
        .map(|i| {
            if per_episode_returns[i].is_empty() {
                0.0
            } else {
                let n = per_episode_returns[i].len();
                let slice = &per_episode_returns[i][n.saturating_sub(20)..];
                slice.iter().copied().sum::<f32>() / slice.len() as f32
            }
        })
        .collect();

    tracing::info!("------------------------------------------------------------");
    tracing::info!(
        "Training complete.  episodes={}  env_steps={}  final_avg_returns(last≤20)={:?}  time={:.1}s",
        episodes_completed,
        total_env_steps,
        final_avg_per_agent.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>(),
        training_start.elapsed().as_secs_f64(),
    );

    Ok(())
}

/// Convert a flat observation buffer `[C*H*W]` to a `[1, C, H, W]` tensor.
fn obs_to_tensor4(obs: &[f32], device: &NdArrayDevice) -> Tensor<B, 4> {
    Tensor::<B, 4>::from_data(
        TensorData::new(obs.to_vec(), [1, NUM_CHANNELS, GRID_H as usize, GRID_W as usize]),
        device,
    )
}

/// Single-trajectory GAE on host-side rollouts.
fn compute_gae_single(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    last_value: f32,
    gamma: f32,
    gae_lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let t = rewards.len();
    let mut advantages = vec![0.0_f32; t];
    let mut returns = vec![0.0_f32; t];
    let mut gae = 0.0_f32;
    for i in (0..t).rev() {
        let next_v = if i == t - 1 {
            last_value
        } else {
            values[i + 1]
        };
        let next_nonterminal = 1.0 - dones[i];
        let delta = rewards[i] + gamma * next_v * next_nonterminal - values[i];
        gae = delta + gamma * gae_lambda * next_nonterminal * gae;
        advantages[i] = gae;
        returns[i] = advantages[i] + values[i];
    }
    (advantages, returns)
}
