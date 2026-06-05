//! Train DQN on CartPole-v1.
//!
//! This example demonstrates Thrust's DQN scaffolding end-to-end: a
//! replay-buffer + target-network + ε-greedy training loop on the
//! classic discrete-action control benchmark.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_cartpole_dqn --features training --release
//! ```
//!
//! Override the total step budget via the `TOTAL_TIMESTEPS` env var:
//!
//! ```bash
//! TOTAL_TIMESTEPS=20000 cargo run --example train_cartpole_dqn --features training --release
//! ```
//!
//! Enable Prioritized Experience Replay (Schaul et al., 2015) via
//! `PER=1`:
//!
//! ```bash
//! PER=1 cargo run --example train_cartpole_dqn --features training --release
//! ```
//!
//! Expected behavior: average return over the last 100 episodes climbs
//! past 475 (the CartPole "solved" threshold) within ~50k env steps.

use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use thrust_rl::{
    env::{Environment, cartpole::CartPole},
    train::dqn::{DQNConfig, DQNTrainer},
};

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    tracing::info!("🚀 Starting CartPole DQN Training");

    // Total env steps; default to 60k which is comfortably above the
    // canonical 50k budget for vanilla DQN on CartPole.
    let total_timesteps: usize = std::env::var("TOTAL_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60_000);

    // Prioritized Experience Replay toggle. Set `PER=1` to flip on the
    // sum-tree backed replay buffer with IS-weighted Smooth-L1 loss.
    let use_per: bool = std::env::var("PER")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .map(|n| n != 0)
        .unwrap_or(false);

    // Inspect the environment for dims.
    let probe = CartPole::new();
    let obs_dim = probe.observation_space().shape[0] as i64;
    let n_actions = match probe.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n as i64,
        _ => panic!("CartPole must have a discrete action space"),
    };
    tracing::info!("Environment: CartPole-v1  obs_dim={}  n_actions={}", obs_dim, n_actions);

    // DQN hyperparameters — curator-locked defaults from issue #48,
    // augmented with the issue #58 fixes (Double-DQN target — applied
    // unconditionally by the trainer — plus Polyak soft target updates
    // toggled via `.soft_update_tau(0.005)`).
    //
    // With Double-DQN + soft updates the avg-100 return is expected to
    // cross 475 within ~50k env steps on this seed.
    let mut config = DQNConfig::new()
        .learning_rate(1e-3)
        .batch_size(64)
        .buffer_capacity(50_000)
        .min_buffer_size(1_000)
        .target_update_interval(500) // ignored in soft mode, kept for documentation
        .gamma(0.99)
        .epsilon_start(1.0)
        .epsilon_end(0.05)
        .epsilon_decay_steps(10_000)
        .max_grad_norm(10.0)
        .soft_update_tau(0.005);

    if use_per {
        // Schaul's defaults: α = 0.6, β annealed 0.4 → 1.0 over the
        // training budget (we let it follow ε_decay_steps unless the
        // caller wants something else).
        config = config
            .prioritized_replay(true)
            .per_alpha(0.6)
            .per_beta_start(0.4)
            .per_beta_end(1.0)
            .per_beta_steps(total_timesteps)
            .per_epsilon(1e-6);
        tracing::info!(
            "🎯 Prioritized Experience Replay ENABLED  α={:.2}  β: {:.2}→{:.2} over {} steps  ε={:.0e}",
            config.per_alpha,
            config.per_beta_start,
            config.per_beta_end,
            config.per_beta_steps,
            config.per_epsilon,
        );
    } else {
        tracing::info!("Uniform replay (set PER=1 to enable Prioritized Experience Replay)");
    }

    let mut trainer = DQNTrainer::new(config, obs_dim, n_actions, 64)?;
    tracing::info!("Trainer on device: {:?}", trainer.device());

    // Single-env rollout loop. DQN is off-policy, so we keep it simple:
    // step the env, push to the buffer, periodically train.
    let mut env = CartPole::new();
    env.reset();
    let mut obs = env.get_observation();

    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut episode_return: f32 = 0.0;
    let mut episode_returns: Vec<f32> = Vec::new();
    let mut last_log_step = 0usize;

    let log_interval = 1_000usize;

    while trainer.total_env_steps() < total_timesteps {
        let action = trainer.select_action(&obs, &mut rng);
        let result = env.step(action);
        let next_obs = result.observation.clone();
        let done = result.terminated || result.truncated;
        trainer.push_transition(&obs, action, result.reward, &next_obs, done);

        episode_return += result.reward;
        obs = next_obs;

        trainer.increment_env_step();
        let _synced = trainer.maybe_sync_target()?;

        if done {
            episode_returns.push(episode_return);
            trainer.increment_episodes(1);
            episode_return = 0.0;
            env.reset();
            obs = env.get_observation();
        }

        // One gradient update per env step (the standard DQN cadence).
        let _stats = trainer.train_step(&mut rng)?;

        // Periodic logging.
        let step = trainer.total_env_steps();
        if step.saturating_sub(last_log_step) >= log_interval {
            last_log_step = step;
            let recent_avg = if episode_returns.len() >= 10 {
                let n = episode_returns.len();
                let slice = &episode_returns[n.saturating_sub(100)..];
                slice.iter().copied().sum::<f32>() / slice.len() as f32
            } else {
                0.0
            };
            let last_stats = trainer.train_step(&mut rng)?;
            let td_loss = last_stats.map(|s| s.td_loss).unwrap_or(f64::NAN);
            let beta_str = match last_stats.and_then(|s| s.beta) {
                Some(b) => format!("  β={:.3}", b),
                None => String::new(),
            };
            tracing::info!(
                "step={:>6}  episodes={:>4}  avg(last≤100)={:7.2}  ε={:.3}  buf={:>6}  loss={:.4}{}",
                step,
                trainer.total_episodes(),
                recent_avg,
                trainer.last_epsilon(),
                trainer.buffer_len(),
                td_loss,
                beta_str,
            );
        }
    }

    let final_avg = if episode_returns.len() >= 1 {
        let n = episode_returns.len();
        let slice = &episode_returns[n.saturating_sub(100)..];
        slice.iter().copied().sum::<f32>() / slice.len() as f32
    } else {
        0.0
    };
    tracing::info!(
        "✅ Training complete.  episodes={}  env_steps={}  train_steps={}  avg(last≤100)={:.2}",
        trainer.total_episodes(),
        trainer.total_env_steps(),
        trainer.total_train_steps(),
        final_avg,
    );

    let save_path = "cartpole_dqn_model.pt";
    trainer.online_mut().save(save_path)?;
    tracing::info!("💾 Online Q-network saved to {}", save_path);

    Ok(())
}
