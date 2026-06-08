//! Train PPO on Pong with **self-play**: the right paddle is controlled by
//! frozen policy snapshots periodically copied from the live agent. This is
//! the additive self-play counterpart to [`train_pong`], which plays against
//! a hand-coded rule-based opponent.
//!
//! # How it works
//!
//! - A single live [`MlpPolicy`] is trained with PPO on the left paddle.
//! - A snapshot pool (`Vec<MlpPolicy>`, FIFO, capped at `POOL_MAX`) holds
//!   recent frozen copies of the live policy. Every `SNAPSHOT_INTERVAL`
//!   updates, the live policy is cloned (via save/load through a temp file) and
//!   pushed into the pool; the oldest snapshot is evicted.
//! - The pool is seeded at training start with one clone of the randomly
//!   initialized policy so update 0 has a valid opponent.
//! - For each env, a snapshot is sampled uniformly from the pool. That snapshot
//!   plays the right paddle for the entire episode. When the episode ends (a
//!   score-out or truncation), a new snapshot is sampled.
//! - The right-paddle observation is the left paddle's observation passed
//!   through [`mirror_observation`], so a single policy network can play either
//!   side.
//! - Win/loss is tracked per episode (score delta sign), and the win rate over
//!   the last `WIN_RATE_WINDOW` episodes is logged every 20 updates.
//!
//! Only the left paddle's transitions enter the rollout buffer — the right
//! paddle is frozen and receives no gradient.
//!
//! # Usage
//!
//! ```bash
//! # Local (CPU)
//! cargo run --example train_pong_self_play --release --features training
//!
//! # Smoke test with a small step budget via env var
//! TOTAL_TIMESTEPS=500000 cargo run --example train_pong_self_play --release --features training
//!
//! # Remote GPU
//! ssh alc-2 "cd ~/thrust && cargo run --example train_pong_self_play --release --features training"
//! ```

use std::collections::{HashMap, VecDeque};

use anyhow::Result;
use rand::{Rng, SeedableRng, rngs::StdRng};
use thrust_rl::{
    buffer::rollout::RolloutBuffer,
    env::{
        Environment,
        pong::{Pong, mirror_observation},
    },
    policy::{inference::TrainingMetadata, mlp::MlpPolicy},
    train::ppo::{PPOConfig, PPOTrainer},
};

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

const NUM_ENVS: usize = 32;
const NUM_STEPS: usize = 128;
/// Default total environment steps. Overridable at runtime via the
/// `TOTAL_TIMESTEPS` env var so a short smoke run can be triggered without
/// recompiling.
const TOTAL_TIMESTEPS_DEFAULT: usize = 20_000_000;
const LEARNING_RATE: f64 = 0.0003;
const HIDDEN_DIM: i64 = 128;
const CHECKPOINT_INTERVAL_SECS: u64 = 300;

/// Maximum number of frozen snapshots in the opponent pool. FIFO eviction:
/// when the pool is full, the oldest snapshot is dropped before pushing.
const POOL_MAX: usize = 8;
/// Push a new snapshot into the pool every `SNAPSHOT_INTERVAL` PPO updates.
/// At `NUM_ENVS * NUM_STEPS = 4096` steps/update this is ~102K env steps per
/// snapshot — frequent enough for a diverse pool, infrequent enough to keep
/// save/load overhead negligible.
const SNAPSHOT_INTERVAL: usize = 25;
/// Window over which the opponent-pool win rate is computed (in episodes).
const WIN_RATE_WINDOW: usize = 200;
/// Log win rate / PPO stats every N updates.
const LOG_INTERVAL: usize = 20;

// ---------------------------------------------------------------------------
// Snapshot pool helpers
// ---------------------------------------------------------------------------

/// Clone a policy by copying tensor values through `VarStore::copy`. This
/// preserves the architecture and is much cheaper than save/load.
///
/// The snapshot is frozen because frozen snapshots never receive gradients
/// (they're opponents, not learners).
fn clone_policy(src: &MlpPolicy, obs_dim: i64, action_dim: i64) -> Result<MlpPolicy> {
    let mut dst = MlpPolicy::new(obs_dim, action_dim, HIDDEN_DIM);
    dst.var_store_mut().copy(src.var_store())?;
    dst.freeze(); // snapshots never receive gradients
    Ok(dst)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    tracing::info!("Starting Pong self-play PPO training");

    // Honor TOTAL_TIMESTEPS override for smoke runs without recompilation.
    let total_timesteps: usize = std::env::var("TOTAL_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(TOTAL_TIMESTEPS_DEFAULT);

    let env_probe = Pong::new();
    let obs_dim = env_probe.observation_space().shape[0] as i64;
    let action_dim = match env_probe.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n as i64,
        _ => panic!("Expected discrete action space"),
    };

    tracing::info!(
        "obs_dim={obs_dim}, action_dim={action_dim}, num_envs={NUM_ENVS}, steps={NUM_STEPS}, total_timesteps={total_timesteps}, pool_max={POOL_MAX}, snapshot_interval={SNAPSHOT_INTERVAL}"
    );

    // Use a plain Vec<Pong> rather than EnvPool because we need step_two,
    // which is specific to Pong and not part of the Environment trait. The
    // outer loop is the dominant cost (GPU forward passes), so per-env step
    // parallelism is not the bottleneck for 32 envs on CPU.
    let mut envs: Vec<Pong> = (0..NUM_ENVS).map(|_| Pong::new()).collect();
    let mut observations: Vec<Vec<f32>> = envs
        .iter_mut()
        .map(|e| {
            e.reset();
            e.get_observation()
        })
        .collect();

    let mut policy = MlpPolicy::new(obs_dim, action_dim, HIDDEN_DIM);
    let device = policy.device();
    tracing::info!("Device: {device:?}");

    let optimizer = policy.optimizer(LEARNING_RATE);

    let config = PPOConfig::new()
        .learning_rate(LEARNING_RATE)
        .n_epochs(10)
        .batch_size(512)
        .gamma(0.99)
        .gae_lambda(0.95)
        .clip_range(0.2)
        .clip_range_vf(f64::INFINITY)
        .vf_coef(0.5)
        .ent_coef(0.01)
        .max_grad_norm(0.5);

    let dummy_policy = MlpPolicy::new(obs_dim, action_dim, HIDDEN_DIM);
    let mut trainer = PPOTrainer::new(config, dummy_policy)?;
    trainer.set_optimizer(optimizer);

    let mut buffer = RolloutBuffer::new(NUM_STEPS, NUM_ENVS, obs_dim as usize);

    // ---------- Opponent snapshot pool ----------
    // Seed the pool with one clone of the freshly initialized random policy
    // so update 0 has a valid opponent. No pretraining against the rule-based
    // heuristic — the entire learning curve is measured against self-play.
    let mut snapshot_pool: VecDeque<MlpPolicy> = VecDeque::with_capacity(POOL_MAX);
    snapshot_pool.push_back(clone_policy(&policy, obs_dim, action_dim)?);
    tracing::info!("Seeded snapshot pool with initial random policy (size=1)");

    // Per-env snapshot assignment: which snapshot in the pool controls the
    // right paddle of env `i` for the current episode.
    let mut rng = StdRng::from_entropy();
    let mut env_snapshot_idx: Vec<usize> = (0..NUM_ENVS).map(|_| 0).collect();

    // Win/loss tracking: recent episode results from the left agent's
    // perspective (1.0 = win, 0.0 = loss, NaN-ish ignored). We treat each
    // completed episode as one match.
    let mut recent_results: VecDeque<f32> = VecDeque::with_capacity(WIN_RATE_WINDOW);
    // Per-env running score so we can determine the winner at episode end.
    let mut env_left_score: Vec<u32> = vec![0; NUM_ENVS];
    let mut env_right_score: Vec<u32> = vec![0; NUM_ENVS];

    let num_updates = total_timesteps / (NUM_STEPS * NUM_ENVS);
    let mut last_checkpoint = std::time::Instant::now();
    let train_start = std::time::Instant::now();

    tracing::info!("Training for {num_updates} updates ({total_timesteps} total steps)...");

    for update in 0..num_updates {
        buffer.reset();

        for step in 0..NUM_STEPS {
            // ---- Left paddle (live policy) forward pass ----
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_tensor = tch::Tensor::from_slice(&obs_flat)
                .reshape([NUM_ENVS as i64, obs_dim])
                .to_device(device);

            let (actions, log_probs, values) = policy.get_action(&obs_tensor);
            let actions_vec: Vec<i64> = Vec::try_from(actions)?;
            let log_probs_vec: Vec<f32> = Vec::try_from(log_probs)?;
            let values_vec: Vec<f32> = Vec::try_from(values)?;

            // ---- Right paddle (snapshot per env) forward pass ----
            // Mirror the left-perspective obs once, then group envs by which
            // snapshot they're assigned to so each snapshot does one batched
            // forward pass instead of NUM_ENVS individual ones.
            let mut right_actions: Vec<i64> = vec![1; NUM_ENVS]; // default stay

            // Build per-snapshot index lists
            let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
            for (env_id, &snap_idx) in env_snapshot_idx.iter().enumerate() {
                groups.entry(snap_idx).or_default().push(env_id);
            }

            for (snap_idx, env_ids) in &groups {
                let snap = match snapshot_pool.get(*snap_idx) {
                    Some(s) => s,
                    None => continue, // pool may have shrunk; default action stays
                };
                let snap_device = snap.device();

                // Gather mirrored obs for this group
                let mut group_obs_flat: Vec<f32> =
                    Vec::with_capacity(env_ids.len() * obs_dim as usize);
                for &env_id in env_ids {
                    let mirrored = mirror_observation(&observations[env_id]);
                    group_obs_flat.extend_from_slice(&mirrored);
                }
                let group_obs_tensor = tch::Tensor::from_slice(&group_obs_flat)
                    .reshape([env_ids.len() as i64, obs_dim])
                    .to_device(snap_device);

                let (acts, _, _) = tch::no_grad(|| snap.get_action(&group_obs_tensor));
                let acts_vec: Vec<i64> = Vec::try_from(acts)?;
                for (i, &env_id) in env_ids.iter().enumerate() {
                    right_actions[env_id] = acts_vec[i];
                }
            }

            // ---- Step all envs with (left_action, right_action) ----
            for env_id in 0..NUM_ENVS {
                let prev_left = env_left_score[env_id];
                let prev_right = env_right_score[env_id];

                let result = envs[env_id].step_two(actions_vec[env_id], right_actions[env_id]);

                buffer.add(
                    step,
                    env_id,
                    &observations[env_id],
                    actions_vec[env_id],
                    result.reward,
                    values_vec[env_id],
                    log_probs_vec[env_id],
                    result.terminated,
                    result.truncated,
                );
                observations[env_id] = result.observation.clone();

                // Track running score by detecting reward sign on a score event.
                // Reward of -1.0 means right scored, +1.0 means left scored.
                if result.reward >= 0.99 {
                    env_left_score[env_id] = prev_left + 1;
                } else if result.reward <= -0.99 {
                    env_right_score[env_id] = prev_right + 1;
                }

                if result.terminated || result.truncated {
                    // Record win rate: terminated = a side hit MAX_SCORE.
                    // Truncated episodes use the score delta sign; ties are
                    // recorded as 0.5 (rare but possible if the step counter
                    // hits MAX_STEPS exactly at parity).
                    let l = env_left_score[env_id] as i32;
                    let r = env_right_score[env_id] as i32;
                    let outcome = match l.cmp(&r) {
                        std::cmp::Ordering::Greater => 1.0,
                        std::cmp::Ordering::Less => 0.0,
                        std::cmp::Ordering::Equal => 0.5,
                    };
                    if recent_results.len() == WIN_RATE_WINDOW {
                        recent_results.pop_front();
                    }
                    recent_results.push_back(outcome);

                    trainer.increment_episodes(1);
                    envs[env_id].reset();
                    observations[env_id] = envs[env_id].get_observation();
                    env_left_score[env_id] = 0;
                    env_right_score[env_id] = 0;

                    // Re-sample opponent snapshot for the next episode
                    env_snapshot_idx[env_id] = rng.gen_range(0..snapshot_pool.len());
                }
            }

            trainer.increment_steps(NUM_ENVS);
        }

        // ---- Bootstrap last values ----
        let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let obs_tensor = tch::Tensor::from_slice(&obs_flat)
            .reshape([NUM_ENVS as i64, obs_dim])
            .to_device(device);
        let (_, _, last_values) = policy.get_action(&obs_tensor);
        let last_values_vec: Vec<f32> = Vec::try_from(last_values)?;

        buffer.compute_advantages(&last_values_vec, 0.99, 0.95);

        let batch = buffer.get_batch();

        // Convert to tensors via the shared helper.
        let t = batch.to_tch_tensors(device);

        let stats = trainer.train_step_with_policy(
            &policy,
            &t.observations,
            &t.actions,
            &t.old_log_probs,
            &t.old_values,
            &t.advantages,
            &t.returns,
            |p, obs, acts| p.evaluate_actions(obs, acts),
        )?;

        // ---- Snapshot push ----
        if (update + 1) % SNAPSHOT_INTERVAL == 0 {
            let snap = clone_policy(&policy, obs_dim, action_dim)?;
            if snapshot_pool.len() == POOL_MAX {
                snapshot_pool.pop_front(); // FIFO eviction
            }
            snapshot_pool.push_back(snap);

            // Persist to disk so future evaluation tooling can replay matches.
            let snap_path = format!("pong_snapshot_{}.pt", update + 1);
            if let Err(e) = policy.save(&snap_path) {
                tracing::warn!("Failed to save snapshot {snap_path}: {e}");
            } else {
                tracing::info!(
                    "Pushed snapshot at update {} (pool size = {}); saved {}",
                    update + 1,
                    snapshot_pool.len(),
                    snap_path
                );
            }
        }

        // ---- Logging ----
        if update % LOG_INTERVAL == 0 {
            let steps = trainer.total_steps();
            let episodes = trainer.total_episodes();
            let avg = if episodes > 0 {
                steps as f64 / episodes as f64
            } else {
                0.0
            };
            let win_rate = if recent_results.is_empty() {
                f32::NAN
            } else {
                recent_results.iter().sum::<f32>() / recent_results.len() as f32
            };
            tracing::info!(
                "Update {}/{} | Steps: {} | Episodes: {} | Avg: {:.1} | WinRate: {:.3} (n={}) | PoolSize: {} | Loss: {:.3} | Policy: {:.3} | Value: {:.3} | Entropy: {:.3} | ExpVar: {:.3}",
                update + 1,
                num_updates,
                steps,
                episodes,
                avg,
                win_rate,
                recent_results.len(),
                snapshot_pool.len(),
                stats.total_loss,
                stats.policy_loss,
                stats.value_loss,
                stats.entropy,
                stats.explained_var,
            );
        }

        if last_checkpoint.elapsed().as_secs() >= CHECKPOINT_INTERVAL_SECS {
            let steps = trainer.total_steps();
            let cp = format!("pong_self_play_checkpoint_{steps}steps.pt");
            if let Err(e) = policy.save(&cp) {
                tracing::warn!("Failed to save checkpoint {cp}: {e}");
            } else {
                tracing::info!("Checkpoint: {cp}");
            }
            last_checkpoint = std::time::Instant::now();
        }
    }

    tracing::info!(
        "Training complete! Steps={} Episodes={}",
        trainer.total_steps(),
        trainer.total_episodes()
    );

    let elapsed = train_start.elapsed().as_secs_f64();
    let total_episodes = trainer.total_episodes().max(1);
    let avg_steps = trainer.total_steps() as f64 / total_episodes as f64;

    // ---------- Final exports ----------
    policy.save("pong_self_play_model.pt")?;
    tracing::info!("Saved pong_self_play_model.pt");

    let mut model = policy.export_for_inference();
    let mut hparams = HashMap::new();
    hparams.insert("num_envs".to_string(), serde_json::json!(NUM_ENVS));
    hparams.insert("num_steps".to_string(), serde_json::json!(NUM_STEPS));
    hparams.insert("hidden_dim".to_string(), serde_json::json!(HIDDEN_DIM));
    hparams.insert("learning_rate".to_string(), serde_json::json!(LEARNING_RATE));
    hparams.insert("gamma".to_string(), serde_json::json!(0.99));
    hparams.insert("ent_coef".to_string(), serde_json::json!(0.01));
    hparams.insert("snapshot_pool_size".to_string(), serde_json::json!(POOL_MAX));
    hparams.insert("snapshot_interval".to_string(), serde_json::json!(SNAPSHOT_INTERVAL));
    hparams.insert("total_timesteps".to_string(), serde_json::json!(total_timesteps));

    let final_win_rate = if recent_results.is_empty() {
        f32::NAN
    } else {
        recent_results.iter().sum::<f32>() / recent_results.len() as f32
    };

    model.metadata = Some(TrainingMetadata {
        total_steps: trainer.total_steps(),
        total_episodes: trainer.total_episodes(),
        final_performance: avg_steps,
        training_time_secs: elapsed,
        device: format!("{device:?}"),
        environment: "Pong".to_string(),
        algorithm: "PPO + self-play".to_string(),
        timestamp: Some(chrono::Utc::now().to_rfc3339()),
        hyperparameters: Some(hparams),
        notes: Some(format!(
            "Self-play training: left paddle = live PPO policy; right paddle = uniform sample from FIFO snapshot pool (max {POOL_MAX}, refreshed every {SNAPSHOT_INTERVAL} updates). Final opponent-pool win rate over last {} episodes: {:.3}.",
            recent_results.len(),
            final_win_rate
        )),
    });

    model.save_json("pong_self_play_model.json")?;
    tracing::info!("Saved pong_self_play_model.json");

    Ok(())
}
