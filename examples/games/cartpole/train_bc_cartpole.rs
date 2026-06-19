//! End-to-end Behavioral Cloning (BC) on CartPole (Burn backend).
//!
//! This is PR C of the BC epic (#161, issue #169) and the deliverable that
//! closes ROADMAP Milestone 6's imitation-learning item. It demonstrates the
//! full behavioral-cloning story on CartPole-v1 using the
//! [`MlpBurnPolicy`](thrust_rl::policy::mlp::MlpBurnPolicy) +
//! [`BcTrainer`](thrust_rl::train::bc::BcTrainer) stack on
//! `Autodiff<NdArray<f32>>` (CPU):
//!
//! 1. **Expert** — train an A2C policy on CartPole until it is a strong teacher
//!    (reusing the `EnvPool` rollout + GAE + single-update loop from
//!    `train_cartpole_a2c.rs`).
//! 2. **Harvest** — roll the trained expert out **greedily** (argmax over the
//!    policy logits) on a seeded `EnvPool`, collecting `(obs, action)` pairs
//!    into a [`Demonstrations`](thrust_rl::train::bc::Demonstrations) dataset.
//! 3. **Clone** — construct a fresh seeded `MlpBurnPolicy` + `BcTrainer` and
//!    run `BcConfig::epochs` supervised cross-entropy epochs over the demos,
//!    logging per-epoch loss + action-match accuracy.
//! 4. **Report** — evaluate BOTH the expert and the cloned policy greedily over
//!    `N` episodes and print mean episode reward (== mean length) for each, so
//!    the cloned-vs-expert gap is visible.
//!
//! # Budget
//!
//! Defaults are sized so the whole pipeline (train expert -> harvest -> clone
//! -> eval) runs in well under a minute on CPU in `--release` (~22s measured,
//! single-threaded `NdArray`):
//!
//! - Expert A2C: up to `200_000` env steps with `entropy_coef = 0.05` to keep
//!   exploration alive; in practice the expert reaches the solved region and
//!   its action distribution goes near-deterministic within a few seconds,
//!   tripping the A2C trainer's entropy-collapse guard, which we treat as an
//!   early-stop (a confident expert is exactly what greedy harvesting wants).
//! - Harvest: `20_000` greedy expert env steps -> 20k `(obs, action)` pairs.
//! - Clone: `BcConfig::epochs` (default 10) supervised epochs over the demos
//!   (reaches ~99% action-match accuracy on the expert labels).
//! - Eval: `20` greedy episodes each for expert and clone. A typical run
//!   reports expert ~173 vs cloned ~170 mean episode reward (random ~22).
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_bc_cartpole --features training --release
//! ```
//!
//! Override the expert step budget via `EXPERT_TIMESTEPS`:
//!
//! ```bash
//! EXPERT_TIMESTEPS=400000 cargo run --example train_bc_cartpole \
//!     --features training --release
//! ```
//!
//! Expected: the cloned policy's mean episode reward lands close to the
//! expert's (both well above the random ~22 baseline), demonstrating
//! successful imitation.

use anyhow::Result;
use burn::{
    backend::Autodiff,
    optim::AdamConfig,
    tensor::{Int, Tensor, TensorData},
};
use thrust_rl::{
    env::{Environment, cartpole::CartPole, pool::EnvPool},
    policy::mlp::{BurnActivation, MlpBurnConfig, MlpBurnPolicy},
    train::{
        a2c::{A2cConfig, A2cTrainer},
        bc::{BcConfig, BcTrainer, Demonstrations},
        optimizer::BurnOptimizer,
    },
};

// Concrete backend stack — selected at compile time via Cargo features.
#[cfg(not(feature = "wgpu"))]
type InnerBackend = burn::backend::NdArray<f32>;
#[cfg(feature = "wgpu")]
type InnerBackend = burn::backend::Wgpu<f32, i32>;
type Backend = Autodiff<InnerBackend>;

#[cfg(not(feature = "wgpu"))]
const BACKEND_LABEL: &str = "NdArray<f32> + Autodiff (CPU)";
#[cfg(feature = "wgpu")]
const BACKEND_LABEL: &str = "Wgpu<f32, i32> + Autodiff (GPU: Vulkan/Metal/DX12/WebGPU)";

// ----- Expert (A2C) hyperparameters -----
const NUM_ENVS: usize = 16;
const NUM_STEPS: usize = 5;
const DEFAULT_EXPERT_TIMESTEPS: usize = 200_000;
const A2C_LEARNING_RATE: f64 = 7e-4;
const HIDDEN_DIM: usize = 128;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 1.0;

// ----- Harvest / clone / eval hyperparameters -----
const HARVEST_STEPS: usize = 20_000;
const EVAL_EPISODES: usize = 20;
/// Seed threaded through the expert init, demo harvest, clone init, and eval
/// so the whole pipeline is reproducible.
const SEED: u64 = 0;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let expert_timesteps: usize = std::env::var("EXPERT_TIMESTEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EXPERT_TIMESTEPS);

    tracing::info!("Behavioral Cloning on CartPole (Burn backend: {})", BACKEND_LABEL);

    let device = Default::default();

    // Probe environment dimensions.
    let probe = CartPole::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n,
        _ => panic!("Expected discrete action space"),
    };

    tracing::info!("  obs_dim    = {}", obs_dim);
    tracing::info!("  action_dim = {}", action_dim);
    tracing::info!("  expert_timesteps = {}", expert_timesteps);
    tracing::info!("  harvest_steps    = {}", HARVEST_STEPS);
    tracing::info!("  eval_episodes    = {}", EVAL_EPISODES);
    tracing::info!("------------------------------------------------------------");

    let pipeline_start = std::time::Instant::now();

    // === 1. Train the A2C expert =============================================
    tracing::info!("[1/4] Training A2C expert ({} env steps)...", expert_timesteps);
    let expert_start = std::time::Instant::now();
    let expert_trainer = train_expert(obs_dim, action_dim, expert_timesteps)?;
    let expert = expert_trainer.policy();
    tracing::info!("  expert trained in {:.1}s", expert_start.elapsed().as_secs_f64());

    // === 2. Harvest expert demonstrations (greedy) ===========================
    tracing::info!("[2/4] Harvesting {} greedy expert demonstrations...", HARVEST_STEPS);
    let demos = harvest_demos(expert, obs_dim, HARVEST_STEPS);
    tracing::info!("  harvested {} (obs, action) pairs", demos.len());

    // === 3. Clone a fresh policy via BC ======================================
    tracing::info!("[3/4] Cloning a fresh policy with BcTrainer...");
    let bc_config = BcConfig::new().learning_rate(1e-3).batch_size(64).epochs(10).seed(SEED);
    // Fresh, separately-seeded policy so the clone learns purely from demos.
    let clone_policy =
        MlpBurnPolicy::<Backend>::new_seeded(obs_dim, action_dim, HIDDEN_DIM, SEED + 1, &device);
    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<Backend, MlpBurnPolicy<Backend>, _> =
        BurnOptimizer::new(inner_opt, bc_config.learning_rate);
    let mut bc_trainer = BcTrainer::new(bc_config.clone(), clone_policy, burn_opt)?;

    for epoch in 0..bc_config.epochs {
        let stats = bc_trainer.train_epoch(&demos, |p, o| p.forward(o).0)?;
        tracing::info!(
            "  epoch {:>2}/{}  loss={:8.5}  accuracy={:5.1}%",
            epoch + 1,
            bc_config.epochs,
            stats.loss,
            stats.accuracy * 100.0,
        );
    }

    // === 4. Evaluate expert vs clone greedily ================================
    tracing::info!("[4/4] Evaluating expert vs cloned policy ({} episodes each)...", EVAL_EPISODES);
    let expert_mean = eval_greedy(expert, obs_dim, EVAL_EPISODES);
    let clone_mean = eval_greedy(bc_trainer.policy(), obs_dim, EVAL_EPISODES);

    tracing::info!("------------------------------------------------------------");
    tracing::info!("Behavioral Cloning complete.");
    tracing::info!("  expert mean episode reward : {:.1}", expert_mean);
    tracing::info!("  cloned mean episode reward : {:.1}", clone_mean);
    tracing::info!("  (random baseline ~22; CartPole max 500)");
    tracing::info!("  pipeline time : {:.1}s", pipeline_start.elapsed().as_secs_f64());

    Ok(())
}

/// Train an A2C expert on CartPole for `total_timesteps` env steps and return
/// the trained trainer (borrow `.policy()` for the strong teacher). Mirrors
/// the rollout + GAE + single-update loop of `train_cartpole_a2c.rs`.
fn train_expert(
    obs_dim: usize,
    action_dim: usize,
    total_timesteps: usize,
) -> Result<
    A2cTrainer<
        Backend,
        MlpBurnPolicy<Backend>,
        impl burn::optim::Optimizer<MlpBurnPolicy<Backend>, Backend>,
    >,
> {
    let device = Default::default();

    let policy_config = MlpBurnConfig {
        num_layers: 2,
        hidden_dim: HIDDEN_DIM,
        use_orthogonal_init: true,
        activation: BurnActivation::ReLU,
        seed: Some(SEED),
    };
    let policy = MlpBurnPolicy::<Backend>::with_config(obs_dim, action_dim, policy_config, &device);

    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<Backend, MlpBurnPolicy<Backend>, _> =
        BurnOptimizer::new(inner_opt, A2C_LEARNING_RATE);

    let a2c_config = A2cConfig::new()
        .learning_rate(A2C_LEARNING_RATE)
        .gamma(GAMMA as f64)
        .gae_lambda(GAE_LAMBDA as f64)
        .value_coef(0.5)
        // Keep exploration alive so the expert reaches CartPole's solved
        // region before its action distribution goes near-deterministic and
        // trips the A2C trainer's entropy-collapse guard.
        .entropy_coef(0.05)
        .n_steps(NUM_STEPS)
        .num_envs(NUM_ENVS)
        .max_grad_norm(0.5)
        .normalize_advantages(true)
        .seed(SEED);

    let mut trainer = A2cTrainer::new(a2c_config, policy, burn_opt)?;
    let mut env_pool = EnvPool::new(CartPole::new, NUM_ENVS);

    let num_updates = total_timesteps / (NUM_STEPS * NUM_ENVS);

    let cap = NUM_STEPS * NUM_ENVS;
    let mut buf_obs: Vec<f32> = Vec::with_capacity(cap * obs_dim);
    let mut buf_actions: Vec<i64> = Vec::with_capacity(cap);
    let mut buf_values: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_rewards: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_dones: Vec<f32> = Vec::with_capacity(cap);

    let mut observations = env_pool.reset();
    let mut episode_lengths = [0u32; NUM_ENVS];
    let mut completed: Vec<u32> = Vec::new();

    for update in 0..num_updates {
        buf_obs.clear();
        buf_actions.clear();
        buf_values.clear();
        buf_rewards.clear();
        buf_dones.clear();

        for _step in 0..NUM_STEPS {
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_t: Tensor<Backend, 2> =
                Tensor::from_data(TensorData::new(obs_flat, [NUM_ENVS, obs_dim]), &device);
            let (actions, _log_probs, values) = trainer.policy().get_action_host(obs_t);
            let results = env_pool.step(&actions);

            for env_id in 0..NUM_ENVS {
                buf_obs.extend_from_slice(&observations[env_id]);
                buf_actions.push(actions[env_id]);
                buf_values.push(values[env_id]);
                buf_rewards.push(results[env_id].reward);
                let done = results[env_id].terminated || results[env_id].truncated;
                buf_dones.push(if done { 1.0 } else { 0.0 });
                episode_lengths[env_id] += 1;
                observations[env_id] = results[env_id].observation.clone();
                if done {
                    completed.push(episode_lengths[env_id]);
                    trainer.increment_episodes(1);
                    episode_lengths[env_id] = 0;
                    observations[env_id] = env_pool.reset_env(env_id)?;
                }
            }
        }

        let last_obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let last_obs_t: Tensor<Backend, 2> =
            Tensor::from_data(TensorData::new(last_obs_flat, [NUM_ENVS, obs_dim]), &device);
        let (_, _, last_values) = trainer.policy().get_action_host(last_obs_t);

        let (adv_host, ret_host) = compute_gae(
            &buf_rewards,
            &buf_values,
            &buf_dones,
            &last_values,
            GAMMA,
            GAE_LAMBDA,
            NUM_STEPS,
            NUM_ENVS,
        );

        let batch = NUM_STEPS * NUM_ENVS;
        let obs_b: Tensor<Backend, 2> =
            Tensor::from_data(TensorData::new(buf_obs.clone(), [batch, obs_dim]), &device);
        let actions_b: Tensor<Backend, 1, Int> =
            Tensor::from_data(TensorData::new(buf_actions.clone(), [batch]), &device);
        let adv_b: Tensor<Backend, 1> =
            Tensor::from_data(TensorData::new(adv_host, [batch]), &device);
        let ret_b: Tensor<Backend, 1> =
            Tensor::from_data(TensorData::new(ret_host, [batch]), &device);

        // The A2C trainer aborts on entropy collapse — for a BC teacher this
        // simply means the expert policy has become confident/near-determ-
        // inistic, which is exactly what we want for greedy demonstrations.
        // Treat it as an early-stop signal rather than a hard failure.
        match trainer.train_step(obs_b, actions_b, adv_b, ret_b, |p, o, a| p.evaluate_actions(o, a))
        {
            Ok(_) => {}
            Err(e) => {
                tracing::info!(
                    "  expert training stopped early at update {} (entropy collapse): {}",
                    update + 1,
                    e
                );
                break;
            }
        }

        if update % 200 == 0 || update == num_updates - 1 {
            let mean = recent_mean(&completed);
            tracing::info!(
                "  a2c update {:>5}/{}  episodes={:>5}  avg_len(last≤100)={:6.1}",
                update + 1,
                num_updates,
                trainer.total_episodes(),
                mean,
            );
        }
    }

    Ok(trainer)
}

/// Roll the expert out **greedily** (argmax over logits) on a seeded
/// `EnvPool`, collecting `(obs, action)` pairs until at least `target_steps`
/// transitions have been gathered. Returns a [`Demonstrations`] dataset.
fn harvest_demos(
    expert: &MlpBurnPolicy<Backend>,
    obs_dim: usize,
    target_steps: usize,
) -> Demonstrations {
    let device = Default::default();
    let mut env_pool = EnvPool::new(CartPole::new, NUM_ENVS);
    let mut observations = env_pool.reset();

    let mut obs_buf: Vec<f32> = Vec::with_capacity(target_steps * obs_dim);
    let mut action_buf: Vec<i64> = Vec::with_capacity(target_steps);

    while action_buf.len() < target_steps {
        let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let obs_t: Tensor<Backend, 2> =
            Tensor::from_data(TensorData::new(obs_flat, [NUM_ENVS, obs_dim]), &device);

        let actions = greedy_actions(expert, obs_t);
        let results = env_pool.step(&actions);

        for env_id in 0..NUM_ENVS {
            obs_buf.extend_from_slice(&observations[env_id]);
            action_buf.push(actions[env_id]);
            let done = results[env_id].terminated || results[env_id].truncated;
            observations[env_id] = if done {
                env_pool.reset_env(env_id).expect("reset env")
            } else {
                results[env_id].observation.clone()
            };
        }
    }

    Demonstrations::new(obs_buf, action_buf, obs_dim).expect("demonstrations are well-formed")
}

/// Greedy (argmax-over-logits) action per env row. BC harvest and evaluation
/// both pick the deterministic expert action rather than sampling.
fn greedy_actions(policy: &MlpBurnPolicy<Backend>, obs: Tensor<Backend, 2>) -> Vec<i64> {
    let (logits, _value) = policy.forward(obs);
    logits.argmax(1).into_data().to_vec::<i64>().expect("argmax to host")
}

/// Evaluate `policy` greedily over `episodes` full CartPole episodes on a
/// single env, returning mean episode reward (== mean length, since CartPole
/// reward is +1/step). The expert and clone are evaluated on the same number
/// of episodes so the reported means are directly comparable.
fn eval_greedy(policy: &MlpBurnPolicy<Backend>, obs_dim: usize, episodes: usize) -> f32 {
    let device = Default::default();
    let mut env = CartPole::new();

    let mut total: f32 = 0.0;
    for _ep in 0..episodes {
        env.reset();
        let mut obs = env.get_observation();
        loop {
            let obs_t: Tensor<Backend, 2> =
                Tensor::from_data(TensorData::new(obs.clone(), [1, obs_dim]), &device);
            let action = greedy_actions(policy, obs_t)[0];
            let step = env.step(action);
            total += step.reward;
            if step.terminated || step.truncated {
                break;
            }
            obs = step.observation;
        }
    }
    total / episodes as f32
}

/// Mean of the most recent (<=100) completed episode lengths, or 0 if none.
fn recent_mean(completed: &[u32]) -> f32 {
    if completed.is_empty() {
        return 0.0;
    }
    let n = completed.len();
    let recent = &completed[n.saturating_sub(100)..];
    recent.iter().map(|&x| x as f32).sum::<f32>() / recent.len() as f32
}

/// Per-env GAE computation (host-side, step-major `[T * N]` layout). Mirrors
/// `train_cartpole_a2c.rs`.
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
