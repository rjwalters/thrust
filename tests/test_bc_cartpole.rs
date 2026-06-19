//! Behavioral Cloning eval tests on the [`CartPole`] discrete-control env.
//!
//! This is PR C of the BC epic (#161, issue #169) and the deliverable that
//! closes ROADMAP Milestone 6's imitation-learning item. Two tiers, mirroring
//! the A2C / SAC convention (`tests/test_a2c_cartpole.rs`,
//! `tests/test_sac_pendulum.rs`):
//!
//! 1. **`bc_cartpole_clone_runs`** (always runs) — a fast, unit-level check
//!    that the BC pipeline wires together end-to-end on a tiny demo set: a
//!    `BcTrainer` runs a couple of supervised epochs over a small synthetic
//!    CartPole-shaped demonstration set, reports finite loss/accuracy, and the
//!    cloned policy produces in-range discrete actions. This is the CI default
//!    — sub-second, no convergence bar. (The trainer's own minibatch smoke test
//!    lives in `src/train/bc/trainer.rs`; this one additionally exercises
//!    greedy action extraction on the cloned policy.)
//!
//! 2. **`bc_cartpole_clones_expert`** (`#[ignore]`) — the architect's eval bar:
//!    train an A2C expert on CartPole, harvest greedy demonstrations, clone a
//!    fresh policy with BC, and assert the cloned policy reaches **mean episode
//!    length >= 150 over >= 20 episodes**. CartPole reward is +1/step, so
//!    episode length == episode reward; a random policy sits near ~22 steps and
//!    the expert near ~195+, so 150 cleanly separates "cloned the expert" from
//!    "did not learn".
//!
//! ## Why the heavy bar is `#[ignore]`d
//!
//! The eval run trains a full A2C expert before harvesting and cloning —
//! ~80s wall-clock on the CPU `NdArray` backend (the expert early-stops on
//! entropy collapse within seconds; harvest + 10 clone epochs + eval make up
//! the rest), too slow to gate every `cargo test` run. It is opt-in:
//!
//! ```text
//! cargo test --release --features training --test test_bc_cartpole -- --ignored
//! ```
//!
//! The fast clone test above guarantees the BC pipeline keeps working on
//! every CI run; the eval test is the periodic / release-gate check.

#![cfg(feature = "training")]

use burn::{
    backend::{Autodiff, NdArray},
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

type B = Autodiff<NdArray<f32>>;

const HIDDEN_DIM: usize = 128;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 1.0;

/// Greedy (argmax-over-logits) action per env row, on the host.
fn greedy_actions(policy: &MlpBurnPolicy<B>, obs: Tensor<B, 2>) -> Vec<i64> {
    let (logits, _value) = policy.forward(obs);
    logits.argmax(1).into_data().to_vec::<i64>().expect("argmax to host")
}

/// Probe CartPole's observation / action dimensions.
fn cartpole_dims() -> (usize, usize) {
    let probe = CartPole::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n,
        _ => panic!("expected discrete action space"),
    };
    (obs_dim, action_dim)
}

/// Fast, always-on smoke test: clone a fresh policy over a tiny synthetic
/// CartPole-shaped demonstration set and confirm the cloned policy emits
/// in-range greedy actions. Keeps the BC pipeline wired up on every CI run
/// without a convergence bar.
#[test]
fn bc_cartpole_clone_runs() {
    let device = Default::default();
    let (obs_dim, action_dim) = cartpole_dims();

    // Tiny synthetic demos: 8 CartPole-shaped observations with arbitrary but
    // valid discrete action labels. Enough to drive a couple of gradient
    // steps; this test asserts wiring, not convergence.
    let obs: Vec<f32> = (0..8)
        .flat_map(|i| {
            let f = i as f32 * 0.01;
            vec![f, -f, 0.5 * f, -0.5 * f]
        })
        .collect();
    let actions: Vec<i64> = (0..8).map(|i| (i % action_dim) as i64).collect();
    let demos = Demonstrations::new(obs, actions, obs_dim).expect("well-formed demos");

    let config = BcConfig::new().batch_size(4).epochs(2).seed(7);
    let policy = MlpBurnPolicy::<B>::new_seeded(obs_dim, action_dim, 16, config.seed, &device);
    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> =
        BurnOptimizer::new(inner_opt, config.learning_rate);
    let mut trainer = BcTrainer::new(config.clone(), policy, burn_opt).expect("trainer constructs");

    for _ in 0..config.epochs {
        let stats = trainer.train_epoch(&demos, |p, o| p.forward(o).0).expect("epoch runs");
        assert!(stats.loss.is_finite(), "loss must be finite, got {}", stats.loss);
        assert!(
            (0.0..=1.0).contains(&stats.accuracy),
            "accuracy in [0,1], got {}",
            stats.accuracy
        );
    }
    assert!(trainer.total_steps() >= 2, "expected at least 2 gradient steps");

    // The cloned policy must produce valid in-range discrete actions on a
    // fresh CartPole observation.
    let mut env = CartPole::new();
    env.reset();
    let obs = env.get_observation();
    let obs_t: Tensor<B, 2> = Tensor::from_data(TensorData::new(obs, [1, obs_dim]), &device);
    let action = greedy_actions(trainer.policy(), obs_t)[0];
    assert!(
        (0..action_dim as i64).contains(&action),
        "cloned action {action} out of range 0..{action_dim}"
    );
}

/// Build a seeded A2C trainer + EnvPool for the CartPole expert.
#[allow(clippy::type_complexity)]
fn build_expert(
    num_envs: usize,
    n_steps: usize,
    seed: u64,
) -> (
    A2cTrainer<B, MlpBurnPolicy<B>, impl burn::optim::Optimizer<MlpBurnPolicy<B>, B>>,
    EnvPool<CartPole>,
    usize,
) {
    let device = Default::default();
    let (obs_dim, action_dim) = cartpole_dims();

    let policy_config = MlpBurnConfig {
        num_layers: 2,
        hidden_dim: HIDDEN_DIM,
        use_orthogonal_init: true,
        activation: BurnActivation::ReLU,
        seed: Some(seed),
    };
    let policy = MlpBurnPolicy::<B>::with_config(obs_dim, action_dim, policy_config, &device);

    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> = BurnOptimizer::new(inner_opt, 7e-4);
    let config = A2cConfig::new()
        .learning_rate(7e-4)
        .gamma(GAMMA as f64)
        .gae_lambda(GAE_LAMBDA as f64)
        .value_coef(0.5)
        // Keep exploration alive so the expert reaches CartPole's solved
        // region before going near-deterministic and tripping the A2C
        // trainer's entropy-collapse guard (matches the example).
        .entropy_coef(0.05)
        .n_steps(n_steps)
        .num_envs(num_envs)
        .max_grad_norm(0.5)
        .normalize_advantages(true)
        .seed(seed);

    let trainer = A2cTrainer::new(config, policy, burn_opt).expect("trainer constructs");
    let env_pool = EnvPool::new(CartPole::new, num_envs);
    (trainer, env_pool, obs_dim)
}

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

/// Train the A2C expert for `num_updates` updates against `env_pool`.
fn train_expert(
    trainer: &mut A2cTrainer<B, MlpBurnPolicy<B>, impl burn::optim::Optimizer<MlpBurnPolicy<B>, B>>,
    env_pool: &mut EnvPool<CartPole>,
    obs_dim: usize,
    num_envs: usize,
    n_steps: usize,
    num_updates: usize,
) {
    let device = Default::default();
    let cap = n_steps * num_envs;
    let mut buf_obs: Vec<f32> = Vec::with_capacity(cap * obs_dim);
    let mut buf_actions: Vec<i64> = Vec::with_capacity(cap);
    let mut buf_values: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_rewards: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_dones: Vec<f32> = Vec::with_capacity(cap);

    let mut observations = env_pool.reset();
    let mut episode_lengths = vec![0u32; num_envs];

    for _update in 0..num_updates {
        buf_obs.clear();
        buf_actions.clear();
        buf_values.clear();
        buf_rewards.clear();
        buf_dones.clear();

        for _step in 0..n_steps {
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_t: Tensor<B, 2> =
                Tensor::from_data(TensorData::new(obs_flat, [num_envs, obs_dim]), &device);
            let (actions, _log_probs, values) = trainer.policy().get_action_host(obs_t);
            let results = env_pool.step(&actions);

            for env_id in 0..num_envs {
                buf_obs.extend_from_slice(&observations[env_id]);
                buf_actions.push(actions[env_id]);
                buf_values.push(values[env_id]);
                buf_rewards.push(results[env_id].reward);
                let done = results[env_id].terminated || results[env_id].truncated;
                buf_dones.push(if done { 1.0 } else { 0.0 });
                episode_lengths[env_id] += 1;
                observations[env_id] = results[env_id].observation.clone();
                if done {
                    trainer.increment_episodes(1);
                    episode_lengths[env_id] = 0;
                    observations[env_id] = env_pool.reset_env(env_id).expect("reset env");
                }
            }
        }

        let last_obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let last_obs_t: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(last_obs_flat, [num_envs, obs_dim]), &device);
        let (_, _, last_values) = trainer.policy().get_action_host(last_obs_t);

        let (adv_host, ret_host) = compute_gae(
            &buf_rewards,
            &buf_values,
            &buf_dones,
            &last_values,
            GAMMA,
            GAE_LAMBDA,
            n_steps,
            num_envs,
        );

        let batch = n_steps * num_envs;
        let obs_b: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(buf_obs.clone(), [batch, obs_dim]), &device);
        let actions_b: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(buf_actions.clone(), [batch]), &device);
        let adv_b: Tensor<B, 1> = Tensor::from_data(TensorData::new(adv_host, [batch]), &device);
        let ret_b: Tensor<B, 1> = Tensor::from_data(TensorData::new(ret_host, [batch]), &device);

        // The A2C trainer aborts on entropy collapse; for a BC teacher that
        // just means the expert policy went confident/near-deterministic,
        // which is what we want. Treat it as an early-stop, not a failure.
        if trainer
            .train_step(obs_b, actions_b, adv_b, ret_b, |p, o, a| p.evaluate_actions(o, a))
            .is_err()
        {
            break;
        }
    }
}

/// Harvest `target_steps` greedy expert `(obs, action)` pairs into a
/// [`Demonstrations`] dataset.
fn harvest_demos(
    expert: &MlpBurnPolicy<B>,
    env_pool: &mut EnvPool<CartPole>,
    obs_dim: usize,
    num_envs: usize,
    target_steps: usize,
) -> Demonstrations {
    let device = Default::default();
    let mut observations = env_pool.reset();
    let mut obs_buf: Vec<f32> = Vec::with_capacity(target_steps * obs_dim);
    let mut action_buf: Vec<i64> = Vec::with_capacity(target_steps);

    while action_buf.len() < target_steps {
        let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let obs_t: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(obs_flat, [num_envs, obs_dim]), &device);
        let actions = greedy_actions(expert, obs_t);
        let results = env_pool.step(&actions);

        for env_id in 0..num_envs {
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

    Demonstrations::new(obs_buf, action_buf, obs_dim).expect("well-formed demos")
}

/// Evaluate `policy` greedily over `episodes` full CartPole episodes, returning
/// mean episode length (== mean reward; CartPole reward is +1/step).
fn eval_mean_length(policy: &MlpBurnPolicy<B>, obs_dim: usize, episodes: usize) -> f32 {
    let device = Default::default();
    let mut env = CartPole::new();
    let mut total = 0u64;
    for _ep in 0..episodes {
        env.reset();
        let mut obs = env.get_observation();
        let mut len = 0u64;
        loop {
            let obs_t: Tensor<B, 2> =
                Tensor::from_data(TensorData::new(obs.clone(), [1, obs_dim]), &device);
            let action = greedy_actions(policy, obs_t)[0];
            let step = env.step(action);
            len += 1;
            if step.terminated || step.truncated {
                break;
            }
            obs = step.observation;
        }
        total += len;
    }
    total as f32 / episodes as f32
}

/// Heavy eval test (opt-in via `--ignored`): train an A2C expert, harvest
/// greedy demos, clone a fresh policy with BC, and assert the cloned policy
/// reaches mean episode length >= 150 over >= 20 episodes.
///
/// Run with:
/// ```text
/// cargo test --release --features training --test test_bc_cartpole -- --ignored
/// ```
#[test]
#[ignore = "~80s expert-train + clone + eval; opt in with --ignored (prefer --release)"]
fn bc_cartpole_clones_expert() {
    let num_envs = 16;
    let n_steps = 5;
    let seed = 0;

    // --- 1. Train the A2C expert. A modest 200k-step teacher already produces
    // near-optimal greedy rollouts, which is all BC needs.
    let expert_timesteps = 200_000;
    let num_updates = expert_timesteps / (n_steps * num_envs);
    let (mut expert_trainer, mut expert_pool, obs_dim) = build_expert(num_envs, n_steps, seed);
    train_expert(&mut expert_trainer, &mut expert_pool, obs_dim, num_envs, n_steps, num_updates);
    let expert = expert_trainer.policy();

    // Sanity: the teacher must actually be strong, otherwise the bar is
    // meaningless.
    let expert_mean = eval_mean_length(expert, obs_dim, 20);
    assert!(
        expert_mean >= 150.0,
        "expert teacher too weak ({expert_mean:.1}); BC cannot exceed its teacher"
    );

    // --- 2. Harvest greedy demonstrations.
    let mut harvest_pool = EnvPool::new(CartPole::new, num_envs);
    let demos = harvest_demos(expert, &mut harvest_pool, obs_dim, num_envs, 20_000);
    assert!(demos.len() >= 20_000, "expected >= 20k demos, got {}", demos.len());

    // --- 3. Clone a fresh, separately-seeded policy with BC.
    let (_, action_dim) = cartpole_dims();
    let device = Default::default();
    let bc_config = BcConfig::new().learning_rate(1e-3).batch_size(64).epochs(10).seed(seed);
    let clone_policy =
        MlpBurnPolicy::<B>::new_seeded(obs_dim, action_dim, HIDDEN_DIM, seed + 1, &device);
    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<B, MlpBurnPolicy<B>, _> =
        BurnOptimizer::new(inner_opt, bc_config.learning_rate);
    let mut bc_trainer = BcTrainer::new(bc_config.clone(), clone_policy, burn_opt).unwrap();
    for _ in 0..bc_config.epochs {
        bc_trainer.train_epoch(&demos, |p, o| p.forward(o).0).expect("epoch runs");
    }

    // --- 4. Eval the cloned policy.
    let episodes = 20;
    let clone_mean = eval_mean_length(bc_trainer.policy(), obs_dim, episodes);
    assert!(
        clone_mean >= 150.0,
        "cloned mean episode length over {episodes} episodes was {clone_mean:.1}, expected >= 150 \
         (random baseline ~22, expert {expert_mean:.1})"
    );
}
