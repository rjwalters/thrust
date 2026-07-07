# Tutorial 2 — Solving CartPole with PPO

> **You will:** train a PPO agent to balance CartPole, this time letting
> `PPOTrainerBurn` own the update step, and learn to read the learning curve it
> produces.
> **Prerequisites:** [Tutorial 1](01-your-first-agent.md) — you should recognize
> the rollout → advantage → update loop and Burn's move-through optimizer.
> **Time:** ~15 minutes.

Tutorial 1 hand-wrote the whole PPO update to demystify it. Now we graduate to
a **real sequential task** and a **real trainer**. Two things change from the
bandit:

1. **The future matters.** In CartPole, the action you take now affects the
   states you'll see for the rest of the episode. That means we need proper
   temporal credit assignment — **GAE** (Generalized Advantage Estimation) —
   instead of the bandit's one-line `reward − value`.
2. **We stop writing the loss by hand.** `PPOTrainerBurn` owns the clipped
   surrogate, value loss, entropy bonus, gradient clipping, and the optimizer
   step. We keep control of the rollout and the advantage computation, and hand
   the trainer a batch of tensors. This is the division of labor you'll use for
   every on-policy trainer in Thrust.

## The problem: CartPole-v1

A pole is hinged on a cart; you push the cart left (`0`) or right (`1`) each
step to keep the pole upright. The observation is four floats
(`[cart position, cart velocity, pole angle, pole angular velocity]`), the
reward is `+1` per step the pole stays up, and the episode ends when the pole
falls or at 500 steps. So **episode reward == episode length**, and a solved
agent scores near 500. Random pushing scores ~22.

## GAE, briefly

The advantage answers "how much better than expected was this action?". For a
sequential task, "expected" has to account for the future, so GAE walks the
rollout **backwards**, discounting future rewards by `gamma` and blending
multi-step estimates with `gae_lambda`:

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t     = delta_t + gamma * lambda * A_{t+1}
```

`gamma = 0.99` says "future reward is almost as valuable as immediate reward"
(appropriate when episodes are long and every step matters). `gae_lambda = 0.95`
trades a little bias for much lower variance. Contrast this with Tutorial 1,
where we set both to `0` precisely because the bandit has no future to
bootstrap from. We compute GAE ourselves so you can see it; the helper is at the
bottom of the code block.

## The trainer surface

`PPOConfig` is where the on-policy knobs live. The important ones:

- `learning_rate`, `n_epochs`, `batch_size` — optimization.
- `gamma`, `gae_lambda` — credit assignment (must match how you compute
  advantages).
- `clip_range` — the PPO trust-region clip (the same `0.2` from Tutorial 1).
- `vf_coef`, `ent_coef` — value-loss and entropy weights.
- `max_grad_norm` — gradient clipping for stability.

You build a `PPOConfig`, wrap your policy and a `BurnOptimizer` in a
`PPOTrainerBurn`, then call `train_step` once per rollout. See
[PPO_BEST_PRACTICES.md](../PPO_BEST_PRACTICES.md) for how to tune these.

## The code

This runs a short training loop (small budget so it's fast in CI; bump
`TOTAL_TIMESTEPS` for a real run). It is a doc-test, so it always compiles
against the current API.

```rust
use burn::{
    backend::{Autodiff, NdArray},
    optim::AdamConfig,
    tensor::{Int, Tensor, TensorData},
};
use thrust_rl::prelude::*;
use thrust_rl::train::optimizer::BurnOptimizer;

type Backend = Autodiff<NdArray<f32>>;

// --- Hyperparameters -------------------------------------------------------
const NUM_ENVS: usize = 16; // parallel CartPole copies
const NUM_STEPS: usize = 256; // rollout length per env
const TOTAL_TIMESTEPS: usize = 8_192; // tiny for CI; use ~200_000 for real
const LEARNING_RATE: f64 = 3e-4;
const HIDDEN_DIM: usize = 128;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;
const SEED: u64 = 0; // seed policy init + resets for reproducibility

let device = Default::default();

// --- Environment -----------------------------------------------------------
let probe = CartPole::new();
let obs_dim = probe.observation_space().shape[0];
let action_dim = match probe.action_space().space_type {
    SpaceType::Discrete(n) => n,
    SpaceType::Box => panic!("CartPole is discrete"),
};
let mut env_pool = EnvPool::new(CartPole::new, NUM_ENVS);

// --- Policy: a seeded 2-layer ReLU MLP with orthogonal init ----------------
// Seeding makes the run reproducible, which is what lets a learning-curve CSV
// be overlaid across algorithms on the same axis (see below).
let policy_config = MlpBurnConfig {
    num_layers: 2,
    hidden_dim: HIDDEN_DIM,
    use_orthogonal_init: true,
    activation: BurnActivation::ReLU,
    seed: Some(SEED),
};
let policy = MlpBurnPolicy::<Backend>::with_config(obs_dim, action_dim, policy_config, &device);

// --- Optimizer + trainer ---------------------------------------------------
let inner_opt = AdamConfig::new().init();
let burn_opt: BurnOptimizer<Backend, MlpBurnPolicy<Backend>, _> =
    BurnOptimizer::new(inner_opt, LEARNING_RATE);

let ppo_config = PPOConfig::new()
    .learning_rate(LEARNING_RATE)
    .n_epochs(10)
    .batch_size(128)
    .gamma(GAMMA as f64)
    .gae_lambda(GAE_LAMBDA as f64)
    .clip_range(0.2)
    .clip_range_vf(0.2)
    .vf_coef(0.5)
    .ent_coef(0.01)
    .max_grad_norm(0.5)
    // Disable KL early-stop for a short, small-budget run.
    .target_kl(1.0);

let mut trainer = PPOTrainerBurn::new(ppo_config, policy, burn_opt).unwrap();

let cap = NUM_STEPS * NUM_ENVS;
let num_updates = TOTAL_TIMESTEPS / cap;

// Rollout buffers, reused each update.
let mut buf_obs: Vec<f32> = Vec::with_capacity(cap * obs_dim);
let mut buf_actions: Vec<i64> = Vec::with_capacity(cap);
let mut buf_log_probs: Vec<f32> = Vec::with_capacity(cap);
let mut buf_values: Vec<f32> = Vec::with_capacity(cap);
let mut buf_rewards: Vec<f32> = Vec::with_capacity(cap);
let mut buf_dones: Vec<f32> = Vec::with_capacity(cap);

let mut observations = env_pool.reset();
let mut episode_lengths = [0u32; NUM_ENVS];
let mut completed: Vec<u32> = Vec::new();

for _update in 0..num_updates {
    buf_obs.clear();
    buf_actions.clear();
    buf_log_probs.clear();
    buf_values.clear();
    buf_rewards.clear();
    buf_dones.clear();

    // --- Phase 1: rollout -------------------------------------------------
    for _step in 0..NUM_STEPS {
        let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let obs_t: Tensor<Backend, 2> =
            Tensor::from_data(TensorData::new(obs_flat, [NUM_ENVS, obs_dim]), &device);
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

            episode_lengths[env_id] += 1;
            observations[env_id] = results[env_id].observation.clone();
            if done {
                completed.push(episode_lengths[env_id]);
                trainer.increment_episodes(1);
                episode_lengths[env_id] = 0;
                observations[env_id] = env_pool.reset_env(env_id).unwrap();
            }
        }
    }

    // --- Phase 2: GAE (needs a value bootstrap for the final observation) --
    let last_obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
    let last_obs_t: Tensor<Backend, 2> =
        Tensor::from_data(TensorData::new(last_obs_flat, [NUM_ENVS, obs_dim]), &device);
    let (_, _, last_values) = trainer.policy().get_action_host(last_obs_t);

    let (advantages, returns) = compute_gae(
        &buf_rewards, &buf_values, &buf_dones, &last_values, GAMMA, GAE_LAMBDA, NUM_STEPS, NUM_ENVS,
    );

    // --- Phase 3: hand the batch to the trainer ---------------------------
    let batch = cap;
    let obs_b: Tensor<Backend, 2> =
        Tensor::from_data(TensorData::new(buf_obs.clone(), [batch, obs_dim]), &device);
    let actions_b: Tensor<Backend, 1, Int> =
        Tensor::from_data(TensorData::new(buf_actions.clone(), [batch]), &device);
    let old_log_probs_b: Tensor<Backend, 1> =
        Tensor::from_data(TensorData::new(buf_log_probs.clone(), [batch]), &device);
    let old_values_b: Tensor<Backend, 1> =
        Tensor::from_data(TensorData::new(buf_values.clone(), [batch]), &device);
    let advantages_b: Tensor<Backend, 1> =
        Tensor::from_data(TensorData::new(advantages, [batch]), &device);
    let returns_b: Tensor<Backend, 1> =
        Tensor::from_data(TensorData::new(returns, [batch]), &device);

    // The trainer runs n_epochs of minibatched clipped-surrogate updates and
    // returns per-update stats. The closure tells it how to score
    // (obs, action) batches with the current policy.
    let _stats = trainer
        .train_step(
            obs_b,
            actions_b,
            old_log_probs_b,
            old_values_b,
            advantages_b,
            returns_b,
            |p, o, a| p.evaluate_actions(o, a),
        )
        .unwrap();
}

// A short CI run won't fully solve CartPole, but the trainer should have run.
// We assert only that the loop executed, to keep the doc-test fast and
// deterministic.
assert!(num_updates >= 1);
let _ = completed;

/// Per-env GAE. `rewards`, `values`, `dones` are flat `[T * N]` row-major
/// (step-major: index = step * num_envs + env_id). `last_values[n]` is the
/// value bootstrap for env `n` at the step just past the end of the rollout.
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
    // Walk the rollout in reverse per env so each advantage can fold in the
    // (already-computed) advantage of the following step.
    for t in (0..num_steps).rev() {
        for n in 0..num_envs {
            let idx = t * num_envs + n;
            let next_value = if t == num_steps - 1 {
                last_values[n]
            } else {
                values[(t + 1) * num_envs + n]
            };
            // If the episode ended at this step, don't bootstrap across the
            // boundary — the next state belongs to a fresh episode.
            let next_nonterminal = 1.0 - dones[idx];
            let delta = rewards[idx] + gamma * next_value * next_nonterminal - values[idx];
            last_gae[n] = delta + gamma * gae_lambda * next_nonterminal * last_gae[n];
            advantages[idx] = last_gae[n];
            returns[idx] = advantages[idx] + values[idx];
        }
    }
    (advantages, returns)
}
```

## Reading the learning curve

The packaged `train_cartpole_modern` example is the same loop with a bigger
budget and one extra feature: set `CURVE_CSV` and it writes one
`env_steps,mean_episode_reward` row per update.

```bash
TOTAL_TIMESTEPS=200000 CURVE_CSV=/tmp/ppo.csv \
    cargo run --release --features training --example train_cartpole_modern
```

Because CartPole's reward is `+1`/step, `mean_episode_reward` **is** the mean
episode length. Plot `env_steps` (x) against `mean_episode_reward` (y) and you
should see it climb off the ~22 random baseline and head toward 500. What to
look for:

- **A rising curve** = credit assignment is working.
- **A curve stuck near ~22** = something is off (a common cause is a
  `gamma`/`gae_lambda` mismatch between `PPOConfig` and your `compute_gae`
  call — they must agree).
- **A curve that climbs then crashes** = the policy collapsed; lower the
  learning rate or the entropy coefficient, or re-enable `target_kl`.

The seed makes runs reproducible, so two algorithms (say PPO here and A2C from
the `train_cartpole_a2c` example) can be overlaid on the same env, seed, and
budget for an honest comparison.

## Try it yourself

- **Mismatch the discount on purpose:** pass `gamma = 0.0` to `compute_gae`
  while leaving `PPOConfig::gamma(0.99)` — the returns the critic is trained on
  no longer match the advantages, and learning stalls. Concrete proof the two
  must agree.
- **Shrink the pool:** set `NUM_ENVS = 1`. Fewer parallel envs means noisier
  gradients per update; you'll need more updates to reach the same bar.
- **Full run:** drop the CI-sized `TOTAL_TIMESTEPS` and run the packaged
  example above with the CSV, then plot it.

## Next

The path splits from here — see the [tutorial index](README.md) for the full
dependency-ordered series. Next up (Tutorial 3) is **off-policy training with
DQN**: replay buffers, target networks, and when to reach for DQN over PPO.
