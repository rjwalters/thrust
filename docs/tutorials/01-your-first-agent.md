# Tutorial 1 — Your first agent

> **You will:** train an actor-critic policy to solve `SimpleBandit` from
> scratch, writing the full rollout → loss → update loop by hand in ~60 lines.
> **Prerequisites:** a Rust nightly toolchain (see
> [GETTING_STARTED.md](../GETTING_STARTED.md)). No GPU, no libtorch — the
> default backend is pure-Rust CPU.
> **Time:** ~10 minutes, including the training run.

This is the "hello world" of Thrust. We pick the simplest possible learning
problem so that nothing hides behind a trainer abstraction: you will see every
tensor, every gradient step, and exactly where the learning signal comes from.
Later tutorials hand the boilerplate to a `PPOTrainerBurn`; here we do it by
hand so the loop holds no mysteries.

## The problem: a contextual bandit

`SimpleBandit` is deliberately trivial:

- The **state** is a single bit, `0.0` or `1.0`, drawn fresh each step.
- There are **two actions**, `0` and `1`.
- The **reward** is `1.0` if `action == state`, else `0.0`.
- Every episode is **one step long**.

The optimal policy is just `action = state`, and a perfect agent scores a
mean reward of `1.0`. We chose this because it is a *contextual bandit*, not a
sequential task: each state is independent of the last, so there is **no credit
assignment across time**. That property lets us strip the advantage estimator
down to its simplest form, which we'll explain when we get there.

## The pieces

Three types do the work, all re-exported from the crate's prelude:

- `SimpleBandit` — the environment (implements the `Environment` trait).
- `EnvPool` — a vectorized wrapper that steps *N* copies of the env at once, so
  each rollout gathers a healthy batch of independent transitions.
- `MlpBurnPolicy` — a small multi-layer perceptron with two heads: an **actor**
  (action logits) and a **critic** (a scalar state-value estimate). This is the
  actor-critic architecture that underpins every on-policy algorithm in Thrust.

We drive the Burn tensor framework directly. The one piece of Burn ceremony to
know up front: its optimizer has a **move-in, move-out** ownership model —
`optim.step(...)` *consumes* the policy and returns the updated copy. There is
no in-place update. We keep the policy in an `Option` and swap it through
`take().unwrap() → step() → Some(...)` each iteration. This surprises everyone
the first time; it is not a bug.

## The loop

Every on-policy RL update is three phases:

1. **Rollout** — run the current policy in the env and record what happened
   (observations, actions, the log-probability of each action, the critic's
   value estimate, and the reward). No gradients here.
2. **Advantage** — turn rewards into a learning signal: *how much better than
   expected* was each action? For a bandit that is simply `advantage = reward −
   value`. There is no future to bootstrap from, so — unlike CartPole in
   Tutorial 2 — we deliberately do **not** discount or run GAE. Bootstrapping
   from the *next* state's value would only inject noise here, because the next
   state is random and unrelated.
3. **Update** — nudge the actor to make high-advantage actions more likely (a
   clipped PPO surrogate) and train the critic to predict the reward (MSE),
   plus a small entropy bonus to keep exploring.

Here is the whole thing. It compiles and runs as-is (it is a doc-test in the
crate, so CI keeps it honest):

```rust
use burn::{
    backend::{Autodiff, NdArray},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Int, Tensor, TensorData},
};
use thrust_rl::prelude::*;

// The backend is chosen at compile time. `NdArray<f32>` is Burn's pure-Rust
// CPU backend; `Autodiff<...>` wraps it to record gradients. Swapping in a GPU
// backend is a one-line type change plus a Cargo feature — see BURN_BACKENDS.md.
type Backend = Autodiff<NdArray<f32>>;

// --- Hyperparameters -------------------------------------------------------
const NUM_ENVS: usize = 4; // parallel bandit copies per rollout
const NUM_STEPS: usize = 100; // steps collected per env per rollout
const TOTAL_TIMESTEPS: usize = 4_000; // small so this runs fast in CI
const LEARNING_RATE: f64 = 1e-3;
const N_EPOCHS: usize = 10; // gradient passes over each rollout
const CLIP_RANGE: f32 = 0.2; // PPO surrogate clip
const VF_COEF: f32 = 0.5; // weight on the value (critic) loss
const ENT_COEF: f32 = 0.1; // weight on the entropy bonus
const HIDDEN_DIM: usize = 64;

let device = Default::default();

// --- Environment -----------------------------------------------------------
// Probe one env for its dimensions, then build a pool of NUM_ENVS copies.
let probe = SimpleBandit::new();
let obs_dim = probe.observation_space().shape[0];
let action_dim = match probe.action_space().space_type {
    SpaceType::Discrete(n) => n,
    SpaceType::Box => panic!("SimpleBandit is discrete"),
};
let mut env_pool = EnvPool::new(SimpleBandit::new, NUM_ENVS);

// --- Policy + optimizer ----------------------------------------------------
// The policy lives in an Option because Burn's optimizer moves it through
// `step()`. `optim` is Adam; we pass the learning rate to `step`, not the
// config, so it can be scheduled later if you want.
let mut policy: Option<MlpBurnPolicy<Backend>> =
    Some(MlpBurnPolicy::new(obs_dim, action_dim, HIDDEN_DIM, &device));
let mut optim = AdamConfig::new().init();

// --- Rollout buffers (host-side Vecs, reused each iteration) ---------------
let cap = NUM_STEPS * NUM_ENVS;
let mut buf_obs: Vec<f32> = Vec::with_capacity(cap * obs_dim);
let mut buf_actions: Vec<i64> = Vec::with_capacity(cap);
let mut buf_log_probs: Vec<f32> = Vec::with_capacity(cap);
let mut buf_values: Vec<f32> = Vec::with_capacity(cap);
let mut buf_rewards: Vec<f32> = Vec::with_capacity(cap);

let mut observations = env_pool.reset();
let num_updates = TOTAL_TIMESTEPS / cap;

let mut total_reward = 0.0_f64;
let mut total_steps = 0_usize;

for _update in 0..num_updates {
    buf_obs.clear();
    buf_actions.clear();
    buf_log_probs.clear();
    buf_values.clear();
    buf_rewards.clear();

    // --- Phase 1: rollout (no gradients) ----------------------------------
    for _step in 0..NUM_STEPS {
        let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let obs_t: Tensor<Backend, 2> =
            Tensor::from_data(TensorData::new(obs_flat, [NUM_ENVS, obs_dim]), &device);

        // `get_action_host` samples an action from the actor, and returns the
        // sampled actions, their log-probs, and the critic's value estimates —
        // all as plain host Vecs, so no gradient graph is retained.
        let (actions, log_probs, values) = policy.as_ref().unwrap().get_action_host(obs_t);

        let results = env_pool.step(&actions);

        for env_id in 0..NUM_ENVS {
            buf_obs.extend_from_slice(&observations[env_id]);
            buf_actions.push(actions[env_id]);
            buf_log_probs.push(log_probs[env_id]);
            buf_values.push(values[env_id]);
            buf_rewards.push(results[env_id].reward);

            total_reward += results[env_id].reward as f64;
            total_steps += 1;

            observations[env_id] = results[env_id].observation.clone();
            // One-step episodes: every step ends the episode, so reset.
            if results[env_id].terminated || results[env_id].truncated {
                observations[env_id] = env_pool.reset_env(env_id).unwrap();
            }
        }
    }

    // --- Phase 2: advantages (A = reward − value) -------------------------
    // No discounting, no GAE: SimpleBandit is a contextual bandit, so the only
    // signal is "did this action beat the critic's expectation?".
    let advantages: Vec<f32> = buf_rewards
        .iter()
        .zip(buf_values.iter())
        .map(|(r, v)| r - v)
        .collect();
    let returns: Vec<f32> = buf_rewards.clone();

    // Normalize advantages to zero mean / unit std — a standard PPO trick that
    // keeps the gradient scale stable across updates.
    let mean = advantages.iter().sum::<f32>() / advantages.len() as f32;
    let var = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / advantages.len() as f32;
    let std = (var + 1e-8).sqrt();
    let advantages: Vec<f32> = advantages.iter().map(|a| (a - mean) / std).collect();

    // --- Phase 3: PPO update (N_EPOCHS full-batch passes) -----------------
    let batch = advantages.len();
    let obs_b: Tensor<Backend, 2> =
        Tensor::from_data(TensorData::new(buf_obs.clone(), [batch, obs_dim]), &device);
    let actions_b: Tensor<Backend, 1, Int> =
        Tensor::from_data(TensorData::new(buf_actions.clone(), [batch]), &device);
    let old_log_probs_b: Tensor<Backend, 1> =
        Tensor::from_data(TensorData::new(buf_log_probs.clone(), [batch]), &device);
    let adv_b: Tensor<Backend, 1> =
        Tensor::from_data(TensorData::new(advantages, [batch]), &device);
    let returns_b: Tensor<Backend, 1> =
        Tensor::from_data(TensorData::new(returns, [batch]), &device);

    for _epoch in 0..N_EPOCHS {
        // Move the policy out, evaluate the *current* actor/critic on the
        // stored batch, and get back new log-probs, per-sample entropy, and
        // fresh value predictions.
        let p = policy.take().unwrap();
        let (new_log_probs, entropy, values_pred) =
            p.evaluate_actions(obs_b.clone(), actions_b.clone());

        // Clipped PPO surrogate: the ratio of new-to-old action probability,
        // clamped so a single update can't move the policy too far.
        let ratio = (new_log_probs - old_log_probs_b.clone()).exp();
        let unclipped = ratio.clone() * adv_b.clone();
        let clipped = ratio.clamp(1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * adv_b.clone();
        let policy_loss = -unclipped.min_pair(clipped).mean();

        // Critic loss: predict the return (here, the reward) via MSE.
        let value_diff = values_pred - returns_b.clone();
        let value_loss = (value_diff.clone() * value_diff).mean();

        // Entropy bonus: subtracting entropy from the loss *encourages* it,
        // keeping the actor exploring instead of collapsing prematurely.
        let entropy_mean = entropy.mean();

        let loss =
            policy_loss + value_loss.mul_scalar(VF_COEF) - entropy_mean.mul_scalar(ENT_COEF);

        // Burn's two-step gradient dance: backward() produces raw grads, then
        // `from_grads` ties each grad to its parameter, then `step` consumes
        // the policy + grads and returns the updated policy.
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &p);
        policy = Some(optim.step(LEARNING_RATE, p, grads));
    }
}

// After training, the running mean reward should be well above the 0.5
// random-chance baseline. (We assert a loose bar so the doc-test is robust.)
let success_rate = total_reward / total_steps as f64;
assert!(
    success_rate > 0.5,
    "expected better-than-random after training, got {success_rate:.3}"
);
```

## What just happened

The running success rate climbs from ~0.5 (random) toward ~1.0 as the actor
learns `action = state`. If you print it each update (the runnable
`train_simple_bandit` example does), you'll watch it converge within a dozen
updates and then the entropy collapses — which is *correct* here, because the
optimal policy is deterministic. In a sequential task you'd want to keep some
entropy for exploration; a bandit has nothing left to explore once it's solved.

The three-phase structure — rollout, advantage, update — is the skeleton of
every on-policy algorithm in this library. Tutorial 2 keeps the exact same
shape but swaps the by-hand loop for `PPOTrainerBurn` and adds real temporal
credit assignment (GAE) for a task where the future actually matters.

## Try it yourself

- **Break it on purpose:** set `ENT_COEF = 0.0` and `NUM_STEPS = 8`. With a
  tiny batch and no exploration pressure the policy can lock onto a wrong action
  early. This is the failure mode entropy bonuses exist to prevent.
- **Watch it live:** run the packaged example, which logs every 10 updates:
  ```bash
  cargo run --release --features training --example train_simple_bandit
  ```
- **Bootstrap on purpose (and regret it):** change the advantage to discount
  future value (`gamma = 0.99`) and watch learning *degrade* — concrete proof
  that GAE is wrong for bandits. The [SimpleBandit training
  guide](../TRAINING_SIMPLEBANDIT.md) has the full analysis.

## Next

[Tutorial 2 — Solving CartPole with PPO](02-cartpole-ppo.md): a real sequential
task, the `PPOTrainerBurn`, GAE, and reading learning curves.
