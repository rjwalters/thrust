# Tutorial 5 — Memory and POMDPs with recurrent PPO

> **You will:** train a recurrent (LSTM) PPO agent on `FlickeringCartPole` — a
> version of CartPole whose observation is randomly blanked to zeros — and watch
> it beat a memoryless MLP baseline that sees the *identical* stream. Along the
> way you'll learn what a POMDP is, when memory is worth its cost, and how a
> recurrent rollout differs from the feedforward one from Tutorial 2.
> **Prerequisites:** [Tutorial 2](02-cartpole-ppo.md) — you should be
> comfortable with the rollout → GAE → `train_step` loop, `EnvPool`, and the
> `PPOConfig` surface. You do **not** need the off-policy tutorials (3 and 4).
> **Time:** ~20 minutes.

Every tutorial so far has quietly assumed something powerful: that the
observation the agent receives *is* the full state of the world. CartPole hands
you `[x, x_dot, theta, theta_dot]` — cart position and velocity, pole angle and
angular velocity — which is literally everything the physics needs to predict
the next step. A policy that maps the current observation to an action can be
optimal, because the current observation misses nothing. This is a **fully
observable** Markov Decision Process (MDP), and a memoryless (feedforward)
policy is all you ever need.

The real world is rarely so kind. Sensors drop frames, occlude, and lag; a robot
sees a slice of its surroundings, not a god's-eye state vector. When the
observation is a *partial, noisy shadow* of the true state, you have a
**Partially Observable MDP (POMDP)**, and a memoryless policy is no longer
enough — the information it needs to act well isn't in the current frame. It has
to be *remembered* from earlier ones. That is what this tutorial is about.

## The problem: making CartPole partially observable

CartPole-v1 is not a POMDP, so we have to *make* it one. The trick is to keep the
physics identical but corrupt the observation channel. `FlickeringCartPole` does
exactly this: on each step, with probability `p` (default `0.5`) it **blanks the
entire observed frame to all-zeros** before handing it to the agent. The cart is
still there, the pole is still falling — the agent just can't see it this step.

This is the canonical Atari-POMDP protocol from **Hausknecht & Stone (2015)**,
"Deep Recurrent Q-Learning for Partially Observable MDPs." Blanking the whole
frame (rather than adding noise) is a clean, brutal partial-observability signal:
half the time, on average, the policy is acting *blind*.

Why does this make memory load-bearing **by construction**? Consider a
feedforward policy on a blanked frame. Its input is `[0, 0, 0, 0]` — a state
CartPole physics never produces on its own. It has no idea whether the pole is
tipping left or right, fast or slow. With no memory it can only output a fixed,
uninformed action, and it will drop the pole. A recurrent policy, by contrast,
carries a **hidden state** across the blank: it saw the pole tipping right two
frames ago, integrates that with its own action, and keeps balancing through the
gap. Memory is not a nice-to-have here — it is the only way to recover the
hidden state the observation withholds.

## Why velocity-masking didn't work (a documented negative result)

The obvious first idea for a CartPole POMDP is not flickering at all — it's to
**hide the velocities**, showing the agent only `[x, theta]` (position and
angle). That *feels* like it should force the agent to infer velocity from a
history of positions, which is exactly the kind of thing memory is for.

It doesn't work, and the failure is instructive. Thrust ships `MaskedCartPole`
as a **documented negative result**: a real 500k-step run disproved it as a
POMDP. A *memoryless* reactive controller on `[x, theta]` solves it outright and
actually **beats** the LSTM (MLP mean return ~324 vs. LSTM ~222). The reason is
that a proportional angle/position feedback loop — "pole leaning right, push
right" — balances CartPole *without ever estimating velocity*. Masking the
velocities removed information the optimal policy didn't need, so memory bought
nothing and just added variance.

The lesson: **partial observability is only real if the withheld information is
actually required to act well.** Flickering closes the loophole that
velocity-masking left open. On a blanked frame there is *no* reactive feedback to
fall back on — position and angle are gone too — so the only way through the gap
is to remember. That's the difference between a task that merely *looks* like a
POMDP and one that is a POMDP by construction.

## When an LSTM earns its cost

Memory isn't free. An LSTM policy costs you, concretely:

- **More parameters and compute.** The recurrent trunk runs a gated update every
  step; training does backpropagation-through-time (BPTT) over whole
  trajectories, not independent transitions.
- **Bookkeeping.** You have to thread the hidden state `(h, c)` across rollout
  steps, zero it at episode boundaries, and keep the training-time forward pass
  consistent with the behavior-time one (more on this below). Get any of that
  wrong and the PPO ratio silently corrupts.
- **Touchier optimization.** Reusing a recurrent rollout across many epochs
  amplifies off-policy drift through the hidden state, so recurrent PPO wants
  *fewer* epochs and more careful gradient clipping than feedforward PPO.

So when does that cost pay off? **Exactly when the task is a genuine POMDP** —
when the information needed to act well is spread across time and missing from
any single frame. If a memoryless baseline already solves your task (as on
`MaskedCartPole`, or plain CartPole), an LSTM is pure overhead and often *worse*.
The honest way to know is to run both arms on the identical stream and compare —
which is exactly what the code below does.

## How the LSTM handles a rollout

In the feedforward loop from Tutorial 2, each step was independent: observation
in, action out, nothing carried between steps. The recurrent loop threads a
hidden state through:

- At each step you call `policy.forward_step(obs, state)`, which returns the
  action logits, a value estimate, **and a new hidden state** to feed into the
  next step. Passing `None` for the state starts from zeros.
- **On episode end, you zero the hidden state** for that env. A fresh episode
  shares no history with the one that just ended, so its memory must start
  blank — otherwise stale state leaks across the boundary and poisons the new
  episode.
- **At the boundary between rollout windows, this simple version resets the
  collection state to zeros** rather than carrying it across. This is a
  deliberate BPTT truncation: the training-time forward pass (`evaluate_sequences`)
  always reconstructs the hidden state from zeros at the start of each window, so
  the behavior policy that collected the data must do the same. If it didn't, the
  `old_log_probs` stored during collection (computed under a warm state) would be
  inconsistent with the `log_probs` recomputed during training (from a cold
  state), and the PPO importance ratio would be garbage. Keeping both cold is the
  cheap, correct choice — Thrust's design note calls this "Strategy A."

There is one subtlety worth naming because it looks like a bug but isn't. The
hidden-state reset and the GAE bootstrap use **different** notions of "episode
over." GAE stops bootstrapping only on `terminated` (the pole actually fell); the
hidden-state reset (`episode_starts`) fires on `terminated || truncated` (fell
*or* hit the 500-step time limit). A truncated episode isn't a real terminal
state — the pole was still up — so its value *should* bootstrap, but its memory
*must* still reset because the next episode is unrelated. The buffer and policy
handle this asymmetry internally; you just need to know it's intentional.

## Recurrent training vs. feedforward training

Four things change from the Tutorial 2 PPO loop. If you internalize these, the
code reads as "the PPO loop you know, with memory bolted on":

1. **The buffer is a `RecurrentRolloutBuffer`.** It stores transitions
   *trajectory-major* (`[n_env][T]`) instead of flat, because the training
   forward needs whole temporally-intact sequences, not a shuffled bag of
   timesteps. It has an extra `add_recurrent_state(step, env, h, c)` alongside
   the usual `add(...)`.
2. **The trainer is a `RecurrentPPOTrainer`, and it takes a `device`.** The
   feedforward trainer never materializes tensors on its own; the recurrent one
   has to build rank-3 `[n_env, T, obs_dim]` sequence batches internally, so it
   needs a device handle.
3. **You score sequences, not timesteps.** The training closure calls
   `policy.evaluate_sequences(obs_seq, actions, initial_state, episode_starts)`
   over rank-3 batches, replacing the rank-2 `evaluate_actions` from Tutorial 2.
4. **Minibatches are whole trajectories.** You pass `envs_per_minibatch` (how
   many env-trajectories per minibatch) instead of `batch_size` (loose
   timesteps). You cannot shuffle individual timesteps without destroying the
   sequences the LSTM has to replay.

One more load-bearing detail that isn't about recurrence at all:
**observation normalization.** CartPole's raw observations are tiny early on
(initial perturbations ~0.05). An LSTM's tanh gates squash such small inputs to
near-zero features, starving both the policy and value heads of gradient. So the
code below standardizes each *visible* coordinate to unit scale with a running
mean/std (a `VecNormalize`-style wrapper). The catch: a **blanked frame is
exactly all-zeros**, and must be passed through unchanged and excluded from the
running statistics — otherwise you'd fold zeros into the mean, pollute the
stats, and turn the "no observation" signal into a recognizable non-zero
constant. That's why the code inlines a small `ObsNormalizer` with a
skip-blanked-frames special case rather than using the library's plain
`RunningMeanStd`.

## The code

This trains **both** arms — the LSTM and the memoryless MLP — on the identical
flickering stream (same seed, same normalizer), then prints the contrast. The
budget is tiny so it runs fast in CI as a doc-test; bump `TOTAL_TIMESTEPS` (and
run the packaged example) for a real result. A short run won't solve the POMDP,
so the assertion at the end only checks that both loops executed.

```rust
use std::sync::atomic::{AtomicU64, Ordering};

use burn::{
    backend::{Autodiff, NdArray},
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

type Backend = Autodiff<NdArray<f32>>;

// --- Hyperparameters (CI budget) -------------------------------------------
const NUM_ENVS: usize = 4; // parallel FlickeringCartPole copies
const NUM_STEPS: usize = 32; // rollout window length per env
const TOTAL_TIMESTEPS: usize = 4_096; // tiny for CI; use ~500_000 for the real run
const HIDDEN_DIM: usize = 64;
const LSTM_LR: f64 = 1.5e-3; // the POMDP wants a higher LR than plain CartPole
const MLP_LR: f64 = 3e-4; // standard CartPole PPO learning rate
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;
const N_EPOCHS: usize = 4; // low: recurrent rollouts drift off-policy fast
const ENVS_PER_MINIBATCH: usize = 2; // whole trajectories per minibatch
const FLICKER_PROB: f64 = 0.5; // blank half the frames, on average
const SEED: u64 = 0;

// CartPole's +1/step reward gives discounted returns of magnitude ~40-60.
// Against targets that large the clipped value loss (clip_range_vf) can only
// move ~0.2 per iteration and the critic freezes; scaling rewards to O(1)
// returns fixes it at the source. Reported returns stay in RAW units.
const REWARD_SCALE: f32 = 0.02;

// Train both arms on the IDENTICAL flickering stream, then print the contrast.
let (lstm_final, lstm_best) = run_lstm();
let (mlp_final, mlp_best) = run_mlp();

println!("FlickeringCartPole (p = {FLICKER_PROB}) — mean return of last <=100 episodes:");
println!("  LSTM (recurrent)   : final {lstm_final:.1}  best {lstm_best:.1}");
println!("  MLP  (feedforward) : final {mlp_final:.1}  best {mlp_best:.1}");
// With a real budget the LSTM clears the ~195 "solved" bar and the MLP plateaus
// well below it. The CI budget is far too small to solve the task, so we only
// assert both loops ran.
assert!(lstm_best >= 0.0 && mlp_best >= 0.0);

/// Build an `EnvPool` with per-env seeded flicker streams so the schedule is
/// reproducible yet decorrelated across the pool. Both arms call this with the
/// same seed, so they see the *identical* flicker stream — the only difference
/// between the runs is the policy class (memory vs. no memory).
fn make_pool() -> EnvPool<FlickeringCartPole> {
    let ctr = AtomicU64::new(SEED.wrapping_mul(1000));
    EnvPool::new(
        || {
            let s = ctr.fetch_add(1, Ordering::Relaxed);
            FlickeringCartPole::with_seed_and_probability(s, FLICKER_PROB)
        },
        NUM_ENVS,
    )
}

/// Probe an env for its observation and (discrete) action dimensions.
fn env_dims() -> (usize, usize) {
    let probe = FlickeringCartPole::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        SpaceType::Discrete(n) => n,
        _ => panic!("FlickeringCartPole is discrete"),
    };
    (obs_dim, action_dim)
}

/// Train the recurrent LSTM policy. Returns `(final_mean, best_mean)` return
/// over the last <=100 completed episodes (final = end of run, best = peak of
/// the moving average).
fn run_lstm() -> (f32, f32) {
    let device = Default::default();
    let (obs_dim, action_dim) = env_dims();
    let mut env_pool = make_pool();

    // Seeded LSTM policy: obs -> (action logits, value), carrying a hidden state.
    let policy_config =
        LstmBurnConfig { hidden_dim: HIDDEN_DIM, ..Default::default() }.with_seed(SEED);
    let policy =
        LstmBurnPolicy::<Backend>::with_config(obs_dim, action_dim, policy_config, &device);

    let inner_opt = AdamConfig::new().init();
    let burn_opt: BurnOptimizer<Backend, LstmBurnPolicy<Backend>, _> =
        BurnOptimizer::new(inner_opt, LSTM_LR);

    let ppo_config = PPOConfig::new()
        .learning_rate(LSTM_LR)
        .n_epochs(N_EPOCHS)
        .gamma(GAMMA as f64)
        .gae_lambda(GAE_LAMBDA as f64)
        .clip_range(0.2)
        // Load-bearing WITH reward scaling: a meaningful value clip keeps the
        // critic tracking its (now O(1)) targets instead of freezing.
        .clip_range_vf(0.2)
        .vf_coef(0.5)
        .ent_coef(0.01)
        // Load-bearing for the recurrent trunk: without it, value-loss spikes
        // wipe out the shared LSTM's policy features.
        .max_grad_norm(0.5)
        .target_kl(1.0); // disable KL early-stop for a short run

    // The recurrent trainer takes a `device` (it materializes rank-3 batches).
    let mut trainer = RecurrentPPOTrainer::new(ppo_config, policy, burn_opt, device)
        .expect("valid PPO config");

    let num_updates = TOTAL_TIMESTEPS / (NUM_STEPS * NUM_ENVS);

    let mut norm = ObsNormalizer::new(obs_dim);
    let mut buffer = RecurrentRolloutBuffer::new(NUM_STEPS, NUM_ENVS, obs_dim, HIDDEN_DIM);
    let mut observations: Vec<Vec<f32>> =
        env_pool.reset().iter().map(|o| norm.normalize(o)).collect();
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut episode_returns = [0.0_f32; NUM_ENVS];
    let mut completed: Vec<f32> = Vec::new();
    // Collection-time recurrent state (None => zeros). Reset to None at each
    // window boundary to stay consistent with the training forward (Strategy A).
    let mut lstm_state: Option<LstmState<Backend, 2>> = None;
    // Host copy of the state ENTERING the current step, recorded into the buffer.
    let mut entering_h = vec![0.0_f32; NUM_ENVS * HIDDEN_DIM];
    let mut entering_c = vec![0.0_f32; NUM_ENVS * HIDDEN_DIM];
    // Host copy of the state exiting the last collected step (for warm-start).
    let mut last_h = vec![0.0_f32; NUM_ENVS * HIDDEN_DIM];
    let mut last_c = vec![0.0_f32; NUM_ENVS * HIDDEN_DIM];
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

            // Thread the hidden state through the step.
            let (logits, values_t, new_state) =
                trainer.policy().forward_step(obs_t, lstm_state.take());
            let (actions, log_probs, values_host) = sample_actions(&logits, &values_t, &mut rng);
            let results = env_pool.step(&actions);

            // Pull the state EXITING this step to the host so we can zero the
            // rows of envs whose episode just ended.
            let mut h_host: Vec<f32> = new_state.hidden.into_data().to_vec().unwrap();
            let mut c_host: Vec<f32> = new_state.cell.into_data().to_vec().unwrap();

            for env in 0..NUM_ENVS {
                let r = &results[env];
                buffer.add(
                    step,
                    env,
                    &observations[env],
                    actions[env],
                    r.reward * REWARD_SCALE, // train on scaled reward; report raw
                    values_host[env],
                    log_probs[env],
                    r.terminated,
                    r.truncated,
                );
                // Record the state that ENTERED this step (Strategy-A hook; the
                // training forward recomputes from zeros, but the buffer stores
                // it for completeness and future strategies).
                buffer.add_recurrent_state(
                    step,
                    env,
                    &entering_h[env * HIDDEN_DIM..(env + 1) * HIDDEN_DIM],
                    &entering_c[env * HIDDEN_DIM..(env + 1) * HIDDEN_DIM],
                );

                episode_returns[env] += r.reward;
                observations[env] = norm.normalize(&r.observation);

                if r.terminated || r.truncated {
                    completed.push(episode_returns[env]);
                    episode_returns[env] = 0.0;
                    trainer.increment_episodes(1);
                    // Fresh episode => zeroed memory for this env's row.
                    for k in 0..HIDDEN_DIM {
                        h_host[env * HIDDEN_DIM + k] = 0.0;
                        c_host[env * HIDDEN_DIM + k] = 0.0;
                    }
                    observations[env] = norm.normalize(&env_pool.reset_env(env).unwrap());
                }
            }

            // The state exiting this step (post per-env reset) enters the next.
            entering_h.copy_from_slice(&h_host);
            entering_c.copy_from_slice(&c_host);
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

        // Linear LR annealing stabilizes the late-run policy above the bar.
        let frac = 1.0 - (update as f64) / (num_updates.max(1) as f64);
        trainer.set_learning_rate(LSTM_LR * frac.max(0.05));

        // Score whole sequences, not timesteps. `None` = start each window's
        // forward from a zeroed state (Strategy A); `starts` are the per-step
        // episode-boundary flags the LSTM uses to reset its state mid-sequence.
        let _stats = trainer
            .train_step(&buffer, ENVS_PER_MINIBATCH, |p, obs_seq, actions, starts| {
                p.evaluate_sequences(obs_seq, actions, None, starts)
            })
            .expect("recurrent train step");

        // Record the cross-window carry flag, then truncate BPTT: next window's
        // collection restarts from a zeroed state to match the training forward.
        let final_hidden: Vec<Vec<f32>> = (0..NUM_ENVS)
            .map(|e| last_h[e * HIDDEN_DIM..(e + 1) * HIDDEN_DIM].to_vec())
            .collect();
        let final_cell: Vec<Vec<f32>> = (0..NUM_ENVS)
            .map(|e| last_c[e * HIDDEN_DIM..(e + 1) * HIDDEN_DIM].to_vec())
            .collect();
        buffer.seed_warm_start(NUM_STEPS - 1, &final_hidden, &final_cell);
        lstm_state = None;
        entering_h.iter_mut().for_each(|x| *x = 0.0);
        entering_c.iter_mut().for_each(|x| *x = 0.0);

        if !completed.is_empty() {
            let recent = &completed[completed.len().saturating_sub(100)..];
            last_mean = recent.iter().sum::<f32>() / recent.len() as f32;
            best_mean = best_mean.max(last_mean);
        }
    }

    (last_mean, best_mean)
}

/// Train the feedforward MLP baseline on the SAME flickering stream. It cannot
/// act on a blanked frame and has no memory to bridge the gaps, so it plateaus
/// below the LSTM. Returns `(final_mean, best_mean)`.
fn run_mlp() -> (f32, f32) {
    let device = Default::default();
    let (obs_dim, action_dim) = env_dims();
    let mut env_pool = make_pool();

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
        .n_epochs(N_EPOCHS)
        .batch_size(128)
        .gamma(GAMMA as f64)
        .gae_lambda(GAE_LAMBDA as f64)
        .clip_range(0.2)
        .clip_range_vf(0.2)
        .vf_coef(0.5)
        .ent_coef(0.01)
        .max_grad_norm(0.5)
        .target_kl(1.0);

    let mut trainer = PPOTrainerBurn::new(ppo_config, policy, burn_opt).expect("valid PPO config");

    let cap = NUM_STEPS * NUM_ENVS;
    let num_updates = TOTAL_TIMESTEPS / cap;

    // Same normalizer as the LSTM arm so the ONLY difference is the policy class.
    let mut norm = ObsNormalizer::new(obs_dim);
    let mut observations: Vec<Vec<f32>> =
        env_pool.reset().iter().map(|o| norm.normalize(o)).collect();
    let mut episode_returns = [0.0_f32; NUM_ENVS];
    let mut completed: Vec<f32> = Vec::new();
    let mut last_mean = 0.0_f32;
    let mut best_mean = 0.0_f32;

    let mut buf_obs: Vec<f32> = Vec::with_capacity(cap * obs_dim);
    let mut buf_actions: Vec<i64> = Vec::with_capacity(cap);
    let mut buf_log_probs: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_values: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_rewards: Vec<f32> = Vec::with_capacity(cap);
    let mut buf_dones: Vec<f32> = Vec::with_capacity(cap);

    for _update in 0..num_updates {
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

            for env in 0..NUM_ENVS {
                buf_obs.extend_from_slice(&observations[env]);
                buf_actions.push(actions[env]);
                buf_log_probs.push(log_probs[env]);
                buf_values.push(values[env]);
                buf_rewards.push(results[env].reward * REWARD_SCALE);
                let done = results[env].terminated || results[env].truncated;
                buf_dones.push(if done { 1.0 } else { 0.0 });

                episode_returns[env] += results[env].reward;
                observations[env] = norm.normalize(&results[env].observation);
                if done {
                    completed.push(episode_returns[env]);
                    episode_returns[env] = 0.0;
                    trainer.increment_episodes(1);
                    observations[env] = norm.normalize(&env_pool.reset_env(env).unwrap());
                }
            }
        }

        let last_obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
        let last_obs_t = Tensor::<Backend, 2>::from_data(
            TensorData::new(last_obs_flat, [NUM_ENVS, obs_dim]),
            &device,
        );
        let (_, _, last_values) = trainer.policy().get_action_host(last_obs_t);

        let (advantages, returns) = compute_gae(
            &buf_rewards, &buf_values, &buf_dones, &last_values, GAMMA, GAE_LAMBDA, NUM_STEPS,
            NUM_ENVS,
        );

        let obs_b = Tensor::<Backend, 2>::from_data(
            TensorData::new(buf_obs.clone(), [cap, obs_dim]),
            &device,
        );
        let actions_b = Tensor::<Backend, 1, Int>::from_data(
            TensorData::new(buf_actions.clone(), [cap]),
            &device,
        );
        let old_log_probs_b =
            Tensor::<Backend, 1>::from_data(TensorData::new(buf_log_probs.clone(), [cap]), &device);
        let old_values_b =
            Tensor::<Backend, 1>::from_data(TensorData::new(buf_values.clone(), [cap]), &device);
        let advantages_b =
            Tensor::<Backend, 1>::from_data(TensorData::new(advantages, [cap]), &device);
        let returns_b =
            Tensor::<Backend, 1>::from_data(TensorData::new(returns, [cap]), &device);

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
            .expect("train step");

        if !completed.is_empty() {
            let recent = &completed[completed.len().saturating_sub(100)..];
            last_mean = recent.iter().sum::<f32>() / recent.len() as f32;
            best_mean = best_mean.max(last_mean);
        }
    }

    (last_mean, best_mean)
}

/// Host-side categorical sampling from LSTM `forward_step` outputs. `logits` is
/// `[N_env, action_dim]`, `values` is `[N_env]`. Returns
/// `(actions, action_log_probs, values)` as host `Vec`s.
fn sample_actions(
    logits: &Tensor<Backend, 2>,
    values: &Tensor<Backend, 1>,
    rng: &mut StdRng,
) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
    let [n, a] = logits.dims();
    let probs_host: Vec<f32> =
        activation::softmax(logits.clone(), 1).into_data().to_vec().unwrap();
    let log_probs_host: Vec<f32> =
        activation::log_softmax(logits.clone(), 1).into_data().to_vec().unwrap();
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

/// Per-env GAE for the feedforward baseline. Flat `[T * N]` step-major layout
/// (index = step * num_envs + env). Identical to Tutorial 2's helper.
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

/// Running per-dimension observation standardizer (Welford mean/variance) with
/// flicker-aware pass-through. Standardizes each *visible* coordinate to unit
/// scale (load-bearing: an LSTM's tanh gates squash CartPole's tiny raw
/// observations to near-zero features). A **blanked frame is all-zeros** — a
/// state CartPole physics never produces — so it is passed through unchanged and
/// excluded from the running statistics, keeping the "no observation" signal a
/// clean zero for both policies.
struct ObsNormalizer {
    mean: Vec<f64>,
    m2: Vec<f64>, // Welford's sum of squared deviations
    count: f64,
}

impl ObsNormalizer {
    fn new(dim: usize) -> Self {
        Self { mean: vec![0.0; dim], m2: vec![0.0; dim], count: 0.0 }
    }

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
```

## Reading the output

Two numbers per arm, both in **raw** return units (undo the `REWARD_SCALE` in
your head — the code reports raw returns even though it trains on scaled ones):

- **`final`** — mean return over the last ≤100 episodes at the *end* of
  training. Honest but noisy at episode granularity; a single unlucky late
  episode moves it.
- **`best`** — the *peak* of that moving average at any point during the run.
  This is the honest answer to "did the policy ever *reach* a good level within
  budget?", reported alongside `final` so end-of-run oscillation can't hide a
  real success (or manufacture a fake one).

On a real run (`TOTAL_TIMESTEPS ≈ 500_000`), CartPole's "solved" bar is ~195,
and the pattern you want to see is:

- **LSTM `best` clears ~195** — memory recovers the hidden state through the
  blanks and balances the pole.
- **MLP `best` plateaus well below** — it cannot act on a blanked frame and has
  no memory to bridge the gap, so it caps out far short of solved.

That gap — same env, same seed, same normalizer, *only* the policy class
differs — is the whole point. It's the clean, controlled demonstration that
memory is load-bearing on this task, in a way it was **not** on `MaskedCartPole`.

The CI-sized budget in the doc-test above is far too small to reach any of this;
it exists only to keep the copy-paste code compiling and running against the live
API. For real numbers, run the packaged example.

## Going further

This tutorial covers **i.i.d. flickering** — each frame blanked independently
with probability `p`, the Hausknecht & Stone (2015) protocol. A few directions
from here:

- **Run the real thing.** The packaged
  [`recurrent_ppo_flickering_cartpole`](../../examples/games/cartpole/recurrent_ppo_flickering_cartpole.rs)
  example is this exact contrast at full budget (16 envs, 128-step rollouts,
  500k steps), with per-update logging and a CLI to run one arm at a time:

  ```bash
  cargo run --release --features training \
      --example recurrent_ppo_flickering_cartpole
  ```

- **Correlated dropout.** Real sensor failures come in *bursts*, not
  independent coin flips. The `recurrent_ppo_burst_flickering_cartpole` example
  blanks frames in runs of consecutive steps, which stresses memory over longer
  horizons. (Its burst constructor carries a stricter validity assert relating
  `flicker_prob` and burst length — out of scope here, covered in that example.)

- **The design rationale.** [`docs/RECURRENT_POLICY_DESIGN.md`](../RECURRENT_POLICY_DESIGN.md)
  is the full design note behind the recurrent stack: why Strategy A (BPTT
  truncation at window boundaries) over stored-subsequence alternatives, the
  GAE-vs-hidden-state-reset asymmetry in detail, and the warm-start hooks the
  buffer exposes for future strategies.

## Next

See the [tutorial index](README.md) for the full dependency-ordered series.
Tutorial 6 goes the other direction — down to the metal — and shows you how to
implement the `Environment` trait from scratch, including the seeding and
determinism contract that `EnvPool` and every env in this tutorial rely on.
