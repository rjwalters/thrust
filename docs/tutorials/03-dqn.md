# Tutorial 3 — Off-policy training with DQN

> **You will:** train a Double-DQN agent to balance CartPole using a replay
> buffer and a target network, and learn *when* to reach for off-policy DQN
> instead of the on-policy PPO loop from Tutorial 2.
> **Prerequisites:** [Tutorial 2](02-cartpole-ppo.md) — you should be
> comfortable with the rollout → advantage → update loop, `CartPole`, and
> Burn's move-through optimizer.
> **Time:** ~20 minutes.

Tutorial 2 was **on-policy**: every gradient step used data collected by the
*current* policy, and once you called `train_step` that rollout was spent — the
next update needed a fresh rollout. That is a lot of throwing data away.

This tutorial is **off-policy**. The core idea is to keep a big **replay
buffer** of past transitions `(s, a, r, s', done)` and train by sampling *old*
experience out of it, over and over. Data collected by an earlier version of
the policy is still useful, so you get far more gradient updates per step of
real environment interaction — DQN is dramatically more **sample-efficient**
than PPO on small discrete tasks.

Four things change from the PPO loop:

1. **We learn Q-values, not a policy distribution.** Instead of a network that
   outputs action probabilities, the Q-network outputs one scalar per action —
   the estimated return of taking that action and then acting greedily. The
   policy is implicit: "pick the action with the highest Q."
2. **We keep a replay buffer.** Transitions go in as we collect them; each
   gradient step samples a random minibatch back out.
3. **We keep a second, slowly-moving target network.** It supplies the "answer"
   the online network regresses toward, and its lag is what keeps training from
   chasing its own tail.
4. **We explore with ε-greedy, not entropy.** There is no policy distribution to
   inject entropy into, so exploration is a separate, explicit knob: with
   probability ε take a random action, otherwise take the greedy one. ε starts
   high and **anneals** down.

## The problem: CartPole-v1 (again, on purpose)

We reuse the exact environment from Tutorial 2 — a pole hinged on a cart, push
left (`0`) or right (`1`), `+1` reward per step the pole stays up, episode ends
when it falls or at 500 steps. So **episode reward == episode length**, random
pushing scores ~22, and a solved agent scores near 500.

Reusing the same env is deliberate: it makes the PPO-vs-DQN contrast concrete.
Same task, same reward scale, same seed budget — the only thing that changes is
the learning algorithm underneath.

## Why off-policy? Why a replay buffer?

Consecutive transitions in a single CartPole episode are **highly correlated**:
the pole angle at step `t+1` is almost the pole angle at step `t`. If you
trained on transitions in the order you collected them, each minibatch would be
a tight cluster of near-identical states, and the gradient would swing wildly as
the agent walked through different regions of the state space. That is unstable
and sample-hungry.

A replay buffer fixes this two ways:

- **It breaks correlation.** Sampling a random minibatch mixes transitions from
  many different episodes and many different times, so each update sees a
  roughly i.i.d. slice of the agent's experience — much closer to the
  assumption supervised optimizers are built on.
- **It reuses data.** A transition can be sampled into many minibatches over its
  lifetime in the buffer, so one step of (expensive) environment interaction
  feeds many (cheap) gradient steps. That is where the sample-efficiency comes
  from.

Thrust's [`ReplayBuffer`](../../src/buffer/replay/storage.rs) is a
fixed-capacity FIFO ring: once it is full, each new transition overwrites the
oldest. The capacity is a memory-vs-recency tradeoff — big enough to decorrelate
and remember, small enough that ancient, off-distribution transitions eventually
age out.

## The four pieces

DQN is really "regression with two stabilizers bolted on." The four moving
parts:

1. **The Q-network** — a small MLP `state → [Q(s, a₀), Q(s, a₁), …]`. Thrust's
   [`QNetworkBurn`](../../src/policy/q_network.rs) is a 2-layer Tanh MLP with
   orthogonal init. One forward pass scores every action at once.
2. **The replay buffer** — described above.
3. **The target network** — a *lagged copy* of the Q-network used to compute the
   regression target (below).
4. **The ε-schedule** — the exploration knob that decays over training.

### Why a target network?

DQN trains the Q-network to satisfy the Bellman equation:

```text
Q(s, a)  ≈  r + γ · maxₐ' Q(s', a')
```

The catch: the thing on the right (the "target") is computed with the *same*
network you are updating. If you regress a network toward a target that shifts
every time you take a gradient step, you get a feedback loop — the target moves
because the network moved because the target moved — and training oscillates or
diverges. This is the "chasing your own tail" problem.

The fix is a **target network**: a separate copy of the Q-network, held (nearly)
fixed, that supplies the `Q(s', a')` term. Because it changes slowly, the
regression target is *stable* over many updates, which is what makes the whole
thing converge. Periodically the target is nudged toward the online network so
it does not fall permanently behind.

### Why ε must anneal (not stay fixed)

Early in training the Q-values are random noise, so "greedy" is meaningless — you
**must** explore broadly to discover which actions are good. That argues for a
high ε (lots of random actions) at the start.

But late in training you have a good policy, and continuing to take random
actions 100% of the time would just add noise and cap your performance. So ε
must **come down** as the agent learns. A fixed ε forces a bad compromise: too
high and the final policy is noisy; too low and early exploration is starved. We
linearly anneal ε from `1.0` (pure exploration) down to a small floor like
`0.05`, which keeps a trickle of exploration forever so the agent can still
adapt.

## Double-DQN, briefly

Plain DQN uses `maxₐ' Q_target(s', a')` as the target. Because the `max` picks
whichever action the (noisy) network happens to *overestimate*, plain DQN
systematically **overestimates** action values — the max of noisy estimates is
biased upward. Over training this optimism compounds.

**Double-DQN** decouples the two roles of that `max`:

```text
a*  =  argmaxₐ' Q_online(s', a')      ← the ONLINE net picks the action
y   =  r + γ · (1 − done) · Q_target(s', a*)   ← the TARGET net scores it
```

Selecting the action with one network and evaluating it with another cancels
most of the upward bias, because the two networks rarely share the *same* noise.
It is nearly free — same two networks, one argmax moved — and strictly better on
most tasks. Thrust's DQN trainer uses the Double-DQN target
[unconditionally](../../src/train/dqn/loss.rs), so you get it for free.

## Hard copy vs. Polyak soft updates

How the target network tracks the online network is a knob. Two strategies:

- **Hard copy** (the default): every `target_update_interval` env steps, copy
  the online weights wholesale into the target (`θ_target ← θ_online`). Between
  copies the target is frozen. This is `DQNConfig`'s behavior when
  `soft_update_tau` is left unset (`None`).
- **Polyak soft update**: every step, blend the target a tiny fraction of the
  way toward the online network:

  ```text
  θ_target  ←  τ · θ_online  +  (1 − τ) · θ_target
  ```

  With a small `τ` (e.g. `0.005`), the target drifts smoothly toward the online
  network instead of jumping in discrete steps — often a bit more stable. You
  opt in with `.soft_update_tau(0.005)`; the trainer then invokes your blend
  closure *every* step instead of hard-copying on the interval. (SAC in
  Tutorial 4 leans on exactly this mechanism for its twin critics.)

In `DQNTrainerBurn`, the update is driven by `maybe_sync_target`, which takes a
closure `(online, target, τ) → new_target`. That closure is the seam where the
blend lives: a full Polyak blend computes the convex combination above per
parameter. `QNetworkBurn` exposes a whole-network parameter copy rather than a
per-parameter blend, so the code below passes the simplest closure that fits the
seam and lets `soft_update_tau` control the *cadence*. We set `τ = 0.005`
explicitly to mirror the packaged `train_cartpole_dqn` example.

## The trainer surface

`DQNConfig` is where the off-policy knobs live. The important ones:

- `learning_rate`, `batch_size` — optimization.
- `buffer_capacity`, `min_buffer_size` — the replay buffer size, and how many
  transitions to collect before the *first* gradient step (you cannot sample a
  minibatch from an empty buffer, and training on a handful of transitions
  overfits instantly).
- `gamma` — the discount, same meaning as in Tutorial 2.
- `epsilon_start`, `epsilon_end`, `epsilon_decay_steps` — the ε-anneal schedule.
- `target_update_interval` — hard-copy cadence (ignored once you set
  `soft_update_tau`).
- `soft_update_tau` — opt into Polyak soft updates.
- `max_grad_norm` — gradient clipping for stability.

You build a `DQNConfig`, wrap the online network and a `BurnOptimizer` in a
`DQNTrainerBurn`, then drive the loop yourself: select an action, step the env,
push the transition, sync the target, and call `train_step` once per env step.

## The code

This runs a short training loop (small budget so it is fast in CI; bump
`TOTAL_TIMESTEPS` for a real run). It is a doc-test, so it always compiles
against the current API.

```rust
use burn::{
    backend::{Autodiff, NdArray},
    optim::AdamConfig,
    tensor::{Tensor, TensorData},
};
use rand::{SeedableRng, rngs::StdRng};
use thrust_rl::{
    env::{Environment, cartpole::CartPole},
    policy::q_network::QNetworkBurn,
    train::{
        dqn::{DQNConfig, DQNTrainerBurn},
        optimizer::BurnOptimizer,
    },
};

type Backend = Autodiff<NdArray<f32>>;

// --- Hyperparameters -------------------------------------------------------
const TOTAL_TIMESTEPS: usize = 4_096; // tiny for CI; use ~60_000 for a real run
const HIDDEN_DIM: usize = 64;
const SEED: u64 = 0; // seeds the Q-network init for a reproducible run

let device = Default::default();

// --- Environment probe -----------------------------------------------------
let probe = CartPole::new();
let obs_dim = probe.observation_space().shape[0];
let n_actions = match probe.action_space().space_type {
    thrust_rl::env::SpaceType::Discrete(n) => n as i64,
    _ => panic!("CartPole is discrete"),
};

// --- The online Q-network: a seeded 2-layer Tanh MLP -----------------------
// It maps a 4-float observation to two Q-values, one per action. Seeding the
// init (together with the seeded action/replay RNG below) makes the run
// reproducible.
let online =
    QNetworkBurn::<Backend>::with_seed(obs_dim, n_actions as usize, HIDDEN_DIM, SEED, &device);

// --- Config: identical to the packaged `train_cartpole_dqn` example --------
// (just a shorter budget). Every field has a builder setter.
let config = DQNConfig::new()
    .learning_rate(1e-3)
    .batch_size(64)
    .buffer_capacity(50_000) // decorrelate + remember; overwrites oldest when full
    .min_buffer_size(1_000) // collect this many transitions before the first update
    .target_update_interval(500) // hard-copy cadence (ignored once tau is set)
    .gamma(0.99)
    .epsilon_start(1.0) // start fully exploring: greedy is meaningless at init
    .epsilon_end(0.05) // keep a trickle of exploration forever
    .epsilon_decay_steps(10_000) // linearly anneal ε over this many env steps
    .max_grad_norm(10.0)
    .soft_update_tau(0.005); // opt into Polyak soft updates (every step)

// --- Optimizer + trainer ---------------------------------------------------
let inner_opt = AdamConfig::new().init();
let burn_opt: BurnOptimizer<Backend, QNetworkBurn<Backend>, _> =
    BurnOptimizer::new(inner_opt, config.learning_rate);

// The trainer owns the online net, the optimizer, the replay buffer, and an
// internally-managed target network (a lagged clone of `online`).
let mut trainer = DQNTrainerBurn::new(config, online, burn_opt, obs_dim, n_actions, device)
    .expect("valid DQN config");

// --- The single-env interaction loop ---------------------------------------
let mut env = CartPole::new();
env.reset();
let mut obs = env.get_observation();
let mut rng = StdRng::seed_from_u64(0xC0FFEE);

while trainer.total_env_steps() < TOTAL_TIMESTEPS {
    // --- ε-greedy action selection ---------------------------------------
    // The trainer draws ε for us (from the anneal schedule) and, with prob.
    // 1 − ε, calls this closure to get the greedy action. It has to be a
    // closure because the trainer is generic over the network type `Q` and
    // cannot call `.forward()` itself without the caller spelling out the
    // concrete type and how to argmax the result.
    let action = trainer.select_action(&obs, &mut rng, |q: &QNetworkBurn<Backend>, o_host: &[f32]| {
        let o_t: Tensor<Backend, 2> =
            Tensor::from_data(TensorData::new(o_host.to_vec(), [1, o_host.len()]), &device);
        let q_values = q.forward(o_t); // [1, n_actions]
        let q_host: Vec<f32> = q_values.into_data().to_vec().unwrap_or_default();
        // argmax over actions.
        let mut best = 0_i64;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in q_host.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i as i64;
            }
        }
        best
    });

    // --- Step the env and store the transition ---------------------------
    let result = env.step(action);
    let next_obs = result.observation.clone();
    let done = result.terminated || result.truncated;
    trainer.buffer_mut().push(&obs, action, result.reward, &next_obs, done);
    obs = next_obs;

    trainer.increment_env_step();

    // --- Sync the target network -----------------------------------------
    // Because we set `soft_update_tau`, this fires every step. The closure is
    // the blend seam described above; `soft_update_tau` controls the cadence.
    let _ = trainer.maybe_sync_target(|online, _target, _tau| online.clone());

    if done {
        trainer.increment_episodes(1);
        env.reset();
        obs = env.get_observation();
    }

    // --- One gradient step against the Double-DQN TD target --------------
    // The two closures are the online forward pass (grad-bearing) and the
    // target forward pass (the trainer detaches it for the TD target). The
    // trainer no-ops until the buffer holds `min_buffer_size` transitions.
    let _ = trainer
        .train_step(
            &mut rng,
            |q: &QNetworkBurn<Backend>, o: Tensor<Backend, 2>| q.forward(o),
            |q: &QNetworkBurn<Backend>, o: Tensor<Backend, 2>| q.forward(o),
        )
        .expect("train step");
}

// A short CI run won't fully solve CartPole; we assert only that the loop ran,
// to keep the doc-test fast and deterministic.
assert!(trainer.total_env_steps() >= TOTAL_TIMESTEPS);
```

## Reading the learning curve

The packaged `train_cartpole_dqn` example is the same loop with a bigger budget
and one extra feature: set `CURVE_CSV` and it writes one
`env_steps,mean_episode_reward` row per logging interval.

```bash
TOTAL_TIMESTEPS=60000 CURVE_CSV=/tmp/dqn.csv \
    cargo run --release --features training --example train_cartpole_dqn
```

Because CartPole's reward is `+1`/step, `mean_episode_reward` **is** the mean
episode length. It runs on the *same* env, seed, and budget as the PPO
(`train_cartpole_modern`) and A2C (`train_cartpole_a2c`) examples, so you can
overlay all three curves on one axis for an honest algorithm comparison. What to
look for:

- **A rising curve** = the Q-values are learning to rank actions correctly.
- **A curve stuck near ~22** = something is off. Common causes: the buffer
  never warms up (`min_buffer_size` larger than your budget), ε decays so fast
  the agent never explores, or the learning rate is too high and the Q-values
  diverge.
- **A curve that climbs then collapses** = classic DQN instability (often
  Q-value overestimation running away). Lengthen the target lag, lower the
  learning rate, or confirm the Double-DQN target is in use.

Compared to the PPO curve on the same axis, DQN typically **reaches a good
policy in far fewer env steps** (replay reuse), while PPO's curve is often
smoother and less prone to the climb-then-collapse failure mode. That tradeoff
is the whole point of the next section.

## When to prefer DQN over PPO

Reach for **off-policy DQN** when:

- The action space is **discrete** (DQN's `argmax` over actions needs a finite,
  enumerable set — it does not apply to continuous control).
- **Environment steps are expensive** and you want maximum learning per step;
  replay reuse makes DQN far more sample-efficient on small discrete tasks.
- You can tolerate more hyperparameter fuss (buffer size, target cadence, ε
  schedule) in exchange for that sample efficiency.

Reach for **on-policy PPO** (Tutorial 2) when:

- The action space is **continuous**, or you specifically want a stochastic
  policy.
- You value **training stability and robustness** over raw sample efficiency —
  PPO's trust-region clip makes it forgiving of hyperparameters.
- You can collect lots of environment interaction cheaply (e.g. a fast
  simulator with many parallel envs), so sample efficiency matters less than
  wall-clock throughput.

## Try it yourself

- **Turn off the target lag.** Set `.soft_update_tau(1.0)` (or a very short
  `target_update_interval` with `soft_update_tau` unset). The target now tracks
  the online net almost exactly — watch training get noticeably less stable, the
  concrete cost of removing the stabilizer.
- **Freeze exploration.** Set `epsilon_start` and `epsilon_end` both to `0.05`.
  With no early exploration the agent never discovers the good actions and the
  curve stalls near the random baseline — direct proof ε must start high.
- **Swap in a harder env.** Point the loop at
  [`GridWorld`](../../examples/games/grid_world/train_dqn_grid_world.rs) instead
  of `CartPole` (16-dim one-hot obs, 4 actions, sparse ±1 reward). Its packaged
  example uses `gamma(0.95)`, `epsilon_end(0.10)`, and
  `epsilon_decay_steps(20_000)` — a good exercise in re-tuning the off-policy
  knobs for a sparse-reward task.
- **Full run:** drop the CI-sized `TOTAL_TIMESTEPS` and run the packaged example
  above with the CSV, then overlay it against the PPO curve from Tutorial 2.

## Next

The path splits from here — see the [tutorial index](README.md) for the full
dependency-ordered series. Next up (Tutorial 4) is **continuous control with
SAC**: Box action spaces, tanh action squashing, automatic entropy tuning, and
twin critics — which reuse the off-policy replay-and-target machinery you just
learned, now with a real per-parameter Polyak blend.
