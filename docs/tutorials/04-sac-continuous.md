# Tutorial 4 — Continuous control with SAC

> **You will:** train a Soft Actor-Critic (SAC) agent to swing up an inverted
> pendulum — a *continuous* control task where the action is a real-valued
> torque, not a discrete button press — and learn the pieces continuous control
> adds on top of the off-policy replay machinery from Tutorial 3.
> **Prerequisites:** [Tutorial 3](03-dqn.md) — you should be comfortable with
> replay buffers, target networks, and Polyak soft updates. This tutorial
> *reuses* those ideas rather than re-deriving them.
> **Time:** ~20 minutes.

Every environment so far has had a **discrete** action space: push the cart left
or right, move up/down/left/right on a grid. DQN's whole strategy leans on that
finiteness — it scores every action with one forward pass and takes the
`argmax`. But a lot of real control is **continuous**: how *much* torque to apply
to a joint, how hard to press the accelerator, what steering angle to hold. You
cannot enumerate those actions, so `argmax` has nothing to iterate over.

This tutorial is about that world. We train **SAC** — the standard modern
algorithm for continuous control — on `PendulumSwingUp`. The good news: SAC is
**off-policy**, so the replay buffer and slowly-moving target networks you built
intuition for in Tutorial 3 carry over almost unchanged. What's new is entirely
about the *action*: a policy that outputs a probability distribution over the
real line, a squashing function that keeps that action in bounds, and an
entropy term that replaces ε-greedy as the exploration knob.

## Why continuous control is different

In Tutorial 3 the policy was implicit: "pick the action with the highest
Q-value." That works because the actions are `{0, 1}` — you can look at both
Q-values and take the bigger one. Now the action is a real number, say a torque
in `[-2, 2]`. There is no finite set to `argmax` over; `maxₐ Q(s, a)` becomes a
continuous optimization problem you would have to solve *inside every gradient
step*. That is intractable, so continuous-control methods take a different tack.

Instead of deriving the policy from the Q-function, SAC keeps an **explicit
actor**: a network that maps a state to a *distribution* over actions. For a
one-dimensional torque, that distribution is a Gaussian — the actor outputs a
mean `μ(s)` and a standard deviation `σ(s)`, and to act you *sample* `a ~
𝒩(μ, σ)`. The critic's job is no longer to rank a menu of actions; it is to
*score the specific action the actor chose*, `Q(s, a)`, so the actor can be
nudged toward higher-value actions by gradient ascent. Selection and evaluation
split across two networks — a theme you will recognize from Double-DQN.

## The problem: PendulumSwingUp

The task is the classic inverted pendulum. A rigid pole hangs from a pivot; you
apply a torque each step and try to swing it upright and hold it there. It is a
genuinely continuous problem — a good policy modulates torque smoothly, and
"bang-bang" (full torque one way or the other) is not enough to balance at the
top.

Three facts about the env shape the code:

- **Observation is 3-dimensional: `[cos θ, sin θ, θ̇]`.** The angle is encoded as
  the pair `(cos θ, sin θ)` rather than the raw angle `θ`. This is deliberate:
  `θ` wraps discontinuously at `±π` (just past `+π` you snap to `−π`), and a
  network fed that raw value would see a cliff in its input right at the top of
  the swing — exactly where precision matters most. `(cos, sin)` is smooth and
  continuous all the way around the circle, so the observation never jumps.
- **Action is 1-dimensional: a torque the env clamps to `[-2.0, 2.0]`.** The
  actor emits a value in `(-1, 1)` (see squashing, below), so we multiply by
  `MAX_TORQUE = 2.0` before calling `env.step()`.
- **Reward is always ≤ 0.** Each step you are penalized by how far the pole is
  from vertical (plus small costs for speed and torque). Perfect balance would
  score `0`; every real trajectory accumulates negative reward. A near-optimal
  policy earns roughly `-150` per episode; a random policy earns roughly
  `-1200` to `-1600`. So "learning" means the (negative) return climbs *toward*
  zero — a point that matters when you read the learning curve later.

Episodes run a fixed **200 steps** and then set `truncated = true`; the pole
never "falls off" or reaches a terminal state, so `terminated` is never set.
We construct the env with `PendulumSwingUp::with_seed(SEED)` for reproducibility.

## SAC's four pieces

SAC is best understood as **Tutorial 3's off-policy skeleton with a
continuous-action head grafted on**. Two of its four pieces you already know:

1. **A replay buffer.** Same idea as DQN — a fixed-capacity FIFO ring of
   `(s, a, r, s', done)` transitions, sampled in random minibatches to
   decorrelate experience and reuse each transition across many gradient steps.
   The only difference is that `a` is now a vector of floats, so SAC uses a
   `ContinuousReplayBuffer`. Everything you learned about *why* replay works in
   Tutorial 3 applies verbatim.
2. **Slowly-moving target networks, updated by Polyak blending.** Same
   `θ_target ← τ·θ_online + (1−τ)·θ_target` update from Tutorial 3, with the
   same small `τ = 0.005` default. SAC uses it to stabilize the critic targets
   exactly as DQN did — this is the "real per-parameter Polyak blend" Tutorial 3
   pointed ahead to.

The two *new* pieces are what make continuous control work:

3. **A stochastic actor** — the explicit policy distribution described above.
   It replaces DQN's implicit "argmax the Q-values" policy.
4. **An automatically-tuned entropy temperature** — the exploration knob that
   replaces ε-greedy. Instead of occasionally taking a uniformly random action,
   SAC keeps the *whole policy distribution* deliberately wide, and tunes exactly
   how wide as training proceeds.

The rest of this section unpacks the two new pieces, plus one wrinkle (twin
critics) that continuous Q-learning needs for the same reason Double-DQN did.

### Tanh action squashing

The actor's Gaussian is defined over all of ℝ — a sampled action could be `+7.3`
or `−4.1`, but the env only accepts a torque in a bounded range. SAC solves this
by pushing the raw Gaussian sample through a **`tanh`**: `tanh` maps `(−∞, +∞)`
smoothly onto `(−1, 1)`, so no matter what the Gaussian emits, the squashed
action is in bounds. We then rescale that `(−1, 1)` value by `MAX_TORQUE = 2.0`
to hit the env's `[−2, 2]` range.

Squashing is not free. SAC's actor loss needs the log-probability of the action
it took, and applying a nonlinear function to a random variable *changes* its
density — you have to correct the log-prob with the change-of-variables Jacobian
of `tanh` (the `log(1 − tanh²(x))` term derived in Haarnoja et al. 2018,
Appendix C). Thrust's `SacActor` computes this correction internally, so you get
a numerically-stable, correctly-normalized log-prob for free — but it is worth
knowing *why* the actor is more than a plain Gaussian: the squashing that keeps
actions in bounds is the same squashing that complicates the math underneath.

### Automatic entropy tuning

DQN explored with ε-greedy: an explicit, hand-scheduled probability of taking a
random action, annealed down over training. SAC explores differently and more
elegantly. It maximizes not just reward but **reward plus the entropy of the
policy** — it is rewarded for keeping its action distribution *wide* (uncertain)
as long as that doesn't cost too much return. High entropy = broad exploration;
low entropy = a sharp, near-deterministic policy. The trade-off is governed by a
**temperature** `α`: the coefficient on the entropy term.

Geometrically, `α` sets how much the policy is willing to spread out its Gaussian
to keep its options open. Too high and the agent flails randomly forever; too low
and it collapses to a deterministic policy before it has explored enough — the
same failure modes as a badly-tuned ε, but now on a continuous dial.

The elegant part: you don't schedule `α` by hand. SAC **tunes it
automatically** by gradient descent against a target entropy. The standard
heuristic is `target_entropy = −action_dim` (here `−1`): the optimizer raises
`α` when the policy is more deterministic than the target (encouraging
exploration) and lowers it when the policy is more random than the target
(encouraging exploitation). This is why SAC is famously robust to
hyperparameters — the exploration knob that you had to hand-anneal in DQN tunes
*itself*.

### Twin critics

There is one continuous-control wrinkle. In Tutorial 3 you saw that plain DQN
**overestimates** action values, because taking the `max` over noisy Q-estimates
is biased upward, and Double-DQN fixed it by decoupling action *selection* from
action *evaluation*. Continuous Q-learning has the same overestimation disease,
and SAC's cure is from the same family of insight: keep **two** independent
critics `Q₁` and `Q₂` (twin critics), and when computing the target, use the
*minimum* of the two. Taking the min systematically pulls the estimate back down,
cancelling the optimistic bias. It is the continuous analog of Double-DQN's
selection/evaluation split — decouple, then combine conservatively.

The `SacTrainer` builds and maintains all of this — twin online critics, their
two Polyak-tracked targets, the squashing actor, the auto-tuned temperature, and
the replay buffer — internally.

## The trainer call-site is simpler than DQN's

Here is a pleasant surprise: even though SAC has *more* moving parts than DQN,
its call-site is **simpler**. In Tutorial 3 you had to pass the trainer two
closures — one to argmax the Q-values for ε-greedy selection, one for the forward
pass — because the trainer was generic over your network type and couldn't call
`.forward()` itself. SAC needs none of that:

- **No action-selection closure.** `trainer.select_action(&obs)` returns the
  action directly. During the warmup phase (`learning_starts`) it samples a
  uniform random action; afterward it samples from the actor. That switch is
  handled internally via `in_warmup()` — there is no ε schedule to thread
  through and no argmax to write.
- **No explicit target-sync call.** DQN made you call `maybe_sync_target(...)`
  every step. SAC's `train()` performs the critic update, actor update, alpha
  update, *and* the Polyak target blend all in one call.

So the loop is just: select an action, rescale it, step the env, push the
transition, and call `train()`. The entire orchestration of the four pieces lives
behind that one method.

## The trainer surface

`SacConfig` holds the knobs; every field has a builder-style setter and the
defaults are the Haarnoja 2018 v2 paper values. The ones this tutorial touches:

- `buffer_capacity`, `min_buffer_size` — the replay ring size, and how many
  transitions to collect before the *first* gradient step (same meaning as DQN).
- `learning_starts` — steps of pure-random warmup before the actor is used for
  selection. Bootstraps the buffer with diverse experience.
- `batch_size` — minibatch size per gradient step.
- `hidden_dim`, `num_hidden_layers` — width and depth of the actor and critic
  MLPs.
- `actor_lr`, `critic_lr`, `alpha_lr` — three *independent* Adam learning rates
  (the actor, the critics, and the temperature each get their own optimizer).
- `gamma` — the discount, same meaning as always.
- `tau` — the Polyak blend coefficient (default `0.005`, applied every gradient
  step).
- `auto_alpha`, `init_alpha`, `target_entropy` — the entropy-tuning controls.
- `max_grad_norm` — optional global gradient-norm clip.
- `seed` — reproducibility; the actor init, both critic inits, and the replay
  RNG are all derived from it.

`SacConfig::new().validate()` runs automatically inside `SacTrainer::new`, so
there is no explicit validate call to remember.

`SacTrainer::new(config, obs_dim, action_dim, device)` constructs everything —
twin critics and their targets, the actor, the log-temperature, three Adam
optimizers, and the continuous replay buffer.

## The code

This runs a short training loop with a small step budget so it stays fast in CI;
bump `TOTAL_TIMESTEPS` (and the buffer/network sizes — see the table below the
block) for a real run. It is a doc-test, so it always compiles against the
current API.

```rust
use burn::backend::{Autodiff, NdArray};
use thrust_rl::{
    env::{Environment, games::pendulum::PendulumSwingUp},
    train::sac::{SacConfig, SacTrainer},
};

type Backend = Autodiff<NdArray<f32>>;

// --- Hyperparameters -------------------------------------------------------
// CI-sized budget so the doc-test runs in a few seconds. For a real swing-up,
// use the full-run values in the table below (TOTAL_TIMESTEPS ~30_000, wider
// networks, a bigger buffer).
const TOTAL_TIMESTEPS: usize = 4_096; // tiny for CI; ~30_000 for a real run
const HIDDEN_DIM: usize = 32; // 256 for a real run
const SEED: u64 = 0; // seeds actor + twin critics + replay RNG

// Pendulum's fixed dims: obs is [cos θ, sin θ, θ̇]; action is a scalar torque.
const OBS_DIM: usize = 3;
const ACTION_DIM: usize = 1;

// The actor emits a tanh-squashed action in (-1, 1); the env clamps torque to
// [-2, 2], so we rescale by MAX_TORQUE before every env step.
const MAX_TORQUE: f32 = 2.0;

let device = Default::default();

// --- Config: same shape as the packaged `train_sac` example, CI-sized -------
// Every field has a builder setter; unset fields keep the Haarnoja 2018 v2
// defaults (gamma 0.99, tau 0.005, auto-tuned alpha, target_entropy = -1).
let config = SacConfig::new()
    .buffer_capacity(4_096) // FIFO replay ring; overwrites oldest when full
    .min_buffer_size(512) // collect this many transitions before the first update
    .learning_starts(512) // steps of pure-random warmup before the actor is used
    .batch_size(64) // minibatch per gradient step
    .hidden_dim(HIDDEN_DIM)
    .num_hidden_layers(2)
    .seed(SEED);

// The trainer builds twin critics + targets, the squashing actor, the auto-tuned
// temperature, three Adam optimizers, and the continuous replay buffer.
let mut trainer =
    SacTrainer::<Backend>::new(config, OBS_DIM, ACTION_DIM, device).expect("valid SAC config");

// --- The single-env interaction loop ---------------------------------------
let mut env = PendulumSwingUp::with_seed(SEED);
env.reset();
let mut obs = env.get_observation();

while trainer.total_env_steps() < TOTAL_TIMESTEPS {
    // No closure, no ε schedule: select_action samples uniform (-1, 1) during
    // warmup and from the actor afterward, switching internally via in_warmup().
    let action = trainer.select_action(&obs);

    // Rescale the (-1, 1) actor output to the env's [-2, 2] torque range.
    let scaled: Vec<f32> = action.iter().map(|a| a * MAX_TORQUE).collect();
    let result = env.step(scaled);
    let done = result.terminated || result.truncated;

    // Push the *unscaled* action: the buffer stores what the actor emitted.
    trainer
        .buffer_mut()
        .push(&obs, &action, result.reward, &result.observation, done);
    trainer.increment_env_step();

    // One call runs the full SAC update — critics, actor, alpha, and the Polyak
    // target blend — and no-ops until the buffer holds `min_buffer_size`
    // transitions. No explicit target-sync call needed (unlike DQN).
    let _ = trainer.train().expect("train step");

    if done {
        trainer.increment_episodes(1);
        env.reset();
        obs = env.get_observation();
    } else {
        obs = result.observation;
    }
}

// A short CI run won't swing the pendulum up; we assert only that the loop ran,
// to keep the doc-test fast and deterministic. There is no reward bar here.
assert!(trainer.total_env_steps() >= TOTAL_TIMESTEPS);
```

For a real run, scale these up (the values the packaged `train_sac` example
uses):

| Parameter | CI value (above) | Full-run value |
|---|---|---|
| `TOTAL_TIMESTEPS` | 4_096 | 30_000 |
| `buffer_capacity` | 4_096 | 50_000 |
| `min_buffer_size` | 512 | 1_000 |
| `learning_starts` | 512 | 1_000 |
| `batch_size` | 64 | 256 |
| `hidden_dim` | 32 | 256 |
| `num_hidden_layers` | 2 | 2 |

## Reading the learning curve

The packaged
[`train_sac`](../../examples/games/pendulum/train_sac.rs) example is the same
loop with the full-run budget and one extra feature: set `CURVE_CSV` and it
writes one `env_steps,mean_episode_reward` row per logging interval.

```bash
TOTAL_TIMESTEPS=30000 CURVE_CSV=/tmp/sac.csv \
    cargo run --release --features training --example train_sac
```

Reading this curve takes one mental adjustment from the CartPole tutorials:
**Pendulum returns are negative**. `mean_episode_reward` here is the mean
*episode return* (a sum of per-step penalties), not an episode length. A random
policy sits around `−1200` to `−1600`; a converged policy climbs *toward* `−150`.
So "learning" is the curve rising from a deeply negative floor up toward zero —
progress moves **up and to the right**, but the whole axis lives below zero.

For that reason, **do not overlay Pendulum curves on the CartPole DQN/PPO/A2C
curves** from the earlier tutorials. Those are positive "reward == episode
length" curves topping out near `+500`; Pendulum's are negative return curves.
They live on different scales with different signs, and plotting them on one axis
is meaningless. Pendulum forms its own negative-return comparison group.

What to look for:

- **A curve rising from ~−1400 toward ~−200** = the actor is learning to swing
  up and hold. This is the healthy shape.
- **A curve stuck near the random floor** = the buffer never warms up
  (`min_buffer_size`/`learning_starts` larger than your budget), the networks
  are too small, or the learning rates are off.
- **A curve that climbs then collapses** = critic overestimation running away
  (rare with twin critics, but possible) or `α` mis-tuned. Confirm auto-alpha is
  on and the twin-critic min is in use.

## When to prefer SAC over DQN or PPO

Reach for **SAC** when:

- The action space is **continuous** (torques, forces, steering angles). This is
  the decisive factor — DQN's `argmax` cannot handle continuous actions at all,
  and SAC is purpose-built for them.
- **Environment steps are expensive** and you want maximum learning per step.
  Like DQN, SAC is off-policy and replay-based, so it is far more
  sample-efficient than on-policy PPO on the same continuous task.
- You want exploration handled **automatically**. SAC's entropy temperature tunes
  itself, sparing you the ε schedule (DQN) or entropy-coefficient fiddling (PPO).

Reach for **DQN** (Tutorial 3) when the action space is **discrete** and small —
`argmax` over a handful of actions is simpler and needs no actor network.

Reach for **PPO** (Tutorial 2) when you value **training stability and
robustness** over sample efficiency, or when environment interaction is cheap
(many parallel envs) so on-policy throughput wins on wall-clock time. PPO also
handles continuous actions — SAC's edge there is sample efficiency, not
capability.

## Try it yourself

- **Disable automatic entropy tuning.** Set `.auto_alpha(false).init_alpha(0.2)`
  to freeze the temperature at a hand-picked value. Watch how sensitive training
  becomes to that one number — direct evidence for *why* SAC tunes `α` itself.
- **Shrink or grow the networks.** Swap `hidden_dim` between `32`, `64`, and
  `256`. Too narrow and the actor can't represent a good swing-up policy; the
  sweet spot trades capacity against per-step cost.
- **Try a harder continuous env.** Point the loop at `MountainCarContinuous`
  instead of `PendulumSwingUp` (also continuous, but with a *sparse* reward —
  you only score on reaching the goal). Sparse reward makes SAC's entropy-driven
  exploration earn its keep; expect to re-tune the warmup and buffer knobs.
- **Full run:** drop the CI-sized `TOTAL_TIMESTEPS` and run the packaged
  `train_sac` example above with `CURVE_CSV`, then watch the negative-return
  curve climb toward `−200`.

## Next

That completes the core algorithm tour — on-policy PPO (Tutorials 1–2),
off-policy discrete DQN (Tutorial 3), and off-policy continuous SAC (this one).
See the [tutorial index](README.md) for what comes next: memory and POMDPs with
recurrent PPO, writing your own environment, and deploying a trained policy to
the browser.
