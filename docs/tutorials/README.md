# 📚 Thrust Tutorials

A guided path from `cargo add thrust-rl` to a trained, deployed policy. Where
the [Example Gallery](../EXAMPLES.md) is a *reference* (one runnable trainer per
algorithm), this series is a *learning path*: each tutorial is a single Markdown
file with complete, copy-paste code, and they are meant to be read in order —
each one assumes the concepts introduced by the ones before it.

## Why these are trustworthy

Every `rust` code block in a landed tutorial is a **doc-test**: it is compiled
and run by `cargo test --features training` in CI. If the library API changes
out from under a tutorial, CI goes red — so the copy-paste code here can never
silently rot. The mechanism lives in
[`src/tutorials.rs`](../../src/tutorials.rs), which pulls each Markdown file in
via `#[doc = include_str!(...)]`.

## The path

The series is ordered by concept dependency. Follow it top to bottom; each
tutorial names its prerequisites in a header block.

| # | Tutorial | You'll learn | Algorithm · Environment | Status |
|---|----------|--------------|-------------------------|--------|
| 1 | [Your first agent](01-your-first-agent.md) | The rollout → loss → update loop, by hand; actor-critic; why bandits skip GAE | Actor-critic (PPO-style) · `SimpleBandit` | ✅ Landed |
| 2 | [Solving CartPole with PPO](02-cartpole-ppo.md) | `EnvPool`, GAE, the `PPOConfig` surface, reading learning curves (`CURVE_CSV`) | PPO · `CartPole` | ✅ Landed |
| 3 | Off-policy training with DQN | Replay buffers, target networks, ε-annealing; when to prefer DQN over PPO | DQN (Double-DQN) · `CartPole` / `GridWorld` | 🚧 Planned ([#309](https://github.com/rjwalters/thrust/issues/309)) |
| 4 | Continuous control with SAC | Box action spaces, tanh action squashing, automatic entropy tuning, twin critics | SAC · `PendulumSwingUp` | 🚧 Planned ([#310](https://github.com/rjwalters/thrust/issues/310)) |
| 5 | Memory and POMDPs | When an LSTM earns its cost; recurrent rollouts and hidden-state handling | Recurrent PPO · `FlickeringCartPole` | 🚧 Planned ([#311](https://github.com/rjwalters/thrust/issues/311)) |
| 6 | Writing your own environment | Implementing the `Environment` trait from scratch; the seeding/determinism contract | (trait impl) · tiny gridworld | 🚧 Planned ([#312](https://github.com/rjwalters/thrust/issues/312)) |
| 7 | Train in Rust, run in the browser | Exporting to the JSON inference format and wiring it into the WASM demo | Inference export · WASM | 🚧 Planned ([#313](https://github.com/rjwalters/thrust/issues/313)) |

### Dependency order, spelled out

- **1 → 2**: Tutorial 1 teaches the on-policy update loop by hand so Tutorial 2
  can hand that loop to `PPOTrainerBurn` and focus on GAE and the config surface.
- **2 → 3**: DQN is introduced *after* PPO so the tutorial can frame off-policy
  learning (replay, target networks) as a contrast to the on-policy loop you
  already understand.
- **3 → 4**: SAC (continuous control) builds on the off-policy replay machinery
  from DQN and adds the continuous-action pieces (squashing, entropy tuning).
- **2 → 5**: Recurrent PPO extends the Tutorial 2 PPO loop with memory; it needs
  the on-policy loop but not the off-policy tutorials, so it can be read right
  after Tutorial 2 by readers chasing POMDPs.
- **1 → 6**: Writing an environment only needs the `Environment` trait from
  Tutorial 1; it's placed late because it's most useful once you've *used*
  several built-in envs and want your own.
- **any → 7**: Deployment is last because it consumes a policy trained by any of
  the earlier tutorials and takes it to the browser.

## Landing plan

Tutorials 1–2 land first: they unblock the crates.io newcomer path (install →
first agent → a real trained policy). Tutorials 3–7 follow incrementally, each
tracked by its own child issue under
[#303](https://github.com/rjwalters/thrust/issues/303):
[#309](https://github.com/rjwalters/thrust/issues/309) (DQN),
[#310](https://github.com/rjwalters/thrust/issues/310) (SAC),
[#311](https://github.com/rjwalters/thrust/issues/311) (recurrent PPO),
[#312](https://github.com/rjwalters/thrust/issues/312) (custom env), and
[#313](https://github.com/rjwalters/thrust/issues/313) (WASM deploy). Each future
tutorial must meet the same bar as 1–2: a single Markdown file, complete
copy-paste code, CI-enforced as a doc-test.

## See also

- [GETTING_STARTED.md](../GETTING_STARTED.md) — install and first build.
- [EXAMPLES.md](../EXAMPLES.md) — the reference gallery of runnable trainers.
- [PPO_BEST_PRACTICES.md](../PPO_BEST_PRACTICES.md) — tuning the on-policy knobs.
- [BURN_BACKENDS.md](../BURN_BACKENDS.md) — swapping in a GPU backend.
