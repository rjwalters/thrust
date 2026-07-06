# 🎮 Example Gallery

Thrust ships **eighteen** runnable training/evaluation examples under
[`examples/games/`](../examples/games). Each one is a self-contained `[[example]]`
target registered in [`Cargo.toml`](../Cargo.toml), so you can run any of them
with a single `cargo run` command — no setup beyond a Rust toolchain.

All examples default to Burn's **pure-Rust NdArray CPU backend**, so a fresh
checkout runs them with no external system libraries (libtorch, CUDA, etc.).
The `training` feature is enabled by default; it is shown explicitly in the
commands below for clarity and to keep them copy-pasteable even if you build
with `--no-default-features`. Opt into a GPU backend by appending `wgpu`
(cross-platform: Vulkan / Metal / DX12 / WebGPU) or `cuda` (Linux + NVIDIA) to
the feature list, e.g. `--features "training,wgpu"`. See
[BURN_BACKENDS.md](BURN_BACKENDS.md) for backend details.

> **Tip:** always use `--release`. Debug builds of the training loop are
> dramatically slower.

## Index

| Example | Algorithm | Environment | Type |
| --- | --- | --- | --- |
| [`train_simple_bandit`](#train_simple_bandit) | Actor-critic (PPO-style) | SimpleBandit | Single-agent |
| [`train_cartpole_modern`](#train_cartpole_modern) | PPO | CartPole | Single-agent |
| [`train_cartpole_dqn`](#train_cartpole_dqn) | DQN (Double-DQN) | CartPole | Single-agent |
| [`train_cartpole_a2c`](#train_cartpole_a2c) | A2C | CartPole | Single-agent |
| [`train_cartpole_async`](#train_cartpole_async) | PPO (async actor-learner, V-trace) | CartPole | Single-agent |
| [`recurrent_ppo_flickering_cartpole`](#recurrent_ppo_flickering_cartpole) | Recurrent PPO (LSTM) vs MLP | FlickeringCartPole | POMDP contrast |
| [`recurrent_ppo_masked_cartpole`](#recurrent_ppo_masked_cartpole) | Recurrent PPO (LSTM) vs MLP | MaskedCartPole | Negative result |
| [`train_dqn_grid_world`](#train_dqn_grid_world) | DQN | GridWorld | Single-agent |
| [`train_bc_cartpole`](#train_bc_cartpole) | Behavioral Cloning | CartPole | Imitation |
| [`train_sac`](#train_sac) | SAC | PendulumSwingUp | Continuous control |
| [`train_sac_mountain_car`](#train_sac_mountain_car) | SAC | MountainCarContinuous | Continuous control |
| [`train_snake_multi_v2`](#train_snake_multi_v2) | PPO (per-agent self-play) | Snake | Multi-agent |
| [`train_pong_self_play`](#train_pong_self_play) | PPO (self-play) | Pong | Multi-agent |
| [`eval_pong`](#eval_pong) | Evaluation / model export | Pong | Tooling |
| [`train_psro`](#train_psro) | PSRO + α-rank | BucketBrigade | Multi-agent (feature-gated) |
| [`train_nfsp`](#train_nfsp) | NFSP | BucketBrigade | Multi-agent (feature-gated) |
| [`train_br_probe`](#train_br_probe) | Single best-response A/B probe | BucketBrigade | Research (feature-gated) |
| [`br_oracle`](#br_oracle) | Coalition improvability oracle (k*) | BucketBrigade | Research (feature-gated) |

---

## Single-agent control

### `train_simple_bandit`

The canonical end-to-end Burn example: an actor-critic policy on the
`SimpleBandit` contextual-bandit environment. Demonstrates the
rollout → loss → update loop with `MlpBurnPolicy`, the move-through optimizer
ownership model, and the vectorized `EnvPool` wrapper.

```bash
cargo run --release --features training --example train_simple_bandit
```

Source: [`examples/games/bandit/train_simple_bandit.rs`](../examples/games/bandit/train_simple_bandit.rs)

### `train_cartpole_modern`

PPO on CartPole-v1 — the reference on-policy trainer. A 2-layer MLP over an
`EnvPool` of 16 parallel envs, GAE advantages, clipped surrogate loss.

```bash
cargo run --release --features training --example train_cartpole_modern
```

**Environment variables:**

| Var | Effect |
| --- | --- |
| `TOTAL_TIMESTEPS` | Override the env-step budget (default ~200k). |
| `CURVE_CSV` | Path to write an `env_steps,mean_episode_reward` learning-curve CSV. |

```bash
TOTAL_TIMESTEPS=50000 CURVE_CSV=/tmp/ppo.csv \
    cargo run --release --features training --example train_cartpole_modern
```

Source: [`examples/games/cartpole/train_cartpole_modern.rs`](../examples/games/cartpole/train_cartpole_modern.rs)

### `train_cartpole_dqn`

Double-DQN on CartPole-v1: replay buffer, target network, ε-annealing, and
Polyak (soft) target updates. The off-policy counterpart to the PPO example.

```bash
cargo run --release --features training --example train_cartpole_dqn
```

**Environment variables:** `TOTAL_TIMESTEPS` (default ~60k), `CURVE_CSV`.

```bash
TOTAL_TIMESTEPS=20000 CURVE_CSV=/tmp/dqn.csv \
    cargo run --release --features training --example train_cartpole_dqn
```

Source: [`examples/games/cartpole/train_cartpole_dqn.rs`](../examples/games/cartpole/train_cartpole_dqn.rs)

### `train_cartpole_a2c`

Synchronous Advantage Actor-Critic (A2C) on CartPole-v1. Mirrors the PPO
example's `EnvPool` rollout + GAE pattern but takes a single un-clipped
policy-gradient + MSE-value gradient step per rollout (no epoch loop, no
importance ratio).

```bash
cargo run --release --features training --example train_cartpole_a2c
```

**Environment variables:** `TOTAL_TIMESTEPS`, `CURVE_CSV`.

```bash
TOTAL_TIMESTEPS=200000 CURVE_CSV=/tmp/a2c.csv \
    cargo run --release --features training --example train_cartpole_a2c
```

Source: [`examples/games/cartpole/train_cartpole_a2c.rs`](../examples/games/cartpole/train_cartpole_a2c.rs)

### `train_bc_cartpole`

End-to-end **Behavioral Cloning** on CartPole-v1: train an A2C expert, harvest
greedy `(obs, action)` demonstrations from it, clone a fresh policy with
supervised cross-entropy via `BcTrainer`, then report mean episode reward for
both expert and clone.

```bash
cargo run --release --features training --example train_bc_cartpole
```

**Environment variables:**

| Var | Effect |
| --- | --- |
| `EXPERT_TIMESTEPS` | Override how long the A2C expert trains before demos are harvested. |

```bash
EXPERT_TIMESTEPS=200000 \
    cargo run --release --features training --example train_bc_cartpole
```

Source: [`examples/games/cartpole/train_bc_cartpole.rs`](../examples/games/cartpole/train_bc_cartpole.rs)

### `train_cartpole_async`

Single-host **async actor-learner** PPO on CartPole-v1: actor threads collect
rollouts over crossbeam channels while the learner updates concurrently,
reaching the learning bar at ~2.1× synchronous throughput. Optional **V-trace**
(IMPALA) off-policy correction compensates for policy staleness in the queued
rollouts.

```bash
cargo run --release --features training --example train_cartpole_async
```

**Environment variables:** `TOTAL_TIMESTEPS`, `NUM_ACTORS`, `USE_VTRACE`,
`MAX_LEAD_STEPS`, `BROADCAST_EVERY`, `GAE_LAMBDA`, `SEED`.

```bash
NUM_ACTORS=4 USE_VTRACE=1 \
    cargo run --release --features training --example train_cartpole_async
```

Source: [`examples/games/cartpole/train_cartpole_async.rs`](../examples/games/cartpole/train_cartpole_async.rs)

### `recurrent_ppo_flickering_cartpole`

The **memory benchmark**: recurrent (LSTM) PPO vs a feedforward MLP baseline on
`FlickeringCartPole`, a POMDP where the whole observation is zeroed with
probability `p = 0.5` per step (Hausknecht & Stone). Memory is load-bearing by
construction — the LSTM solves the env (peak ~484/500) while the memoryless MLP
plateaus far below (~176). Both arms get identical training treatment.

```bash
cargo run --release --features training --example recurrent_ppo_flickering_cartpole
```

**Environment variables:** `TOTAL_TIMESTEPS` (per arm), `FLICKER_PROB`
(default `0.5`).

Source: [`examples/games/cartpole/recurrent_ppo_flickering_cartpole.rs`](../examples/games/cartpole/recurrent_ppo_flickering_cartpole.rs)

### `recurrent_ppo_masked_cartpole`

A documented **negative result**, kept runnable on purpose: the same
LSTM-vs-MLP contrast on `MaskedCartPole` (velocity components zeroed).
Empirically this is *not* a POMDP — a reactive `[x, θ]` controller solves it
outright, and the MLP beats the LSTM. Useful as a cautionary baseline when
designing memory tasks.

```bash
cargo run --release --features training --example recurrent_ppo_masked_cartpole
```

**Environment variables:** `TOTAL_TIMESTEPS` (per arm).

Source: [`examples/games/cartpole/recurrent_ppo_masked_cartpole.rs`](../examples/games/cartpole/recurrent_ppo_masked_cartpole.rs)

### `train_dqn_grid_world`

DQN on the discrete `GridWorld` navigation env — a second off-policy reference
alongside the CartPole DQN example.

```bash
cargo run --release --features training --example train_dqn_grid_world
```

**Environment variables:** `TOTAL_TIMESTEPS`, `CURVE_CSV`.

Source: [`examples/games/grid_world/train_dqn_grid_world.rs`](../examples/games/grid_world/train_dqn_grid_world.rs)

---

## Continuous control

### `train_sac`

Soft Actor-Critic (SAC) on the continuous `PendulumSwingUp` env (Gym's
`Pendulum-v1`). Off-policy, twin critics, automatic entropy tuning, Polyak
target updates; one env step + one replay-sampled gradient update per step.
The actor emits a tanh-squashed action rescaled by `MAX_TORQUE = 2.0`.

```bash
cargo run --release --features training --example train_sac
```

**Environment variables:** `TOTAL_TIMESTEPS`, `CURVE_CSV`.

```bash
TOTAL_TIMESTEPS=4000 CURVE_CSV=/tmp/sac.csv \
    cargo run --release --features training --example train_sac
```

Source: [`examples/games/pendulum/train_sac.rs`](../examples/games/pendulum/train_sac.rs)

### `train_sac_mountain_car`

SAC on the continuous `MountainCarContinuous` env — the deceptive-reward
sibling of the Pendulum example, with a momentum-pumping exploration warmup.
The action range is already `[-1, 1]`, so the actor output is passed straight
to `env.step`.

```bash
cargo run --release --features training --example train_sac_mountain_car
```

**Environment variables:** `TOTAL_TIMESTEPS`, `CURVE_CSV`.

```bash
TOTAL_TIMESTEPS=4000 CURVE_CSV=/tmp/mc.csv \
    cargo run --release --features training --example train_sac_mountain_car
```

Source: [`examples/games/mountain_car/train_sac_mountain_car.rs`](../examples/games/mountain_car/train_sac_mountain_car.rs)

---

## Multi-agent / self-play

### `train_snake_multi_v2`

Multi-agent Snake: N independent PPO learners (one `SnakeCnnBurnPolicy` +
`PPOTrainerBurn` per snake) share a single `SnakeEnv`. Each snake gets a
per-agent grid observation and is updated independently on its own rollout
buffer — the canonical multi-agent training recipe.

```bash
cargo run --release --features training --example train_snake_multi_v2
```

**Environment variables:** `TOTAL_TIMESTEPS` (total budget across all agents).

Source: [`examples/games/snake/train_snake_multi_v2.rs`](../examples/games/snake/train_snake_multi_v2.rs)

### `train_pong_self_play`

PPO self-play on Pong: the left paddle trains while the right paddle is a frozen
snapshot of the live policy, refreshed every `SNAPSHOT_INTERVAL` updates. A
single policy network plays either side via mirrored observations.

```bash
cargo run --release --features training --example train_pong_self_play
```

**Environment variables:** `TOTAL_TIMESTEPS` (default ~200k).

```bash
TOTAL_TIMESTEPS=200000 \
    cargo run --release --features training --example train_pong_self_play
```

Source: [`examples/games/pong/train_pong_self_play.rs`](../examples/games/pong/train_pong_self_play.rs)

### `eval_pong`

Tooling, not training. Two jobs in one binary:

1. **Convert** a Burn-recorded `MlpBurnPolicy` (`.bin` from
   `train_pong_self_play`) into the `InferenceModel` JSON format the WASM demo
   loads.
2. **Evaluate** any number of `InferenceModel` JSON files head-to-head against
   Pong's built-in rule-based opponent over a configurable number of episodes.

```bash
# Evaluate over 1000 episodes (defaults match train_pong_self_play's filenames):
cargo run --release --features training --example eval_pong -- --episodes 1000

# Convert + evaluate explicitly:
cargo run --release --features training --example eval_pong -- \
    --self-play-bin pong_self_play_model \
    --self-play-json pong_self_play_model.json \
    --rule-based-json web/public/pong_model.json \
    --episodes 1000
```

Source: [`examples/games/pong/eval_pong.rs`](../examples/games/pong/eval_pong.rs)

---

## Cooperative MARL (feature-gated)

> **Note:** The Bucket Brigade examples require the `env-bucket-brigade` feature
> **and** the `envs/bucket-brigade` git submodule, which is path-only and **not
> part of the published crate** (see [RELEASING.md](RELEASING.md)). They run only
> from a local checkout with the submodule initialized:
>
> ```bash
> git submodule update --init --recursive
> ```

### `train_psro`

PSRO (Policy-Space Response Oracles) with an **α-rank** meta-solver on the
Bucket Brigade workshop-paper "no-convergence" cells. Drives an N-tensor PSRO
trainer over a factored `[house, mode, signal]` multi-discrete action policy:
each outer iteration solves the meta-Nash, trains a fresh best-response per
agent, then re-evaluates exploitability.

```bash
cargo run --release --features "training,env-bucket-brigade" --example train_psro
```

**Environment variables:**

| Var | Effect |
| --- | --- |
| `CELL` | Which no-convergence cell to train (`beta01` \| `beta05` \| `beta09`; default `beta05`). |
| `TOTAL_ITERATIONS` | Number of PSRO outer iterations. |
| `ROLLOUT_STEPS` | Rollout length per best-response training step. |

```bash
TOTAL_ITERATIONS=20 ROLLOUT_STEPS=4096 CELL=beta01 \
    cargo run --release --features "training,env-bucket-brigade" --example train_psro
```

Source: [`examples/games/bucket_brigade/train_psro.rs`](../examples/games/bucket_brigade/train_psro.rs)

### `train_nfsp`

NFSP (Neural Fictitious Self-Play), approximate and N-player, on the same
Bucket Brigade cells. Each agent's reservoir stores factored multi-discrete
actions natively, driving the `[house, mode, signal]` action space without a
Cartesian-product wrapper.

```bash
cargo run --release --features "training,env-bucket-brigade" --example train_nfsp
```

**Environment variables:** `CELL` (`beta01` | `beta05` | `beta09`, default
`beta05`), `TOTAL_ITERATIONS`, `ROLLOUT_STEPS`.

```bash
TOTAL_ITERATIONS=20 ROLLOUT_STEPS=4096 CELL=beta01 \
    cargo run --release --features "training,env-bucket-brigade" --example train_nfsp
```

Source: [`examples/games/bucket_brigade/train_nfsp.rs`](../examples/games/bucket_brigade/train_nfsp.rs)

### `train_br_probe`

Research harness, not a tutorial: a standalone **single best-response A/B
probe** for fast critic-fit investigation on the Bucket Brigade cells
(explained-variance diagnostics without a full PSRO/NFSP outer loop).

```bash
cargo run --release --features "training,env-bucket-brigade" --example train_br_probe
```

**Environment variables:** `CELL`, `ITERATIONS`, `ROLLOUT_STEPS`,
`BR_TRAIN_STEPS`, `BR_REWARD_SCALE`, `BR_MAX_MINIBATCHES_PER_EPOCH`,
`CRITIC_LR`, `VF_COEF`, `SEED`.

Source: [`examples/games/bucket_brigade/train_br_probe.rs`](../examples/games/bucket_brigade/train_br_probe.rs)

### `br_oracle`

The non-PPO **coalition improvability oracle**: scores scripted/searched
policies against frozen-uniform opponents to measure the k* coalition-size
threshold, and can sweep raw `(beta, kappa, c)` grids to produce the full
phase-diagram artifact (`docs/research/2026-07-kstar-phase-diagram.md`).

```bash
cargo run --release --features "training,env-bucket-brigade" --example br_oracle
```

**Environment variables:** `PHASE`, `CELL` / `CELLS`, `K`, `ALPHA`, `OUT_DIR`.

Source: [`examples/games/bucket_brigade/br_oracle.rs`](../examples/games/bucket_brigade/br_oracle.rs)
