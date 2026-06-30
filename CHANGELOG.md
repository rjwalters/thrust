# Changelog

All notable changes to **thrust-rl** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with the pre-1.0 convention that **breaking API changes will land in MINOR
bumps** (e.g. `0.1.x -> 0.2.0`) until we cut `1.0.0`. Patch releases
(`0.1.x -> 0.1.y`) are bug fixes and additive, non-breaking improvements.

See [`docs/RELEASING.md`](docs/RELEASING.md) for the maintainer-facing
release procedure (tagging, GitHub Release creation, and `cargo publish`).

## [Unreleased]

Nothing yet.

## [0.2.0] - 2026-06-30

### Summary

The multi-agent research release. 0.2.0 matures the population-based
multi-agent stack — PSRO (with an α-rank meta-solver) and approximate
NFSP — to the point of running the Bucket Brigade workshop paper's
no-convergence cells end to end, adds the A2C algorithm, and lands
substantial training-throughput work. It also closes out that research
question: an extensive validation effort (documented under
`docs/research/`) concluded — and root-caused via a non-PPO improvability
oracle — that neither PSRO nor NFSP beats PPO on those cells because the
single best-response has ~no team-return headroom against frozen-uniform
opponents. The negative result and its mechanism are fully documented.

All changes are additive; existing public APIs are unchanged (MINOR bump
per the pre-1.0 policy).

### Added

#### Algorithms & training
- **A2C**: `A2cTrainer` with an un-clipped advantage-actor-critic loss,
  smoke tests, a CartPole example, opt-in learning-curve CSV export, and a
  convergence test (#156, #158).
- **PSRO**: payoff-magnitude fixes for α-rank saturation plus best-response
  critic reward scaling, making the α-rank meta-solver usable at the
  no-convergence cells' payoff scale (#225).
- **NFSP**: adaptive average-policy supervised-step floor and reward scaling
  to address average-policy under-training (#223).
- **Joint trainer**: tunable best-response-fit defaults and an optional
  separate critic optimizer (#239).

#### Bucket Brigade research tooling
- Critic **explained-variance** diagnostic surfaced from the joint PPO
  update, plus a standalone single-best-response A/B probe harness
  (`train_br_probe`) for fast critic-fit investigation (#241, #247).
- Non-PPO **best-response improvability oracle** (`bucket_brigade_oracle`
  module + `br_oracle` example): scores scripted/searched policies against
  frozen-uniform opponents to bound achievable best-response return (#259).
- Cell-specific `gap_closed_cell` reporting in the bucket-brigade examples
  (#220).
- `SEED` env override across the bucket-brigade training examples for
  seeded multi-host replication, plus cluster run/dispatch scripts (#256).

#### Features
- Off-by-default `blas-threaded` feature for threaded NdArray matmul (#237).
- Opt-in `max_minibatches_per_epoch` best-response-update throughput lever
  (#254).

### Changed
- Dropped the `linker = "clang"` pin in `.cargo/config.toml`, keeping the
  fast `lld` linker via `-fuse-ld=lld` (works with the universal gcc/cc
  driver; removes a needless clang requirement) (#257).

### Fixed
- Best-response PPO update now applies the configured grad-norm clip and
  iterates all minibatches per epoch (previously the clip was dead and most
  of each rollout was discarded) (#240).
- Snake env food respawn is now deterministic via an owned seeded RNG.
- Cleared `env-bucket-brigade` clippy lints and gated them in CI (#231).

### Performance
- PSRO round-robin best-response loop parallelized with rayon (#238).
- NFSP per-agent average-policy supervised training parallelized with rayon
  (#236), and per-step seeded sampling batched through a single model
  forward (#249).

## [0.1.0] - 2026-06-08

First publishable release of thrust-rl. Locks in the foundation laid by
the initial development sprints: two on-policy / off-policy training
algorithms (PPO and DQN), four environments with both single-agent and
multi-agent variants, the multi-agent population/matchmaking
infrastructure, WASM-deployable pure-Rust inference, browser demos, and
Bayesian-style hyperparameter exploration. Everything below is brand new
relative to "no published version".

### Added

#### Algorithms
- **PPO** (`src/train/ppo*`): clipped-surrogate Proximal Policy
  Optimization with Generalized Advantage Estimation (GAE), an
  auxiliary-loss hook (`PPOTrainer::set_aux_loss_fn`), and the
  `MlpPolicy` / `MultiDiscreteMlpPolicy` policy heads. Solves CartPole
  (~300-470 steps/episode on the curated configs in
  `examples/games/cartpole/`).
- **DQN** (`src/train/dqn*`): vanilla DQN MVP plus Double-DQN target
  computation and Polyak soft target updates (#58), and Prioritized
  Experience Replay with proportional sampling and importance-sampling
  correction (#59).

#### Environments
- **CartPole** (`src/env/games/cartpole.rs`): classic balancing task,
  trained to >300 step episodes by the bundled configs.
- **Snake** (`src/env/games/snake/`): single-agent and multi-agent
  variants with configurable grid size; multi-agent uses a population
  through `multi_agent::PolicyLearner`.
- **Pong** (`src/env/games/pong.rs`): single-player vs. rule-based
  opponent and self-play training pipeline (#47, #53).
- **SimpleBandit** (`src/env/games/simple_bandit.rs`): contextual bandit
  used as a sanity test for PPO and the WASM inference stack.
- **ContinuousLqr** (`src/env/games/continuous_lqr.rs`): 1-D LQR
  placeholder env exercising the `Vec<f32>` continuous-action surface
  introduced alongside #61.
- **BucketBrigade** (Slepian-Wolf MARL research env, see
  `src/env/games/bucket_brigade/`): adapter is in-tree but the
  `env-bucket-brigade` feature is **disabled in the published v0.1.0
  manifest** because the underlying `bucket-brigade-core` crate has not
  been published to crates.io. Local checkouts can still enable it by
  un-commenting the feature in `Cargo.toml`. See "Known limitations"
  below.

#### Multi-agent infrastructure
- `multi_agent::Population` and `Agent` for population-based training.
- `GameSimulator` skeleton and `JointMultiAgentTrainer` for synchronized
  joint PPO across agents (#15).
- `multi_agent::PolicyLearner` with the per-epoch / per-step bugs from
  the original Slepian-Wolf port resolved (#4, #41, #34, #39).
- Four matchmaking strategies: `Random`, `RoundRobin`, `Fitness`,
  `SelfPlay`.
- Multi-discrete action support in `MultiAgentEnvironment` (#14).
- `Environment::clone_state` / `restore_state` for MCTS, replay, and
  rollback workflows (#62).

#### WASM and browser demos
- Pure-Rust inference module (`src/inference/`) compiles to WASM with no
  PyTorch dependency.
- Universal inference system with JSON model format
  (`docs/UNIVERSAL_INFERENCE.md`).
- React + Vite web app under `web/` (excluded from the crate tarball)
  with live CartPole, Snake, and SimpleBandit demos; SimpleBandit
  agent visualizer; Pong web demo with self-play model and rule-based
  fallback (#76).

#### Hyperparameter optimization
- Bayesian / Pareto-style hyperparameter search machinery
  (`src/optim/`) used by the `optimize_cartpole` and `optimize_snake`
  examples to sweep PPO clip ranges, value-function coefficients, and
  network widths.

#### Documentation
- `README.md` with project vision, status, and demo links.
- `ROADMAP.md` and `MULTI_AGENT_DESIGN.md` for forward planning.
- `docs/PPO_BEST_PRACTICES.md`, `docs/RESEARCH_PAPERS.md`,
  `docs/RL_TOYBOX_SURVEY.md` (#49).
- `docs/GPU_SETUP.md`, `docs/LIBTORCH_SETUP.md`,
  `docs/REMOTE_TRAINING.md`, `docs/TRAINING_*` per-environment guides.
- Public-API rustdoc audit completed (#36, #38).

#### Tooling
- `build.rs` defeats GNU ld's `--as-needed` for `libtorch_cuda` on
  Linux so CUDA-equipped boxes don't silently fall back to CPU (#13).
- GPU training scripts (`scripts/gpu-*.sh`, excluded from the crate
  tarball) for an NVIDIA L4 sandbox.
- Loom orchestration framework installed under `.loom/` for AI-assisted
  development (excluded from the crate tarball).

### Known limitations

- **DQN does not reliably solve CartPole.** PPO does. Both Double-DQN
  and PER landed in v0.1.0, but the canonical CartPole DQN config still
  shows high variance run-to-run. Tracked by the open follow-up issues
  surfaced during #58/#59 review.
- **`env-bucket-brigade` is source-only in the published crate.** The
  underlying `bucket-brigade-core` Cargo crate lives in a git submodule
  and is not on crates.io, so `cargo publish` cannot include it as an
  optional dependency. Users who need the env should depend on
  `thrust-rl` from a git checkout with submodules initialized and
  un-comment the relevant lines in `Cargo.toml`. Re-publishing
  `bucket-brigade-core` is tracked as a v0.2.x follow-up.
- **`multi_agent::PolicyLearner` API is experimental** and is expected
  to change as multi-agent training matures. Pin to a specific 0.1.x
  patch if you depend on its current shape.
- **`tch-rs` version pinned to 0.22** (PyTorch 2.9), which requires
  Rust nightly + `edition = "2024"`. See `rust-toolchain.toml`.
- **No binary `release` artifacts.** Consumers build from source via
  `cargo install thrust-rl` (library) or by cloning the repo for
  examples. A binary release workflow is intentionally out of scope
  for v0.1.0.

### Removed (relative to never-published state)

- Empty `BayesianOptimizer` / `ParetoFrontier` / `TrialScheduler`
  placeholder modules were dropped before publish.
- `bincode` dependency was removed to resolve RUSTSEC-2025-0141 (#37).
- Misleading `target_synced` field on `DQNStepStats` (#66).

[Unreleased]: https://github.com/rjwalters/thrust/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rjwalters/thrust/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rjwalters/thrust/releases/tag/v0.1.0
