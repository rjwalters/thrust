# rl-toybox Survey: What's Worth Duplicating in Thrust

**Status**: Plan only. No ports happen in this PR. Concrete port work is enumerated in the "Follow-up issues to file" section at the bottom.

## Scope and Sources

This survey is deliberately Thrust-side. Per the scope constraint in issue #45, **rl-toybox was not cloned, fetched, or built** while writing this document. The characterizations of the upstream project come from two public sources only:

1. The rl-toybox README and repository metadata at https://github.com/bzznrc/rl-toybox.
2. The summary table provided in the body of issue #45.

Wherever this document would otherwise need to peek at upstream Python source to be confident about an implementation detail (e.g. exact replay-buffer capacity, exact network shapes, exact MCTS rollout policy), it instead flags the unknown as a gap. The point of this document is not to faithfully describe rl-toybox; it is to decide what Thrust should build next, using rl-toybox as a forcing function for breadth.

## Thrust today (verified against the source tree on `main`)

This section was written by walking the tree, not by paraphrasing the issue body. It supersedes any conflicting summary elsewhere.

**Environments** (`src/env/games/`, registered in `src/env/games/mod.rs`):
- `src/env/games/cartpole.rs` — single-agent classic control, discrete action space.
- `src/env/games/simple_bandit.rs` — multi-armed bandit, used as a sanity check for the trainer.
- `src/env/games/snake.rs` + `src/env/games/snake/{environment,multi_agent,types}.rs` — single- and multi-agent snake. The CNN policy that consumes it lives in `src/policy/snake_cnn.rs`.
- `src/env/games/bucket_brigade.rs` — feature-gated (`env-bucket-brigade`) placeholder that points at the external `bucket-brigade` submodule under `envs/`. Not a working in-tree Thrust env.

Note: an earlier Pong env existed in tree but was removed on `main` in commit `fcf200a` ("Remove orphaned src/env/games/pong.rs (YAGNI)"). The Pong WASM demo at `web/src/components/Pong/` runs against a pre-trained exported model and does not require an in-tree Rust env.

The `Environment` trait in `src/env/mod.rs` only accepts discrete actions today (`fn step(&mut self, action: i64)`). `SpaceType::Box` exists as an enum variant on `SpaceInfo`, but **no environment or policy in the repository currently consumes continuous actions**. Adding SAC therefore requires either extending the trait or introducing a parallel `ContinuousEnvironment` trait.

**Algorithms** (`src/train/`):
- `src/train/ppo/{trainer,config,loss,stats}.rs` — PPO with clipping, value-function loss, and entropy bonus. This is the only trainer in the repo.

**Buffers** (`src/buffer/`):
- `src/buffer/rollout/{storage,gae,sampling,tests}.rs` plus `src/buffer/rollout.rs` — on-policy rollout buffer with GAE. **No replay buffer of any kind exists.** Adding any off-policy algorithm (DQN, SAC, TD3, etc.) means introducing a new buffer family.

**Policies** (`src/policy/`):
- `src/policy/mlp.rs` — `MlpPolicy` for discrete actions over flat observations.
- `src/policy/multi_discrete_mlp.rs` — discrete-action MLP for multi-head action spaces.
- `src/policy/snake_cnn.rs` — `SnakeCnnPolicy` for grid observations.
- `src/policy/{inference,universal_inference}.rs` — pure-Rust inference path used by `src/wasm.rs` for WASM demos. No tch-rs at runtime in the browser.

No Q-network, no twin critics, no Gaussian-head continuous policy.

**Multi-agent infrastructure** (`src/multi_agent/`):
- `src/multi_agent/population.rs` — `Population`, `Agent`, `AgentId`, `LearningMode`.
- `src/multi_agent/simulator.rs` — `GameSimulator` (parallel game thread).
- `src/multi_agent/learner.rs` — `PolicyLearner` (per-agent PPO trainer thread).
- `src/multi_agent/matchmaking.rs` — four `MatchmakingStrategy` variants (`Random`, `RoundRobin`, `FitnessBased`, `SelfPlay`).
- `src/multi_agent/messages.rs` — `Experience`, `PolicyUpdate`, `ControlMessage`, `TrainingStats` (channel payloads between simulator and learner).
- `src/multi_agent/joint.rs` — synchronized joint trainer that owns N policies and N optimizers in one process for cross-agent-coupled losses (P3-style auxiliary objectives).

There is **no centralized critic** today. Each `PolicyLearner` runs an independent PPO loop on its own agent's trajectories; the only way agents currently share information at training time is through the joint trainer's cross-agent auxiliary loss term, which is a representational regularizer, not a critic.

**Hyperparameter optimization** (`src/optimize/`): Bayesian search + Pareto-frontier tooling. Orthogonal to this survey.

**Reference docs already in repo**: `ROADMAP.md` (Milestone 6 already lists DQN / SAC / A2C / behavioral cloning), `MULTI_AGENT_DESIGN.md` (under `docs/archive/` for an earlier proposal, plus the current state captured in `src/multi_agent/mod.rs` doc-comments), `docs/PPO_BEST_PRACTICES.md`, `docs/RESEARCH_PAPERS.md`.

## rl-toybox at a glance

Reproduced from the issue body, since that is one of the two permitted sources. Not independently verified against upstream.

| Env | Style | Algorithm |
|-----|-------|-----------|
| Snake | Grid control | Q-learning |
| Bang | Discrete-control arena shooter | DQN (double-Q, prioritized replay) |
| Jump | Platformer | PPO (actor-critic) |
| Vroom | Racing, continuous control | SAC |
| Flip | Board game, self-play | MCTS |
| Kick | Multi-agent football (3v3/5v5/7v7) | Centralized critic + shared policy |

Stack: Python + PyTorch, Arcade for rendering. Layout: `core/` (shared algo + env contracts), `games/` (per-env code/config/docs), `scripts/` (CLI for train/eval/play), `docs/`.

**Known gaps in our picture of upstream** (do not fix by cloning; flag and move on):
- Exact replay-buffer capacity and prioritization exponent (Bang).
- Exact actor/critic network widths (all envs).
- Whether SAC uses entropy-tuning (learned alpha) or a fixed alpha (Vroom).
- MCTS rollout policy: pure random rollouts vs. learned policy guidance vs. AlphaZero-style PUCT (Flip).
- Whether Kick uses a true MAPPO/COMA centralized critic or a simpler shared-value baseline.

These gaps do not block the recommendation. If we port any of these, the porting issue should be free to make sensible choices and document them; we should not feel obligated to mirror upstream.

## Comparison table

| rl-toybox env | rl-toybox algorithm | Thrust env analogue | Algorithm gap in Thrust | Effort to close gap |
|---|---|---|---|---|
| Snake | Q-learning | `src/env/games/snake` (much richer, has CNN policy and multi-agent variant) | No tabular or NN Q-learning; PPO only. Adding a Q-network policy + simple replay = first step toward DQN. | **S** (env reuse + small new trainer file + tiny replay buffer for tabular case) |
| Bang | DQN with double-Q + prioritized replay | None. Closest is the discrete-action skeleton in cartpole; we have no arena-shooter env. | Full DQN trainer (`src/train/dqn/`), Q-network policy (`src/policy/q_network.rs`), replay buffer (`src/buffer/replay/`), prioritized sampling. | **M** (algorithm stack) / **L** (if also porting the env, which we do not recommend; see non-goals) |
| Jump | PPO | None. Could reuse cartpole as a PPO sanity check; no platformer env. | None for algorithm. Porting the env is orthogonal to the survey. | **N/A** (algorithm already covered) |
| Vroom | SAC | None. Continuous-control env does not exist in Thrust. | SAC trainer (`src/train/sac/`), twin critics, Gaussian policy head, replay buffer (shared with DQN), and a `Box`-action Environment trait extension. | **L** (changes `Environment` trait or adds a parallel continuous trait) |
| Flip | MCTS, self-play | None. We have `SelfPlay` matchmaking but no game-tree search. | MCTS trainer (`src/train/mcts/`), env-side state-clone hook (Thrust envs do not currently expose `clone_state`), optional learned prior network. | **L** (touches `Environment` trait to add a state-clone API; without it, MCTS cannot do rollouts) |
| Kick | Centralized-critic shared-policy MARL | None (would map to bucket-brigade or a new soccer env). `SelfPlay` matchmaking is the closest existing piece. | Centralized critic that conditions on all agents' observations and actions, an extension to `src/multi_agent/learner.rs` to do joint critic updates, and new message variants in `src/multi_agent/messages.rs`. Shared policy weights are already easy with `Population`. | **M** (extends multi_agent, no trait changes) |

Effort tier calibration (also used in the next section):
- **S** = fits in one PR, touches three files or fewer, no new module directories.
- **M** = new module under `src/train/` (and probably a sibling under `src/buffer/` or `src/policy/`), but no breaking trait change.
- **L** = changes the `Environment` trait, the multi-agent message protocol, or some other API the rest of the crate already depends on.

## Ranked port recommendations

These are ranked by impact-per-unit-effort, not by which is most "fun" to build. The bar to make this list is: "would change the kinds of papers Thrust can reproduce."

### 1. Replay buffer with uniform and prioritized sampling — **S/M**

**What to build.** A new module `src/buffer/replay/` with two structs: `ReplayBuffer` (uniform `(s, a, r, s', done)` ring buffer, FIFO eviction) and `PrioritizedReplayBuffer` (sum-tree-backed with `alpha`/`beta` annealing). Expose a trait so trainers can be generic over which one they use. No GPU work; this is pure storage and indexing.

**Thrust modules touched.** New: `src/buffer/replay/{mod,uniform,prioritized,tests}.rs`. Edit: `src/buffer/mod.rs` to re-export. Nothing else.

**Why it matters.** Every off-policy algorithm we want next (DQN, SAC, TD3, BC-with-replay) needs this. Building it standalone, with tests, before either DQN or SAC means those follow-ups stop being "two new things at once." This is the cheapest unblocker on the board.

### 2. DQN trainer with double-Q and target network — **M**

**What to build.** A new trainer `src/train/dqn/` (mirroring the layout of `src/train/ppo/`: `trainer.rs`, `config.rs`, `loss.rs`, `stats.rs`). A Q-network policy at `src/policy/q_network.rs` that returns Q-values per discrete action and supports a separate target-network copy with periodic hard or Polyak updates. Loss: standard double-DQN target (online net selects argmax, target net evaluates).

**Thrust modules touched.** New: `src/train/dqn/`, `src/policy/q_network.rs`. Edit: `src/train/mod.rs`, `src/policy/mod.rs` for re-exports. Depends on port #1 (replay buffer).

**Why it matters.** DQN is the canonical entry point to off-policy RL and the most common comparison baseline for any new discrete-action paper. It also forces us to decouple "policy" from "actor-critic" in `src/policy/`, which today implicitly assumes a stochastic actor head. Done right, it makes the policy module honest about what it represents.

### 3. SAC trainer + continuous-action support — **L**

**What to build.** A new trainer `src/train/sac/` (twin critics, Gaussian-policy actor, optional learned alpha). A continuous policy head at `src/policy/gaussian_mlp.rs`. The hard part is the `Environment` trait: today `fn step(&mut self, action: i64)`. SAC requires `Vec<f32>` actions. Two options to discuss in the issue: (a) generalize the trait via an associated `Action` type, (b) introduce a parallel `ContinuousEnvironment` trait with its own pool / rollout path. Either is non-trivial; both touch every existing env. The follow-up issue should pick one and document the choice; the trait change is the L tier of work.

**Thrust modules touched.** `src/env/mod.rs` (trait), every file under `src/env/games/` (implementation updates if the trait changes), `src/train/sac/`, `src/policy/gaussian_mlp.rs`, `src/buffer/replay/` (shared with #2). Possibly `src/wasm.rs` for inference compatibility.

**Why it matters.** SAC unlocks continuous-control research entirely. Without it, Thrust cannot reproduce locomotion, robotic-manipulation, or driving benchmarks. The trait change is also long overdue: hard-coding `i64` actions is a debt we will pay eventually regardless.

### 4. Centralized critic for multi-agent training — **M**

**What to build.** Extend `src/multi_agent/learner.rs` so a `PolicyLearner` can optionally consume a centralized critic that conditions on the joint observation and joint action of all agents in its game. Add new `Experience` and `PolicyUpdate` variants (or fields) in `src/multi_agent/messages.rs` to carry the joint state/action. The shared policy story (every agent uses the same network) is already easy with `Population::with_shared_policy` semantics; this work is purely about the critic side.

**Thrust modules touched.** Edit: `src/multi_agent/learner.rs`, `src/multi_agent/messages.rs`, `src/multi_agent/simulator.rs` (to plumb joint observations through). New: probably `src/multi_agent/centralized_critic.rs`. No trait changes.

**Why it matters.** Cooperative multi-agent (the rl-toybox Kick use case, and a substantial fraction of bucket-brigade's reason for existing) is qualitatively different from competitive self-play. PPO-with-self-play does not learn the bucket-brigade-style coordination problems well; centralized-critic methods (MAPPO, COMA) are the standard answer. This is the single most natural extension of the multi-agent stack we already shipped.

### 5. MCTS trainer + Environment::clone_state — **L**

**What to build.** A new trainer `src/train/mcts/` implementing a basic UCT search, optionally with a learned policy/value prior (AlphaZero-style PUCT). The blocking dependency is that `Environment` does not expose state cloning; without it MCTS cannot do rollouts. Adding `fn clone_state(&self) -> Self::State` (or `Box<dyn Any>`) to the trait, plus a `fn restore_state(&mut self, state: ...)` partner, is the hard part — every existing env implementer must opt in.

**Thrust modules touched.** `src/env/mod.rs` (trait), every env (state-clone implementation), `src/train/mcts/`. Optional: a learned-prior policy variant that reuses the existing actor-critic policies for guidance.

**Why it matters.** MCTS is the only family on this list that opens up zero-/partial-information board games and any setting where you have a perfect simulator and want to leverage it. The cost is real (trait change, every env updated) and the payoff is narrower than SAC's. Ranked below SAC for that reason, but ahead of nothing because the use case is otherwise unreachable.

## Algorithmic tricks worth adopting independently

These are things to consider lifting even if we do not port the corresponding env. Each entry says what Thrust has today and what would change.

- **Prioritized experience replay.** *Thrust today*: nothing. The rollout buffer is on-policy FIFO; no replay exists at all. *Recommendation*: implement as part of port #1 above. Use a sum-tree for `O(log n)` weighted sampling. Important to get importance-sampling weights right when annealing `beta`.

- **Double-Q target updates with a target network.** *Thrust today*: nothing. PPO has no target network; the value head is updated in-place. *Recommendation*: introduce target-network bookkeeping as part of port #2 (DQN). The pattern (hold a frozen copy of the network parameters, refresh every K steps or via Polyak averaging) generalizes to SAC's twin critics and any future TD3-style trainer.

- **Centralized critic / decentralized policy (CTDE).** *Thrust today*: nothing. The closest existing piece is `src/multi_agent/joint.rs`, which couples agent encoders through an auxiliary loss term but does not give the critic access to other agents' observations or actions during the value update. *Recommendation*: port #4. This is the single biggest qualitative gap in our multi-agent story.

- **MCTS rollout policy.** *Thrust today*: nothing — no game-tree search anywhere. We do have `SelfPlay` matchmaking, which is a different beast (it controls *who plays whom* during data collection, not search inside an episode). *Recommendation*: port #5. Note that even without porting MCTS, the underlying enabler (`Environment::clone_state`) would be useful for several other things, including value-function bootstrapping experiments and counterfactual rollouts; it might be worth filing the trait change separately as a smaller predecessor issue.

- **Soft target updates (Polyak averaging) for critics.** *Thrust today*: nothing. *Recommendation*: bundle with the target-network work in port #2 / #3, since it is the natural smoothing variant of hard target swaps.

- **Entropy bonus in continuous action spaces (learned alpha).** *Thrust today*: PPO uses a fixed entropy coefficient. *Recommendation*: include in port #3 (SAC), where temperature auto-tuning is standard.

## Explicit non-goals

This section is here to prevent scope creep when the follow-up issues get filed.

- **Do not port the Bang env.** rl-toybox uses Bang as a vehicle for DQN. We want the *algorithm* (DQN, double-Q, prioritized replay) and we can validate it against CartPole and Snake, both of which we already have. Building a Python-arcade-style shooter env in Rust is high effort and low value — it does not validate anything algorithmic that CartPole + a tuned reward function does not validate equally well.

- **Do not port the Jump env.** It is a PPO platformer; PPO is what Thrust already has, and CartPole already serves as our PPO smoke test. Adding a platformer to the repo is an art project, not an RL project.

- **Do not port rl-toybox's tabular Q-learning Snake variant.** Our Snake is richer (CNN policy, multi-agent variant) and a tabular Q-table over our state space is impractical. If we want Q-learning on Snake, it should be neural Q-learning (DQN) using the Snake env we already have — which is what port #2 already covers.

- **Do not faithfully replicate rl-toybox's Python layout.** Their `core/` vs `games/` split is good for a Python repo with Arcade rendering; Thrust's `src/env/games/` + `src/train/` + `src/policy/` split is better aligned with how Rust modules want to be organized. Use rl-toybox for the *what*, not the *how*.

- **Do not clone, vendor, or fetch any upstream code.** Every port we file should be implemented from algorithm descriptions in the standard literature (Mnih 2013/2015 for DQN, Haarnoja 2018 for SAC, Silver 2017 for MCTS, Lowe 2017 / Yu 2021 for MAPPO/COMA), not from rl-toybox.

- **Do not port the Kick env (separately from the centralized-critic work).** We already have bucket-brigade as our cooperative-MARL target and infrastructure under `envs/bucket-brigade/`. Re-doing it as soccer would duplicate the multi-agent integration work without adding a new algorithmic capability.

## Follow-up issues to file

These are intended to be pasted directly into `gh issue create -t "<title>"`. They are roughly in dependency order (replay buffer first, then off-policy algorithms that need it).

1. **Add replay buffer infrastructure with uniform and prioritized sampling**
   Implements port #1. Pure storage and indexing; no trainer changes. Prerequisite for #2 and #3.

2. **Add DQN trainer with double-Q targets and target network**
   Implements port #2. Depends on issue #1. Includes a Q-network policy at `src/policy/q_network.rs` and end-to-end CartPole and Snake validation runs.

3. **Add `Environment::clone_state` / `restore_state` and use it from a smoke test**
   Smaller predecessor of port #5. Lands the trait change in isolation so the MCTS PR can focus on the search algorithm. Touches every env to implement the new methods.

4. **Add centralized-critic MARL trainer (MAPPO-style) under `src/multi_agent/`**
   Implements port #4. Extends `Experience` and `PolicyUpdate` in `src/multi_agent/messages.rs`, adds `src/multi_agent/centralized_critic.rs`, validates on bucket-brigade.

5. **Extend `Environment` trait to support continuous (`Box`) action spaces**
   Predecessor of port #3. Pick one of two strategies (associated `Action` type vs. parallel `ContinuousEnvironment` trait), document the rationale, and migrate existing envs. SAC depends on this.

6. **Add SAC trainer with twin critics, Gaussian policy head, and learned alpha**
   Implements port #3. Depends on issues #1 and #5. First continuous-control trainer in Thrust.

(Filing all six is fine, but the priority order is 1 -> 2 -> 4 in parallel with 3 -> 5 -> 6. Issues 4 and 5 are independent of each other.)
