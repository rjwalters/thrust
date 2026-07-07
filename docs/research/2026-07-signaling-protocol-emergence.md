# SignalingGame Phase 1–2 Experiment: Does a Discrete Protocol Emerge?

**Date:** 2026-07-06
**Issue:** [#304](https://github.com/rjwalters/thrust/issues/304) (epic: [#266](https://github.com/rjwalters/thrust/issues/266); adjudicates the [#276](https://github.com/rjwalters/thrust/issues/276) deferral)
**Hardware:** macOS arm64, release build, NdArray (CPU) backend, single host
**Artifact:** [`data/2026-07-signaling-protocol-emergence.json`](data/2026-07-signaling-protocol-emergence.json) — 6 arms × 3 seeds
**Harness:** `examples/games/signaling/signaling_protocol.rs` (this issue; wires shipped pieces only, no core `src/` changes)

## TL;DR — Verdict

**Partial emergence, with two diagnosed bottlenecks — one structural, one that
directly re-motivates #276.**

1. **On the shipped single-shot surface, no protocol can emerge — structurally.**
   The shipped `impl JointEnv for SignalingGame` terminates every step, and
   `collect_rollout` resets on `done`, so the speaker's message is erased
   before the listener ever observes it. The comms-on and comms-ablated arms
   are **bit-identical** (reward 0.248 ± 0.002 = the 1/V chance floor). The
   channel is dead weight, not merely unused.
2. **When the message actually reaches the listener** (an episode-persistence
   wrapper around the unmodified env; `episode_len = 8`), **a load-bearing
   protocol partially emerges**: reward 0.677 ± 0.172 vs 0.246 ± 0.008 for the
   ablated control (chance = 0.25), end-to-end `I(referent; guess)` = 1.22 of
   2.00 bits. Per-seed spread is wide — the best seed reaches 0.885 reward /
   1.88 bits (a near-perfect 4-token protocol), the worst 0.464 / 0.58 bits —
   the classic joint-exploration instability of discrete Lewis games.
3. **The Phase 2 comms-loss hook cannot shape the protocol.** The
   `comms_coef ∈ {0, 0.01, 0.1}` sweep produced **bit-identical** trained
   policies in every arm. This is by construction: the Phase 2 penalty is
   computed from sampled discrete tokens and folded into the joint loss as a
   gradient-free constant (`joint.rs::comms_penalty`, documented as
   "monitored regularizer"). Confirmed empirically here.

Both bottlenecks are precisely the evidence class #276 (differentiable
Gumbel-softmax comms) was deferred to wait for: the discrete channel trains
only via the sparse shared reward, converges seed-dependently at V = 4 in the
easiest possible setting, and the existing regularization hook has no gradient
pathway to help. **#276's motivation stands** (with a prerequisite: the
turn-structure fix below is needed before any comms mechanism can matter on
the shipped surface).

## Setup

The smallest possible instantiation of the epic-#266 question, using only
shipped v0.3.0 infrastructure:

- **Env:** `SignalingGame` (#292) — 2 agents, vocab V = 4. The speaker
  observes a hidden referent (one-hot) and emits one message token; the
  listener observes the received token and guesses the referent. Both get +1
  iff the guess is correct. Message routing runs through the shipped
  `split_action` / `place_message` comms helpers inside the env's own
  `step_multi` — the harness never touches tokens.
- **Trainer:** `JointMultiAgentTrainer` (#295) — joint PPO, one
  `MultiDiscreteMlpBurnPolicy` per agent (hidden_dim 32, LR 3e-3, γ 0.9,
  ent_coef 0.01, 4 epochs, minibatch 64, all-minibatch iteration), 128-step
  rollouts × 500 iterations ≈ 64k env steps per seed. Seeds {1000, 1001,
  1002}; fully deterministic per seed.
- **Harness wrapper (`Arena`):** the shipped `JointEnv` impl for
  `SignalingGame` keeps the referent fixed and terminates every step, which
  makes it degenerate as a *learning* task (one referent to name, and the
  message erased by the reset before delivery — see Diagnosis). The wrapper
  (a) redraws the referent uniformly each episode (the definition of a Lewis
  game; the env's own `set_hidden` API exists for this) and (b) optionally
  holds an episode open for `episode_len` steps so the message persists into
  the listener's next observation. `episode_len = 1` reproduces the shipped
  single-shot surface exactly.

### Arms

| arm | episode_len | channel | comms_coef |
|-----|-------------|---------|-----------|
| `single_shot_comms_on` | 1 | live | 0 |
| `single_shot_ablated` | 1 | severed | 0 |
| `persistent_comms_on` | 8 | live | 0 |
| `persistent_ablated` | 8 | severed | 0 |
| `persistent_coef_default` | 8 | live | 0.01 |
| `persistent_coef_high` | 8 | live | 0.1 |

"Severed" = the listener's received-message slot is forced to the `-1`
sentinel every step (the no-comms control from the issue: if reward is
identical with the channel ablated, no protocol is load-bearing).

### Metrics

Protocol emergence is measured per the COMMS_DESIGN §5 intent (mutual
information between state, message, and action), via direct policy probes
(300 samples/condition) after training:

- `I(T;M)` — referent vs emitted message (does the speaker *encode*?);
- `I(M;G)` — received token vs guess (does the listener *decode*?);
- `I(T;G)` — the composed end-to-end channel `p(g|t) = Σ_m p(m|t)p(g|m)`;
- `decode ceiling` — accuracy an informed listener would achieve,
  `Σ_t p(g=t|t)/V` (separates protocol quality from turn-structure penalty);
- **reward** — mean rollout reward over the final quarter of training (what
  PPO actually optimizes, turn structure included). Chance = 1/V = 0.25;
  max MI = log2(V) = 2 bits.

## Results

6 arms × 3 seeds, 500 iterations each (~7 min total wall-clock, CPU):

| arm | reward (final ¼) | I(T;M) | I(M;G) | I(T;G) | decode ceiling |
|-----|------------------|--------|--------|--------|----------------|
| `single_shot_comms_on` | 0.248 ± 0.002 | 0.57 | 0.52 | 0.29 | 0.245 |
| `single_shot_ablated` | 0.248 ± 0.002 | 0.57 | 0.52 | 0.29 | 0.245 |
| `persistent_comms_on` | **0.677 ± 0.172** | 1.47 | 1.32 | **1.22** | 0.744 |
| `persistent_ablated` | 0.246 ± 0.008 | 0.88 | 0.10 | 0.01 | 0.256 |
| `persistent_coef_default` | 0.677 ± 0.172 | 1.47 | 1.32 | 1.22 | 0.744 |
| `persistent_coef_high` | 0.677 ± 0.172 | 1.47 | 1.32 | 1.22 | 0.744 |

Mean learning curves (rollout reward, every 50th iteration):

```
single_shot_comms_on     0.24 0.26 0.25 0.23 0.26 0.23 0.24 0.20 0.27 0.24
single_shot_ablated      0.24 0.26 0.25 0.23 0.26 0.23 0.24 0.20 0.27 0.24
persistent_comms_on      0.25 0.38 0.41 0.59 0.48 0.71 0.67 0.66 0.68 0.60
persistent_ablated       0.25 0.22 0.21 0.35 0.28 0.30 0.20 0.27 0.22 0.24
persistent_coef_*        (bit-identical to persistent_comms_on)
```

Per-seed detail for the headline arm (`persistent_comms_on`):

| seed | reward | I(T;M) | I(T;G) |
|------|--------|--------|--------|
| 1000 | 0.464 | 1.01 | 0.58 |
| 1001 | 0.885 | 1.97 | 1.88 |
| 1002 | 0.683 | 1.43 | 1.20 |

Seed 1001 is a near-perfect protocol (1.97/2.00 encode bits, 94% decode
ceiling); seed 1000 converged to a partial 2-of-4-token code. Full per-seed
tables and curves are in the JSON artifact.

## Diagnosis

### Bottleneck 1 (structural): the shipped single-shot surface never delivers the message

`SignalingGame::step_multi` terminates every step (`terminated = [true,
true]`), and `JointMultiAgentTrainer::collect_rollout` calls `reset_joint`
whenever `done` — which clears `received` back to the `-1` sentinel. Under
simultaneous moves the listener's observation at the moment it acts is
therefore *always* the sentinel; the emitted message exists only inside the
post-step observation that the reset immediately discards. The comms-on and
ablated single-shot arms being **bit-identical** (same RNG stream, same
trajectories) proves the channel is unreachable, not merely unlearned.

This is not a bug in any one component — env, comms helpers, and trainer each
do what they say — but their composition gives the listener no step on which a
message is observable. Any future comms experiment on this env needs either
multi-step episodes (as this harness's wrapper does) or a sequential
speaker-then-listener turn inside one step.

### Bottleneck 2 (learning): discrete-channel joint exploration is seed-unstable, and the Phase 2 hook has no gradient to stabilize it

With delivery fixed, the protocol must bootstrap from a sparse shared reward:
the listener's decode gradient is noise until the speaker encodes, and the
speaker's encode gradient is noise until the listener decodes (a moving-target
/ credit-assignment loop). At V = 4 — the easiest non-trivial setting — 500
iterations produced one near-perfect, one good, and one partial protocol.

The comms-loss-weight sweep confirms the shipped mitigation cannot help:
`comms_coef` folds a token-entropy penalty into the joint loss **as a
constant** (no gradient, by documented Phase 2 design), and the trained
policies at coef 0 / 0.01 / 0.1 are bit-identical. Sender-side gradient
pathways — exactly #276's Gumbel-softmax proposal — are the mechanism this
experiment shows is missing.

### What was NOT done (honest-reporting guardrail)

No reward shaping, no curriculum, no vocabulary shrinking, no hyperparameter
search beyond the documented defaults (LR 3e-3 and γ 0.9 were chosen a priori
for the tiny env and short horizon; nothing was tuned against the verdict).
The single-shot arms are reported as observed — a negative result on the
shipped surface — rather than adjusted until they complied.

## Implications

- **#276 (differentiable comms): motivation confirmed, not retired.** The
  discrete channel does carry a protocol when delivery works, but convergence
  is unreliable even at V = 4, and the only shipped shaping hook is
  gradient-free by construction. Gumbel-softmax sender gradients are the
  natural next mechanism. Scoping note: #276 should *also* fix delivery
  semantics (bottleneck 1), or it will inherit a surface where no comms
  mechanism can matter.
- **#266 (epic): Phases 1–2 are functionally validated.** Message routing
  (`split_action`/`place_message`), heterogeneous per-agent obs dims in the
  joint trainer, and the comms-coef plumbing all behaved exactly as specified
  under a real training load.
- **Possible cheap follow-up** (not required by this issue): a multi-round
  `JointEnv` surface for `SignalingGame` (referent redraw + persistent
  message) upstreamed from this harness's `Arena`, so future runs don't need
  a wrapper.

## Reproduction

```bash
# Full run (6 arms x 3 seeds x 500 iterations, ~7 min on an M-series CPU):
cargo run --release --example signaling_protocol --features training

# Quick smoke:
ITERATIONS=50 SEEDS=1 cargo run --release --example signaling_protocol \
    --features training

# Custom output path (default: docs/research/data/2026-07-signaling-protocol-emergence.json):
OUT=/tmp/out.json cargo run --release --example signaling_protocol --features training
```

Deterministic per seed (seeded policy init, seeded referent stream, seeded
action sampling); re-running reproduces the committed artifact's per-seed
numbers on the same backend.

## Caveats

- **3 seeds** bounds the seed-variance claim loosely; the qualitative verdict
  (works sometimes, never fully reliably) is robust across all three, but the
  0.677 ± 0.172 mean should not be over-read.
- The MI probe composes `p(g|t)` from independently probed `p(m|t)` and
  `p(g|m)` (valid here because the listener is memoryless and sees only the
  token); it measures the *learned protocol's* capacity, deliberately
  excluding the turn-structure penalty that depresses raw reward (the first
  step of every episode is blind, capping expected reward at
  `(0.25 + 7) / 8 ≈ 0.91` for `episode_len = 8` under a perfect protocol —
  consistent with the best seed's 0.885 measured on the final quarter).
- Single host, CPU NdArray backend; no cross-backend replication was
  attempted (the run is minutes, so this is cheap to redo elsewhere).
