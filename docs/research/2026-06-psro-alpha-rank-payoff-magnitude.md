# PSRO Non-Convergence on Bucket Brigade: α-rank Payoff-Magnitude Saturation (#215)

**Date:** 2026-06-22
**Issue:** [#215](https://github.com/rjwalters/thrust/issues/215)
**Related:** [#199](https://github.com/rjwalters/thrust/issues/199) (NFSP reward scaling),
[#134](https://github.com/rjwalters/thrust/issues/134) (cluster validation),
[#198](https://github.com/rjwalters/thrust/issues/198) / [#212](https://github.com/rjwalters/thrust/issues/212) (PSRO perf)
**Status:** Code-level root-cause fixes landed; full documented decreasing-exploitability
cluster run remains gated on #134.

## The finding being investigated

A capped PSRO calibration on alc-2 (`CELL=beta01`, 2048 rollout, payoff-eval
cap = 64) showed exploitability **increasing** across iterations rather than
decreasing:

| iter | pop | exploitability |
|------|-----|----------------|
| 1 | 2 | 6404 |
| 2 | 3 | 9253 |
| 3 | 4 | 8423 |
| 4 | 5 | 12029 |

PSRO is not solving the 4-player bucket-brigade game. Issue #215 frames two
blockers: (1) **non-convergence** (gating), and (2) the **α-rank solve cost**
(super-linear, follow-on, explicitly gated on #1). This writeup addresses #1 at
the code level and demonstrates the mechanism at a locally-runnable scale.

## Root cause (two contributing factors, both magnitude-driven)

The two PSRO halves — the **best-response (BR) learner** and the **α-rank
meta-solver** — are *both* miscalibrated for the bucket-brigade `[−700, 0]` payoff
band, which is ~3 orders of magnitude larger than the `{−1, +1}` matching-pennies
game on which both were validated.

### 1. α-rank fixation probability saturates on the `[−700, 0]` band (PSRO-specific, decisive)

α-rank builds a Markov chain over joint pure strategies whose transition
probabilities are Moran fixation probabilities

```text
ρ(α, m, δ) = (1 − exp(−α δ)) / (1 − exp(−m α δ)),   δ = π_τ − π_σ
```

driven by `α · δ`. The defaults (`α = 10`, `m = 50`) were validated on
matching-pennies, where `|δ| ≤ 2` so `|α δ| ≤ 20` — comfortably inside the regime
where `ρ` is a *graded* function of the payoff advantage. On the bucket-brigade
band, `|δ|` reaches **~700**, so `|α δ| ≈ 7000`. Every non-neutral transition
**saturates** to a hard 0 or 1: the graded Moran dynamics collapse into a
degenerate deterministic best-response graph whose stationary distribution is
acutely sensitive to tiny payoff-estimate noise. That brittleness — the meta-Nash
mixture flipping between near-degenerate distributions as the empirical payoffs
wobble iteration-to-iteration — is a direct mechanism for exploitability that
*increases* instead of decreasing.

**Measured (unit test, `test_alpha_rank_span_normalization_is_magnitude_invariant`):**
a 2×2 symmetric game in which strategy 0 strictly dominates is solved at two
scales with identical ordering.

| Payoff scale | α-rank mass on dominant strategy (unnormalized) |
|--------------|--------------------------------------------------|
| `±2` (unit) | **> 0.9** (correct — concentrates on the dominant strategy) |
| `±700` (bucket-brigade band) | **≈ 0.5** (wrong — saturation erases the dominance signal) |

At the large scale the *unnormalized* solver returns a near-uniform distribution
even though one strategy strictly dominates: the solver has stopped tracking the
strategy ordering entirely. This is not a slow-convergence artifact; it is the
fixation probability degenerating.

### 2. Unscaled `[−700, 0]` reward band wrecks the BR critic (shared with #199)

The PSRO BR is trained with the *same* joint PPO update as NFSP's BR side, so it
inherits the identical pathology #199 measured on the NFSP arm: a value function
regressed against returns in `[−700, 0]` produces multi-million-magnitude critic
targets and near-useless advantages, so the BR policy barely moves. A weak BR
appends near-random policies to the population, which cannot reduce
exploitability. #199 fixed this on NFSP with `br_reward_scale`; this PR adds the
analogous knob to PSRO.

## Code-level fixes (this PR)

All in `src/multi_agent/psro.rs` (config + trainer + meta-solver) and the
bucket-brigade example.

1. **`PsroConfig::br_reward_scale: f32`** — uniform reward scaling applied to the
   BR rollout rewards in `train_best_response` before the joint PPO update,
   mirroring [`NfspConfig::br_reward_scale`](../../src/multi_agent/nfsp.rs). Scaling
   rewards is an affine transform of the return (optimal policy unchanged) but
   keeps the critic's regression targets and advantage statistics numerically
   sane. `1.0` (default) is a bit-identical no-op.

2. **`AlphaRankMetaSolver::normalize_payoff_span: bool`** (builder:
   `with_payoff_span_normalization`) — when enabled, every Moran payoff
   differential `δ` is divided by the payoff **span** (`max − min` over the input
   tensor) before multiplying by `α`, so the effective selection strength
   `α · (δ / span)` lands in the same `[−α, α]` band the defaults were tuned for
   *regardless of the absolute payoff magnitude*. This is the α-rank analogue of
   `br_reward_scale`: a magnitude-invariance fix, not a change to the ranking
   semantics on a fixed scale. A degenerate zero/non-finite span falls back to a
   unit divisor (no-op). `false` (default) is bit-identical to the pre-#215 solver
   — confirmed by `test_alpha_rank_span_normalization_default_off_is_bit_identical`
   and the existing `test_psro_exploitability_trace_is_bit_identical` reproducibility
   bar.

3. **Example knobs** in `examples/games/bucket_brigade/train_psro.rs`: new env
   overrides `BR_REWARD_SCALE` and `ALPHA_RANK_NORMALIZE_SPAN`, both logged at
   startup. The per-iteration `exploitability` is already logged live via the
   `on_iteration` callback (#202), so divergence-vs-convergence is visible during
   a run.

## Local-scale evidence

### α-rank magnitude invariance (controlled, fast, always-on)

`test_alpha_rank_span_normalization_is_magnitude_invariant` (in
`src/multi_agent/psro.rs`) isolates and proves the mechanism:

- **Unnormalized:** the `±2` game concentrates correctly (> 0.9 on the dominant
  strategy); the *same-ordering* `±700` game collapses to ≈ uniform (< 0.6) — the
  saturation bug.
- **Span-normalized:** the `±700` game recovers the same concentrated answer as
  the `±2` game (within `1e-3` per-entry), and still puts > 0.9 on the dominant
  strategy. Magnitude no longer changes the answer.

This is exactly the brittleness that would make the meta-Nash mixture (and hence
the reported exploitability) jump around on the large-payoff cells.

### End-to-end plumbing (controlled, fast, always-on)

`psro_n4_issue215_knobs_run_and_are_finite` (in
`tests/test_psro_n_player_matching_pennies.rs`) drives a tiny-budget N=4 PSRO run
with *both* knobs set to non-default values (`br_reward_scale = 0.01`,
span-norm on) and asserts the NashConv curve stays finite/non-negative with valid
per-agent simplex marginals — guarding the plumbing on every CISC run.

### Real-env local run (beta01, small budget)

A 3-iteration `beta01` run (128 rollout, CPU NdArray, `target/release`) compares
the default trainer against both knobs on. The headline is the **trend** of the
per-iteration α-rank exploitability (NashConv on the empirical meta-game):

| iter | pop | baseline (no knobs) | both knobs (`BR_REWARD_SCALE=0.01`, `ALPHA_RANK_NORMALIZE_SPAN=1`) |
|------|-----|---------------------|--------------------------------------------------------------------|
| 1 | 2 | 6412.0 | 8363.9 |
| 2 | 3 | 7002.6 (↑) | 7612.7 (↓) |
| 3 | 4 | 6447.1 (↓) | 5344.9 (↓) |
| **trend** | | **6412 → 7003 → 6447 — non-monotone, no descent** | **8364 → 7613 → 5345 — monotonically decreasing** |

The baseline reproduces the cluster non-convergence *locally*: exploitability
**increases** on the first step (6412 → 7003) and never establishes a downward
trend. With both magnitude fixes the curve is **monotonically decreasing**
(8364 → 7613 → 5345) over the same budget — the qualitative behavior the issue
asks for.

Caveats, stated honestly:
- The absolute NashConv values are **not comparable across the two columns** —
  NashConv is computed on the raw `[−700, 0]` payoffs, and span normalization
  changes only the *meta-solver's* internal selection strength, not the payoff
  scale the metric is reported on. Read the **per-column trend**, not the
  cross-column magnitude.
- This is a **3-iteration, 128-rollout** run on CPU — an *observability* check
  that the fixes move the trend in the right direction, **not** a converged
  result. `gap_closed_cell` is still far from the PPO reference in both runs (the
  BR side has barely trained at this micro-budget — the deeper #199-shared
  question). A documented, multi-iteration decreasing curve *to a competitive
  `gap_closed_cell`* at a meaningful budget requires the cluster (#134).

The local budget (few iterations, 256 rollout) is far below the cluster scale;
the numbers above are an *observability* check that the knobs change the run in
the expected direction, not a convergence claim. A documented decreasing-
exploitability run at a meaningful budget requires the cluster (see below).

## What remains gated on #134 (cluster runs)

- A documented PSRO run where exploitability **decreases** over iterations on at
  least one bucket-brigade cell, with `br_reward_scale < 1` and α-rank span
  normalization on. The full-scale `population⁴` α-rank solve + BR training at a
  feasible budget needs the operator-gated alcubierro cluster.
- Determining whether, *with* both magnitude fixes, the BR (RL) side can learn
  structured play on these hard, sparse-reward cells at all — the deeper,
  shared-with-NFSP (#199) research question. A well-evidenced negative result
  ("α-rank PSRO is unsuited to this payoff structure because the BR cannot learn
  at feasible budget") is an acceptable outcome per #215.

## Blocker #2 (α-rank solve cost) — explicitly deferred

Per #215, α-rank *performance* work is gated on convergence being established
first, so this PR does **not** optimize it. For the record, the next step once
convergence is shown would be to bound the `population⁴`-state stationary solve:
either (a) a sparse/iterative eigensolve that exploits the response graph's
single-agent-deviation sparsity (each state has only `N·(k−1)` out-edges, already
the representation in `solve_n_player_impl`), or (b) capping the response-graph
size by pruning dominated joint strategies before the power iteration. Both are
follow-on perf work, not convergence work.

## Reproduction

```bash
# Fast, always-on mechanistic + plumbing tests:
cargo test --features training --lib multi_agent::psro::tests::test_alpha_rank_span
cargo test --features training --test test_psro_n_player_matching_pennies \
    psro_n4_issue215_knobs_run_and_are_finite

# Local real-env run (requires the bucket-brigade env feature):
TOTAL_ITERATIONS=5 ROLLOUT_STEPS=256 CELL=beta01 CHECKPOINT_INTERVAL_ITERATIONS=0 \
  BR_REWARD_SCALE=0.01 ALPHA_RANK_NORMALIZE_SPAN=1 \
  cargo run --release --example train_psro --features "training,env-bucket-brigade"
```
