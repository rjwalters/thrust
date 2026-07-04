# k\* Coordination-Threshold Sweep Across the Full (β, κ, c) Phase-Diagram Grid

**Date:** 2026-07-04
**Issue:** [#269](https://github.com/rjwalters/thrust/issues/269) (gate: [#268](https://github.com/rjwalters/thrust/issues/268), infrastructure: PR [#271](https://github.com/rjwalters/thrust/pull/271))
**Hardware:** alcubierre cluster — 6 × 32-core Linux x86_64 nodes (alc-2/5/6/8/9/10), release build, scripted policies only (no NN, no Burn, no PPO)
**Artifact:** [`data/2026-07-kstar-phase-diagram.json`](data/2026-07-kstar-phase-diagram.json) — 75 records keyed by `cell_tag`

## TL;DR

The coalition improvability gate (k\*: smallest coalition size `k ∈ 1..=4` whose
episode-level bootstrap CI on the team-return gap clears zero) was measured on
the **full 5 × 5 × 3 = 75-cell paper grid** (β ∈ {0.1, 0.3, 0.5, 0.7, 0.9},
κ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, c ∈ {0.5, 1.0, 2.0}) from
`compute_nash_phase_diagram.py`. Three headline findings:

1. **Every cell has a finite k\* ≤ 4** — there is no flat/near-degenerate cell
   anywhere on the grid. Distribution: **k\* = 1 on 15 cells, k\* = 2 on 45
   cells, k\* = 4 on 15 cells** (no cell lands at k\* = 3).
2. **k\* is a pure function of κ** (the solo-extinguish probability):
   κ = 0.1 → k\* = 4, κ ∈ {0.3, 0.5, 0.7} → k\* = 2, κ = 0.9 → k\* = 1.
   Work cost c shifts gap magnitudes slightly (monotonically up with c) but
   never moves k\*.
3. **The β axis is degenerate**: per-cell results are **bit-identical across
   all five β values** at every (κ, c). This is not a measurement artifact —
   `prob_fire_spreads_to_neighbor` is structurally dead in the engine's
   Bernoulli extinguish mode (see "The β axis measures nothing" below), which
   all cells here inherit via `minimal_specialization-v1`.

The #268 gate verdict is reproduced exactly: the canonical no-convergence cells
(κ = 0.1, c = 0.5) show k\* = 4 with gap **+33.24, 95% CI (+18.99, +50.14)** —
byte-identical to PR #271's headline numbers, cross-checked macOS-arm64 vs
Linux-x86_64.

## Protocol

Identical to the #268 gate, generalized to raw (β, κ, c) triples
(`run_phase_cell` in `src/multi_agent/bucket_brigade_oracle.rs`):

- 4 agents; `k` scripted coordinated deviators vs `N − k` frozen uniform
  opponents; scripted coalition battery (always-rest / specialist /
  owned- and any-house firefighters / randomized firefighter search with 64
  samples × 40 episodes / hero+specialists / rest+specialists heterogeneous
  mixes for k ≥ 2).
- 400 eval episodes per (cell, k), shared per-episode seed stream between the
  baseline and every candidate (paired comparison), seed 42, step cap 1000.
- Gap statistic: episode-mean of the per-episode per-step team-return gap
  (ceiling − all-uniform baseline); 1000-resample episode-level percentile
  bootstrap, α = 0.05.
- k\* rule: smallest `k` with CI lower bound strictly > 0.
- Base scenario `minimal_specialization-v1` with the three swept fields
  overridden — the same construction as the Python `_make_scenario`.

## Results

### k\* over the grid

k\* by (β, κ) — the table below holds **for every c ∈ {0.5, 1.0, 2.0}** and is
constant down each column (β-degeneracy):

| β \ κ | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|-------|-----|-----|-----|-----|-----|
| 0.1   | 4   | 2   | 2   | 2   | 1   |
| 0.3   | 4   | 2   | 2   | 2   | 1   |
| 0.5   | 4   | 2   | 2   | 2   | 1   |
| 0.7   | 4   | 2   | 2   | 2   | 1   |
| 0.9   | 4   | 2   | 2   | 2   | 1   |

### Per-k gap detail

Because results are bit-identical across β, the 15 (κ, c) combinations carry
the grid's full information (shown at β = 0.5; `gap_mean [CI_lo, CI_hi]`,
episode-mean per-step team-return gap):

| κ | c | k\* | k=1 | k=2 | k=3 | k=4 | ceiling @ k\* |
|-----|-----|----|-----------------------|-----------------------|--------------------------|--------------------------|---------------|
| 0.1 | 0.5 | 4 | −4.93 [−13.61, +1.97] | −4.76 [−13.22, +2.11] | +3.61 [−5.63, +13.75]    | +33.24 [+18.99, +50.14]  | search-best firefighter (any, work=0.485) |
| 0.1 | 1.0 | 4 | −4.67 [−13.36, +2.23] | −4.26 [−12.72, +2.61] | +4.35 [−4.89, +14.48]    | +34.17 [+19.92, +51.07]  | search-best firefighter (any, work=0.485) |
| 0.1 | 2.0 | 4 | −4.17 [−12.85, +2.74] | −3.26 [−11.72, +3.62] | +5.81 [−3.42, +15.93]    | +36.02 [+21.78, +52.92]  | search-best firefighter (any, work=0.485) |
| 0.3 | 0.5 | 2 | +8.93 [−2.96, +20.32] | +42.18 [+22.42, +62.81] | +47.74 [+28.25, +65.89] | +74.56 [+51.76, +97.84]  | all firefighter (owned, work=1.0) |
| 0.3 | 1.0 | 2 | +9.17 [−2.71, +20.57] | +42.65 [+22.89, +63.27] | +48.46 [+28.97, +66.60] | +75.50 [+52.71, +98.78]  | all firefighter (owned, work=1.0) |
| 0.3 | 2.0 | 2 | +9.66 [−2.21, +21.07] | +43.59 [+23.84, +64.20] | +49.89 [+30.40, +68.04] | +77.39 [+54.60, +100.65] | all firefighter (owned, work=1.0) |
| 0.5 | 0.5 | 2 | −4.37 [−16.82, +9.63] | +64.61 [+40.44, +87.88] | +349.12 [+312.71, +384.78] | +449.80 [+415.98, +482.15] | search-best firefighter (any, work=0.345) |
| 0.5 | 1.0 | 2 | −4.12 [−16.57, +9.88] | +65.08 [+40.92, +88.36] | +349.69 [+313.29, +385.33] | +450.51 [+416.70, +482.86] | search-best firefighter (any, work=0.345) |
| 0.5 | 2.0 | 2 | −3.61 [−16.06, +10.39] | +66.03 [+41.88, +89.30] | +350.81 [+314.45, +386.43] | +451.93 [+418.15, +484.28] | search-best firefighter (any, work=0.345) |
| 0.7 | 0.5 | 2 | −6.26 [−21.86, +8.84] | +403.17 [+369.17, +436.93] | +506.91 [+477.31, +533.27] | +527.88 [+499.93, +552.08] | all firefighter (any, work=1.0) |
| 0.7 | 1.0 | 2 | −6.01 [−21.61, +9.10] | +403.52 [+369.53, +437.26] | +507.42 [+477.83, +533.77] | +528.55 [+500.60, +552.75] | all firefighter (any, work=1.0) |
| 0.7 | 2.0 | 2 | −5.50 [−21.11, +9.61] | +404.23 [+370.27, +437.92] | +508.43 [+478.87, +534.77] | +529.89 [+501.93, +554.09] | all firefighter (any, work=1.0) |
| 0.9 | 0.5 | 1 | +400.53 [+366.35, +434.58] | +524.62 [+497.22, +551.36] | +541.72 [+514.62, +567.93] | +560.70 [+535.75, +587.87] | all firefighter (any, work=1.0) |
| 0.9 | 1.0 | 1 | +400.70 [+366.54, +434.76] | +524.94 [+497.56, +551.69] | +542.20 [+515.12, +568.40] | +561.55 [+536.61, +588.71] | all firefighter (any, work=1.0) |
| 0.9 | 2.0 | 1 | +401.05 [+366.91, +435.12] | +525.60 [+498.24, +552.35] | +543.18 [+516.12, +569.35] | +563.24 [+538.33, +590.39] | all firefighter (any, work=1.0) |

### Interpretation

- **κ is the coordination knob.** At κ = 0.9 a single deviating firefighter
  already clears the gate by ~+400 per-step team return per episode-mean —
  solo extinguishing works, so one competent agent lifts the whole team. At
  κ = 0.1 no coalition smaller than the full team of 4 is statistically
  distinguishable from uniform play, exactly the "trainability cliff" the
  slepian-wolf-marl-2 k\* reframe predicted for the no-convergence band. The
  intermediate κ band uniformly needs pairs.
- **The ceiling policy family tracks κ.** In the hard band (κ = 0.1) the best
  coalition is a *throttled* any-house firefighter (work ≈ 0.49 — working
  every night is a net loss when extinguishing rarely succeeds); by κ ≥ 0.7 it
  is the maximal any-house firefighter (work = 1.0).
- **c is a second-order effect**: raising the work cost slightly *raises* the
  measured gap at every (κ, k) (the all-uniform baseline pays the higher cost
  on wasted work more often than the coalition does) but never moves k\*.
- **For the downstream join** (`recalibrated_verdict.json` /
  `conditional_entropy.json` in rjwalters/bucket-brigade): joining on
  `cell_tag` is well-defined for all 75 records, but any β-structure found in
  those artifacts cannot be explained by env dynamics — see below.

## The β axis measures nothing (engine finding)

Per-cell results are **bit-identical** across β at every (κ, c) — verified by
hashing all 75 per-cell records with `cell_tag`/`beta` fields stripped (15
groups of 5 identical hashes). Root cause, in
`envs/bucket-brigade/bucket-brigade-core/src/engine/core.rs::step`:

```text
3. extinguish phase   — burning houses may become SAFE
4. burn-out phase     — every still-BURNING house becomes RUINED  (bernoulli mode)
5. spread phase       — fire spreads only from BURNING houses
6. spontaneous ignition
```

In the default `"bernoulli"` extinguish mode (which `minimal_specialization`
uses), the burn-out phase clears **all** BURNING houses before `spread_fires`
runs, so the spread loop's `houses[house_idx] == 1` guard never passes:
`prob_fire_spreads_to_neighbor` is dead code regardless of value. Only the
`"continuous"` extinguish mode (issue #253), whose burn-out returns early,
can ever exercise fire spread.

Implications:

- The "β = fire-spread rate" axis of the phase diagram does not vary the
  dynamics for any Bernoulli-mode scenario; the historical `beta01` /
  `beta05` / `beta09` no-convergence cells are one cell measured three times.
  (Consistent with the #268 gate quoting a single headline number for all
  three cells.)
- Any cross-β differences in downstream Nash/entropy artifacts computed on
  Bernoulli-mode scenarios are sampling noise, not β structure.

This is a pre-existing engine property, not introduced or fixed by this
sweep; it is tracked separately in
[#289](https://github.com/rjwalters/thrust/issues/289).

## Reproduction

Per-cell records were produced by the `PHASE=1` grid mode added to the
`br_oracle` example (this issue), distributed over 6 alcubierre nodes
(12–13 cells per node, one tmux session each, ~5 s of compute per cell after
the release build; zero node failures, zero re-runs):

```bash
# Full 75-cell grid on one host (writes one JSON per cell to phase_out/):
PHASE=1 K=4 EVAL_EPISODES=400 OUT_DIR=phase_out \
    cargo run --release --example br_oracle \
        --features "training,env-bucket-brigade"

# Arbitrary cell subset (the distribution knob used on the cluster):
PHASE=1 K=4 EVAL_EPISODES=400 OUT_DIR=out \
    CELLS="0.1,0.1,0.5;0.1,0.1,1.0" \
    cargo run --release --example br_oracle \
        --features "training,env-bucket-brigade"
```

Existing per-cell outputs are skipped on re-run, so partial failures lose one
cell, not the run. The committed artifact is the concatenation of the 75
per-cell records under a `protocol` / `grid` header (see
`data/2026-07-kstar-phase-diagram.json`; determinism means byte-identical
records regardless of which host produced them).

## Caveats

- The scripted-battery ceiling is a **lower bound** on improvability: a
  richer policy class could clear zero at smaller k. k\* here is "smallest k
  at which *this battery* provably beats uniform," the same operational
  definition the #268 gate used.
- The bootstrap CI is per-(cell, k); no multiple-comparison correction is
  applied across the 300 (cell, k) tests. With the κ-banded structure being
  15-fold replicated (β-degeneracy) and effect sizes far from the decision
  boundary everywhere except κ = 0.1 k = 3 (CI [−5.63, +13.75], comfortably
  spanning zero), correction would not change any k\*.
