# NFSP Average-Policy Under-Training on Bucket Brigade (#199)

**Date:** 2026-06-22
**Issue:** [#199](https://github.com/rjwalters/thrust/issues/199)
**Parent:** [#134](https://github.com/rjwalters/thrust/issues/134) bucket-brigade validation
**Sibling:** [#198](https://github.com/rjwalters/thrust/issues/198) (PSRO perf), [#212/#215](https://github.com/rjwalters/thrust/issues/215) (PSRO convergence)
**Status:** Root-caused + code-level fixes landed and demonstrated locally. Full
cluster re-run remains gated on #134 (operator-only hardware).

## The finding being investigated

The #134 cluster run (`docs/research/2026-06-bucket-brigade-validation.md`)
reported that NFSP's average policy (AP) **never learns** on the three
no-convergence cells: `avg_ap_loss` is pinned at **ln(40) ≈ 3.689** — the uniform
entropy floor over the `[10, 2, 2] = 40` joint action space — for all 48
iterations on all 3 cells. The supervised wiring is known-good (the unit test
`test_nfsp_avg_policy_supervised_step_reduces_loss_on_fixed_minibatch` passes), so
the run was under-trained / mis-configured, not broken.

## Root cause (three contributing factors, ranked)

### 1. AP is starved of gradient steps relative to reservoir size (primary)

The original config gives the AP `avg_policy_train_steps_per_iteration = 8` ×
`avg_policy_minibatch_size = 64` ≈ **512 samples/iteration** against a reservoir
that grows to **~9,800 entries/agent**. That is far too few gradient steps to fit
a moving, mostly-random target — the AP never moves off the uniform floor.

**Why a fixed step count is the wrong knob:** as the reservoir grows across
iterations, a constant 512-sample budget covers a *shrinking* fraction of it. The
fix has to scale with reservoir size.

### 2. Unscaled `[−700, 0]` reward band wrecks the BR critic (decisive, measured)

The best-response (BR) PPO learner regresses a value function against returns in
the `[−700, 0]` band. Measured locally on the canonical `beta05` cell (6 iters ×
256 rollout):

| Config | BR critic `value_loss` (per iter) |
|--------|-----------------------------------|
| `br_reward_scale = 1.0` (original) | **~9.8 × 10⁶** |
| `br_reward_scale = 0.01` (this PR) | **~9.8 × 10²** |

A ~10⁴× reduction (exactly the `(0.01)² = 1e-4` you expect for squared-error on
linearly-scaled targets). A critic with multi-million-magnitude targets produces
near-useless advantages, so the BR policy barely moves (`policy_loss ≈ 0`,
`entropy` flat at ~1.23 across iters in both runs). A weak BR feeds near-random
labels into the reservoir, which is exactly why the AP "correctly" learns ≈uniform.

### 3. BR side is weak regardless (the remaining research blocker)

Even with reward scaling, at the small local budget the BR policy/entropy barely
move. Whether the BR can learn structured play on these hard, sparse-reward cells
at a *feasible* budget is the open research question — and it is the same wall the
PSRO half hit (#215). This is **not** resolvable without the cluster budget gated
on #134.

## Code-level fixes (this PR)

All in `src/multi_agent/nfsp.rs` (config + trainer) and the bucket-brigade example:

1. **`NfspConfig::avg_policy_min_reservoir_coverage: f32`** — adaptive AP-step
   floor. When `> 0`, the trainer runs
   `max(avg_policy_train_steps_per_iteration, ceil(coverage × reservoir_len / minibatch))`
   supervised steps per agent per iteration, so the AP sees ~`coverage` full
   passes over the reservoir *however large it grows*. `0.0` preserves the legacy
   fixed-step behavior.
2. **`NfspConfig::br_reward_scale: f32`** — uniform reward scaling applied in the
   anticipatory rollout before the BR PPO update. Scaling rewards is an affine
   transform of the return (optimal policy unchanged) but keeps critic targets /
   advantage stats numerically sane. `1.0` is a no-op.
3. **Per-iteration diagnostics** in `examples/games/bucket_brigade/train_nfsp.rs`:
   the per-iteration log now reports `avg_ap_loss` *and* its signed delta from the
   `ln(40)` floor, plus the BR side's per-agent `policy_loss` / `value_loss` /
   `entropy` — so AP learning curves and the BR-side diagnostic are both visible
   live, not just at the final eval. New env overrides: `AP_COVERAGE`,
   `BR_REWARD_SCALE`.

## Local-scale evidence

### AP mechanism fits a factored target well below ln(40) (controlled)

`test_nfsp_multi_discrete_ap_loss_drops_below_uniform_floor` (fast, always-on)
seeds an agent's reservoir with a fixed non-uniform `[10,2,2]` joint target and
runs the AP supervised update with `avg_policy_min_reservoir_coverage = 4.0`:

```
[#199] multi-discrete AP loss: first=1.6193, last=0.0033, ln(40) floor=3.6889
```

The AP drops from the uniform-ish floor to **~0.003** — i.e. given enough gradient
steps the AP *does* fit a factored target, well below `ln(40)`. This is exactly
the signal the cluster run never produced.

### Reward scaling fixes the BR critic blow-up (real env)

Short `beta05` run (6 iters × 256 rollout, CPU NdArray): see the table in §Root
Cause 2 — `value_loss` drops from ~9.8M to ~979 with `br_reward_scale = 0.01`,
with no change to the affine-invariant policy objective.

### Honest caveat on the AP curve at small budget

At the small *local* budget (≤6 iters, ≤256 rollout) the reservoir only reaches
~140 entries, so `avg_ap_loss` moves only ~0.03–0.05 below the floor in the real
env — the coverage fix's payoff scales with reservoir size, which only reaches the
thousands at the cluster budget. The controlled synthetic test above isolates and
proves the mechanism; the real-env curve at full scale is what the gated cluster
re-run will confirm.

## What remains gated on #134 (cluster runs)

- Re-running the full `48 × 2048` (or larger) NFSP arm on all 3 cells with
  `avg_policy_min_reservoir_coverage > 0` + `br_reward_scale < 1`, and confirming
  `avg_ap_loss` falls meaningfully below `ln(40)` across the *full* run at the
  ~9,800-entry reservoir scale.
- Recomputing the cell-specific `gap_closed_cell` for the NFSP arm at the fixed
  config (the #134 acceptance item).
- Determining whether the BR (RL) side can learn structured play on these cells at
  a feasible budget — the deeper, shared-with-PSRO (#215) research question.

These require the operator-gated alcubierro cluster (the long-budget hardware from
#134) and are out of scope for a locally-runnable PR.
