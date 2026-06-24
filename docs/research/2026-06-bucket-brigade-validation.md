# Bucket-Brigade Research Validation — PSRO / NFSP on the No-Convergence Cells

**Date:** 2026-06-20 / 21
**Issue:** [#134](https://github.com/rjwalters/thrust/issues/134)
**Hardware:** alcubierro cluster — Intel Core i9-14900K (32 threads) nodes, Ubuntu 24.04, Burn 0.21 **NdArray (CPU)** backend
**Status:** Negative result + two engineering blockers (see TL;DR)

## TL;DR

The research question — *do PSRO with α-rank or approximate NFSP beat PPO on the
Bucket Brigade workshop paper's no-convergence cells?* — is answered **No, not at
any budget reachable with the current implementation.**

- **NFSP** ran the full job (3 cells × 48 outer iterations × 2048 rollout steps)
  but **does not learn**: the average-policy supervised loss is pinned at
  **ln(40) ≈ 3.689** (uniform random over the 40-action joint space) for all 48
  iterations across **all three cells**, giving `gap_closed ≈ −7.0` versus PPO's
  `−0.049` reference. No cell achieves `gap_closed > 0`.
- **PSRO** could **not be run** at a meaningful budget — single-threaded, with a
  `population⁴` α-rank payoff tensor and an opaque `run()` (no progress output,
  no mid-run checkpointing). Even **4 iterations did not complete in >40 min**.
  Tracked as a perf/observability blocker: **[#198](https://github.com/rjwalters/thrust/issues/198)**.

This satisfies the #134 acceptance criterion of *"a clear documented finding that
neither [trainer] does [achieve `gap_closed > 0`] at this budget (a separate
research thread)."* No canonical artifact is promoted (no winner).

## Cells

The three no-convergence cells from the workshop paper (κ=0.1, c=0.5 fixed;
β = `prob_fire_spreads_to_neighbor`):

| Cell | β | κ | c |
|------|-----|-----|-----|
| `beta01` | 0.1 | 0.1 | 0.5 |
| `beta05` | 0.5 | 0.1 | 0.5 |
| `beta09` | 0.9 | 0.1 | 0.5 |

## NFSP results

Full runs: **48 outer iterations × 2048 rollout steps**, one cell per node
(alc-4/5/6), CPU NdArray backend, deterministic eval over `EVAL_STEPS = 200`.

| Cell | AP `gap_closed` | AP per-step team | BR `gap_closed` | BR per-step team | wall-clock |
|------|-----------------|------------------|-----------------|------------------|-----------|
| `beta01` | **−6.991** | −546.65 | −9.991 | −743.62 | 108 min |
| `beta05` | **−6.991** | −546.65 | −9.991 | −743.62 | 144 min |
| `beta09` | **−6.990** | −546.64 | −9.991 | −743.62 | 112 min |
| *PPO (paper)* | *−0.049* | — | — | — | — |

AP = average policy (NFSP's paper-recommended deploy artifact); BR = best response.

### The runs are degenerate, not just unconverged

Three signals show NFSP never learned anything on these cells:

1. **`avg_ap_loss` is pinned at ln(40) ≈ 3.689 for all 48 iterations.** The joint
   action space is `[10, 2, 2] = 40` actions, and `ln(40) = 3.6889`. The reported
   per-iteration average supervised loss never moves off the uniform-random
   entropy floor — the average-policy network does not fit the reservoir at all.
2. **All three cells produce bit-near-identical results** — identical
   `per_step_team` to 3 decimals and **identical reservoir sizes to the exact
   integer** (`[9826, 9716, 9768, 9932]`). Eval *is* cell-specific
   (`make_cell_env(beta, …)`), so this is a *consequence* of a uniform policy:
   under random play, β barely moves the team reward. It is not a cell-plumbing bug
   (logs confirm β = 0.1 / 0.5 / 0.9 were applied).
3. **BR is worse than AP** (`−9.99` vs `−6.99`): the RL best-response side is also
   not producing useful structure, so the reservoir it feeds the average policy is
   near-random — the average policy then "correctly" learns ≈uniform.

### Likely cause (lead for follow-up, not yet root-caused)

The average policy is almost certainly **under-trained**, not broken: a passing
unit test (`test_nfsp_avg_policy_supervised_step_reduces_loss_on_fixed_minibatch`)
confirms one supervised step *does* reduce loss on fixed data. But the run config
gives the average policy only `avg_policy_train_steps_per_iteration = 8` ×
`avg_policy_minibatch_size = 64` ≈ **512 samples/iteration** against a reservoir
that grows to **~9,800 entries/agent** — far too few gradient steps to fit a
moving, mostly-random target. Combined with a BR side that isn't learning on this
hard, sparse, large-magnitude-reward (`[−700, 0]`) env, nothing converges.

Candidate follow-up experiments: raise `avg_policy_train_steps_per_iteration`
substantially; diagnose/strengthen the BR (RL) side first (a weak BR starves the
AP target); revisit reward scaling for the `[−700, 0]` payoff band.

## PSRO: could not be run (blocked)

PSRO never completed a usable run. Calibration on alc-2 (release, CPU NdArray):

| Probe (PSRO, beta01) | Result |
|----------------------|--------|
| 6 iters × 2048 | **>85 min, never completed**, zero iteration output |
| 4 iters × 2048 | **>40 min, never completed**, zero iteration output |

Root causes (all in the PSRO path):

- **Single-threaded** (`src/multi_agent/psro.rs` has no rayon) — pins one of 32 cores.
- **α-rank payoff tensor scales as `population^num_agents` = `population⁴`**, one
  rollout per entry; cost grows superlinearly with iterations.
- **`PsroTrainer::run()` is opaque** — no per-iteration progress and **no mid-run
  checkpointing**; a killed run yields nothing.

This is not a budget-tuning problem and was tracked separately as a perf +
observability pass: **[#198](https://github.com/rjwalters/thrust/issues/198)** (now done).

### Update (2026-06-21): post-#198 parallel calibration

#198 shipped: the payoff-tensor evaluation is now **rayon-parallel** (bit-identical
to serial, [#207](https://github.com/rjwalters/thrust/issues/203)), PSRO emits
**live per-iteration logging** ([#202](https://github.com/rjwalters/thrust/issues/202)),
and writes **mid-run checkpoints** ([#204](https://github.com/rjwalters/thrust/issues/204)).
Re-calibrated on alc-2 (33 threads active, `CELL=beta01`, 2048 rollout):

| iter | population | payoff evals (pop⁴) | wall-clock |
|------|-----------|---------------------|-----------|
| 1 | 2 | 16 | ~180 s |
| 2 | 3 | 81 | ~472 s |
| 3 | 4 | 256 | >620 s (killed mid-iter) |

**Parallelization works** — all 33 cores utilized, iter 1 in ~3 min where the
single-threaded baseline couldn't finish 4 iters in 40 min — **but it is not
sufficient.** Per-iteration cost grows ~2.6×/iter because the α-rank payoff tensor
is `population⁴`; a constant ~33× core speedup cannot beat super-linear growth, so a
12-iteration run still projects to hours-to-days. The remaining lever is **caching
the reusable payoff entries** (PSRO adds one policy/agent/iter, so most of the
`population⁴` tensor is unchanged between iterations → only the new-policy slabs
need re-evaluation): **[#212](https://github.com/rjwalters/thrust/issues/212)**.

Secondary observation: exploitability *increased* across the calibration iterations
(6404 → 8864), i.e. PSRO did not converge on bucket-brigade — a sibling concern to
the NFSP non-convergence ([#199](https://github.com/rjwalters/thrust/issues/199)),
to revisit once the payoff cost is tractable.

### Update (2026-06-21, cont.): payoff-cap calibration — convergence is the real blocker

[#212](https://github.com/rjwalters/thrust/issues/212) shipped an opt-in
`max_payoff_evals_per_iteration` cap ([#214](https://github.com/rjwalters/thrust/issues/214))
to bound the boundary-slab cost. Re-calibrated on alc-2 (33 threads, `CELL=beta01`,
2048 rollout, **cap=64**, 24 iterations requested):

| iter | population | exploitability | iteration wall-clock |
|------|-----------|----------------|---------------------|
| 1 | 2 | 6404 | — |
| 2 | 3 | 9253 | 8m03s |
| 3 | 4 | 8423 | 13m06s |
| 4 | 5 | 12029 | 23m18s |

Two conclusions, both decisive for the PSRO half of #134:

1. **The cap is necessary but not sufficient.** From iter 2 on, payoff evals are
   pinned at 64, yet per-iteration time still grows ~1.6×/iter (8→13→23 min). So the
   payoff tensor was not the *only* super-linear cost — the remaining growth is the
   **α-rank stationary-distribution solve over the `population⁴` response graph**
   (plus best-response training), which the payoff-eval cap does not touch.
2. **PSRO does not converge here.** Exploitability *increases* (6404 → 12029) rather
   than decreasing, with or without the cap. Cost optimization is moot if the
   algorithm doesn't converge on this game.

**Net:** after parallelization (#198), observability (#202), checkpointing (#204),
and the payoff cap (#212), PSRO on the 4-player bucket-brigade is *both* still slow
(compounding super-linear costs) *and* non-convergent. The gating question is no
longer cost but **whether PSRO can converge on these cells at all** — a research
question, tracked alongside the α-rank-solve cost in
[#215](https://github.com/rjwalters/thrust/issues/215). This mirrors the NFSP
non-convergence (#199): **neither trainer converges on the no-convergence cells**,
which is itself the answer to #134's research question at the budgets explored.

> **Update (2026-06-22, #215):** root-caused at the code level — see
> [`2026-06-psro-alpha-rank-payoff-magnitude.md`](./2026-06-psro-alpha-rank-payoff-magnitude.md).
> The α-rank Moran fixation probability **saturates** on the `[−700, 0]` band
> (`α·δ ≈ 7000` vs the `≤ 20` it was tuned for on matching-pennies), collapsing the
> meta-solver into a brittle hard-max — a direct mechanism for diverging
> exploitability. `AlphaRankMetaSolver` gained an opt-in
> `with_payoff_span_normalization` knob (restores magnitude invariance) and
> `PsroConfig` gained `br_reward_scale` (mirrors #199's BR critic fix). A unit test
> shows a strictly-dominant strategy that the unnormalized solver correctly
> concentrates on at unit scale degrades to ≈uniform at the `±700` scale, and span
> normalization recovers the correct concentrated answer. A documented
> decreasing-exploitability *cluster* run remains gated on #134.

## Budget note: 8192 rollout is impractical for NFSP on this hardware

The #134-recommended `48 × 8192` budget proved infeasible. At 8192 rollout, NFSP
iterations ran **40–49 min each and growing** (vs ~150 s/iter, flat, at 2048),
projecting to **≥36 h per run**. Per-iteration cost is superlinear in rollout size
here (rollout collection + larger reservoir + per-iter RL/SL updates). The runs
above therefore use **2048 rollout**, where iteration time is flat (~150 s) and a
full 48-iteration run finishes in ~2 h. Given NFSP's flat learning curve, more
*iterations* at the cheaper rollout is the more informative lever than a 4× larger
rollout; a future longer run gated on the perf/learning fixes can revisit 8192.

## Hyperparameters

NFSP (`examples/games/bucket_brigade/train_nfsp.rs`):

| Knob | Value |
|------|-------|
| `max_iterations` | 48 |
| `rollout_steps` | 2048 |
| `anticipatory_param` (η) | 0.1 |
| `reservoir_capacity` | 16,384 |
| `br_train_steps_per_iteration` | 1 |
| `avg_policy_train_steps_per_iteration` | 8 |
| `avg_policy_minibatch_size` | 64 |
| `avg_policy_lr` | 5e-3 |
| BR optimizer (Adam) lr | 3e-4 |
| `n_epochs` (BR) | 4 |
| `minibatch_size` (BR) | 256 |
| `hidden_dim` | 64 |
| `num_agents` | 4 |
| seed | 42 |

## Caveats

- **Metric is base-scenario, not cell-specific.** The example's `gap_closed` uses
  base-scenario baselines, flagged in-run as *"baselines are base-scenario, not
  cell-specific (#128); treat as diagnostic."* Because NFSP collapses to a uniform
  policy, the cell-specific metric would not change the qualitative conclusion
  (no learning, far below PPO), but a cell-specific recomputation is the correct
  next step once a policy that actually learns exists.
- **CPU backend.** Per the GPU benchmark ([`BURN_BACKENDS.md`](../BURN_BACKENDS.md)),
  NdArray (CPU) is the right backend for these small MLPs (wgpu was 4–9× slower).

## Acceptance-criteria status (#134)

- [x] Training runs attempted on the cluster with full logs captured — **NFSP: 3 cells
      complete**; **PSRO: blocked (#198)**.
- [x] Per-policy deterministic evaluation with `gap_closed` recorded (base-scenario; see caveat).
- [x] Results table + writeup in `docs/research/` (this document).
- [x] **Documented finding that neither trainer achieves `gap_closed > 0` at this budget** —
      satisfied (the "separate research thread" branch of the criterion).
- [ ] ~~Tighten the `-25.0` test floor~~ — N/A; only applicable on a *publishable* (`> 0`) result.

## Next steps

1. ~~**PSRO perf + observability pass** — #198~~ **DONE** — parallel payoff eval (#207), live logging (#202), mid-run checkpoints (#204). Calibration showed parallelism alone is insufficient (see the 2026-06-21 update above).
2. **PSRO payoff-tensor caching** — [#212](https://github.com/rjwalters/thrust/issues/212): reuse unchanged `population⁴` entries across iterations (`O(pop⁴)` → `O(pop³)` new work/iter). This is the remaining blocker for a feasible PSRO run.
3. **NFSP learning investigation** — [#199](https://github.com/rjwalters/thrust/issues/199): raise average-policy training steps, diagnose the BR side, revisit reward scaling. **Root-caused + code-level fixes landed** — see [`2026-06-nfsp-avg-policy-undertraining.md`](./2026-06-nfsp-avg-policy-undertraining.md): `NfspConfig` gained an adaptive `avg_policy_min_reservoir_coverage` AP-step floor and a `br_reward_scale` knob (the unscaled `[−700, 0]` band drove the BR critic's `value_loss` to ~9.8M; ×0.01 fixes it). The AP demonstrably fits a factored `[10,2,2]` target to ~0.003 (well below `ln(40)`) when given enough gradient steps. The **full cluster re-run** at the ~9,800-entry reservoir scale remains gated on #134.
4. **PSRO convergence investigation** — [#215](https://github.com/rjwalters/thrust/issues/215): root-cause the exploitability divergence. **Root-caused + code-level fixes landed** — see [`2026-06-psro-alpha-rank-payoff-magnitude.md`](./2026-06-psro-alpha-rank-payoff-magnitude.md): the α-rank Moran fixation probability saturates on the `[−700, 0]` band (`AlphaRankMetaSolver::with_payoff_span_normalization` restores magnitude invariance), and `PsroConfig` gained `br_reward_scale` mirroring #199. The α-rank-solve **perf** work (blocker #2) stays gated on a demonstrated convergence. The **full cluster re-run** remains gated on #134.
5. Once #212 + #199 + #215 land, re-run both at a feasible, observable budget and recompute **cell-specific** `gap_closed_cell`.

---

# Re-run #2 — after the #239 best-response fix (2026-06-24)

**Date:** 2026-06-24
**HEAD:** `669692b` (current `main`, includes #239 BR fix, #241 EV-logging + `train_br_probe`, #235 batched-sampling, #236/#238 rayon parallelism)
**Hardware:** alcubierro cluster — alc-2 / alc-6 / alc-8 (i9-14900K, 32 threads), Burn NdArray (CPU)
**Status:** **Inconclusive-to-weak.** The [#239](https://github.com/rjwalters/thrust/issues/239)
best-response fix *does* move the BR off the uniform floor at cluster scale, but **weakly,
slowly, and cell-dependently** — and the full validation is now **performance-gated** by the
same fix's training cost. No full 48-iteration run was completed; no canonical artifact promoted.

## Why a re-run

The original (2026-06-23) re-run with the #199/#215 knobs reproduced non-convergence and was
root-caused to **[#239](https://github.com/rjwalters/thrust/issues/239)**: the PPO best-response
does not learn (critic `value_loss` flat, policy entropy pinned at the `ln(40)` uniform floor).
The #239 fix (apply grad-clip, `vf_coef 0.5`, **all-minibatch iteration**, `BR_TRAIN_STEPS=8`,
`BR_REWARD_SCALE=0.001`) merged in PR #248 with a **local** result of critic EV `0.00 → 0.33`
and entropy `1.22 → 0.99` on beta05. This re-run tests whether that reproduces at cluster scale.

## Method

A short smoke, not the full job. Per cell (beta01/05/09, one per node):

- **BR-learning probe** — the #241 `train_br_probe` harness (freeze N−1 opponents, train one BR),
  12 iterations × 2048 rollout, `BR_REWARD_SCALE=0.001 VF_COEF=0.5 BR_TRAIN_STEPS=8`,
  `RAYON_NUM_THREADS=16`. This is the purpose-built fast A/B tool for the pivotal
  "does the BR fit?" question, decoupled from the slow NFSP/PSRO outer loop.
- **PSRO** — `train_psro` at 6×2048 with the #215 knobs (`ALPHA_RANK_NORMALIZE_SPAN=1`,
  `MAX_PAYOFF_EVALS_PER_ITER=64`) + the #239 BR knobs.
- **NFSP outer loop** — attempted at 6×2048 but **abandoned**: see "Performance wall" below.

## Result 1 — the BR fix reproduces, but weakly (critic EV)

Full 12-iteration `train_br_probe` trajectories (EV = critic explained variance; entropy floor ≈ `ln(40)/dim` per-dim ≈ 1.23 uniform):

| Cell | EV iter 1–8 | EV iter 9–12 | Entropy 1 → 12 (min) | Verdict |
|------|-------------|--------------|----------------------|---------|
| **beta05** | ~0.00 | 0.011, **0.129, 0.126, 0.169** | 1.22 → 1.05 (**0.99 @ iter 8**) | reproduces #239 (EV up, entropy crosses 1.0) |
| beta09 | ~0.00 | 0.035, 0.042, **0.132, 0.080** | 1.19 → 1.12 (1.10) | partial (EV up; entropy stays ≥ 1.0) |
| beta01 | ~0.00 | 0.034, 0.072, **0.128** | 1.19 → 1.17 (1.13) | weakest (EV only starts climbing at iter 10) |

**Reading:**

- The fix **is real**: EV is genuinely > 0 by the end on all three cells, where the prior
  run had it pinned at exactly 0. The mechanism #239 identified is correct.
- But it is **much weaker than the local PR claimed**: EV reaches only **~0.1–0.17**, not 0.33,
  and only **beta05** drops entropy below 1.0. The critic stays at EV 0 for the first **~8
  iterations** before moving — the fit is slow to start.
- **`mean_ep_return` does not improve** on any cell (flat ~−25k to −30k throughout). The policy
  drifts off uniform but that has not yet translated into a stronger best response. This is the
  crux: a marginally-fitting critic is not yet producing a useful BR.

## Result 2 — PSRO exploitability (incomplete)

Only **beta09** produced PSRO iterations before the co-scheduled probe contended for cores:

| Cell | iters done | exploitability | note |
|------|-----------|----------------|------|
| beta09 | 2 / 6 | 11320 → **10498** (decreasing) | the desired direction, but only 2 points |
| beta01 | 0 / 6 | — | **did not complete iteration 1 in ~67 min** of full-core time |
| beta05 | 0 / 6 | — | same — iteration 1 incomplete at kill |

beta09's two-point downward trend is encouraging versus the prior run's oscillation, but two
points are not a convergence claim, and the other two cells produced no data at all.

## Performance wall (the load-bearing finding)

The #239 fix that makes the BR learn — `BR_TRAIN_STEPS=8` **× all-minibatch iteration per epoch**
— multiplies BR-training cost per outer iteration, and that work runs over the bucket-brigade
rollout that **[#235](https://github.com/rjwalters/thrust/issues/235) proved cannot be batched**
(distinct per-agent policy modules in a shared joint episode). Measured on current `main` at the
**reduced** 2048 rollout:

- **`train_br_probe`** (single BR, frozen opponents): **~1.5 min/iter** — fast, this is why the probe was usable.
- **NFSP outer loop**: **> 1 h/iter** — *no* NFSP iteration completed in ~1 h on any node. Abandoned in favor of the probe.
- **PSRO outer loop**: **~25–65+ min/iter**, cell-dependent (beta09 ~23–33 min with full cores; beta01/05 > 67 min for iteration 1).

The earlier "~150 s/iter at 48×2048" figure was **pre-#239** (the old, non-learning BR). The fix
trades non-convergence for a throughput wall: a full 48-iteration × 6-run validation at this cost
is **days-to-weeks** of cluster time for what the probe suggests is a **marginal** signal. Running
it blind is not justified by the smoke.

## Conclusion

The #239 fix is **directionally correct but under-delivers at scale**: the BR critic begins to fit
(EV 0 → ~0.1–0.17) where it previously did not, but the effect is weak, slow (~8 iters to onset),
fails to lower entropy below the uniform floor on 2 of 3 cells, and does not yet improve
best-response return. Combined with the new per-iteration cost, this does **not** clear the #134
bar (`gap_closed_cell > 0` on ≥ 1 cell), and the full run is not currently worth its cluster cost.
The result is logged as **inconclusive/weak**, superseding neither the original negative finding
nor establishing a positive one.

## Follow-ups filed

- **BR-training throughput** — [#251](https://github.com/rjwalters/thrust/issues/251): the
  all-minibatch × `BR_TRAIN_STEPS=8` cost over the un-batchable rollout makes the NFSP/PSRO outer
  loops infeasible at validation budget. The cheapest tractability lever is reducing per-iter BR
  cost (e.g. subsample minibatches, or parallel-env rollouts per #235's deferred EnvPool note).
- **#239 efficacy gap at scale** — [#252](https://github.com/rjwalters/thrust/issues/252): local
  probe gave EV → 0.33 / entropy → 0.99; cluster gives EV → ~0.15 / entropy ≥ 1.0 on 2/3 cells,
  no return improvement. The fix needs strengthening (more BR iters? critic LR? the gated
  separate-critic-optimizer?) before another full attempt.

## Reproduction

```bash
# BR-learning probe (the fast A/B used here), per cell:
CELL=beta05 ITERATIONS=12 ROLLOUT_STEPS=2048 \
  BR_REWARD_SCALE=0.001 VF_COEF=0.5 BR_TRAIN_STEPS=8 \
  cargo run --release --example train_br_probe --features "training,env-bucket-brigade"
```

Logs for this re-run: `run134-out/probe_{beta01,beta05,beta09}.log` and
`run134-out/smoke_psro_{beta01,beta05,beta09}.log` on alc-2/6/8 (and pulled locally during the session).
