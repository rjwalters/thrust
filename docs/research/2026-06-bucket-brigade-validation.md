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

---

# Throughput profile + minibatch-cap lever (issue #251, 2026-06-24)

**Date:** 2026-06-24
**Branch:** `feature/issue-251`
**Backend:** NdArray<f32> CPU, local workstation (Apple Silicon). All timings below are
**local, single-machine** numbers — directional, not cluster figures (no cluster access this session).

## AC#1 — where the outer-loop wall-clock actually goes

Issue #251's premise was that the dominant new cost is the #239 **all-minibatch ×
`BR_TRAIN_STEPS=8`** BR-update product. Coarse `Instant` profiling on `main` gives a **more nuanced**
picture: the *single-BR probe* is rollout-bound, but the *multi-agent NFSP/PSRO outer loop* has a
**co-dominant BR-update phase (~42–52%)** that the lever directly attacks.

**Single-BR probe (`train_br_probe`, beta05, 2048 rollout, 8 iters × `BR_TRAIN_STEPS=8`) — only ONE
agent's optimizer steps (frozen N−1):**

| Phase | wall-clock | share |
|-------|-----------|-------|
| BR rollout (`collect_rollout`) | 841.8 s | **96%** |
| BR PPO update (single active agent, 4 epochs × 8 mb) | 39.3 s | **4%** |

The probe's update is tiny because only one agent is trained and the per-update cost there is
dominated by fixed overhead (per-agent GAE/returns/advantage setup), so it is **not representative**
of the real outer loop. The ~105 s/iter rollout is the **batch-1 per-agent rollout** that
[#235](https://github.com/rjwalters/thrust/issues/235) proved cannot be batched.

**NFSP outer loop (`train_nfsp`, beta05) — all 4 agents active in the joint update:**

| Run | br_rollout | br_update | ap_train | total/iter |
|-----|-----------|-----------|----------|-----------|
| 512 rollout, iter 1 | 172.0 s (47%) | 190.0 s (52%) | 2.0 s (1%) | 364.1 s |
| 512 rollout, iter 2 | 189.5 s (36%) | 339.0 s (64%) | 1.8 s (0%) | 530.3 s |
| 2048 rollout, iter 1 | 343.2 s (57%) | 253.3 s (42%) | 2.0 s (0%) | 598.5 s |

(2048 figures use `BR_TRAIN_STEPS=2`; 512 figures use `BR_TRAIN_STEPS=8`. Phase *shares* are the
stable signal — absolute seconds vary with machine load.)

Here the BR **update is co-dominant** (42–64% across the runs measured). Unlike the single-agent
probe, the multi-agent update is **variable-dominated** — each minibatch does 4 per-agent
forward+backward passes, a global-norm grad-clip (two module-visitor passes per agent), and 4
optimizer steps — so reducing the minibatch count per epoch reduces it ~proportionally. The AP
supervised step is negligible (≤2%) at the reservoir occupancies seen in early iterations.

**Conclusion (AC#1):** the outer loop is split between the un-batchable rollout (~57%) and the
**multi-agent BR update (~42%)** at 2048. The BR update is exactly what a minibatch cap reduces; the
rollout share remains the deferred parallel-env (EnvPool, #235) follow-up.

## The change — `max_minibatches_per_epoch` throughput lever (opt-in)

`JointTrainerConfig::max_minibatches_per_epoch: Option<usize>` (default `None`). When `Some(cap)`,
each epoch's globally-shuffled minibatch set is truncated to at most `cap` chunks — a uniformly
random subsample of the rollout (the indices are shuffled before chunking). Behavior-preserving by
default:

- `None` (default) preserves the full #239 all-minibatch coverage **bit-identical**; grad-clip,
  `vf_coef`, `iterate_all_minibatches`, and `BR_TRAIN_STEPS` are all unchanged. Each retained
  minibatch is still a full forward+backward over `minibatch_size` samples.
- Exposed via `BR_MAX_MINIBATCHES_PER_EPOCH` in `train_nfsp`, `train_psro`, and `train_br_probe`.
- More effective at higher rollout (more minibatches per epoch to cap): at 2048 there are 8
  minibatches/epoch, so `cap=2` is a 4× minibatch-count reduction.

## AC#3 — before/after, controlled back-to-back A/B (local, beta05, 2048 rollout)

Measured with `BR_TRAIN_STEPS=2` as a fast timing proxy (the per-update cap effect is independent of
the step count); cap OFF then cap=2 run back-to-back to minimise machine drift:

| | cap OFF (8 mb/epoch) | cap=2 (2 mb/epoch) | change |
|---|---|---|---|
| br_rollout | 343.2 s (57%) | 277.3 s (70%) | unchanged by cap (Δ = machine drift) |
| **br_update** | **253.3 s (42%)** | **114.3 s (29%)** | **2.2× faster** |
| total / iter | 598.5 s | 393.9 s | **1.52× faster (−34%)** |

The minibatch cap cuts the multi-agent BR update **~2.2×** (the residual is the per-update fixed
GAE/setup overhead the cap cannot touch), shrinking the whole outer iteration **~1.5×** even though
the rollout is the larger single phase. A higher `BR_TRAIN_STEPS` (the #239 default of 8) multiplies
the update savings linearly. The cap is the **proportional, in-scope lever for the update phase**;
the remaining rollout share is the parallel-env follow-up.

## AC#2 — the #239 BR-learning behaviour is preserved with the lever on

`train_br_probe` (beta05, 2048, 8 iters, `BR_REWARD_SCALE=0.001 VF_COEF=0.5 BR_TRAIN_STEPS=8`),
critic explained-variance `ev` and policy entropy `ent` per iter:

| iter | cap OFF `ev` / `ent` | cap=2 `ev` / `ent` |
|------|----------------------|--------------------|
| 1 | 0.000 / 1.220 | 0.000 / 1.224 |
| 4 | 0.000 / 1.136 | 0.000 / 1.211 |
| 6 | 0.000 / 1.050 | 0.000 / 1.202 |
| 7 | 0.000 / 1.009 | 0.000 / 1.114 |
| 8 | 0.011 / **0.993** | 0.000 / **1.122** |

With the cap on, the BR **still moves off the uniform-entropy floor** (`ent` 1.224 → 1.122) — the
#239 learning direction is preserved, just at a slower per-iteration rate (fewer gradient steps per
epoch). `ev` has not yet risen at 8 iters in **either** arm (per #239 the EV onset is ~iter 16); the
cap does not collapse the BR back to the floor. This is the documented throughput/fit trade-off: the
cap is opt-in and the operator chooses `cap` per their iteration budget (default = full coverage).

## Reproduction (issue #251)

```bash
# Profile + baseline (lever OFF, full #239 coverage):
CELL=beta05 ITERATIONS=8 ROLLOUT_STEPS=2048 \
  BR_REWARD_SCALE=0.001 VF_COEF=0.5 BR_TRAIN_STEPS=8 \
  cargo run --release --example train_br_probe --features "training,env-bucket-brigade"

# Lever ON (cap minibatch steps/epoch to 2):
CELL=beta05 ITERATIONS=8 ROLLOUT_STEPS=2048 \
  BR_REWARD_SCALE=0.001 VF_COEF=0.5 BR_TRAIN_STEPS=8 BR_MAX_MINIBATCHES_PER_EPOCH=2 \
  cargo run --release --example train_br_probe --features "training,env-bucket-brigade"

# NFSP outer-loop phase breakdown (per-iter br_rollout vs br_update vs ap_train):
TOTAL_ITERATIONS=2 ROLLOUT_STEPS=2048 CELL=beta05 \
  BR_REWARD_SCALE=0.001 VF_COEF=0.5 BR_TRAIN_STEPS=8 \
  cargo run --release --example train_nfsp --features "training,env-bucket-brigade"

# Controlled outer-loop throughput A/B (cap OFF vs cap=2), fast BR_TRAIN_STEPS=2 timing proxy:
for CAP in "" "2"; do TOTAL_ITERATIONS=1 ROLLOUT_STEPS=2048 CELL=beta05 \
  BR_REWARD_SCALE=0.001 VF_COEF=0.5 BR_TRAIN_STEPS=2 BR_MAX_MINIBATCHES_PER_EPOCH=$CAP \
  cargo run --release --example train_nfsp --features "training,env-bucket-brigade"; done
```

---

# Lever sweep for the #239 efficacy gap (issue #252, 2026-06-24)

**Date:** 2026-06-24
**Branch:** `feature/issue-252`
**Backend:** NdArray<f32> CPU, local workstation (28 cores). **Every number below is local,
small/short-budget, and directional** — there was no cluster access this session. Do **not** read
these as cluster figures.

Issue #252 asked which lever, if any, closes the gap between the #239 best-response fix's weak
cluster behaviour (critic explained-variance `ev` plateauing at ~0.15, entropy ≥ 1.0 on 2/3 cells,
flat `mean_ep_return`) and the ~0.48 achievable EV ceiling established by the #242 fittability
diagnostic (`tests/test_bucket_brigade_critic_fittability.rs`: best held-out `local_obs → return`
EV ≈ 0.48). One lever at a time was swept off the current default baseline
(`BR_REWARD_SCALE=0.001 VF_COEF=0.5 BR_TRAIN_STEPS=8`, 2048 rollout), beta05.

## Headline result — the EV "plateau" was premature stopping, not a cap

The single most important question was **"does `ev` keep climbing past iter 12, or is it capped at
~0.15?"** Answer: **it keeps climbing, all the way to (and past) the ~0.48 ceiling.** The cluster's
~0.15 reading was an artifact of stopping the probe at iter 12 — the EV onset is slow (~iter 8) and
the rise continues for another ~15 iterations.

`train_br_probe`, beta05, 2048 rollout, baseline knobs, **40 iters** (selected rows; the run was
captured through iter 39):

| iter | `ev` | `ent` | `mean_ep_return` | note |
|------|------|-------|------------------|------|
| 1 | 0.000 | 1.220 | −25 549 | uniform start |
| 7 | 0.000 | 1.009 | −27 107 | critic still flat |
| 8 | 0.011 | **0.993** | −26 643 | EV onset; the *only* sub-1.0 entropy reading |
| 9 | 0.132 | 1.004 | −26 760 | EV rising |
| 12 | 0.172 | 1.056 | −30 293 | **where the cluster stopped (~0.15–0.17)** |
| 16 | 0.329 | 1.120 | −24 334 | matches the original local "0.33" claim |
| 20 | 0.341 | 1.111 | −26 832 | still climbing |
| 25 | **0.478** | 1.156 | −28 214 | **reaches the ~0.48 #242 ceiling** |
| 29 | 0.489 | 1.178 | −24 256 | at ceiling |
| 33 | 0.580 | 1.125 | −26 629 | on-policy EV overshoots the offline 0.48 |
| 38 | 0.564 | 1.115 | −27 585 | plateaued ~0.48–0.58 |
| 39 | 0.552 | 1.134 | −27 927 | plateaued |

**Reading:**

- **`ev` is NOT capped at ~0.15.** Given enough iterations the per-agent BR critic fits to the
  full #242 ceiling (`ev` ≈ 0.48 by iter 25, oscillating 0.48–0.58 thereafter; the on-policy PPO
  `ev`, measured on the current batch, runs a touch above the offline held-out 0.48). The headroom
  the issue worried about **is captured — by Lever 1 (more iterations) alone.** The cluster's
  "EV → 0.15 plateau" was reading the trajectory at iter 12, before the rise finished.
- **But the objective is still not met.** Two things the issue cares about do **not** improve, even
  with a fully-fit critic:
  - **Entropy never settles below the uniform floor.** There is a single transient dip to 0.993 at
    iter 8; as `ev` climbs the policy entropy *recovers* to ~1.10–1.18 and stays there. The
    "entropy < 1.0" signal is fragile and, here, non-monotone.
  - **`mean_ep_return` is flat** (~−24k to −30k, no trend) across all 39 iters. A critic that
    predicts returns at the achievable ceiling does **not** translate into a stronger best response.

This is the crux finding: **critic *fit* is not the bottleneck** (it is solvable with more
iterations), so strengthening the critic further cannot be the unblock for #134. The bottleneck is
that a well-fit per-agent critic does not yield a higher-return policy — which points at the
policy-improvement / advantage side and **re-opens the centralized-critic question #242 set aside**
(a per-agent critic that fits its *own* return target is still not enough to drive the BR).

## Lever 2 — separate critic optimizer (`CRITIC_LR`, #239 fix #4): harmful

`train_br_probe`, beta05, 2048, baseline knobs + `CRITIC_LR=1e-3` (splits actor/critic into two
backward passes, steps the value head at 1e-3 vs the actor's 3e-4), 20 iters:

| iter | `ev` | `ent` | `mean_ep_return` |
|------|------|-------|------------------|
| 1 | 0.000 | 1.209 | −26 064 |
| 8 | 0.000 | 1.190 | −30 558 |
| 14 | 0.000 | 1.079 | −24 782 |
| 20 | 0.000 | 0.922 | −29 153 |

`ev` stays pinned at ~0.000 for the **entire** run — the separate critic optimizer **breaks** the
EV-rising behaviour the shared optimizer produces (value loss stays ~11–16, never collapses). This
confirms #239's "unhelpful in isolation" finding and sharpens it: at the tested LR the split is
**actively harmful** to critic fit. Note the entropy still drifts to 0.92 *without any critic fit* —
reinforcing that entropy decline is **not** evidence the critic is working (see also the 512-rollout
control below).

## Control — entropy collapse without critic fit (512 rollout)

A cheap control that screened whether a smaller rollout could stand in for 2048 (it cannot, but the
result is independently informative). `train_br_probe`, beta05, **512** rollout, baseline knobs, 40
iters:

| iter | `ev` | `ent` | `mean_ep_return` |
|------|------|-------|------------------|
| 1 | −0.001 | 1.193 | −25 520 |
| 8 | 0.000 | 0.944 | −26 465 |
| 20 | 0.000 | 0.617 | −22 900 |
| 40 | −0.001 | 0.306 | −22 680 |

At 512 the critic **never fits** (`ev` ≈ 0 throughout — the per-update batch is too small for the
value head), yet the policy entropy **collapses to 0.31** and `mean_ep_return` stays flat. Two
consequences: (1) the critic is **data-hungry** — it needs the larger 2048 batch to fit, so the
#251 minibatch-cap lever (which *reduces* per-epoch data) would hurt, not help, EV; and (2) the
"entropy < 1.0" acceptance signal is **unreliable in isolation** — here entropy crosses well below
1.0 with *zero* critic fit and *zero* return improvement. The real objective is `mean_ep_return`,
and it does not move in any 512 or 2048 configuration tested.

# Stage-1 cluster probe sweep — the diagnostic-first ladder (2026-06-26/28)

**Issue:** [#134](https://github.com/rjwalters/thrust/issues/134) · **PR:** #256 ·
**Hardware:** alcubierre cluster, 6 worker nodes (alc-2/4/5/6/7/8), NdArray CPU backend.

## Why this run

Every prior probe was *local micro-budget* (512–2048 rollout). The open question
the negative result hinges on is whether the BR inner loop can raise team return
**at all** at a larger budget, or across an untried lever. Rather than spend the
cluster window on the full 6-run PSRO/NFSP protocol (which #239/#251/#252 predicted
would reproduce non-convergence slowly and expensively), we ran a **diagnostic-first
ladder**: a parallel `train_br_probe` sweep at **8192 rollout** over 3 cells × 6 knob
sets as a *gate* — only a knob set that raises `mean_ep_return` would have advanced to
seeded full runs. (`scripts/weekend_alc_dispatch.sh`; gate + collection in
`scripts/alc_finalize_stage1.sh`.)

## Knob grid

`base` (8192×8, the #239 defaults), `vf1` (`VF_COEF=1.0`), `br16` (`BR_TRAIN_STEPS=16`),
`rs01` (`BR_REWARD_SCALE=0.01`), `bigroll` (`ROLLOUT_STEPS=16384`), `combo`
(16384 × 16 trainsteps × `VF_COEF=1.0`). One probe per (cell, knobset); `BR_MAX_MINIBATCHES_PER_EPOCH`
deliberately left unset (the 512 control above shows capping data hurts the data-hungry critic).

## Result — gate FAIL: 0 of 15 evaluable runs raise team return

Gate metric = mean of the last-5 vs first-5 iters' `mean_ep_return`; "win" = >5% less
negative. **No knob set won on any cell.** Every run sits in the −27k to −28.5k band
(±2% noise; several drift *worse* late-than-early). Representative β=0.5 results:

| knobset (β=0.5) | `ev` early→late | `mean_ep_return` early5→late5 | verdict |
|------|------|------|------|
| `base` 8192×8 | 0.00 → **0.565** | −27 884 → −27 285 | flat |
| `vf1` VF_COEF=1.0 | 0.00 → **0.583** | −27 455 → −28 454 | flat |
| `br16` 16 trainsteps | 0.00 → **0.575** | −27 727 → −27 302 | flat |
| `bigroll` 16384 rollout | 0.00 → 0.328¹ | −28 046 → −27 983 | flat |
| `rs01` BR_REWARD_SCALE=0.01 | 0.00 → 0.053² | −28 076 → −27 303 | flat |

¹ `bigroll`/`combo` were stopped early (16384 rollout ran ~3.8 h/iter and ~6.9 h/iter
respectively — a 5–11 day tail for confirmatory data on an already-flat trend); they
were flat through the iters completed. ² `BR_REWARD_SCALE=0.01` (vs the 0.001 default)
visibly *degrades* critic fit (EV 0.05 vs 0.57) — the larger band hurts the value head.

## The load-bearing finding — the critic fits, and it does not matter

This sweep **settles** the question the earlier `≤2048` probes left ambiguous. At 8192
rollout the critic reaches **EV ≈ 0.57** on `base`/`vf1`/`br16` — well past the
~0.48 "fittability ceiling" reported at 2048 (#242/#252), i.e. a genuinely *well-fit*
value head — and `mean_ep_return` **still does not move**. Combined with the 512 control
(entropy collapses to 0.31 with zero critic fit and zero return gain), the BR inner loop
now shows return is flat **across the entire EV spectrum**: EV≈0 (512), EV≈0.5 (8192),
larger rollout, more trainsteps, higher value weight. **Critic fit is conclusively not
the bottleneck**, and no tested PPO-side lever raises team return on these cells.

## Implication for #134 / #230

The gate failed, so the seeded full PSRO/NFSP runs (Stage 2) were **not** launched — a
fast non-convergent BR has no research value, and PSRO/NFSP both inherit this BR. This is
the most thoroughly-evidenced statement of the #134 negative result to date: across three
re-runs and now a 5-lever 8192-rollout sweep, **neither trainer beats PPO on the
no-convergence cells, because the underlying PPO best-response does not improve team
return regardless of critic quality.** The bottleneck is upstream of NFSP/PSRO — in the
BR objective/credit-assignment on this game, not in the meta-solver or the average policy.
Accordingly **#230** (α-rank solve cost) stays correctly gated: there is still no
demonstrated PSRO convergence to make that perf work worthwhile.

---

# Improvability gate — does a better-than-uniform BR even exist? (issue #259, 2026-06-28)

## Why this run

The Stage-1 sweep above settled that the BR's **critic** fits (EV ≈ 0.57) yet team return
stays flat — but it did not answer the prior question: **is there anything for the policy
to find?** Every closed predecessor (#239, #252, #241) patched the PPO/critic side and
under-delivered. Issue #259 imposes an **improvability gate**: before touching the PPO
update, establish — via a **non-PPO** method — whether a better-than-uniform best-response
*exists at all* against the same frozen-uniform opponents `train_br_probe` trains against.

## Method — a scripted (NN-free, PPO-free) oracle

New harness: `examples/games/bucket_brigade/br_oracle.rs` + library
`src/multi_agent/bucket_brigade_oracle.rs`. It freezes `N−1 = 3` **uniform-random**
opponents (the clean idealization of `train_br_probe`'s freshly-initialized,
≈uniform frozen nets) and scores a battery of **scripted** policies for the single BR
agent (agent 0), reporting best-achievable per-step **team** return and per-step / per-episode
**BR-agent** return vs the all-uniform baseline. Candidates: `uniform` (baseline),
`always_rest`, the `specialist` baseline (the literal `gap_closed_cell == 1.0` endpoint),
deterministic firefighters (owned-only and any-house, `work=1.0`), and a 64-sample random
search over a `FirefighterParams { scope_owned_only, work_prob }` family that *contains*
the specialist. All candidates share one per-episode seed stream (variance reduction).
400 eval episodes/cell, all three no-convergence cells.

## Result — no improvable team-return gap (gate: branch 1, flat)

| BR policy (agent 0; other 3 uniform) | per-step **team** | per-step **BR-agent** | per-ep **BR-agent** |
|------|------|------|------|
| `uniform` (baseline) | −673.950 | −205.448 | −27 634 |
| `always_rest` | −673.886 | −201.979 | −27 372 |
| `specialist` | −678.519 | −196.504 | −27 831 |
| `firefighter[owned, work=1.0]` | −675.822 | −194.403 | −27 144 |
| `firefighter[any, work=1.0]` | −674.063 | −204.604 | −29 732 |
| `search_best firefighter[owned, work=0.853]` | −676.583 | −197.760 | −27 587 |

**Ceiling (best team return) = `always_rest` at −673.886 vs baseline −673.950 → team gap
= +0.064/step = +0.01% of |baseline|.** The strongest *known* policy (specialist) and an
aggressive any-house firefighter are both **worse** on team return than doing nothing,
because fighting fires costs `c = 0.5`/night while three uniform-random teammates let the
village ruin regardless. Even the BR agent's **own** return — the quantity `train_br_probe`
actually optimizes — has a ceiling only **+0.95%** above uniform (best `firefighter[owned]`
−194.4 vs −205.4/step), and that small edge comes from *resting more* (avoiding wasted
work cost), not from coordinating a save. All three cells (β = 0.1 / 0.5 / 0.9) produced
**identical** aggregate statistics under the shared seed stream — matching this repo's own
committed per-cell baselines (`MINSPEC_RANDOM_BETA01/05/09` and
`MINSPEC_SPECIALIST_BETA01/05/09` are byte-identical) and the documented
"wasteland-collapse" dynamics: once a handful of fires ignite, Bernoulli burn-out ruins
the ring within a few nights and the per-step ruined-house penalty swamps any single
agent's fire-fighting.

## Implication — the #134 direction is exhausted at the BR level

This is **outcome 1** of the #259 gate (oracle ≈ uniform): **no PPO/policy fix is
warranted.** The flat `mean_ep_return` the Stage-1 sweep observed is not a bug in the PPO
update — it is a property of the *game* in the single-BR-vs-frozen-uniform regime. With
3 of 4 agents frozen uniform, the BR agent's marginal contribution to **team** return is
~0 (+0.01% ceiling) and to its **own** return is ~+0.95% (achieved by resting, not
saving). There is nothing for the policy gradient to climb toward, so a well-fit critic
correctly reports near-zero advantage and the policy correctly stays near uniform.

The #259 **AC#2** (instrument/patch the PPO update) is therefore **correctly skipped** —
its precondition (a real improvable gap) is not met. For **#134 / #230**: this is the BR-level
confirmation that the no-convergence cells are not unblockable by tuning the best-response.
The single-agent best-response problem against uniform opponents is itself near-flat;
PSRO/NFSP inherit a BR with no team-return headroom, so the negative result stands and
**#230** remains correctly gated. Any future attempt to revive the #134 direction must
change the **game/credit-assignment regime** (e.g. a per-agent shaped reward that gives a
single firefighter a non-trivial marginal return, or co-adapting opponents rather than
frozen-uniform), not the PPO optimizer.

Reproduce:

```bash
for c in beta01 beta05 beta09; do
  CELL=$c EVAL_EPISODES=400 \
    cargo run --release --example br_oracle --features "training,env-bucket-brigade"
done
```
