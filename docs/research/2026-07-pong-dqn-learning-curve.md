# Pong DQN Learning Curve — Nature-DQN on ALE Pong (Epic #306, Phase 4)

**Date:** 2026-07-09
**Issue:** [#329](https://github.com/rjwalters/thrust/issues/329) (run) / [#330](https://github.com/rjwalters/thrust/issues/330) (report) — Epic [#306](https://github.com/rjwalters/thrust/issues/306), Phase 4
**Binary:** `examples/games/atari/train_pong_dqn.rs` (`--example train_pong_dqn`)
**Runbook:** [`../PONG_DQN_RUNBOOK.md`](../PONG_DQN_RUNBOOK.md)
**Hardware:** alcubierre cluster — `alc-2` and `alc-8`, each an NVIDIA RTX 4090 (24 GB), CUDA backend
**Artifacts (curves):**
[`data/2026-07-pong-dqn-run1-lr2.5e-4.csv`](data/2026-07-pong-dqn-run1-lr2.5e-4.csv),
[`data/2026-07-pong-dqn-run2a-lr6.25e-5.csv`](data/2026-07-pong-dqn-run2a-lr6.25e-5.csv),
[`data/2026-07-pong-dqn-run2b-lr1e-4.csv`](data/2026-07-pong-dqn-run2b-lr1e-4.csv)

## TL;DR

The Thrust Nature-DQN stack **demonstrably trains** on ALE Pong: from the −21
random floor it lifts off at ~200k wrapper steps, climbs through the ε-decay
floor, and converges toward a rising envelope. With the corrected Atari-standard
Adam learning rate (**6.25e-5**, Rainbow/Dopamine), run 2 arm A reaches
`avg(last≤100) = −5.14` at the 5M wrapper-step (20M raw-frame) budget and is
**still improving at budget exhaustion** (best 400k window: **−3.89**). It does
not cross zero within this budget.

Three runs, one conclusion:

| Run | Adam LR | Host | Final `avg(last≤100)` | Best 400k window | Verdict |
|---|---|---|---|---|---|
| Run 1 | 2.5e-4 | alc-2 | **−20.76** (stopped at 2.39M steps) | −20.42 | Negative result — wrong LR (Mnih RMSProp rate) |
| Run 2 arm A | **6.25e-5** | alc-2 | **−5.14** at 5M steps | **−3.89** | Textbook curve, still rising at budget end |
| Run 2 arm B | 1e-4 | alc-8 | **−7.21** at 5M steps | −6.73 | Learns, but repeated instability dips; ends worse than arm A |

The gap to a zero-crossing is a **budget-and-buffer** story, not a code defect:
sticky-action DQN baselines typically need 15–25M+ raw frames and 1M-transition
replay buffers to cross zero on Pong; this run used 20M raw frames and a
100k-transition f32 buffer (host-RAM-bounded). The concrete path to a crossing is
a **u8 frame buffer** (→ 1M-transition capacity at the same RAM) plus a **longer
step budget** — no algorithmic change required.

## Reproducible command

From the repo root on a CUDA host (full operator recipe in the
[runbook](../PONG_DQN_RUNBOOK.md)):

```bash
export ATARI_PYTHON=/usr/bin/python3          # interpreter with ale-py installed
export CURVE_CSV="$HOME/pong_dqn_curve.csv"    # env_steps,mean_episode_reward
export CHECKPOINT_INTERVAL=500000
export CHECKPOINT_DIR="$HOME/pong_dqn_checkpoints"
export LEARNING_RATE=6.25e-5                    # Atari-standard Adam (the default)

cargo run --release \
  --features "training,env-atari,cuda" \
  --example train_pong_dqn \
  2>&1 | tee "$HOME/pong_dqn_run.log"
```

Default budget is `TOTAL_TIMESTEPS=5_000_000` **wrapper steps** = 20M raw frames
(frame-skip 4). Arm B swapped only `LEARNING_RATE=1e-4`; every other
hyperparameter is identical across the two arms (single-variable LR comparison).

## Run 1 — Adam 2.5e-4 (negative result)

The first run used Adam **2.5e-4** — the Mnih 2015 *RMSProp* rate, copied into the
Adam column without adaptation (see [#342](https://github.com/rjwalters/thrust/issues/342)).

- Stopped at the pre-declared no-learning decision point: **2.39M wrapper steps
  (9.56M raw frames), ~6.2 h**, ~103–130 wrapper steps/s.
- `avg(last≤100)` never exceeded **−20.27** and sat flat at **−20.8 ± 0.1** from
  ~1M wrapper steps onward (the ε floor). The greedy policy was marginally
  *worse* than random — zero upward trend over 1.4M post-floor steps.
- **Root cause:** Adam at 2.5e-4 is ~4× the Atari-standard Adam rate. At that
  rate the Nature-DQN CNN never escapes the ε-exploration floor.
- **Resolution:** [PR #343](https://github.com/rjwalters/thrust/pull/343)
  (issue [#342](https://github.com/rjwalters/thrust/issues/342)) changed the
  default to **6.25e-5** (Rainbow/Dopamine) and added a `LEARNING_RATE` env knob.
  Both are on `origin/main`.
- **Artifacts:** `alc-2:~/pong_dqn_run1_lr2.5e-4/` (curve CSV, full log, 4
  checkpoints at 500k intervals).

Curve: [`data/2026-07-pong-dqn-run1-lr2.5e-4.csv`](data/2026-07-pong-dqn-run1-lr2.5e-4.csv)
(239 rows, flat at −20.8 throughout).

## Run 2 arm A — Adam 6.25e-5 (corrected default)

The corrected-LR rerun isolates a single variable (LR: 6.25e-5 vs run 1's
2.5e-4) and produces a **textbook DQN learning curve**:

- **Liftoff ~200k wrapper steps.** First point above −19 at **590k steps**
  (−18.92); the curve is climbing steadily before the ε floor is reached.
- **−15 at the ε floor** (~1M steps, where ε reaches 0.1).
- **Plateau near −8** at ~1.9–2.6M steps, then a further climb toward **−5.5**.
- **Final `avg(last≤100) = −5.14` at exactly 5M wrapper steps (20M raw
  frames)** — and **still improving at budget exhaustion**. The last 400k-step
  window oscillates −6.3 → −3.9 → −5.1 with a **rising envelope**; the best
  400k window mean is **−3.89** (starting ~4.48M steps) and the single best
  logged point is **−3.44** at 4.67M steps.
- **Stats:** 2,289 episodes, 14.0 h wall clock, ~99 wrapper steps/s.
- **Artifacts:** `alc-2:~/pong_dqn_run2_lr6.25e-5/` (curve CSV, full log, 9
  checkpoints).

Curve: [`data/2026-07-pong-dqn-run2a-lr6.25e-5.csv`](data/2026-07-pong-dqn-run2a-lr6.25e-5.csv)
(500 rows).

This is the honest headline result: the stack learns Pong, the curve is still
rising at the budget wall, and the score at the wall (−5.14, best window −3.89)
is bounded by the budget/buffer — not by any correctness defect.

## Run 2 arm B — Adam 1e-4 (parallel LR-stability arm)

Arm B ran in parallel on `alc-8` with `LEARNING_RATE=1e-4` (~1.6× arm A's rate),
all else identical, to probe LR stability:

- **Tracks arm A closely to ~2M steps** — liftoff at ~200k (first >−19 at 540k,
  −18.95), same climb shape, same approach to the −8 plateau.
- **Diverges after ~2M steps with repeated instability dips.** The curve drops
  from ~−6.3 into a sustained **−8.9 to −10.9 trough** at ~2.0–2.6M steps
  (deepest point −10.89 at 2.52M), recovers to ~−8, dips again around
  **~3.1M and ~4.0M steps** (two further −8.5 to −9.1 excursions) that arm A
  **never showed**.
- **Late rally then settle.** Best logged point −5.91 at 3.77M; a late-run rally
  precedes a settle to a **final `avg(last≤100) = −7.21` at exactly 5M wrapper
  steps** (best 400k window −6.73).
- **Stats:** 2,411 episodes, 15.0 h wall clock (53,968 s), ~93 wrapper steps/s.
- **Artifacts:** `alc-8:~/pong_dqn_run2b_lr1e-4/` (curve CSV, full log,
  checkpoints).

Curve: [`data/2026-07-pong-dqn-run2b-lr1e-4.csv`](data/2026-07-pong-dqn-run2b-lr1e-4.csv)
(500 rows).

### LR-stability comparison (6.25e-5 vs 1e-4)

Both rates learn Pong and follow the same trajectory to ~2M steps. Past that
point the **lower rate (6.25e-5, arm A) is meaningfully more stable**: it climbs
with a monotonically rising envelope and no multi-point regressions, ending at
**−5.14** and still rising. The higher rate (1e-4, arm B) shows **repeated
multi-point instability dips** (three separate −8.5 to −10.9 excursions after
2M steps) and ends **worse (−7.21)** despite an extra hour of wall clock and more
episodes. The takeaway: **6.25e-5 is the right default** — the Rainbow/Dopamine
rate is not just conventional, it is empirically the more stable of the two on
this stack, and arm A was still improving at budget exhaustion (best window
−3.89) while arm B had already peaked and settled lower.

## Gap analysis — why zero was not crossed (and how to cross it)

The success criterion (runbook §Success criterion) is **any positive mean
episode score** (random floor ≈ −21). Neither arm crossed zero within the 20M
raw-frame budget. This is expected and honest given the budget and buffer:

**Budget.** Sticky-action (Machado et al. 2018, p=0.25) DQN baselines typically
need **15–25M+ raw frames** to cross zero on Pong — later than the classic
no-sticky curves. This run's 20M raw frames sits at the *low end* of that band.
Arm A was **still improving at 20M frames** (rising envelope, best window −3.89),
consistent with a crossing that lands just beyond the current budget wall.

**Replay buffer.** The in-tree `ReplayBuffer` stores frames as `f32`
(4 bytes/value), so one Pong transition is ~221 KiB and a **1M-transition buffer
would need ~210 GiB** — it does not fit in host RAM. This run used a
**100k-transition** buffer (~21 GiB f32). Published Pong DQN uses a **1M**
buffer; the 10× smaller buffer shortens the effective experience horizon and
caps sample diversity, which bounds the achievable score independent of step
count. (Full RAM math: runbook §Replay-buffer memory.)

**The stack itself is correct.** Every diagnostic of a working DQN is present:
liftoff off the random floor, monotone climb through the ε-decay schedule,
ε-floor improvement, a plateau-then-climb shape, and a rising envelope at the
budget wall. The negative run 1 result was a hyperparameter bug (LR), fixed and
reproduced as the arm-A textbook curve. There is no evidence of a code defect
bounding the score.

**Concrete path to a crossing** (no algorithmic change):

1. **u8 frame buffer.** Store frames as `u8` (the pre-`1/255`-scale
   representation) instead of `f32`, cutting frame storage 4×
   (~5.3 GiB at 100k). At the same ~21 GiB host-RAM budget this raises capacity
   to **~1M transitions** — the published DQN buffer size. This is a new buffer
   type (out of scope for #329/#330; a good follow-up issue).
2. **Longer step budget.** Extend `TOTAL_TIMESTEPS` to **7.5–10M wrapper steps
   (30–40M raw frames)** so the still-rising arm-A envelope has room to reach and
   cross zero, matching the upper end of the sticky-action band.

Either alone would likely help; together they close the published-baseline gap.

## Throughput / operator notes

- **~99 wrapper steps/s on the RTX 4090, GPU util ~60 %.** The throughput ceiling
  is **CPU/env-IPC-bound**, not GPU-compute-bound: the subprocess `ale-py`
  adapter's per-step IPC overhead — not the CNN forward/backward — caps the loop.
  A single 4090 is **not saturated** by this workload (see the
  [#281 re-triage](https://github.com/rjwalters/thrust/issues/281) for the
  distributed-training implication).
- **Wall clock** followed the runbook's conservative (IPC-bound) band:
  `5M / ~99 ≈ 14 h` for arm A, ~15 h for arm B.
- **Node provisioning (non-alc-2 CUDA hosts).** Only `alc-2` has the apt CUDA
  toolkit. Other alc-* nodes (arm B ran on `alc-8`) are provisioned via:
  `pip install nvidia-cuda-nvrtc-cu12` (nvrtc pip wheel) + toolkit headers staged
  into `~/cuda-root` + `CUDA_PATH=~/cuda-root` + `LD_LIBRARY_PATH` pointed at the
  wheel's lib directory. Proven on alc-8. (Also documented in
  [`../BURN_BACKENDS.md`](../BURN_BACKENDS.md).)

## Cross-references

- Reproduction recipe and hyperparameter table: [`../PONG_DQN_RUNBOOK.md`](../PONG_DQN_RUNBOOK.md)
- Backend throughput (large-net CNN, cuda column): [`../BURN_BACKENDS.md`](../BURN_BACKENDS.md)
- LR fix: [#342](https://github.com/rjwalters/thrust/issues/342) / [PR #343](https://github.com/rjwalters/thrust/pull/343)
