#!/usr/bin/env bash
#
# weekend_alc_run.sh — operator-driven cluster run for issues #134, #230, #219.
#
# Strategy: DIAGNOSTIC-FIRST LADDER (chosen 2026-06-26).
#   Stage 1 (gate): train_br_probe at the larger 8192 rollout across all 3
#                   cells. The open question from the validation doc is whether
#                   the single-BR inner loop can EVER raise mean_ep_return — at
#                   2048/512 it never did (ev rose to ~0.48 but return stayed
#                   flat). If the larger budget still doesn't move return, the
#                   full PSRO/NFSP protocol is dead on arrival; we stop and
#                   write the negative result instead of burning the window.
#   Stage 2 (gated on Stage 1 PASS): the full #134 protocol — 3 cells x
#                   {PSRO, NFSP} at 48 iters x 8192 rollout, with the current
#                   convergence knobs.
#   Stage 3 (independent): #219 CUDA-backend throughput bench (a CUDA-toolkit
#                   node is available this window).
#
# Run from the repo root on a release-capable node:
#   bash scripts/weekend_alc_run.sh 2>&1 | tee weekend_driver.log
#
# Override knobs (all optional):
#   PROBE_ITERS=40 PROBE_ROLLOUT=8192   # Stage 1 budget
#   TOTAL_ITERATIONS=48 ROLLOUT_STEPS=8192  # Stage 2 budget
#   FORCE_STAGE2=1   # run Stage 2 even if the gate says FAIL (operator override)
#   SKIP_CUDA=1      # skip Stage 3
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$REPO_ROOT/alc_results_$STAMP"
mkdir -p "$OUT"
echo "==> Results dir: $OUT"

CELLS=(beta01 beta05 beta09)
PROBE_ITERS="${PROBE_ITERS:-40}"
PROBE_ROLLOUT="${PROBE_ROLLOUT:-8192}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-48}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-8192}"
FEATURES="training,env-bucket-brigade"

# ---------------------------------------------------------------------------
# Stage 0 — build once so per-run timing excludes compilation.
# ---------------------------------------------------------------------------
echo "==> Stage 0: building release binaries"
cargo build --release --features "$FEATURES" \
  --example train_br_probe --example train_psro --example train_nfsp \
  2>&1 | tee "$OUT/stage0_build.log"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  echo "!! Build failed — aborting." ; exit 1
fi

# ---------------------------------------------------------------------------
# Stage 1 — GATE: does mean_ep_return improve at 8192 on ANY cell?
# Compares mean of the last 5 iters' mean_ep_return vs the first 5. Returns are
# negative payoff units (~-25000), so "improvement" = LESS negative.
# ---------------------------------------------------------------------------
echo "==> Stage 1 (gate): train_br_probe, rollout=$PROBE_ROLLOUT, iters=$PROBE_ITERS"
GATE_PASS=0
GATE_TABLE="$OUT/stage1_gate_summary.txt"
: > "$GATE_TABLE"

for CELL in "${CELLS[@]}"; do
  LOG="$OUT/probe_${CELL}.log"
  echo "    -> probe $CELL"
  CELL="$CELL" ITERATIONS="$PROBE_ITERS" ROLLOUT_STEPS="$PROBE_ROLLOUT" \
    ./target/release/examples/train_br_probe 2>&1 | tee "$LOG"

  # Pull the mean_ep_return series (one per iter) from the log.
  mapfile -t RET < <(grep -oE 'mean_ep_return=[-0-9.]+' "$LOG" | sed 's/mean_ep_return=//')
  N=${#RET[@]}
  if [[ $N -lt 6 ]]; then
    echo "$CELL: INSUFFICIENT DATA ($N iters logged)" | tee -a "$GATE_TABLE"
    continue
  fi
  # mean of first 5 and last 5
  early=$(printf '%s\n' "${RET[@]:0:5}" | awk '{s+=$1} END{print s/NR}')
  late=$(printf '%s\n' "${RET[@]: -5}" | awk '{s+=$1} END{print s/NR}')
  verdict=$(awk -v e="$early" -v l="$late" 'BEGIN{
    # improvement = less negative => l > e. Relative gain vs |early|.
    gain = (l - e) / (e<0 ? -e : (e==0?1:e));
    printf (gain > 0.05) ? "IMPROVED" : "FLAT";
  }')
  printf '%s: early5=%.1f late5=%.1f  -> %s\n' "$CELL" "$early" "$late" "$verdict" \
    | tee -a "$GATE_TABLE"
  [[ "$verdict" == "IMPROVED" ]] && GATE_PASS=1
done

echo "==> Stage 1 gate summary:"
cat "$GATE_TABLE"

if [[ "$GATE_PASS" -eq 1 ]]; then
  echo "==> GATE: PASS — at least one cell improved mean_ep_return. Proceeding to Stage 2."
else
  echo "==> GATE: FAIL — no cell improved mean_ep_return at $PROBE_ROLLOUT rollout."
  echo "    This is the strongest negative result yet (the inner BR loop cannot"
  echo "    raise team return even at the larger budget). Recommendation: write up"
  echo "    the negative result and close #134/#230 rather than run Stage 2."
  if [[ "${FORCE_STAGE2:-0}" != "1" ]]; then
    echo "    Skipping Stage 2 (set FORCE_STAGE2=1 to override)."
  fi
fi

# ---------------------------------------------------------------------------
# Stage 2 — full #134 protocol (gated). 3 cells x {PSRO, NFSP}.
#   PSRO: convergence knobs are NOT default-on — must set explicitly.
#   NFSP: AP_COVERAGE=2.0 and BR_REWARD_SCALE=0.001 are ALREADY the defaults;
#         we set them explicitly for a self-documenting, reproducible log.
#   Do NOT set BR_MAX_MINIBATCHES_PER_EPOCH: the validation doc shows the critic
#   is data-hungry and capping minibatches HURTS EV.
# ---------------------------------------------------------------------------
if [[ "$GATE_PASS" -eq 1 || "${FORCE_STAGE2:-0}" == "1" ]]; then
  echo "==> Stage 2: full protocol, total_iters=$TOTAL_ITERATIONS, rollout=$ROLLOUT_STEPS"
  for CELL in "${CELLS[@]}"; do
    echo "    -> PSRO $CELL"
    CELL="$CELL" TOTAL_ITERATIONS="$TOTAL_ITERATIONS" ROLLOUT_STEPS="$ROLLOUT_STEPS" \
      ALPHA_RANK_NORMALIZE_SPAN=1 BR_REWARD_SCALE=0.01 \
      CHECKPOINT_INTERVAL_ITERATIONS=5 \
      ./target/release/examples/train_psro 2>&1 | tee "$OUT/psro_${CELL}.log"

    echo "    -> NFSP $CELL"
    CELL="$CELL" TOTAL_ITERATIONS="$TOTAL_ITERATIONS" ROLLOUT_STEPS="$ROLLOUT_STEPS" \
      AP_COVERAGE=2.0 BR_REWARD_SCALE=0.001 \
      ./target/release/examples/train_nfsp 2>&1 | tee "$OUT/nfsp_${CELL}.log"
  done
  echo "==> Stage 2 complete. Logs + checkpoints in $OUT"
  echo "    Next (manual): 1000-ep eval + gap_closed_cell per policy, then writeup."
else
  echo "==> Stage 2 skipped (gate FAIL)."
fi

# ---------------------------------------------------------------------------
# Stage 3 — #219 CUDA throughput bench (independent of Stages 1-2).
# Requires a CUDA-toolkit node (nvcc). Fills the blank `cuda` column in
# docs/BURN_BACKENDS.md.
# ---------------------------------------------------------------------------
if [[ "${SKIP_CUDA:-0}" != "1" ]]; then
  echo "==> Stage 3 (#219): CUDA throughput bench"
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "!! nvcc not found on PATH — CUDA toolkit missing. Skipping Stage 3."
    echo "   (Install the CUDA toolkit or run on a CUDA-equipped node.)"
  else
    nvcc --version 2>&1 | tee "$OUT/stage3_nvcc.txt"
    cargo bench --features "training,cuda" --bench trainer_throughput -- \
      --warm-up-time 2 --measurement-time 10 2>&1 | tee "$OUT/stage3_cuda_bench.log"
    echo "==> Stage 3 complete. Extract the 8 */cuda groups from stage3_cuda_bench.log"
    echo "    and add the cuda column to docs/BURN_BACKENDS.md (record host spec)."
  fi
fi

echo "==> ALL DONE. Results: $OUT"
echo "    Pull this dir back and commit logs/artifacts per #134 task 3-4."
