#!/usr/bin/env bash
#
# weekend_alc_dispatch.sh — fan-out cluster run for #134 (probe sweep -> seeded
# full runs). Coordinator-driven; ~10 independent SSH hosts, no shared FS.
#
# Cluster reality (probed 2026-06-26, user=sphere):
#   - cargo lives at ~/.cargo/bin/cargo (NOT on the non-interactive PATH);
#     present on all hosts except alc-3 (this script installs rustup there).
#   - No host is a git clone of thrust; we PROVISION by rsync'ing the source
#     from this coordinator to $REMOTE_REPO (auth-free, exact committed tree).
#   - No host has the CUDA toolkit (nvcc) -> #219 is NOT runnable this weekend;
#     Stage 3 is omitted. Track CUDA-toolkit install on the 4090 node separately.
#
# Strategy (diagnostic-first ladder):
#   Provision : rustup-if-missing + rsync source + release build, all hosts.
#   Stage 1   : PROBE SWEEP — train_br_probe over {cells} x {knob sets}, fanned
#               across hosts. A (cell,knobset) "wins" if mean_ep_return improves
#               >5% (less negative) late-vs-early. Detached + sentinel-polled.
#   Stage 2   : SEEDED FULL RUNS — winners x $SEEDS x {PSRO, NFSP} at full budget.
#               No winners => strongest negative result (write up, close #134/#230;
#               override with FORCE_STAGE2=1).
#
# Remote jobs run under `setsid nohup` writing <label>.log + <label>.done so the
# run survives coordinator/SSH disconnects. Re-running re-uses finished sentinels.
#
# Usage:  bash scripts/weekend_alc_dispatch.sh 2>&1 | tee weekend_dispatch.log
#
set -uo pipefail

# ===========================================================================
# CONFIG
# ===========================================================================
HOSTS=(alc-0 alc-2 alc-3 alc-4 alc-5 alc-6 alc-7 alc-8 alc-9 alc-10)
REMOTE_REPO="${REMOTE_REPO:-/home/sphere/thrust-alc}"  # provisioned per host
CARGO="${CARGO:-\$HOME/.cargo/bin/cargo}"              # explicit (non-interactive PATH)
FEATURES="training,env-bucket-brigade"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CELLS=(beta01 beta05 beta09)
SEEDS=(42 43 44 45)
PROBE_ITERS="${PROBE_ITERS:-40}"
PROBE_ROLLOUT="${PROBE_ROLLOUT:-8192}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-48}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-8192}"
POLL_SECS="${POLL_SECS:-30}"

# Stage 1 knob grid: "label;ENV". Probe-only levers most likely to move return.
# Do NOT add BR_MAX_MINIBATCHES_PER_EPOCH (validation doc: it hurts the critic).
KNOBSETS=(
  "base;"
  "vf1;VF_COEF=1.0"
  "br16;BR_TRAIN_STEPS=16"
  "bigroll;ROLLOUT_STEPS=16384"
  "rs01;BR_REWARD_SCALE=0.01"
  "combo;VF_COEF=1.0 BR_TRAIN_STEPS=16 ROLLOUT_STEPS=16384"
)

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$LOCAL_REPO/alc_results_$STAMP"; mkdir -p "$OUT"
REMOTE_OUT="/tmp/thrust_alc_$STAMP"
NHOSTS=${#HOSTS[@]}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"
echo "==> coordinator out: $OUT"
echo "==> ${NHOSTS} hosts: ${HOSTS[*]}  | remote out: $REMOTE_OUT"

# ===========================================================================
# Provision — rustup-if-missing + rsync source + build, all hosts in parallel.
# ===========================================================================
echo "==> Provisioning ${NHOSTS} hosts (rustup-if-missing, rsync source, build)"
prov_pids=()
for host in "${HOSTS[@]}"; do
  (
    # 1. ensure rust
    $SSH "$host" 'test -x $HOME/.cargo/bin/cargo' 2>/dev/null \
      || $SSH "$host" 'curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y' \
      || { echo "[$host] rustup install FAILED"; exit 1; }
    # 2. rsync source (exclude build artifacts + vcs + web deps)
    rsync -az --delete -e "$SSH" \
      --exclude target/ --exclude .git/ --exclude 'alc_results_*' \
      --exclude node_modules/ --exclude web/node_modules/ \
      "$LOCAL_REPO/" "$host:$REMOTE_REPO/" \
      || { echo "[$host] rsync FAILED"; exit 1; }
    # 3. build release examples
    $SSH "$host" "cd $REMOTE_REPO && $CARGO build --release --features '$FEATURES' \
        --example train_br_probe --example train_psro --example train_nfsp \
        && echo BUILD_OK" \
      || { echo "[$host] build FAILED"; exit 1; }
    echo "[$host] provisioned"
  ) > "$OUT/provision_${host}.log" 2>&1 &
  prov_pids+=($!)
done
PROV_OK=()
for i in "${!prov_pids[@]}"; do
  if wait "${prov_pids[$i]}"; then PROV_OK+=("${HOSTS[$i]}")
  else echo "!! provision FAILED: ${HOSTS[$i]} (see $OUT/provision_${HOSTS[$i]}.log)"; fi
done
HOSTS=("${PROV_OK[@]}"); NHOSTS=${#HOSTS[@]}
[[ "$NHOSTS" -eq 0 ]] && { echo "!! no hosts provisioned — aborting"; exit 1; }
echo "==> Provisioned ${NHOSTS}/${#PROV_OK[@]} hosts: ${HOSTS[*]}"

# ===========================================================================
# dispatch <stage> <job...>   job = "LABEL|ENVSTRING|BINARY"
# Detached remote launch (setsid nohup), then poll <label>.done sentinels.
# ===========================================================================
dispatch() {
  local stage="$1"; shift
  local -a jobs=("$@"); local njobs=${#jobs[@]} h i
  echo "==> [$stage] $njobs jobs across $NHOSTS hosts (detached)"
  for ((h=0; h<NHOSTS; h++)); do
    local host="${HOSTS[$h]}"
    local remote="mkdir -p $REMOTE_OUT; cd $REMOTE_REPO || exit 1;"
    for ((i=h; i<njobs; i+=NHOSTS)); do
      IFS='|' read -r label env bin <<< "${jobs[$i]}"
      remote+=" setsid bash -c '$env ./target/release/examples/$bin \
        > $REMOTE_OUT/$label.log 2>&1; echo \$? > $REMOTE_OUT/$label.done' </dev/null >/dev/null 2>&1 &"
    done
    $SSH "$host" "$remote" </dev/null
  done
  # poll for all sentinels
  local total=$njobs done_n=0
  while :; do
    done_n=0
    for ((h=0; h<NHOSTS; h++)); do
      local host="${HOSTS[$h]}"
      local cnt
      cnt=$($SSH "$host" "ls $REMOTE_OUT/*.done 2>/dev/null | wc -l" </dev/null 2>/dev/null || echo 0)
      done_n=$((done_n + cnt))
    done
    echo "    [$stage] $done_n/$total done"
    [[ "$done_n" -ge "$total" ]] && break
    sleep "$POLL_SECS"
  done
  echo "==> [$stage] collecting results"
  for host in "${HOSTS[@]}"; do
    mkdir -p "$OUT/$host"
    rsync -az -e "$SSH" "$host:$REMOTE_OUT/" "$OUT/$host/" 2>/dev/null || true
  done
}

# ===========================================================================
# Stage 1 — probe sweep
# ===========================================================================
S1_JOBS=()
for cell in "${CELLS[@]}"; do
  for ks in "${KNOBSETS[@]}"; do
    kslabel="${ks%%;*}"; ksenv="${ks#*;}"
    S1_JOBS+=("probe_${cell}_${kslabel}|CELL=$cell ITERATIONS=$PROBE_ITERS ROLLOUT_STEPS=$PROBE_ROLLOUT $ksenv|train_br_probe")
  done
done
dispatch "Stage1" "${S1_JOBS[@]}"

WINNERS="$OUT/winners.tsv"; : > "$WINNERS"
SUMMARY="$OUT/stage1_summary.txt"; : > "$SUMMARY"
echo "==> Stage 1 analysis (early5 vs late5 mean_ep_return):"
for cell in "${CELLS[@]}"; do
  for ks in "${KNOBSETS[@]}"; do
    kslabel="${ks%%;*}"; ksenv="${ks#*;}"
    log=$(ls "$OUT"/*/"probe_${cell}_${kslabel}.log" 2>/dev/null | head -1)
    [[ -z "$log" ]] && { echo "$cell/$kslabel: NO LOG" | tee -a "$SUMMARY"; continue; }
    mapfile -t RET < <(grep -oE 'mean_ep_return=[-0-9.]+' "$log" | sed 's/mean_ep_return=//')
    N=${#RET[@]}
    [[ $N -lt 6 ]] && { echo "$cell/$kslabel: INSUFFICIENT ($N)" | tee -a "$SUMMARY"; continue; }
    early=$(printf '%s\n' "${RET[@]:0:5}" | awk '{s+=$1} END{print s/NR}')
    late=$(printf '%s\n' "${RET[@]: -5}" | awk '{s+=$1} END{print s/NR}')
    verdict=$(awk -v e="$early" -v l="$late" 'BEGIN{g=(l-e)/(e<0?-e:(e==0?1:e)); printf (g>0.05)?"WIN":"flat";}')
    printf '%-8s %-8s early5=%-10.1f late5=%-10.1f -> %s\n' "$cell" "$kslabel" "$early" "$late" "$verdict" | tee -a "$SUMMARY"
    [[ "$verdict" == "WIN" ]] && printf '%s\t%s\n' "$cell" "$ksenv" >> "$WINNERS"
  done
done
NWIN=$(wc -l < "$WINNERS" | tr -d ' ')
echo "==> Stage 1: $NWIN winning (cell,knobset) pair(s). Summary: $SUMMARY"

# ===========================================================================
# Stage 2 — seeded full runs (gated on winners)
# ===========================================================================
if [[ "$NWIN" -eq 0 && "${FORCE_STAGE2:-0}" != "1" ]]; then
  echo "==> GATE FAIL: no knob set raised mean_ep_return on any cell at rollout=$PROBE_ROLLOUT."
  echo "    Strongest negative result yet. Recommend: write up + close #134/#230."
  echo "    (Override: FORCE_STAGE2=1.)  Done. Results: $OUT"
  exit 0
fi
[[ "$NWIN" -eq 0 ]] && for cell in "${CELLS[@]}"; do printf '%s\t%s\n' "$cell" "" >> "$WINNERS"; done

echo "==> Stage 2: seeded full runs (seeds: ${SEEDS[*]}, budget ${TOTAL_ITERATIONS}x${ROLLOUT_STEPS})"
S2_JOBS=()
while IFS=$'\t' read -r cell ksenv; do
  [[ -z "$cell" ]] && continue
  for seed in "${SEEDS[@]}"; do
    common="CELL=$cell SEED=$seed TOTAL_ITERATIONS=$TOTAL_ITERATIONS ROLLOUT_STEPS=$ROLLOUT_STEPS CHECKPOINT_INTERVAL_ITERATIONS=5 $ksenv"
    S2_JOBS+=("psro_${cell}_s${seed}|$common ALPHA_RANK_NORMALIZE_SPAN=1 BR_REWARD_SCALE=0.01|train_psro")
    S2_JOBS+=("nfsp_${cell}_s${seed}|$common AP_COVERAGE=2.0 BR_REWARD_SCALE=0.001|train_nfsp")
  done
done < "$WINNERS"
echo "==> Stage 2: ${#S2_JOBS[@]} runs queued"
dispatch "Stage2" "${S2_JOBS[@]}"
echo "==> Stage 2 complete. Logs+checkpoints under $OUT/<host>/"
echo "    Next (manual): 1000-ep eval + gap_closed_cell per policy; aggregate"
echo "    across seeds (mean +/- sd) for the #134 writeup."
echo "==> ALL DONE. Everything under $OUT"
