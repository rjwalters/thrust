#!/usr/bin/env bash
#
# alc_finalize_stage1.sh — collect + gate the #134 Stage 1 probe sweep.
#
# Standalone replacement for the dispatcher's end-of-stage logic (the
# coordinator died on a laptop sleep; the remote setsid jobs survived). Safe to
# run repeatedly: it rsyncs the latest remote logs and recomputes the gate.
# Does NOT re-dispatch or launch Stage 2 — run that manually if a winner lands.
#
# Usage: bash scripts/alc_finalize_stage1.sh
#
set -uo pipefail
RO="/tmp/thrust_alc_20260626_174949"          # remote out dir (this run)
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/alc_results_20260626_174949"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
# host -> knobset assignment for this run (parallel indexed arrays; bash 3.2-safe)
HOSTS=(alc-2 alc-4 alc-5 alc-6 alc-7 alc-8)
KSETS=(base  vf1   br16  bigroll rs01 combo)
CELLS=(beta01 beta05 beta09)

echo "==> collecting logs from ${HOSTS[*]}"
for h in "${HOSTS[@]}"; do
  mkdir -p "$OUT/$h"
  rsync -az -e "$SSH" "$h:$RO/" "$OUT/$h/" 2>/dev/null || echo "  (rsync $h failed, skipping)"
done

echo "==> sentinel status"
tot=0
for i in "${!HOSTS[@]}"; do
  h="${HOSTS[$i]}"
  c=$($SSH "$h" "ls $RO/*.done 2>/dev/null | wc -l" 2>/dev/null || echo 0)
  echo "  $h (${KSETS[$i]}): $c/3 done"; tot=$((tot+c))
done
echo "  TOTAL: $tot/18"

echo "==> gate analysis (early5 vs late5 mean_ep_return; WIN if >5% less negative)"
SUMMARY="$OUT/stage1_summary.txt"; : > "$SUMMARY"
WINNERS="$OUT/winners.tsv"; : > "$WINNERS"
for i in "${!HOSTS[@]}"; do
  h="${HOSTS[$i]}"; ks="${KSETS[$i]}"
  for cell in "${CELLS[@]}"; do
    log="$OUT/$h/probe_${cell}_${ks}.log"
    [ -f "$log" ] || { printf '%-8s %-8s NO LOG\n' "$cell" "$ks" | tee -a "$SUMMARY"; continue; }
    RET=()
    while IFS= read -r v; do RET+=("$v"); done < <(grep -oE 'mean_ep_return=[-0-9.]+' "$log" | sed 's/mean_ep_return=//')
    n=${#RET[@]}
    if [ "$n" -lt 6 ]; then
      printf '%-8s %-8s n=%-3s INSUFFICIENT\n' "$cell" "$ks" "$n" | tee -a "$SUMMARY"; continue
    fi
    early=$(printf '%s\n' "${RET[@]:0:5}" | awk '{s+=$1}END{printf "%.0f",s/NR}')
    late=$(printf '%s\n' "${RET[@]: -5}" | awk '{s+=$1}END{printf "%.0f",s/NR}')
    verdict=$(awk -v e="$early" -v l="$late" 'BEGIN{g=(l-e)/(e<0?-e:1);printf (g>0.05)?"WIN":"flat"}')
    printf '%-8s %-8s n=%-3s early5=%-9s late5=%-9s -> %s\n' "$cell" "$ks" "$n" "$early" "$late" "$verdict" | tee -a "$SUMMARY"
    [ "$verdict" = "WIN" ] && printf '%s\t%s\n' "$cell" "$ks" >> "$WINNERS"
  done
done
nwin=$(wc -l < "$WINNERS" | tr -d ' ')
echo "==> WINNERS: $nwin  (summary: $SUMMARY)"
[ "$nwin" -eq 0 ] && echo "==> GATE FAIL so far — no knobset raises mean_ep_return. (combo killed; bigroll pending.)"
