#!/usr/bin/env bash
# Full GR-1 Tabletop eval on the ORIGINAL (no-conv-fix) policy — RLDX_PATCHEMBED_FP32=0
# forces the slow bf16 Conv3d path (~6 s/obs). Writes to its OWN output dir so it
# never overwrites the conv-fix results under output_final/gr1_tabletop/<model>.
#
# WARNING: the un-fixed conv is ~6 s per policy query, so a full 50-episode run is
# VERY slow — roughly 2-3 DAYS for all 24 tasks x 50 eps. Use N_EPISODES to shrink.
# It is resumable (skips episodes already in each task's simulation_results.csv).
#
# Usage:
#   run_scripts/bench/run_gr1_full_original.sh [MODEL_PATH]
#
# Env overrides:
#   N_EPISODES   episodes per task   (default 50 = paper-comparable)
#   OUT_ROOT     output dir          (default output_final/gr1_tabletop/<tag>__original_noconvfix_full)
#   PORT         server port         (default 20106 — distinct from the default 20100)
#   MAX_STEPS    max env steps       (default 720)
#   TASKS_FILE   task subset         (default: all 24)
set -euo pipefail

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$BASE_DIR"

MODEL_PATH="${1:-RLWRLD/RLDX-1-FT-GR1}"
TAG="$(echo "$MODEL_PATH" | tr '/' '_')"
N_EPISODES="${N_EPISODES:-50}"
PORT="${PORT:-20106}"
OUT_ROOT="${OUT_ROOT:-$BASE_DIR/output_final/gr1_tabletop/${TAG}__original_noconvfix_full}"

if [[ "$OUT_ROOT" == "$BASE_DIR/output_final/gr1_tabletop/$TAG" ]]; then
  echo "[!] refusing to run: OUT_ROOT collides with the conv-fix results dir." >&2
  echo "    pick a different OUT_ROOT." >&2
  exit 1
fi

echo "================================================================"
echo " FULL eval — ORIGINAL (no conv fix, RLDX_PATCHEMBED_FP32=0, ~6 s/obs)"
echo "   model     : $MODEL_PATH"
echo "   episodes  : $N_EPISODES / task x 24 tasks"
echo "   out dir   : $OUT_ROOT   (separate — won't touch the fix results)"
echo "   port      : $PORT"
echo "   est. time : ~2-3 days at 50 eps (the un-fixed conv is ~6 s/query)"
echo " started: $(date)"
echo "================================================================"

# RLDX_PATCHEMBED_FP32=0 is exported here so the policy server subprocess that
# eval_gr1_thor.sh launches inherits it and runs the original bf16 conv.
export RLDX_PATCHEMBED_FP32=0

N_EPISODES="$N_EPISODES" PORT="$PORT" OUT_ROOT="$OUT_ROOT" \
  MAX_STEPS="${MAX_STEPS:-720}" \
  bash "$BASE_DIR/run_scripts/eval/gr1_tabletop/eval_gr1_thor.sh" "$MODEL_PATH"

echo "================================================================"
echo " ORIGINAL full eval finished: $(date)"
echo " results: $OUT_ROOT"
echo "================================================================"
