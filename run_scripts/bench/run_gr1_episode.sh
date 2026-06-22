#!/usr/bin/env bash
# Run ONE real closed-loop GR-1 Tabletop episode (policy server + mujoco sim
# client, with video) — the in-sim counterpart to infer_gr1.py's in-process
# inference loop. A real episode needs both venvs (model in the main venv, sim in
# the robocasa venv) talking over ZeroMQ, so it can't run inside infer_gr1.py.
#
# This is a thin front-end over eval_gr1_thor.sh restricted to one task / one
# episode, so it inherits the same server startup, EGL rendering, and resume logic.
#
# Usage:
#   run_scripts/bench/run_gr1_episode.sh [TASK]
#
# Env overrides (all optional):
#   MODEL_PATH            policy checkpoint     (default RLWRLD/RLDX-1-FT-GR1)
#   N_EPISODES           episodes to run       (default 1)
#   MAX_STEPS            max env steps          (default 720)
#   PORT                 server port            (default 20105)
#   OUT_ROOT             output dir             (default output_final/gr1_episode)
#   COMPILE              server --compile level (default none; submodule/fullgraph)
#   RLDX_PATCHEMBED_FP32 toggle the conv fix    (unset=auto/on for Thor; 0=original bf16)
set -euo pipefail

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
TASK="${1:-PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env}"
MODEL_PATH="${MODEL_PATH:-RLWRLD/RLDX-1-FT-GR1}"
OUT_ROOT="${OUT_ROOT:-$BASE_DIR/output_final/gr1_episode}"

TMP_TASKS="$(mktemp)"
echo "$TASK" > "$TMP_TASKS"
trap 'rm -f "$TMP_TASKS"' EXIT

echo "[i] One closed-loop episode  | task=$TASK | model=$MODEL_PATH"
echo "[i] conv fix: RLDX_PATCHEMBED_FP32=${RLDX_PATCHEMBED_FP32:-auto}  | compile=${COMPILE:-none}"

# eval_gr1_thor.sh reads TASKS_FILE / N_EPISODES / PORT / MAX_STEPS / OUT_ROOT /
# COMPILE from the env; RLDX_PATCHEMBED_FP32 flows through to the server process.
TASKS_FILE="$TMP_TASKS" \
  N_EPISODES="${N_EPISODES:-1}" \
  PORT="${PORT:-20105}" \
  MAX_STEPS="${MAX_STEPS:-720}" \
  N_ACTION="${N_ACTION:-16}" \
  OUT_ROOT="$OUT_ROOT" \
  COMPILE="${COMPILE:-}" \
  bash "$BASE_DIR/run_scripts/eval/gr1_tabletop/eval_gr1_thor.sh" "$MODEL_PATH"

echo "[i] Result + video under: $OUT_ROOT/$TASK/"
