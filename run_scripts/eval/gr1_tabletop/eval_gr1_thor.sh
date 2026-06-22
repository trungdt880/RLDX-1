#!/usr/bin/env bash
# GR-1 Tabletop evaluation, tailored for the aarch64 Jetson Thor setup.
#
# Differences from the stock run_scripts/eval/gr1_tabletop/eval_gr1.sh:
#   - Serves the policy from the main venv's python directly (RLDX_ATTN_IMPL=sdpa),
#     since flash-attn is not installed on aarch64 and `uv run` is unnecessary.
#   - Waits for the server port instead of a fixed `sleep 30`.
#   - EGL headless rendering for the mujoco client.
#   - Configurable task subset + episode count; writes per-task CSVs and prints
#     an aggregated success-rate summary at the end (via aggregate_gr1.py).
#   - Resumable: rollout_policy.py skips episodes already recorded in each
#     task's simulation_results.csv, so re-running continues where it left off.
#
# Usage:
#   run_scripts/eval/gr1_tabletop/eval_gr1_thor.sh [MODEL_PATH]
#
# Env overrides:
#   N_EPISODES   episodes per task           (default 3; use 50 for paper-comparable)
#   PORT         policy server port          (default 20100)
#   MAX_STEPS    max env steps per episode   (default 720)
#   N_ACTION     action steps per chunk      (default 16)
#   TASKS_FILE   newline-separated task list (default: all 24 built in below)
#   OUT_ROOT     output directory           (default output_final/gr1_tabletop/<tag>)
#   COMPILE      server --compile level      (default none; only "submodule"/
#                "fullgraph" are valid — neither helps on Thor, see
#                run_scripts/bench/REPORT.md. cudagraph was withdrawn: it freezes
#                the camera frame and breaks closed-loop control.)
set -euo pipefail

MODEL_PATH="${1:-RLWRLD/RLDX-1-FT-GR1}"
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$BASE_DIR"

MAIN_PY="$BASE_DIR/.venv/bin/python"
GR1_PY="$BASE_DIR/rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python"
PORT="${PORT:-20100}"
N_EPISODES="${N_EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-720}"
N_ACTION="${N_ACTION:-16}"
COMPILE="${COMPILE:-}"

TAG="$(echo "$MODEL_PATH" | tr '/' '_')"
OUT_ROOT="${OUT_ROOT:-$BASE_DIR/output_final/gr1_tabletop/$TAG}"

# --- task list -------------------------------------------------------------
DEFAULT_TASKS=(
  PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)
if [[ -n "${TASKS_FILE:-}" ]]; then
  mapfile -t TASKS < <(grep -vE '^\s*(#|$)' "$TASKS_FILE")
else
  TASKS=("${DEFAULT_TASKS[@]}")
fi

mkdir -p "$OUT_ROOT"
echo "[i] Model     : $MODEL_PATH"
echo "[i] Tasks     : ${#TASKS[@]}"
echo "[i] Episodes  : $N_EPISODES per task   (max_steps=$MAX_STEPS, n_action=$N_ACTION)"
echo "[i] Output    : $OUT_ROOT"

# --- policy server (main venv, sdpa attention) -----------------------------
SERVER_LOG="$OUT_ROOT/server.log"
echo "[i] Starting policy server on 127.0.0.1:$PORT (log: $SERVER_LOG)"
COMPILE_ARGS=()
if [[ -n "$COMPILE" && "$COMPILE" != "none" ]]; then
  COMPILE_ARGS=(--compile "$COMPILE")
  echo "[i] Inference acceleration: --compile $COMPILE"
  # CUDA-graph/Triton paths need CUDA-13 ptxas for sm_11x Triton codegen.
  export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda-13.0/bin/ptxas}"
fi
RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 "$MAIN_PY" rldx/eval/run_rldx_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --use-sim-policy-wrapper \
    "${COMPILE_ARGS[@]}" \
    --host 127.0.0.1 --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

echo "[i] Waiting for server to bind port $PORT ..."
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "[!] Server died during startup. Tail of $SERVER_LOG:"; tail -20 "$SERVER_LOG"; exit 1
  fi
  if ss -tln 2>/dev/null | grep -q ":$PORT "; then echo "[i] Server is up."; break; fi
  sleep 5
done
if ! ss -tln 2>/dev/null | grep -q ":$PORT "; then
  echo "[!] Server did not bind port $PORT within timeout. Tail of $SERVER_LOG:"; tail -20 "$SERVER_LOG"; exit 1
fi

# --- rollout loop (sim venv, EGL headless) ---------------------------------
for task in "${TASKS[@]}"; do
  out_dir="$OUT_ROOT/$task"
  mkdir -p "$out_dir"
  echo "[i] === $task ($(date +%H:%M:%S)) ==="
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NO_ALBUMENTATIONS_UPDATE=1 "$GR1_PY" rldx/eval/rollout_policy.py \
      --policy_client_host 127.0.0.1 --policy_client_port "$PORT" \
      --env_name "gr1_unified/$task" \
      --n_episodes "$N_EPISODES" \
      --max_episode_steps "$MAX_STEPS" \
      --n_action_steps "$N_ACTION" \
      --n_envs 1 \
      --video_dir "$out_dir" 2>&1 | tee "$out_dir/eval.log" | grep -E "success rate|results:" || true
done

# --- aggregate -------------------------------------------------------------
echo
echo "[i] All tasks done. Aggregating ..."
"$MAIN_PY" "$BASE_DIR/run_scripts/eval/gr1_tabletop/aggregate_gr1.py" "$OUT_ROOT"
