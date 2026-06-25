#!/bin/bash
# Single-GPU robocasa smoke eval: 1 episode per task, all 24 kitchen tasks.
# Serves the model once on one GPU, then runs every task sequentially.
# Usage: eval_robocasa_1ep.sh <CKPT_NAME> [GPU_ID]
set -u
export NO_ALBUMENTATIONS_UPDATE=1

CKPT_NAME=${1:?"Usage: $0 <CKPT_NAME> [GPU_ID]"}
GPU_ID=${2:-0}
BASE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
MODEL_PATH="$CKPT_NAME"

OUT_ROOT="$BASE_DIR/output_final/robocasa_1ep/$CKPT_NAME"
LOG_ROOT="$OUT_ROOT/_launcher_logs"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

TASK_NAMES=(
  "TurnSinkSpout"  "TurnOnStove"  "TurnOnSinkFaucet"  "TurnOnMicrowave"
  "TurnOffStove"  "TurnOffSinkFaucet"
  "TurnOffMicrowave"  "PnPStoveToCounter"  "PnPSinkToCounter"  "PnPMicrowaveToCounter"
  "PnPCounterToStove"  "PnPCounterToSink"
  "PnPCounterToMicrowave"  "PnPCounterToCab"  "PnPCabToCounter"  "OpenSingleDoor"
  "OpenDrawer"  "OpenDoubleDoor"
  "CoffeeSetupMug"  "CoffeeServeMug"  "CoffeePressButton"  "CloseSingleDoor"
  "CloseDrawer"  "CloseDoubleDoor"
)
PORT=20100
SHARD_LOG="$LOG_ROOT/server.log"

echo "[launcher] GPU=${GPU_ID} PORT=${PORT} tasks=${#TASK_NAMES[@]} episodes=1/task" | tee -a "$SHARD_LOG"

# Start the policy server once.
RLDX_PATCHEMBED_FP32=1 RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=$GPU_ID \
  uv run python "$BASE_DIR/rldx/eval/run_rldx_server.py" \
    --model-path "$MODEL_PATH" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --use-sim-policy-wrapper \
    --host 127.0.0.1 \
    --port "$PORT" >> "$SHARD_LOG" 2>&1 &
SERVE_PID=$!
echo "[launcher] server PID=${SERVE_PID}" | tee -a "$SHARD_LOG"

cleanup() { kill "$SERVE_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Wait for the server to come up (poll port, up to ~4 min).
for _ in $(seq 1 120); do
  if ss -lnt | awk '{print $4}' | grep -q ":$PORT$"; then
    echo "[launcher] server ready on port ${PORT}" | tee -a "$SHARD_LOG"
    break
  fi
  sleep 2
done

for task_name in "${TASK_NAMES[@]}"; do
  out_dir="$OUT_ROOT/$task_name"
  mkdir -p "$out_dir"
  echo "[launcher] running ${task_name}" | tee -a "$SHARD_LOG"
  RLDX_SKIP_MODEL_REGISTRY=1 \
  "$BASE_DIR/rldx/eval/sim/robocasa/robocasa_uv/.venv/bin/python" \
    "$BASE_DIR/rldx/eval/rollout_policy.py" \
      --n_episodes 1 \
      --policy_client_host 127.0.0.1 \
      --policy_client_port "$PORT" \
      --max_episode_steps 720 \
      --env_name "robocasa_panda_omron/${task_name}_PandaOmron_Env" \
      --n_action_steps 16 \
      --n_envs 1 \
      --video_dir "$out_dir" \
      >> "$out_dir/eval.log" 2>&1
  echo "[launcher] ${task_name} done (exit=$?)" | tee -a "$SHARD_LOG"
done

echo "[launcher] All tasks complete" | tee -a "$SHARD_LOG"
