#!/usr/bin/env bash
# Launch the RLDX-1-PT inference server for the DROID embodiment over ZeroMQ.
#
# This is the SERVER side for sim-evals (arhanjain/sim-evals, Isaac Sim DROID).
# The Isaac Sim process runs a ZeroMQ *client* (rldx.policy.server_client.PolicyClient,
# or any client speaking the same msgpack wire protocol) and calls the `get_action`
# endpoint each control step.
#
# Wire format (what the client must send to `get_action`):
#   observation (flat keys; use_sim_policy_wrapper translates these to the nested
#   format RLDXPolicy expects):
#     video.primary    uint8  (B, 4, H, W, 3)   # 4-frame history, delta_indices [-6,-4,-2,0]
#     video.secondary  uint8  (B, 4, H, W, 3)
#     video.wrist      uint8  (B, 4, H, W, 3)
#     state.end_effector_position   float32 (B, 1, 3)
#     state.end_effector_rotation   float32 (B, 1, 3)   # euler xyz
#     state.gripper_position        float32 (B, 1, 1)
#     annotation.human.action.task_description   list[str] length B
#   options (optional): {"session_ids": [...], "reset_memory": [bool, ...]}
#
#   action returned (flat keys), each a 16-step chunk:
#     action.end_effector_position  float32 (B, 16, 3)   # DELTA EEF position
#     action.end_effector_rotation  float32 (B, 16, 3)   # DELTA EEF rotation (euler)
#     action.gripper_close          float32 (B, 16, 1)   # ABSOLUTE gripper-close
#
# NOTE: this is an end-effector *delta* action space. sim-evals expects joint-
# position actions, so the sim side must convert delta-EEF -> absolute pose ->
# IK -> joint targets (and gripper_close -> gripper command).
#
# Usage:
#   run_scripts/serve/run_rldx_pt_droid_server.sh                 # 0.0.0.0:5555
#   PORT=6000 HOST=127.0.0.1 run_scripts/serve/run_rldx_pt_droid_server.sh
#   MODEL_PATH=/abs/path/to/ckpt run_scripts/serve/run_rldx_pt_droid_server.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL_PATH="${MODEL_PATH:-RLWRLD/RLDX-1-PT}"
EMBODIMENT="${EMBODIMENT:-OXE_DROID}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5555}"
PYTHON="${PYTHON:-.venv/bin/python}"
# AUTOBATCH=1 tolerates unbatched per-step obs from the client
# (state (D,)->(1,1,D), video (T,H,W,C)->(1,T,H,W,C)). Off by default.
AUTOBATCH="${AUTOBATCH:-0}"
# SAVE_FRAMES=1 dumps each incoming video.* frame to SAVE_FRAMES_DIR
# (every SAVE_FRAMES_EVERY-th get_action). Off by default.
SAVE_FRAMES="${SAVE_FRAMES:-0}"
SAVE_FRAMES_DIR="${SAVE_FRAMES_DIR:-./server_frames}"
SAVE_FRAMES_EVERY="${SAVE_FRAMES_EVERY:-1}"

# Jetson Thor (sm_11x) runtime knobs:
#   - sdpa attention (no flash-attn kernel on sm_110; see memory notes)
#   - skip albumentations' online version check (no network on-device)
#   - the bf16 Conv3d patch-embed fp32 fallback is AUTO-detected on sm_11x
#     (rldx/model/modules/backbone/modeling_qwen3_vl.py); override with
#     RLDX_PATCHEMBED_FP32=1/0 if needed.
export RLDX_ATTN_IMPL="${RLDX_ATTN_IMPL:-sdpa}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

echo "Launching RLDX-1-PT DROID ZeroMQ server"
echo "  model_path  : ${MODEL_PATH}"
echo "  embodiment  : ${EMBODIMENT}"
echo "  bind        : tcp://${HOST}:${PORT}"
echo "  attn impl   : ${RLDX_ATTN_IMPL}"
echo

EXTRA_FLAGS=()
[ "${AUTOBATCH}" = "1" ] && EXTRA_FLAGS+=(--auto-batch-obs)
if [ "${SAVE_FRAMES}" = "1" ]; then
  EXTRA_FLAGS+=(--save-frames --save-frames-dir "${SAVE_FRAMES_DIR}" \
               --save-frames-every "${SAVE_FRAMES_EVERY}")
fi

exec "${PYTHON}" -m rldx.eval.run_rldx_server \
  --model-path "${MODEL_PATH}" \
  --embodiment-tag "${EMBODIMENT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --use-sim-policy-wrapper \
  "${EXTRA_FLAGS[@]}" \
  --strict
