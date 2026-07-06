#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# setup_sim.sh — one-shot, sim-ONLY bootstrap for the ALLEX ⇄ RoboCasa harness.
#
# This sets up EVERYTHING needed to load a RoboCasa scene with the ALLEX humanoid
# and replay a baked trajectory (`replay_in_scene.py`) — and NOTHING from the
# RLDX-1 model side (no 16GB checkpoint, no torch, no ZeroMQ policy server). It is
# for the person who only works on the simulation with `allex_model`.
#
# What it does (idempotent — safe to re-run):
#   1. sanity-checks prerequisites (uv, the robocasa fork, your allex_model repo)
#   2. creates a Python 3.10 venv with `uv`
#   3. installs the pinned sim stack (robosuite 1.5.1 + the ALLEX robocasa fork
#      editable + mujoco/numpy/imageio) — versions matched to the working machine
#   4. symlinks the ALLEX meshes into the fork
#   5. downloads the RoboCasa scene/object assets (~8 GB) if they are missing
#   6. runs a headless smoke test (build the ALLEX env + one replay step)
#
# Usage (from anywhere):
#   run_scripts/eval/robocasa_allex/setup_sim.sh
#
# Knobs (env vars):
#   ALLEX_MODEL_DIR   path to the allex_model repo   [default: ~/workspace/allex_model]
#   ALLEX_SIM_VENV    where to create the venv        [default: <fork>/robocasa_uv/.venv]
#   ALLEX_SKIP_ASSETS =1 to skip the ~8 GB asset download (already have them)
#   ALLEX_SKIP_SMOKE  =1 to skip the final headless smoke test
#   ALLEX_FORCE_VENV  =1 to delete and recreate the venv from scratch
#
set -euo pipefail

# ----------------------------------------------------------------------------- paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then :; else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # run_scripts/eval/robocasa_allex -> repo root
fi
FORK="$REPO_ROOT/external_dependencies/robocasa-gr1-tabletop-tasks"
ALLEX_MODEL_DIR="${ALLEX_MODEL_DIR:-$HOME/workspace/allex_model}"
VENV="${ALLEX_SIM_VENV:-$FORK/robocasa_uv/.venv}"
PY="$VENV/bin/python"
MESH_LINK="$FORK/robocasa/models/assets/robots/allex/meshes"

say()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup] WARN:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------- 1. preflight
say "repo root      : $REPO_ROOT"
say "robocasa fork  : $FORK"
say "allex_model    : $ALLEX_MODEL_DIR"
say "target venv    : $VENV"

command -v uv >/dev/null 2>&1 || die \
    "uv is not installed. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh   (then re-open the shell)"

[ -f "$FORK/setup.py" ] || die "robocasa fork not found at $FORK (are you in the RLDX-1 checkout?)"
[ -f "$FORK/robocasa/models/robots/manipulators/allex_robot.py" ] || die \
    "ALLEX integration missing from the fork — expected robocasa/models/robots/manipulators/allex_robot.py"

[ -d "$ALLEX_MODEL_DIR/meshes" ] || die \
    "allex_model meshes not found at $ALLEX_MODEL_DIR/meshes — clone allex_model there, or set ALLEX_MODEL_DIR=/path/to/allex_model"
[ -d "$ALLEX_MODEL_DIR/examples/motions" ] || warn \
    "no $ALLEX_MODEL_DIR/examples/motions — replay needs a motion .npz (hello/demo1); scenes still build."

# ----------------------------------------------------------------------------- 2. venv
if [ "${ALLEX_FORCE_VENV:-0}" = "1" ] && [ -d "$VENV" ]; then
    say "ALLEX_FORCE_VENV=1 -> removing existing venv"; rm -rf "$VENV"
fi
if [ -x "$PY" ]; then
    say "venv exists ($("$PY" --version 2>&1)) — reusing (set ALLEX_FORCE_VENV=1 to rebuild)"
else
    say "creating Python 3.10 venv with uv ..."
    uv venv --python 3.10 "$VENV"
fi

# ----------------------------------------------------------------------------- 3. deps
# Pinned to the versions verified working on the reference machine. The editable
# robocasa install pulls the rest (numpy==1.26.4, mujoco==3.2.6, numba, scipy,
# tianshou, ...) from the fork's setup.py. gymnasium/imageio-ffmpeg are needed by
# the gym wrappers + mp4 rendering but are not in robocasa's install_requires.
say "installing sim stack (robosuite 1.5.1 + robocasa fork [editable] + deps) ..."
uv pip install --python "$VENV" \
    "robosuite==1.5.1" \
    "gymnasium==0.29.1" \
    "imageio==2.37.3" \
    "imageio-ffmpeg==0.6.0" \
    -e "$FORK"

# ----------------------------------------------------------------------------- 4. meshes
say "linking ALLEX meshes -> $MESH_LINK"
ln -sfn "$ALLEX_MODEL_DIR/meshes" "$MESH_LINK"

# ----------------------------------------------------------------------------- 5. assets (~8 GB)
ASSET_PROBE="$FORK/robocasa/models/assets/fixtures"
if [ "${ALLEX_SKIP_ASSETS:-0}" = "1" ]; then
    say "ALLEX_SKIP_ASSETS=1 -> skipping asset download"
elif [ -d "$ASSET_PROBE" ] && [ "$(find "$ASSET_PROBE" -type f 2>/dev/null | head -1)" ]; then
    say "RoboCasa assets already present -> skipping download"
else
    say "downloading RoboCasa scene/object assets (~8 GB, one time) ..."
    warn "this is large and network-bound; re-run the script if it is interrupted (downloads resume/skip)."
    # the download scripts do sibling imports, so run them from their own dir
    ( cd "$FORK/robocasa/scripts" && "$PY" download_kitchen_assets.py -y )
    ( cd "$FORK/robocasa/scripts" && "$PY" download_tabletop_assets.py -y )
fi

# ----------------------------------------------------------------------------- 6. smoke test
if [ "${ALLEX_SKIP_SMOKE:-0}" = "1" ]; then
    say "ALLEX_SKIP_SMOKE=1 -> skipping smoke test"
else
    say "headless smoke test: build the ALLEX env + reset + one step ..."
    ALLEX_CONTRACT_DIR="$SCRIPT_DIR/contract" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$PY" - <<'PYEOF'
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import gymnasium as gym
import robocasa  # noqa: F401  (registers robocasa_allex/* env ids)
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
env = gym.make("robocasa_allex/PnPCanToBowl_AllexRobot_Env", enable_render=True, seed=0)
obs, info = env.reset()
robot = env.unwrapped.env.robots[0]
neq = int(env.unwrapped.env.sim.model._model.neq)
# one no-op step (hold reset pose) through the 6-group ALLEX action contract
sys.path.insert(0, os.environ["ALLEX_CONTRACT_DIR"])
import allex_contract as C
action = {f"action.{g}": np.zeros(C.GROUP_DIMS[g]) for g in C.GROUP_ORDER}
obs, r, term, trunc, info = env.step(action)
ego = np.asarray(obs["video.camera_ego_left"])
env.close()
assert neq == 12, f"expected 12 equality couplings, got {neq}"
assert ego.ndim == 3 and ego.shape[-1] == 3 and np.var(ego) > 1.0, "ego camera returned a blank frame"
print(f"  OK  gripperless={len(robot.arms)==0}  neq={neq}  ego={ego.shape} var={np.var(ego):.1f}  reward={float(r)}")
PYEOF
fi

# ----------------------------------------------------------------------------- done
cat <<EOF

$(printf '\033[1;32m[setup] DONE — ALLEX sim harness is ready.\033[0m')

  venv python : $PY

Try it (headless -> mp4):
  MUJOCO_GL=egl $PY \\
      $REPO_ROOT/run_scripts/eval/robocasa_allex/replay_in_scene.py --motion hello

Live window (needs a display, e.g. a monitor on the box):
  DISPLAY=:0 $PY \\
      $REPO_ROOT/run_scripts/eval/robocasa_allex/replay_in_scene.py --viewer

More options + full docs:
  $REPO_ROOT/run_scripts/eval/robocasa_allex/SIM_ONLY_SETUP.md
EOF
