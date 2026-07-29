#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Find an ALLEX mount pose whose ego view frames the workspace naturally.

Problem (see ego_view_geometry.py): with the current mount
(_ROBOT_POS_OFFSETS['AllexRobot'] = [0, -0.4, 0.9]) the ego camera at the
TRAINING-MEAN neck pitch (0.385 rad) points at a blank wall. Real ALLEX training
frames show the tabletop, both hands, and the manipulated objects. Feeding the
policy a featureless wall is maximally out-of-distribution.

Sweeps mount height x back-offset, renders the ego view at the training-mean
neck pitch, and scores how much task content is in frame:
  * ``struct``  : pixel std (a blank wall is ~9; a framed workspace is ~65)
  * ``lower2/3``: fraction of NON-wall pixels in the lower two thirds
Higher is better on both. Writes a labelled contact sheet.

Run (robosuite venv):
  MUJOCO_GL=egl <sim-venv-python> checks/mount_sweep.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "contract"))
import allex_contract as C  # noqa: E402

ENV_ID = os.environ.get("ALLEX_ENV_ID", "robocasa_allex/PnPCanToBowl_AllexRobot_Env")
CAM = "robot0_zed_left_camera_optical_frame"
CAM_DX = 0.01
NECK_PITCH = 0.385           # training mean — the pose the policy expects
HEIGHTS = [0.55, 0.70, 0.90]  # z of the mount
BACKS = [0.20, 0.30, 0.40]    # metres back from the counter


def main() -> None:
    import gymnasium as gym
    import mujoco
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    import imageio.v2 as imageio
    from robocasa.environments.tabletop import tabletop as TT

    stats = json.load(open(HERE.parent / "contract" / "allex_stats.json"))["general_embodiment"]["state"]
    mean = {g: np.asarray(stats[g]["mean"], dtype=np.float64) for g in C.GROUP_ORDER}
    mean["neck_joints"] = np.array([NECK_PITCH, mean["neck_joints"][1]])

    print("=" * 74)
    print(f"ALLEX mount sweep — ego view at training-mean neck pitch {NECK_PITCH} rad")
    print("=" * 74)
    print(f"{'height':>8}{'back':>7}{'struct':>9}{'lower2/3':>10}{'contacts':>10}")
    print("-" * 74)

    rows, labels = [], []
    orig = list(TT._ROBOT_POS_OFFSETS["AllexRobot"])
    for z in HEIGHTS:
        strip = []
        for b in BACKS:
            TT._ROBOT_POS_OFFSETS["AllexRobot"] = [orig[0], -abs(b), z]
            env = gym.make(ENV_ID, enable_render=True, seed=0)
            env.reset()
            base = env.unwrapped.env
            sim = base.sim
            model, data = sim.model._model, sim.data._data
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM)
            model.cam_pos[cid] = model.cam_pos[cid] + np.array([CAM_DX, 0.0, 0.0])
            for _ in range(50):
                env.step({f"action.{g}": mean[g] for g in C.GROUP_ORDER})
            img = sim.render(width=256, height=256, camera_name=CAM)[::-1]
            lum = img.astype(np.float32).mean(axis=-1)
            struct = float(lum.std())
            lower = lum[lum.shape[0] // 3:, :]
            nonwall = float((lower < 235).mean())
            ncon = int(data.ncon)
            print(f"{z:>8.2f}{b:>7.2f}{struct:>9.1f}{nonwall:>10.2f}{ncon:>10}")
            strip.append(img)
            labels.append((z, b, struct, nonwall))
            env.close()
        rows.append(np.concatenate(strip, axis=1))

    TT._ROBOT_POS_OFFSETS["AllexRobot"] = orig
    out = HERE.parent / "rollout" / "frames" / "mount_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out, np.concatenate(rows, axis=0))
    print("-" * 74)
    print(f"grid rows=heights {HEIGHTS} (top->bottom), cols=backs {BACKS} (left->right)")
    print(f"-> {out}")
    best = max(labels, key=lambda r: r[2])
    print(f"\nmost structured view: height={best[0]} back={best[1]} (struct={best[2]:.1f})")


if __name__ == "__main__":
    main()
