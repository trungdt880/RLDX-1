#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Is the ALLEX ego camera occluded by its own head housing?

The rendered ego view shows a heavy black ring: the `zed_left_camera_optical_frame`
origin sits INSIDE the head/camera-housing meshes (`Neck_Yaw_Cam`, `Neck_Yaw_Head`),
so a large fraction of the model's only visual input is the inside of the robot's
own shell. Real ALLEX training frames are unobstructed.

Sweeps the camera forward along the head's +x and reports the fraction of
near-black pixels, so the smallest offset that clears the housing can be picked.

Run (robosuite venv):
  MUJOCO_GL=egl <sim-venv-python> checks/ego_camera_check.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

HERE = Path(__file__).resolve().parent
ENV_ID = os.environ.get("ALLEX_ENV_ID", "robocasa_allex/PnPCanToBowl_AllexRobot_Env")
CAM = "robot0_zed_left_camera_optical_frame"
OFFSETS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08]


def main() -> None:
    import gymnasium as gym
    import mujoco
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    import imageio.v2 as imageio

    env = gym.make(ENV_ID, enable_render=True, seed=0)
    env.reset()
    base = env.unwrapped.env
    sim = base.sim
    model = sim.model._model
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM)
    assert cid >= 0, f"camera {CAM} not found"
    base_pos = model.cam_pos[cid].copy()
    print(f"camera '{CAM}' id={cid} pos={base_pos} fovy={model.cam_fovy[cid]}")
    print(f"\n{'dx (m)':>8}  {'black %':>8}  {'mean lum':>9}")
    print("-" * 30)

    strips = []
    for dx in OFFSETS:
        model.cam_pos[cid] = base_pos + np.array([dx, 0.0, 0.0])
        sim.forward()
        img = sim.render(width=256, height=256, camera_name=CAM)[::-1]
        lum = img.astype(np.float32).mean(axis=-1)
        black = float((lum < 25).mean()) * 100.0
        print(f"{dx:>8.3f}  {black:>7.1f}%  {lum.mean():>9.1f}")
        strips.append(img)

    model.cam_pos[cid] = base_pos  # restore
    out = HERE.parent / "rollout" / "frames" / "ego_camera_offsets.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out, np.concatenate(strips, axis=1))
    print(f"\nstrip ({len(OFFSETS)} offsets, left->right {OFFSETS}) -> {out}")
    env.close()


if __name__ == "__main__":
    main()
