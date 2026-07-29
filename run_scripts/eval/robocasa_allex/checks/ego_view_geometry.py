#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Where is the ALLEX ego camera actually looking, and can it see the workspace?

The rendered ego view is a blank wall, so the policy's only visual input carries
no task information. This measures the geometry behind that:

  * world position/height of the ego camera vs the counter surface and the
    robot's own palms;
  * the down-angle the head WOULD need to centre the workspace, vs the
    Neck_Pitch joint limit and the training distribution (mean 0.385 rad);
  * an ego render swept across the Neck_Pitch range (housing occlusion removed
    via the +1 cm camera offset from ego_camera_check.py) so the view can be
    inspected directly.

Run (robosuite venv):
  MUJOCO_GL=egl <sim-venv-python> checks/ego_view_geometry.py
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
CAM_DX = 0.01          # clears the head-housing occlusion
PITCHES = [-0.35, 0.0, 0.385, 0.6, 0.8726]   # last = joint upper limit


def main() -> None:
    import gymnasium as gym
    import mujoco
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    import imageio.v2 as imageio

    stats = json.load(open(HERE.parent / "contract" / "allex_stats.json"))["general_embodiment"]["state"]
    mean = {g: np.asarray(stats[g]["mean"], dtype=np.float64) for g in C.GROUP_ORDER}

    env = gym.make(ENV_ID, enable_render=True, seed=0)
    env.reset()
    base = env.unwrapped.env
    sim = base.sim
    model, data = sim.model._model, sim.data._data
    pf = base.robots[0].robot_model.naming_prefix
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM)
    model.cam_pos[cid] = model.cam_pos[cid] + np.array([CAM_DX, 0.0, 0.0])

    def hold(pose, n=40):
        for _ in range(n):
            env.step({f"action.{g}": pose[g] for g in C.GROUP_ORDER})

    pose = {g: mean[g].copy() for g in C.GROUP_ORDER}
    hold(pose)

    lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pf}L_Palm_Link")
    cam_world = data.cam_xpos[cid].copy()
    palm_world = data.xpos[lp].copy()

    # counter / table surface: highest table-ish geom under the robot
    surf_z, surf_name = None, None
    for gi in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if any(k in nm.lower() for k in ("counter", "table", "island")):
            top = float(data.geom_xpos[gi][2] + model.geom_size[gi][2])
            if surf_z is None or top > surf_z:
                surf_z, surf_name = top, nm

    print("=" * 68)
    print("ALLEX ego-camera geometry")
    print("=" * 68)
    print(f"  ego camera world pos : ({cam_world[0]:+.3f}, {cam_world[1]:+.3f}, {cam_world[2]:+.3f})")
    print(f"  left palm  world pos : ({palm_world[0]:+.3f}, {palm_world[1]:+.3f}, {palm_world[2]:+.3f})")
    if surf_z is not None:
        print(f"  work surface         : z={surf_z:.3f}  ({surf_name})")
        drop = cam_world[2] - surf_z
        print(f"  camera sits {drop:.3f} m ABOVE the work surface")
        for horiz in (0.3, 0.5, 0.7):
            ang = np.degrees(np.arctan2(drop, horiz))
            print(f"    to centre a point {horiz:.1f} m in front -> look down {ang:.0f}deg "
                  f"({np.radians(ang):.2f} rad)")
    print(f"  Neck_Pitch joint limit : [-0.698, +0.873] rad  (max {np.degrees(0.8726):.0f}deg)")
    print(f"  training Neck_Pitch    : mean 0.385 rad ({np.degrees(0.385):.0f}deg), "
          f"q99 0.726 ({np.degrees(0.726):.0f}deg)")

    print("\n  sweeping Neck_Pitch (ego render):")
    strips = []
    for p in PITCHES:
        pose["neck_joints"] = np.array([p, mean["neck_joints"][1]])
        hold(pose, 60)
        img = sim.render(width=256, height=256, camera_name=CAM)[::-1]
        lum = img.astype(np.float32).mean(axis=-1)
        print(f"    pitch {p:+.3f} rad ({np.degrees(p):+5.0f}deg): "
              f"mean_lum={lum.mean():6.1f}  std={lum.std():5.1f}  "
              f"(low std = featureless wall)")
        strips.append(img)

    out = HERE.parent / "rollout" / "frames" / "ego_neck_pitch_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out, np.concatenate(strips, axis=1))
    print(f"\n  strip (pitches {PITCHES}) -> {out}")
    env.close()


if __name__ == "__main__":
    main()
