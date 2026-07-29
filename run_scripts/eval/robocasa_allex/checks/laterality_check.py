#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Does `action.left_arm_joints` actually move the robot's LEFT arm?

Rules out a left/right swap in the ALLEX contract→robosuite wiring before any
laterality claim is made about the policy. Pure open-loop causality test, no
model involved:

  * start at the checkpoint's training-mean pose;
  * command ONLY ``action.left_arm_joints`` (shoulder pitch sweep), hold every
    other group fixed; record both palm positions;
  * repeat for ``action.right_arm_joints``.

PASS iff each group moves its OWN palm and leaves the other essentially still.
Also reports the world-frame Y of each palm so "left" is checked against the
robot's own frame, not just naming.

Run (robosuite venv):
  MUJOCO_GL=egl <sim-venv-python> checks/laterality_check.py
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


def main() -> None:
    import gymnasium as gym
    import mujoco
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    stats = json.load(open(HERE.parent / "contract" / "allex_stats.json"))
    mean = {g: np.asarray(stats["general_embodiment"]["state"][g]["mean"], dtype=np.float64)
            for g in C.GROUP_ORDER}

    env = gym.make(ENV_ID, enable_render=False, seed=0)
    env.reset()
    base = env.unwrapped.env

    def handles():
        """Re-acquire model/data/body-ids: env.reset() rebuilds the sim."""
        m, d = base.sim.model._model, base.sim.data._data
        pf = base.robots[0].robot_model.naming_prefix
        return (m, d,
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{pf}L_Palm_Link"),
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{pf}R_Palm_Link"))

    print("=" * 70)
    print("ALLEX laterality check — does action.left_* drive the LEFT arm?")
    print("=" * 70)

    def settle(pose, n=25):
        for _ in range(n):
            env.step({f"action.{g}": pose[g] for g in C.GROUP_ORDER})

    def sweep(group: str, delta: float):
        env.reset()
        _m, data, lp, rp = handles()          # re-acquire AFTER the rebuild
        settle(mean, 40)
        l0, r0 = data.xpos[lp].copy(), data.xpos[rp].copy()
        pose = {g: mean[g].copy() for g in C.GROUP_ORDER}
        # index 0 of an arm group == Shoulder_Pitch (frozen contract order)
        for k in range(30):
            pose[group] = mean[group].copy()
            pose[group][0] = mean[group][0] + delta * (k + 1) / 30.0
            env.step({f"action.{g_}": pose[g_] for g_ in C.GROUP_ORDER})
        settle(pose, 15)
        dl = data.xpos[lp] - l0
        dr = data.xpos[rp] - r0
        return l0, r0, dl, dr

    ok = True
    for group, other, palm_name in (("left_arm_joints", "right_arm_joints", "LEFT"),
                                    ("right_arm_joints", "left_arm_joints", "RIGHT")):
        l0, r0, dl, dr = sweep(group, -0.6)  # shoulder pitch -0.6 rad
        moved_l, moved_r = float(np.linalg.norm(dl)), float(np.linalg.norm(dr))
        own, oth = (moved_l, moved_r) if palm_name == "LEFT" else (moved_r, moved_l)
        print(f"\n  command ONLY action.{group} (shoulder pitch -0.6 rad):")
        print(f"    L_Palm start=({l0[0]:+.3f},{l0[1]:+.3f},{l0[2]:+.3f})  moved {moved_l:.4f} m  Δz={dl[2]:+.4f}")
        print(f"    R_Palm start=({r0[0]:+.3f},{r0[1]:+.3f},{r0[2]:+.3f})  moved {moved_r:.4f} m  Δz={dr[2]:+.4f}")
        good = own > 0.02 and own > 5 * max(oth, 1e-6)
        ok = ok and good
        print(f"    -> drives the {palm_name} palm: {'PASS' if good else 'FAIL'} "
              f"(own={own:.4f} m vs other={oth:.4f} m)")

    # where do the two palms sit relative to each other in world coords?
    env.reset()
    _m, data, lp, rp = handles()
    settle(mean, 40)
    print(f"\n  world-frame palms:  L_Palm=({data.xpos[lp][0]:+.3f},{data.xpos[lp][1]:+.3f},"
          f"{data.xpos[lp][2]:+.3f})  R_Palm=({data.xpos[rp][0]:+.3f},{data.xpos[rp][1]:+.3f},"
          f"{data.xpos[rp][2]:+.3f})")

    env.close()
    print("\n" + "-" * 70)
    print(f"VERDICT [laterality wiring]: {'PASS — no left/right swap' if ok else 'FAIL — swapped!'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
