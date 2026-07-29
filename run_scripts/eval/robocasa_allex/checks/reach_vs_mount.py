#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""How far back can ALLEX mount and still REACH the can?

reachability_check.py showed that at the current mount (back 0.40 m) the can sits
~3 cm beyond the right arm's full extension and the bowl ~10 cm beyond -- i.e. the
pick-and-place task is not physically achievable, independent of the policy. The
0.40 m back-offset was originally chosen to stop the hands hitting the table
during gesture replays; it directly costs reach.

Sweeps the back-offset and reports, per mount:
  * shoulder->can and shoulder->bowl distance,
  * the closest the palm can actually get (brute-force over shoulder/elbow),
  * hand-table contact count at the resting pose (the reason for the offset).

Pick the largest offset that still reaches both objects with contacts near zero.

Run (robosuite venv):
  MUJOCO_GL=egl <sim-venv-python> checks/reach_vs_mount.py
"""
from __future__ import annotations

import itertools
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
BACKS = [0.40, 0.30, 0.20, 0.10, 0.0]
GRASP_OK = 0.12   # palm-origin to object; palm origin sits behind the fingers


def main() -> None:
    import gymnasium as gym
    import mujoco
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    from robocasa.environments.tabletop import tabletop as TT

    stats = json.load(open(HERE.parent / "contract" / "allex_stats.json"))["general_embodiment"]["state"]
    mean = {g: np.asarray(stats[g]["mean"], dtype=np.float64) for g in C.GROUP_ORDER}
    orig = list(TT._ROBOT_POS_OFFSETS["AllexRobot"])

    print("=" * 86)
    print(f"ALLEX reach vs mount back-offset (height z={orig[2]})")
    print("=" * 86)
    print(f"{'back':>6}{'d(can)':>9}{'d(bowl)':>9}{'palm→can':>11}{'palm→bowl':>11}"
          f"{'contacts':>10}  verdict")
    print("-" * 86)

    for b in BACKS:
        TT._ROBOT_POS_OFFSETS["AllexRobot"] = [orig[0], -abs(b), orig[2]]
        env = gym.make(ENV_ID, enable_render=False, seed=0)
        env.reset()
        base = env.unwrapped.env
        model, data = base.sim.model._model, base.sim.data._data
        pf = base.robots[0].robot_model.naming_prefix

        def hold(pose, n=40):
            for _ in range(n):
                env.step({f"action.{g}": pose[g] for g in C.GROUP_ORDER})

        hold(mean, 50)
        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        rsh, rp = bid(f"{pf}R_Shoulder_Pitch_Link"), bid(f"{pf}R_Palm_Link")
        if rsh < 0:
            rsh = bid(f"{pf}R_Upper_Arm_Link")
        can = data.xpos[bid("obj_main")].copy()
        bowl = data.xpos[bid("container_main")].copy()
        ncon = int(data.ncon)
        d_can = float(np.linalg.norm(can - data.xpos[rsh]))
        d_bowl = float(np.linalg.norm(bowl - data.xpos[rsh]))

        # Pure FORWARD KINEMATICS over the arm joint grid: set qpos + mj_forward.
        # Stepping the full physics per sample is ~1000x slower and adds nothing —
        # reachability is a kinematic question.
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{pf}{j}")
                for j in C.JOINT_NAMES["right_arm_joints"]]
        adrs = [int(model.jnt_qposadr[j]) for j in jids]
        lims = [model.jnt_range[j] for j in jids[:5]]
        grid = [np.linspace(lo, hi, 7) for lo, hi in lims]
        best_can, best_bowl = 1e9, 1e9
        for combo in itertools.product(*grid):
            for adr, v in zip(adrs[:5], combo):
                data.qpos[adr] = v
            mujoco.mj_forward(model, data)
            p = data.xpos[rp]
            best_can = min(best_can, float(np.linalg.norm(p - can)))
            best_bowl = min(best_bowl, float(np.linalg.norm(p - bowl)))

        ok = best_can < GRASP_OK and best_bowl < GRASP_OK
        verdict = ("BOTH reachable" if ok else
                   "can only" if best_can < GRASP_OK else "NEITHER")
        print(f"{b:>6.2f}{d_can:>9.3f}{d_bowl:>9.3f}{best_can:>11.3f}{best_bowl:>11.3f}"
              f"{ncon:>10}  {verdict}")
        env.close()

    TT._ROBOT_POS_OFFSETS["AllexRobot"] = orig
    print("-" * 86)
    print(f"reachable threshold: palm-origin within {GRASP_OK} m of the object")
    print("contacts = hand/arm-vs-table contacts at the resting pose (want ~0)")


if __name__ == "__main__":
    main()
