#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Can ALLEX physically REACH the task objects, or is the scene set up unfairly?

No policy involved. Before blaming a model for not picking the can, confirm the
can is inside the arm's kinematic workspace at the current mount:

  1. measure each arm's true max reach = shoulder->palm distance with the arm
     driven to full extension (respecting joint limits, via the real controller);
  2. locate the task objects (can, bowl, ...) and the shoulders;
  3. compare object distance vs max reach, and report the margin;
  4. brute-force search the shoulder/elbow joint space for the closest the palm
     can actually get to the can -- the decisive number, since max reach alone
     ignores direction and joint limits.

Run (robosuite venv):
  MUJOCO_GL=egl <sim-venv-python> checks/reachability_check.py
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


def main() -> None:
    import gymnasium as gym
    import mujoco
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    stats = json.load(open(HERE.parent / "contract" / "allex_stats.json"))["general_embodiment"]["state"]
    mean = {g: np.asarray(stats[g]["mean"], dtype=np.float64) for g in C.GROUP_ORDER}

    env = gym.make(ENV_ID, enable_render=False, seed=0)
    env.reset()
    base = env.unwrapped.env
    model, data = base.sim.model._model, base.sim.data._data
    pf = base.robots[0].robot_model.naming_prefix

    def hold(pose, n=40):
        for _ in range(n):
            env.step({f"action.{g}": pose[g] for g in C.GROUP_ORDER})

    hold(mean)

    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    lp, rp = bid(f"{pf}L_Palm_Link"), bid(f"{pf}R_Palm_Link")
    lsh, rsh = bid(f"{pf}L_Shoulder_Pitch_Link"), bid(f"{pf}R_Shoulder_Pitch_Link")
    if lsh < 0:
        lsh, rsh = bid(f"{pf}L_Upper_Arm_Link"), bid(f"{pf}R_Upper_Arm_Link")

    print("=" * 74)
    print("ALLEX reachability — is the can physically within arm's reach?")
    print("=" * 74)
    print(f"  L_shoulder {np.round(data.xpos[lsh], 3)}   R_shoulder {np.round(data.xpos[rsh], 3)}")
    print(f"  L_palm     {np.round(data.xpos[lp], 3)}   R_palm     {np.round(data.xpos[rp], 3)}")

    # ---------------------------------------------------- task objects
    print("\n  task objects (free-jointed bodies in the scene):")
    objs = {}
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if nm.startswith(("robot0", "mobilebase", "world", "table", "wall", "floor")):
            continue
        if model.body_jntnum[b] > 0 and model.body_dofnum[b] >= 6:
            objs[nm] = data.xpos[b].copy()
    for nm, p in sorted(objs.items()):
        dl = np.linalg.norm(p - data.xpos[lsh])
        dr = np.linalg.norm(p - data.xpos[rsh])
        print(f"    {nm:<28} pos={np.round(p, 3)}  d(Lsh)={dl:.3f}  d(Rsh)={dr:.3f}")

    # ---------------------------------------------------- max reach
    print("\n  measuring true max reach (arm driven to full extension):")
    reach = {}
    for side, sh, pl, grp in (("L", lsh, lp, "left_arm_joints"),
                              ("R", rsh, rp, "right_arm_joints")):
        best = 0.0
        for sp in (-1.2, -0.6, 0.0, 0.6):
            pose = {g: mean[g].copy() for g in C.GROUP_ORDER}
            a = pose[grp]
            a[0] = sp        # shoulder pitch
            a[3] = 0.0       # elbow straight (index 3 == Elbow in frozen order)
            pose[grp] = a
            hold(pose, 60)
            best = max(best, float(np.linalg.norm(data.xpos[pl] - data.xpos[sh])))
        reach[side] = best
        print(f"    {side} arm max shoulder->palm distance: {best:.3f} m")

    # ---------------------------------------------------- closest approach
    # robocasa convention: 'obj_main' is the manipulated object (the can),
    # 'container_main' is the target receptacle (the bowl). Do NOT fuzzy-match
    # "can" -- it also matches nothing here and silently falls back to the bowl.
    can = objs.get("obj_main")
    if can is None and objs:
        can = list(objs.values())[0]
    bowl = objs.get("container_main")
    if bowl is not None:
        print(f"\n  target can  (obj_main)      = {np.round(can, 3)}")
        print(f"  target bowl (container_main) = {np.round(bowl, 3)}")
    if can is not None:
        print(f"\n  brute-force closest approach to the can at {np.round(can, 3)}:")
        for side, sh, pl, grp in (("L", lsh, lp, "left_arm_joints"),
                                  ("R", rsh, rp, "right_arm_joints")):
            jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{pf}{j}")
                    for j in C.JOINT_NAMES[grp]]
            lims = [model.jnt_range[j] for j in jids]
            best, bestq = 1e9, None
            grid = [np.linspace(lo, hi, 5) for lo, hi in [lims[0], lims[1], lims[2], lims[3]]]
            for combo in itertools.product(*grid):
                pose = {g: mean[g].copy() for g in C.GROUP_ORDER}
                a = pose[grp]
                a[0], a[1], a[2], a[3] = combo
                pose[grp] = a
                hold(pose, 12)
                d = float(np.linalg.norm(data.xpos[pl] - can))
                if d < best:
                    best, bestq = d, combo
            d_sh = float(np.linalg.norm(can - data.xpos[sh]))
            verdict = ("REACHABLE" if best < 0.12 else
                       "MARGINAL" if best < 0.25 else "OUT OF REACH")
            print(f"    {side} arm: closest palm-to-can = {best:.3f} m   "
                  f"(shoulder->can {d_sh:.3f} m vs max reach {reach[side]:.3f} m)  -> {verdict}")
            print(f"        best shoulder/elbow = {np.round(bestq, 2)}")

    env.close()
    print("\n  NOTE: palm-to-can distance, not a grasp test; a few cm is normal since")
    print("  the palm origin sits behind the fingers.")


if __name__ == "__main__":
    main()
