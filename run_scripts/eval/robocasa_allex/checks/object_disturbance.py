#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Does ALLEX knock the task objects around just by RESTING there?

Earlier mount sweeps counted robot-vs-TABLE contacts but never robot-vs-OBJECT
contacts -- so a mount that pushes the can/bowl at t=0 scored as fine. If the
objects drift or topple before the policy does anything, the episode is corrupt
before it starts (and the initial observation is not the scene the task means).

Per mount, with NO policy (robot just holding the training-mean pose):
  objΔ / bowlΔ   how far the can / bowl move during settling (want ~0)
  tipped         object tilted >20deg from upright (toppled)
  rob-obj        robot<->object contacts (want 0)
  rob-tbl/pen    robot<->table contacts and deepest penetration (mm)
  palm→can/bowl  best FK approach of the right palm (want <0.12)
  ego            ego-view pixel std (blank wall ~9, framed workspace ~55+)

Run (robosuite venv):
  MUJOCO_GL=egl <sim-venv-python> checks/object_disturbance.py
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
BACKS = [float(x) for x in os.environ.get("ALLEX_BACKS", "0.30,0.33,0.35,0.40").split(",")]
Z = float(os.environ.get("ALLEX_Z", "0.62"))
SEEDS = [0, 1, 2]


def tilt_deg(quat) -> float:
    """Angle between the body's local +z and world +z."""
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=np.float64))
    return float(np.degrees(np.arccos(np.clip(R.reshape(3, 3)[2, 2], -1.0, 1.0))))


def main() -> None:
    import gymnasium as gym
    import mujoco
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    from robocasa.environments.tabletop import tabletop as TT

    stats = json.load(open(HERE.parent / "contract" / "allex_stats.json"))["general_embodiment"]["state"]
    mean = {g: np.asarray(stats[g]["mean"], dtype=np.float64) for g in C.GROUP_ORDER}
    orig = list(TT._ROBOT_POS_OFFSETS["AllexRobot"])

    print("=" * 100)
    print(f"ALLEX resting-pose object disturbance (z={Z}, no policy, {len(SEEDS)} seeds)")
    print("=" * 100)
    print(f"{'back':>6}{'objΔ':>8}{'bowlΔ':>8}{'tipped':>8}{'rob-obj':>9}"
          f"{'rob-tbl':>9}{'pen(mm)':>9}{'palm→can':>10}{'palm→bowl':>11}{'ego':>7}  verdict")
    print("-" * 100)

    for b in BACKS:
        TT._ROBOT_POS_OFFSETS["AllexRobot"] = [orig[0], -abs(b), Z]
        dobj = dbowl = 0.0
        tipped = 0
        robobj = robtbl = 0
        pen = 0.0
        for sd in SEEDS:
            env = gym.make(ENV_ID, enable_render=(sd == 0), seed=sd)
            env.reset()
            base = env.unwrapped.env
            model, data = base.sim.model._model, base.sim.data._data
            pf = base.robots[0].robot_model.naming_prefix
            bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
            ob, cb = bid("obj_main"), bid("container_main")
            p0, q0 = data.xpos[ob].copy(), data.xpos[cb].copy()

            for _ in range(150):
                obs, *_ = env.step({f"action.{g}": mean[g] for g in C.GROUP_ORDER})

            dobj = max(dobj, float(np.linalg.norm(data.xpos[ob] - p0)))
            dbowl = max(dbowl, float(np.linalg.norm(data.xpos[cb] - q0)))
            tipped += int(tilt_deg(data.xquat[ob]) > 20.0 or tilt_deg(data.xquat[cb]) > 20.0)

            obj_bodies = {ob, cb}
            for c in range(data.ncon):
                con = data.contact[c]
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
                r1, r2 = g1.startswith(pf), g2.startswith(pf)
                if not (r1 or r2):
                    continue
                other = model.geom_bodyid[con.geom2 if r1 else con.geom1]
                # walk up to a root-ish body to test membership
                root = other
                while root != 0 and root not in obj_bodies:
                    root = model.body_parentid[root]
                if root in obj_bodies:
                    robobj += 1
                else:
                    robtbl += 1
                    pen = min(pen, float(con.dist))
            if sd == 0:
                ego = float(np.asarray(obs["video.camera_ego_left"]).astype(np.float32).mean(-1).std())
                # Reach must be measured to the whole RIGHT HAND, not the palm
                # origin: the fingers extend ~8-10 cm past it, so palm-origin
                # distance understates real grasp reach.
                hand = [b for b in range(model.nbody)
                        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                        .startswith(f"{pf}R_") and any(
                            k in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                            for k in ("Palm", "Finger", "Thumb"))]
                can, bowl = data.xpos[ob].copy(), data.xpos[cb].copy()
                jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{pf}{j}")
                        for j in C.JOINT_NAMES["right_arm_joints"]]
                adrs = [int(model.jnt_qposadr[j]) for j in jids]
                grid = [np.linspace(*model.jnt_range[j], 7) for j in jids[:5]]
                bc = bb = 1e9
                for combo in itertools.product(*grid):
                    for adr, v in zip(adrs[:5], combo):
                        data.qpos[adr] = v
                    mujoco.mj_forward(model, data)
                    hp = data.xpos[hand]
                    bc = min(bc, float(np.linalg.norm(hp - can, axis=1).min()))
                    bb = min(bb, float(np.linalg.norm(hp - bowl, axis=1).min()))
            env.close()

        clean = dobj < 0.005 and dbowl < 0.005 and robobj == 0 and tipped == 0
        reach = bc < 0.12 and bb < 0.12
        verdict = ("CLEAN + reachable" if clean and reach else
                   "clean, OUT OF REACH" if clean else
                   "DISTURBS OBJECTS" + ("" if reach else " + out of reach"))
        print(f"{b:>6.2f}{dobj:>8.4f}{dbowl:>8.4f}{tipped:>8}{robobj:>9}"
              f"{robtbl:>9}{pen*1000:>9.1f}{bc:>10.3f}{bb:>11.3f}{ego:>7.1f}  {verdict}")

    TT._ROBOT_POS_OFFSETS["AllexRobot"] = orig
    print("-" * 100)
    print("want: objΔ/bowlΔ ~0, tipped 0, rob-obj 0, palm→* < 0.12, ego > 40")


if __name__ == "__main__":
    main()
