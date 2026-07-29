#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Joint mount sweep: hand-table CONTACT vs REACH vs EGO VIEW.

Three constraints fight each other and were tuned one at a time, so each fix
broke another:
  * too high  -> ego camera sees only wall (policy input useless)
  * too far back -> can/bowl beyond arm reach (task impossible)
  * too low/close -> hands press into the table at the resting pose, and the
    underdamped servos buzz (the visible start-of-scene shake)

This scores all three at once over (height x back) so a mount can be chosen that
satisfies every constraint instead of trading one for another.

Columns:
  contacts   robot-vs-table contacts at the resting pose (want ~0)
  pen        deepest penetration in mm (want ~0; negative dist = pressing in)
  buzz       robot max|qvel| after settling, rad/s (want <0.05)
  palm→can   best FK approach of the right palm to the can (want <0.12)
  palm→bowl  same for the bowl
  ego        ego-view pixel std (blank wall ~9, framed workspace ~55+)

Run (robosuite venv):
  MUJOCO_GL=egl <sim-venv-python> checks/mount_tradeoff.py
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
CAM = "robot0_zed_left_camera_optical_frame"
HEIGHTS = [0.55, 0.62, 0.70]
BACKS = [0.20, 0.30]


def main() -> None:
    import gymnasium as gym
    import mujoco
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
    from robocasa.environments.tabletop import tabletop as TT

    stats = json.load(open(HERE.parent / "contract" / "allex_stats.json"))["general_embodiment"]["state"]
    mean = {g: np.asarray(stats[g]["mean"], dtype=np.float64) for g in C.GROUP_ORDER}
    orig = list(TT._ROBOT_POS_OFFSETS["AllexRobot"])

    print("=" * 92)
    print("ALLEX mount trade-off: contact vs reach vs ego view")
    print("=" * 92)
    print(f"{'z':>6}{'back':>6}{'contacts':>10}{'pen(mm)':>9}{'buzz':>8}"
          f"{'palm→can':>10}{'palm→bowl':>11}{'ego':>7}  verdict")
    print("-" * 92)

    best = None
    for z, b in itertools.product(HEIGHTS, BACKS):
        TT._ROBOT_POS_OFFSETS["AllexRobot"] = [orig[0], -abs(b), z]
        env = gym.make(ENV_ID, enable_render=True, seed=0)
        env.reset()
        base = env.unwrapped.env
        model, data = base.sim.model._model, base.sim.data._data
        pf = base.robots[0].robot_model.naming_prefix
        for _ in range(120):
            env.step({f"action.{g}": mean[g] for g in C.GROUP_ORDER})

        # contacts + deepest penetration involving the robot
        ncon, pen = 0, 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
            if g1.startswith(pf) or g2.startswith(pf):
                ncon += 1
                pen = min(pen, float(con.dist))

        dofs = []
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if nm.startswith(pf):
                dofs.append(int(model.jnt_dofadr[j]))
        buzz = float(np.abs(data.qvel[dofs]).max())

        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        rp = bid(f"{pf}R_Palm_Link")
        can, bowl = data.xpos[bid("obj_main")].copy(), data.xpos[bid("container_main")].copy()

        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{pf}{j}")
                for j in C.JOINT_NAMES["right_arm_joints"]]
        adrs = [int(model.jnt_qposadr[j]) for j in jids]
        grid = [np.linspace(*model.jnt_range[j], 7) for j in jids[:5]]
        bc, bb = 1e9, 1e9
        for combo in itertools.product(*grid):
            for adr, v in zip(adrs[:5], combo):
                data.qpos[adr] = v
            mujoco.mj_forward(model, data)
            p = data.xpos[rp]
            bc = min(bc, float(np.linalg.norm(p - can)))
            bb = min(bb, float(np.linalg.norm(p - bowl)))

        env.reset()
        for _ in range(60):
            obs, *_ = env.step({f"action.{g}": mean[g] for g in C.GROUP_ORDER})
        ego = float(np.asarray(obs["video.camera_ego_left"]).astype(np.float32).mean(-1).std())

        good = ncon <= 5 and bc < 0.12 and bb < 0.12 and ego > 40
        verdict = "OK" if good else ",".join(
            x for x in [("contact" if ncon > 5 else ""), ("reach" if bc >= 0.12 or bb >= 0.12 else ""),
                        ("view" if ego <= 40 else "")] if x)
        print(f"{z:>6.2f}{b:>6.2f}{ncon:>10}{pen*1000:>9.1f}{buzz:>8.3f}"
              f"{bc:>10.3f}{bb:>11.3f}{ego:>7.1f}  {verdict}")
        if good and best is None:
            best = (z, b)
        env.close()

    TT._ROBOT_POS_OFFSETS["AllexRobot"] = orig
    print("-" * 92)
    print(f"first mount satisfying all three: {best}")


if __name__ == "__main__":
    main()
