#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Phase 1.4 — does the ALLEX robot.xml survive robosuite's merge + prefix, and
still track absolute joint targets?  (bypasses the Robot/controller wrapper)

Isolates two unknowns that a full robosuite.make would conflate:
  (a) merged + "robot0_" prefixed model keeps all 48 <position> actuators, 60
      joints, 12 <equality> couplings, and still position-tracks (== 0.5c inside
      robosuite's arena), from
  (b) whether the custom AllexRobot/CompositeController wrapper works (later).

Run in the robosuite venv:
  .../robocasa_uv/.venv/bin/python run_scripts/eval/robocasa_allex/checks/robosuite_merge_check.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
from robosuite.models import MujocoWorldBase  # noqa: E402
from robosuite.models.arenas import EmptyArena  # noqa: E402
from robosuite.models.base import MujocoXMLModel  # noqa: E402

ROBOT_XML = Path("/home/thor/RLDX-1/external_dependencies/robocasa-gr1-tabletop-tasks/"
                 "robocasa/models/assets/robots/allex/robot.xml")
MOTION = Path.home() / "workspace/allex_model/examples/motions/hello.npz"
PREFIX = "robot0_"


class AllexXMLModel(MujocoXMLModel):
    """Minimal wrapper so robosuite runs add_prefix (rewrites <equality> refs)."""

    def __init__(self, idn=0):
        super().__init__(str(ROBOT_XML), idn=idn)

    @property
    def naming_prefix(self):
        return PREFIX

    @property
    def _important_sites(self):
        return {}

    @property
    def _important_geoms(self):
        return {}

    @property
    def _important_sensors(self):
        return {}

    @property
    def contact_geoms(self):
        return []

    @property
    def contact_geom_rgba(self):
        return np.array([0.0, 0.5, 0.5, 1.0])

    @property
    def bottom_offset(self):
        return np.zeros(3)

    @property
    def top_offset(self):
        return np.zeros(3)

    @property
    def horizontal_radius(self):
        return 0.5


def main():
    world = MujocoWorldBase()
    world.merge(EmptyArena())
    robot = AllexXMLModel()
    world.merge(robot)
    model = world.get_model(mode="mujoco")
    data = mujoco.MjData(model)

    # ---- structural assertions (merge/prefix preserved the model) ----
    n_pos = sum(int(model.actuator_trntype[i]) == 0 for i in range(model.nu))
    print(f"nu={model.nu} (position={n_pos})  njnt={model.njnt}  neq={model.neq}")
    assert model.nu == 48, f"expected 48 actuators, got {model.nu}"
    assert model.neq == 12, f"expected 12 equality couplings, got {model.neq}"
    # every actuator prefixed + still a <position> actuator
    anames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert all(a.startswith(PREFIX) for a in anames), "actuators not prefixed"
    assert n_pos == 48, "some actuators are no longer <position>"
    print("[merge] 48 prefixed position actuators + 12 couplings preserved: OK")

    # ---- reproduce 0.5c tracking inside the merged model ----
    z = np.load(MOTION, allow_pickle=True)
    names = [str(x) for x in z["joint_names"]]
    q = z["q"].astype(np.float64)
    dt = float(z["dt"])
    # robosuite's arena <option> overrides ALLEX's; restore what ALLEX.xml needs
    # for its stiff position-PD + equality couplings (implicitfast, small dt).
    print(f"  [merged opt] integrator={int(model.opt.integrator)} "
          f"timestep={model.opt.timestep} (ALLEX.xml wants implicitfast)")
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.timestep = dt  # same as validated standalone 0.5c
    # DIAGNOSTIC: EmptyArena floor sits at z=0 but ALLEX mounts at origin (its own
    # scene puts the floor 0.685m below the base) -> the robot penetrates the floor
    # and contact forces shove the arms. Disable contact to isolate that from the
    # controller/model physics. (Real env will mount the robot at proper height.)
    if os.environ.get("ALLEX_NO_CONTACT", "1") == "1":
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        print("  [diag] contacts DISABLED (isolating floor-penetration effect)")

    # map npz joint name -> actuator id (prefixed actuator drives that joint)
    name2act = {}
    for i, an in enumerate(anames):
        jid = int(model.actuator_trnid[i, 0])
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid).replace(PREFIX, "")
        name2act[jn] = i
    qadr = {n: int(model.jnt_qposadr[model.actuator_trnid[name2act[n], 0]]) for n in names if n in name2act}
    missing = [n for n in names if n not in name2act]
    assert not missing, f"npz joints not driven by an actuator: {missing[:5]}"

    sd = mujoco.MjData(model)

    def gravity_comp():
        sd.qpos[:] = data.qpos
        sd.qvel[:] = 0.0
        mujoco.mj_forward(model, sd)
        return sd.qfrc_bias

    # ramp from home to first frame (don't teleport coupled joints)
    home = np.array([data.qpos[qadr[n]] for n in names])
    nwarm = max(1, int(round(0.5 / dt)))
    for k in range(nwarm):
        a = (k + 1) / nwarm
        tgt = home * (1 - a) + q[0] * a
        data.qfrc_applied[:] = gravity_comp()
        for c, n in enumerate(names):
            data.ctrl[name2act[n]] = tgt[c]
        mujoco.mj_step(model, data)

    errs = {n: [] for n in names}
    for i in range(len(q)):
        data.qfrc_applied[:] = gravity_comp()
        for c, n in enumerate(names):
            data.ctrl[name2act[n]] = q[i, c]
        mujoco.mj_step(model, data)
        for c, n in enumerate(names):
            errs[n].append(abs(data.qpos[qadr[n]] - q[i, c]))

    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), "NaN/inf in sim"
    deg = 180.0 / np.pi
    groups = {"arm": [], "hand": [], "neck": [], "waist": []}
    for n, e in errs.items():
        rms = float(np.sqrt(np.mean(np.square(e)))) * deg
        key = ("hand" if any(f in n for f in ("Thumb", "Index", "Middle", "Ring", "Little"))
               else "neck" if "Neck" in n else "waist" if "Waist" in n else "arm")
        groups[key].append(rms)
    print("  per-group tracking RMS (deg) inside merged model:")
    for g, v in groups.items():
        print(f"    {g:6s} mean={np.mean(v):.3f} max={np.max(v):.3f}")
    worst_arm = max(groups["arm"])
    verdict = "PASS" if worst_arm < 5.0 and model.neq == 12 else "FAIL"
    print(f"\nVERDICT [1.4 robosuite-merge]: {verdict} "
          f"(merged+prefixed model tracks; couplings intact)")


if __name__ == "__main__":
    main()
