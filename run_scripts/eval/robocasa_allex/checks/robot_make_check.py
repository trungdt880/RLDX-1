#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Phase 1.5 — bring-up test for the ALLEX robosuite robot wrapper.

Builds the robosuite ``AllexPositionRobot`` (fixed-base, gripperless, 48-joint
absolute position control) in an EMPTY arena and drives the 6 contract parts
with a REPLAYED npz motion THROUGH the composite passthrough controller
(robot.control -> AllexCompositeController -> JointPositionPassthroughController
-> sim.data.ctrl), i.e. it tests the full wrapper stack, not raw sim.data.ctrl.

We construct the Robot object directly (rather than robosuite.make) because
every shipped task env assumes arms/eef/grippers for reward+placement; the empty
bring-up exercises exactly the Phase 1 (steps 1-4) code with no task noise. This
is the "directly constructs the robosuite Robot/env" branch of the deliverable.

Asserts:
  * robot builds + resets without error
  * model.neq == 12 (equality couplings survive)
  * 6-part action space dims are 7/15/2/7/15/2 (== 48)
  * per-group joint tracking RMS small (arm/neck/waist < 5 deg, hand < 12 deg)

Run in the robosuite venv:
  .../robocasa_uv/.venv/bin/python run_scripts/eval/robocasa_allex/checks/robot_make_check.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import robocasa  # noqa: E402,F401  (triggers ALLEX registration)
from robosuite.controllers import load_composite_controller_config  # noqa: E402
from robosuite.models import MujocoWorldBase  # noqa: E402
from robosuite.models.arenas import EmptyArena  # noqa: E402
from robosuite.robots import ROBOT_CLASS_MAPPING  # noqa: E402
from robosuite.utils.binding_utils import MjSim  # noqa: E402

# import the FROZEN contract to cross-check the fork's embedded ordering
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))
import allex_contract as C  # noqa: E402

from robocasa.models.robots.manipulators.allex_robot import (  # noqa: E402
    ALLEX_JOINT_NAMES,
    ALLEX_PART_ORDER,
)

CONFIG = Path(
    "/home/thor/RLDX-1/external_dependencies/robocasa-gr1-tabletop-tasks/"
    "robocasa/models/assets/robots/allex/default_allex_position.json"
)
MOTION = Path.home() / "workspace/allex_model/examples/motions/hello.npz"
DEG = 180.0 / np.pi
GROUP_DIMS = [7, 15, 2, 7, 15, 2]


def _group_key(joint_name: str) -> str:
    if any(f in joint_name for f in ("Thumb", "Index", "Middle", "Ring", "Little")):
        return "hand"
    if "Neck" in joint_name:
        return "neck"
    if "Waist" in joint_name:
        return "waist"
    return "arm"


def main() -> None:
    # ---- 0. contract consistency: fork's embedded ordering == frozen contract ----
    assert ALLEX_PART_ORDER == C.GROUP_ORDER, (ALLEX_PART_ORDER, C.GROUP_ORDER)
    for g in C.GROUP_ORDER:
        assert ALLEX_JOINT_NAMES[g] == C.JOINT_NAMES[g], g
    print("[contract] fork embedded joint ordering == frozen allex_contract: OK")

    # ---- 1. build the robot object + model ----
    cc_config = load_composite_controller_config(controller=str(CONFIG))
    robot = ROBOT_CLASS_MAPPING["AllexRobot"](
        robot_type="AllexRobot",
        idn=0,
        composite_controller_config=cc_config,
        control_freq=20,
    )
    robot.load_model()
    print("[build] AllexPositionRobot.load_model(): OK "
          f"(wrapper={type(robot).__name__})")

    # ---- 2. merge into an empty arena + create sim (mirrors env._initialize_sim) ----
    world = MujocoWorldBase()
    world.merge(EmptyArena())
    world.merge(robot.robot_model)
    sim = MjSim.from_xml_string(world.get_xml())
    sim.forward()

    model = sim.model._model  # raw mujoco.MjModel
    assert model.neq == 12, f"expected 12 equality couplings, got {model.neq}"
    n_pos = sum(int(model.actuator_trntype[i]) == 0 for i in range(model.nu))
    print(f"[merge] nu={model.nu} (position={n_pos}) njnt={model.njnt} neq={model.neq}")
    assert model.nu == 48 and n_pos == 48, "expected 48 <position> actuators"

    # ---- 3. wire references + reset (this runs the wrapper _load_controller) ----
    robot.reset_sim(sim)
    robot.setup_references()
    robot.reset(deterministic=True)
    print("[reset] setup_references + reset (composite+passthrough loaded): OK")

    # ---- 3b. action-space dims from the composite controller ----
    split = robot.composite_controller._action_split_indexes
    part_dims = [split[p][1] - split[p][0] for p in ALLEX_PART_ORDER]
    print(f"[action-space] parts={list(split.keys())}")
    print(f"[action-space] dims={part_dims} total={robot.action_dim}")
    assert part_dims == GROUP_DIMS, f"part dims {part_dims} != {GROUP_DIMS}"
    assert robot.action_dim == 48, robot.action_dim

    # ---- 4. physics setup per Phase 1.4 gotchas ----
    z = np.load(MOTION, allow_pickle=True)
    names = [str(x) for x in z["joint_names"]]
    q = z["q"].astype(np.float64)
    dt = float(z["dt"])
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # stiff kp needs it
    model.opt.timestep = dt
    if os.environ.get("ALLEX_NO_CONTACT", "1") == "1":
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        print("  [diag] contacts DISABLED (robot mounts at origin over EmptyArena floor)")

    data = sim.data._data
    pf = robot.robot_model.naming_prefix
    qadr = {n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pf + n)])
            for n in names}

    # helper: build a full 48-vec absolute action for a target frame via the
    # composite controller's own action-vector assembler (keyed by part name).
    def action_for(frame: np.ndarray) -> np.ndarray:
        name2val = dict(zip(names, frame))
        action_dict = {
            part: np.array([name2val[j] for j in ALLEX_JOINT_NAMES[part]])
            for part in ALLEX_PART_ORDER
        }
        return robot.create_action_vector(action_dict)

    # ---- 5. ramp from home (zeros) to first frame, then replay ----
    nwarm = max(1, int(round(0.5 / dt)))
    home = np.array([data.qpos[qadr[n]] for n in names])
    for k in range(nwarm):
        a = (k + 1) / nwarm
        tgt = home * (1 - a) + q[0] * a
        robot.control(action_for(tgt), policy_step=True)
        sim.step()

    errs = {n: [] for n in names}
    for i in range(len(q)):
        robot.control(action_for(q[i]), policy_step=True)
        sim.step()
        for c, n in enumerate(names):
            errs[n].append(abs(data.qpos[qadr[n]] - q[i, c]))

    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), "NaN/inf in sim"

    groups = {"arm": [], "hand": [], "neck": [], "waist": []}
    for n, e in errs.items():
        rms = float(np.sqrt(np.mean(np.square(e)))) * DEG
        groups[_group_key(n)].append(rms)
    print("  per-group tracking RMS (deg) through the composite wrapper:")
    for g in ("arm", "hand", "neck", "waist"):
        v = groups[g]
        print(f"    {g:6s} mean={np.mean(v):7.4f} max={np.max(v):7.4f}  (n={len(v)})")

    ok_arm = max(groups["arm"]) < 5.0
    ok_neck = max(groups["neck"]) < 5.0
    ok_waist = max(groups["waist"]) < 5.0
    ok_hand = max(groups["hand"]) < 12.0
    ok_struct = (model.neq == 12) and (part_dims == GROUP_DIMS) and (robot.action_dim == 48)
    verdict = "PASS" if (ok_arm and ok_neck and ok_waist and ok_hand and ok_struct) else "FAIL"
    print(f"\n  gates: arm<5={ok_arm} neck<5={ok_neck} waist<5={ok_waist} "
          f"hand<12={ok_hand} struct={ok_struct}")
    print(f"VERDICT [1.5 robosuite robot bring-up]: {verdict}")
    if verdict != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
