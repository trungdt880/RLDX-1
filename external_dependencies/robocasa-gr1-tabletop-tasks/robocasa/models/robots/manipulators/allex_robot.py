# SPDX-License-Identifier: Apache-2.0
"""ALLEX humanoid for robosuite/robocasa — Phase 1 (steps 1-4).

Design (staff-reviewed): ALLEX hands are modeled AS HANDS (not grippers). ONE
gripperless robosuite robot holding the full 48-joint chain, driven by ABSOLUTE
joint-position targets (radians) written straight into ALLEX's own <position>
actuators (per-joint kp/kv + actuatorgravcomp). NO IK, NO computed-torque.

This single module provides the four coupled pieces so registration is atomic:
  1. ``AllexRobot``                   — the ManipulatorModel (arms=[], fixed base)
  2. ``JointPositionPassthroughController`` — a part controller that returns the
     absolute goal qpos (NOT torques) for its part's actuators
  3. ``AllexCompositeController``     — a BASIC-style composite that instantiates
     the passthrough controller for each of the 6 contract parts
  4. ``AllexPositionRobot``           — the Robot wrapper that wires the 6 parts'
     joint/actuator references and drives sim.data.ctrl

The 6 parts + intra-group joint ordering mirror the FROZEN contract in
``run_scripts/eval/robocasa_allex/contract/allex_contract.py`` (kept in-sync by
``robot_make_check.py``, which asserts equality). They are embedded here so the
fork has no dependency on run_scripts.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np

from robosuite.controllers.composite.composite_controller import (
    CompositeController,
    register_composite_controller,
)
from robosuite.controllers.parts.controller import Controller
from robosuite.controllers import composite_controller_factory
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.models.robots.robot_model import RobotModel
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.fixed_base_robot import FixedBaseRobot

# --------------------------------------------------------------------------- #
# Frozen contract (mirror of allex_contract.GROUP_ORDER / JOINT_NAMES)
# --------------------------------------------------------------------------- #
ALLEX_PART_ORDER = [
    "left_arm_joints",   # 7
    "left_hand_joints",  # 15
    "neck_joints",       # 2
    "right_arm_joints",  # 7
    "right_hand_joints", # 15
    "waist_joints",      # 2
]
ALLEX_JOINT_NAMES = {
    "left_arm_joints": [
        "L_Shoulder_Pitch_Joint", "L_Shoulder_Roll_Joint", "L_Shoulder_Yaw_Joint",
        "L_Elbow_Joint", "L_Wrist_Yaw_Joint", "L_Wrist_Roll_Joint", "L_Wrist_Pitch_Joint",
    ],
    "left_hand_joints": [
        "L_Thumb_Yaw_Joint", "L_Thumb_CMC_Joint", "L_Thumb_MCP_Joint",
        "L_Index_ABAD_Joint", "L_Index_MCP_Joint", "L_Index_PIP_Joint",
        "L_Middle_ABAD_Joint", "L_Middle_MCP_Joint", "L_Middle_PIP_Joint",
        "L_Ring_ABAD_Joint", "L_Ring_MCP_Joint", "L_Ring_PIP_Joint",
        "L_Little_ABAD_Joint", "L_Little_MCP_Joint", "L_Little_PIP_Joint",
    ],
    "neck_joints": ["Neck_Pitch_Joint", "Neck_Yaw_Joint"],
    "right_arm_joints": [
        "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
        "R_Elbow_Joint", "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
    ],
    "right_hand_joints": [
        "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
        "R_Index_ABAD_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint",
        "R_Middle_ABAD_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
        "R_Ring_ABAD_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
        "R_Little_ABAD_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint",
    ],
    "waist_joints": ["Waist_Yaw_Joint", "Waist_Lower_Pitch_Joint"],
}

# Absolute path to the fork's ALLEX robot.xml (adapted for robosuite in Phase 1.x).
ALLEX_XML = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "assets", "robots", "allex", "robot.xml",
    )
)


def _joint_to_actuator(joint_name: str) -> str:
    """ALLEX convention: joint ``X_Joint`` is driven by <position> actuator ``X_Position``."""
    assert joint_name.endswith("_Joint"), joint_name
    return joint_name[: -len("_Joint")] + "_Position"


# --------------------------------------------------------------------------- #
# 1. Robot model
# --------------------------------------------------------------------------- #
class AllexRobot(ManipulatorModel):
    """ALLEX 48-DoF fixed-base humanoid, modeled with hands (no grippers).

    ``arms = []`` neutralizes robosuite's arm/eef/gripper assumptions. We also
    bypass ``ManipulatorModel.__init__`` (which does an eef-body lookup that
    assumes single/bimanual arms) by calling the grandparent ``RobotModel``
    initializer directly and re-creating the minimal ManipulatorModel state.
    """

    arms = []

    def __init__(self, idn=0):
        # Grandparent init: parses XML, sets frictionloss/damping/armature over dof.
        RobotModel.__init__(self, ALLEX_XML, idn=idn)

        # Minimal ManipulatorModel state (skip the eef-body lookup it does).
        self.grippers = OrderedDict()
        self.hand_rotation_offset = {}
        self.cameras = self.get_element_names(self.worldbody, "camera")
        self._base_actuators = []
        self._torso_actuators = []
        self._head_actuators = []
        self._legs_actuators = []
        self._arms_actuators = []
        self._base_joints = []
        self._torso_joints = []
        self._head_joints = []
        self._legs_joints = []
        self._arms_joints = []

    # -- required RobotModel/ManipulatorModel properties --------------------- #
    @property
    def default_base(self):
        return "NoActuationBase"

    @property
    def default_gripper(self):
        return {}

    @property
    def default_controller_config(self):
        # Controller config is always supplied explicitly (controller_configs=...),
        # so this is only a fallback and is not exercised in the normal path.
        return {}

    @property
    def init_qpos(self):
        # One entry per model joint (60 = 48 actuated + 12 coupled/passive).
        # Zeros = neutral; equality couplings resolve at forward().
        return np.zeros(len(self._joints))

    @property
    def base_xpos_offset(self):
        # NOTE: EmptyArena floor is at z=0 and ALLEX's Base_Link is ~0.685 m above
        # its feet, so the robot must be mounted elevated to avoid floor
        # penetration shoving the arms. z≈0.9 clears the floor; tune per arena.
        return {
            "bins": (-0.5, -0.1, 0.9),
            "empty": (-0.6, 0.0, 0.9),
            "table": lambda table_length: (-0.15 - table_length / 2, 0.0, 0.9),
        }

    @property
    def top_offset(self):
        return np.array((0.0, 0.0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        # Not used (arms=[]); provided so the abstract property is concrete.
        return "single"

    @property
    def _eef_name(self):
        return {}

    @property
    def _important_sites(self):
        return {}

    @property
    def _important_geoms(self):
        return {}

    @property
    def _important_sensors(self):
        return {}


# --------------------------------------------------------------------------- #
# 2. Passthrough part controller (absolute joint-position setpoints)
# --------------------------------------------------------------------------- #
class JointPositionPassthroughController(Controller):
    """Writes absolute joint-position targets straight to <position> actuators.

    This is deliberately NOT the stock ``JointPositionController`` (which is a
    computed-TORQUE controller and would write torques into ALLEX's radian
    position setpoints -> garbage). ``run_controller`` returns the goal qpos; the
    Robot wrapper writes it (clipped to ctrlrange) into ``sim.data.ctrl``.
    """

    def __init__(
        self,
        sim,
        joint_indexes,
        actuator_range,
        ref_name=None,
        input_type="absolute",
        part_name=None,
        naming_prefix=None,
        lite_physics=True,
        policy_freq=20,
        **kwargs,  # sink JSON extras (type, input_max, ramp_ratio, ndim, ...)
    ):
        super().__init__(
            sim,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            ref_name=ref_name,
            part_name=part_name,
            naming_prefix=naming_prefix,
            lite_physics=lite_physics,
        )
        assert input_type in ("absolute", "delta"), input_type
        self.input_type = input_type
        self.control_dim = len(joint_indexes["joints"])
        self.control_freq = policy_freq
        # Action space == actuator ctrlrange (absolute radians). actuator_min/max
        # come from the base Controller (== ctrlrange low/high of the part's actuators).
        self.input_min = self.actuator_min
        self.input_max = self.actuator_max
        self.output_min = self.actuator_min
        self.output_max = self.actuator_max
        self.goal_qpos = None

    def set_goal(self, action, set_qpos=None):
        self.update()
        action = np.asarray(action, dtype=np.float64).flatten()
        assert action.shape[0] == self.control_dim, (
            f"{self.part_name}: expected {self.control_dim} targets, got {action.shape[0]}"
        )
        if set_qpos is not None:
            goal = np.asarray(set_qpos, dtype=np.float64).flatten()
        elif self.input_type == "absolute":
            goal = action
        else:  # delta
            goal = np.asarray(self.joint_pos, dtype=np.float64) + action
        self.goal_qpos = np.clip(goal, self.actuator_min, self.actuator_max)

    def run_controller(self):
        if self.goal_qpos is None:
            self.goal_qpos = np.array(self.joint_pos, dtype=np.float64)
        self.update()
        super().run_controller()  # resets new_update flag
        return np.array(self.goal_qpos, dtype=np.float64)

    def reset_goal(self):
        self.goal_qpos = np.array(self.joint_pos, dtype=np.float64)

    @property
    def control_limits(self):
        return self.actuator_min, self.actuator_max

    @property
    def name(self):
        return "JOINT_POSITION_PASSTHROUGH"


# --------------------------------------------------------------------------- #
# 3. Composite controller (BASIC-style, 6 passthrough parts, no arm/eef tracking)
# --------------------------------------------------------------------------- #
@register_composite_controller
class AllexCompositeController(CompositeController):
    name = "ALLEX_JOINT_POSITION"

    def _init_controllers(self):
        # Bypass robosuite's controller_factory (which dispatches by known part
        # names like "right"/"legs" and would reject our contract part names).
        for part_name, params in self.part_controller_config.items():
            self.part_controllers[part_name] = JointPositionPassthroughController(**params)

    def update_state(self):
        # No arm/eef "*_center" sites to track (BASIC would loop over self.arms).
        return

    @property
    def action_limits(self):
        low, high = [], []
        for _, controller in self.part_controllers.items():
            lo, hi = controller.control_limits
            low = np.concatenate([low, lo])
            high = np.concatenate([high, hi])
        return low, high


# --------------------------------------------------------------------------- #
# 4. Robot wrapper (fixed base; wires the 6 parts and drives sim.data.ctrl)
# --------------------------------------------------------------------------- #
class AllexPositionRobot(FixedBaseRobot):
    """Robot wrapper that sets up joint/actuator references for the 6 ALLEX
    contract parts and drives absolute position setpoints through the composite
    passthrough controller."""

    def setup_references(self):
        # Base FixedBaseRobot.setup_references handles the (empty) arm loop and
        # generic joint/actuator refs; then we add our 6 parts.
        super().setup_references()
        pf = self.robot_model.naming_prefix
        model = self.sim.model
        for part_name in ALLEX_PART_ORDER:
            joints = [pf + n for n in ALLEX_JOINT_NAMES[part_name]]
            actuators = [pf + _joint_to_actuator(n) for n in ALLEX_JOINT_NAMES[part_name]]
            self._ref_joints_indexes_dict[part_name] = [model.joint_name2id(j) for j in joints]
            self._ref_actuators_indexes_dict[part_name] = [model.actuator_name2id(a) for a in actuators]

    def _load_controller(self):
        self.composite_controller = composite_controller_factory(
            type=self.composite_controller_config.get("type", "ALLEX_JOINT_POSITION"),
            sim=self.sim,
            robot_model=self.robot_model,
            grippers={},
        )
        pf = self.robot_model.naming_prefix
        model = self.sim.model
        for part_name in ALLEX_PART_ORDER:
            cfg = self.part_controller_config.setdefault(part_name, {})
            joints = [pf + n for n in ALLEX_JOINT_NAMES[part_name]]
            act_ids = self._ref_actuators_indexes_dict[part_name]
            cfg["robot_name"] = self.name
            cfg["sim"] = self.sim
            cfg["part_name"] = part_name
            cfg["naming_prefix"] = pf
            cfg["ref_name"] = None
            cfg["ndim"] = len(joints)
            cfg["policy_freq"] = self.control_freq
            cfg["lite_physics"] = self.lite_physics
            cfg.setdefault("input_type", "absolute")
            cfg["joint_indexes"] = {
                "joints": [model.joint_name2id(j) for j in joints],
                "qpos": [model.get_joint_qpos_addr(j) for j in joints],
                "qvel": [model.get_joint_qvel_addr(j) for j in joints],
            }
            low = model.actuator_ctrlrange[act_ids, 0]
            high = model.actuator_ctrlrange[act_ids, 1]
            cfg["actuator_range"] = (low, high)

        self.composite_controller.load_controller_config(
            self.part_controller_config,
            self.composite_controller_config.get("composite_controller_specific_configs", {}),
        )
        self.enable_parts()

    def enable_parts(self):
        self._enabled_parts = {part_name: True for part_name in ALLEX_PART_ORDER}

    def control(self, action, policy_step=False):
        assert len(action) == self.action_dim, (
            f"invalid action dim -- expected {self.action_dim}, got {len(action)}"
        )
        self.composite_controller.update_state()
        if policy_step:
            self.composite_controller.set_goal(action)

        applied_action_dict = self.composite_controller.run_controller(self._enabled_parts)
        for part_name, applied_action in applied_action_dict.items():
            idxs = self._ref_actuators_indexes_dict[part_name]
            low = self.sim.model.actuator_ctrlrange[idxs, 0]
            high = self.sim.model.actuator_ctrlrange[idxs, 1]
            self.sim.data.ctrl[idxs] = np.clip(applied_action, low, high)

        if policy_step:
            self.recent_qpos.push(self._joint_positions)
            self.recent_actions.push(action)
            self.recent_torques.push(np.zeros(len(self.joint_indexes)))


# --------------------------------------------------------------------------- #
# Registration: map the model type-string "AllexRobot" -> our Robot wrapper.
# (register_robot_class only maps to the built-in wrappers, so we register the
# custom wrapper directly.)
# --------------------------------------------------------------------------- #
ROBOT_CLASS_MAPPING["AllexRobot"] = AllexPositionRobot
