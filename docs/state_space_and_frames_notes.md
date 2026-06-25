# RLDX-1 State Space, Rotation Conventions & Coordinate Frames

Reference notes covering: the policy input state space, absolute vs. relative,
rotation representations, per-simulation conventions, DROID specifics, and a
from-scratch explainer of coordinate frames.

---

## 1. Input state space — the big picture

- **State (proprioception) is ABSOLUTE.** Each timestep the policy receives a
  full snapshot of the robot pose (end-effector pose + gripper), not a change.
  The LIBERO config even names the keys `eef_pos_absolute`, `eef_rot_absolute`
  (`rldx/configs/data/libero_config.py:37-41`).
- **Action is mostly DELTA (relative); gripper is ABSOLUTE.** Each action
  component carries its own `ActionRepresentation` (`DELTA` / `RELATIVE` /
  `ABSOLUTE`). LIBERO/DROID emit `eef_pos_delta`, `eef_rot_delta` as `DELTA`
  and `gripper_close` as `ABSOLUTE`.
- **Layout differs per embodiment**, selected by an embodiment tag, then
  zero-padded to a fixed width so one network handles every robot.
- **Position is always `xyz`.** **Orientation representation varies** by
  benchmark: axis-angle, quaternion (wxyz or xyzw), or euler roll-pitch-yaw.

### Action horizon
Actions are predicted as a **16-step chunk** (`delta_indices=list(range(16))`).

---

## 2. Embodiment tags (per-robot layouts)

Each embodiment tag (~42 of them: `GENERAL_EMBODIMENT`, the `OXE_*` family,
`LIBERO_PANDA`, `GR1`, `UNITREE_G1`, `BEHAVIOR_R1_PRO`, `OXE_DROID`, …) maps to:

1. A `MODALITY_CONFIGS` entry defining the ordered `modality_keys`
   (`rldx/configs/data/embodiment_configs.py`).
2. An integer projector index (`EMBODIMENT_TAG_TO_PROJECTOR_INDEX`,
   `rldx/model/core/processing_rldx.py:30-78`) selecting a per-embodiment row
   in category-specific MLP encoders/decoders.
3. Per-embodiment normalization stats (min/max/mean/std per joint group).

> ⚠️ Picking the **wrong tag does not error** — it silently slices the obs
> under the wrong joint convention (`docs/embodiment_tags.md`).

The raw layout is embodiment-specific; the model-facing tensor is normalized +
padded to a shared format (`max_state_dim`, `max_action_dim`).

---

## 3. Per-simulation-framework state extraction

| Benchmark (sim API) | State extraction | Frame | Orientation | State dims |
|---|---|---|---|---|
| **LIBERO / LIBERO-Plus** (robosuite) | `robot0_eef_pos`, `quat2axisangle(robot0_eef_quat)`, `robot0_gripper_qpos` | World, absolute | **Axis-angle** (3) — fields labeled roll/pitch/yaw but are axis-angle | xyz(3)+rot(3)+gripper(2) = **8** |
| **SimplerEnv Google/Fractal** (ManiSkill2) | pos=`proprio[0:3]`; `quat=roll(proprio[3:7],-1)` (wxyz→xyzw); `gripper=1-proprio[7]` | World, absolute | **Quaternion xyzw** (4) | xyz(3)+quat(4)+gripper(1) = **8** |
| **SimplerEnv WidowX/Bridge** (ManiSkill2) | `mat2euler(quat2mat(quat) @ default_rot.T)` w/ Bridge frame correction | World, absolute (frame-corrected) | **Euler RPY** (3) + 1 pad | xyz(3)+rpy(3)+pad(1)+gripper(1) = **8** |
| **RoboCasa / GR1** (robosuite) | `end_effector_position_relative`, `end_effector_rotation_relative`, `gripper_qpos`, base pos/rot | EEF **relative** to base; base in world | Euler | ~**14** |
| **BEHAVIOR R1-Pro** (OmniGibson) | 258-D `obs["proprio"]` sliced into ~28 named keys | World, absolute | **Quaternion wxyz** for EEF | **258** raw |
| **DROID** (`OXE_DROID`) | `end_effector_position`, `end_effector_rotation`, `gripper_position` | **Robot base frame** | **Euler xyz** (3) | ~**7** |

Sources: `rldx/eval/sim/LIBERO/libero_env.py:44-159`,
`rldx/eval/sim/SimplerEnv/simpler_env.py:104-224`,
`rldx/eval/sim/BEHAVIOR/behavior_env.py:31-73`, config files under
`rldx/configs/data/`.

### Gotchas
- **Two quaternion orderings coexist** (wxyz internal/BEHAVIOR; xyzw
  SimplerEnv/LIBERO source) — a classic silent rotation bug.
- **"roll/pitch/yaw" field names are not always euler** — in LIBERO they hold
  **axis-angle** values; only WidowX/RoboCasa/DROID are true euler.

---

## 4. Axis-angle vs. roll-pitch-yaw (they are NOT the same)

Both describe a rotation with 3 numbers, but the meaning differs.

**Roll-pitch-yaw (Euler angles):** three *separate, sequential* rotations about
axes, e.g. `R = Rz(yaw) · Ry(pitch) · Rx(roll)`. Order matters (6 conventions);
suffers gimbal lock.

**Axis-angle (rotation vector):** a *single* rotation of angle θ about one
arbitrary unit axis **û**, packed as `v = θ·û`. The vector's **direction** is
the axis; its **magnitude** is the angle (radians). Not three separate angles.

They **only coincide** for a pure rotation about a single X/Y/Z axis. Example
of divergence — 90° about X then 90° about Y:
- Euler RPY = `(π/2, π/2, 0)`
- Axis-angle ≈ `(1.209, 1.209, 1.209)` (a 120° rotation about axis (1,1,1)/√3)

The conversion is **not linear** — you must go through a rotation matrix or
quaternion.

---

## 5. DROID specifics

From the server wire schema (`run_scripts/serve/run_rldx_pt_droid_server.sh:16,23`):

```
state.end_effector_position   float32 (B,1,3)
state.end_effector_rotation   float32 (B,1,3)   # euler xyz
state.gripper_position        float32 (B,1,1)

action.end_effector_position  float32 (B,16,3)  # DELTA EEF position
action.end_effector_rotation  float32 (B,16,3)  # DELTA EEF rotation (euler)
action.gripper_close          float32 (B,16,1)  # ABSOLUTE gripper-close
```

- **Rotation = Euler, `xyz` order** (true roll-pitch-yaw), **not** axis-angle.
- **State = absolute** EEF pose **in the robot base frame (`panda_link0`)**.
- **Action = delta** EEF in that same frame; gripper absolute.
- In code, `"xyz"` is fed to scipy (`rldx/data/state_action/pose.py:491-492`):
  **lowercase = extrinsic** rotations → `R = Rz·Ry·Rx`.
- ⚠️ **Degrees vs radians:** the `Pose` class defaults to `degrees=True`, but
  the DROID wire values are **radians**. Verify the flag on the DROID data path.

### Convert RPY (euler xyz) → axis-angle

Using scipy (matches this codebase):
```python
from scipy.spatial.transform import Rotation as R
rotvec = R.from_euler("xyz", rpy, degrees=False).as_rotvec()  # θ·û
# reverse: R.from_rotvec(rotvec).as_euler("xyz", degrees=False)
```

Explicit math (euler → matrix → axis-angle):
```python
import numpy as np
def rpy_xyz_to_axisangle(roll, pitch, yaw):
    cx, sx = np.cos(roll),  np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw),   np.sin(yaw)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    M = Rz @ Ry @ Rx                                   # extrinsic xyz
    theta = np.arccos(np.clip((np.trace(M)-1)/2, -1, 1))
    if np.isclose(theta, 0):
        return np.zeros(3)
    axis = np.array([M[2,1]-M[1,2], M[0,2]-M[2,0], M[1,0]-M[0,1]]) / (2*np.sin(theta))
    return theta * axis
```

---

## 6. sim-evals (`arhanjain/sim-evals`) — important mismatch

The repo is wired for **openpi / π0-FAST-DROID**, which uses a **joint-position**
interface, NOT EEF pose:

- **Observation** (`droid_environment.py`): `arm_joint_pos` (7 `panda_joint*`),
  `gripper_pos` (rescaled `finger_joint`), 3 cameras. **No EEF pose term.**
- **Action** (`ActionCfg`): `JointPositionActionCfg(joint_names=["panda_joint.*"])`
  + binary gripper. **Applies joint targets directly — no IK, no EEF frame.**

So RLDX-1-PT's EEF-pose/EEF-delta interface does **not** line up with stock
sim-evals. You must build two bridges yourself:
- **FK** to produce the EEF-pose observation (base frame).
- **IK** to consume the EEF-delta action → joint targets.

(Your own server header already warns about this.)

### Getting EEF pose in base frame in Isaac Lab
```python
import isaaclab.utils.math as math_utils
from scipy.spatial.transform import Rotation as R

def eef_pose_base_frame(env, asset_cfg=SceneEntityCfg("robot", body_names=["<EEF_BODY>"])):
    robot = env.scene[asset_cfg.name]
    ee_idx = asset_cfg.body_ids[0]
    ee_pos_w  = robot.data.body_state_w[:, ee_idx, 0:3]   # world frame
    ee_quat_w = robot.data.body_state_w[:, ee_idx, 3:7]   # wxyz
    base_pos_w  = robot.data.root_state_w[:, 0:3]
    base_quat_w = robot.data.root_state_w[:, 3:7]
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
        base_pos_w, base_quat_w, ee_pos_w, ee_quat_w)        # -> base frame
    q_xyzw = ee_quat_b[:, [1,2,3,0]].cpu().numpy()           # wxyz -> xyzw
    euler_xyz = R.from_quat(q_xyzw).as_euler("xyz", degrees=False)  # radians
    return torch.cat([ee_pos_b, torch.as_tensor(euler_xyz, device=ee_pos_b.device)], dim=-1)
```
Gotchas: `body_state_w`/`root_state_w` are **world frame**;
`subtract_frame_transforms` puts EEF in **base frame**. In this scene the robot
root spawns at origin+identity, so world≈base — but do the transform anyway.
Pick the **right EEF body** (TCP/flange matching DROID's `O_T_EE`), mind
**wxyz↔xyzw** and **extrinsic xyz radians**.

---

## 7. Coordinate frames — from scratch (robotics-noob primer)

### A "frame" = an origin + 3 axes
Three numbers `(x, y, z)` are meaningless until you say **measured from where,
along which axes**. That "where + which axes" is a **coordinate frame**
(reference frame). The same physical point gets **different numbers** in
different frames — like describing a cup "30 cm in front of me" vs. "2 m from
the north wall." Both correct; you must say which.

### Pose = position + orientation
A **pose** = `(x,y,z)` **position** + an **orientation** (rotation). A pose is
always "of *something*, expressed *in some frame*."

### The frames you meet
- **World frame** (global/inertial): fixed to the room/scene origin; never moves.
- **Base frame** (robot root, `panda_link0`): fixed to the arm's foot. Doesn't
  move during a task if the robot is bolted down.
- **End-effector frame** (EEF / TCP / tool / flange): rides on the gripper;
  moves constantly. DROID's `end_effector_*` is *this frame in the base frame*.
- **Link & joint frames**: the arm is links connected by joints; each joint
  relates a **parent** link's frame to a **child** link's frame by the joint angle.

```
world ─▶ base(panda_link0) ─▶ link1 ─▶ … ─▶ link7 ─▶ hand/EEF(TCP)
 fixed       fixed            each joint rotates child relative to parent
```

### "Expressed in / relative to / w.r.t. frame X"
All mean: the numbers are measured using X's origin and axes.
- EEF **in world** = gripper relative to the room.
- EEF **in base** = gripper relative to the arm's foot. ← DROID was trained on this.

Feeding world-frame numbers to a base-frame-trained policy looks fine but means
the wrong thing → silent failure. (In your sim the base sits at the world
origin, so they coincide *there* — a coincidence, not a rule.)

### Transform = convert a pose between frames
A **transform** bundles a translation + rotation. Written `T_base_ee` =
"express EEF in base frame."
- **Compose** transforms along the chain (base→link1→…→EEF) — multiplying joint
  transforms to get the gripper pose from joint angles is **Forward Kinematics (FK)**.
- **Invert/subtract** to go the other way (`subtract_frame_transforms`).
- The reverse problem, "what joint angles put the gripper at this pose?" is
  **Inverse Kinematics (IK)**. FK = angles→pose; IK = pose→angles.

### Absolute vs. delta, restated
- **Absolute pose** = full pose in a frame → your **state**.
- **Delta/relative** = a small change to add → your **action**. Sim adds delta
  onto current pose, then IK → joint targets.

### One-paragraph summary
A **frame** is an origin + 3 axes; a **pose** only has meaning relative to a
chosen frame. **World** is fixed to the room, **base** to the robot's foot,
**EEF** rides on the gripper. DROID gives the gripper's pose **in the base
frame**, and a **transform** (built via **FK**, inverted via **IK**) converts a
pose between frames. Get the frame right → it works; wrong → it fails quietly.
Always ask: *which frame?*
