# RLDX-1-PT — `OXE_DROID` head: zero-shot input / output spec

Reference for running the **pretrained** checkpoint
[`RLWRLD/RLDX-1-PT`](https://huggingface.co/RLWRLD/RLDX-1-PT) zero-shot through
the `OXE_DROID` (Franka / DROID) embodiment head, via
`rldx.policy.rldx_policy.RLDXPolicy`.

> ⚠️ **RLDX-1-PT is a pretraining checkpoint.** The `OXE_DROID` head runs and
> produces correctly-shaped, sanely-scaled actions, but it reuses DROID's
> (Franka) normalization stats and 3-camera layout — task performance on a
> different robot will be poor until fine-tuned. The tags `GENERAL_EMBODIMENT`
> and `NEW_EMBODIMENT` have **no** normalization stats in PT and cannot be used
> zero-shot.

---

## At a glance

| | |
|---|---|
| Checkpoint | `RLWRLD/RLDX-1-PT` (~14 GB, 6.9B params) |
| Embodiment tag | `EmbodimentTag.OXE_DROID` (`"droid"`) |
| Control rate | **10 Hz** (100 ms / step) |
| Cameras | 3 (`primary`, `secondary`, `wrist`) |
| Frames per camera | **4** (so 12 images / call) |
| Action chunk | **16 steps** of delta end-effector + gripper |
| Fits on | RTX 4090 (24 GB): ~15.4 GB load, ~17.1 GB peak |

---

## INPUT — nested `obs` dict passed to `policy.get_action(obs)`

```python
obs = {
    "video":    { <camera_key>: np.uint8  array (B, T, H, W, 3) },
    "state":    { <state_key>:  np.float32 array (B, 1, D) },
    "language": { "annotation.human.action.task_description": [["<task string>"]] },
}
```

with `B = 1`, `T = 4`.

### Video (3 cameras, 4 frames each)

| Key | dtype | shape | meaning |
|---|---|---|---|
| `video["primary"]` | `uint8` | `(1, 4, H, W, 3)` | exterior scene cam #1 |
| `video["secondary"]` | `uint8` | `(1, 4, H, W, 3)` | exterior scene cam #2 (different angle) |
| `video["wrist"]` | `uint8` | `(1, 4, H, W, 3)` | wrist-mounted cam |

- All 3 cameras are **required** (the observation validator errors otherwise).
- `H, W` are free — the processor downsamples to a fixed token budget, so
  resolution barely affects VRAM. DROID used **168×336**.
- RGB, `uint8`, range 0–255.

### State (current robot proprioception, single timestep)

| Key | dtype | shape | meaning |
|---|---|---|---|
| `state["end_effector_position"]` | `float32` | `(1, 1, 3)` | current eef position (m), base frame |
| `state["end_effector_rotation"]` | `float32` | `(1, 1, 3)` | current eef orientation, 3-vector (axis-angle / euler) |
| `state["gripper_position"]` | `float32` | `(1, 1, 1)` | current gripper opening |

Pass **raw** (un-normalized) values — normalization is applied internally from
the checkpoint's DROID statistics.

### Language

| Key | type | value |
|---|---|---|
| `language["annotation.human.action.task_description"]` | `list[list[str]]` | `[["pick up the red cube"]]` |

---

## Frame timing (the 4 frames)

The 4 frames per camera are the **last 4 at 200 ms spacing** (a 0.6 s window):

```
capture/buffer rate : 10 Hz          (1 frame per camera per 100 ms control step)
video_stride        : 2              (sample every 2nd buffered frame)
frame timestamps    : t = {0, -200, -400, -600} ms   (most-recent last)
ring buffer length  : (4-1)*2 + 1 = 7 frames per camera
```

To reproduce: run the loop at 10 Hz, push one rendered frame per camera into a
length-7 buffer each step, then take every other frame (4 total).

> The raw DROID **dataset** was recorded at 15 Hz, but RLDX operates at **10 Hz**
> control — match 10 Hz, not 15.

---

## OUTPUT — `action, info = policy.get_action(obs)`

`action` is a dict; each value is a **16-step action chunk** shaped `(1, 16, D)`:

| Key | shape | representation | meaning |
|---|---|---|---|
| `action["end_effector_position"]` | `(1, 16, 3)` | **delta**, meters | eef position offset from the current pose, per step (cumulative from current: step 0 ≈ 0, grows outward) |
| `action["end_effector_rotation"]` | `(1, 16, 3)` | **delta**, radians | eef orientation offset from current pose, 3-vector |
| `action["gripper_close"]` | `(1, 16, 1)` | **absolute**, ~[0, 1] | gripper command (1 = closed, 0 = open) |

`info` is an auxiliary dict (timing / metadata), not needed to drive the robot.

---

## Mapping output → a joint-controlled sim (e.g. MuJoCo)

The eef outputs are **offsets relative to the current end-effector pose**. For
each step `i` in `0..15`:

```text
T_target[i] = compose(T_current_eef, Δpos[i], Δrot[i])   # absolute target pose
q_arm[i]    = IK(T_target[i])                            # 7 joint angles
q_grip[i]   = map(gripper_close[i])                      # absolute → gripper joint
```

> The exact rotation convention of the 3-vector (axis-angle vs euler XYZ, and
> eef-frame vs base-frame composition) is dataset-specific; verify on a
> fine-tuned checkpoint before trusting closed-loop behavior.

---

## Minimal usage

```python
import numpy as np
from rldx.policy.rldx_policy import RLDXPolicy
from rldx.data.embodiment_tags import EmbodimentTag

policy = RLDXPolicy(
    model_path="RLWRLD/RLDX-1-PT",
    embodiment_tag=EmbodimentTag.OXE_DROID,
    device="cuda:0",
)

obs = {
    "video": {
        "primary":   prim,    # (1, 4, H, W, 3) uint8
        "secondary": sec,     # (1, 4, H, W, 3) uint8
        "wrist":     wrist,   # (1, 4, H, W, 3) uint8
    },
    "state": {
        "end_effector_position": pos,   # (1, 1, 3) float32
        "end_effector_rotation": rot,   # (1, 1, 3) float32
        "gripper_position":      grip,  # (1, 1, 1) float32
    },
    "language": {
        "annotation.human.action.task_description": [["pick up the red cube"]],
    },
}

action, info = policy.get_action(obs)
# action["end_effector_position"] : (1, 16, 3)  delta xyz (m)
# action["end_effector_rotation"] : (1, 16, 3)  delta rot (rad)
# action["gripper_close"]         : (1, 16, 1)  absolute gripper
```

---

## Measured performance (RTX 4090, 24 GB, eager / no compile)

| Input res | Latency | Throughput | VRAM peak |
|---|---|---|---|
| 224×224 | 103 ms | 9.7 Hz | 17.10 GB |
| 320×320 | 123 ms | 8.2 Hz | 17.15 GB |
| 448×448 | 127 ms | 7.9 Hz | 17.15 GB |

- Load footprint ~15.4 GB; peak ~17.1 GB → **no OOM**, ~8 GB headroom.
- VRAM is ~flat across input resolution (processor caps the token budget).
- For lower latency use `--compile submodule` (appropriate for the 4090);
  `--compile fullgraph` is tuned for RTX 5090 / Blackwell.

Reproduce:

```bash
uv run python rldx/eval/benchmark_policy.py \
    --model-path RLWRLD/RLDX-1-PT --embodiment-tag OXE_DROID \
    --warmup 10 --iters 50
```
