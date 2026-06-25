# RoboCasa × Gymnasium × RLDX-1 — How the Env & Serving Stack Works

A from-scratch walkthrough of how a RoboCasa kitchen task is driven by the RLDX-1
model: how gymnasium envs are registered and built, the wrapper "onion" between
MuJoCo physics and the model, where robot state comes from (and how to tell
relative vs absolute), how the client/server split works, and how to enable RTC.

Written for someone new to gymnasium/robosuite/MuJoCo. Every claim is anchored to
a `file:line` so you can read the source yourself.

> **Paths.** The `robocasa` package is **not** in the rldx codebase. It is an
> editable install at `external_dependencies/robocasa/robocasa/`, installed into a
> **separate sim venv** (Python 3.10):
> `rldx/eval/sim/robocasa/robocasa_uv/.venv/bin/python`.
> That is why `import robocasa` fails in the main venv.

---

## 1. Gymnasium in 60 seconds

Gymnasium (the maintained fork of OpenAI Gym) is a **standard interface** for "an
environment you control step by step." The whole contract is:

```python
env = gym.make("env-id")                                   # construct
obs, info = env.reset()                                     # start an episode
obs, reward, terminated, truncated, info = env.step(action) # one step
env.close()
env.observation_space   # schema of obs (keys, shapes, dtypes)
env.action_space        # schema of a valid action
```

A **policy** is just `obs -> action`. The control loop:

```python
obs, _ = env.reset()
while not (terminated or truncated):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

Vocabulary: **observation** = what the agent sees (cameras + robot state +
instruction text); **action** = what it does (eef motion + gripper); **reward** /
`info["success"]` = task feedback; **terminated** = ended naturally;
**truncated** = cut off by a time limit; **episode** = one `reset()` → many
`step()` → done.

---

## 2. Registration: how a string becomes an env

You never instantiate an env class directly. Classes are **registered under a
string ID** into a global table (`gym.registry`); `gym.make(id)` looks it up.

For RoboCasa the IDs are **generated programmatically at import time** —
`external_dependencies/robocasa/robocasa/utils/gym_utils/gymnasium_groot.py:170-172`:

```python
for ENV in REGISTERED_ENVS:                                  # task names
    for ROBOT, ROBOT_ALIAS in GROOT_ROBOCASA_ENVS_ROBOTS.items():  # robots
        create_grootrobocasa_env_class(ENV, ROBOT, ROBOT_ALIAS)
```

and inside `create_grootrobocasa_env_class` (`gymnasium_groot.py:141,157`):

```python
id_name = f"robocasa_{robot_alias}/{class_name}"   # robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env
register(id=id_name, entry_point=f"...gymnasium_groot:{class_name}")
```

Key consequences:

- **Registration is an import side effect.** `import robocasa...gymnasium_groot`
  is what populates the registry. No import → `gym.make` raises "unknown env id".
  That is why the client imports it with `# noqa: F401`
  (`robocasa_client_from_scratch.py:424-425`) — the import looks unused but its
  *side effect* is the point.
- **The env classes are synthesized, not files.** `type(class_name,
  (GrootRoboCasaEnv,), {...})` (`gymnasium_groot.py:143`) builds them at runtime,
  so grepping for the class name finds nothing.
- **ID shape:** `robocasa_<robot_alias>/<TaskName>_<Robot>_Env`.
- There are **two** registries in play: gymnasium's (env IDs) and robosuite's own
  `REGISTERED_ENVS` (task name → task class, `gymnasium_basic.py:20,71`). Don't
  confuse them.

List everything registered:

```bash
PY=rldx/eval/sim/robocasa/robocasa_uv/.venv/bin/python
$PY -c "import gymnasium as gym, robocasa.utils.gym_utils.gymnasium_groot as g; \
print([k for k in gym.registry if k.startswith(('robocasa_','gr1_unified/'))][:20])"
```

The other module in that dir, `gymnasium_basic.py`, registers the older
`RoboCasaEnv` path; `gymnasium_groot.py` is the one the model client uses.

---

## 3. The wrapper "onion" — one transformation per layer

`gym.make(...)` returns an **onion of wrappers**, each adding one concern and
delegating downward via `super().step()/reset()`. **Actions flow down**
(model → physics); **observations flow up** (physics → model).

| Layer | Where | Role |
|---|---|---|
| 0. MuJoCo | external | raw rigid-body physics; knows only bodies/joints/contacts/pixels |
| 1. robosuite | external (`self.env`) | robot + scene + controllers + cameras; the "Panda in a kitchen" |
| 2. `RoboCasaEnv` | `gymnasium_basic.py:88` | gym⇄robosuite adapter; dict⇄flat-vector actions; defines spaces |
| 3. `GrootRoboCasaEnv` | `gymnasium_groot.py:21` | rename + resize into the **model's** vocabulary |
| 4. gym auto-wrappers | added by `gym.make` | enforce reset/step contract (OrderEnforcing, PassiveEnvChecker, TimeLimit) |
| 5. `MultiStepWrapper` / `ObsHistory` | `rldx/eval/sim/wrapper/multistep_wrapper.py` / client | stack a short frame history `(T,H,W,C)` |
| 6. client (batch + ZeroMQ) | `robocasa_client_from_scratch.py` | add batch dim `(B,...)`, talk to the model server |

### Layer 1 — robosuite

Owns the task (`Kitchen`), robot (`PandaOmron`), controllers (e.g. OSC turning
"move eef here" into joint torques), and cameras. Has its own task registry
(`REGISTERED_ENVS`). Built in `create_env_robosuite` (`gymnasium_basic.py:28-85`)
via `robosuite.make(**env_kwargs)`. Down: flat action vector → torques. Up: raw
obs dict with robosuite key names (`robot0_agentview_left_image`, joint sensors).

### Layer 2 — `RoboCasaEnv` (the gym⇄robosuite bridge)

- **Defines the spaces** by introspecting the controllers: `action_space` is a
  `Dict`, one `Box` per controllable part (`gymnasium_basic.py:151-168`);
  `observation_space` from a probe reset (`:170-196`).
- **Action: dict → flat vector.** `step(action_dict)` (`:275`) uses each part's
  `_action_split_indexes` to scatter dict values into one contiguous
  `env_action`, then calls robosuite (`:280-293`). `assert len(action_dict)==0`
  (`:289`) guarantees you supplied every part exactly once.
- **Observation: gather + normalize.** `get_basic_observation` (`:228`) flips
  images upright (MuJoCo renders bottom-up, `:232-233`), casts to float32,
  injects the task string under `"language"` (`:251`).
- Normalizes the API: robosuite returns 4 values, gym wants 5 → adds `truncated`
  (`:298,305`) and derives `info["success"]` from `reward>0` (`:300`).

### Layer 3 — `GrootRoboCasaEnv` (the vocabulary translator)

A pure **rename + resize** shim, both directions:

- **Obs (up):** `get_groot_observation` (`gymnasium_groot.py:90`) uses the
  `key_converter` to rename robosuite keys → `hand.*`/`body.*` → `state.*`,
  **resizes every camera to 256×256** (`process_img`, `:59-69`), renames cameras
  to canonical names (`video.res256_image_side_0`, …), and sets
  `annotation.human.action.task_description` (`:116`).
- **Action (down):** `step` (`:124`) calls `key_converter.unmap_action(action)`
  (`:128`) to turn model action names back into robosuite part-names, then
  `super().step()`.

This is the class your registered `entry_point` actually constructs.

### Layer 4 — gymnasium auto-wrappers

Added invisibly by `gym.make`: **OrderEnforcing** (step-before-reset guard),
**PassiveEnvChecker** (validates obs/action match the declared spaces on the first
steps — catches shape bugs), **TimeLimit** (sets `truncated` after N steps). Pure
guard rails; no data transformation.

### Layer 5 — `MultiStepWrapper` / `ObsHistory`

Bridges "env gives one frame at a time" vs "model wants a short history." Keeps a
ring buffer and stacks frames along a new time axis at the model's
`delta_indices` (`multistep_wrapper.py:329` `_get_obs`). The from-scratch client
reimplements this as `ObsHistory` so it needs zero `rldx` imports.

### Layer 6 — client (batch + wire)

Adds the batch dim `(T,H,W,C) → (1,T,H,W,C)`, msgpacks, sends over ZeroMQ;
receives an action **chunk** `(B,T_action,D)`, slices per step, feeds each back
into `env.step()`.

### The two data flows end to end

```
OBSERVATION (up):
  MuJoCo state/pixels
    → robosuite raw dict (robot0_agentview_left_image, joint sensors)
    → RoboCasaEnv: flip images, float32, add "language"
    → GrootRoboCasaEnv: rename→video.*/state.*, resize 256², add task_description
    → gym checkers (pass-through)
    → MultiStepWrapper/ObsHistory: stack T frames  →  (T,H,W,C)
    → client: add batch  →  (1,T,H,W,C)  →  model

ACTION (down):
  model action chunk (1,T_action,D)
    → client: slice one step → dict of model-named actions
    → GrootRoboCasaEnv.unmap_action: model names → robosuite part-names
    → RoboCasaEnv.step: scatter dict into one flat vector
    → robosuite controllers: vector → joint torques
    → MuJoCo: integrate physics → next state
```

---

## 4. Where robot state comes from, and relative vs absolute

### Two raw sources, merged in `RoboCasaEnv`

- **A — robosuite observables:** `self.env._get_observations(force_update=True)`
  (`gymnasium_basic.py:170-174`) → `robot0_eef_pos`, `robot0_joint_pos`,
  `robot0_base_to_eef_pos`, `robot0_gripper_qpos`, …
- **B — direct MuJoCo qpos:** `gather_robot_observations(env)`
  (`models/robots/__init__.py:112`) reads `sim.data.qpos` per joint, using
  `sim.model.jnt_qposadr` (offset) and `sim.model.jnt_type`
  (free=7 / ball=4 / hinge=1 DOFs) to slice each joint's numbers
  (`:120-140`). Gripper joints are reordered to match the real robot (`:138`).

Merged via `obs.update(gather_robot_observations(self.env))`
(`gymnasium_basic.py:175,229`). `sim.data.qpos` is the ground truth.

### Raw → `state.*` keys

The per-robot key converter renames them. For `robocasa_panda_omron/` that is
`PandaOmronKeyConverter.map_obs` (`models/robots/__init__.py:432-449`):

```python
{
  "hand.gripper_qpos":                   robot0_gripper_qpos,
  "body.base_position":                  robot0_base_pos,
  "body.end_effector_position_relative": robot0_base_to_eef_pos,   # base frame
  "body.end_effector_rotation_relative": robot0_base_to_eef_quat,
  "body.end_effector_position_absolute": robot0_eef_pos,           # world frame
  "body.end_effector_rotation_absolute": robot0_eef_quat,
  "body.joint_position":                 robot0_joint_pos,         # arm joint angles (rad)
  "body.joint_position_cos":             robot0_joint_pos_cos,
  "body.joint_position_sin":             robot0_joint_pos_sin,
  "body.joint_velocity":                 robot0_joint_vel,
  "hand.gripper_qvel":                   robot0_gripper_qvel,
}
```

Then `get_groot_observation` strips the prefix and prepends `state.`
(`gymnasium_groot.py:93-97`): `body.joint_position → state.joint_position`. The
model's `modality_config` selects which subset it actually consumes.

### How to know relative vs absolute

**(a) The key-name suffix.** PandaOmron exposes **both** frames side by side:

| `state.*` key | robosuite source | frame |
|---|---|---|
| `…_position_absolute` | `robot0_eef_pos` | **world** |
| `…_position_relative` | `robot0_base_to_eef_pos` | **robot base** |
| `…_rotation_absolute` | `robot0_eef_quat` | world, quaternion |
| `…_rotation_relative` | `robot0_base_to_eef_quat` | base, quaternion |
| `joint_position` | `robot0_joint_pos` | joint angles (rad), absolute |

"relative" = expressed in the robot's base frame; "absolute" = world frame.

**(b) The authoritative lookup: `get_metadata(name)`**
(`PandaOmronKeyConverter.get_metadata`, `models/robots/__init__.py:496-521`):

```python
absolute=False : body.base_position, body.end_effector_position_relative,
                 body.end_effector_position (action)        # → relative / delta
absolute=False + rotation_type=QUATERNION : base_rotation, end_effector_rotation_relative
absolute=False + rotation_type=AXIS_ANGLE : end_effector_rotation (action)
absolute=True  : everything else (joint_position, gripper_qpos, velocities, …)
```

`rotation_type` tells you whether a rotation vector is a **quaternion** (obs) or
**axis-angle** (action delta).

> ⚠️ Naming gotcha: `body.base_position` is `absolute=False` (a delta), despite
> the word "position." Trust the `_relative`/`_absolute` suffix or
> `get_metadata`, not the bare word.

**Action side:** `map_action`/`unmap_action` (`:451,473`) — PandaOmron eef
actions are `absolute=False` (delta commands the OSC controller consumes), with
axis-angle rotation. Gripper / control-mode are discretized to ±1 (`:478-492`).
Contrast GR1 robots, which set `control_delta=False` (`gymnasium_basic.py:129`)
for absolute joint control.

Inspect at runtime:

```bash
PY=rldx/eval/sim/robocasa/robocasa_uv/.venv/bin/python
$PY -c "
from robocasa.models.robots import make_key_converter
kc = make_key_converter('PandaOmron')
for n in ['body.joint_position','body.end_effector_position_relative',
          'body.end_effector_position_absolute','body.base_position']:
    print(f'{n:45s}', kc.get_metadata(n))"
```

---

## 5. The client/server split (and the #1 gotcha)

The model wants a modern CUDA/torch stack; the simulator wants an older
numpy/torch. They cannot share one venv, so RLDX-1 uses a **client/server split**
over ZeroMQ + msgpack:

```
client (sim venv)  --- observation request --->  server (model on GPU)
  robosuite/mujoco  <--- action chunk reply  ---  rldx/eval/run_rldx_server.py
```

### Flat vs nested observation — the contract mismatch

There are two observation formats:

- **Flat** (RLDX sim format): keys like `video.left_view`, `state.<key>`, and a
  language key — what the from-scratch client and sim wrappers produce.
- **Nested** (bare `RLDXPolicy` format):
  `{"video": {...}, "state": {...}, "language": {...}}`.

The translator between them is **`RLDXSimPolicyWrapper`** (`rldx/policy/rldx_policy.py:248`):
its `check_observation` validates the **flat** format (`:280`), and its
`_get_action` (`:427`) rewrites flat → nested (`video.left_view →
observation["video"]["left_view"]`, `:492-517`) before the inner policy runs.

**Symptom if the wrapper is missing:**

```
AssertionError: Observation must contain a 'video' key
  policy.py:104 get_action → rldx_policy.py:203 RLDXPolicy.check_observation
  → observation_validator.py:74
```

Line 203 is the **inner** `RLDXPolicy` (nested validator), proving the wrapper was
not installed — so the flat keys the client sends are rejected.

**Fix:** start the server with `--use-sim-policy-wrapper`
(`run_rldx_server.py:224-227`):

```bash
RLDX_PATCHEMBED_FP32=1 RLDX_ATTN_IMPL=sdpa CUDA_VISIBLE_DEVICES=0 \
  uv run python rldx/eval/run_rldx_server.py \
    --model-path RLWRLD/RLDX-1-FT-ROBOCASA \
    --embodiment-tag GENERAL_EMBODIMENT \
    --use-sim-policy-wrapper \
    --host 127.0.0.1 --port 20200
```

(Alternative: have the client emit the nested format directly — short modality
names as keys, language as `list[list[str]]` of shape `(B,1)`. But the wrapper
path is the intended design.)

### Client checklist (the "five things")

From the from-scratch client header (`robocasa_client_from_scratch.py:40-58`):
wire format (ZeroMQ+msgpack with a numpy hook), the contract (read
`get_modality_config()` at runtime, never hardcode), key mapping (sim key → model
key), temporal stacking (ring buffer at `delta_indices`), action chunking (run N
of the returned chunk open-loop, then re-query).

---

## 6. Real-Time Chunking (RTC) serving

RTC keeps consecutive action chunks continuous: each step the server freezes the
first `d` actions of the **previous** chunk (the "prefix") and inpaints the rest.
The previous chunk is cached **per `session_id`** in the `SessionRegistry`
(`policy_runtime.py:155` save, `:324` reload), so the client doesn't round-trip
the prefix — it just sends stable `session_ids`.

RTC is **off by default** (`rtc_inference_mode="none"`, `configs/model/rldx.py:187`).

### Server flags (tyro → kebab-case), `run_rldx_server.py:102-118`

| Flag | Meaning | Default |
|---|---|---|
| `--rtc-inference-mode {none,trained,guided}` | algorithm | checkpoint's value |
| `--rtc-inference-delay` | `d` = frozen prefix length | 0 |
| `--rtc-inference-exec-horizon` | `s` = actions executed per chunk | 0 → `action_horizon − d` |
| `--rtc-jacobian-beta` | guidance clip (**guided only**) | — |
| `--rtc-jacobian-steps-only` | guide only first N denoise steps (**guided only**) | 3 |

Modes: **trained** (RTC inpainting via attention; cheap, no autograd) and
**guided** (Jacobian-VJP guidance; needs autograd). Guided is **incompatible with
`--compile=fullgraph`** — `_validate_cli` rejects it (`run_rldx_server.py:156`);
use eager or `--compile=submodule`.

### Launch

```bash
RLDX_PATCHEMBED_FP32=1 RLDX_ATTN_IMPL=sdpa CUDA_VISIBLE_DEVICES=0 \
  uv run python rldx/eval/run_rldx_server.py \
    --model-path RLWRLD/RLDX-1-FT-ROBOCASA \
    --embodiment-tag GENERAL_EMBODIMENT \
    --use-sim-policy-wrapper \
    --rtc-inference-mode trained \
    --rtc-inference-delay 4 \
    --rtc-inference-exec-horizon 8 \
    --host 127.0.0.1 --port 20200
```

### Client side

RTC is mostly transparent via the server cache. The client must:

1. Send a **stable `session_id`** per env, and `reset_memory=[True]` on the first
   call of each episode (invalidates the RTC cache, `policy_runtime.py:244`). The
   from-scratch client already does both
   (`robocasa_client_from_scratch.py:385,461`).
2. **Execute exactly `s` actions** before re-querying — set the client's
   `--exec-horizon` equal to the server's `s`, or the freeze boundary won't line
   up with what was executed.

Wire options decoded in `step_request.py` (`decode_options_to_step_request`,
`:148`): `session_ids`, `reset_memory`, and optionally `action_prefix` +
`rtc_prefix_len`. The optional client-supplied prefix (real-robot latency) takes
priority over the cache (`policy_runtime.py:248-321`) and is sent in **physical
units** — the server re-normalizes it (`:269-307`).

---

## 7. Quick reference — key files

| Concern | File |
|---|---|
| Env ID registration (generated) | `external_dependencies/robocasa/robocasa/utils/gym_utils/gymnasium_groot.py:139-172` |
| gym⇄robosuite adapter, spaces, dict⇄vector actions | `…/gym_utils/gymnasium_basic.py:88` |
| Model-vocabulary translation (obs/action rename, resize) | `…/gym_utils/gymnasium_groot.py:21` |
| Raw state from MuJoCo qpos | `external_dependencies/robocasa/robocasa/models/robots/__init__.py:112` |
| PandaOmron key map + abs/rel metadata | `…/models/robots/__init__.py:416-521` |
| Converter dispatch by robot | `…/models/robots/__init__.py:748` |
| Temporal history stacking (reference) | `rldx/eval/sim/wrapper/multistep_wrapper.py` |
| Flat↔nested translation wrapper | `rldx/policy/rldx_policy.py:248` |
| Observation validator (nested) | `rldx/policy/observation_validator.py` |
| Server entry point + RTC/compile flags | `rldx/eval/run_rldx_server.py` |
| RTC prefix injection / cache | `rldx/policy/policy_runtime.py:223-358` |
| From-scratch client (teaching edition) | `run_scripts/eval/robocasa_kitchen/robocasa_client_from_scratch.py` |

**Sim venv python:** `rldx/eval/sim/robocasa/robocasa_uv/.venv/bin/python`
(needs `MUJOCO_GL=egl` for headless rendering).
