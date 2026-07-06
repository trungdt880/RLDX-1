# ALLEX → RoboCasa integration

Roll out the **`RLDX-1-MT-ALLEX`** checkpoint (registry `rldx_1_midtrain_allex`, HF
`RLWRLD/RLDX-1-MT-ALLEX`) on the **ALLEX** humanoid inside RoboCasa. ALLEX is
WIRobotics' fixed-base upper-body humanoid (48 actuated DOF: 2 arms × 7, 2 hands ×
15, neck 2, waist 2), model at `~/workspace/allex_model`.

**Status: working closed-loop harness.** The policy drives ALLEX in a RoboCasa scene
end-to-end, stably. This is a **harness — no task-success claim** (the checkpoint was
trained on real ALLEX data only; sim success is dominated by the visual+dynamics
domain gap and is not evaluated here).

> **Just want the sim (scenes + trajectory replay), not the RLDX-1 model?**
> See [`SIM_ONLY_SETUP.md`](./SIM_ONLY_SETUP.md) — one command
> (`setup_sim.sh`) sets up ALLEX-in-RoboCasa with no checkpoint/server.

---

## What was built

The model + embodiment already existed (ALLEX is a first-class `general_embodiment`
in the checkpoint). What was missing — and what this adds — is the RoboCasa/robosuite
side that speaks the ALLEX policy contract:

1. **`AllexRobot`** — a gripperless robosuite robot holding the full 48-joint ALLEX
   chain (hands modeled **as hands**, not grippers), driven by a **passthrough
   joint-position controller** that writes absolute targets straight to ALLEX's own
   `<position>` actuators (NO IK, NO computed-torque). 6 controller parts = the 6
   contract joint groups.
2. **`AllexKeyConverter`** — maps robosuite obs/action ↔ the GROOT contract
   (`state.<group>` ↔ `robot0_<group>`; `video.camera_ego_left`; absolute joint
   metadata; **torque omitted**). Registers 197 `robocasa_allex/*` gym env IDs.
3. **Gripper-optional task layer** — the tabletop success/reward path was
   gripper-only; now it's safe for a gripperless robot (returns `success=False` —
   see `TODO(ALLEX)` for a real geometry-based metric).
4. **Serve + client** — the model server (RLDX venv) + an ALLEX rollout client
   (robosuite venv) over ZeroMQ.

### How it was de-risked (before any sim code)
An offline **falsification gate** ran against the real checkpoint: frozen the
observation/action contract, cross-checked joint ordering vs MJCF limits, proved
absolute joint-position control tracks in MuJoCo, and measured torque-input
sensitivity. Result → **GO**, with torque **omitted** (`allow_missing_physics=True`
makes mask=0 in-distribution; feeding a MuJoCo torque surrogate would be OOD). See
`GO_NOGO.md` and `contract/allex_contract.md` for the frozen contract, and
`checks/` for the falsification tests.

---

## Repo layout

```
run_scripts/eval/robocasa_allex/
├─ README.md                     # this file
├─ GO_NOGO.md                    # Phase-0/0.5 gate decision + rollout config
├─ contract/
│  ├─ allex_contract.md          # FROZEN obs/action contract (source of truth)
│  ├─ allex_contract.py          # executable: GROUP_ORDER/JOINT_NAMES + normalize + assert
│  ├─ allex_stats.json           # general_embodiment norm stats (from the checkpoint)
│  └─ verify_joint_order.py      # ordering cross-check vs MJCF joint limits
├─ checks/                       # phase-by-phase verification (all PASS)
│  ├─ coupling_check.py          # 0.5c pure-MuJoCo absolute-position tracking
│  ├─ action_sanity.py           # 0.5a real checkpoint -> in-range actions
│  ├─ torque_sensitivity.py      # 0.5b torque-OOD (decides: omit torque)
│  ├─ robosuite_merge_check.py   # 1.4 merge+prefix preserves model + tracks
│  ├─ robot_make_check.py        # 1.5 AllexRobot builds + tracks via composite wrapper
│  ├─ converter_check.py         # 2  key-converter unit + env gym.make integration
│  └─ episode_check.py           # 3  full 150-step episode, no crash
├─ allex_rollout_client.py       # 4  CLOSED-LOOP client (env <-> ZMQ <-> model server)
├─ replay_in_scene.py            # visual: replay a baked motion in the scene (mp4/gif/GUI)
└─ rollout/                      # generated videos / frames

external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/   (vendored fork)
├─ models/robots/manipulators/allex_robot.py   # AllexRobot + passthrough controller + wrapper
├─ models/assets/robots/allex/{robot.xml, default_allex_position.json, meshes->}
├─ models/robots/__init__.py                    # AllexKeyConverter + registration
├─ utils/gym_utils/{gymnasium_basic,gymnasium_groot}.py  # ALLEX env wiring
├─ utils/object_utils.py                         # gripper-optional guards
└─ environments/tabletop/tabletop.py             # gripper-optional reward + mount offset
```

> **Setup dependency:** `models/assets/robots/allex/meshes` is a symlink to
> `~/workspace/allex_model/meshes`. On a fresh checkout, recreate it:
> `ln -sfn ~/workspace/allex_model/meshes <fork>/robocasa/models/assets/robots/allex/meshes`.

---

## How to run

All commands from repo root `/home/thor/RLDX-1`. Two venvs:
- **robosuite venv** (env, torch 2.5): `rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python`
- **RLDX venv** (model, torch 2.9): `/home/thor/RLDX-1/.venv/bin/python`

### 1. Watch a baked trajectory in the scene (no policy, no server)
```bash
# smooth mp4 (headless; native-dt drive + gravity/velocity feed-forward, like allex_model/replay.py)
MUJOCO_GL=egl rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python \
    run_scripts/eval/robocasa_allex/replay_in_scene.py --motion hello
# live interactive window (needs a display)
DISPLAY=:0 rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python \
    run_scripts/eval/robocasa_allex/replay_in_scene.py --viewer
```
Options: `--motion hello|demo1|<path.npz>`, `--camera robot0_frontview|robot0_zed_left_camera_optical_frame(ego)`,
`--robot-back <m>` (mount further back from the counter; default 0.4), `--robot-height <m>`,
`--speed`, `--fps`, `--env robocasa_allex/<Task>_AllexRobot_Env`, `--out <file.mp4|.gif>`.

### 2. Run the verification checks
```bash
PY=rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python
MUJOCO_GL=egl $PY run_scripts/eval/robocasa_allex/checks/converter_check.py   # env builds + emits contract obs
MUJOCO_GL=egl $PY run_scripts/eval/robocasa_allex/checks/episode_check.py     # full 150-step episode
# 0.5a/0.5b load the 16GB model -> run in the RLDX venv:
RLDX_PATCHEMBED_FP32=1 RLDX_ATTN_IMPL=sdpa /home/thor/RLDX-1/.venv/bin/python \
    run_scripts/eval/robocasa_allex/checks/action_sanity.py
```

### 3. Closed-loop rollout (policy drives the robot)
Two processes. **Terminal A — server** (RLDX venv, GPU; loads the 16GB checkpoint):
```bash
SNAP=$(/home/thor/RLDX-1/.venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('RLWRLD/RLDX-1-MT-ALLEX'))")
RLDX_PATCHEMBED_FP32=1 RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/thor/RLDX-1/.venv/bin/python rldx/eval/run_rldx_server.py \
    --model-path "$SNAP" --embodiment-tag GENERAL_EMBODIMENT \
    --use-sim-policy-wrapper --host 127.0.0.1 --port 20250
```
(action horizon 40 comes from the checkpoint modality — there is no `--action-horizon` flag.)
**Terminal B — client** (robosuite venv), once the server is listening:
```bash
MUJOCO_GL=egl rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python \
    run_scripts/eval/robocasa_allex/allex_rollout_client.py
```
Expect a stable rollout (~380 ms/server call steady-state), `reward=0` (not a success
claim), sim finite, `neq==12`. Logs/video land in `run_scripts/eval/robocasa_allex/`.

---

## What to do next

1. **Real success metric.** The reward is a gripperless stub (`success=False`). Add a
   geometry-based check (e.g. `object in receptacle`, not eef-based) at the
   `TODO(ALLEX)` marker in `tabletop.py:reward` to turn this from a harness into an
   actual eval.
2. **Byte-exact joint ordering.** The intra-group ordering is cross-check-verified
   (uniquely-best of all permutations) but not confirmed against the `real_allex`
   dataset's `meta/modality.json`. Get that file and diff against
   `allex_contract.JOINT_NAMES` before trusting any rollout *number*.
3. **Close the domain gap.** The checkpoint saw zero sim frames. This harness now
   exists to *measure* the gap — next is sim co-training / fine-tuning against it.
4. **Execution smoothness (optional).** The policy path runs through robocasa's 20 Hz
   `env.step` (coarser than the 200 Hz open-loop replay). If smoother execution
   matters, tune the env `control_freq` / interpolate the 40-step action chunk across
   env steps.
5. **Portability.** The mesh symlink and absolute paths are machine-local; parametrize
   before running elsewhere.
