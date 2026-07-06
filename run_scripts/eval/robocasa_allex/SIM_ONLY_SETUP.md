# ALLEX ⇄ RoboCasa — sim-only setup

For working on the **simulation** with the ALLEX humanoid: load a RoboCasa scene,
drop ALLEX in, and replay baked trajectories from `allex_model`. This path uses
**none** of the RLDX-1 model side — no 16 GB checkpoint, no policy server, no
ZeroMQ. Just RoboCasa + MuJoCo + ALLEX.

> If you *also* want the closed-loop policy rollout (the `RLDX-1-MT-ALLEX`
> checkpoint driving the robot), that's a separate, heavier setup — see the main
> [`README.md`](./README.md). Everything below is self-contained without it.

---

## What you get

- ALLEX (48-DOF fixed-base upper-body humanoid, hands modeled **as hands**)
  integrated as a first-class robosuite robot inside the RoboCasa tabletop envs.
- `replay_in_scene.py` — drop a baked motion (`allex_model/examples/motions/*.npz`)
  into a real RoboCasa scene and watch it, either as an **mp4** (headless) or in a
  **live MuJoCo window**. Motion is driven exactly like `allex_model`'s own
  `replay.py` (native-dt stepping + gravity/velocity feed-forward), so it looks
  smooth, not laggy.

This is a **harness** — it does not score task success (the reward path is a
gripperless stub). It's for building/looking at ALLEX-in-RoboCasa, not for eval
numbers yet.

---

## Prerequisites

1. **This repo** checked out (you're reading a file in it).
2. **`uv`** — the Python package manager. Install once:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # then re-open your shell so `uv` is on PATH
   ```
3. **The `allex_model` repo** cloned locally (provides the robot meshes + the
   sample motions). Default expected location: `~/workspace/allex_model`.
   If it lives elsewhere, set `ALLEX_MODEL_DIR` (see below).
4. **Disk + network** for the one-time RoboCasa asset download (~8 GB of scene
   textures, fixtures, and objects). A GPU is *not* required — rendering uses EGL
   (headless) or GLFW (a window); the smoke test runs on CPU-class hardware.

---

## Install (one command)

```bash
run_scripts/eval/robocasa_allex/setup_sim.sh
```

That's it. The script is idempotent (safe to re-run) and does everything:

1. checks `uv`, the robocasa fork, and your `allex_model`;
2. creates a **Python 3.10** venv with `uv` at
   `external_dependencies/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv`;
3. installs the pinned sim stack — `robosuite 1.5.1`, the ALLEX robocasa fork
   (editable), `mujoco 3.2.6`, `numpy 1.26.4`, `gymnasium 0.29.1`,
   `imageio` + `imageio-ffmpeg` (for mp4);
4. symlinks the ALLEX meshes from `allex_model` into the fork;
5. downloads the RoboCasa scene/object assets (~8 GB) **if missing**;
6. runs a headless smoke test (builds the ALLEX env, resets, steps once, checks
   the ego camera renders and the 12 joint couplings survive).

On success it prints the exact `python` path and a ready-to-paste replay command.

### Knobs (optional env vars)

| var | default | meaning |
|-----|---------|---------|
| `ALLEX_MODEL_DIR` | `~/workspace/allex_model` | where your `allex_model` repo lives |
| `ALLEX_SIM_VENV` | `<fork>/robocasa_uv/.venv` | where to create the venv |
| `ALLEX_SKIP_ASSETS` | `0` | `1` = skip the ~8 GB download (already have them) |
| `ALLEX_SKIP_SMOKE` | `0` | `1` = skip the final headless smoke test |
| `ALLEX_FORCE_VENV` | `0` | `1` = delete and rebuild the venv from scratch |

Example — `allex_model` in a custom spot, assets already present:
```bash
ALLEX_MODEL_DIR=/data/allex_model ALLEX_SKIP_ASSETS=1 \
    run_scripts/eval/robocasa_allex/setup_sim.sh
```

---

## Run it

Let `PY` be the venv python the installer printed
(`external_dependencies/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python`).

**Headless → mp4** (server / no display):
```bash
MUJOCO_GL=egl $PY run_scripts/eval/robocasa_allex/replay_in_scene.py --motion hello
# -> run_scripts/eval/robocasa_allex/rollout/allex_replay.mp4
```

**Live interactive window** (needs a display — e.g. a monitor on the box):
```bash
DISPLAY=:0 $PY run_scripts/eval/robocasa_allex/replay_in_scene.py --viewer
```

Handy flags for `replay_in_scene.py`:

| flag | what it does |
|------|--------------|
| `--motion hello \| demo1 \| /abs/path.npz` | which baked trajectory to replay |
| `--camera robot0_frontview \| robot0_zed_left_camera_optical_frame` | render view (the second is ALLEX's ego camera) |
| `--robot-back <m>` | mount ALLEX further back from the counter (default 0.4 m — keeps the hands off the table) |
| `--robot-height <m>` | ALLEX base height (default 0.9 m) |
| `--env robocasa_allex/<Task>_AllexRobot_Env` | which scene (many `robocasa_allex/*` ids are registered) |
| `--speed`, `--fps`, `--out <file.mp4\|.gif>` | pacing + output |

**Sanity checks** (optional, no policy needed):
```bash
MUJOCO_GL=egl $PY run_scripts/eval/robocasa_allex/checks/converter_check.py  # env builds + emits contract obs
MUJOCO_GL=egl $PY run_scripts/eval/robocasa_allex/checks/episode_check.py    # full 150-step episode, no crash
```

---

## Where things live

- **ALLEX robot + controller**: `external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/models/robots/manipulators/allex_robot.py`
- **ALLEX MuJoCo model + config**: `.../robocasa/models/assets/robots/allex/{robot.xml, default_allex_position.json, meshes→}`
- **Env wiring / registration**: `.../robocasa/models/robots/__init__.py`, `.../robocasa/utils/gym_utils/`, `.../robocasa/environments/tabletop/tabletop.py`
- **Replay + checks**: `run_scripts/eval/robocasa_allex/`

If you edit any of the `allex_robot.py` / `robot.xml` / env-wiring files, no
reinstall is needed — robocasa is installed **editable**, so changes take effect
on the next run.

---

## Troubleshooting

- **`uv: command not found`** — install uv (Prerequisites #2), re-open the shell.
- **`allex_model meshes not found`** — clone `allex_model`, or point the installer
  at it: `ALLEX_MODEL_DIR=/path/to/allex_model run_scripts/eval/robocasa_allex/setup_sim.sh`.
- **mesh symlink broke after moving `allex_model`** — just re-run `setup_sim.sh`
  (it recreates the symlink), or manually:
  `ln -sfn $ALLEX_MODEL_DIR/meshes external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/models/assets/robots/allex/meshes`.
- **Blank / black mp4, or `MUJOCO_GL` errors** — for headless use `MUJOCO_GL=egl`;
  for a window use `--viewer` with a valid `DISPLAY`. Don't mix them.
- **`robosuite WARNING: No private macro file` / `mimicgen not imported` /
  `mink IK` / `robosuite_models`** — harmless. None of them are used by the ALLEX
  sim path.
- **A `torch` (+ tensorboard/triton) install shows up** — that's `tianshou==0.5.1`,
  pinned by RoboCasa's own `setup.py`, pulling it in transitively. It's the CPU
  build and is unrelated to the RLDX-1 model; the sim path doesn't use it.
- **Asset download interrupted** — just re-run `setup_sim.sh`; it skips what's
  already downloaded.
