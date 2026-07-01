#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Load a RoboCasa scene with ALLEX and replay a recorded motion — SMOOTHLY.

Drops ALLEX into a real robocasa scene and replays a baked trajectory from the
allex_model repo (`examples/motions/*.npz`). It reproduces the native
`allex_model/examples/mujoco/replay.py` driving scheme so the motion looks the
same as there (NOT laggy/shaking):
  * steps the sim ONCE per motion frame at the motion's native dt (not the env's
    20 Hz control loop, which would hold each target for 25 substeps);
  * gravity-compensation feed-forward (`qfrc_bias`) on all dofs;
  * velocity (error-derivative) feed-forward `+kv * q̇_ref` per driven joint;
  * ramps from the reset pose to the first frame (no teleport → coupled joints
    don't snap).
It drives `sim.data.ctrl` (ALLEX's own `<position>` actuators) directly — this is
a VISUAL replay, not the policy control path (that path is the 20 Hz env.step in
allex_rollout_client.py).

Two modes:
  * default : render a camera to .mp4 (or .gif).
  * --viewer: live interactive MuJoCo window (needs a display / DISPLAY set).

Run (robosuite venv):
  MUJOCO_GL=egl .../robocasa_uv/.venv/bin/python \
      run_scripts/eval/robocasa_allex/replay_in_scene.py --motion hello
  DISPLAY=:0 .../robocasa_uv/.venv/bin/python \
      run_scripts/eval/robocasa_allex/replay_in_scene.py --viewer
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_VIEWER = "--viewer" in sys.argv
os.environ.setdefault("MUJOCO_GL", "glfw" if _VIEWER else "egl")
if not _VIEWER:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
MOTION_DIR = Path.home() / "workspace/allex_model/examples/motions"


def _resolve_motion(name: str) -> Path:
    if os.path.sep in name or name.endswith(".npz"):
        return Path(name)
    return MOTION_DIR / f"{name}.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", default="hello", help="hello | demo1 | /abs/path.npz")
    ap.add_argument("--env", default="robocasa_allex/PnPCanToBowl_AllexRobot_Env")
    ap.add_argument("--viewer", action="store_true",
                    help="live interactive MuJoCo window (needs a display)")
    ap.add_argument("--camera", default="robot0_frontview",
                    help="robot0_frontview | robot0_agentview_center | "
                         "robot0_zed_left_camera_optical_frame (ego)  [file mode]")
    ap.add_argument("--robot-back", type=float, default=None,
                    help="override how far ALLEX mounts BACK from the counter, "
                         "in metres (e.g. 0.4). Larger = further back / more table "
                         "clearance. Default: the env's baked-in offset (-0.4).")
    ap.add_argument("--robot-height", type=float, default=None,
                    help="override ALLEX base height z in metres (default 0.9).")
    ap.add_argument("--fps", type=int, default=30, help="output/pacing fps")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed multiplier (2.0 = 2x faster)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default=str(HERE / "rollout" / "allex_replay.mp4"),
                    help=".mp4 (needs imageio-ffmpeg) or .gif")
    args = ap.parse_args()

    import gymnasium as gym
    import mujoco
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    # Optional runtime override of the ALLEX mount offset [x, y_depth, z] before
    # the env is built. y<0 = further back from the counter.
    if args.robot_back is not None or args.robot_height is not None:
        from robocasa.environments.tabletop import tabletop as _TT
        off = list(_TT._ROBOT_POS_OFFSETS["AllexRobot"])
        if args.robot_back is not None:
            off[1] = -abs(args.robot_back)
        if args.robot_height is not None:
            off[2] = args.robot_height
        _TT._ROBOT_POS_OFFSETS["AllexRobot"] = off
        print(f"[mount] ALLEX offset override -> {off}")

    sys.path.insert(0, str(HERE / "contract"))
    import allex_contract as C  # noqa: F401  (kept for parity / joint sanity)

    motion = _resolve_motion(args.motion)
    z = np.load(motion, allow_pickle=True)
    names = [str(x) for x in z["joint_names"]]
    q = z["q"].astype(np.float64)
    qd = z["qd"].astype(np.float64) if "qd" in z else np.zeros_like(q)
    dt = float(z["dt"])
    print(f"[motion] {motion.name}: {len(q)} frames, dt={dt}s ({len(q)*dt:.1f}s)")

    print(f"[env] gym.make({args.env})  mode={'viewer' if args.viewer else 'file'}")
    env = gym.make(args.env, enable_render=not args.viewer, seed=0)
    env.reset()
    base = env.unwrapped.env
    sim = base.sim
    model = sim.model._model
    data = sim.data._data
    model.opt.timestep = dt  # step per motion frame at native rate
    task = base.get_ep_meta().get("lang", "")
    print(f"[env] reset OK — task: {task!r}")

    # Map each recorded joint -> (actuator id, dof addr, qpos addr, kv). ALLEX
    # position actuator biasprm = [0, -kp, -kv]; robosuite prefixes names 'robot0_'.
    pf = base.robots[0].robot_model.naming_prefix  # 'robot0_'
    pairs, qadr = [], {}
    for jn in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{pf}{jn.replace('_Joint', '_Position')}")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{pf}{jn}")
        assert aid >= 0 and jid >= 0, jn
        kv = float(-model.actuator_biasprm[aid, 2])
        pairs.append((names.index(jn), aid, int(model.jnt_dofadr[jid]), kv))
        qadr[jn] = int(model.jnt_qposadr[jid])

    sd = mujoco.MjData(model)  # scratch for gravity comp at qvel=0

    def gravity_comp():
        sd.qpos[:] = data.qpos
        sd.qvel[:] = 0.0
        mujoco.mj_forward(model, sd)
        return sd.qfrc_bias

    def drive(frame_q, frame_qd):
        data.qfrc_applied[:] = gravity_comp()
        for c, aid, dof, kv in pairs:
            data.ctrl[aid] = frame_q[c]
            data.qfrc_applied[dof] += kv * frame_qd[c]     # error-term D via +kv*q̇_ref
        mujoco.mj_step(model, data)

    # ramp reset-pose -> first frame (no teleport; coupled joints stay stable)
    home = np.array([data.qpos[qadr[n]] for n in names])
    nwarm = max(1, int(round(0.5 / dt)))
    for k in range(nwarm):
        a = (k + 1) / nwarm
        drive(home * (1 - a) + q[0] * a, np.zeros(len(names)))

    render_every = max(1, round(1.0 / (args.fps * dt) * args.speed))
    print(f"[drive] gravity+velocity feed-forward, {dt*1000:.1f}ms steps, "
          f"render every {render_every} → ~{args.fps}fps @ {args.speed}x")

    if args.viewer:
        import mujoco.viewer
        print("[viewer] opening window — close it or Ctrl-C to stop.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for i in range(len(q)):
                if not viewer.is_running():
                    break
                t0 = time.time()
                drive(q[i], qd[i])
                if i % render_every == 0:
                    viewer.sync()
                    time.sleep(max(0.0, dt * render_every / args.speed - (time.time() - t0)))
        env.close()
        print("[done] viewer closed.")
        return

    import imageio.v2 as imageio
    print(f"[render] camera {args.camera!r} -> {args.out}")
    vid = []
    for i in range(len(q)):
        drive(q[i], qd[i])
        if i % render_every == 0:
            img = sim.render(width=args.width, height=args.height,
                             camera_name=args.camera)[::-1]
            vid.append(np.ascontiguousarray(img, dtype=np.uint8))
    env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".mp4":
        imageio.mimsave(out, vid, fps=args.fps, quality=8, macro_block_size=None)
    else:
        imageio.mimsave(out, vid, fps=args.fps, loop=0)
    print(f"\n[done] wrote {len(vid)} frames -> {out}  ({args.width}x{args.height}, "
          f"{args.fps}fps). Smooth replay of '{motion.stem}' in the scene.")


if __name__ == "__main__":
    main()
