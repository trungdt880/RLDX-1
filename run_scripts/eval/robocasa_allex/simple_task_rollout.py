#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Minimal single-instruction rollout: can the policy move ALLEX on command?

The smallest useful closed-loop test of `RLDX-1-MT-ALLEX` in the RoboCasa ALLEX
env: give ONE short natural-language instruction ("raise your left hand"), let
the policy drive, and measure what actually moved. Deliberately narrow -- no
object interaction, no task success, just "does the model produce coherent,
in-range, instruction-responsive motion".

Beyond the rollout it also VERIFIES the live wire contract against the server's
advertised modality config (video/state/action delta indices + key sets), so a
mismatch fails loudly instead of silently degrading motion.

Two things it does that the generic client does not:
  * ``--init-pose train_mean`` starts ALLEX at the per-joint MEAN of the
    checkpoint's own training statistics. The env's default reset is all-zeros,
    which is ~31% out-of-distribution (e.g. elbows straight at 0.0 rad vs a
    training band of [-2.5,-0.8]) and would make the policy extrapolate from
    step 1. See checks/io_alignment_check.py.
  * per-step telemetry: palm heights (L/R), per-group motion, action ranges,
    plus ego + third-person video.

Run (robosuite venv; server must be up):
  MUJOCO_GL=egl <sim-venv-python> simple_task_rollout.py \
      --task "raise your left hand" --steps 120 --tag left
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import msgpack  # noqa: E402
import numpy as np  # noqa: E402
import zmq  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "contract"))
import allex_contract as C  # noqa: E402


# ----------------------------------------------------------------- wire layer
class Client:
    def __init__(self, host: str, port: int, timeout_ms: int = 180_000):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{host}:{port}")
        self.addr = f"tcp://{host}:{port}"

    @staticmethod
    def _enc(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            b = io.BytesIO()
            np.save(b, o, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": b.getvalue()}
        return o

    @staticmethod
    def _dec(o: dict) -> Any:
        if isinstance(o, dict):
            if "__ndarray_class__" in o:
                return np.load(io.BytesIO(o["as_npy"]), allow_pickle=False)
            if "__ModalityConfig_class__" in o:
                return o["as_json"]
        return o

    def call(self, endpoint: str, data: dict | None = None, needs_input: bool = True) -> Any:
        req: dict = {"endpoint": endpoint}
        if needs_input:
            req["data"] = data or {}
        self.sock.send(msgpack.packb(req, default=self._enc))
        try:
            raw = self.sock.recv()
        except zmq.error.Again:
            raise RuntimeError(f"no reply from {self.addr}; is the server past model-load?")
        rep = msgpack.unpackb(raw, object_hook=self._dec)
        if isinstance(rep, dict) and "error" in rep:
            raise RuntimeError(f"server error: {rep['error']}")
        return rep


# --------------------------------------------------- live contract validation
WALL_Y = 0.02                          # room back wall (wall_room_g0) world y
EXPECT_VIDEO_DELTA = [-6, -4, -2, 0]   # policy_loader._compute_inference_video_delta_indices(4,2)
EXPECT_STATE_DELTA = [0]
EXPECT_HORIZON = 40


def verify_contract(mc: dict) -> dict:
    """Assert the SERVED modality config is what the client is built for."""
    v, s, a, lang = (mc[k] for k in ("video", "state", "action", "language"))
    got = {
        "video_keys": list(v["modality_keys"]),
        "video_delta": list(v["delta_indices"]),
        "state_keys": list(s["modality_keys"]),
        "state_delta": list(s["delta_indices"]),
        "action_keys": list(a["modality_keys"]),
        "action_delta": list(a["delta_indices"]),
        "lang_keys": list(lang["modality_keys"]),
    }
    print("[contract] server advertises:")
    print(f"    video : keys={got['video_keys']} delta={got['video_delta']}")
    print(f"    state : keys={got['state_keys']} delta={got['state_delta']}")
    print(f"    action: keys={got['action_keys']} horizon={len(got['action_delta'])}")
    print(f"    lang  : keys={got['lang_keys']}")

    problems = []
    if got["video_delta"] != EXPECT_VIDEO_DELTA:
        problems.append(f"video delta {got['video_delta']} != {EXPECT_VIDEO_DELTA}")
    if got["state_delta"] != EXPECT_STATE_DELTA:
        problems.append(f"state delta {got['state_delta']} != {EXPECT_STATE_DELTA}")
    if len(got["action_delta"]) != EXPECT_HORIZON:
        problems.append(f"action horizon {len(got['action_delta'])} != {EXPECT_HORIZON}")
    if got["state_keys"] != list(C.GROUP_ORDER):
        problems.append(f"state keys {got['state_keys']} != contract {list(C.GROUP_ORDER)}")
    if got["action_keys"] != list(C.GROUP_ORDER):
        problems.append(f"action keys {got['action_keys']} != contract {list(C.GROUP_ORDER)}")
    if got["video_keys"] != ["camera_ego_left"]:
        problems.append(f"video keys {got['video_keys']} != ['camera_ego_left']")
    if got["lang_keys"] != ["annotation.human.task_description"]:
        problems.append(f"language keys {got['lang_keys']} unexpected")
    if problems:
        raise SystemExit("[contract] MISMATCH:\n  - " + "\n  - ".join(problems))
    print("[contract] OK — served config matches the frozen ALLEX contract.")
    return got


# ------------------------------------------------------------- in-dist pose
def training_mean_pose() -> dict[str, np.ndarray]:
    stats = json.load(open(HERE / "contract" / "allex_stats.json"))["general_embodiment"]["state"]
    return {g: np.asarray(stats[g]["mean"], dtype=np.float64) for g in C.GROUP_ORDER}


def aim_front_camera(base, cam_name: str = "robot0_frontview",
                     az: float = 270.0, elev: float = -18.0, dist: float = 2.2) -> str:
    """Retarget a scene camera to look at ALLEX from the FRONT (robot + table).

    Every stock camera in this scene sits behind/above the robot and shows only
    its back. Rather than add a camera (impossible post-compile) or use a free
    ``mujoco.Renderer`` (which bypasses robosuite's texture pipeline and renders
    wrong materials), this re-points an existing camera and keeps rendering
    through ``sim.render``.

    ``cam_pos``/``cam_quat`` are in the camera's PARENT BODY frame
    (``mobilebase0_support`` here, not the world), so the world-space eye/target
    are converted into that frame.
    """
    import mujoco

    sim = base.sim
    model, data = sim.model._model, sim.data._data
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        return cam_name
    sim.forward()
    pf = base.robots[0].robot_model.naming_prefix
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pf}Base_Link")
    lookat = data.xpos[bid] + np.array([0.0, 0.35, 0.55])

    a, e = np.radians(az), np.radians(elev)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    eye = lookat - dist * fwd
    # The room's back wall is at y~+0.02. Moving the robot forward pushes the eye
    # past it and the view becomes the inside of a wall. Shrink the distance so
    # the camera always stays inside the room.
    if eye[1] > WALL_Y - 0.15:
        room = (WALL_Y - 0.15 - lookat[1]) / max(-fwd[1], 1e-6)
        dist = max(0.8, min(dist, room))
        eye = lookat - dist * fwd

    f = lookat - eye
    f /= np.linalg.norm(f)
    x = np.cross(f, np.array([0.0, 0.0, 1.0]))
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    Rw = np.column_stack([x, y, -f])          # camera looks along its -z

    pb = model.cam_bodyid[cid]
    Rb = data.xmat[pb].reshape(3, 3)
    model.cam_pos[cid] = Rb.T @ (eye - data.xpos[pb])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (Rb.T @ Rw).flatten())
    model.cam_quat[cid] = q
    sim.forward()
    return cam_name


def robot_dofs(base) -> np.ndarray:
    """DOF indices belonging to the ROBOT only.

    ``data.qvel`` spans the whole sim, including the free-floating can/bowl/
    vegetables that are still settling after a reset. Testing "is it quiet" over
    all of qvel therefore never converges and says nothing about the robot.
    """
    import mujoco

    model = base.sim.model._model
    pf = base.robots[0].robot_model.naming_prefix
    idx = []
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith(pf):
            n = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}.get(
                model.jnt_type[j], 1)
            idx.extend(range(int(model.jnt_dofadr[j]), int(model.jnt_dofadr[j]) + n))
    return np.asarray(idx, dtype=int)


def apply_init_pose(base, pose: dict[str, np.ndarray], settle_s: float = 0.6,
                    quiet_tol: float = 5e-3, max_s: float = 4.0):
    """Put ALLEX at `pose` (clipped to MJCF limits) and let the couplings settle."""
    import mujoco

    sim = base.sim
    model, data = sim.model._model, sim.data._data
    pf = base.robots[0].robot_model.naming_prefix  # 'robot0_'
    for g in C.GROUP_ORDER:
        for j, jn in enumerate(C.JOINT_NAMES[g]):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{pf}{jn}")
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    f"{pf}{jn.replace('_Joint', '_Position')}")
            if jid < 0 or aid < 0:
                continue
            lo, hi = model.jnt_range[jid]
            val = float(np.clip(pose[g][j], lo, hi))
            data.qpos[model.jnt_qposadr[jid]] = val
            data.ctrl[aid] = val
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Settle until the robot is genuinely QUIET, not just for a fixed duration:
    # the teleport + equality-coupling resolution rings for a while, and starting
    # inference on a moving robot is both a visible shake and an off-distribution
    # state. Step until max|qvel| drops below tol (bounded by max_s).
    dt = model.opt.timestep
    dofs = robot_dofs(base)
    nmin, nmax = int(settle_s / dt), int(max_s / dt)
    for i in range(nmax):
        mujoco.mj_step(model, data)
        if i >= nmin and float(np.abs(data.qvel[dofs]).max()) < quiet_tol:
            break
    vmax = float(np.abs(data.qvel[dofs]).max())

    # The teleport happened behind the controller's back: its stored target is
    # still the pre-teleport (reset) pose. With target interpolation enabled the
    # next set_goal would ramp FROM that stale pose, yanking the arm ~1.3 rad on
    # step 0. Re-sync each part controller to where the robot actually is.
    for ctrl in base.robots[0].composite_controller.part_controllers.values():
        if hasattr(ctrl, "reset_goal"):
            ctrl.reset_goal()
    return vmax, (i + 1) * dt


# ------------------------------------------------------------------ rollout
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="raise your left hand")
    p.add_argument("--env-name", default="robocasa_allex/PnPCanToBowl_AllexRobot_Env")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=20250)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--exec-horizon", type=int, default=8,
                   help="steps of the 40-step chunk to execute before re-querying")
    p.add_argument("--init-pose", choices=["train_mean", "zeros"], default="train_mean")
    p.add_argument("--settle-steps", type=int, default=120,
                   help="max env hold-steps after the init pose, stopping early once "
                        "the robot is quiet, so inference never starts mid-shake. "
                        "20 was too few: contact-induced buzz needs ~120 steps to "
                        "decay from ~0.7 to <0.05 rad/s (checks/mount_tradeoff.py)")
    p.add_argument("--action-smooth", type=float, default=0.0,
                   help="EMA factor in [0,1) applied across the executed action "
                        "sequence. The policy's own chunk dithers (~63%% direction-"
                        "flip rate, +-0.01-0.02 rad); 0.5-0.7 removes that jitter. "
                        "0 = raw policy output.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="run")
    p.add_argument("--out-dir", default=str(HERE / "rollout"))
    p.add_argument("--camera", default="robot0_frontview")
    p.add_argument("--front-view", action="store_true", default=True,
                   help="re-aim the third-person camera to show the robot's FRONT + table")
    p.add_argument("--no-front-view", dest="front_view", action="store_false")
    p.add_argument("--front-az", type=float, default=270.0)
    p.add_argument("--front-elev", type=float, default=-18.0)
    p.add_argument("--front-dist", type=float, default=2.2)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cl = Client(args.host, args.port)
    print(f"[client] ping -> {cl.call('ping', needs_input=False)}")
    got = verify_contract(cl.call("get_modality_config", needs_input=False))

    import gymnasium as gym
    import mujoco
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    print(f"[env] {args.env_name}")
    env = gym.make(args.env_name, enable_render=True, seed=args.seed)
    obs, _ = env.reset()
    base = env.unwrapped.env
    model, data = base.sim.model._model, base.sim.data._data
    pf = base.robots[0].robot_model.naming_prefix

    if args.init_pose == "train_mean":
        print("[pose] initializing at the checkpoint's training MEAN pose (in-distribution)")
        pose = training_mean_pose()
        vmax, tset = apply_init_pose(base, pose)
        # hold through the NORMAL env path until quiet as well (obs must also be a
        # real GROOT obs), so the first policy query sees a static robot.
        rdofs = robot_dofs(base)
        for _ in range(args.settle_steps):
            obs, *_ = env.step({f"action.{g}": pose[g] for g in C.GROUP_ORDER})
            if float(np.abs(data.qvel[rdofs]).max()) < 5e-3:
                break
        print(f"[pose] settled in {tset:.2f}s (raw, robot max|qvel|={vmax:.5f}) -> "
              f"{float(np.abs(data.qvel[rdofs]).max()):.5f} rad/s before first query")

    if args.front_view:
        aim_front_camera(base, args.camera, args.front_az, args.front_elev, args.front_dist)
        print(f"[camera] third-person '{args.camera}' re-aimed to FRONT "
              f"(az={args.front_az} elev={args.front_elev} dist={args.front_dist})")

    lpid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pf}L_Palm_Link")
    rpid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pf}R_Palm_Link")
    palm = lambda: (float(data.xpos[lpid][2]), float(data.xpos[rpid][2]))  # noqa: E731

    # ring buffer sized from the SERVED deltas
    span = lambda d: max(d) - min(d) + 1  # noqa: E731
    hist = max(span(got["video_delta"]), span(got["state_delta"])) + 1
    buf: deque = deque(maxlen=hist)
    for _ in range(hist):
        buf.append(obs)

    def build_request() -> dict:
        req: dict = {}
        for cam in got["video_keys"]:
            req[f"video.{cam}"] = np.stack(
                [buf[d - 1][f"video.{cam}"] for d in got["video_delta"]], axis=0
            )[None].astype(np.uint8, copy=False)
        for g in got["state_keys"]:
            req[f"state.{g}"] = np.stack(
                [buf[d - 1][f"state.{g}"] for d in got["state_delta"]], axis=0
            )[None].astype(np.float32, copy=False)
        for lk in got["lang_keys"]:
            req[lk] = [args.task]
        return req

    sid = f"simple-{args.tag}"
    cl.call("reset", {"options": {"reset_memory": [True], "session_ids": [sid]}})
    print(f"[task] {args.task!r}   init_pose={args.init_pose}  steps={args.steps}")

    lz0, rz0 = palm()
    print(f"[start] palm height L={lz0:.4f} R={rz0:.4f} m")

    ego, third, traj, palms, lats = [], [], [], [], []
    step, first, ema = 0, True, None
    q0 = np.concatenate([np.asarray(obs[f"state.{g}"]).ravel() for g in C.GROUP_ORDER])

    while step < args.steps:
        t0 = time.perf_counter()
        action, _ = cl.call("get_action", {
            "observation": build_request(),
            "options": {"reset_memory": [first], "session_ids": [sid]},
        })
        lats.append(time.perf_counter() - t0)
        first = False

        n = min(args.exec_horizon, min(np.asarray(v).shape[1] for v in action.values()))
        for t in range(n):
            act = {k: np.asarray(v)[0, t] for k, v in action.items()}
            if args.action_smooth > 0.0:
                # EMA across executed targets; carries over chunk boundaries so
                # re-queries don't reintroduce a jump.
                a = args.action_smooth
                if ema is None:
                    ema = {k: v.copy() for k, v in act.items()}
                else:
                    ema = {k: a * ema[k] + (1.0 - a) * act[k] for k in act}
                act = {k: v.copy() for k, v in ema.items()}
            obs, _r, term, trunc, _i = env.step(act)
            buf.append(obs)
            step += 1
            ego.append(np.asarray(obs["video.camera_ego_left"]).copy())
            third.append(base.sim.render(width=640, height=480, camera_name=args.camera)[::-1].copy())
            traj.append(np.concatenate([np.asarray(obs[f"state.{g}"]).ravel()
                                        for g in C.GROUP_ORDER]))
            palms.append(palm())
            if step >= args.steps or term or trunc:
                break
        lz, rz = palms[-1]
        print(f"  [step {step:>3}] lat={lats[-1]*1e3:6.0f}ms  palm L={lz:.4f} R={rz:.4f}  "
              f"(ΔL={lz-lz0:+.4f} ΔR={rz-rz0:+.4f})  finite={np.isfinite(data.qpos).all()}")
        if term or trunc:
            break

    # ------------------------------------------------------------- analysis
    traj = np.asarray(traj)
    palms = np.asarray(palms)
    dq = traj - q0
    print("\n" + "=" * 68)
    print(f"RESULT — task {args.task!r}")
    print("=" * 68)
    print(f"  steps={step}  server calls={len(lats)}  latency mean={np.mean(lats)*1e3:.0f}ms")
    print(f"  palm height  L: {lz0:+.4f} -> {palms[-1,0]:+.4f}  (Δ {palms[-1,0]-lz0:+.4f} m, "
          f"peak {palms[:,0].max()-lz0:+.4f})")
    print(f"  palm height  R: {rz0:+.4f} -> {palms[-1,1]:+.4f}  (Δ {palms[-1,1]-rz0:+.4f} m, "
          f"peak {palms[:,1].max()-rz0:+.4f})")
    print("  per-group motion (max |Δangle| from start, rad):")
    i = 0
    for g in C.GROUP_ORDER:
        d = C.GROUP_DIMS[g]
        print(f"    {g:<18} {np.abs(dq[:, i:i+d]).max():.3f}")
        i += d
    print(f"  sim finite: {bool(np.isfinite(data.qpos).all())}   neq={int(model.neq)}")

    # ---------------------------------------------------------------- video
    try:
        import imageio.v2 as imageio
        for name, frames in (("ego", ego), ("third", third)):
            f = out / f"simple_{args.tag}_{name}.mp4"
            imageio.mimsave(f, [x.astype(np.uint8) for x in frames], fps=20,
                            quality=8, macro_block_size=None)
            print(f"  video: {f}")
    except Exception as e:  # noqa: BLE001
        print(f"  video encode failed: {e}")
    np.savez(out / f"simple_{args.tag}.npz", traj=traj, palms=palms,
             task=args.task, q0=q0)
    env.close()


if __name__ == "__main__":
    main()
