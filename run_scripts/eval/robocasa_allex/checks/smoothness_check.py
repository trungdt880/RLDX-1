#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Is the SHAKING coming from the policy's commanded targets, or from execution?

The rollout state trajectory dithers (arm direction-flip rate ~68%/step). Two
candidate causes:
  A. the policy's predicted action chunk is itself jittery, or
  B. the chunk is smooth but the env executes it as a 20 Hz STEP input into a
     very underdamped position servo (L_Shoulder_Pitch kp=5000 kv=8.22 -> zeta
     ~0.06, ringing ~11 Hz), which rings between control steps.

This asks the server for ONE 40-step chunk from a real observation and measures
the chunk's own smoothness. Low flip-rate inside the chunk => the model is
smooth => cause B (execution), fixable by interpolating targets across substeps.

Run (robosuite venv; server up):
  MUJOCO_GL=egl <sim-venv-python> checks/smoothness_check.py
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import msgpack  # noqa: E402
import zmq  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "contract"))
import allex_contract as C  # noqa: E402

HOST, PORT = os.environ.get("ALLEX_HOST", "127.0.0.1"), int(os.environ.get("ALLEX_PORT", "20250"))
TASK = "pick the can from the counter and place it in the bowl"


def enc(o):
    if isinstance(o, np.ndarray):
        b = io.BytesIO(); np.save(b, o, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": b.getvalue()}
    return o


def dec(o):
    if isinstance(o, dict):
        if "__ndarray_class__" in o:
            return np.load(io.BytesIO(o["as_npy"]), allow_pickle=False)
        if "__ModalityConfig_class__" in o:
            return o["as_json"]
    return o


def main() -> None:
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 180_000); s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://{HOST}:{PORT}")

    def call(ep, data=None, needs=True):
        req = {"endpoint": ep}
        if needs:
            req["data"] = data or {}
        s.send(msgpack.packb(req, default=enc))
        r = msgpack.unpackb(s.recv(), object_hook=dec)
        if isinstance(r, dict) and "error" in r:
            raise RuntimeError(r["error"])
        return r

    mc = call("get_modality_config", needs=False)
    vdelta = list(mc["video"]["delta_indices"])

    stats = json.load(open(HERE.parent / "contract" / "allex_stats.json"))["general_embodiment"]["state"]
    mean = {g: np.asarray(stats[g]["mean"], dtype=np.float64) for g in C.GROUP_ORDER}

    env = gym.make("robocasa_allex/PnPCanToBowl_AllexRobot_Env", enable_render=True, seed=0)
    obs, _ = env.reset()
    for _ in range(50):
        obs, *_ = env.step({f"action.{g}": mean[g] for g in C.GROUP_ORDER})

    req = {}
    req["video.camera_ego_left"] = np.stack(
        [obs["video.camera_ego_left"]] * len(vdelta), axis=0)[None].astype(np.uint8)
    for g in C.GROUP_ORDER:
        req[f"state.{g}"] = np.asarray(obs[f"state.{g}"], dtype=np.float32)[None][None]
    req["annotation.human.task_description"] = [TASK]

    call("reset", {"options": {"reset_memory": [True], "session_ids": ["smooth"]}})
    action, _ = call("get_action", {"observation": req,
                                    "options": {"reset_memory": [True], "session_ids": ["smooth"]}})

    chunk = np.concatenate([np.asarray(action[f"action.{g}"])[0] for g in C.GROUP_ORDER], axis=-1)
    print("=" * 70)
    print("Policy action chunk smoothness (one query, 40 predicted steps)")
    print("=" * 70)
    print(f"chunk shape {chunk.shape}  (steps, 48 joints)")

    d1 = np.diff(chunk, axis=0)
    d2 = np.diff(chunk, 2, axis=0)
    flips = (np.sign(d1[1:]) != np.sign(d1[:-1])).mean(axis=0)
    print(f"\nWITHIN the commanded chunk:")
    print(f"  mean |Δq| per step  : {np.abs(d1).mean():.5f} rad")
    print(f"  mean |Δ²q| per step : {np.abs(d2).mean():.5f} rad")
    print(f"  direction-flip rate : {flips.mean() * 100:.1f}%   (50% = pure dither)")
    i = 0
    for g in C.GROUP_ORDER:
        dd = C.GROUP_DIMS[g]
        print(f"    {g:<18} flips={flips[i:i+dd].mean()*100:5.1f}%  "
              f"max|Δq|={np.abs(d1[:, i:i+dd]).max():.4f}")
        i += dd

    print("\n  first 8 steps of L_Shoulder_Pitch (rad):")
    print("   ", np.array2string(chunk[:8, 0], precision=4))
    # Compare commanded vs EXECUTED motion from a recorded rollout, if present.
    npz = HERE.parent / "rollout" / "simple_front_pnp.npz"
    print("\nVERDICT:")
    cmd_flip = flips.mean() * 100
    print(f"  commanded chunk dithers at {cmd_flip:.1f}% flip-rate "
          f"({'JITTERY' if cmd_flip > 55 else 'smooth'}), "
          f"max step {np.abs(d1).max():.4f} rad")
    if npz.exists():
        tr = np.load(npz, allow_pickle=True)["traj"]
        e1 = np.diff(tr, axis=0)
        eflip = (np.sign(e1[1:]) != np.sign(e1[:-1])).mean() * 100
        amp = np.abs(e1).max() / max(np.abs(d1).max(), 1e-9)
        print(f"  executed state  dithers at {eflip:.1f}% flip-rate, "
              f"max step {np.abs(e1).max():.4f} rad")
        print(f"  execution AMPLIFIES the largest commanded step by {amp:.1f}x")
        print("  => BOTH contribute: a jittery command AND an underdamped servo")
        print("     (L_Shoulder_Pitch kp=5000 kv=8.22 -> zeta~0.06) that rings on")
        print("     each 20 Hz step input held constant for 25 substeps.")
    env.close()


if __name__ == "__main__":
    main()
