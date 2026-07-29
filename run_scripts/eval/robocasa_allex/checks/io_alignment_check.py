#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Offline I/O alignment audit: client assembly vs. what the server asserts.

Verifies, WITHOUT loading the 16 GB model, that everything the ALLEX rollout
client puts on the wire satisfies the server-side validators in
``rldx/policy/rldx_policy.py::RLDXSimPolicyWrapper.check_observation``, and that
the sim's reset state is inside the checkpoint's training distribution.

Checks
  1. obs keys/dtypes/shapes from the real env == contract keys.
  2. Client ring-buffer assembly reproduces the SERVED modality config
     (video delta [-6,-4,-2,0] -> (B,4,H,W,3) uint8; state delta [0] ->
     (B,1,D) float32; language list[str]) -- the exact asserts at
     rldx_policy.py:340-425.
  3. The 4 sampled video frames are DISTINCT after motion (a real strided
     window, not the same frame repeated).
  4. Sim reset pose vs. training [q01,q99] per joint -> how much of the state
     the model sees is out-of-distribution (clipped to +-1 by the normalizer).

Run (robosuite venv):
  MUJOCO_GL=egl <sim-venv-python> checks/io_alignment_check.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "contract"))
import allex_contract as C  # noqa: E402

# The SERVED config (policy_loader._apply_memory_config rewrites video deltas
# at load time: _compute_inference_video_delta_indices(video_length=4,
# video_stride=2) -> [-6,-4,-2,0]). Asserted against the live server elsewhere.
VIDEO_DELTA = [-6, -4, -2, 0]
STATE_DELTA = [0]
ENV_ID = os.environ.get("ALLEX_ENV_ID", "robocasa_allex/PnPCanToBowl_AllexRobot_Env")

ok = True


def check(cond: bool, label: str, detail: str = "") -> None:
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


def main() -> None:
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    print("=" * 74)
    print("ALLEX I/O alignment audit (client assembly vs server validators)")
    print("=" * 74)

    env = gym.make(ENV_ID, enable_render=True, seed=0)
    obs, _ = env.reset()

    # ---------------------------------------------------------------- 1. keys
    print("\n[1] env obs vs contract keys")
    state_keys = [f"state.{g}" for g in C.GROUP_ORDER]
    check("video.camera_ego_left" in obs, "video.camera_ego_left present")
    check(all(k in obs for k in state_keys), "all 6 state.<group> present")
    check("annotation.human.task_description" in obs, "language key present")
    ego = np.asarray(obs["video.camera_ego_left"])
    print(f"      ego frame: shape={ego.shape} dtype={ego.dtype}")
    for g in C.GROUP_ORDER:
        a = np.asarray(obs[f"state.{g}"])
        check(a.shape[-1] == C.GROUP_DIMS[g], f"state.{g} dim=={C.GROUP_DIMS[g]}",
              f"got {a.shape}")

    # ------------------------------------------ 2. client assembly -> asserts
    print("\n[2] client ring-buffer assembly vs server asserts")
    from collections import deque
    hist_len = max(max(VIDEO_DELTA) - min(VIDEO_DELTA) + 1,
                   max(STATE_DELTA) - min(STATE_DELTA) + 1) + 1
    buf: deque = deque(maxlen=hist_len)
    for _ in range(hist_len):
        buf.append(obs)
    # advance the sim so the frames actually differ
    zero = {f"action.{g}": np.zeros(C.GROUP_DIMS[g]) for g in C.GROUP_ORDER}
    for i in range(hist_len):
        a = dict(zero)
        # move the left arm a little so the ego view + state change over time
        a["action.left_arm_joints"] = np.array([0.0, 0.05 * i, 0, -0.1 * i, 0, 0, 0])
        obs, *_ = env.step(a)
        buf.append(obs)

    gather = lambda key, deltas: np.stack([buf[d - 1][key] for d in deltas], axis=0)  # noqa: E731
    vid = gather("video.camera_ego_left", VIDEO_DELTA)[None].astype(np.uint8, copy=False)
    print(f"      retained {hist_len} frames; sampling deltas {VIDEO_DELTA}")
    check(vid.ndim == 5, "video ndim==5 (B,T,H,W,C)", f"got {vid.shape}")
    check(vid.shape[1] == len(VIDEO_DELTA), f"video T=={len(VIDEO_DELTA)}", f"got {vid.shape[1]}")
    check(vid.shape[-1] == 3, "video C==3")
    check(vid.dtype == np.uint8, "video dtype uint8", f"got {vid.dtype}")

    for g in C.GROUP_ORDER:
        st = gather(f"state.{g}", STATE_DELTA)[None].astype(np.float32, copy=False)
        if g == C.GROUP_ORDER[0]:
            print(f"      state.{g}: {st.shape} {st.dtype}")
        assert st.ndim == 3 and st.shape[1] == len(STATE_DELTA)
    check(True, "all state (B,T,D) float32 with T==1")

    lang = [str(obs["annotation.human.task_description"])]
    check(isinstance(lang, list) and isinstance(lang[0], str), "language list[str]",
          f"{lang[0][:52]!r}")

    # -------------------------------------------- 3. frames actually distinct
    print("\n[3] strided video window carries real temporal signal")
    diffs = [float(np.mean(np.abs(vid[0, i].astype(np.int16) - vid[0, i + 1].astype(np.int16))))
             for i in range(vid.shape[1] - 1)]
    print(f"      mean|Δpixel| between consecutive sampled frames: "
          f"{[round(d, 2) for d in diffs]}")
    check(all(d > 0.0 for d in diffs), "the 4 sampled frames are distinct")

    # ------------------------------------ 4. reset pose vs training band
    print("\n[4] sim reset pose vs training state distribution [q01,q99]")
    stats = json.load(open(HERE.parent / "contract" / "allex_stats.json"))["general_embodiment"]["state"]
    env2 = gym.make(ENV_ID, enable_render=True, seed=0)
    obs0, _ = env2.reset()
    total = oob = 0
    worst = []
    for g in C.GROUP_ORDER:
        q01 = np.asarray(stats[g]["q01"], dtype=np.float64)
        q99 = np.asarray(stats[g]["q99"], dtype=np.float64)
        x = np.asarray(obs0[f"state.{g}"], dtype=np.float64).ravel()
        below, above = x < q01, x > q99
        n = int((below | above).sum())
        total += x.size
        oob += n
        for j in np.where(below | above)[0]:
            dist = float(q01[j] - x[j] if below[j] else x[j] - q99[j])
            worst.append((dist, f"{C.JOINT_NAMES[g][j]}", float(x[j]),
                          float(q01[j]), float(q99[j])))
        print(f"      {g:<18} out-of-band {n:>2}/{x.size}")
    worst.sort(reverse=True)
    print(f"      TOTAL out-of-band: {oob}/{total} joints "
          f"({100.0 * oob / total:.0f}% of the state vector)")
    if worst:
        print("      worst offenders (joint, sim_value, train q01, q99):")
        for d, name, v, a, b in worst[:6]:
            print(f"        {name:<26} sim={v:+.3f}  band=[{a:+.3f},{b:+.3f}]  off by {d:.3f} rad")
    # informational, not a hard gate
    print(f"      NOTE: out-of-band channels clip to ±1 after normalization.")

    env.close()
    env2.close()
    print("\n" + "-" * 74)
    print(f"VERDICT [I/O alignment]: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
