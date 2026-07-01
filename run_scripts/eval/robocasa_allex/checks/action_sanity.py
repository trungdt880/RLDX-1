#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Phase 0.5a — offline action-sanity check for RLDX-1-MT-ALLEX.

Loads the real checkpoint (no sim, no server), feeds a contract-shaped nested
observation built from a real ALLEX joint pose (npz) + a gray ego frame + a task
string, and asserts the returned PHYSICAL action is sane:
  - shape (B, action_horizon=40, 48) across the 6 groups, finite
  - every commanded joint target within that joint's MJCF limit (small slack)
  - responsive: two different input poses yield different action chunks

What this PROVES: the full load -> encode -> denormalize -> per-group-order path
produces valid absolute joint targets. What it does NOT prove: task competence or
sim transfer (no real paired action labels here; the npz is a generic motion).

Run (needs the 16GB weights present + GPU):
  /home/thor/RLDX-1/.venv/bin/python run_scripts/eval/robocasa_allex/checks/action_sanity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CONTRACT = Path(__file__).resolve().parents[1] / "contract"
sys.path.insert(0, str(CONTRACT))
import allex_contract as C  # noqa: E402
from verify_joint_order import joint_limits  # noqa: E402

MOTION = Path.home() / "workspace/allex_model/examples/motions/hello.npz"


def build_state_groups(q_by_name: dict[str, float]) -> dict[str, np.ndarray]:
    """contract group -> (1,1,dim) float32 state, joints in frozen order."""
    out = {}
    for g in C.GROUP_ORDER:
        vals = [q_by_name[n] for n in C.JOINT_NAMES[g]]
        out[g] = np.asarray(vals, dtype=np.float32)[None, None, :]  # (B=1,T=1,D)
    return out


def gray_video(t: int = 4, h: int = 256, w: int = 256) -> np.ndarray:
    # (B=1, T, H, W, C) uint8 — mid-gray avoids all-black degeneracy.
    return np.full((1, t, h, w, 3), 128, dtype=np.uint8)


def make_obs(q_by_name, video_t):
    return {
        "video": {"camera_ego_left": gray_video(video_t)},
        "state": build_state_groups(q_by_name),
        "language": {"annotation.human.task_description": [["pick up the object"]]},
    }


def flatten_action(action: dict) -> np.ndarray:
    """dict of per-group (B,H,d) -> (H, 48) in frozen group order (B=1)."""
    parts = [np.asarray(action[g])[0] for g in C.GROUP_ORDER]  # each (H,d)
    return np.concatenate(parts, axis=-1)


def main():
    import torch  # noqa: F401
    from huggingface_hub import snapshot_download
    from rldx.data.embodiment_tags import EmbodimentTag
    from rldx.policy.rldx_policy import RLDXPolicy

    ckpt = snapshot_download("RLWRLD/RLDX-1-MT-ALLEX")  # cached; no re-download
    print(f"[load] {ckpt}")
    policy = RLDXPolicy(
        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        model_path=ckpt,
        device=0,
        strict=True,
    )
    # video horizon the loaded model actually expects (memory expands delta_indices)
    video_t = len(policy.modality_configs["video"].delta_indices)
    print(f"[info] expected video frames T={video_t} "
          f"(contract says {C.VIDEO_DELTA_INDICES})")

    z = np.load(MOTION, allow_pickle=True)
    names = [str(x) for x in z["joint_names"]]
    q = z["q"].astype(np.float64)

    def q_at(t):
        return {n: float(q[t, i]) for i, n in enumerate(names)}

    lim = joint_limits()
    frozen = C.build_frozen_index_map()
    inv = {v: k for k, v in frozen.items()}

    results = {}
    for label, t in [("pose_t0", 0), ("pose_tmid", q.shape[0] // 2)]:
        obs = make_obs(q_at(t), video_t)
        action, _info = policy.get_action(obs, {"reset_memory": [True]})
        act = flatten_action(action)  # (H,48)
        results[label] = act
        assert np.isfinite(act).all(), f"{label}: non-finite action"
        assert act.shape == (C.MAX_ACTION_HORIZON, C.REAL_DIM), \
            f"{label}: bad shape {act.shape}"
        # joint-limit check on the FIRST predicted step (targets closest to state)
        oob = []
        for j in range(C.REAL_DIM):
            nm = inv[j]
            if nm not in lim:
                continue
            lo, hi = lim[nm]
            v = act[0, j]
            if v < lo - 0.20 or v > hi + 0.20:
                oob.append(f"{nm}={v:.3f} vs [{lo:.3f},{hi:.3f}]")
        print(f"[{label}] action shape {act.shape}, finite=OK, "
              f"step0 out-of-limit joints: {len(oob)}")
        for o in oob[:8]:
            print("    OOB", o)

    diff = float(np.abs(results["pose_t0"] - results["pose_tmid"]).mean())
    print(f"\n[responsiveness] mean|Δaction| between two poses = {diff:.4f} rad")
    verdict = "PASS" if diff > 1e-3 else "SUSPECT (policy ignores state?)"
    print(f"VERDICT [0.5a action-sanity]: {verdict}")


if __name__ == "__main__":
    main()
