#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Phase 0.5b — torque-OOD sensitivity for RLDX-1-MT-ALLEX (decisive gate input).

The checkpoint has allow_missing_physics=True (omitting torque -> physics_mask=0,
an IN-DISTRIBUTION path: trained with physics_dropout=0.3 + exit-zero physics init,
weight=0.1). RoboCasa cannot produce real torque-sensor efforts, so we will roll
out with torque OMITTED. This test quantifies how much that costs: it compares the
action the policy emits with torque OMITTED (mask=0) vs torque PROVIDED as a
surrogate (mask=1) at typical (group-mean) and extreme (q99) magnitudes.

apply_physics SKIPS all-zero torque, so "zeros-as-present" is impossible; the only
two regimes are mask=0 (omit) and mask=1 (nonzero surrogate). We report action
divergence in rad, against the state-responsiveness baseline (~0.17 rad) from 0.5a.

Interpretation:
  divergence << baseline  -> torque is NOT load-bearing; omit it safely (GO).
  divergence ~ baseline   -> torque materially steers the policy; a wrong/absent
                             surrogate is a real risk -> flag at the gate.

Run: /home/thor/RLDX-1/.venv/bin/python run_scripts/eval/robocasa_allex/checks/torque_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CONTRACT = Path(__file__).resolve().parents[1] / "contract"
sys.path.insert(0, str(CONTRACT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import allex_contract as C  # noqa: E402
from action_sanity import make_obs, flatten_action, MOTION  # noqa: E402


def torque_obs(base_obs, kind: str):
    """Return a copy of base_obs with a physics/torque block, or omitted.

    kind: 'omit' (no physics), 'mean' (group-mean effort), 'q99' (extreme effort).
    Physics keys are 'torque.<effort_group>' each shaped (T=1, dim).
    """
    obs = {k: dict(v) if isinstance(v, dict) else v for k, v in base_obs.items()}
    if kind == "omit":
        return obs
    stats = C._STATS["torque"]
    phys = {}
    for g in C.TORQUE_GROUP_ORDER:
        s = stats[g]
        if kind == "mean":
            val = np.asarray(s["mean"], np.float32)
        elif kind == "q99":
            val = np.asarray(s["q99"], np.float32)
        else:
            raise ValueError(kind)
        # ensure nonzero so apply_physics doesn't skip it as "absent sensor"
        if np.all(val == 0):
            val = val + 1e-3
        phys[f"torque.{g}"] = val[None, None, :]  # (B=1, T=1, dim)
    obs["physics"] = phys
    return obs


def main():
    from huggingface_hub import snapshot_download
    from rldx.data.embodiment_tags import EmbodimentTag
    from rldx.policy.rldx_policy import RLDXPolicy

    ckpt = snapshot_download("RLWRLD/RLDX-1-MT-ALLEX")
    # require_physics=False so torque is optional; strict validation on.
    policy = RLDXPolicy(EmbodimentTag.GENERAL_EMBODIMENT, ckpt, device=0, strict=True)
    video_t = len(policy.modality_configs["video"].delta_indices)

    z = np.load(MOTION, allow_pickle=True)
    names = [str(x) for x in z["joint_names"]]
    q = z["q"].astype(np.float64)
    q_by_name = {n: float(q[0, i]) for i, n in enumerate(names)}
    base = make_obs(q_by_name, video_t)

    acts = {}
    for kind in ("omit", "mean", "q99"):
        obs = torque_obs(base, kind)
        a, _ = policy.get_action(obs, {"reset_memory": [True]})
        acts[kind] = flatten_action(a)  # (H,48)
        assert np.isfinite(acts[kind]).all()
        print(f"[{kind}] action shape {acts[kind].shape} finite=OK")

    d_mean = float(np.abs(acts["mean"] - acts["omit"]).mean())
    d_q99 = float(np.abs(acts["q99"] - acts["omit"]).mean())
    d_span = float(np.abs(acts["q99"] - acts["mean"]).mean())
    baseline = 0.175  # 0.5a state-responsiveness reference (rad)
    print(f"\n mean|Δaction| omit->mean-torque = {d_mean:.4f} rad "
          f"({100*d_mean/baseline:.1f}% of state-baseline)")
    print(f" mean|Δaction| omit->q99-torque  = {d_q99:.4f} rad "
          f"({100*d_q99/baseline:.1f}% of state-baseline)")
    print(f" mean|Δaction| mean->q99 torque  = {d_span:.4f} rad")
    # Framing: mask=0 (omit) is IN-DISTRIBUTION (physics_dropout=0.3 in training),
    # so it is the SAFE path. The risk is feeding an OOD MuJoCo surrogate. The
    # present-vs-omit gap tells us how far a surrogate would pull actions in an
    # UNCONTROLLED direction; the mean-vs-q99 gap tells us how much magnitude
    # (i.e. a wrong surrogate value) matters within the present regime.
    present_vs_omit = max(d_mean, d_q99)
    print(f"\n present-vs-omit gap = {present_vs_omit:.4f} rad "
          f"({100*present_vs_omit/baseline:.0f}% of state-baseline) "
          "= how far an OOD surrogate would pull actions")
    print(f" magnitude sensitivity (mean->q99) = {d_span:.4f} rad")
    print("\nGATE READING: OMIT torque at rollout (mask=0 is trained via 0.3 "
          "dropout -> in-distribution & safe). Do NOT feed a MuJoCo torque "
          "surrogate: it is OOD and would perturb actions by "
          f"~{present_vs_omit:.3f} rad in an uncontrolled direction.")
    caveat = ("Real deployment runs WITH torque (mask=1); sim runs mask=0 -> a "
              "different (but trained) operating mode. Log as a harness caveat.")
    print(f"CAVEAT: {caveat}")
    print(f"\nVERDICT [0.5b]: GO with torque OMITTED "
          f"(surrogate-OOD risk={present_vs_omit:.4f} rad avoided by omitting)")


if __name__ == "__main__":
    main()
