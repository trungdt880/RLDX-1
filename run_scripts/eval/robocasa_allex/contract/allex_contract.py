#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
FROZEN ALLEX (general_embodiment) observation/action contract — Phase 0.

Authoritative, executable spec of the RLDX-1-MT-ALLEX checkpoint's
state/action layout + normalization. Downstream RoboCasa sim / client code
MUST import this module rather than re-deriving order or stats.

Run directly for a self-test:
    /home/thor/RLDX-1/.venv/bin/python allex_contract.py

Sources (all verified against the checkpoint JSONs + RLDX source):
  - group concat order  : processor_config.json
                          .processor_kwargs.modality_configs.general_embodiment
                          .state.modality_keys   (== action.modality_keys)
                          == rldx/policy/policy_runtime.py:48-55
  - per-group dims       : statistics.json['general_embodiment']['state'][group]
  - normalization        : rldx/data/state_action/state_action_processor.py:143-163
                           (_compute_normalization_parameters, use_percentiles=True
                            => min:=q01, max:=q99)
                           rldx/data/utils.py:75-154 (normalize/unnormalize minmax)
  - clip_outliers=True   : state_action_processor.py:262-263, 410-411
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# 1. FROZEN GROUP CONCAT ORDER + DIMS
# --------------------------------------------------------------------------
# The modality_keys list order IS the order groups are concatenated into the
# flat 48-vector (state, action, and torque all share this grouping/order).
GROUP_ORDER: list[str] = [
    "left_arm_joints",   # 7
    "left_hand_joints",  # 15
    "neck_joints",       # 2
    "right_arm_joints",  # 7
    "right_hand_joints", # 15
    "waist_joints",      # 2
]
GROUP_DIMS: dict[str, int] = {
    "left_arm_joints": 7,
    "left_hand_joints": 15,
    "neck_joints": 2,
    "right_arm_joints": 7,
    "right_hand_joints": 15,
    "waist_joints": 2,
}
TORQUE_GROUP_ORDER: list[str] = [
    "left_arm_effort",   # 7
    "left_hand_effort",  # 15
    "neck_effort",       # 2
    "right_arm_effort",  # 7
    "right_hand_effort", # 15
    "waist_effort",      # 2
]

REAL_DIM = 48            # sum(GROUP_DIMS)
MAX_STATE_DIM = 64       # processor: state zero-padded 48 -> 64
MAX_ACTION_DIM = 64      # processor: action zero-padded 48 -> 64
MAX_ACTION_HORIZON = 40  # model predicts 40 future steps
MEMORY_LENGTH = 4        # server buffers cognition tokens across calls
# Inference video frames the CLIENT must send per call (camera_ego_left),
# derived by policy_loader._apply_memory_config with video_length=4,
# video_stride=2 (default): [(i-3)*2 for i in range(4)].
VIDEO_DELTA_INDICES = [-6, -4, -2, 0]

# --------------------------------------------------------------------------
# 2. FROZEN INTRA-GROUP JOINT ORDERING  (ALLEX MJCF joint names, per index)
# --------------------------------------------------------------------------
# CONFIDENCE: STRONGLY SUPPORTED (not byte-exact verified). No in-repo per-joint
# name list exists for general_embodiment. This ordering = per-group slice of the
# npz joint_names order, cross-checked by verify_joint_order.py: each channel's
# [q01,q99] percentile band must fit inside that joint's MJCF limit. Result:
# 500/500 random hand-permutations were strictly worse (uniquely-best), 0 gross
# arm/waist/neck violations. Two right-hand PIP channels (R_Middle_PIP,
# R_Ring_PIP) show wide/negative bands that NO permutation fixes -> a real-data
# quirk, not a mis-ordering. Confirm against real_allex meta/modality.json before
# trusting a live rollout NUMBER (goal is a harness, so this bar is acceptable).
JOINT_NAMES: dict[str, list[str]] = {
    "left_arm_joints": [
        "L_Shoulder_Pitch_Joint", "L_Shoulder_Roll_Joint", "L_Shoulder_Yaw_Joint",
        "L_Elbow_Joint", "L_Wrist_Yaw_Joint", "L_Wrist_Roll_Joint", "L_Wrist_Pitch_Joint",
    ],
    "left_hand_joints": [
        "L_Thumb_Yaw_Joint", "L_Thumb_CMC_Joint", "L_Thumb_MCP_Joint",
        "L_Index_ABAD_Joint", "L_Index_MCP_Joint", "L_Index_PIP_Joint",
        "L_Middle_ABAD_Joint", "L_Middle_MCP_Joint", "L_Middle_PIP_Joint",
        "L_Ring_ABAD_Joint", "L_Ring_MCP_Joint", "L_Ring_PIP_Joint",
        "L_Little_ABAD_Joint", "L_Little_MCP_Joint", "L_Little_PIP_Joint",
    ],
    "neck_joints": ["Neck_Pitch_Joint", "Neck_Yaw_Joint"],
    "right_arm_joints": [
        "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
        "R_Elbow_Joint", "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
    ],
    "right_hand_joints": [
        "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
        "R_Index_ABAD_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint",
        "R_Middle_ABAD_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
        "R_Ring_ABAD_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
        "R_Little_ABAD_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint",
    ],
    "waist_joints": ["Waist_Yaw_Joint", "Waist_Lower_Pitch_Joint"],
}

_STATS_PATH = Path(__file__).with_name("allex_stats.json")


def _load_stats() -> dict:
    with open(_STATS_PATH) as f:
        return json.load(f)["general_embodiment"]


_STATS = _load_stats()


def _params(modality: str, group: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) normalization bounds == (q01, q99) since use_percentiles=True.

    Mirrors StateActionProcessor._compute_normalization_parameters:
    with use_percentiles=True it sets min:=q01, max:=q99, then
    normalize_values_minmax maps [min,max] -> [-1,1].
    """
    st = _STATS[modality][group]
    lo = np.asarray(st["q01"], dtype=np.float64)
    hi = np.asarray(st["q99"], dtype=np.float64)
    return lo, hi


def normalize(group: str, raw: np.ndarray, modality: str = "state") -> np.ndarray:
    """raw joint values -> normalized [-1,1] (percentile minmax, clipped).

    Formula (rldx/data/utils.py:normalize_values_minmax with min:=q01,max:=q99):
        n = 2*(x - q01)/(q99 - q01) - 1 ; clip to [-1,1]
    Features with q99==q01 map to 0 (mask), matching RLDX.
    """
    lo, hi = _params(modality, group)
    raw = np.asarray(raw, dtype=np.float64)
    out = np.zeros_like(raw)
    mask = ~np.isclose(hi, lo)
    out[..., mask] = (raw[..., mask] - lo[mask]) / (hi[mask] - lo[mask])
    out[..., mask] = 2.0 * out[..., mask] - 1.0
    return np.clip(out, -1.0, 1.0)  # clip_outliers=True


def denormalize(group: str, norm: np.ndarray, modality: str = "action") -> np.ndarray:
    """normalized [-1,1] -> raw joint target (absolute radians).

    Inverse of normalize (rldx/data/utils.py:unnormalize_values_minmax):
        x = (clip(n,-1,1)+1)/2 * (q99 - q01) + q01
    NOTE: because q99==q01 features were mapped to 0 (not invertible), those
    channels return q01 (== q99); RLDX has the identical irreversibility.
    """
    lo, hi = _params(modality, group)
    norm = np.asarray(norm, dtype=np.float64)
    return (np.clip(norm, -1.0, 1.0) + 1.0) / 2.0 * (hi - lo) + lo


def build_frozen_index_map() -> dict[str, int]:
    """The authoritative sim joint-name -> flat-48-index map."""
    idx = 0
    out: dict[str, int] = {}
    for g in GROUP_ORDER:
        names = JOINT_NAMES.get(g)
        if names is None or len(names) != GROUP_DIMS[g]:
            raise RuntimeError(
                f"JOINT_NAMES['{g}'] not frozen / wrong length "
                f"(need {GROUP_DIMS[g]}); see allex_contract.md."
            )
        for n in names:
            out[n] = idx
            idx += 1
    assert idx == REAL_DIM
    return out


def assert_state_vector_order(joint_name_to_index: dict) -> None:
    """Hard-fail if a sim joint-name->index map disagrees with the frozen order.

    Raises AssertionError listing every mismatch.
    """
    frozen = build_frozen_index_map()
    errs = []
    for name, fidx in frozen.items():
        if name not in joint_name_to_index:
            errs.append(f"missing joint '{name}' (frozen idx {fidx})")
        elif joint_name_to_index[name] != fidx:
            errs.append(
                f"joint '{name}' at idx {joint_name_to_index[name]}, "
                f"frozen expects {fidx}"
            )
    extra = set(joint_name_to_index) - set(frozen)
    for name in sorted(extra):
        errs.append(f"unexpected joint '{name}' (idx {joint_name_to_index[name]})")
    if errs:
        raise AssertionError(
            "state vector order mismatch vs frozen ALLEX contract:\n  "
            + "\n  ".join(errs)
        )


def _self_test() -> None:
    rng = np.random.default_rng(0)
    max_abs = 0.0
    for modality in ("state", "action"):
        for g in GROUP_ORDER:
            lo, hi = _params(modality, g)
            # sample raw values strictly inside [q01, q99] so clipping is a no-op
            raw = lo + rng.random(lo.shape) * (hi - lo)
            n = normalize(g, raw, modality)
            back = denormalize(g, n, modality)
            live = ~np.isclose(hi, lo)  # invertible channels only
            err = np.max(np.abs(back[live] - raw[live])) if live.any() else 0.0
            max_abs = max(max_abs, err)
            assert (n >= -1.0 - 1e-9).all() and (n <= 1.0 + 1e-9).all()
    print(f"[normalize round-trip] max abs err (invertible channels) = {max_abs:.3e}")
    assert max_abs < 1e-9, "round-trip error too large"

    # order assertion self-check (only if JOINT_NAMES frozen)
    if all(g in JOINT_NAMES for g in GROUP_ORDER):
        fm = build_frozen_index_map()
        assert_state_vector_order(dict(fm))  # identity must pass
        bad = dict(fm)
        k0 = GROUP_ORDER[0]
        bad[JOINT_NAMES[k0][0]] = 999
        try:
            assert_state_vector_order(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("assert_state_vector_order failed to detect a mismatch")
        print(f"[order assertion] frozen 48-joint map OK; "
              f"tamper detection OK ({len(fm)} joints)")
    else:
        print("[order assertion] SKIPPED — JOINT_NAMES not yet frozen "
              "(see allex_contract.md, marked UNVERIFIED)")

    print("PASS")


if __name__ == "__main__":
    _self_test()
