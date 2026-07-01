#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Verify the hypothesized ALLEX intra-group joint ordering WITHOUT the dataset.

Principle: the checkpoint's per-channel normalization percentiles [q01, q99]
(general_embodiment state stats) are observed real-robot joint values, so each
channel's [q01, q99] MUST lie inside the MJCF joint limit of whatever joint sits
at that index. A wrong joint->index assignment puts a percentile band outside a
joint's range (e.g. a positive-only finger band on a negative-range thumb joint).

We test the HYPOTHESIS ordering = per-group slice of the npz joint_names order,
concatenated in the frozen contract group order, and also check that random
permutations produce violations (i.e. the ordering is discriminative, not that
every ordering trivially fits).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MJCF = Path.home() / "workspace/allex_model/mjcf/ALLEX.xml"
STATS = json.load(open(HERE / "allex_stats.json"))["general_embodiment"]["state"]

# Frozen contract group concat order + the HYPOTHESIS intra-group joint names
# (per-group slice of the npz joint_names order).
GROUPS = {
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


def joint_limits() -> dict[str, tuple[float, float]]:
    """Parse joint name -> (lo, hi) range from the ALLEX MJCF."""
    txt = MJCF.read_text()
    lim = {}
    for m in re.finditer(r'<joint\b[^>]*\bname="([^"]+)"[^>]*>', txt):
        tag = m.group(0)
        name = m.group(1)
        rng = re.search(r'\brange="([-\d.eE]+)\s+([-\d.eE]+)"', tag)
        if rng:
            lim[name] = (float(rng.group(1)), float(rng.group(2)))
    return lim


def check(order_by_group: dict[str, list[str]], lim: dict,
          tol: float = 0.05, gross: float = 0.35):
    """Return (n_minor, n_gross, details) for a candidate ordering.

    A channel is out-of-range if [q01,q99] escapes [lo-tol, hi+tol]. The escape
    is MINOR if <= `gross` rad (real-vs-sim ROM / calibration slack) or GROSS if
    larger (a likely wrong joint->index assignment).
    """
    n_minor = n_gross = 0
    details = []
    for g, names in order_by_group.items():
        q01 = np.asarray(STATS[g]["q01"], float)
        q99 = np.asarray(STATS[g]["q99"], float)
        for i, nm in enumerate(names):
            if nm not in lim:
                continue  # no explicit range (e.g. coupled) — skip
            lo, hi = lim[nm]
            esc = max(0.0, (lo - tol) - q01[i], q99[i] - (hi + tol))
            if esc <= 0:
                continue
            if esc > gross:
                sev = "GROSS"
                n_gross += 1
            else:
                sev = "minor"
                n_minor += 1
            details.append(f"{sev} {g}[{i}]={nm}: band[{q01[i]:.3f},{q99[i]:.3f}] "
                           f"vs range[{lo:.3f},{hi:.3f}] escape={esc:.3f}")
    return n_minor, n_gross, details


def main():
    lim = joint_limits()
    n_minor, n_gross, details = check(GROUPS, lim)
    tot = n_minor + n_gross
    print(f"=== HYPOTHESIS ordering: {n_gross} GROSS + {n_minor} minor "
          f"out-of-range across 48 channels ===")
    for d in details:
        print("  ", d)

    # Discriminative control: shuffle joints WITHIN each hand group; a good check
    # must make wrong orderings clearly worse (more total escapes).
    rng = np.random.default_rng(0)
    worse = equal = 0
    trials = 500
    for _ in range(trials):
        shuf = {}
        for g, names in GROUPS.items():
            nm = list(names)
            if "hand" in g:
                rng.shuffle(nm)
            shuf[g] = nm
        m, gr, _ = check(shuf, lim)
        if (gr, m) > (n_gross, n_minor):
            worse += 1
        elif (gr, m) == (n_gross, n_minor):
            equal += 1
    print(f"\n=== discriminativeness: {worse}/{trials} random hand-permutations "
          f"strictly WORSE than hypothesis; {equal} tied ===")

    # Verdict: ordering assignment is sound iff NO gross mis-assignment AND the
    # hypothesis is the unique best (minor overshoots = real-vs-sim ROM slack).
    if n_gross == 0 and worse >= 0.99 * trials:
        verdict = "VERIFIED (uniquely-best, only minor ROM slack)"
    elif n_gross <= 2 and worse >= 0.95 * trials:
        verdict = "STRONGLY SUPPORTED (uniquely-best; few wide/noisy channels)"
    else:
        verdict = "UNCERTAIN — needs real_allex modality.json"
    print(f"\nVERDICT: intra-group ordering {verdict}")
    print("NOTE: byte-exact confirmation still requires the real_allex "
          "dataset meta/modality.json; this test rules out gross mis-ordering.")


if __name__ == "__main__":
    main()
