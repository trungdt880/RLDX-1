# SPDX-License-Identifier: Apache-2.0
"""Opt-in observation shape adapter for the inference server.

A tolerant shim that promotes the "natural" per-step observation shapes a sim
client often sends into the batched (B, T, ...) shapes RLDXPolicy requires,
*before* the strict validator runs:

    state.<key>   (D,)        -> (1, 1, D)        # prepend batch + time
    state.<key>   (T, D)      -> (1, T, D)        # prepend batch
    video.<key>   (T,H,W,C)   -> (1, T, H, W, C)  # prepend batch
    <language>    "text"      -> ["text"]         # wrap scalar string

It only ever *prepends* missing leading dims; it never fabricates a temporal
history. So a single video frame (H, W, C) becomes (1, 1, H, W, C) and the
validator still rejects it for not having the required T frames — that stays a
loud error rather than a silent wrong-history bug.

Enable with `--auto-batch-obs` on run_rldx_server (or AUTOBATCH=1 in the launch
script). Off by default: the correct fix is for the client to send batched arrays.
"""
from typing import Any

import numpy as np


class ObsShapeAdapter:
    """Wraps a policy and normalizes obs shapes before delegating get_action."""

    def __init__(self, policy: Any):
        self.policy = policy

    @staticmethod
    def _promote(arr: np.ndarray, target_ndim: int) -> np.ndarray:
        # Prepend singleton leading dims until arr has target_ndim.
        while arr.ndim < target_ndim:
            arr = arr[None, ...]
        return arr

    def _normalize(self, observation: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, val in observation.items():
            if isinstance(key, str) and key.startswith("video.") and isinstance(val, np.ndarray):
                out[key] = self._promote(val, 5)  # (B, T, H, W, C)
            elif isinstance(key, str) and key.startswith("state.") and isinstance(val, np.ndarray):
                out[key] = self._promote(val, 3)  # (B, T, D)
            elif isinstance(val, str):
                # language: a bare string -> list[str] of batch 1
                out[key] = [val]
            else:
                out[key] = val
        return out

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.policy.get_action(self._normalize(observation), options)

    # Delegate everything else (reset, get_modality_config, check_*, attrs).
    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)
