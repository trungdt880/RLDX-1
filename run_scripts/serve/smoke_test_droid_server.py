#!/usr/bin/env python3
"""Smoke-test the RLDX-1-PT DROID ZeroMQ inference server.

Connects to a running server (started by run_rldx_pt_droid_server.sh), pings it,
pulls the modality config, builds ONE synthetic DROID observation with the exact
keys / shapes / dtypes the server expects, calls the `get_action` endpoint, and
validates that a 16-step EEF-delta action chunk comes back.

This exercises the full server path a sim-evals ZeroMQ client will use
(serialize obs -> ZMQ -> RLDXSimPolicyWrapper -> RLDXPolicy -> action -> ZMQ back),
without needing Isaac Sim.

Usage:
    .venv/bin/python run_scripts/serve/smoke_test_droid_server.py
    .venv/bin/python run_scripts/serve/smoke_test_droid_server.py --host 1.2.3.4 --port 5555
"""
import argparse
import sys
import time

import numpy as np

from rldx.policy.server_client import PolicyClient


# DROID state/action feature dims (from RLDX-1-PT processor/statistics.json).
DROID_STATE_DIMS = {
    "end_effector_position": 3,
    "end_effector_rotation": 3,
    "gripper_position": 1,
}
LANG_KEY = "annotation.human.action.task_description"
INSTRUCTION = "pick up the object and place it in the bin"
EXPECTED_ACTION_KEYS = {
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
}
ACTION_HORIZON = 16
IMG_HW = (224, 224)  # arbitrary; the processor resizes. Any HxW RGB uint8 works.


def _state_dim(key: str) -> int:
    # Strip a leading "state." if a fully-qualified key sneaks in.
    return DROID_STATE_DIMS[key.split(".")[-1]]


def build_observation(modality_cfg: dict, batch: int = 1) -> dict:
    """Build one synthetic flat DROID observation from the server's modality config."""
    obs: dict = {}

    video_cfg = modality_cfg["video"]
    t_video = len(video_cfg.delta_indices)  # DROID: 4-frame history [-6,-4,-2,0]
    for key in video_cfg.modality_keys:  # primary, secondary, wrist
        obs[f"video.{key}"] = np.random.randint(
            0, 256, size=(batch, t_video, *IMG_HW, 3), dtype=np.uint8
        )

    state_cfg = modality_cfg["state"]
    t_state = len(state_cfg.delta_indices)  # DROID: 1
    for key in state_cfg.modality_keys:
        obs[f"state.{key}"] = np.zeros((batch, t_state, _state_dim(key)), dtype=np.float32)

    # Language: flat key, list[str] of length B (the wrapper expands to (B, 1)).
    lang_keys = modality_cfg["language"].modality_keys
    obs[lang_keys[0]] = [INSTRUCTION] * batch

    return obs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--timeout-ms", type=int, default=180000,
                    help="generous: the first get_action pays one-time CUDA/cuDNN warmup")
    args = ap.parse_args()

    client = PolicyClient(host=args.host, port=args.port, timeout_ms=args.timeout_ms)

    print(f"[1/4] ping  tcp://{args.host}:{args.port} ...")
    if not client.ping():
        print("  FAIL: server did not respond to ping. Is it up and done loading?")
        return 1
    print("  ok")

    print("[2/4] get_modality_config ...")
    modality_cfg = client.get_modality_config()
    vkeys = modality_cfg["video"].modality_keys
    skeys = modality_cfg["state"].modality_keys
    akeys = modality_cfg["action"].modality_keys
    print(f"  video={vkeys} (T={len(modality_cfg['video'].delta_indices)})")
    print(f"  state={skeys} (T={len(modality_cfg['state'].delta_indices)})")
    print(f"  action={akeys} (horizon={len(modality_cfg['action'].delta_indices)})")

    print("[3/4] build synthetic obs + call get_action ...")
    obs = build_observation(modality_cfg, batch=args.batch)
    for k, v in obs.items():
        shape = v.shape if isinstance(v, np.ndarray) else f"list[{len(v)}]"
        dtype = v.dtype if isinstance(v, np.ndarray) else "str"
        print(f"    {k:42s} {str(dtype):8s} {shape}")
    options = {
        "session_ids": [f"smoke_{i}" for i in range(args.batch)],
        "reset_memory": [True] * args.batch,
    }
    t0 = time.perf_counter()
    action, info = client.get_action(obs, options)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  get_action returned in {dt_ms:.0f} ms")

    print("[4/4] validate action chunk ...")
    ok = True
    for key, dim in EXPECTED_ACTION_KEYS.items():
        if key not in action:
            print(f"  FAIL: missing action key '{key}' (got {list(action)})")
            ok = False
            continue
        arr = np.asarray(action[key])
        exp = (args.batch, ACTION_HORIZON, dim)
        status = "ok" if arr.shape == exp else "BAD SHAPE"
        if arr.shape != exp:
            ok = False
        print(f"  {key:32s} {str(arr.dtype):8s} {arr.shape}  expected {exp}  [{status}]")
        print(f"      first step: {np.round(arr[0, 0], 4)}")

    print()
    if ok:
        print(f"PASS — DROID server inference works ({dt_ms:.0f} ms/query, "
              f"{1000.0/dt_ms:.2f} Hz, {ACTION_HORIZON}-step chunk).")
        return 0
    print("FAIL — action chunk did not match the expected DROID schema (see above).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
