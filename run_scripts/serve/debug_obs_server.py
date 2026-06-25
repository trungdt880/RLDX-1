#!/usr/bin/env python3
"""Mock RLDX ZeroMQ inference server for inspecting what a client sends.

Speaks the SAME wire protocol as the real server (rldx.policy.server_client:
PolicyServer / MsgSerializer over a ZMQ REP socket, same endpoints), but loads
NO model. For every `get_action` request it:
  * prints the observation + options structure (key, type, dtype, shape, range),
  * saves each camera frame to a PNG, and
  * returns a correctly-shaped *dummy* (zeros) DROID action chunk so the client
    keeps stepping.

Use this to bring up / debug the other side's ZeroMQ client without waiting on
the 13 GB model. Point their client at this host:port exactly as if it were the
real server.

Usage:
    .venv/bin/python run_scripts/serve/debug_obs_server.py            # 0.0.0.0:5555
    .venv/bin/python run_scripts/serve/debug_obs_server.py --port 5599 --outdir ./zmq_debug
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import zmq

# Line-buffer stdout so the structure dump shows up live even when piped to a file.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

# Reuse the real serializer + frame saver so behavior matches the production server.
from rldx.data.types import ModalityConfig
from rldx.policy.frame_logger import save_video_frames
from rldx.policy.server_client import MsgSerializer


# DROID action schema the real server returns (flat keys, 16-step chunk).
ACTION_DIMS = {
    "action.end_effector_position": 3,
    "action.end_effector_rotation": 3,
    "action.gripper_close": 1,
}
ACTION_HORIZON = 16

# DROID modality config (mirrors RLDX-1-PT processor/processor_config.json) so a
# client calling get_modality_config gets the same answer the real server gives.
DROID_MODALITY_CONFIG = {
    "video": ModalityConfig(delta_indices=[-6, -4, -2, 0],
                            modality_keys=["primary", "secondary", "wrist"]),
    "state": ModalityConfig(delta_indices=[0],
                            modality_keys=["end_effector_position",
                                           "end_effector_rotation", "gripper_position"]),
    "action": ModalityConfig(delta_indices=list(range(16)),
                             modality_keys=["end_effector_position",
                                            "end_effector_rotation", "gripper_close"]),
    "language": ModalityConfig(delta_indices=[0],
                               modality_keys=["annotation.human.action.task_description"]),
}


def _describe(obj, indent=2):
    """Pretty-print a (possibly nested) obs/options object with shapes & dtypes."""
    pad = " " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                extra = ""
                if np.issubdtype(v.dtype, np.number) and v.size:
                    extra = f"  range[{v.min()}, {v.max()}]"
                print(f"{pad}{k:44s} ndarray {str(v.dtype):8s} {tuple(v.shape)}{extra}")
            elif isinstance(v, (list, tuple)):
                sample = v[0] if len(v) else None
                print(f"{pad}{k:44s} {type(v).__name__}[{len(v)}]  e.g. {sample!r}")
            elif isinstance(v, dict):
                print(f"{pad}{k:44s} dict")
                _describe(v, indent + 4)
            else:
                print(f"{pad}{k:44s} {type(v).__name__}  {v!r}")
    else:
        print(f"{pad}{type(obj).__name__}: {obj!r}")


def _dummy_action(observation: dict) -> dict:
    """Build a zeros DROID action chunk with batch inferred from the obs."""
    batch = 1
    for v in observation.values():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            batch = int(v.shape[0])
            break
        if isinstance(v, (list, tuple)):
            batch = len(v)
            break
    return {k: np.zeros((batch, ACTION_HORIZON, d), dtype=np.float32)
            for k, d in ACTION_DIMS.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--outdir", default="./zmq_debug")
    ap.add_argument("--save-steps", type=int, default=3,
                    help="save frames for the first N observations (0 = never)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[debug-obs-server] listening on tcp://{args.host}:{args.port}")
    print(f"[debug-obs-server] frames -> {outdir.resolve()} (first {args.save_steps} obs)\n")

    step = 0
    try:
        while True:
            request = MsgSerializer.from_bytes(sock.recv())

            # Tolerate both wire shapes: {"endpoint","data"} (PolicyClient) and a bare obs.
            if isinstance(request, dict) and "endpoint" in request:
                endpoint = request.get("endpoint", "get_action")
                data = request.get("data", {}) or {}
            else:
                endpoint, data = "get_action", {"observation": request, "options": None}

            if endpoint == "ping":
                sock.send(MsgSerializer.to_bytes({"status": "ok", "message": "debug-obs-server"}))
                continue
            if endpoint == "get_modality_config":
                sock.send(MsgSerializer.to_bytes(DROID_MODALITY_CONFIG))
                continue
            if endpoint == "reset":
                print(f"[reset] options: {data.get('options')}")
                sock.send(MsgSerializer.to_bytes({"status": "ok"}))
                continue
            if endpoint != "get_action":
                sock.send(MsgSerializer.to_bytes({"error": f"unknown endpoint: {endpoint}"}))
                continue

            # ---- get_action: inspect, save, reply with dummy action ----
            observation = data.get("observation", {})
            options = data.get("options")
            print(f"================ get_action  (step {step}) ================")
            print("  OBSERVATION:")
            _describe(observation, indent=4)
            if options is not None:
                print("  OPTIONS:")
                _describe(options, indent=4)

            if step < args.save_steps:
                saved = save_video_frames(observation, outdir, step)
                if saved:
                    print(f"  saved {saved} frame(s) -> {outdir}/step{step:04d}_*")

            action = _dummy_action(observation)
            print("  -> returning dummy action:")
            _describe(action, indent=4)
            print()
            sock.send(MsgSerializer.to_bytes([action, {"debug": True}]))
            step += 1
    except KeyboardInterrupt:
        print("\n[debug-obs-server] shutting down")
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
