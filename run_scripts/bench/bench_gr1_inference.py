#!/usr/bin/env python3
"""End-to-end RLDX-1 inference latency on a REAL GR-1 Tabletop observation.

Loads the policy exactly as the eval server does (RLDXPolicy + sim wrapper,
``RLDX_ATTN_IMPL`` attention), replays a single real observation captured from
the live mujoco rollout (see save_gr1_observation.py), and times the full
per-observation pipeline: image/state preprocessing → Qwen3-VL backbone →
MSAT diffusion action head (all denoising steps) → action decode.

This is the latency the robot actually waits on for one policy query. Because
RLDX runs action chunking (one query returns ``n_action_steps`` actions executed
open-loop), the policy is queried once every ``n_action_steps`` control steps —
so we also report the effective control rate that this latency can sustain.

Usage:
    RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 \
        .venv/bin/python run_scripts/bench/bench_gr1_inference.py \
        [--obs run_scripts/bench/gr1_sample_obs.pkl] [--iters 30] [--json out.json]
"""
import argparse
import copy
import json
import os
import pickle
import statistics
import time

import torch

from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper

# GR-1 Tabletop sim runs at 20 Hz control; the eval default executes 16 actions
# per policy query open-loop.
CONTROL_HZ = 20.0


def fmt(ms):
    return f"{ms:8.1f} ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="RLWRLD/RLDX-1-FT-GR1")
    ap.add_argument("--obs", default="run_scripts/bench/gr1_sample_obs.pkl")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--n-action-steps", type=int, default=16)
    ap.add_argument("--compile", choices=["none", "submodule", "fullgraph"],
                    default="none", help="inference optimization path (none=A, "
                    "submodule=B, fullgraph=D)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    attn = os.environ.get("RLDX_ATTN_IMPL", "(default)")
    dev_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"Device: {dev_name}  cap {cap}  | torch {torch.__version__} | attn={attn}")

    with open(args.obs, "rb") as f:
        blob = pickle.load(f)
    observation, options = blob["observation"], blob["options"]
    instr = observation["annotation.human.coarse_action"][0]
    print(f"Observation: real GR-1 Tabletop step  | task: {instr!r}")

    print(f"Loading policy from {args.model_path} ...")
    policy = RLDXPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        device="cuda",
    )

    # Apply inference optimization to the inner policy BEFORE wrapping, exactly
    # as run_rldx_server does. Path A=eager, B=per-submodule torch.compile,
    # D=fullgraph capture.
    if args.compile != "none":
        from rldx.inference.serve_optimization import apply_optimization

        path = {"submodule": "B", "fullgraph": "D"}[args.compile]
        print(f"Applying inference optimization path={path} ({args.compile}) ...")
        info = apply_optimization(policy, path=path)
        print(f"  optimization info: {info}")

    policy = RLDXSimPolicyWrapper(policy, strict=True)

    # Cold-start (first) call resets session memory + RTC like a fresh episode.
    cold_opts = copy.deepcopy(options)
    # Steady-state calls: keep the same session but don't re-reset memory, which
    # is the dominant regime during an episode.
    warm_opts = copy.deepcopy(options)
    if isinstance(warm_opts, dict) and "reset_memory" in warm_opts:
        warm_opts["reset_memory"] = [False] * len(warm_opts["reset_memory"])

    # ---- warmup (includes one cold-start) ----
    print(f"Warmup x{args.warmup} ...")
    action, _ = policy.get_action(observation, cold_opts)
    torch.cuda.synchronize()
    for _ in range(max(0, args.warmup - 1)):
        policy.get_action(observation, warm_opts)
        torch.cuda.synchronize()

    # ---- timed (steady-state) ----
    print(f"Timing x{args.iters} (steady-state) ...")
    times = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        policy.get_action(observation, warm_opts)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    mean = statistics.mean(times)
    median = statistics.median(times)
    std = statistics.pstdev(times)
    p90 = times[int(0.9 * (len(times) - 1))]
    lo, hi = times[0], times[-1]
    infer_hz = 1000.0 / median

    # action chunk shape (per joint key) -> total actions returned per query
    chunk_len = None
    for k, v in action.items():
        chunk_len = v.shape[1] if hasattr(v, "shape") and v.ndim >= 2 else v.shape[0]
        break
    n_action = args.n_action_steps
    required_hz = CONTROL_HZ / n_action  # policy queries needed per second
    realtime_ok = infer_hz >= required_hz

    print("\n" + "=" * 64)
    print(" RLDX-1 per-observation inference  (GR-1 Tabletop, real obs)")
    print("=" * 64)
    print(f"  median     {fmt(median)}   ->  {infer_hz:6.3f} Hz")
    print(f"  mean       {fmt(mean)}   +/- {std:.1f} ms")
    print(f"  min / p90  {fmt(lo)} / {fmt(p90)}")
    print(f"  max        {fmt(hi)}")
    print("-" * 64)
    print(f"  action chunk returned per query : {chunk_len} steps "
          f"(executed open-loop)")
    print(f"  control rate (sim)              : {CONTROL_HZ:.0f} Hz")
    print(f"  policy queried every            : {n_action} steps "
          f"-> need {required_hz:.2f} Hz of inference")
    print(f"  real-time capable @ {n_action}-step chunk : "
          f"{'YES' if realtime_ok else 'NO'} "
          f"(inference {infer_hz:.2f} Hz vs need {required_hz:.2f} Hz)")
    print("=" * 64)

    if args.json:
        out = {
            "device": dev_name,
            "capability": list(cap),
            "torch": torch.__version__,
            "attn_impl": attn,
            "compile": args.compile,
            "model_path": args.model_path,
            "task": instr,
            "iters": args.iters,
            "latency_ms": {
                "median": median, "mean": mean, "std": std,
                "min": lo, "p90": p90, "max": hi,
            },
            "all_times_ms": times,
            "inference_hz": infer_hz,
            "chunk_len": int(chunk_len),
            "control_hz": CONTROL_HZ,
            "n_action_steps": n_action,
            "required_inference_hz": required_hz,
            "realtime_capable": bool(realtime_ok),
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[i] wrote {args.json}")


if __name__ == "__main__":
    main()
