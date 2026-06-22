#!/usr/bin/env python3
"""Break the ~6 s RLDX-1 per-observation latency into pipeline stages.

Patches PolicyRuntime stage methods with CUDA-synced timers so we can see how
the per-query cost splits across:
    prepare   — unbatch + Qwen3-VL processor (image resize/patchify) + collate
    rtc       — RTC action-prefix injection
    inference — model.get_action  (Qwen3-VL backbone + MSAT diffusion head)
    decode    — normalized -> physical action denorm

and, inside inference, backbone vs action-head (diffusion) when those hooks are
reachable. Run on the captured real GR-1 observation.

Usage:
    RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 \
        .venv/bin/python run_scripts/bench/profile_gr1_stages.py [--iters 10]
"""
import argparse
import pickle
import statistics
import time

import torch

from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy import policy_runtime as PR
from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper

TIMINGS = {}


def _sync():
    torch.cuda.synchronize()


def timed(name):
    def deco(fn):
        def wrap(*a, **k):
            _sync()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            _sync()
            TIMINGS.setdefault(name, []).append((time.perf_counter() - t0) * 1000)
            return out
        return wrap
    return deco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="RLWRLD/RLDX-1-FT-GR1")
    ap.add_argument("--obs", default="run_scripts/bench/gr1_sample_obs.pkl")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    # Patch the runtime stage methods before any instance is built.
    PR.PolicyRuntime._prepare_inputs = timed("prepare")(PR.PolicyRuntime._prepare_inputs)
    PR.PolicyRuntime._inject_rtc_prefix = timed("rtc")(PR.PolicyRuntime._inject_rtc_prefix)
    PR.PolicyRuntime._run_inference = timed("inference (backbone+action head)")(
        PR.PolicyRuntime._run_inference)
    PR.PolicyRuntime._decode = timed("decode")(PR.PolicyRuntime._decode)

    with open(args.obs, "rb") as f:
        blob = pickle.load(f)
    observation, options = blob["observation"], blob["options"]

    policy = RLDXPolicy(model_path=args.model_path,
                        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT, device="cuda")
    policy = RLDXSimPolicyWrapper(policy, strict=True)

    policy.get_action(observation, options)  # warmup / cold start
    TIMINGS.clear()

    for _ in range(args.iters):
        policy.get_action(observation, options)

    total = sum(statistics.median(v) for v in TIMINGS.values())
    print("\n" + "=" * 60)
    print(f" RLDX-1 stage breakdown  (median of {args.iters}, GR-1 real obs)")
    print("=" * 60)
    order = ["prepare", "rtc", "inference (backbone+action head)", "decode"]
    for name in order:
        if name in TIMINGS:
            ms = statistics.median(TIMINGS[name])
            print(f"  {name:36s} {ms:8.1f} ms  {100*ms/total:5.1f}%")
    print("-" * 60)
    print(f"  {'TOTAL (sum of medians)':36s} {total:8.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
