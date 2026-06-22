#!/usr/bin/env python3
"""Sweep num_inference_timesteps (flow-matching Euler steps): latency vs output drift.

One model load. Reference = the shipped 4-step output (seeded, fixed noise). For
each reduced step count we report end-to-end latency AND how far the resulting
action chunk drifts from the 4-step reference — the accuracy cost of trading
denoising steps for speed.

    RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 \
      .venv/bin/python run_scripts/bench/step_sweep.py [--steps 4 3 2 1]
"""
import argparse
import copy
import pickle
import statistics
import time

import numpy as np
import torch

from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper

CONTROL_HZ = 20.0


def _sync():
    torch.cuda.synchronize()


def seeded_action(policy, obs, opts, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    a, _ = policy.get_action(obs, copy.deepcopy(opts))
    _sync()
    return {k: np.asarray(v) for k, v in a.items()}


def latency(policy, obs, opts, iters, seed):
    ts = []
    for _ in range(iters):
        torch.manual_seed(seed)
        _sync(); t0 = time.perf_counter()
        policy.get_action(obs, copy.deepcopy(opts))
        _sync(); ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="RLWRLD/RLDX-1-FT-GR1")
    ap.add_argument("--obs", default="run_scripts/bench/gr1_sample_obs.pkl")
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 3, 2, 1])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.obs, "rb") as f:
        blob = pickle.load(f)
    obs, options = blob["observation"], blob["options"]
    opts = copy.deepcopy(options)
    if isinstance(opts, dict) and "reset_memory" in opts:
        opts["reset_memory"] = [False] * len(opts["reset_memory"])

    print(f"Loading {args.model_path} ...")
    policy = RLDXPolicy(model_path=args.model_path,
                        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT, device="cuda")
    am = policy.model.action_model
    base_n = am.num_inference_timesteps
    wrapped = RLDXSimPolicyWrapper(policy, strict=True)
    print(f"shipped num_inference_timesteps = {base_n}")

    # warmup + reference at shipped step count
    for _ in range(3):
        wrapped.get_action(obs, copy.deepcopy(opts)); _sync()
    am.num_inference_timesteps = base_n
    ref = seeded_action(wrapped, obs, opts, args.seed)

    print(f"\n{'steps':>5} {'median ms':>10} {'Hz':>7} {'speedup':>8} {'max|Δ| vs '+str(base_n)+'-step':>20} {'mean|Δ|':>10}")
    print("-" * 70)
    base_med = None
    for n in args.steps:
        am.num_inference_timesteps = n
        for _ in range(2):
            wrapped.get_action(obs, copy.deepcopy(opts)); _sync()
        got = seeded_action(wrapped, obs, opts, args.seed)
        med = latency(wrapped, obs, opts, args.iters, args.seed)
        if n == base_n:
            base_med = med
        max_d = max(float(np.max(np.abs(ref[k] - got[k]))) for k in ref)
        mean_d = float(np.mean([np.mean(np.abs(ref[k] - got[k])) for k in ref]))
        sp = (base_med / med) if base_med else float("nan")
        print(f"{n:>5} {med:>10.1f} {1000/med:>7.2f} {sp:>7.2f}x {max_d:>20.4e} {mean_d:>10.4e}")


if __name__ == "__main__":
    main()
