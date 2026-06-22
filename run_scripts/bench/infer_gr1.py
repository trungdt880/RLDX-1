#!/usr/bin/env python3
"""Minimal RLDX-1 inference example (HuggingFace-style): load → observe → act.

Loads the RLDX-1 GR-1 policy, loads a real observation (captured from the GR-1
Tabletop sim by save_gr1_observation.py), runs inference, prints the predicted
action chunk, and reports timing in ms and Hz.

Two modes:
  (default)     one inference call — show the action chunk + single-query latency.
  --episode     run a FULL EPISODE'S WORTH of policy queries back-to-back
                (ceil(max_steps / n_action_steps) calls) and report per-episode
                inference cost, per-query latency, and the real-time margin.
                --steps N forces an explicit query count.

NOTE on --episode: this replays the captured observation each query — it measures
the *sustained inference cost* of an episode, NOT a physics rollout. The GR-1
mujoco sim lives in a separate venv and talks to the policy over ZeroMQ, so a real
closed-loop episode cannot run in this single (main-venv) process. For a real
task rollout (with environment feedback + video) use:
    run_scripts/bench/run_gr1_episode.sh

Usage:
    # one-time: capture a real observation (needs the GR-1 sim) — see save_gr1_observation.py
    RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 \
        .venv/bin/python run_scripts/bench/infer_gr1.py                 # single inference
    RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 \
        .venv/bin/python run_scripts/bench/infer_gr1.py --episode       # full-episode inference loop
"""
import argparse
import copy
import math
import pickle
import statistics
import time

import numpy as np
import torch

from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper

CONTROL_HZ = 20.0  # GR-1 Tabletop sim control rate


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="RLWRLD/RLDX-1-FT-GR1")
    ap.add_argument("--obs", default="run_scripts/bench/gr1_sample_obs.pkl")
    ap.add_argument("--episode", action="store_true",
                    help="run a full episode's worth of policy queries "
                         "(ceil(max_steps / n_action_steps)) and report per-episode cost")
    ap.add_argument("--steps", type=int, default=None,
                    help="explicit number of policy queries (overrides --episode)")
    ap.add_argument("--max-steps", type=int, default=720,
                    help="env steps in one episode (to derive query count for --episode)")
    ap.add_argument("--n-action-steps", type=int, default=16,
                    help="control steps executed open-loop per policy query (chunk length)")
    args = ap.parse_args()

    # 1. Load the policy (downloads from HF hub on first run, then cached).
    print(f"Loading RLDX-1 from {args.model_path} ...")
    policy = RLDXPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        device="cuda",
    )
    # The sim wrapper accepts the flat observation format the GR-1 env emits
    # ('video.*', 'state.*', 'annotation.*') and returns flat 'action.*' chunks.
    policy = RLDXSimPolicyWrapper(policy, strict=True)

    # 2. Load one real observation (a single env step).
    with open(args.obs, "rb") as f:
        blob = pickle.load(f)
    observation, options = blob["observation"], blob["options"]
    print(f"\nTask: {observation['annotation.human.coarse_action'][0]!r}")
    print("Observation modalities:")
    for k, v in observation.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:42s} {str(v.dtype):8s} {tuple(v.shape)}")

    # 3. Warmup (first call also pays one-time CUDA/cuDNN warmup).
    print("\nWarming up ...")
    action, _ = policy.get_action(observation, copy.deepcopy(options))
    _sync()

    # 4. Single inference — show the action chunk + latency.
    _sync()
    t0 = time.perf_counter()
    action, info = policy.get_action(observation, copy.deepcopy(options))
    _sync()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print("\nPredicted action chunk (one open-loop sequence per joint group):")
    for k, v in action.items():
        v = np.asarray(v)
        print(f"  {k:24s} {tuple(v.shape)}  first-step={np.round(v.reshape(v.shape[0], -1)[0][:4], 3)} ...")
    print(f"\nSingle-query inference latency: {dt_ms:.1f} ms  ->  {1000.0 / dt_ms:.3f} Hz")

    # 5. Optional: a full episode's worth of policy queries.
    n_queries = args.steps
    if n_queries is None and args.episode:
        n_queries = math.ceil(args.max_steps / args.n_action_steps)
    if not n_queries or n_queries <= 1:
        return

    # Steady-state regime: don't re-reset session memory each query.
    warm_opts = copy.deepcopy(options)
    if isinstance(warm_opts, dict) and "reset_memory" in warm_opts:
        warm_opts["reset_memory"] = [False] * len(warm_opts["reset_memory"])

    print(f"\n=== full-episode inference loop: {n_queries} policy queries "
          f"(= one {args.max_steps}-step episode @ {args.n_action_steps}-step chunks) ===")
    print("    (replays the captured obs — sustained inference cost, NOT a physics rollout)")
    per_query = []
    ep_t0 = time.perf_counter()
    for i in range(n_queries):
        _sync()
        t0 = time.perf_counter()
        policy.get_action(observation, copy.deepcopy(warm_opts))
        _sync()
        per_query.append((time.perf_counter() - t0) * 1000.0)
    episode_s = time.perf_counter() - ep_t0

    med = statistics.median(per_query)
    mean = statistics.mean(per_query)
    hz = 1000.0 / med
    required_hz = CONTROL_HZ / args.n_action_steps
    # robot-time covered = (queries × chunk length) control steps at CONTROL_HZ
    robot_s = n_queries * args.n_action_steps / CONTROL_HZ
    print(f"  per-query latency : median {med:.1f} ms  mean {mean:.1f} ms  "
          f"min/max {min(per_query):.1f}/{max(per_query):.1f}")
    print(f"  episode inference : {episode_s:.2f} s of policy compute over {n_queries} queries")
    print(f"  control rate      : {hz:.2f} Hz inference  (need {required_hz:.2f} Hz for "
          f"{CONTROL_HZ:.0f} Hz / {args.n_action_steps}-step chunks)")
    print(f"  real-time capable : {'YES' if hz >= required_hz else 'NO'}  "
          f"({n_queries} queries = {robot_s:.0f}s robot time at {CONTROL_HZ:.0f} Hz, "
          f"{episode_s:.1f}s of it in policy inference)")
    print("\nFor a REAL closed-loop episode (env feedback + video), run:")
    print("  run_scripts/bench/run_gr1_episode.sh")


if __name__ == "__main__":
    main()
