#!/usr/bin/env python3
"""Minimal RLDX-1 inference example (HuggingFace-style): load → observe → act.

Loads the RLDX-1 GR-1 policy, loads a single real observation (captured from the
GR-1 Tabletop sim by save_gr1_observation.py), runs one inference, prints the
predicted action chunk, and reports timing in ms and Hz.

    # one-time: capture a real observation (needs the GR-1 sim running)
    #   see run_scripts/bench/save_gr1_observation.py
    #
    # then, anywhere with the main venv:
    RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 \
        .venv/bin/python run_scripts/bench/infer_gr1.py
"""
import pickle
import time

import numpy as np
import torch

from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper

MODEL_PATH = "RLWRLD/RLDX-1-FT-GR1"
OBS_PATH = "run_scripts/bench/gr1_sample_obs.pkl"


def main():
    # 1. Load the policy (downloads from HF hub on first run, then cached).
    print(f"Loading RLDX-1 from {MODEL_PATH} ...")
    policy = RLDXPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        device="cuda",
    )
    # The sim wrapper accepts the flat observation format the GR-1 env emits
    # ('video.*', 'state.*', 'annotation.*') and returns flat 'action.*' chunks.
    policy = RLDXSimPolicyWrapper(policy, strict=True)

    # 2. Load one real observation (a single env step).
    with open(OBS_PATH, "rb") as f:
        blob = pickle.load(f)
    observation, options = blob["observation"], blob["options"]
    print(f"\nTask: {observation['annotation.human.coarse_action'][0]!r}")
    print("Observation modalities:")
    for k, v in observation.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:42s} {str(v.dtype):8s} {tuple(v.shape)}")

    # 3. Inference (first call also pays one-time CUDA/cuDNN warmup).
    print("\nWarming up ...")
    action, _ = policy.get_action(observation, options)
    torch.cuda.synchronize()

    print("Inferring ...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    action, info = policy.get_action(observation, options)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Show the predicted action chunk.
    print("\nPredicted action chunk (one open-loop sequence per joint group):")
    for k, v in action.items():
        v = np.asarray(v)
        print(f"  {k:24s} {tuple(v.shape)}  first-step={np.round(v.reshape(v.shape[0], -1)[0][:4], 3)} ...")

    print(f"\nInference latency: {dt_ms:.1f} ms  ->  {1000.0/dt_ms:.3f} Hz")


if __name__ == "__main__":
    main()
