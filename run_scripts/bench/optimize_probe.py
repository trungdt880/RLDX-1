#!/usr/bin/env python3
"""Post-conv-fix inference probe: config + fine-grained breakdown + reference action.

With the Thor fp32-Conv3d patch-embed fix in place, the vision encoder is no
longer the bottleneck. This probe re-measures where the (now ~280 ms) per-obs
latency goes — splitting the model forward into vision patch-embed / vision
blocks / LLM / diffusion head (per Euler step) / decode — and dumps a seeded,
deterministic reference action chunk so later optimization runs can be compared
for output drift.

Usage:
    RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 \
        .venv/bin/python run_scripts/bench/optimize_probe.py \
        [--iters 30] [--seed 0] [--save-action out.npz]
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
TIMINGS = {}
COUNTS = {}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def wrap(obj, attr, name):
    """Monkeypatch obj.attr (a bound forward) with a CUDA-synced timer."""
    fn = getattr(obj, attr)

    def timed(*a, **k):
        _sync()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        _sync()
        TIMINGS.setdefault(name, []).append((time.perf_counter() - t0) * 1000)
        COUNTS[name] = COUNTS.get(name, 0) + 1
        return out

    setattr(obj, attr, timed)


def resolve(root, dotted):
    obj = root
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def hook_submodules(model):
    """Best-effort instance-level timing hooks. Prints what was hooked."""
    candidates = {
        "vision_encoder": "backbone.qwen_model.model.visual",
        "  vision.patch_embed": "backbone.qwen_model.model.visual.patch_embed",
        "LLM": "backbone.qwen_model.model.language_model",
        "action_head (all steps)": "action_model.model",
        "  action_encoder (all steps)": "action_model.action_encoder",
        "  action_decoder (all steps)": "action_model.action_decoder",
    }
    for name, path in candidates.items():
        mod = resolve(model, path)
        if mod is not None and hasattr(mod, "forward"):
            wrap(mod, "forward", name)
            print(f"  hooked {name:34s} <- {path}  ({type(mod).__name__})")
        else:
            print(f"  MISS   {name:34s} <- {path}")


def print_config(model):
    cfg = model.config
    am = getattr(model, "action_model", None)
    msat = getattr(am, "model", None) if am is not None else None
    print("\n--- model config ---")
    for k in ("num_inference_timesteps", "action_horizon", "rtc_inference_mode",
              "rtc_inference_delay", "use_memory", "use_physics", "add_pos_embed",
              "noise_s"):
        print(f"  {k:28s} = {getattr(cfg, k, '<n/a>')}")
    if am is not None:
        print(f"  action_model.action_dim      = {getattr(am, 'action_dim', '<n/a>')}")
        dt = getattr(am, 'denoising_timesteps', None)
        print(f"  denoising_timesteps          = {dt}")
    if msat is not None:
        nd = len(msat.double_blocks) if hasattr(msat, "double_blocks") else "?"
        ns = len(msat.single_blocks) if hasattr(msat, "single_blocks") else "?"
        print(f"  MSAT double/single blocks    = {nd} / {ns}")


def get_seeded_action(policy, observation, options, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    action, _ = policy.get_action(observation, copy.deepcopy(options))
    _sync()
    return {k: np.asarray(v) for k, v in action.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="RLWRLD/RLDX-1-FT-GR1")
    ap.add_argument("--obs", default="run_scripts/bench/gr1_sample_obs.pkl")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-action-steps", type=int, default=16)
    ap.add_argument("--save-action", default=None)
    args = ap.parse_args()

    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    import os
    print(f"Device: {dev}  cap {cap} | torch {torch.__version__} | "
          f"attn={os.environ.get('RLDX_ATTN_IMPL','(default)')} | "
          f"PATCHEMBED_FP32={os.environ.get('RLDX_PATCHEMBED_FP32','(auto)')}")

    with open(args.obs, "rb") as f:
        blob = pickle.load(f)
    observation, options = blob["observation"], blob["options"]
    instr = observation["annotation.human.coarse_action"][0]
    print(f"Task: {instr!r}")

    print(f"Loading policy from {args.model_path} ...")
    policy = RLDXPolicy(model_path=args.model_path,
                        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT, device="cuda")
    print_config(policy.model)
    wrapped = RLDXSimPolicyWrapper(policy, strict=True)

    # ---- determinism self-check: same seed twice -> identical action? ----
    print("\nDeterminism self-check (same seed x2) ...")
    a1 = get_seeded_action(wrapped, observation, options, args.seed)
    a2 = get_seeded_action(wrapped, observation, options, args.seed)
    max_d = max(float(np.max(np.abs(a1[k] - a2[k]))) for k in a1)
    print(f"  max|a1 - a2| (seed {args.seed}) = {max_d:.3e}   "
          f"({'DETERMINISTIC' if max_d == 0 else 'NON-DETERMINISTIC'})")

    # reference action (seeded)
    ref = get_seeded_action(wrapped, observation, options, args.seed)
    if args.save_action:
        np.savez(args.save_action, **{k.replace(".", "__"): v for k, v in ref.items()})
        print(f"  saved reference action -> {args.save_action}")
    print("  reference action chunk:")
    for k, v in ref.items():
        flat = v.reshape(v.shape[0], v.shape[1], -1) if v.ndim >= 3 else v
        print(f"    {k:26s} shape={tuple(v.shape)} first4={np.round(v.reshape(-1)[:4], 4)}")

    # ---- fine-grained breakdown ----
    print("\nInstalling timing hooks ...")
    hook_submodules(policy.model)
    warm_opts = copy.deepcopy(options)
    if isinstance(warm_opts, dict) and "reset_memory" in warm_opts:
        warm_opts["reset_memory"] = [False] * len(warm_opts["reset_memory"])

    for _ in range(args.warmup):
        wrapped.get_action(observation, copy.deepcopy(warm_opts))
        _sync()
    TIMINGS.clear()
    COUNTS.clear()

    print(f"Timing breakdown x{args.iters} ...")
    for _ in range(args.iters):
        wrapped.get_action(observation, copy.deepcopy(warm_opts))
        _sync()

    print("\n--- submodule breakdown (median per get_action) ---")
    for name in ["vision_encoder", "  vision.patch_embed", "LLM",
                 "action_head (all steps)", "  action_encoder (all steps)",
                 "  action_decoder (all steps)"]:
        if name in TIMINGS:
            calls = COUNTS[name]
            per_call = statistics.median(TIMINGS[name])
            # group medians by get_action: action head runs n_steps times/call
            total_per_obs = sum(TIMINGS[name]) / args.iters
            print(f"  {name:34s} {total_per_obs:7.1f} ms/obs  "
                  f"({calls // args.iters} calls/obs, {per_call:.2f} ms/call)")

    # ---- end-to-end ----
    print(f"\nEnd-to-end timing x{args.iters} ...")
    times = []
    for _ in range(args.iters):
        _sync()
        t0 = time.perf_counter()
        wrapped.get_action(observation, copy.deepcopy(warm_opts))
        _sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median = statistics.median(times)
    hz = 1000.0 / median
    req = CONTROL_HZ / args.n_action_steps
    print("=" * 60)
    print(f"  median {median:8.1f} ms  ->  {hz:.3f} Hz")
    print(f"  mean   {statistics.mean(times):8.1f} ms  +/- {statistics.pstdev(times):.1f}")
    print(f"  min/max {times[0]:.1f} / {times[-1]:.1f} ms")
    print(f"  realtime @ {args.n_action_steps}-step chunk: "
          f"{'YES' if hz >= req else 'NO'} (need {req:.2f} Hz)")
    print("=" * 60)


if __name__ == "__main__":
    main()
