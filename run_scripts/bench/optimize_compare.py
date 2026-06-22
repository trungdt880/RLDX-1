#!/usr/bin/env python3
"""Apply an inference optimization, then compare latency AND action output vs eager.

In one process: load policy → capture eager reference action (seeded) → apply the
optimization (compile / cuda-graph / fewer denoising steps) → re-run with the SAME
initial diffusion noise → report end-to-end latency and the max/relative action
drift vs eager. Because the model is bitwise-deterministic given a seed, an
output-preserving optimization should match eager to fp roundoff; an
output-changing one (fewer steps) will not, and we quantify by how much.

Usage:
    RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas \
      .venv/bin/python run_scripts/bench/optimize_compare.py --opt cudagraph [--iters 30]

  --opt eager | compileB | cudagraph | triton | steps
  --steps N        (only for --opt steps) number of denoising Euler steps
  --fixed-noise    pin the initial diffusion noise so compute-path differences
                   (not RNG draws) are isolated — needed for cudagraph/triton.
"""
import argparse
import copy
import os
import pickle
import statistics
import time

import numpy as np
import torch

from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper

CONTROL_HZ = 20.0


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---- fixed-noise monkeypatch (isolates compute path from RNG draw order) ----
_orig_randn = torch.randn
_orig_normal_ = torch.Tensor.normal_
_NOISE_CACHE = {}
_NOISE_SHAPE = None


def _norm_shape(size):
    if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
        return tuple(size[0])
    return tuple(size)


def _fixed_randn(*size, **kw):
    if "size" in kw:
        shp = tuple(kw["size"])
    else:
        shp = _norm_shape(size)
    if _NOISE_SHAPE is not None and shp == _NOISE_SHAPE:
        if shp not in _NOISE_CACHE:
            _NOISE_CACHE[shp] = _orig_randn(*size, **kw)
        t = _NOISE_CACHE[shp]
        dev = kw.get("device", t.device)
        dt = kw.get("dtype", t.dtype)
        return t.to(device=dev, dtype=dt).clone()
    return _orig_randn(*size, **kw)


def _fixed_normal_(self, *a, **k):
    shp = tuple(self.shape)
    if _NOISE_SHAPE is not None and shp == _NOISE_SHAPE and shp in _NOISE_CACHE:
        self.copy_(_NOISE_CACHE[shp].to(self.device, self.dtype))
        return self
    return _orig_normal_(self, *a, **k)


def enable_fixed_noise(shape):
    global _NOISE_SHAPE
    _NOISE_SHAPE = tuple(shape)
    torch.randn = _fixed_randn
    torch.Tensor.normal_ = _fixed_normal_


def get_seeded_action(policy, observation, options, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    action, _ = policy.get_action(observation, copy.deepcopy(options))
    _sync()
    return {k: np.asarray(v) for k, v in action.items()}


def compare(ref, got):
    rows = []
    max_abs = 0.0
    for k in ref:
        d = np.abs(ref[k] - got[k])
        denom = np.abs(ref[k]).max() + 1e-8
        rows.append((k, float(d.max()), float(d.mean()), float(d.max() / denom)))
        max_abs = max(max_abs, float(d.max()))
    return max_abs, rows


def median_latency(policy, observation, warm_opts, iters, seed):
    times = []
    for _ in range(iters):
        torch.manual_seed(seed)
        _sync()
        t0 = time.perf_counter()
        policy.get_action(observation, copy.deepcopy(warm_opts))
        _sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return statistics.median(times), statistics.mean(times), statistics.pstdev(times), times[0], times[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="RLWRLD/RLDX-1-FT-GR1")
    ap.add_argument("--obs", default="run_scripts/bench/gr1_sample_obs.pkl")
    ap.add_argument("--opt", choices=["eager", "compileB", "cudagraph", "triton", "steps",
                                      "msatRO", "msatMA", "headRO"],
                    default="eager")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--compile-mode", default="max-autotune")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-action-steps", type=int, default=16)
    ap.add_argument("--fixed-noise", action="store_true")
    args = ap.parse_args()

    dev = torch.cuda.get_device_name(0)
    print(f"Device: {dev} cap {torch.cuda.get_device_capability(0)} | torch {torch.__version__} | "
          f"opt={args.opt} | attn={os.environ.get('RLDX_ATTN_IMPL','(default)')} | "
          f"ptxas={os.environ.get('TRITON_PTXAS_PATH','(default)')}")

    with open(args.obs, "rb") as f:
        blob = pickle.load(f)
    observation, options = blob["observation"], blob["options"]

    print(f"Loading policy from {args.model_path} ...")
    policy = RLDXPolicy(model_path=args.model_path,
                        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT, device="cuda")
    am = policy.model.action_model
    action_dim = am.action_dim
    horizon = policy.model.config.action_horizon
    noise_shape = (1, horizon, action_dim)
    if args.fixed_noise:
        enable_fixed_noise(noise_shape)
        print(f"  fixed-noise ON for shape {noise_shape}")

    warm_opts = copy.deepcopy(options)
    if isinstance(warm_opts, dict) and "reset_memory" in warm_opts:
        warm_opts["reset_memory"] = [False] * len(warm_opts["reset_memory"])

    wrapped = RLDXSimPolicyWrapper(policy, strict=True)

    # ---- eager reference (in-process, before any mutation) ----
    print("Capturing eager reference action (seed", args.seed, ") ...")
    ref = get_seeded_action(wrapped, observation, warm_opts, args.seed)
    # eager latency baseline (quick)
    for _ in range(2):
        wrapped.get_action(observation, copy.deepcopy(warm_opts)); _sync()
    eager_med, *_ = median_latency(wrapped, observation, warm_opts, min(args.iters, 15), args.seed)
    print(f"  eager median latency: {eager_med:.1f} ms")

    # ---- apply the optimization ----
    if args.opt == "eager":
        pass
    elif args.opt == "steps":
        am.num_inference_timesteps = args.steps
        am.denoising_timesteps = None
        print(f"  set num_inference_timesteps = {args.steps}")
    elif args.opt in ("msatRO", "msatMA", "headRO"):
        # compile-driven CUDA graphs on the launch-bound diffusion head, keeping
        # real eager numerics (no GraphSafe reimplementation). use_physics=False
        # here so MSAT.forward has no per-step mutating closure hazard.
        mode = "reduce-overhead" if args.opt.endswith("RO") else "max-autotune"
        if args.opt == "headRO":
            am.action_encoder = torch.compile(am.action_encoder, mode=mode)
            am.model = torch.compile(am.model, mode=mode)
            am.action_decoder = torch.compile(am.action_decoder, mode=mode)
            print(f"  torch.compile(action_encoder+MSAT+action_decoder, mode={mode})")
        else:
            am.model = torch.compile(am.model, mode=mode)
            print(f"  torch.compile(MSAT only, mode={mode})")
    else:
        path = {"compileB": "B", "cudagraph": "C", "triton": "D"}[args.opt]
        from rldx.inference.serve_optimization import apply_optimization
        print(f"Applying optimization path={path} (mode={args.compile_mode}) ...")
        t0 = time.time()
        info = apply_optimization(policy, path=path, compile_mode=args.compile_mode)
        print(f"  apply_optimization returned in {time.time()-t0:.1f}s: {info}")

    # ---- warmup (compile / first-call capture happen here) ----
    print(f"Warmup x{args.warmup} (compile/capture) ...")
    t0 = time.time()
    for i in range(args.warmup):
        torch.manual_seed(args.seed)
        wrapped.get_action(observation, copy.deepcopy(warm_opts))
        _sync()
    print(f"  warmup done in {time.time()-t0:.1f}s")

    # ---- compare output ----
    got = get_seeded_action(wrapped, observation, warm_opts, args.seed)
    max_abs, rows = compare(ref, got)
    print("\n--- action output drift vs eager (seed {}) ---".format(args.seed))
    for k, mx, mn, rel in rows:
        print(f"  {k:22s} max|Δ|={mx:.4e}  mean|Δ|={mn:.4e}  rel={rel:.3e}")
    verdict = ("IDENTICAL" if max_abs == 0 else
               "fp-roundoff (output-preserving)" if max_abs < 5e-3 else
               "SMALL drift" if max_abs < 5e-2 else
               "CHANGED")
    print(f"  >>> max|Δ| over all keys = {max_abs:.4e}  -> {verdict}")

    # ---- latency ----
    print(f"\nEnd-to-end timing x{args.iters} ...")
    med, mean, std, lo, hi = median_latency(wrapped, observation, warm_opts, args.iters, args.seed)
    hz = 1000.0 / med
    req = CONTROL_HZ / args.n_action_steps
    speedup = eager_med / med if med > 0 else 0.0
    print("=" * 60)
    print(f"  opt={args.opt}  median {med:.1f} ms -> {hz:.3f} Hz   (eager {eager_med:.1f} ms, {speedup:.2f}x)")
    print(f"  mean {mean:.1f} +/- {std:.1f} | min/max {lo:.1f}/{hi:.1f}")
    print(f"  realtime @ {args.n_action_steps}-step: {'YES' if hz>=req else 'NO'} (need {req:.2f} Hz)")
    print(f"  output drift vs eager: max|Δ|={max_abs:.4e} ({verdict})")
    print("=" * 60)


if __name__ == "__main__":
    main()
