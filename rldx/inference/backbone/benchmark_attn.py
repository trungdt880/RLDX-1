"""FlashAttention-4 vs SDPA backbone-attention benchmark (Jetson Thor).

Answers "how fast is FA4 on Thor": times the eager backbone forward under one
attention backend and, optionally, checks that its output numerically agrees
with a previously-saved reference run (e.g. SDPA).

FA4 only affects the *eager* backbone attention (the accelerated Path C/D use
custom Triton kernels, not flash-attn), so this benchmark measures the eager
forward — the path FA4 actually changes.

The attention backend is selected by ``RLDX_ATTN_IMPL``, which the adapter
reads once at import time, so a clean A/B needs one process per impl. Run twice
and compare via the saved reference:

    pixi run -e thor python rldx/inference/backbone/benchmark_attn.py \
        --impl sdpa --out /tmp/sdpa.pt
    pixi run -e thor python rldx/inference/backbone/benchmark_attn.py \
        --impl flash_attention_4 --out /tmp/fa4.pt --compare-to /tmp/sdpa.pt

The second invocation prints FA4 latency AND max-abs-diff vs the SDPA run.
"""

import argparse
import ctypes
import os
import sys


# Fix NVRTC builtins path (mirrors benchmark_backbone.py; cu13 toolchain).
try:
    import nvidia.cu13 as _cu13

    _cu13_lib = os.path.join(os.path.dirname(os.path.abspath(_cu13.__path__[0])), "cu13", "lib")
    if os.path.isdir(_cu13_lib):
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _cu13_lib not in _ld:
            os.environ["LD_LIBRARY_PATH"] = f"{_cu13_lib}:{_ld}" if _ld else _cu13_lib
        _builtins = os.path.join(_cu13_lib, "libnvrtc-builtins.so.13.0")
        if os.path.isfile(_builtins):
            ctypes.CDLL(_builtins)
except (ImportError, OSError):
    pass


_MODE_TO_MODEL_TYPE = {
    "video": "rldx_1_pretrain",
    "all": "rldx_1_midtrain_allex",
}


def parse_args():
    parser = argparse.ArgumentParser(description="FA4 vs SDPA backbone attention benchmark")
    parser.add_argument(
        "--impl",
        default="flash_attention_4",
        help="RLDX_ATTN_IMPL value for this run (flash_attention_4 | sdpa | flash_attention_2).",
    )
    parser.add_argument("--out", type=str, default=None, help="Save this run's output for compare.")
    parser.add_argument(
        "--compare-to",
        type=str,
        default=None,
        help="Reference .pt from a prior run; report max-abs-diff vs it.",
    )
    parser.add_argument(
        "--tol", type=float, default=2e-2, help="Max-abs-diff PASS threshold (bf16)."
    )
    parser.add_argument("--mode", default="video", choices=list(_MODE_TO_MODEL_TYPE.keys()))
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--concat-frames", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--iter", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cog-tokens", action="store_true")
    parser.add_argument("--n-cog-tokens", type=int, default=64)
    args = parser.parse_args()
    args.model_type = _MODE_TO_MODEL_TYPE[args.mode]
    args.model_path = None
    return args


def main():
    args = parse_args()

    # MUST be set before importing the rldx backbone: the adapter resolves the
    # attention impl from this env var at import time.
    os.environ["RLDX_ATTN_IMPL"] = args.impl

    import torch

    # Path setup (mirror benchmark_backbone.py) so ``from utils import ...``
    # resolves to rldx/inference/utils.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import _path  # noqa: E402

    _path.setup(__file__)

    from utils import (  # noqa: E402
        generate_synthetic_input,
        load_backbone,
        measure_times,
        print_latency_table,
    )

    backbone, meta = load_backbone(args)
    device = meta["device"]

    # Confirm the impl actually took effect (the adapter falls back to sdpa when
    # FA4 is unavailable — surface that instead of silently mis-reporting).
    effective = getattr(backbone.qwen_model.config, "_attn_implementation", "?")
    print(
        f"\nRequested RLDX_ATTN_IMPL={args.impl!r} -> effective _attn_implementation={effective!r}"
    )
    if args.impl in ("flash_attention_4", "fa4") and effective == "sdpa":
        print("  [w] FA4 was requested but fell back to SDPA on this host.")

    processor_path = meta["model_cfg"].get(
        "processor_path", args.model_path or meta["model_cfg"]["hf_path"]
    )
    print("Generating input...")
    vl_input, input_info = generate_synthetic_input(
        processor_path,
        args.num_images,
        args.image_size,
        args.image_size,
        args.concat_frames,
        device,
        args.seed,
        custom_prompt=args.prompt,
    )
    print(
        f"  seq_len={input_info.get('seq_len', '?')} "
        f"(vision={input_info.get('vision_tokens', '?')}, "
        f"prompt_tokens={input_info.get('prompt_tokens', '?')})"
    )

    def fwd():
        with torch.no_grad():
            return backbone(vl_input)["backbone_features"]

    out = fwd()
    if torch.isnan(out).any() or torch.isinf(out).any():
        print("  [!] NaN/Inf in output")
    ref_out = out.detach().float().cpu()

    print(f"Warming up ({args.warmup})...")
    for _ in range(args.warmup):
        fwd()
    torch.cuda.synchronize()
    print(f"Benchmarking ({args.iter})...")
    times = measure_times(fwd, args.iter)

    print_latency_table(f"Backbone eager forward [{effective}]", {args.impl: times})
    print(
        f"Output shape: {list(ref_out.shape)}  peak GPU mem: "
        f"{torch.cuda.max_memory_allocated(device) / (1024**2):.1f} MB"
    )

    if args.out:
        torch.save({"impl": args.impl, "effective": effective, "features": ref_out}, args.out)
        print(f"Saved reference -> {args.out}")

    if args.compare_to:
        ref = torch.load(args.compare_to, map_location="cpu", weights_only=False)
        other = ref["features"]
        if other.shape != ref_out.shape:
            print(f"  [!] shape mismatch {list(other.shape)} vs {list(ref_out.shape)} — skip diff")
        else:
            diff = (ref_out - other).abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            cos = torch.nn.functional.cosine_similarity(
                ref_out.flatten(), other.flatten(), dim=0
            ).item()
            verdict = "PASS" if max_abs <= args.tol else "FAIL"
            print(
                f"\nCorrectness {args.impl!r} vs {ref['impl']!r} ({args.compare_to}):\n"
                f"  max_abs_diff={max_abs:.4e} mean_abs_diff={mean_abs:.4e} "
                f"cosine={cos:.6f}  [{verdict} @ tol={args.tol:g}]"
            )


if __name__ == "__main__":
    main()
