#!/usr/bin/env python3
"""Render the real-data RLDX-1 inference benchmark (GR-1 Tabletop, Thor) to PDF.

Numbers are measured medians from the harness in this directory
(bench_gr1_inference.py / profile_gr1_stages.py + inline submodule probes).
Edit the constants below if you re-run. See REPORT.md for the full writeup.

Usage:  .venv/bin/python run_scripts/bench/gr1_inference_report.py [out.pdf]
"""
import sys
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = sys.argv[1] if len(sys.argv) > 1 else "gr1_inference_thor_report.pdf"

# ---- measured data (Jetson Thor sm_110, RLDX-1-FT-GR1, real GR-1 obs) -------
E2E_SDPA_MS = 6161.4     # as shipped: eager, RLDX_ATTN_IMPL=sdpa, bf16
E2E_FIX_MS = 281.0       # patch-embed conv forced to fp32 (autocast off)
E2E_COMPILE_MS = 6240.1  # --compile submodule (path B)
CONTROL_HZ = 20.0
N_ACTION = 16

# pipeline stages (median ms)
STAGES = [("preprocess\n(Qwen3-VL processor)", 17.2),
          ("RTC prefix", 0.05),
          ("model forward", 6157.2),
          ("action decode", 0.5)]

# model-forward split (median ms)
FORWARD = [("vision encoder\n(ViT 27blk, 4 frames)", 5960.3),
           ("language model\n(18/36 blk)", 96.4),
           ("diffusion head\n(MSAT)", 112.1)]

# inside the vision encoder (median ms)
VISION = [("patch_embed\n(Conv3d)", 5866.4),
          ("27 transformer\nblocks", 48.0),
          ("rotary / pos emb", 0.3)]

# the Conv3d patch-embed in isolation, by dtype (median ms)
CONV_DTYPE = [("bf16", 5670.4), ("fp16", 5716.1), ("fp32", 2.2)]

GREEN, RED, BLUE, ORANGE, GREY = "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#888"


def main():
    e2e_hz = 1000.0 / E2E_SDPA_MS
    fix_hz = 1000.0 / E2E_FIX_MS
    need_hz = CONTROL_HZ / N_ACTION
    with PdfPages(OUT) as pdf:
        # ===================== Page 1: headline + where time goes ===========
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle("RLDX-1 inference on real GR-1 Tabletop data — Jetson Thor (sm_110)",
                     fontsize=13.5, fontweight="bold", y=0.975)
        fig.text(0.5, 0.945,
                 f"RLWRLD/RLDX-1-FT-GR1  ·  RLDX_ATTN_IMPL=sdpa  ·  bf16  ·  {date.today()}",
                 ha="center", fontsize=9.5, color="#444")

        fig.text(0.07, 0.915, "Setup", fontsize=12, fontweight="bold")
        fig.text(0.07, 0.895,
                 "• One REAL observation captured from the live mujoco rollout (PnPCupToDrawerClose):\n"
                 "  ego-view camera 1x4x256x256x3 uint8 + proprio state (arms/hands/waist) + task text.\n"
                 "• Latency = full per-query pipeline: preprocess -> Qwen3-VL backbone -> MSAT\n"
                 "  diffusion action head -> decode. Median of 30 steady-state calls, CUDA-synced.\n"
                 "• One query returns a 16-step action chunk, executed open-loop; sim control = 20 Hz.",
                 fontsize=9, va="top", family="monospace")

        # headline
        fig.text(0.07, 0.80, "Headline", fontsize=12, fontweight="bold")
        fig.text(0.07, 0.775,
                 f"{E2E_SDPA_MS/1000:.2f} s / obs   →   {e2e_hz:.3f} Hz",
                 fontsize=15, fontweight="bold", color=RED)
        fig.text(0.55, 0.775,
                 f"with 1-line fix:  {E2E_FIX_MS/1000:.2f} s   →   {fix_hz:.2f} Hz",
                 fontsize=13, fontweight="bold", color=GREEN)
        fig.text(0.07, 0.745,
                 f"Action chunking: 1 query / {N_ACTION} steps  →  policy must run at {need_hz:.2f} Hz "
                 f"to feed a {CONTROL_HZ:.0f} Hz controller.\n"
                 f"As shipped: {e2e_hz:.3f} Hz, ~{need_hz/e2e_hz:.1f}x too slow (offline sim eval is fine — "
                 f"not wall-clock bound).\n"
                 f"After the fix (page 2): {fix_hz:.2f} Hz — {E2E_SDPA_MS/E2E_FIX_MS:.0f}x faster, clears "
                 f"real-time ({fix_hz:.2f} > {need_hz:.2f} Hz).",
                 fontsize=9.5, va="top")

        # ---- pipeline stage breakdown ----
        fig.text(0.07, 0.645, "Where the time goes", fontsize=12, fontweight="bold")
        ax1 = fig.add_axes([0.42, 0.45, 0.50, 0.15])
        names = [s[0] for s in STAGES][::-1]
        vals = [s[1] for s in STAGES][::-1]
        cols = [GREEN if v < 100 else RED for v in vals]
        ax1.barh(range(len(names)), vals, color=cols)
        ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names, fontsize=7.5)
        ax1.set_xscale("log"); ax1.set_xlabel("median ms (log)", fontsize=8)
        ax1.set_title("Pipeline stages", fontsize=10, fontweight="bold")
        for i, v in enumerate(vals):
            ax1.text(v * 1.15, i, f"{v:.1f} ms", va="center", fontsize=7.5)
        ax1.set_xlim(0.03, 60000); ax1.grid(axis="x", alpha=0.3)
        fig.text(0.07, 0.59,
                 "The model forward is\n99.7% of the latency.\n"
                 "Everything else is <18 ms.",
                 fontsize=8.5, va="top")

        # ---- model-forward split ----
        ax2 = fig.add_axes([0.47, 0.225, 0.45, 0.15])
        fwd_total = sum(f[1] for f in FORWARD)
        fn = [f[0] for f in FORWARD][::-1]
        fv = [f[1] for f in FORWARD][::-1]
        fc = [ORANGE, BLUE, RED][::-1]
        ax2.barh(range(len(fn)), fv, color=fc)
        ax2.set_yticks(range(len(fn))); ax2.set_yticklabels(fn, fontsize=7)
        ax2.set_xscale("log"); ax2.set_xlabel("median ms (log)", fontsize=8)
        ax2.set_title("Inside the model forward", fontsize=10, fontweight="bold")
        for i, v in enumerate(fv):
            ax2.text(v * 1.15, i, f"{v:.0f} ms ({100*v/fwd_total:.1f}%)", va="center", fontsize=7.5)
        ax2.set_xlim(30, 60000); ax2.grid(axis="x", alpha=0.3)

        ax3 = fig.add_axes([0.10, 0.225, 0.22, 0.15])
        ax3.bar(["eager\n(sdpa)", "compile\nsubmod"], [E2E_SDPA_MS, E2E_COMPILE_MS],
                color=[BLUE, GREY], width=0.6)
        ax3.set_ylabel("median ms", fontsize=8)
        ax3.set_title("compile: no gain", fontsize=9.5, fontweight="bold")
        for i, v in enumerate([E2E_SDPA_MS, E2E_COMPILE_MS]):
            ax3.text(i, v + 80, f"{v:.0f}", ha="center", fontsize=7.5)
        ax3.set_ylim(0, 7600); ax3.tick_params(labelsize=7.5); ax3.grid(axis="y", alpha=0.3)

        fig.text(0.07, 0.16,
                 "The vision encoder is ~97% of inference — which is surprising, because it is the SMALL\n"
                 "part of the model (576 M params, 27 blocks, only 1024 patch tokens). Page 2 shows why:\n"
                 "it is not the encoder compute at all.",
                 fontsize=9.5, va="top")
        fig.text(0.07, 0.05,
                 "Reproduce:  RLDX_ATTN_IMPL=sdpa .venv/bin/python run_scripts/bench/bench_gr1_inference.py\n"
                 "Stages:     .venv/bin/python run_scripts/bench/profile_gr1_stages.py\n"
                 "Demo:       .venv/bin/python run_scripts/bench/infer_gr1.py   ·   Writeup: REPORT.md",
                 fontsize=7.5, va="top", family="monospace", color="#666")
        pdf.savefig(fig); plt.close(fig)

        # ===================== Page 2: root cause + fix =====================
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle("Why the \"small\" vision encoder takes 6 s — and the fix",
                     fontsize=14, fontweight="bold", y=0.965)

        fig.text(0.07, 0.915, "It is not the encoder. It is one Conv3d.", fontsize=12,
                 fontweight="bold")
        fig.text(0.07, 0.895,
                 "The ViT is small: 576 M params, 27 blocks, hidden 1152, and the real input is just\n"
                 "1024 patch tokens (4 frames x 16x16 -> 256 after the 2x2 merge). Timing inside it:",
                 fontsize=9.5, va="top")

        # vision-internal split
        ax = fig.add_axes([0.30, 0.70, 0.62, 0.14])
        vn = [v[0] for v in VISION][::-1]
        vv = [v[1] for v in VISION][::-1]
        vc = [RED, GREEN, GREEN][::-1]
        ax.barh(range(len(vn)), vv, color=vc)
        ax.set_yticks(range(len(vn))); ax.set_yticklabels(vn, fontsize=8)
        ax.set_xscale("log"); ax.set_xlabel("median ms (log)", fontsize=8)
        ax.set_title("Inside the vision encoder", fontsize=10, fontweight="bold")
        for i, v in enumerate(vv):
            ax.text(v * 1.18, i, f"{v:.1f} ms", va="center", fontsize=8)
        ax.set_xlim(0.1, 30000); ax.grid(axis="x", alpha=0.3)

        fig.text(0.07, 0.64,
                 "The 27 transformer blocks total ~48 ms. The entire 6 s is the patch_embed —\n"
                 "the pixel→patch projection, a Conv3d(3, 1152, kernel=(2,16,16), stride=(2,16,16)).",
                 fontsize=9.5, va="top")

        # conv dtype bar (the killer)
        ax = fig.add_axes([0.12, 0.40, 0.45, 0.17])
        cn = [c[0] for c in CONV_DTYPE]
        cv = [c[1] for c in CONV_DTYPE]
        bars = ax.bar(cn, cv, color=[RED, RED, GREEN], width=0.6)
        ax.set_yscale("log")
        ax.set_ylabel("Conv3d latency, ms (log)", fontsize=8)
        ax.set_title("Same conv, by precision", fontsize=10, fontweight="bold")
        for b, v in zip(bars, cv):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.2, f"{v:.1f}", ha="center", fontsize=8.5)
        ax.set_ylim(1, 20000); ax.grid(axis="y", alpha=0.3)

        # before/after bar
        ax = fig.add_axes([0.66, 0.40, 0.27, 0.17])
        bars = ax.bar(["as\nshipped", "fp32\nconv"], [E2E_SDPA_MS, E2E_FIX_MS],
                      color=[RED, GREEN], width=0.6)
        ax.set_ylabel("end-to-end ms", fontsize=8)
        ax.set_title(f"Fix: {E2E_SDPA_MS/E2E_FIX_MS:.0f}x", fontsize=10, fontweight="bold")
        for b, v in zip(bars, [E2E_SDPA_MS, E2E_FIX_MS]):
            ax.text(b.get_x() + b.get_width() / 2, v + 120, f"{v:.0f}", ha="center", fontsize=8)
        ax.set_ylim(0, 7000); ax.grid(axis="y", alpha=0.3)

        fig.text(0.07, 0.345, "Root cause", fontsize=12, fontweight="bold")
        fig.text(0.07, 0.32,
                 "Half-precision 3D convolution has NO working kernel on Thor (sm_110). Under the model's\n"
                 "torch.autocast(bf16) the patch-embed Conv3d falls back to a ~2600x slower reference path\n"
                 "(bf16 5670 ms / fp16 5716 ms vs fp32 2.2 ms). cudnn enabled/disabled/benchmark: no change.\n"
                 "This is a CUDA/cuDNN-on-Blackwell-Thor gap, unrelated to RLDX or model size. It also\n"
                 "explains the earlier negatives: flash-attn is irrelevant (attention <2%), and torch.compile\n"
                 "leaves the vision tower eager so the conv fallback still dominates.",
                 fontsize=9, va="top")

        fig.text(0.07, 0.20, "The fix  (21.9x  →  3.56 Hz, real-time-capable)", fontsize=12,
                 fontweight="bold")
        fig.text(0.07, 0.175,
                 "Run only the patch-embed conv in fp32 with autocast off; keep the rest of the model bf16:",
                 fontsize=9, va="top")
        code = (
            "def forward(self, hidden_states):            # Qwen3VLVisionPatchEmbed\n"
            "    x = hidden_states.view(-1, 3, 2, 16, 16)\n"
            "    with torch.autocast(device_type='cuda', enabled=False):  # Thor: no fast half conv3d\n"
            "        out = self.proj.float()(x.float()).view(-1, self.embed_dim)\n"
            "    return out.to(hidden_states.dtype)")
        fig.text(0.07, 0.14, code, fontsize=8.5, va="top", family="monospace",
                 bbox=dict(boxstyle="round", facecolor="#f3f3f3", edgecolor="#ccc"))
        fig.text(0.07, 0.045,
                 "Conv is non-overlapping (stride == kernel), so an unfold+bf16-matmul rewrite also works and\n"
                 "stays in bf16; the fp32 cast is just the smallest change. Numerics unchanged (bf16-level).",
                 fontsize=9, va="top", style="italic")
        pdf.savefig(fig); plt.close(fig)

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
