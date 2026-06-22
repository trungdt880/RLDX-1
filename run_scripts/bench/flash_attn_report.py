#!/usr/bin/env python3
"""Render the flash-attn-vs-SDPA comparison (Jetson Thor, sm_110) to a PDF.

Data comes from /home/thor/venv-fa/compare_attn.py (flash_attn 2.8.4 vs
torch-2.11 SDPA, same interpreter, RLDX-1 attention shapes). Numbers are the
measured medians; edit the tables below if you re-run the harness.

Usage:  .venv/bin/python run_scripts/bench/flash_attn_report.py [out.pdf]
"""
import sys
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = sys.argv[1] if len(sys.argv) > 1 else "flash_attn_thor_report.pdf"

SDPA = "#1f77b4"
FLASH = "#d62728"

# ---- measured data ---------------------------------------------------------
# Numerical equivalence: error vs fp32 SDPA reference (rel-RMS), + flash-sdpa ULP gap
NUM = [
    # label, dtype, sdpa_relrms, flash_relrms, fs_max, fs_mean
    ("backbone S=1024", "bf16", 3.322e-3, 3.322e-3, 3.906e-3, 2.387e-8),
    ("backbone S=1024", "fp16", 4.147e-4, 4.147e-4, 4.883e-4, 2.140e-8),
    ("backbone S=256", "bf16", 3.213e-3, 3.213e-3, 3.906e-3, 2.777e-8),
    ("action head S=256", "bf16", 3.655e-3, 3.655e-3, 9.766e-4, 2.691e-8),
    ("action head S=256", "fp16", 4.564e-4, 4.563e-4, 1.221e-4, 2.383e-8),
]

# Prefill (causal): seq, sdpa_ms, flash_ms
PREFILL = [(256, 0.1799, 0.2062), (512, 0.3079, 0.2620), (1024, 0.4559, 0.3892),
           (2048, 0.8905, 0.8909), (4096, 3.1375, 3.0299)]
# Decode (q_len=1 over KV cache): kv_len, sdpa_ms, flash_ms
DECODE = [(256, 0.0955, 0.1221), (512, 0.0614, 0.1067), (1024, 0.0612, 0.1090),
          (2048, 0.0563, 0.0995), (4096, 0.0692, 0.0940)]
# Action head (full attn, short): seq, sdpa_ms, flash_ms
ACTION = [(64, 0.0399, 0.0713), (128, 0.0374, 0.0668), (256, 0.0381, 0.0642)]


def latency_plot(ax, data, title, xlabel):
    xs = [d[0] for d in data]
    s = [d[1] for d in data]
    f = [d[2] for d in data]
    idx = range(len(xs))
    w = 0.38
    ax.bar([i - w / 2 for i in idx], s, w, label="SDPA (torch 2.11)", color=SDPA)
    ax.bar([i + w / 2 for i in idx], f, w, label="flash-attn 2.8.4", color=FLASH)
    ax.set_xticks(list(idx))
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("median latency (ms)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def speedup_plot(ax):
    groups = []
    labels = []
    for seq, s, f in PREFILL:
        groups.append(s / f); labels.append(f"PF {seq}")
    for kv, s, f in DECODE:
        groups.append(s / f); labels.append(f"DEC {kv}")
    for seq, s, f in ACTION:
        groups.append(s / f); labels.append(f"AH {seq}")
    colors = ["#2ca02c" if g >= 1 else "#d62728" for g in groups]
    idx = range(len(groups))
    ax.bar(idx, groups, color=colors)
    ax.axhline(1.0, color="black", lw=1, ls="--")
    ax.text(len(groups) - 0.5, 1.02, "parity (no change)", ha="right", fontsize=8)
    for i, g in enumerate(groups):
        ax.text(i, g + 0.02, f"{g:.2f}", ha="center", fontsize=7)
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("flash-attn speedup over SDPA\n(>1 = flash faster)")
    ax.set_title("Speedup by regime  —  PF=prefill, DEC=decode, AH=action head",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.5)
    ax.grid(axis="y", alpha=0.3)


def main():
    with PdfPages(OUT) as pdf:
        # ---- Page 1: title + summary + numerical table ----
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle("flash-attn vs PyTorch SDPA on Jetson Thor (sm_110)",
                     fontsize=15, fontweight="bold", y=0.97)
        fig.text(0.5, 0.935,
                 f"RLDX-1 attention shapes  ·  flash_attn 2.8.4 vs torch 2.11 SDPA  ·  {date.today()}",
                 ha="center", fontsize=10, color="#444")

        fig.text(0.07, 0.905, "Setup", fontsize=12, fontweight="bold")
        fig.text(0.07, 0.885,
                 "• Both kernels run in ONE interpreter (venv-fa: py3.12 / torch 2.11.0 / flash_attn 2.8.4)\n"
                 "  → identical input tensors, true apples-to-apples.\n"
                 "• Shapes mirror the real model: Qwen3-VL backbone = 32 query / 8 KV heads (GQA),\n"
                 "  head_dim 128, 36 layers;  MSAT diffusion action head = 24 heads, head_dim 64.\n"
                 "• Device: NVIDIA Thor, compute capability (11, 0), CUDA 13.",
                 fontsize=9, va="top", family="monospace")

        fig.text(0.07, 0.78, "1.  Numerical equivalence (error vs fp32 SDPA reference)",
                 fontsize=12, fontweight="bold")
        col = ["shape", "dtype", "SDPA rel-RMS", "flash rel-RMS", "flash-SDPA max", "flash-SDPA mean"]
        cells = [[lab, dt, f"{a:.2e}", f"{b:.2e}", f"{c:.2e}", f"{d:.2e}"]
                 for (lab, dt, a, b, c, d) in NUM]
        ax = fig.add_axes([0.07, 0.60, 0.86, 0.15]); ax.axis("off")
        t = ax.table(cellText=cells, colLabels=col, loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.4)
        for j in range(len(col)):
            t[0, j].set_facecolor("#dddddd"); t[0, j].set_text_props(fontweight="bold")

        fig.text(0.07, 0.55, "Finding:", fontsize=10, fontweight="bold", color="#2ca02c")
        fig.text(0.155, 0.55,
                 "the two kernels match the fp32 reference to identical precision; their mutual\n"
                 "difference is one bf16/fp16 ULP (mean ~2e-8 ≈ 0). Swapping attention impl does NOT\n"
                 "change model outputs — accuracy is set purely by bf16/fp16 rounding, shared by both.",
                 fontsize=9, va="top")

        fig.text(0.07, 0.46, "Verdict", fontsize=12, fontweight="bold")
        fig.text(0.07, 0.43,
                 "Numerically interchangeable; performance gives no net win (see next page). flash-attn\n"
                 "wins only mid-length prefill (~1.17x at 512-1024) but LOSES 1.3-1.8x on decode and on\n"
                 "the short action-head sequences — exactly the paths that dominate RLDX per-step inference.\n"
                 "It also requires py3.12 + torch 2.11, breaking RLDX's py3.10 / torch 2.7-2.9 pin.\n\n"
                 "→  Keep RLDX_ATTN_IMPL=sdpa.  The real Thor latency lever is --compile submodule.",
                 fontsize=9.5, va="top")
        fig.text(0.07, 0.08,
                 "Reproduce:  /home/thor/venv-fa/run.sh python /home/thor/venv-fa/compare_attn.py\n"
                 "Report gen: .venv/bin/python run_scripts/bench/flash_attn_report.py",
                 fontsize=8, va="top", family="monospace", color="#666")
        pdf.savefig(fig); plt.close(fig)

        # ---- Page 2: speed figures ----
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("2.  Speed  —  median latency & speedup (Jetson Thor sm_110)",
                     fontsize=14, fontweight="bold")
        latency_plot(axes[0, 0], PREFILL, "Backbone prefill (causal, GQA 32/8, d128)", "sequence length")
        latency_plot(axes[0, 1], DECODE, "Backbone decode (q_len=1 over KV cache)", "KV-cache length")
        latency_plot(axes[1, 0], ACTION, "Action head (full attn, 24h, d64)", "sequence length")
        speedup_plot(axes[1, 1])
        fig.text(0.5, 0.005,
                 "Decode and short action-head sequences dominate RLDX inference — and there SDPA is "
                 "1.3-1.8x faster than flash-attn.",
                 ha="center", fontsize=9, style="italic")
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
