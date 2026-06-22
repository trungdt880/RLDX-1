> ⚠️ **CORRECTION (2026-06-21, after closed-loop eval).** The cudagraph (Path C)
> recommendation below is **WITHDRAWN**. A full GR-1 Tabletop eval (24 tasks × 5
> eps) measured **eager-fix 58.3% vs cudagraph 1.7%** success. Root cause:
> `_CompiledDispatcher._replay` replays with the *captured* `bi`, so Path C
> **freezes the camera frame + instruction at the first observation** — the
> robot keeps seeing frame 0. The 1.2e-2 micro-benchmark "drift" was an artifact
> of replaying the *same* frame; a single-observation latency test cannot
> validate output equivalence. The `--compile cudagraph` wiring has been
> reverted. **The validated win remains the fp32-Conv3d fix; no further
> output-preserving speedup is currently available on Thor.** See §6/§7 below for
> the original (now-superseded) numbers and the lesson.

# RLDX-1 inference speedup on Jetson Thor — session report

**Date:** 2026-06-20 · **Branch:** `fix/thor-vision-patchembed-fp32-conv3d`
**Hardware:** NVIDIA Jetson Thor (sm_110 / cap 11.0), CUDA 13.0, torch 2.9.0+cu130, aarch64
**Model:** `RLWRLD/RLDX-1-FT-GR1` · `RLDX_ATTN_IMPL=sdpa` · bf16
**Workload:** one real GR-1 Tabletop observation (`PnPCupToDrawerClose`), batch 1, 16-step action chunk

---

## 1. Executive summary

Starting from the just-landed fp32-Conv3d fix (which already took inference from **6161 ms → ~280 ms, 22×**), I investigated whether inference can go faster, implemented the change that works, and verified it barely moves the model's outputs.

**Achieved:**
- Re-profiled the post-fix path and showed it is now **launch-bound**, not compute-bound.
- Benchmarked every acceleration lever in the codebase (torch.compile submodule / CUDA graph / Triton chain / fewer denoising steps / partial reduce-overhead) for **both latency and exact output drift**.
- Found that **whole-forward CUDA-graph capture (Path C) is the only effective lever: 280 ms → 216 ms (1.28×, 3.35 → 4.6 Hz)** with negligible output change.
- Discovered and fixed a real gap: **Path C was unreachable from the server CLI** — the `--compile` flag only exposed the two paths that *don't* work on Thor.
- Verified output equivalence: the speedup changes actions by ≤1.2e-2 (≈0.1–0.5 % of range, within actuator noise).

**Net result:** GR-1 eval can now run at **4.6 Hz instead of 3.35 Hz on Thor** via one flag, with no meaningful behavior change.

---

## 2. Starting point — the conv fix (commit `3a7d6a8`)

Half-precision 3D convolution has no fast kernel on Thor sm_110, so under `autocast(bf16)` the Qwen3-VL vision **patch-embed Conv3d** dispatched to a ~2600× slower reference path. That single op was ~97 % of inference. The fix runs just that conv in fp32 with autocast disabled, keeping the rest bf16.

Re-confirmed on this box: end-to-end **~280 ms / 3.35 Hz**, vision encoder down to 52 ms (conv itself 1.8 ms). The big win was already captured; the question was what's left.

---

## 3. Where the remaining ~280 ms goes (re-profiled)

Median per `get_action`, with `num_inference_timesteps=4`, `rtc=none`, `use_memory/use_physics=False`, MSAT = 4 double + 8 single blocks, action_dim 64:

| stage | median | share |
|---|---|---|
| diffusion head (MSAT, 4 Euler steps × 28 ms) | 113.8 ms | 38 % |
| language model | 92.7 ms | 31 % |
| vision encoder (blocks ~50, conv 1.8) | 52.4 ms | 18 % |
| preprocess + decode + overhead | ~40 ms | 13 % |

The cost is spread across **~150+ small kernels** (27 ViT blocks + 18 LLM layers + 4×12 MSAT blocks). That signature — many tiny ops, none dominant — is **kernel-launch-bound**, which dictates which optimizations can help.

---

## 4. Methodology

- **Exact output comparison.** The model is bitwise-deterministic given a seed (verified: max|Δ| = 0 across two seeded runs). I seed before every call and, for graph/Triton paths, **pin the initial diffusion noise** with a fixed-noise monkeypatch (`optimize_compare.py`) so reported drift is the *true numerical effect* of the optimization, not a different random sample.
- **Fair latency.** In-process eager reference captured before any mutation; CUDA-synced medians over 20–30 steady-state iterations; warmup absorbs compile/capture.
- **Parallel static analysis.** Ran a multi-agent analysis workflow over the optimization framework with **adversarial verification** of every "output-preserving" claim. It correctly flagged the Path C/D `init_noise` resampling subtlety (which my fixed-noise harness already controls for) and the Path D sm_120-vs-sm_110 risk — but it *mispredicted* Path B as a 30 % win; the GPU measurement overruled it.

---

## 5. Results

All vs the same eager (post-conv-fix) baseline, fixed noise, seed 0:

| optimization | latency | speedup | max\|Δ\| vs eager | build time | verdict |
|---|---|---|---|---|---|
| eager (post conv-fix) | ~280 ms | 1.00× | 0 | — | baseline |
| **Path C — CUDA graph** | **216 ms** | **1.28×** | 1.2e-2 | 0.9 s | ✅ **only effective lever** |
| Path D — Triton `fullgraph` | 228 ms | 1.23× | 2.5e-2 | **442 s** | builds on sm_110 but slower than C + huge build |
| denoise steps 4→2 | 226 ms | 1.23× | 5.8e-2 | — | output-changing, dominated by C |
| denoise steps 4→3 | 252 ms | 1.11× | 2.2e-2 | — | output-changing |
| Path B — `submodule` compile | 271 ms | 1.03× | 2.0e-2 | — | ✗ HF KV-cache `recompile_limit` thrash |
| MSAT-only `reduce-overhead` | 270 ms | 1.03× | 1.3e-2 | — | ✗ partial graph misses distributed overhead |

**Why these outcomes:**
- The launch overhead is *distributed* across the whole forward, so **only a single whole-forward CUDA graph (Path C) removes it.** Every partial compile leaves the cross-module / per-step Python+launch glue intact → ~no gain.
- **Path B is doubly dead on Thor:** the HF `DynamicCache.is_initialized` guard blows Dynamo's `recompile_limit (8)`, so the compiled LLM falls back to eager. (This is *why* the old "compile does nothing" note still holds, now for a different reason than the conv.)
- **Path D works but isn't worth it on Thor:** its sm_120-tuned Triton kernels run on sm_110 but don't beat a generic graph, and it costs 442 s to build.
- **Fewer steps** is the only knob that *changes the policy*; it's dominated by Path C on both speed and fidelity. Use only if Path C is unavailable, and validate task success-rate, not just latency.

---

## 6. Output-equivalence verification (the "compare outputs" ask)

With identical seed and pinned initial noise, Path C's decoded action chunk differs from eager by:

```
action.left_arm    max|Δ|=4.4e-03  mean|Δ|=1.1e-03
action.left_hand   max|Δ|=1.2e-02  mean|Δ|=1.8e-03
action.right_arm   max|Δ|=6.5e-03  mean|Δ|=1.2e-03
action.right_hand  max|Δ|=9.9e-03  mean|Δ|=1.3e-03
action.waist       max|Δ|=8.9e-04  mean|Δ|=1.3e-04
>>> overall max|Δ| = 1.2e-02   (mean ~1e-3)
```

This is the GraphSafe **reimplementation's** bf16 rounding (not a different random sample — noise was pinned). At ≈0.1–0.5 % of the action range over a 16-step open-loop chunk at 20 Hz, it is within actuator noise; behavior is effectively unchanged.

---

## 7. What was shipped

**Tracked source change** (`rldx/eval/run_rldx_server.py`, +19/−6): exposed Path C through the server CLI.
- `--compile` literal extended to `{none, submodule, cudagraph, fullgraph}`.
- mapping `{none→A, submodule→B, cudagraph→C, fullgraph→D}` (Path C was previously **unreachable**; the existing `guided`-RTC validation already covered C).

**Untracked eval helper** (`run_scripts/eval/gr1_tabletop/eval_gr1_thor.sh`): opt-in `COMPILE` env (default off, so paper-comparable runs stay bit-faithful; auto-sets `TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas` for sm_11x Triton codegen).

**How to use the fast path:**
```bash
# GR-1 eval at 4.6 Hz on Thor:
COMPILE=cudagraph run_scripts/eval/gr1_tabletop/eval_gr1_thor.sh

# or run the server directly:
RLDX_ATTN_IMPL=sdpa TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas \
  .venv/bin/python rldx/eval/run_rldx_server.py \
  --model-path RLWRLD/RLDX-1-FT-GR1 --embodiment-tag GENERAL_EMBODIMENT \
  --use-sim-policy-wrapper --compile cudagraph
```

**Not committed** — changes sit in the working tree on the current branch, awaiting your go-ahead.

---

## 8. Artifacts produced (`run_scripts/bench/`)

| file | purpose |
|---|---|
| `optimize_probe.py` | config dump + fine-grained submodule breakdown + seeded reference action |
| `optimize_compare.py` | apply an optimization, compare latency **and** action drift vs eager (fixed-noise option) |
| `step_sweep.py` | denoising-step sweep (latency vs output drift) |
| `REPORT.md` | updated with the post-fix optimization section + reproduction commands |
| `ref_action_eager.npz` | saved eager reference action chunk (seed 0) |

---

## 9. Recommendations / next steps

1. **Adopt `--compile cudagraph` for Thor inference** (real-robot or wall-clock-bound eval). It is the only lever that helps and is output-safe.
2. **Smoke-test before a long run.** Path C captures a shape-specific graph; the dispatcher falls back to eager on shape drift, but a quick server+client check on the real env is wise before a multi-hour eval.
3. **Don't bother with `submodule`/`fullgraph` on Thor** — measured useless / not worth it respectively. Consider downgrading them to a warning on sm_11x.
4. **Keep `num_inference_timesteps=4`** unless a task-success ablation justifies fewer; step reduction is dominated by Path C.
5. **Offline sim eval doesn't need any of this** — it isn't wall-clock-bound; the value is for real-time / latency-sensitive deployment.

---

## 10. Honest limitations

- Benchmarked on **one** captured observation (representative of GR-1 Tabletop steady state). Latency is stable; Path C's graph is shape-specific.
- Path C's 1.2e-2 drift is a *reimplementation* difference, not bitwise — fine for control, but if you need bitwise-identical outputs, stay eager.
- (Resolved) The closed-loop success-rate eval was subsequently run — see §11.

---

## 11. Closed-loop validation (GR-1 Tabletop, 24 tasks × 5 eps = 120 ep/config)

Run 2026-06-21 to validate behavior, not just latency. Both fast configs ran ~1 h;
the original (~6 s/obs) ran ~6.5 h.

| config | inference | overall success | vs original |
|---|---|---|---|
| original (no conv fix, bf16 conv) | ~6 s | **60.8 %** (73/120) | — |
| **eager (conv fix, fp32 conv)** | ~280 ms | **58.3 %** (70/120) | **−2.5 pts (within noise)** |
| cudagraph (Path C) | ~216 ms | **1.7 %** (2/120) | **−59.2 pts (BROKEN)** |

**Conclusions:**
1. **The fp32-Conv3d fix is behavior-neutral.** 60.8 % → 58.3 % is within sampling
   noise (120 eps, SE ≈ 4.5 pts; diffusion noise unseeded; per-task swings of
   ±40 pts cancel out — e.g. CupToDrawer 100 %→60 % but CanToDrawer 40 %→80 %).
   Both bracket the paper's 58.7 %. So the 22× speedup costs nothing in task success.
2. **cudagraph (Path C) is broken in closed loop** — it freezes the VL input
   (camera frame + instruction) at the first observation (`_CompiledDispatcher._replay`
   re-feeds the captured `bi`). 1.7 % success. The 1.2e-2 single-obs "drift" in §6
   was an artifact of replaying the same frame.

**Is the −2.5 pts (fix vs original) a real regression? No — it's noise.** The eval
seeds only the environment, not torch, so diffusion sampling varies run-to-run.
Repeating each config:

| config | runs | pooled |
|---|---|---|
| eager-fix (fp32) | 58.3 / 57.5 / 59.2 % | **58.3 %** (210/360) |
| original (bf16) | 60.8 / 55.0 % | **57.9 %** (139/240) |

The original alone swings 5.8 pts (its 60.8 % first run was a lucky draw); pooled
means are statistically identical (Δ +0.4 pts, slightly favoring the fix), both ≈
the paper's 58.7 %. This matches the math: the fix does fp32 accumulation of the
*same* bf16 weights, strictly more accurate than the bf16 fallback it replaces, so
it cannot be systematically worse. **The fp32-Conv3d fix is behavior-neutral.**

**Lesson:** validate an inference optimization with the closed-loop success-rate
eval, never just single-observation latency + numeric drift; and for tight A/B
comparisons, seed torch per `get_action` (the eval currently seeds only the env).

Outputs: `output_final/gr1_tabletop/RLWRLD_RLDX-1-FT-GR1__{original_noconvfix,eager,cudagraph}/`.
Reproduce: `run_scripts/bench/run_gr1_eval_both.sh` + `run_gr1_eval_original.sh`.
