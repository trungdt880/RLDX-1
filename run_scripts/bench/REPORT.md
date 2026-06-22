# RLDX-1 inference on real GR-1 Tabletop data — Jetson Thor (sm_110)

**Model:** `RLWRLD/RLDX-1-FT-GR1` · **attn:** `RLDX_ATTN_IMPL=sdpa` · bf16 · main venv (py3.10 / torch 2.9 cu130)
**Observation:** one *real* step captured from the live mujoco rollout (`PnPCupToDrawerClose`,
*"pick up the cup, place it into the drawer and close the drawer"*): ego-view camera `1×4×256×256×3`
uint8 + arm/hand/waist proprio + task text.

## Headline

| | latency / observation | throughput |
|---|---|---|
| **as shipped (bf16)** | **6.15 s** | **0.163 Hz** |
| **with 1-line patch-embed fix** | **0.28 s** | **3.56 Hz**  (**21.9× faster**) |

One policy query returns a **16-step action chunk** executed open-loop. The GR-1 sim runs at 20 Hz
control, so the policy only needs to run at **20 / 16 = 1.25 Hz**. As shipped Thor is ~7.7× too slow
for real-time (offline sim eval is unaffected — it isn't wall-clock-bound). **With the fix it clears
real-time** (3.56 Hz > 1.25 Hz).

## Where the 6.15 s goes

Profiled by patching `PolicyRuntime` stages and model submodules (median of N steady-state calls,
CUDA-synced):

| stage | median |
|---|---|
| preprocess (Qwen3-VL processor) | 17 ms |
| RTC prefix | 0 ms |
| **model forward** | **6157 ms (99.7%)** |
| action decode | 0.5 ms |

Inside the model forward:

| submodule | median | share |
|---|---|---|
| **Qwen3-VL vision encoder** | **5960 ms** | **96.6%** |
| language model (18 of 36 blocks kept, GQA) | 96 ms | 1.6% |
| MSAT diffusion action head | 112 ms | 1.8% |

## "But the vision encoder is small" — yes it is. It's not the encoder.

The ViT is only **576 M params, 27 blocks, hidden 1152**, and the real input is just **1024 patch
tokens** (4 frames × 16×16 → 256 after the 2×2 merge). Timing *inside* the encoder:

| component | median |
|---|---|
| `rotary_pos_emb` / `pos_embed` | ~0.1 ms |
| one transformer block | 1.78 ms → **27 blocks ≈ 48 ms** |
| **`patch_embed` (pixel→patch projection)** | **5866 ms** |

**The whole 6 s is the patch-embed.** It's a `Conv3d(3, 1152, kernel=(2,16,16), stride=(2,16,16))`.
Isolating it:

| Conv3d dtype | latency |
|---|---|
| **bf16** | **5670 ms** |
| **fp16** | **5716 ms** |
| **fp32** | **2.2 ms** |

`torch.backends.cudnn` enabled / disabled / `benchmark=True` make no difference.

### Root cause

**Half-precision 3D convolution has no working kernel on Thor (sm_110).** Under the model's
`torch.autocast(bf16)`, the patch-embed `Conv3d` dispatches to a ~2600× slower reference/fallback
path. The fp32 `Conv3d` has a proper kernel (2.2 ms). This is a CUDA/cuDNN-on-Blackwell-Thor gap, not
anything about RLDX or the model size. It also explains why the earlier experiments did nothing:

- **flash-attn is irrelevant** — attention is <2% of the cost; the bottleneck is a conv, not attention.
- **`torch.compile` (`--compile submodule`) gives nothing** (6240 ms, 1.5% *slower*) — the vision
  tower is left eager by path B, and the conv fallback dominates regardless.

### The fix (21.9×)

Run just the patch-embed conv in fp32 with autocast disabled; keep the rest of the model bf16:

```python
# rldx/.../Qwen3VLVisionPatchEmbed.forward equivalent
def forward(self, hidden_states):
    x = hidden_states.view(-1, 3, 2, 16, 16)
    with torch.autocast(device_type="cuda", enabled=False):   # Thor sm_110 has no fast half conv3d
        out = self.proj.float()(x.float()).view(-1, self.embed_dim)
    return out.to(hidden_states.dtype)
```

Measured end-to-end: **6161 ms → 281 ms** (0.162 → 3.56 Hz). The conv is non-overlapping
(stride == kernel), so an `unfold`+bf16-matmul rewrite would also work and stay in bf16; the fp32
cast is the smallest change.

## Further speedup after the conv fix (launch-bound regime)

> ⚠️ **CORRECTION (2026-06-21):** the Path C / cudagraph recommendation in this
> section is **WITHDRAWN**. Closed-loop GR-1 Tabletop eval: eager-fix **58.3%**
> vs cudagraph **1.7%**. Path C freezes the VL input (camera frame + instruction)
> at the first observation (`_CompiledDispatcher._replay` re-feeds the captured
> `bi`), so it is broken in closed-loop. The single-obs drift of 1.2e-2 below was
> measured by replaying the *same* frame and does NOT reflect closed-loop
> behavior. `--compile cudagraph` wiring reverted. The conv fix remains the only
> validated speedup on Thor.

With the conv fixed, the ~280 ms is now spread across many small kernels and is **launch-bound**, not compute-bound. Re-measured breakdown (median/obs, real GR-1 obs, `num_inference_timesteps=4`):

| stage | median | share |
|---|---|---|
| diffusion head (MSAT, 4 Euler steps × 28 ms) | 113.8 ms | 38% |
| language model | 92.7 ms | 31% |
| vision encoder (blocks ~50, conv 1.8) | 52.4 ms | 18% |
| preprocess + decode + overhead | ~40 ms | 13% |

The model is **bitwise-deterministic given a seed** (max\|Δ\|=0 across two seeded runs), so output drift below is the *true* numerical effect of each optimization (with the initial diffusion noise pinned via a fixed-noise monkeypatch — see `optimize_compare.py`).

| optimization | latency | speedup | max\|Δ\| vs eager | build | verdict |
|---|---|---|---|---|---|
| eager (post conv-fix) | ~280 ms | 1.00× | 0 | — | baseline |
| **Path C — CUDA graph** (`apply_optimization "C"`) | **216 ms** | **1.28×** | 1.2e-2 | 0.9 s | ✅ only effective lever |
| Path D — Triton `fullgraph` | 228 ms | 1.23× | 2.5e-2 | **442 s** | builds on sm_110 but slower than C + huge build |
| `num_inference_timesteps` 4→2 | 226 ms | 1.23× | 5.8e-2 | — | output-changing, dominated by C |
| `num_inference_timesteps` 4→3 | 252 ms | 1.11× | 2.2e-2 | — | output-changing |
| Path B — `submodule` compile | 271 ms | 1.03× | 2.0e-2 | — | ✗ HF KV-cache `recompile_limit` thrash |
| MSAT-only `reduce-overhead` | 270 ms | 1.03× | 1.3e-2 | — | ✗ per-submodule graph misses the *distributed* launch overhead |

**Key conclusions:**
- The launch overhead is spread across ~150+ kernels (27 ViT blocks + 18 LLM layers + 4×12 MSAT blocks). **Only capturing the *whole* forward as one CUDA graph (Path C) removes it** — every *partial* compile (Path B, single-submodule `reduce-overhead`) gives ~nothing because the cross-module / per-step Python+launch overhead remains.
- `torch.compile` (Path B) is doubly dead on Thor: the HF `DynamicCache` `is_initialized` guard blows Dynamo's `recompile_limit (8)` so the LLM falls back to eager.
- Path C's 1.2e-2 drift is from the GraphSafe *reimplementation* of the forward (not RNG — noise was pinned). It is ≈0.5% of the action range worst-case, ~0.1% mean — within actuator noise over a 16-step open-loop chunk.
- Reducing denoising steps is the only knob that *changes* the policy (5.8e-2 at 2 steps) and is dominated by Path C on both speed and fidelity — only use it if Path C is unavailable, and validate task success-rate, not just latency.

**Gap found & fixed:** the server's `--compile` flag mapped only `{none→A, submodule→B, fullgraph→D}` — i.e. Path C (the one lever that works on Thor) was **unreachable**, and the two reachable options are exactly the two that don't help on sm_110. Added `--compile cudagraph → C` in `rldx/eval/run_rldx_server.py`, and an opt-in `COMPILE` env in `eval_gr1_thor.sh` (off by default; sets `TRITON_PTXAS_PATH` automatically).

```bash
# fast-path GR-1 eval on Thor (1.28x, 4.6 Hz):
COMPILE=cudagraph run_scripts/eval/gr1_tabletop/eval_gr1_thor.sh
# or directly:
RLDX_ATTN_IMPL=sdpa TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas \
  .venv/bin/python rldx/eval/run_rldx_server.py --model-path RLWRLD/RLDX-1-FT-GR1 \
  --embodiment-tag GENERAL_EMBODIMENT --use-sim-policy-wrapper --compile cudagraph
```

Reproduce the optimization sweep: `optimize_probe.py` (breakdown + ref action), `optimize_compare.py --opt {cudagraph,compileB,triton,msatRO,steps}` (latency + output drift), `step_sweep.py` (denoising-step sweep).

## Reproduce

```bash
# capture a real observation (needs the GR-1 sim venv), one-time:
#   start server:  .venv/bin/python run_scripts/bench/save_gr1_observation.py --port 20111
#   then run the normal rollout client against port 20111 for a few steps
RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python run_scripts/bench/bench_gr1_inference.py
RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python run_scripts/bench/profile_gr1_stages.py
RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python run_scripts/bench/infer_gr1.py   # HF-style demo
.venv/bin/python run_scripts/bench/gr1_inference_report.py gr1_inference_thor_report.pdf          # PDF
```
