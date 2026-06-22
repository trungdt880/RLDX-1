#!/usr/bin/env bash
# Config C: the ORIGINAL pre-conv-fix version (RLDX_PATCHEMBED_FP32=0 -> bf16
# Conv3d fallback, ~6 s/obs). Waits for the A/B run (eager vs cudagraph) to
# finish first (single GPU, serial), then runs the same 24 tasks x N eps, then
# prints a 3-way success-rate comparison: original vs eager-fix vs cudagraph.
#
# This confirms the fp32-Conv3d fix itself does not change task success — only
# speed — and that cudagraph likewise preserves behavior.
#
# Usage:  N_EPISODES=5 run_scripts/bench/run_gr1_eval_original.sh [MODEL_PATH]
set -uo pipefail

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$BASE_DIR"

MODEL="${1:-RLWRLD/RLDX-1-FT-GR1}"
N_EPISODES="${N_EPISODES:-5}"
TAG="$(echo "$MODEL" | tr '/' '_')"
ROOT="$BASE_DIR/output_final/gr1_tabletop"
EAGER_ROOT="$ROOT/${TAG}__eager"
CUDA_ROOT="$ROOT/${TAG}__cudagraph"
ORIG_ROOT="$ROOT/${TAG}__original_noconvfix"
EVAL="$BASE_DIR/run_scripts/eval/gr1_tabletop/eval_gr1_thor.sh"
AGG="$BASE_DIR/run_scripts/eval/gr1_tabletop/aggregate_gr1.py"
AB_LOG="$BASE_DIR/run_scripts/bench/gr1_eval_AB.log"
PY="$BASE_DIR/.venv/bin/python"

echo "================================================================"
echo " Config C: ORIGINAL (no conv fix, RLDX_PATCHEMBED_FP32=0, ~6 s/obs)"
echo " waiting for the A/B run to finish before starting (serial GPU)..."
echo " $(date)"
echo "================================================================"

# --- wait for the A/B run to finish (finished marker), max ~8 h ---
for _ in $(seq 1 5760); do
  grep -q "^ finished:" "$AB_LOG" 2>/dev/null && break
  sleep 5
done
# belt-and-suspenders: ensure no policy server is still holding the GPU
while pgrep -f "run_rldx_server.py" >/dev/null 2>&1; do sleep 5; done
sleep 15
echo "[i] A/B finished; starting original (no-conv-fix) run at $(date)"

# --- run the original version (disable the fp32 conv fix) ---
RLDX_PATCHEMBED_FP32=0 N_EPISODES="$N_EPISODES" OUT_ROOT="$ORIG_ROOT" PORT=20103 COMPILE=none \
    bash "$EVAL" "$MODEL" || echo "[!] original run returned nonzero (continuing)"

# --- 3-way comparison ---
echo; echo "######## 3-WAY SIDE-BY-SIDE ########  $(date +%H:%M:%S)"
"$PY" "$AGG" "$ORIG_ROOT"  --json "$ORIG_ROOT/summary.json"  >/dev/null 2>&1 || true
"$PY" "$AGG" "$EAGER_ROOT" --json "$EAGER_ROOT/summary.json" >/dev/null 2>&1 || true
"$PY" "$AGG" "$CUDA_ROOT"  --json "$CUDA_ROOT/summary.json"  >/dev/null 2>&1 || true
"$PY" - "$ORIG_ROOT/summary.json" "$EAGER_ROOT/summary.json" "$CUDA_ROOT/summary.json" <<'PYEOF'
import json, sys
labels = ["original(6s)", "eager-fix", "cudagraph"]
def load(p):
    try: return json.load(open(p))
    except Exception: return None
S = [load(p) for p in sys.argv[1:4]]
short = lambda t: t.replace("_GR1ArmsAndWaistFourierHands_Env","").replace("SplitA","")
tasks = sorted({t for s in S if s for t in s["per_task"]})
w = max((len(short(t)) for t in tasks), default=10)
def cell(s, t):
    if not s or t not in s["per_task"]: return "   -   "
    p = s["per_task"][t]; n = p["episodes"]
    return f'{p["successes"]}/{n} {100*p["successes"]/n:4.0f}%' if n else "   -   "
print(f"\n{'task':<{w}}  {labels[0]:>11}  {labels[1]:>11}  {labels[2]:>11}")
print("-"*(w+42))
for t in tasks:
    print(f"{short(t):<{w}}  {cell(S[0],t):>11}  {cell(S[1],t):>11}  {cell(S[2],t):>11}")
print("-"*(w+42))
def agg(s, key):
    if not s or key not in s: return "   -   "
    d = s[key]; n = d["episodes"]
    return f'{d["successes"]}/{n} {100*d["successes"]/n:4.1f}%' if n else "   -   "
for label,key in [("core 6 (in-dist)","core"),("novel 18 (gen)","novel"),("ALL TASKS","overall")]:
    print(f"{label:<{w}}  {agg(S[0],key):>11}  {agg(S[1],key):>11}  {agg(S[2],key):>11}")
def ov(s):
    d=s["overall"]; return 100*d["successes"]/d["episodes"] if s and d["episodes"] else float("nan")
if all(S):
    o,e,c = ov(S[0]),ov(S[1]),ov(S[2])
    print(f"\nOverall: original {o:.1f}%  |  eager-fix {e:.1f}% (Δ{e-o:+.1f})  |  cudagraph {c:.1f}% (Δ{c-o:+.1f})")
PYEOF
echo "================================================================"
echo " 3-way eval finished: $(date)"
echo "================================================================"
