#!/usr/bin/env bash
# Full GR-1 Tabletop eval, run twice on Thor and compared:
#   (A) eager     — conv fix only (the shipped baseline)
#   (B) cudagraph — Path C whole-forward CUDA graph (the new --compile cudagraph)
#
# Serial on the single GPU (separate ports + OUT_ROOTs so neither clobbers the
# other). Both configs see identical env init states (deterministic eval), so a
# success-rate difference is attributable to Path C's ~1.2e-2 numeric drift.
# Prints a side-by-side per-task + overall comparison at the end.
#
# Usage:  N_EPISODES=5 run_scripts/bench/run_gr1_eval_both.sh [MODEL_PATH]
set -uo pipefail

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$BASE_DIR"

MODEL="${1:-RLWRLD/RLDX-1-FT-GR1}"
N_EPISODES="${N_EPISODES:-5}"
TAG="$(echo "$MODEL" | tr '/' '_')"
ROOT="$BASE_DIR/output_final/gr1_tabletop"
EAGER_ROOT="$ROOT/${TAG}__eager"
CUDA_ROOT="$ROOT/${TAG}__cudagraph"
EVAL="$BASE_DIR/run_scripts/eval/gr1_tabletop/eval_gr1_thor.sh"
AGG="$BASE_DIR/run_scripts/eval/gr1_tabletop/aggregate_gr1.py"
PY="$BASE_DIR/.venv/bin/python"

echo "================================================================"
echo " GR-1 Tabletop A/B eval  | model=$MODEL | $N_EPISODES eps/task x 24 tasks"
echo " A: eager     -> $EAGER_ROOT  (port 20100)"
echo " B: cudagraph -> $CUDA_ROOT  (port 20101)"
echo " started: $(date)"
echo "================================================================"

echo; echo "######## (A) EAGER (conv fix only) ########  $(date +%H:%M:%S)"
N_EPISODES="$N_EPISODES" OUT_ROOT="$EAGER_ROOT" PORT=20100 COMPILE=none \
    bash "$EVAL" "$MODEL" || echo "[!] eager run returned nonzero (continuing)"

echo "[i] cooling down 20s so the server port is released ..."
sleep 20

echo; echo "######## (B) CUDAGRAPH (Path C) ########  $(date +%H:%M:%S)"
N_EPISODES="$N_EPISODES" OUT_ROOT="$CUDA_ROOT" PORT=20101 COMPILE=cudagraph \
    bash "$EVAL" "$MODEL" || echo "[!] cudagraph run returned nonzero (continuing)"

echo; echo "######## SIDE-BY-SIDE ########  $(date +%H:%M:%S)"
"$PY" "$AGG" "$EAGER_ROOT" --json "$EAGER_ROOT/summary.json" >/dev/null 2>&1 || true
"$PY" "$AGG" "$CUDA_ROOT"  --json "$CUDA_ROOT/summary.json"  >/dev/null 2>&1 || true
"$PY" - "$EAGER_ROOT/summary.json" "$CUDA_ROOT/summary.json" <<'PYEOF'
import json, sys
try:
    a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
except Exception as e:
    print(f"[!] could not load summaries: {e}"); sys.exit(0)

def rate(d):
    n = d["episodes"]; return (100.0 * d["successes"] / n) if n else float("nan")

short = lambda t: t.replace("_GR1ArmsAndWaistFourierHands_Env", "").replace("SplitA", "")
tasks = sorted(set(a["per_task"]) | set(b["per_task"]))
w = max((len(short(t)) for t in tasks), default=10)
print(f"\n{'task':<{w}}  {'eager':>10}  {'cudagraph':>10}  {'Δ pts':>7}")
print("-" * (w + 34))
for t in tasks:
    pa, pb = a["per_task"].get(t), b["per_task"].get(t)
    ra = 100.0*pa["successes"]/pa["episodes"] if pa and pa["episodes"] else float("nan")
    rb = 100.0*pb["successes"]/pb["episodes"] if pb and pb["episodes"] else float("nan")
    sa = f'{pa["successes"]}/{pa["episodes"]}' if pa else "-"
    sb = f'{pb["successes"]}/{pb["episodes"]}' if pb else "-"
    d = rb - ra if (ra==ra and rb==rb) else float("nan")
    print(f"{short(t):<{w}}  {sa:>5} {ra:5.1f}%  {sb:>5} {rb:5.1f}%  {d:+6.1f}")
print("-" * (w + 34))
for label, key in [("core 6 (in-dist)","core"), ("novel 18 (gen)","novel"), ("ALL TASKS","overall")]:
    da, db = a.get(key), b.get(key)
    if not da or not db: continue
    ra, rb = rate(da), rate(db)
    print(f"{label:<{w}}  {da['successes']:>3}/{da['episodes']:<3} {ra:5.1f}%  "
          f"{db['successes']:>3}/{db['episodes']:<3} {rb:5.1f}%  {rb-ra:+6.1f}")
print(f"\nOverall: eager {rate(a['overall']):.1f}%  vs  cudagraph {rate(b['overall']):.1f}%  "
      f"(Δ {rate(b['overall'])-rate(a['overall']):+.1f} pts)")
PYEOF
echo "================================================================"
echo " finished: $(date)"
echo "================================================================"
