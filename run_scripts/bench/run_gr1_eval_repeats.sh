#!/usr/bin/env bash
# Measure run-to-run NOISE of the GR-1 Tabletop eval to test whether the
# original(60.8%) vs eager-fix(58.3%) gap is real or sampling noise.
#
# The eval seeds only the ENV (random/np.random via env_seed), NOT torch — so
# scenes are identical across runs but the diffusion action sampling (torch.randn)
# differs every run. Repeating the SAME config therefore reveals the noise floor.
#
# Runs (serial, single GPU):
#   eager-fix  x2 more  -> __eager_r2, __eager_r3   (conv fix ON = default)
#   original   x1 more  -> __original_noconvfix_r2  (RLDX_PATCHEMBED_FP32=0)
# Then aggregates ALL runs (incl. the originals from the first A/B) and prints
# per-config mean ± spread.
#
# Usage:  N_EPISODES=5 run_scripts/bench/run_gr1_eval_repeats.sh [MODEL_PATH]
set -uo pipefail

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$BASE_DIR"
MODEL="${1:-RLWRLD/RLDX-1-FT-GR1}"
N_EPISODES="${N_EPISODES:-5}"
TAG="$(echo "$MODEL" | tr '/' '_')"
ROOT="$BASE_DIR/output_final/gr1_tabletop"
EVAL="$BASE_DIR/run_scripts/eval/gr1_tabletop/eval_gr1_thor.sh"
AGG="$BASE_DIR/run_scripts/eval/gr1_tabletop/aggregate_gr1.py"
PY="$BASE_DIR/.venv/bin/python"

run() {  # $1=label  $2=OUT_ROOT  $3=PORT  $4=extra_env(KEY=VAL or "")
  echo; echo "######## $1 ########  $(date +%H:%M:%S)"
  env $4 N_EPISODES="$N_EPISODES" OUT_ROOT="$2" PORT="$3" COMPILE=none \
      bash "$EVAL" "$MODEL" || echo "[!] $1 returned nonzero (continuing)"
  sleep 15
}

echo "================================================================"
echo " GR-1 eval NOISE-FLOOR repeats | $N_EPISODES eps/task x 24 | started $(date)"
echo "================================================================"

run "eager-fix repeat 2"  "$ROOT/${TAG}__eager_r2"            20100 ""
run "eager-fix repeat 3"  "$ROOT/${TAG}__eager_r3"            20100 ""
run "original repeat 2"   "$ROOT/${TAG}__original_noconvfix_r2" 20100 "RLDX_PATCHEMBED_FP32=0"

echo; echo "######## NOISE-FLOOR SUMMARY ########  $(date +%H:%M:%S)"
# aggregate every run dir to JSON, then group by config and show mean ± range
declare -A GROUP
GROUP[eager]="${TAG}__eager ${TAG}__eager_r2 ${TAG}__eager_r3"
GROUP[original]="${TAG}__original_noconvfix ${TAG}__original_noconvfix_r2"
for d in ${GROUP[eager]} ${GROUP[original]}; do
  [ -d "$ROOT/$d" ] && "$PY" "$AGG" "$ROOT/$d" --json "$ROOT/$d/summary.json" >/dev/null 2>&1 || true
done
EAGER_JSONS=""; for d in ${GROUP[eager]}; do EAGER_JSONS="$EAGER_JSONS $ROOT/$d/summary.json"; done
ORIG_JSONS="";  for d in ${GROUP[original]}; do ORIG_JSONS="$ORIG_JSONS $ROOT/$d/summary.json"; done
"$PY" - "EAGER" $EAGER_JSONS ":::" "ORIGINAL" $ORIG_JSONS <<'PYEOF'
import json, sys, statistics
args = sys.argv[1:]
sep = args.index(":::")
def collect(label, paths):
    rates=[]
    for p in paths:
        try:
            d=json.load(open(p)); o=d["overall"]; n=o["episodes"]
            if n: rates.append((p.split("/")[-2], 100*o["successes"]/n, o["successes"], n))
        except Exception: pass
    print(f"\n{label}:")
    for name,r,s,n in rates: print(f"   {name:42s} {s:>3}/{n:<3} = {r:5.1f}%")
    vals=[r for _,r,_,_ in rates]
    if vals:
        m=statistics.mean(vals)
        sd=statistics.pstdev(vals) if len(vals)>1 else 0.0
        print(f"   -> mean {m:.1f}%  range [{min(vals):.1f}, {max(vals):.1f}]  spread {max(vals)-min(vals):.1f} pts  (n={len(vals)} runs)")
    return vals
eg = collect(args[0], args[1:sep])
og = collect(args[sep+1], args[sep+2:])
if eg and og:
    print(f"\nVERDICT: eager-fix mean {statistics.mean(eg):.1f}% vs original mean {statistics.mean(og):.1f}%.")
    print(f"eager-fix run-to-run spread = {max(eg)-min(eg):.1f} pts. If the original/fix gap (~2.5 pts)")
    print("is <= this spread, fp32-vs-bf16 is NOISE, not a regression.")
PYEOF
echo "================================================================"
echo " repeats finished: $(date)"
echo "================================================================"
