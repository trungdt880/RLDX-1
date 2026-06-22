#!/usr/bin/env python3
"""Aggregate GR-1 Tabletop eval results from per-task simulation_results.csv files.

Reads every ``<OUT_ROOT>/<task>/simulation_results.csv`` written by
``rldx/eval/rollout_policy.py`` and prints a per-task and overall success-rate
summary. Safe to run any time (including mid-eval) — it just reads whatever
episodes have been recorded so far.

Usage:
    python aggregate_gr1.py <OUT_ROOT> [--json out.json]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# The first six tasks are the in-distribution core PnP set; the remaining
# eighteen are the novel-object generalization split. Kept here only to label
# the two sub-aggregates in the printout.
CORE_PREFIXES = (
    "PnPCupToDrawerClose",
    "PnPPotatoToMicrowaveClose",
    "PnPMilkToMicrowaveClose",
    "PnPBottleToCabinetClose",
    "PnPWineToCabinetClose",
    "PnPCanToDrawerClose",
)


def read_task(csv_path: Path) -> tuple[int, int]:
    """Return (successes, episodes) for one task's CSV."""
    successes = total = 0
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            total += 1
            successes += int(float(row["success"])) > 0
    return successes, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root", type=Path, help="eval output dir (one subdir per task)")
    ap.add_argument("--json", type=Path, default=None, help="optional path to dump JSON summary")
    args = ap.parse_args()

    csvs = sorted(args.out_root.glob("*/simulation_results.csv"))
    if not csvs:
        raise SystemExit(f"No simulation_results.csv under {args.out_root}")

    rows, core, novel, tot_s, tot_n = [], [0, 0], [0, 0], 0, 0
    for csv_path in csvs:
        task = csv_path.parent.name
        s, n = read_task(csv_path)
        if n == 0:
            continue
        rows.append((task, s, n))
        tot_s += s
        tot_n += n
        bucket = core if task.startswith(CORE_PREFIXES) else novel
        bucket[0] += s
        bucket[1] += n

    short = lambda t: t.replace("_GR1ArmsAndWaistFourierHands_Env", "").replace("SplitA", "")
    width = max(len(short(t)) for t, _, _ in rows)
    print(f"\n{'task':<{width}}  succ/eps    rate")
    print("-" * (width + 20))
    for task, s, n in rows:
        print(f"{short(task):<{width}}  {s:>3}/{n:<3}   {100*s/n:5.1f}%")
    print("-" * (width + 20))

    def line(label: str, s: int, n: int) -> None:
        if n:
            print(f"{label:<{width}}  {s:>3}/{n:<3}   {100*s/n:5.1f}%")

    line("core 6 (in-dist)", *core)
    line("novel 18 (gen)", *novel)
    line("ALL TASKS", tot_s, tot_n)
    print(f"\nOverall: {tot_s}/{tot_n} = {100*tot_s/tot_n:.1f}%   "
          f"(paper reports 58.7% over 24 tasks @ 50 eps)")

    if args.json:
        summary = {
            "overall": {"successes": tot_s, "episodes": tot_n,
                        "rate": tot_s / tot_n if tot_n else None},
            "core": {"successes": core[0], "episodes": core[1]},
            "novel": {"successes": novel[0], "episodes": novel[1]},
            "per_task": {t: {"successes": s, "episodes": n} for t, s, n in rows},
        }
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"\n[i] Wrote JSON summary to {args.json}")


if __name__ == "__main__":
    main()
