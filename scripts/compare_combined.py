"""Combined model vs the separate models, mean +- std over seeds, from results/*.json.

    python scripts/compare_combined.py [--results results] [--init warm]

Fault rows come from fusion_fault_<split>_seed*.json (separate) and
combined_<init>_<split>_seed*.json (combined). Defect and RUL rows compare vision_* and rul_*
with the same combined runs. Output is plain text plus a markdown table for docs/EXPERIMENT_LOG.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(results: Path, pattern: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results.glob(pattern))]


def ms(runs, getter, fmt):
    v = []
    for r in runs:
        try:
            v.append(float(getter(r)))
        except (KeyError, TypeError):
            pass
    if not v:
        return "-"
    sd = np.std(v, ddof=1) if len(v) > 1 else 0.0
    return f"{fmt.format(np.mean(v))} ± {fmt.format(sd)}" + ("" if len(v) > 1 else " (1 seed)")


def rows_for(results: Path, split: str, init: str):
    sep_f = load(results, f"fusion_fault_{split}_seed*.json")
    com = load(results, f"combined_{init}_{split}_seed*.json")
    sep_v = load(results, "vision_bottle-screw_seed*.json")
    sep_r = load(results, "rul_FD001_seed*.json")
    P, D, C = "{:.3f}", "{:.3f}", "{:.1f}"
    spec = [
        ("Fault", "accuracy", sep_f, lambda r: r["test"]["accuracy"], com, lambda r: r["test"]["fault"]["accuracy"], P),
        ("Fault", "macro-F1", sep_f, lambda r: r["test"]["f1"], com, lambda r: r["test"]["fault"]["f1"], P),
        ("Fault", "per-recording acc", sep_f, lambda r: r["test"]["file_accuracy"], com, lambda r: r["test"]["fault"]["file_accuracy"], P),
        ("Fault", "vibration hidden acc", sep_f, lambda r: r["test_missing_sensor"]["accuracy"], com, lambda r: r["test"]["fault_missing_vibration"]["accuracy"], P),
        ("Fault", "sound hidden acc", sep_f, lambda r: r["test_missing_audio"]["accuracy"], com, lambda r: r["test"]["fault_missing_sound"]["accuracy"], P),
        ("Defect", "accuracy", sep_v, lambda r: r["test"]["accuracy"], com, lambda r: r["test"]["defect"]["accuracy"], P),
        ("Defect", "macro-F1", sep_v, lambda r: r["test"]["f1"], com, lambda r: r["test"]["defect"]["f1"], P),
        ("Defect", "AUC", sep_v, lambda r: r["test"]["roc_auc"], com, lambda r: r["test"]["defect"]["roc_auc"], P),
        ("Defect", "defects caught", sep_v, lambda r: r["test"]["defect_recall"], com, lambda r: r["test"]["defect"]["defect_recall"], P),
        ("RUL", "RMSE (cycles)", sep_r, lambda r: r["test"]["rmse"], com, lambda r: r["test"]["rul"]["rmse"], C),
        ("RUL", "MAE (cycles)", sep_r, lambda r: r["test"]["mae"], com, lambda r: r["test"]["rul"]["mae"], C),
        ("RUL", "NASA score", sep_r, lambda r: r["test"]["nasa_score"], com, lambda r: r["test"]["rul"]["nasa_score"], "{:.0f}"),
    ]
    out = [(t, m, ms(sr, sg, f), ms(cr, cg, f)) for t, m, sr, sg, cr, cg, f in spec]
    return out, len(com)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--init", default="warm")
    args = ap.parse_args()
    results = Path(args.results)
    for split in ("file", "severity"):
        rows, n = rows_for(results, split, args.init)
        if n == 0:
            print(f"\n[{split} split] no combined_{args.init}_{split}_seed*.json found yet")
            continue
        print(f"\n=== MAFAULDA split: {split}  (combined runs: {n}) ===")
        w = max(len(r[1]) for r in rows)
        print(f"{'task':7s} {'metric':{w}s}  {'separate models':20s} combined")
        for t, m, a, b in rows:
            print(f"{t:7s} {m:{w}s}  {a:20s} {b}")
        print("\nMarkdown:\n\n| Task | Metric | Separate models | Combined |\n|---|---|---|---|")
        for t, m, a, b in rows:
            print(f"| {t} | {m} | {a} | {b} |")


if __name__ == "__main__":
    main()
