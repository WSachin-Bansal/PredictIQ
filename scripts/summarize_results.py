"""Aggregate results/*.json into a mean +- std table (copy it into docs/EXPERIMENT_LOG.md).

    python scripts/summarize_results.py [--results results]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = ["accuracy", "f1", "file_accuracy"]


def collect(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob("*.json")):
        r = json.loads(p.read_text())
        a = r["args"]
        base = {"mode": a["mode"], "task": a.get("task", "fault"), "split": a["split"],
                "seed": a["seed"]}
        for variant, key in (("full", "test"), ("sensor hidden", "test_missing_sensor"),
                             ("audio hidden", "test_missing_audio")):
            if key in r:
                rows.append({**base, "input": variant, **{m: r[key][m] for m in METRICS}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    df = collect(Path(ap.parse_args().results))
    if df.empty:
        print("no results found")
        return
    g = df.groupby(["task", "split", "mode", "input"])
    out = g[METRICS].agg(["mean", "std"]).round(4)
    out.insert(0, ("n_seeds", ""), g.size())
    print(out.to_string())
    try:  # needs the optional `tabulate` package
        print("\nMarkdown:\n")
        print(out.to_markdown())
    except ImportError:
        print("\n(pip install tabulate for a markdown version)")


if __name__ == "__main__":
    main()
