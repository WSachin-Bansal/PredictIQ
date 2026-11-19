"""Shared helpers for every dashboard page: paths, result loading, class colours."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
PROCESSED = ROOT / "data" / "processed" / "mafaulda"

MODE_LABEL = {"sensor": "Vibration only", "audio": "Sound only", "fusion": "Fusion (both)"}
SPLIT_LABEL = {"file": "Standard split", "severity": "Unseen-severity split"}

# One fixed colour per fault class, used on EVERY page so the audience learns them once.
CLASS_COLORS = {
    "normal": "#2e9e5b",
    "imbalance": "#e07b00",
    "horizontal-misalignment": "#7a5cc7",
    "vertical-misalignment": "#b04fb0",
    "overhang_ball_fault": "#d64545",
    "overhang_cage_fault": "#e0707a",
    "overhang_outer_race": "#9c2f2f",
    "underhang_ball_fault": "#2f7fd6",
    "underhang_cage_fault": "#5fa8e8",
    "underhang_outer_race": "#1f4f8f",
}


def load_runs(task: str = "fault") -> list[dict]:
    """Every results/<mode>_<task>_<split>_seed<N>.json as a dict (skips RUL and trial files)."""
    runs = []
    for p in sorted(RESULTS.glob(f"*_{task}_*_seed*.json")):
        r = json.loads(p.read_text())
        r["file"] = p.name
        runs.append(r)
    return runs


def summary_table(task: str = "fault") -> pd.DataFrame:
    """One row per (split, model, input) with mean/std over seeds. Metrics are fractions."""
    rows = []
    for r in load_runs(task):
        a = r["args"]
        for variant, key in (("full", "test"), ("vibration hidden", "test_missing_sensor"),
                             ("sound hidden", "test_missing_audio")):
            if key in r:
                rows.append({"split": a["split"], "mode": a["mode"], "input": variant,
                             "seed": a["seed"], "accuracy": r[key]["accuracy"],
                             "f1": r[key]["f1"], "file_accuracy": r[key]["file_accuracy"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby(["split", "mode", "input"])[["accuracy", "f1", "file_accuracy"]]
    out = g.agg(["mean", "std"]).fillna(0.0)
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = df.groupby(["split", "mode", "input"]).size()
    return out.reset_index()


def load_rul_runs() -> list[dict]:
    return [json.loads(p.read_text()) | {"file": p.name} for p in sorted(RESULTS.glob("rul_*_seed*.json"))]
