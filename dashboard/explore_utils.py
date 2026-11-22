"""Tables and figures for the Data Explorer page (no streamlit/torch, so testable)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predictiq.data.mafaulda import relabel_for_task, split_files  # noqa: E402

SPLIT_COLORS = {"train": "#3b6ea5", "val": "#e0a800", "test": "#d64545"}


def class_counts(index: pd.DataFrame, unit: str = "recordings") -> pd.Series:
    s = index.groupby("fault")["n_windows"].sum() if unit == "windows" else index.groupby("fault").size()
    return s.sort_values()


def plot_class_counts(counts: pd.Series, colors: dict, unit: str):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(counts.index, counts.values, color=[colors.get(c, "#888") for c in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=8)
    ax.set(xlabel=f"number of {unit}")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def assign_splits(index: pd.DataFrame, split: str, seed: int = 42) -> pd.DataFrame:
    """Index with a `set` column (train/val/test), built with the SAME call the training uses."""
    idx, _ = relabel_for_task(index, "fault")
    ids = split_files(idx, split, seed)
    where = {int(i): name for name, v in ids.items() for i in v}
    out = index.copy()
    out["set"] = out["id"].map(where)
    return out


def severity_table(indexed: pd.DataFrame) -> pd.DataFrame:
    """One row per (fault, severity level): recordings and the set they were assigned to."""
    g = indexed.groupby(["fault", "severity_label", "set"]).size().unstack("set", fill_value=0)
    for c in ("train", "val", "test"):
        if c not in g:
            g[c] = 0
    g = g[["train", "val", "test"]].reset_index().rename(columns={"severity_label": "severity"})
    g["recordings"] = g[["train", "val", "test"]].sum(axis=1)
    return g


def plot_split_stack(indexed: pd.DataFrame):
    t = indexed.groupby(["fault", "set"]).size().unstack("set", fill_value=0)
    t = t.reindex(columns=["train", "val", "test"], fill_value=0)
    t = t.loc[t.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    left = np.zeros(len(t))
    for c in t.columns:
        ax.barh(t.index, t[c], left=left, color=SPLIT_COLORS[c], label=c)
        left += t[c].to_numpy()
    ax.set(xlabel="recordings")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def plot_speed(index: pd.DataFrame, colors: dict):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for f, g in index.groupby("fault"):
        ax.hist(g["speed_hz"].dropna(), bins=25, alpha=0.45, color=colors.get(f, "#888"), label=f)
    ax.set(xlabel="shaft speed (Hz)", ylabel="recordings")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return fig
