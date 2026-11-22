"""Helpers for the RUL page (no torch / streamlit, so they can be tested)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predictiq.data.sensor import load_cmapss  # noqa: E402

RESULTS = ROOT / "results"
CMAPSS = ROOT / "data" / "raw" / "cmapss"

# Illustrative traffic-light thresholds (cycles). Not a validated maintenance policy.
GREEN_ABOVE, YELLOW_ABOVE = 75, 30


def status(rul: float) -> tuple[str, str]:
    if rul > GREEN_ABOVE:
        return "HEALTHY", "#2e9e5b"
    if rul > YELLOW_ABOVE:
        return "PLAN MAINTENANCE", "#e0a800"
    return "URGENT", "#d64545"


def load_rul_runs(subset: str = "FD001") -> list[dict]:
    return [json.loads(p.read_text()) | {"file": p.name} for p in sorted(RESULTS.glob(f"rul_{subset}_seed*.json"))]


def load_test(subset: str = "FD001", data_dir=None):
    """(test dataframe, final true RUL per engine in file order, engine ids in file order)."""
    d = Path(data_dir or CMAPSS)
    df = load_cmapss(str(d / f"test_{subset}.txt"))
    final = np.loadtxt(d / f"RUL_{subset}.txt")
    return df, final, sorted(df["unit"].unique())


def cycle_windows(feats: np.ndarray, window: int) -> np.ndarray:
    """feats (T, F), already normalised -> (T, window, F): the window ENDING at every cycle
    (front-padded with the first row, the same rule the test evaluation uses)."""
    padded = np.concatenate([np.repeat(feats[:1], window - 1, axis=0), feats])
    return np.stack([padded[t:t + window] for t in range(len(feats))])


def true_rul_curve(n_cycles: int, final_rul: float, cap: float = 125.0) -> np.ndarray:
    """True RUL at every observed cycle of a cut-off test engine (capped like the training labels)."""
    return np.minimum(final_rul + (n_cycles - 1 - np.arange(n_cycles)), cap)


def plot_sensors(g, cols, cycle: int, mean, std, all_cols):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for c in cols:
        k = all_cols.index(c)
        ax.plot(g["cycle"], (g[c].to_numpy() - mean[k]) / std[k], lw=1, label=c)
    ax.axvline(cycle, color="k", ls="--", lw=1)
    ax.set(xlabel="flight cycle", ylabel="sensor value (standardised)", title="Sensor trends")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout()
    return fig


def plot_rul_curve(cycles, pred, true, cycle: int):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.axhspan(GREEN_ABOVE, 130, color="#2e9e5b", alpha=0.10)
    ax.axhspan(YELLOW_ABOVE, GREEN_ABOVE, color="#e0a800", alpha=0.12)
    ax.axhspan(0, YELLOW_ABOVE, color="#d64545", alpha=0.10)
    ax.plot(cycles, true, color="k", lw=2, label="true RUL (capped at 125)")
    ax.plot(cycles, pred, color="#3b6ea5", lw=1.6, label="predicted RUL")
    ax.plot([cycle], [pred[cycle - 1]], "o", color="#3b6ea5", ms=8)
    ax.set(xlabel="flight cycle", ylabel="remaining useful life (cycles)", ylim=(0, 130),
           title="Predicted vs true remaining life")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def plot_pred_vs_true(true, pred):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 130], [0, 130], "k--", lw=1)
    ax.scatter(true, pred, s=14, color="#3b6ea5", alpha=0.8)
    ax.set(xlabel="true RUL at cut-off", ylabel="predicted RUL", title="All 100 test engines")
    fig.tight_layout()
    return fig
