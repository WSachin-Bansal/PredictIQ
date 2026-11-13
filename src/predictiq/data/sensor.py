"""Sensor data utilities: C-MAPSS loading, RUL labels, and leakage-safe windowing."""
from __future__ import annotations

import numpy as np
import pandas as pd

CMAPSS_COLS = (
    ["unit", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)


def load_cmapss(path: str) -> pd.DataFrame:
    """Load a C-MAPSS train_FD00X.txt file (whitespace separated, no header)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLS)
    return df


def add_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """Add piecewise-linear RUL: min(cycles_to_failure, max_rul)."""
    df = df.copy()
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = np.minimum(last - df["cycle"], max_rul)
    return df


def drop_constant(df: pd.DataFrame, cols: list[str], tol: float = 1e-8) -> list[str]:
    """Return feature columns whose std is above tol (constant sensors carry no signal)."""
    return [c for c in cols if df[c].std() > tol]


def split_by_unit(units: np.ndarray, val_frac: float = 0.2, seed: int = 42):
    """Split by engine/run id, NOT by window, to avoid trajectory leakage."""
    rng = np.random.default_rng(seed)
    units = np.array(sorted(set(units)))
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_frac))
    return units[n_val:], units[:n_val]


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int = 50,
    stride: int = 5,
):
    """Slice each unit's trajectory into windows.

    Returns X (N, window, F), y_rul (N,), unit_ids (N,). The label is the RUL at the
    last step of the window. Normalise BEFORE calling (fit scaler on train units only).
    """
    xs, ys, us = [], [], []
    for unit, g in df.groupby("unit"):
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        rul = g["rul"].to_numpy(dtype=np.float32)
        for end in range(window, len(g) + 1, stride):
            xs.append(feats[end - window : end])
            ys.append(rul[end - 1])
            us.append(unit)
    return np.stack(xs), np.array(ys, dtype=np.float32), np.array(us)


def fit_scaler(df: pd.DataFrame, cols: list[str]):
    """Return (mean, std) computed on training data only."""
    mean = df[cols].mean().to_numpy(dtype=np.float32)
    std = df[cols].std().replace(0, 1).to_numpy(dtype=np.float32)
    return mean, std


def apply_scaler(df: pd.DataFrame, cols: list[str], mean, std) -> pd.DataFrame:
    df = df.copy()
    df[cols] = (df[cols].to_numpy(dtype=np.float32) - mean) / std
    return df
