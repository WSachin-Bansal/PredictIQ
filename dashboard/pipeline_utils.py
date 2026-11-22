"""Figures for the Pipeline & Models page (no torch/streamlit, so testable)."""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def stitch(windows: np.ndarray, stride_fraction: float = 0.5) -> np.ndarray:
    """windows (N, ..., T) with 50% overlap -> the full continuous signal (..., T + (N-1)*T/2)."""
    step = int(windows.shape[-1] * stride_fraction)
    parts = [windows[0]] + [w[..., windows.shape[-1] - step:] for w in windows[1:]]
    return np.concatenate(parts, axis=-1)


def plot_timeline(sig: np.ndarray, rate: int, win: int, win_seconds: float, stride_seconds: float,
                  color: str, title: str, ylabel: str):
    """Whole recording with the current 1-second window shaded."""
    t = np.arange(len(sig)) / rate
    start = win * stride_seconds
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.plot(t, sig, color=color, lw=0.5)
    ax.axvspan(start, start + win_seconds, color="#e0a800", alpha=0.3, label="current window")
    ax.set(xlabel="time (s)", ylabel=ylabel, title=title, xlim=(0, t[-1]))
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def plot_before_after(raw: np.ndarray, norm: np.ndarray, rate: int, zoom: float = 0.05, color="#2f7fd6"):
    """One channel before / after standardisation (first `zoom` seconds, so the wave is visible)."""
    n = int(zoom * rate)
    t = np.arange(n) / rate
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 3.4), sharex=True)
    axes[0].plot(t, raw[:n], color=color, lw=0.8)
    axes[0].set_title("raw (after filtering + downsampling)", fontsize=8)
    axes[1].plot(t, norm[:n], color=color, lw=0.8)
    axes[1].set_title("standardised (mean 0, spread 1)", fontsize=8)
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def params_by_block(layers) -> "dict[str, int]":
    """Group a layer table by its top-level block (vibration encoder, audio encoder, fusion, heads)."""
    names = {"enc.sensor": "Vibration encoder", "enc.audio": "Sound encoder", "fusion": "Fusion layer",
             "fault_head": "Fault head", "severity_head": "Severity head"}
    out: dict[str, int] = {}
    for layer, p in zip(layers["layer"], layers["params"]):
        key = next((v for k, v in names.items() if layer == k or layer.startswith(k + ".")), layer)
        out[key] = out.get(key, 0) + int(p)
    return out


def plot_params(blocks: dict):
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    items = sorted(blocks.items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in items], [v for _, v in items], color="#3b6ea5")
    for i, (_, v) in enumerate(items):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    ax.set(xlabel="parameters (learned numbers)")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig
