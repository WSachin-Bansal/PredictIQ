"""Matplotlib figures shared by the dashboard pages (no Streamlit imports, so testable)."""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CH_NAMES = ["Accel 1 · ch1", "Accel 1 · ch2", "Accel 1 · ch3",
            "Accel 2 · ch1", "Accel 2 · ch2", "Accel 2 · ch3"]


def plot_vibration(vib: np.ndarray, fs: int = 10000, color: str = "#2f7fd6", zoom: float = 0.1):
    """vib: (6, T). Six stacked traces; only the first `zoom` seconds so the wave is visible."""
    n = int(zoom * fs)
    t = np.arange(n) / fs
    fig, axes = plt.subplots(6, 1, figsize=(6, 5), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, vib[i, :n], color=color, lw=0.8)
        ax.set_ylabel(CH_NAMES[i], rotation=0, ha="right", va="center", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Vibration, first {zoom:g} s of the window", fontsize=9)
    fig.tight_layout()
    return fig


def plot_spectrum(vib: np.ndarray, fs: int = 10000, color: str = "#2f7fd6", max_hz: int = 1500):
    """Average magnitude spectrum over the 6 channels: peaks at multiples of the rotation speed."""
    x = vib - vib.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.rfft(x, axis=1)).mean(axis=0) / x.shape[1]
    f = np.fft.rfftfreq(x.shape[1], 1 / fs)
    keep = f <= max_hz
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(f[keep], mag[keep], color=color, lw=0.9)
    ax.set(xlabel="frequency (Hz)", ylabel="strength", title="Frequency spectrum (vibration)")
    fig.tight_layout()
    return fig


def plot_mel(spec: np.ndarray, title: str = "Mel spectrogram (what the audio model sees)"):
    """spec: (n_mels, frames) already log-scaled and standardised."""
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
    ax.set(xlabel="time frame", ylabel="Mel band", title=title)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Results page figures
MODE_COLORS = {"sensor": "#3b6ea5", "audio": "#b8862f", "fusion": "#1a9988"}
MODE_NAMES = {"sensor": "Vibration only", "audio": "Sound only", "fusion": "Fusion"}


def plot_accuracy_bars(tab, split: str, metric: str = "accuracy"):
    """Mean +- std over seeds for the three models (full input). tab = utils.summary_table()."""
    rows = tab[(tab["split"] == split) & (tab["input"] == "full")].set_index("mode")
    modes = [m for m in ("sensor", "audio", "fusion") if m in rows.index]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    vals = [rows.loc[m, f"{metric}_mean"] for m in modes]
    errs = [rows.loc[m, f"{metric}_std"] for m in modes]
    ax.bar([MODE_NAMES[m] for m in modes], vals, yerr=errs, capsize=5,
           color=[MODE_COLORS[m] for m in modes])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.1%}", ha="center", fontsize=9)
    ax.set(ylim=(0, 1.08), ylabel=metric.replace("_", " "))
    fig.tight_layout()
    return fig


def plot_missing(tab, split: str, metric: str = "accuracy"):
    """Fusion with each input hidden in turn, against the single-input models."""
    t = tab[tab["split"] == split]
    fus = t[t["mode"] == "fusion"].set_index("input")
    names = [("full", "both inputs"), ("vibration hidden", "vibration hidden"),
             ("sound hidden", "sound hidden")]
    names = [(k, n) for k, n in names if k in fus.index]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar([n for _, n in names], [fus.loc[k, f"{metric}_mean"] for k, _ in names],
           yerr=[fus.loc[k, f"{metric}_std"] for k, _ in names], capsize=5,
           color=MODE_COLORS["fusion"])
    for mode in ("sensor", "audio"):
        r = t[(t["mode"] == mode) & (t["input"] == "full")]
        if len(r):
            ax.axhline(r.iloc[0][f"{metric}_mean"], ls="--", color=MODE_COLORS[mode],
                       label=f"{MODE_NAMES[mode]} model")
    ax.set(ylim=(0, 1.05), ylabel=metric.replace("_", " "), title="Fusion model with one input removed")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def plot_confusion(cm, classes, normalize: bool = True):
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set(xlabel="predicted", ylabel="true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = shown[i, j]
            if v >= (0.005 if normalize else 1):
                ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


def per_class_recall(cm):
    cm = np.asarray(cm, dtype=float)
    return np.diag(cm) / np.maximum(cm.sum(1), 1)


def plot_per_class(cm, classes, colors):
    rec = per_class_recall(cm)
    order = np.argsort(rec)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh([classes[i] for i in order], rec[order], color=[colors.get(classes[i], "#888") for i in order])
    ax.set(xlim=(0, 1.05), xlabel="share of windows of this class found (recall)")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def plot_curves(runs, mode: str, key: str = "val_f1"):
    """Validation score per epoch, one line per seed."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for r in runs:
        h = r["history"]
        ax.plot([e["epoch"] for e in h], [e[key] for e in h], color=MODE_COLORS[mode], alpha=0.6,
                label=f"seed {r['args']['seed']}")
    ax.set(xlabel="epoch", ylabel=key.replace("_", " "), ylim=(0, 1.02),
           title=f"{MODE_NAMES[mode]}: validation score while training")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig
