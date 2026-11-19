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
