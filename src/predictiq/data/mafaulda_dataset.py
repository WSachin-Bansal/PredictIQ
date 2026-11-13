"""Torch dataset over the preprocessed MAFAULDA windows (see data/mafaulda.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_index(processed_dir) -> tuple[pd.DataFrame, list[str]]:
    d = Path(processed_dir)
    index = pd.read_csv(d / "index.csv", dtype={"severity_label": str})
    classes = json.loads((d / "meta.json").read_text())["classes"]
    return index, classes


def fit_sensor_stats(processed_dir, index: pd.DataFrame, ids, n_files: int = 200,
                     seed: int = 0):
    """Per-channel mean/std of vibration from TRAIN files only (no val/test leakage)."""
    d = Path(processed_dir)
    keys = dict(zip(index["id"], index["key"]))
    ids = list(ids)
    rng = np.random.default_rng(seed)
    if len(ids) > n_files:
        ids = list(rng.choice(ids, n_files, replace=False))
    chunks = []
    for i in ids:
        v = np.load(d / "vib" / f"{keys[int(i)]}.npy", mmap_mode="r")
        chunks.append(np.asarray(v[len(v) // 2], dtype=np.float32))  # (C, T)
    x = np.stack(chunks)  # (K, C, T)
    mean = x.mean(axis=(0, 2))
    std = np.maximum(x.std(axis=(0, 2)), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


class MafauldaWindows(Dataset):
    """One item per window.

    Returns a dict: sensor (C, T) normalised vibration, audio (n_samples,) raw waveform
    (the log-Mel transform is applied on-device in the training loop), fault, severity
    (-1 = none), file_id. Only the requested modalities are loaded.
    """

    def __init__(self, processed_dir, index: pd.DataFrame, ids,
                 modalities=("sensor", "audio"), stats=None):
        self.dir = Path(processed_dir)
        self.modalities = tuple(modalities)
        self.stats = stats
        sub = index.set_index("id").loc[list(ids)]
        self.fault = dict(zip(sub.index, sub["label"]))
        self.sev = dict(zip(sub.index, sub["severity_bin"]))
        self.key = dict(zip(sub.index, sub["key"]))
        self.items = [(int(i), w) for i, n in zip(sub.index, sub["n_windows"])
                      for w in range(int(n))]
        self._cache: dict = {}

    def __len__(self):
        return len(self.items)

    def _open(self, fid: int) -> dict:
        if fid not in self._cache:
            if len(self._cache) > 128:
                self._cache.clear()
            arrs = {}
            if "sensor" in self.modalities:
                arrs["vib"] = np.load(self.dir / "vib" / f"{self.key[fid]}.npy", mmap_mode="r")
            if "audio" in self.modalities:
                arrs["aud"] = np.load(self.dir / "aud" / f"{self.key[fid]}.npy", mmap_mode="r")
            self._cache[fid] = arrs
        return self._cache[fid]

    def __getitem__(self, i):
        fid, w = self.items[i]
        arrs = self._open(fid)
        item = {"file_id": fid, "fault": int(self.fault[fid]),
                "severity": int(self.sev[fid])}
        if "sensor" in self.modalities:
            v = np.asarray(arrs["vib"][w], dtype=np.float32)
            if self.stats is not None:
                mean, std = self.stats
                v = (v - mean[:, None]) / std[:, None]
            item["sensor"] = torch.from_numpy(v)
        if "audio" in self.modalities:
            item["audio"] = torch.from_numpy(np.asarray(arrs["aud"][w], dtype=np.float32))
        return item
