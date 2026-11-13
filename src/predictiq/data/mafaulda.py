"""MAFAULDA utilities (pure numpy / pandas / scipy, no torch).

Steps: build an index of recordings -> split BY FILE -> preprocess each recording into
fixed-length, time-aligned windows (vibration + microphone) cached as .npy files.

CLI:
    python -m predictiq.data.mafaulda --raw data/raw/mafaulda --out data/processed/mafaulda

Assumed raw layout (verify against your download; the index builder prints what it found):
    <root>/<fault>[/<severity>]/<recording>.csv
    <root>/{overhang,underhang}/<bearing_fault>/<severity>/<recording>.csv
Each CSV has no header and 8 columns:
    0 tachometer | 1-3 underhang accel (axial, radial, tangential)
    4-6 overhang accel | 7 microphone
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate, resample_poly

COLS_ACCEL = list(range(1, 7))
COL_MIC = 7
UNSPECIFIED = "unspecified"
_NUM = re.compile(r"^(\d+(?:\.\d+)?)")


# --------------------------------------------------------------------------- index
def _descend(root: Path) -> Path:
    """Step into wrapper folders (e.g. root/mafaulda/...) until real content appears."""
    while True:
        if any(root.glob("*.csv")):
            return root
        subs = [p for p in root.iterdir() if p.is_dir()]
        if len(subs) != 1:
            return root
        root = subs[0]


def parse_path(dir_parts: tuple[str, ...], stem: str) -> dict:
    """Infer fault class, severity and speed from the folder names / file name."""
    parts = [p for p in dir_parts]
    if not parts:
        fault, rest = "unknown", []
    elif parts[0].lower() in ("overhang", "underhang") and len(parts) > 1:
        fault, rest = f"{parts[0].lower()}_{parts[1].lower()}", parts[2:]
    else:
        fault, rest = parts[0].lower(), parts[1:]
    # some downloads repeat the class folder (imbalance/imbalance/6g): ignore the repeats
    names = {fault, *(p.lower() for p in parts[:1])}
    if parts and parts[0].lower() in ("overhang", "underhang") and len(parts) > 1:
        names.add(parts[1].lower())
    rest = [p for p in rest if p.lower() not in names]
    numeric = [p for p in rest if _NUM.match(p)]
    sev_label = numeric[0] if numeric else (rest[0] if rest else UNSPECIFIED)
    m = _NUM.match(sev_label)
    severity = float(m.group(1)) if m else float("nan")
    try:
        speed = float(stem)
    except ValueError:
        speed = float("nan")
    return {"fault": fault, "severity_label": sev_label, "severity": severity,
            "speed_hz": speed}


def build_index(raw_root) -> tuple[pd.DataFrame, list[str]]:
    """One row per CSV: id, path, fault, severity_label, severity, severity_bin, label.

    severity_bin is a within-fault rank bucket (0/1/2 = low/mid/high), -1 if the fault
    has no graded severity (e.g. normal).
    """
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"{raw_root} does not exist")
    root = _descend(raw_root)
    rows = []
    for p in sorted(root.rglob("*.csv")):
        rel = p.relative_to(root)
        # `key` names the cache files; derived from the path so adding folders later
        # (e.g. while a download is still running) never invalidates existing caches
        rows.append({"path": str(p), "key": "__".join(rel.with_suffix("").parts),
                     **parse_path(rel.parts[:-1], p.stem)})
    if not rows:
        raise FileNotFoundError(f"no .csv files under {raw_root}")
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    df["severity_bin"] = -1
    for _, g in df.groupby("fault"):
        vals = sorted(g["severity"].dropna().unique())
        if len(vals) >= 2:
            mapping = {v: i * 3 // len(vals) for i, v in enumerate(vals)}
            idx = g.index[g["severity"].notna()]
            df.loc[idx, "severity_bin"] = g.loc[idx, "severity"].map(mapping)
    df["severity_bin"] = df["severity_bin"].astype(int)
    classes = sorted(df["fault"].unique())
    df["label"] = df["fault"].map({c: i for i, c in enumerate(classes)}).astype(int)
    return df, classes


def relabel_for_task(index: pd.DataFrame, task: str = "fault") -> tuple[pd.DataFrame, list[str]]:
    """Choose the classification target.

    task="fault":    one class per fault type (normal, imbalance, ...).
    task="severity": one class per (fault, severity level), e.g. normal, imbalance_6g,
                     imbalance_10g ... Much harder when only a few fault types exist.
    Returns (index with a rewritten `label` column, class names in label order).
    """
    if task not in ("fault", "severity"):
        raise ValueError("task must be 'fault' or 'severity'")
    index = index.copy()
    if task == "fault":
        classes = sorted(index["fault"].unique())
        index["label"] = index["fault"].map({c: i for i, c in enumerate(classes)}).astype(int)
        return index, classes
    graded = index["severity_label"] != UNSPECIFIED
    name = index["fault"].where(~graded, index["fault"] + "_" + index["severity_label"])
    order = (index.assign(name=name)[["name", "fault", "severity"]].drop_duplicates()
             .sort_values(["fault", "severity", "name"], na_position="first"))
    classes = order["name"].tolist()
    index["label"] = name.map({c: i for i, c in enumerate(classes)}).astype(int)
    return index, classes


# --------------------------------------------------------------------------- splits
def _split_list(items, rng, val_frac, test_frac):
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    if n < 3:
        return items, [], []
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    return items[n_test + n_val:], items[n_test:n_test + n_val], items[:n_test]


def split_files(index: pd.DataFrame, mode: str = "file", seed: int = 42,
                val_frac: float = 0.2, test_frac: float = 0.2) -> dict:
    """Split recordings (never windows) into train/val/test, stratified by fault class.

    mode="file":     random files per class. Standard, but neighbouring recordings of the
                     same fault/severity/speed are near-duplicates, so scores are optimistic.
    mode="severity": whole severity levels are held out per class (needs >=3 levels in the
                     class, else falls back to "file" for that class). Harder and more
                     honest: tests generalisation to unseen fault severities.
    The result depends only on (index, mode, seed) so every model uses identical splits.
    """
    if mode not in ("file", "severity"):
        raise ValueError("mode must be 'file' or 'severity'")
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    key = "label" if mode == "file" else "fault"   # file mode: stratify by the target
    for _, g in index.groupby(key, sort=True):
        if mode == "severity" and g["severity_label"].nunique() >= 3:
            groups = sorted(g["severity_label"].unique())
            parts = _split_list(groups, rng, val_frac, test_frac)
            for name, gs in zip(("train", "val", "test"), parts):
                out[name] += g.loc[g["severity_label"].isin(gs), "id"].tolist()
        else:
            parts = _split_list(g["id"].tolist(), rng, val_frac, test_frac)
            for name, ids in zip(("train", "val", "test"), parts):
                out[name] += list(ids)
    return {k: np.array(sorted(v), dtype=int) for k, v in out.items()}


# --------------------------------------------------------------------------- preprocessing
def _starts(n: int, win: int, stride: int) -> list[int]:
    return list(range(0, n - win + 1, stride))


def prepare_file(path, fs: int = 50_000, vib_rate: int = 10_000, audio_rate: int = 16_000,
                 win_s: float = 1.0, stride_s: float = 0.5):
    """Read one recording -> (vib (N, 6, win_v) float16, audio (N, win_a) float16).

    Vibration is low-pass filtered and decimated fs -> vib_rate; the microphone is
    resampled fs -> audio_rate. Window k covers the SAME time span in both modalities.
    """
    with open(path) as f:                      # tolerate an optional header row
        first = f.readline().strip().split(",")
    try:
        [float(v) for v in first]
        skip = 0
    except ValueError:
        skip = 1
    x = pd.read_csv(path, header=None, skiprows=skip, dtype=np.float32).to_numpy()
    if x.ndim != 2 or x.shape[1] < 8:
        raise ValueError(f"expected 8 columns, got shape {x.shape}")
    if fs % vib_rate:
        raise ValueError("fs must be a multiple of vib_rate")
    vib = decimate(x[:, COLS_ACCEL].astype(np.float64), fs // vib_rate, axis=0,
                   zero_phase=True)
    g = math.gcd(audio_rate, fs)
    aud = resample_poly(x[:, COL_MIC].astype(np.float64), audio_rate // g, fs // g)

    wv, sv = int(round(win_s * vib_rate)), int(round(stride_s * vib_rate))
    wa, sa = int(round(win_s * audio_rate)), int(round(stride_s * audio_rate))
    sv_starts, sa_starts = _starts(len(vib), wv, sv), _starts(len(aud), wa, sa)
    n = min(len(sv_starts), len(sa_starts))
    if n == 0:
        raise ValueError("recording shorter than one window")
    vib_w = np.stack([vib[s:s + wv].T for s in sv_starts[:n]]).astype(np.float16)
    aud_w = np.stack([aud[s:s + wa] for s in sa_starts[:n]]).astype(np.float16)
    return vib_w, aud_w


def prepare_dataset(raw_root, out_dir, force: bool = False, **kw) -> pd.DataFrame:
    """Preprocess every recording to out_dir/{vib,aud}/<id>.npy and write index.csv."""
    index, _ = build_index(raw_root)
    out = Path(out_dir)
    (out / "vib").mkdir(parents=True, exist_ok=True)
    (out / "aud").mkdir(parents=True, exist_ok=True)
    try:
        from tqdm import tqdm
        rows = tqdm(list(index.itertuples()), desc="preprocessing")
    except ImportError:
        rows = index.itertuples()
    n_windows = []
    for r in rows:
        vp, ap = out / "vib" / f"{r.key}.npy", out / "aud" / f"{r.key}.npy"
        try:
            if vp.exists() and ap.exists() and not force:
                n = len(np.load(vp, mmap_mode="r"))
            else:
                vib, aud = prepare_file(r.path, **kw)
                np.save(vp, vib)
                np.save(ap, aud)
                n = len(vib)
        except Exception as e:  # keep going; report at the end
            print(f"skipping {r.path}: {e}")
            n = 0
        n_windows.append(n)
    index["n_windows"] = n_windows
    index = index[index["n_windows"] > 0].reset_index(drop=True)
    classes = sorted(index["fault"].unique())
    index["label"] = index["fault"].map({c: i for i, c in enumerate(classes)}).astype(int)
    index.to_csv(out / "index.csv", index=False)
    meta = {"classes": classes, **{k: kw[k] for k in kw}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{len(index)} recordings, {int(index['n_windows'].sum())} windows")
    print(index.groupby("fault").agg(files=("id", "count"),
                                     windows=("n_windows", "sum"),
                                     severities=("severity_label", "nunique")))
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fs", type=int, default=50_000)
    ap.add_argument("--vib-rate", type=int, default=10_000)
    ap.add_argument("--audio-rate", type=int, default=16_000)
    ap.add_argument("--win-seconds", type=float, default=1.0)
    ap.add_argument("--stride-seconds", type=float, default=0.5)
    ap.add_argument("--force", action="store_true", help="re-process existing files")
    a = ap.parse_args()
    prepare_dataset(a.raw, a.out, force=a.force, fs=a.fs, vib_rate=a.vib_rate,
                    audio_rate=a.audio_rate, win_s=a.win_seconds, stride_s=a.stride_seconds)


if __name__ == "__main__":
    main()
