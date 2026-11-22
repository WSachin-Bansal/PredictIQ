"""MVTec AD as a supervised good-vs-defect image task (pure numpy/pandas, no torch).

MVTec AD is normally used for anomaly detection (train on good images only). Here we build a
*supervised* classifier instead, so we pool every image of a category and re-split it ourselves:

    <root>/<category>/train/good/*.png            -> good
    <root>/<category>/test/good/*.png             -> good
    <root>/<category>/test/<defect_type>/*.png    -> defect  (ground_truth/ masks are ignored)

Split: stratified by (category, label), seeded, by IMAGE. Different defect types of one
category all appear in train and test, so this does NOT test unseen defect types.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def find_category(root: str | Path, category: str) -> Path:
    """<root>/<category>, tolerating one extra wrapper folder (e.g. a Kaggle unzip)."""
    root = Path(root)
    for cand in [root / category, *root.glob(f"*/{category}")]:
        if (cand / "test").is_dir():
            return cand
    raise FileNotFoundError(f"no MVTec category '{category}' (with a test/ folder) under {root}")


def build_index(root: str | Path, categories: list[str]) -> pd.DataFrame:
    rows = []
    for cat in categories:
        base = find_category(root, cat)
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            rel = p.relative_to(base).parts          # e.g. ('test', 'broken_large', '000.png')
            if rel[0] == "ground_truth":
                continue
            if rel[0] == "train":
                defect = "good"
            elif rel[0] == "test":
                defect = rel[1]
            else:
                continue
            rows.append({"path": str(p), "category": cat, "defect_type": defect,
                         "label": 0 if defect == "good" else 1})
    if not rows:
        raise FileNotFoundError(f"no images found for {categories} under {root}")
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    return df


def split_images(index: pd.DataFrame, seed: int = 42, val_frac: float = 0.2,
                 test_frac: float = 0.2) -> dict[str, list[int]]:
    """Stratified (category, label) split by image; deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for _, g in index.groupby(["category", "label"]):
        ids = np.array(sorted(g["id"]))
        rng.shuffle(ids)
        n_te = max(1, int(round(len(ids) * test_frac)))
        n_va = max(1, int(round(len(ids) * val_frac)))
        out["test"] += ids[:n_te].tolist()
        out["val"] += ids[n_te:n_te + n_va].tolist()
        out["train"] += ids[n_te + n_va:].tolist()
    return {k: sorted(v) for k, v in out.items()}
