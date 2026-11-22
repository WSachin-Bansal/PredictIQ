"""Helpers for the Vision page (no torch / streamlit, so they can be tested)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predictiq.data.mvtec import find_category  # noqa: E402

RESULTS = ROOT / "results"
MVTEC = ROOT / "data" / "raw" / "mvtec"


def load_vision_runs() -> list[dict]:
    return [json.loads(p.read_text()) | {"file": p.name} for p in sorted(RESULTS.glob("vision_*_seed*.json"))]


def resolve_image(stored_path: str, category: str, root=None) -> Path:
    """The result files store the training PC's absolute path. Rebuild it from
    <category>/<rest> under the local MVTec folder, so it also works on teammates' machines."""
    parts = stored_path.replace("\\", "/").split("/")
    i = max(k for k, p in enumerate(parts) if p == category)
    return find_category(root or MVTEC, category).joinpath(*parts[i + 1:])


def mask_path(img_path: Path):
    """ground_truth mask of a defective test image, or None."""
    p = Path(img_path)
    if len(p.parents) > 2 and p.parents[1].name == "test" and p.parents[0].name != "good":
        m = p.parents[2] / "ground_truth" / p.parents[0].name / f"{p.stem}_mask.png"
        return m if m.exists() else None
    return None


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap (H,W) onto an RGB uint8 image (H,W,3)."""
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return ((1 - alpha) * img / 255.0 + alpha * heat).clip(0, 1)


def defect_detection_table(runs: list[dict]) -> pd.DataFrame:
    """Mean over seeds of the share of test images flagged 'defect', per defect type."""
    rows = [{"defect_type": k, "flagged": v["predicted_defect"], "seed": r["args"]["seed"]}
            for r in runs for k, v in r["per_defect_type"].items()]
    df = pd.DataFrame(rows)
    return df.groupby("defect_type")["flagged"].agg(["mean", "std"]).fillna(0).reset_index()


def plot_defect_detection(df: pd.DataFrame):
    df = df.sort_values("mean")
    colors = ["#2e9e5b" if d == "good" else "#d64545" for d in df["defect_type"]]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(df["defect_type"], df["mean"], xerr=df["std"], color=colors, capsize=3)
    ax.set(xlim=(0, 1.05), xlabel="share of test images flagged as 'defect'")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig
