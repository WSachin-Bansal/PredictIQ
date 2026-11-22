"""Load a trained FaultNet checkpoint and score MAFAULDA windows for the dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predictiq.data.audio import MelExtractor  # noqa: E402
from predictiq.models.faultnet import FaultNet  # noqa: E402

SEVERITY_NAMES = ["low", "medium", "high"]


def load_bundle(tag: str):
    """checkpoints/<tag>.pt -> dict(model, classes, modalities, stats, mel, config)."""
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    m, a = cfg["mafaulda"], cfg["audio"]
    ck = torch.load(ROOT / "checkpoints" / f"{tag}.pt", map_location="cpu", weights_only=False)
    model = FaultNet(ck["modalities"], len(ck["classes"]), m["n_severity"], m["n_vib_channels"],
                     m["embed_dim"], m["hidden"], m["dropout"], m["modality_dropout"])
    model.load_state_dict(ck["model"])
    model.eval()
    mel = MelExtractor(m["audio_rate"], m["win_seconds"], a["n_fft"], a["hop_length"], a["n_mels"])
    return {"model": model, "classes": ck["classes"], "modalities": tuple(ck["modalities"]),
            "stats": ck["sensor_stats"], "mel": mel}


@torch.no_grad()
def score_windows(bundle, vib: np.ndarray, aud: np.ndarray, hide_sensor=False, hide_audio=False):
    """vib (N, 6, T), aud (N, S) -> (fault probs (N, C), severity probs (N, 3)).

    A hidden modality is passed as None, exactly like the missing-input test in training.
    """
    mods = bundle["modalities"]
    inp = {}
    if "sensor" in mods:
        if hide_sensor:
            inp["sensor"] = None
        else:
            mean, std = bundle["stats"]
            inp["sensor"] = torch.from_numpy((vib - mean[None, :, None]) / std[None, :, None])
    if "audio" in mods:
        inp["audio"] = None if hide_audio else bundle["mel"](torch.from_numpy(aud))
    if all(v is None for v in inp.values()):  # nothing left to look at
        return None, None
    out = bundle["model"](inp)
    return out["fault_logits"].softmax(-1).numpy(), out["severity_logits"].softmax(-1).numpy()


# ------------------------------------------------------------------ vision (MVTec)
def load_vision(tag: str):
    """checkpoints/<tag>.pt -> dict(model, size). No pretrained download: weights come from the file."""
    from predictiq.models.vision_clf import VisionClassifier
    ck = torch.load(ROOT / "checkpoints" / f"{tag}.pt", map_location="cpu", weights_only=False)
    v = ck["vision_cfg"]
    model = VisionClassifier(v["backbone"], False, v["embed_dim"])
    model.load_state_dict(ck["model"])
    model.eval()
    return {"model": model, "size": ck["image_size"], "categories": ck["categories"]}


def diagnose_image(bundle, pil_image, target=None):
    """-> (P(defect), Grad-CAM heatmap (size, size) in [0,1], RGB uint8 image used by the model)."""
    from predictiq.data.vision import eval_transform
    from predictiq.explain.gradcam import gradcam
    size = bundle["size"]
    img = pil_image.convert("RGB").resize((size, size))
    cam, probs, _ = gradcam(bundle["model"], eval_transform(size)(img).unsqueeze(0), target)
    return float(probs[1]), cam, np.asarray(img)


# ------------------------------------------------------------------ RUL (C-MAPSS)
def load_rul(tag: str):
    """checkpoints/<tag>.pt (written by train_cmapss.py) -> dict(model, features, mean, std, window)."""
    import torch.nn as nn
    from predictiq.models.encoders import SensorEncoder
    ck = torch.load(ROOT / "checkpoints" / f"{tag}.pt", map_location="cpu", weights_only=False)
    s = ck["sensor_cfg"]
    model = nn.Sequential(SensorEncoder(len(ck["features"]), s["hidden"], s["layers"], s["dropout"],
                                        s["embed_dim"]), nn.Linear(s["embed_dim"], 1))
    model.load_state_dict(ck["model"])
    model.eval()
    return {"model": model, "features": ck["features"], "mean": ck["mean"], "std": ck["std"],
            "window": ck["window"], "max_rul": ck["max_rul"]}


@torch.no_grad()
def predict_rul(bundle, windows: np.ndarray) -> np.ndarray:
    """windows (T, window, F) normalised -> predicted RUL in cycles (T,)."""
    out = [bundle["model"](torch.from_numpy(windows[i:i + 512].astype("float32"))).squeeze(-1)
           for i in range(0, len(windows), 512)]
    return torch.cat(out).numpy() * 125.0


# ------------------------------------------------------------------ architecture tables
def layer_table(mode: str):
    """Real FaultNet for `mode` (sensor | audio | fusion) -> (layers DataFrame, total params).

    One row per leaf layer in execution order, with output shape for ONE example and its
    parameter count, collected with forward hooks on a dummy input.
    """
    import pandas as pd
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    m, a = cfg["mafaulda"], cfg["audio"]
    mods = {"sensor": ("sensor",), "audio": ("audio",), "fusion": ("sensor", "audio")}[mode]
    model = FaultNet(mods, 10, m["n_severity"], m["n_vib_channels"], m["embed_dim"], m["hidden"],
                     m["dropout"], m["modality_dropout"]).eval()
    n_vib = int(m["vib_rate"] * m["win_seconds"])
    frames = int(m["audio_rate"] * m["win_seconds"]) // a["hop_length"] + 1
    inputs = {}
    if "sensor" in mods:
        inputs["sensor"] = torch.randn(1, m["n_vib_channels"], n_vib)
    if "audio" in mods:
        inputs["audio"] = torch.randn(1, 1, a["n_mels"], frames)
    rows, hooks = [], []

    def make(name, mod):
        def hook(_mod, _inp, out):
            rows.append({"layer": name, "type": type(mod).__name__,
                         "output shape": "x".join(str(s) for s in out.shape[1:]),
                         "params": sum(p.numel() for p in mod.parameters(recurse=False))})
        return hook

    for name, mod in model.named_modules():
        if name and not list(mod.children()):
            hooks.append(mod.register_forward_hook(make(name, mod)))
    with torch.no_grad():
        model(inputs)
    for h in hooks:
        h.remove()
    return pd.DataFrame(rows), sum(p.numel() for p in model.parameters())
