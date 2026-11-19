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
