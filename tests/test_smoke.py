"""Shape / loss smoke test on random tensors (no real data needed)."""
import itertools
from pathlib import Path

import pytest
import torch
import yaml

from predictiq.models.multimodal import MODALITIES, PredictIQ
from predictiq.training.losses import MultiTaskLoss

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "default.yaml").read_text())
CFG["vision"]["pretrained"] = False  # no download in tests


def fake_batch(b=4):
    s, a = CFG["sensor"], CFG["audio"]
    frames = int(a["sample_rate"] * a["clip_seconds"] / a["hop_length"]) + 1
    return {
        "sensor": torch.randn(b, s["window"], s["n_features"]),
        "vision": torch.randn(b, 3, 224, 224),
        "audio": torch.randn(b, 1, a["n_mels"], frames),
    }


@pytest.mark.parametrize("mode", ["concat", "weighted", "attention"])
def test_full_model_and_loss(mode):
    cfg = {**CFG, "fusion": {**CFG["fusion"], "mode": mode}}
    model = PredictIQ(cfg)
    batch = fake_batch()
    out = model(batch)
    assert out["fault_logits"].shape == (4, cfg["fusion"]["n_classes"])
    assert out["failure_logit"].shape == (4,)
    assert out["rul"].shape == (4,)
    target = {"fault": torch.randint(0, cfg["fusion"]["n_classes"], (4,)),
              "failure": torch.rand(4).round(), "rul": torch.rand(4) * 125}
    loss = MultiTaskLoss()(out, target)
    loss["total"].backward()


def test_all_modality_subsets_and_missing():
    for r in (1, 2, 3):
        for subset in itertools.combinations(MODALITIES, r):
            model = PredictIQ(CFG, subset).eval()
            batch = {m: v for m, v in fake_batch().items() if m in subset}
            assert model(batch)["rul"].shape == (4,)
    full = PredictIQ(CFG).eval()
    b = fake_batch()
    b["vision"] = None  # missing modality
    assert full(b)["rul"].shape == (4,)
