import pytest
import torch

from predictiq.models.combined import SLOTS, CombinedModel


@pytest.fixture(scope="module")
def make():
    def _make(**kw):
        torch.manual_seed(0)
        return CombinedModel(pretrained=False, **kw)
    return _make


def batch(b=4):
    return {"vib": torch.randn(b, 6, 2000), "audio": torch.randn(b, 1, 64, 32),
            "image": torch.randn(b, 3, 64, 64), "engine": torch.randn(b, 50, 17)}


@pytest.mark.parametrize("slot", SLOTS)
def test_each_source_alone(make, slot):
    out = make().eval()({slot: batch()[slot]})
    assert out["fault_logits"].shape == (4, 10) and out["severity_logits"].shape == (4, 3)
    assert out["defect_logits"].shape == (4, 2) and out["rul_norm"].shape == (4,)


def test_all_sources_and_no_source(make):
    m = make().eval()
    assert m(batch())["fault_logits"].shape == (4, 10)
    with pytest.raises(ValueError):
        m({"vib": None})


def test_missing_source_equals_explicit_none(make):
    m = make().eval()
    b = batch()
    a = m({"vib": b["vib"]})["fault_logits"]
    c = m({"vib": b["vib"], "audio": None, "image": None, "engine": None})["fault_logits"]
    assert torch.allclose(a, c)


def test_gradients_reach_only_the_used_encoder(make):
    m = make().train()
    out = m({"engine": batch()["engine"]})
    out["rul_norm"].sum().backward()
    got = {s: any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.enc[s].parameters())
           for s in SLOTS}
    assert got == {"vib": False, "audio": False, "image": False, "engine": True}


def test_modality_dropout_never_hides_everything(make):
    m = make(modality_dropout=0.99).train()
    torch.manual_seed(1)
    for _ in range(5):
        out = m({"vib": batch()["vib"], "audio": batch()["audio"]})
        assert torch.isfinite(out["fault_logits"]).all()


def test_image_backbone_frozen_except_last_block(make):
    m = make()
    bb = m.enc["image"].backbone
    assert all(p.requires_grad for p in bb.layer4.parameters())
    assert not any(p.requires_grad for p in bb.layer1.parameters())


def test_param_report(make):
    rep = make().encoder_params()
    print(rep)
    assert set(rep) == {"vib encoder", "audio encoder", "image encoder", "engine encoder", "fusion", "heads"}
    assert rep["vib encoder"] == 128_992 and rep["audio encoder"] == 31_840   # same as FaultNet
