"""Torch dataset + FaultNet checks (skipped automatically when torch is missing)."""
import pytest

torch = pytest.importorskip("torch")

from conftest import AUDIO_RATE, FS, VIB_RATE  # noqa: E402
from predictiq.data.mafaulda import prepare_dataset, split_files  # noqa: E402
from predictiq.data.mafaulda_dataset import (MafauldaWindows, fit_sensor_stats,  # noqa: E402
                                             load_index)
from predictiq.models.faultnet import FaultNet  # noqa: E402


def test_dataset_items_and_batching(synthetic_raw, tmp_path):
    out = tmp_path / "processed"
    prepare_dataset(synthetic_raw, out, fs=FS, vib_rate=VIB_RATE, audio_rate=AUDIO_RATE,
                    win_s=1.0, stride_s=0.5)
    index, classes = load_index(out)
    sp = split_files(index, "file", 0)
    stats = fit_sensor_stats(out, index, sp["train"])
    ds = MafauldaWindows(out, index, sp["train"], ("sensor", "audio"), stats)
    item = ds[0]
    assert item["sensor"].shape == (6, VIB_RATE) and item["audio"].shape == (AUDIO_RATE,)
    assert len(ds) == 3 * len(sp["train"])
    only_audio = MafauldaWindows(out, index, sp["train"], ("audio",))
    assert "sensor" not in only_audio[0]
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4)))
    assert batch["sensor"].shape == (4, 6, VIB_RATE) and batch["fault"].shape == (4,)


@pytest.mark.parametrize("mods", [("sensor",), ("audio",), ("sensor", "audio")])
def test_faultnet_shapes_and_missing_modality(mods):
    model = FaultNet(mods, n_classes=6)
    batch = {"sensor": torch.randn(4, 6, 10000), "audio": torch.randn(4, 1, 64, 32)}
    batch = {k: v for k, v in batch.items() if k in mods}
    model.train()
    out = model(batch)
    assert out["fault_logits"].shape == (4, 6) and out["severity_logits"].shape == (4, 3)
    out["fault_logits"].sum().backward()
    if len(mods) == 2:
        model.eval()
        for hidden in ("sensor", "audio"):
            partial = {**batch, hidden: None}
            assert model(partial)["fault_logits"].shape == (4, 6)
