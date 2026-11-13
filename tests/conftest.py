"""Synthetic MAFAULDA-like recordings: tiny, but same folder layout and 8-column format."""
import numpy as np
import pytest

FS = 5000          # scaled-down rates (same 10:1 and 25:8 ratios as the real data)
VIB_RATE = 1000
AUDIO_RATE = 1600
SECONDS = 2


def write_recording(path, fault_hz, seed):
    rng = np.random.default_rng(seed)
    n = FS * SECONDS
    t = np.arange(n) / FS
    cols = [np.full(n, 1.0)]                                   # tachometer
    for _ in range(6):                                         # accelerometers
        cols.append(np.sin(2 * np.pi * fault_hz * t) + 0.1 * rng.standard_normal(n))
    cols.append(np.sin(2 * np.pi * 3 * fault_hz * t) + 0.1 * rng.standard_normal(n))  # mic
    np.savetxt(path, np.stack(cols, axis=1), delimiter=",", fmt="%.5f")


@pytest.fixture
def synthetic_raw(tmp_path):
    root = tmp_path / "raw" / "mafaulda"            # wrapper folder, like a Kaggle unzip
    layout = {                                       # folder -> (n_files, base frequency)
        "normal": (6, 20),
        "imbalance/6g": (3, 40), "imbalance/10g": (3, 45), "imbalance/35g": (3, 50),
        "overhang/ball_fault/6g": (3, 80), "overhang/ball_fault/20g": (3, 85),
    }
    seed = 0
    for folder, (n, hz) in layout.items():
        d = root / folder
        d.mkdir(parents=True)
        for k in range(n):
            write_recording(d / f"{12.0 + k * 1.5:.3f}.csv", hz + k, seed)
            seed += 1
    return tmp_path / "raw"
