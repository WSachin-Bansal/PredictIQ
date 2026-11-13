"""Index, file-level splits and preprocessing (no torch needed)."""
import numpy as np

from conftest import AUDIO_RATE, FS, VIB_RATE
from predictiq.data.mafaulda import build_index, prepare_dataset, split_files


def test_index_parses_classes_severity_speed(synthetic_raw):
    index, classes = build_index(synthetic_raw)          # must see through the wrapper dir
    assert classes == ["imbalance", "normal", "overhang_ball_fault"]
    assert len(index) == 6 + 9 + 6
    imb = index[index.fault == "imbalance"]
    assert set(imb.severity_label) == {"6g", "10g", "35g"}
    assert sorted(imb.severity_bin.unique()) == [0, 1, 2]
    assert (index[index.fault == "normal"].severity_bin == -1).all()
    assert index.speed_hz.notna().all()


def _check_partition(index, sp):
    all_ids = np.concatenate([sp["train"], sp["val"], sp["test"]])
    assert len(all_ids) == len(set(all_ids)) == len(index)   # disjoint and complete


def test_file_split_disjoint_stratified_deterministic(synthetic_raw):
    index, _ = build_index(synthetic_raw)
    sp = split_files(index, "file", seed=1)
    _check_partition(index, sp)
    for name in ("train", "val", "test"):
        assert set(index[index.id.isin(sp[name])].fault) == set(index.fault)
    again = split_files(index, "file", seed=1)
    assert all((sp[k] == again[k]).all() for k in sp)


def test_severity_split_holds_out_severity_levels(synthetic_raw):
    index, _ = build_index(synthetic_raw)
    sp = split_files(index, "severity", seed=1)
    _check_partition(index, sp)
    imb = index[index.fault == "imbalance"]
    levels = {n: set(imb[imb.id.isin(sp[n])].severity_label)
              for n in ("train", "val", "test")}
    assert levels["train"].isdisjoint(levels["val"] | levels["test"])
    assert levels["val"].isdisjoint(levels["test"])


def test_prepare_dataset_windows_are_aligned(synthetic_raw, tmp_path):
    out = tmp_path / "processed"
    index = prepare_dataset(synthetic_raw, out, fs=FS, vib_rate=VIB_RATE,
                            audio_rate=AUDIO_RATE, win_s=1.0, stride_s=0.5)
    assert len(index) == 21 and (index.n_windows == 3).all()
    r = index.iloc[0]
    vib = np.load(out / "vib" / f"{r.key}.npy")
    aud = np.load(out / "aud" / f"{r.key}.npy")
    assert vib.shape == (3, 6, VIB_RATE) and aud.shape == (3, AUDIO_RATE)
    assert vib.dtype == np.float16 and np.isfinite(vib.astype(np.float32)).all()
    assert (out / "index.csv").exists() and (out / "meta.json").exists()
    # re-running is a no-op that reuses cached files
    again = prepare_dataset(synthetic_raw, out, fs=FS, vib_rate=VIB_RATE,
                            audio_rate=AUDIO_RATE, win_s=1.0, stride_s=0.5)
    assert len(again) == len(index)


def test_nested_duplicate_folders_and_header_row(tmp_path):
    """Layout seen in the real download: imbalance/imbalance/6g/<speed>.csv"""
    from conftest import write_recording
    from predictiq.data.mafaulda import prepare_file
    root = tmp_path / "raw" / "mafaulda"
    for folder, hz in (("normal/normal", 20), ("imbalance/imbalance/6g", 40),
                       ("imbalance/imbalance/10g", 45), ("imbalance/imbalance/35g", 50)):
        d = root / folder
        d.mkdir(parents=True)
        for k in range(2):
            write_recording(d / f"{12.0 + k:.3f}.csv", hz + k, k)
    index, classes = build_index(tmp_path / "raw")
    assert classes == ["imbalance", "normal"]
    assert set(index[index.fault == "imbalance"].severity_label) == {"6g", "10g", "35g"}
    assert sorted(index[index.fault == "imbalance"].severity_bin.unique()) == [0, 1, 2]
    # a CSV with a text header row must still load
    p = tmp_path / "hdr.csv"
    p.write_text("a,b,c,d,e,f,g,h\n" + open(index.iloc[0].path).read())
    vib, aud = prepare_file(p, fs=FS, vib_rate=VIB_RATE, audio_rate=AUDIO_RATE)
    assert vib.shape[1:] == (6, VIB_RATE) and aud.shape[1] == AUDIO_RATE


def test_relabel_for_task(synthetic_raw):
    from predictiq.data.mafaulda import relabel_for_task
    index, _ = build_index(synthetic_raw)
    idx_f, cls_f = relabel_for_task(index, "fault")
    assert cls_f == ["imbalance", "normal", "overhang_ball_fault"]
    idx_s, cls_s = relabel_for_task(index, "severity")
    assert cls_s == ["imbalance_6g", "imbalance_10g", "imbalance_35g", "normal",
                     "overhang_ball_fault_6g", "overhang_ball_fault_20g"]
    assert sorted(idx_s.label.unique()) == list(range(6))
    # file split stratifies by the new target: every class appears in train/val/test
    sp = split_files(idx_s, "file", seed=0)
    for name in ("train", "val", "test"):
        assert set(idx_s[idx_s.id.isin(sp[name])].label) == set(range(6)) or name != "train"
    assert set(idx_s[idx_s.id.isin(sp["train"])].label) == set(range(6))


def test_cache_survives_new_folders(synthetic_raw, tmp_path):
    """Adding folders later (e.g. a download still running) must not mix up cached files."""
    import shutil
    from conftest import write_recording
    out = tmp_path / "processed"
    kw = dict(fs=FS, vib_rate=VIB_RATE, audio_rate=AUDIO_RATE, win_s=1.0, stride_s=0.5)
    first = prepare_dataset(synthetic_raw, out, **kw)
    before = {k: np.load(out / "vib" / f"{k}.npy").copy() for k in first.key}
    new = synthetic_raw / "mafaulda" / "aaa_new_fault"      # sorts BEFORE all existing folders
    new.mkdir()
    for k in range(3):
        write_recording(new / f"{10.0 + k:.3f}.csv", 99 + k, 100 + k)
    second = prepare_dataset(synthetic_raw, out, **kw)
    assert len(second) == len(first) + 3
    for k, arr in before.items():
        assert (np.load(out / "vib" / f"{k}.npy") == arr).all()
    shutil.rmtree(new)
