import numpy as np
from PIL import Image

from predictiq.data.mvtec import build_index, find_category, split_images


def _img(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(p)


def make_tree(root, wrapper=False):
    base = root / "mvtec_anomaly_detection" if wrapper else root
    for cat in ("bottle", "screw"):
        for i in range(20):
            _img(base / cat / "train" / "good" / f"{i:03d}.png")
        for i in range(5):
            _img(base / cat / "test" / "good" / f"{i:03d}.png")
        for d in ("broken", "scratch"):
            for i in range(5):
                _img(base / cat / "test" / d / f"{i:03d}.png")
                _img(base / cat / "ground_truth" / d / f"{i:03d}_mask.png")
    return root


def test_index_labels_and_masks_ignored(tmp_path):
    idx = build_index(make_tree(tmp_path), ["bottle"])
    assert len(idx) == 20 + 5 + 10                      # masks excluded
    assert set(idx.defect_type) == {"good", "broken", "scratch"}
    assert (idx[idx.defect_type == "good"].label == 0).all()
    assert (idx[idx.defect_type != "good"].label == 1).all()


def test_wrapper_folder_found(tmp_path):
    make_tree(tmp_path, wrapper=True)
    assert find_category(tmp_path, "screw").name == "screw"


def test_split_disjoint_stratified_deterministic(tmp_path):
    idx = build_index(make_tree(tmp_path), ["bottle", "screw"])
    s = split_images(idx, seed=1)
    allids = s["train"] + s["val"] + s["test"]
    assert sorted(allids) == sorted(idx.id) and len(set(allids)) == len(allids)
    test = idx[idx.id.isin(s["test"])]
    assert set(zip(test.category, test.label)) == {("bottle", 0), ("bottle", 1), ("screw", 0), ("screw", 1)}
    assert s == split_images(idx, seed=1) and s != split_images(idx, seed=2)
