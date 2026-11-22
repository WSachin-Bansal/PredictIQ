"""Train a good-vs-defect classifier on MVTec AD (ResNet18, supervised).

    python -m predictiq.training.train_vision --categories bottle screw --seed 1 --epochs 10

First run downloads the pretrained ResNet18 weights (~45 MB) through torchvision.
Results: results/vision_<categories>_seed<N>.json, checkpoints/vision_<categories>_seed<N>.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix, recall_score
from torch.utils.data import DataLoader, Dataset

from predictiq.data.mvtec import build_index, split_images
from predictiq.data.vision import eval_transform, train_transform
from predictiq.models.vision_clf import VisionClassifier
from predictiq.training.metrics import classification_metrics


class ImageSet(Dataset):
    """Images resized once to size x size and kept in memory (a few hundred images)."""

    def __init__(self, rows: pd.DataFrame, size: int, transform):
        self.labels = rows["label"].to_numpy()
        self.ids = rows["id"].to_numpy()
        self.imgs = [Image.open(p).convert("RGB").resize((size, size)) for p in rows["path"]]
        self.tf = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), int(self.labels[i]), int(self.ids[i])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys, ids = [], [], []
    for x, y, i in loader:
        probs.append(model(x.to(device)).softmax(-1)[:, 1].cpu())
        ys.append(y)
        ids.append(i)
    return torch.cat(ids).numpy(), torch.cat(ys).numpy(), torch.cat(probs).numpy()


def score(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    m = classification_metrics(y, pred, p)
    m["defect_recall"] = float(recall_score(y, pred, pos_label=1, zero_division=0))
    m["good_recall"] = float(recall_score(y, pred, pos_label=0, zero_division=0))
    m["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/mvtec")
    ap.add_argument("--categories", nargs="+", default=["bottle", "screw"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    ap.add_argument("--backbone-lr", type=float, default=1e-4, help="last ResNet block")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    v = yaml.safe_load(open(args.config))["vision"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    index = build_index(args.root, args.categories)
    splits = split_images(index, args.seed)
    size = v["image_size"]
    print("images per class:", index.groupby(["category", "defect_type"]).size().to_dict())
    print({k: len(s) for k, s in splits.items()}, "images (train/val/test)")

    def make(name, train):
        rows = index[index.id.isin(splits[name])]
        return ImageSet(rows, size, train_transform(size) if train else eval_transform(size))

    tr_ds, va_ds, te_ds = make("train", True), make("val", False), make("test", False)
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va = DataLoader(va_ds, batch_size=64)
    te = DataLoader(te_ds, batch_size=64)

    model = VisionClassifier(v["backbone"], not args.no_pretrained, v["embed_dim"]).to(device)
    head = [p for n, p in model.named_parameters() if p.requires_grad and "layer4" not in n]
    last = [p for n, p in model.named_parameters() if p.requires_grad and "layer4" in n]
    opt = torch.optim.Adam([{"params": head, "lr": args.lr}, {"params": last, "lr": args.backbone_lr}])
    counts = np.bincount(tr_ds.labels, minlength=2)
    weight = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    print(f"train good/defect = {counts[0]}/{counts[1]}  class weights = {weight.tolist()}")

    history, best, best_state = [], -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y, _ in tr:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        _, yv, pv = predict(model, va, device)
        m = score(yv, pv)
        history.append({"epoch": ep, "train_loss": total / n, "val_f1": m["f1"],
                        "val_acc": m["accuracy"], "val_auc": m.get("roc_auc")})
        print(f"epoch {ep:02d} loss={total / n:.4f} val_acc={m['accuracy']:.3f} "
              f"val_f1={m['f1']:.3f} val_auc={m.get('roc_auc', float('nan')):.3f} ({time.time() - t0:.0f}s)")
        if m["f1"] > best:
            best, best_state = m["f1"], copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    ids, y, p = predict(model, te, device)
    test = score(y, p)
    sub = index.set_index("id").loc[ids]
    per_type = {}
    for d, g in sub.assign(p=p).groupby("defect_type"):
        per_type[d] = {"n": int(len(g)), "predicted_defect": float((g["p"] >= 0.5).mean())}
    print(f"\nTEST: acc={test['accuracy']:.3f} macro_f1={test['f1']:.3f} auc={test.get('roc_auc', float('nan')):.3f} "
          f"defect_recall={test['defect_recall']:.3f} good_recall={test['good_recall']:.3f}")
    print("share predicted 'defect' by defect type:",
          {k: round(v["predicted_defect"], 2) for k, v in per_type.items()})

    tag = f"vision_{'-'.join(args.categories)}_seed{args.seed}"
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    res = {"args": vars(args), "device_used": str(device), "categories": args.categories,
           "split_sizes": {k: len(s) for k, s in splits.items()}, "history": history, "test": test,
           "per_defect_type": per_type,
           "test_images": [{"path": r.path, "category": r.category, "defect_type": r.defect_type,
                            "label": int(r.label), "prob_defect": float(pp)}
                           for (_, r), pp in zip(sub.iterrows(), p)]}
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2))
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"model": best_state, "categories": args.categories, "image_size": size,
                "vision_cfg": v}, f"checkpoints/{tag}.pt")


if __name__ == "__main__":
    main()
