"""Train sensor-only / audio-only / fusion models on preprocessed MAFAULDA.

Usage (after `python -m predictiq.data.mafaulda --raw ... --out data/processed/mafaulda`):
    python -m predictiq.training.train_mafaulda --mode sensor
    python -m predictiq.training.train_mafaulda --mode audio
    python -m predictiq.training.train_mafaulda --mode fusion

All modes use the same file-level split (it depends only on --split / --split-seed), so
their numbers are directly comparable. Fusion also reports test scores with one modality
zeroed out (missing-modality robustness). Results go to results/<mode>_<split>_seed<N>.json.
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
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from predictiq.data.audio import MelExtractor
from predictiq.data.mafaulda import relabel_for_task, split_files
from predictiq.data.mafaulda_dataset import MafauldaWindows, fit_sensor_stats, load_index
from predictiq.models.faultnet import FaultNet
from predictiq.training.metrics import classification_metrics

MODES = {"sensor": ("sensor",), "audio": ("audio",), "fusion": ("sensor", "audio")}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_inputs(batch, modalities, mel, device, drop=()):
    """Move a batch to the device; `drop` lists modalities to hide (-> zero embedding)."""
    inp = {}
    if "sensor" in modalities:
        inp["sensor"] = None if "sensor" in drop else batch["sensor"].to(device)
    if "audio" in modalities:
        inp["audio"] = None if "audio" in drop else mel(batch["audio"].to(device))
    return inp


@torch.no_grad()
def predict(model, loader, modalities, mel, device, drop=()):
    model.eval()
    fids, ys, sev_t, probs, sev_p = [], [], [], [], []
    for batch in loader:
        out = model(make_inputs(batch, modalities, mel, device, drop))
        probs.append(out["fault_logits"].softmax(-1).cpu())
        sev_p.append(out["severity_logits"].argmax(-1).cpu())
        ys.append(batch["fault"])
        sev_t.append(batch["severity"])
        fids.append(batch["file_id"])
    cat = lambda xs: torch.cat(xs).numpy()  # noqa: E731
    return cat(fids), cat(ys), cat(probs), cat(sev_t), cat(sev_p)


def summarize(fids, y, probs, sev_t, sev_p, n_classes) -> dict:
    pred = probs.argmax(1)
    m = classification_metrics(y, pred)
    # file level: average the window probabilities of each recording, then argmax
    df = pd.DataFrame(probs, columns=list(range(n_classes)))
    df["file"], df["y"] = fids, y
    g = df.groupby("file")
    file_pred = g[list(range(n_classes))].mean().to_numpy().argmax(1)
    m["file_accuracy"] = float((file_pred == g["y"].first().to_numpy()).mean())
    valid = sev_t >= 0
    m["severity_accuracy"] = float((sev_p[valid] == sev_t[valid]).mean()) if valid.any() else None
    m["confusion_matrix"] = confusion_matrix(y, pred, labels=list(range(n_classes))).tolist()
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES), required=True)
    ap.add_argument("--processed", default="data/processed/mafaulda")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--task", choices=["fault", "severity"], default="fault",
                    help="fault: one class per fault type; severity: one class per "
                         "(fault, severity level), e.g. normal, imbalance_6g, ...")
    ap.add_argument("--split", choices=["file", "severity"], default="file")
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    m, a = cfg["mafaulda"], cfg["audio"]
    epochs = args.epochs or m["epochs"]
    bs = args.batch_size or m["batch_size"]
    lr = args.lr or m["lr"]
    split_seed = m["split_seed"] if args.split_seed is None else args.split_seed
    modalities = MODES[args.mode]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if args.task == "severity" and args.split == "severity":
        ap.error("--task severity needs every severity level in training; use --split file")
    index, _ = load_index(args.processed)
    index, classes = relabel_for_task(index, args.task)
    splits = split_files(index, args.split, split_seed)
    print({k: len(v) for k, v in splits.items()}, "recordings (train/val/test), classes:", classes)
    stats = fit_sensor_stats(args.processed, index, splits["train"]) if "sensor" in modalities else None

    def loader(name, shuffle):
        ds = MafauldaWindows(args.processed, index, splits[name], modalities, stats)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=args.num_workers,
                          pin_memory=device.type == "cuda")

    tr, va, te = loader("train", True), loader("val", False), loader("test", False)
    mel = MelExtractor(m["audio_rate"], m["win_seconds"], a["n_fft"], a["hop_length"],
                       a["n_mels"]).to(device)
    model = FaultNet(modalities, len(classes), m["n_severity"], m["n_vib_channels"],
                     m["embed_dim"], m["hidden"], m["dropout"], m["modality_dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg["train"]["weight_decay"])

    def evaluate(loader_, drop=()):
        return summarize(*predict(model, loader_, modalities, mel, device, drop),
                         n_classes=len(classes))

    history, best_f1, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for batch in tr:
            out = model(make_inputs(batch, modalities, mel, device))
            loss = F.cross_entropy(out["fault_logits"], batch["fault"].to(device))
            sev = batch["severity"].to(device)
            if (sev >= 0).any():
                loss = loss + 0.5 * F.cross_entropy(out["severity_logits"], sev, ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(sev)
            n += len(sev)
        val = evaluate(va)
        history.append({"epoch": epoch, "train_loss": total / n, "val_f1": val["f1"],
                        "val_acc": val["accuracy"], "val_file_acc": val["file_accuracy"]})
        print(f"epoch {epoch:02d} loss={total / n:.4f} val_acc={val['accuracy']:.4f} "
              f"val_f1={val['f1']:.4f} val_file_acc={val['file_accuracy']:.4f} "
              f"({time.time() - t0:.0f}s)")
        if val["f1"] > best_f1:
            best_f1, best_state = val["f1"], copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    results = {"args": vars(args), "split_sizes": {k: len(v) for k, v in splits.items()},
               "classes": classes, "history": history, "test": evaluate(te)}
    if args.mode == "fusion":
        results["test_missing_sensor"] = evaluate(te, drop=("sensor",))
        results["test_missing_audio"] = evaluate(te, drop=("audio",))

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.mode}_{args.task}_{args.split}_seed{args.seed}"
    (out_dir / f"{tag}.json").write_text(json.dumps(results, indent=2))
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"model": best_state, "classes": classes, "modalities": modalities,
                "sensor_stats": stats}, f"checkpoints/{tag}.pt")

    print(f"\nTEST ({tag}): " + ", ".join(
        f"{k}={results['test'][k]:.4f}" for k in ("accuracy", "f1", "file_accuracy")))
    for k in ("test_missing_sensor", "test_missing_audio"):
        if k in results:
            print(f"{k}: acc={results[k]['accuracy']:.4f} f1={results[k]['f1']:.4f}")


if __name__ == "__main__":
    main()
