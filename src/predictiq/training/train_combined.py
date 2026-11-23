"""Train ONE CombinedModel on all three datasets (MAFAULDA, MVTec AD, C-MAPSS).

    python -m predictiq.training.train_combined --init warm --seed 1 --epochs 10

Each optimiser step draws one batch from every dataset (the small ones are revisited more
rarely, see source_schedule) and applies ONLY the loss whose labels that dataset has:
    MAFAULDA (vibration + sound)  -> fault + severity
    MVTec    (image)              -> defect (good / defect)
    C-MAPSS  (engine sensors)     -> remaining useful life
Inputs a dataset does not have are zero embeddings. Data splits are identical to the separate
models' runs with the same seed, so the numbers can be compared one to one.

--init warm : start the four encoders from the separately trained checkpoints (needs
              checkpoints/{fusion_fault_<split>,vision_<cats>,rul_<subset>}_seed<N>.pt)
--init scratch : ImageNet ResNet18, everything else random.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from predictiq.data.audio import MelExtractor
from predictiq.data.mafaulda import relabel_for_task, split_files
from predictiq.data.mafaulda_dataset import MafauldaWindows, fit_sensor_stats, load_index
from predictiq.data.mvtec import build_index, split_images
from predictiq.data.sensor import (CMAPSS_COLS, add_rul, apply_scaler, drop_constant, fit_scaler,
                                   load_cmapss, make_windows, split_by_unit)
from predictiq.data.vision import eval_transform, train_transform
from predictiq.models.combined import CombinedModel
from predictiq.training.combined_utils import composite_score, remap_state, source_schedule
from predictiq.training.metrics import rul_metrics
from predictiq.training.train_cmapss import last_windows, nasa_score
from predictiq.training.train_mafaulda import summarize
from predictiq.training.train_vision import ImageSet
from predictiq.training.train_vision import score as vision_score


def cycle(loader):
    while True:
        yield from loader


# ----------------------------------------------------------------------------- data
def build_mafaulda(args, cfg):
    m = cfg["mafaulda"]
    index, _ = load_index(args.processed)
    index, classes = relabel_for_task(index, "fault")
    splits = split_files(index, args.split, m["split_seed"])
    stats = fit_sensor_stats(args.processed, index, splits["train"])

    def loader(name, shuffle):
        ds = MafauldaWindows(args.processed, index, splits[name], ("sensor", "audio"), stats)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)

    return {"train": loader("train", True), "val": loader("val", False), "test": loader("test", False),
            "classes": classes, "stats": stats}


def build_mvtec(args, cfg):
    size = cfg["vision"]["image_size"]
    index = build_index(args.mvtec_root, args.categories)
    splits = split_images(index, args.seed)  # same split as train_vision with this seed

    def make(name, train):
        rows = index[index.id.isin(splits[name])]
        return ImageSet(rows, size, train_transform(size) if train else eval_transform(size))

    tr = make("train", True)
    loaders = {"train": DataLoader(tr, batch_size=args.batch_size, shuffle=True),
               "val": DataLoader(make("val", False), batch_size=64),
               "test": DataLoader(make("test", False), batch_size=64)}
    counts = np.bincount(tr.labels, minlength=2)
    return {**loaders, "index": index, "class_weight": counts.sum() / (2.0 * np.maximum(counts, 1))}


def build_cmapss(args, cfg):
    s, max_rul = cfg["sensor"], cfg["rul"]["max_rul"]
    d = Path(args.cmapss_dir)
    train = add_rul(load_cmapss(str(d / f"train_{args.subset}.txt")), max_rul)
    test = load_cmapss(str(d / f"test_{args.subset}.txt"))
    true_rul = np.minimum(np.loadtxt(d / f"RUL_{args.subset}.txt"), max_rul)
    feats = drop_constant(train, [c for c in CMAPSS_COLS if c.startswith(("s", "op"))])
    tr_units, va_units = split_by_unit(train["unit"].to_numpy(), seed=args.seed)  # as train_cmapss
    tr_df, va_df = train[train.unit.isin(tr_units)], train[train.unit.isin(va_units)]
    mean, std = fit_scaler(tr_df, feats)
    tr_df, va_df = (apply_scaler(x, feats, mean, std) for x in (tr_df, va_df))
    test = apply_scaler(test, feats, mean, std)
    Xtr, ytr, _ = make_windows(tr_df, feats, s["window"], s["stride"])
    Xva, yva, _ = make_windows(va_df, feats, s["window"], s["stride"])
    Xte, _ = last_windows(test, feats, s["window"])
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                        batch_size=cfg["train"]["batch_size"], shuffle=True)
    return {"train": loader, "Xva": Xva, "yva": yva, "Xte": Xte, "true_rul": true_rul,
            "features": feats, "mean": mean, "std": std}


# ----------------------------------------------------------------------------- inputs / eval
def maf_inputs(batch, mel, device, drop=()):
    inp = {}
    if "vib" not in drop:
        inp["vib"] = batch["sensor"].to(device)
    if "audio" not in drop:
        inp["audio"] = mel(batch["audio"].to(device))
    return inp


@torch.no_grad()
def eval_maf(model, loader, mel, device, n_classes, drop=()):
    model.eval()
    fids, ys, st, probs, sp = [], [], [], [], []
    for b in loader:
        out = model(maf_inputs(b, mel, device, drop))
        probs.append(out["fault_logits"].softmax(-1).cpu())
        sp.append(out["severity_logits"].argmax(-1).cpu())
        ys.append(b["fault"])
        st.append(b["severity"])
        fids.append(b["file_id"])
    cat = lambda xs: torch.cat(xs).numpy()  # noqa: E731
    return summarize(cat(fids), cat(ys), cat(probs), cat(st), cat(sp), n_classes)


@torch.no_grad()
def eval_vision(model, loader, device):
    model.eval()
    ids, ys, ps = [], [], []
    for x, y, i in loader:
        ps.append(model({"image": x.to(device)})["defect_logits"].softmax(-1)[:, 1].cpu())
        ys.append(y)
        ids.append(i)
    return torch.cat(ids).numpy(), torch.cat(ys).numpy(), torch.cat(ps).numpy()


@torch.no_grad()
def eval_engine(model, X, device):
    model.eval()
    out = [model({"engine": torch.tensor(X[i:i + 512]).to(device)})["rul_norm"].cpu()
           for i in range(0, len(X), 512)]
    return torch.cat(out).numpy() * 125.0


# ----------------------------------------------------------------------------- warm start
def warm_start(model, args):
    ck = Path("checkpoints")
    jobs = [("vibration + sound", ck / f"fusion_fault_{args.split}_seed{args.seed}.pt",
             [("enc.sensor.", "enc.vib."), ("enc.audio.", "enc.audio.")]),
            ("image", ck / f"vision_{'-'.join(args.categories)}_seed{args.seed}.pt",
             [("encoder.", "enc.image.")]),
            ("engine", ck / f"rul_{args.subset}_seed{args.seed}.pt", [("0.", "enc.engine.")])]
    merged = {}
    for name, path, rules in jobs:
        if not path.exists():
            raise FileNotFoundError(f"warm start needs {path} (train that model first, or use --init scratch)")
        part = remap_state(torch.load(path, map_location="cpu", weights_only=False)["model"], rules)
        merged.update(part)
        print(f"warm start: {len(part):3d} tensors for the {name} encoder(s) from {path.name}")
    missing, unexpected = model.load_state_dict(merged, strict=False)
    left = [k for k in missing if k.startswith("enc.")]
    if left or unexpected:
        raise RuntimeError(f"warm start did not match the model: missing encoder keys {left[:3]}, "
                           f"unexpected {list(unexpected)[:3]}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["warm", "scratch"], default="warm")
    ap.add_argument("--split", choices=["file", "severity"], default="file", help="MAFAULDA split")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10, help="1 epoch = one pass over the MAFAULDA training windows")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-4, help="last ResNet block")
    ap.add_argument("--w-fault", type=float, default=1.0)
    ap.add_argument("--w-severity", type=float, default=0.5)
    ap.add_argument("--w-defect", type=float, default=1.0)
    ap.add_argument("--w-rul", type=float, default=5.0, help="RUL loss is numerically small, so it gets a larger weight")
    ap.add_argument("--vision-passes", type=float, default=1.0, help="passes over the MVTec train set per epoch")
    ap.add_argument("--engine-passes", type=float, default=2.0, help="passes over the C-MAPSS train set per epoch")
    ap.add_argument("--processed", default="data/processed/mafaulda")
    ap.add_argument("--mvtec-root", default="data/raw/mvtec")
    ap.add_argument("--categories", nargs="+", default=["bottle", "screw"])
    ap.add_argument("--cmapss-dir", default="data/raw/cmapss")
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    m, a, v, s = cfg["mafaulda"], cfg["audio"], cfg["vision"], cfg["sensor"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    maf, vis, eng = build_mafaulda(args, cfg), build_mvtec(args, cfg), build_cmapss(args, cfg)
    classes = maf["classes"]
    steps = len(maf["train"])
    print(f"steps per epoch: {steps} (MAFAULDA batches) | MVTec batches/pass {len(vis['train'])} | "
          f"C-MAPSS batches/pass {len(eng['train'])} | engine features {len(eng['features'])}")

    model = CombinedModel(len(classes), m["n_severity"], m["n_vib_channels"], len(eng["features"]),
                          m["embed_dim"], m["hidden"], m["dropout"], m["modality_dropout"],
                          s["hidden"], s["layers"], s["dropout"], v["backbone"],
                          pretrained=args.init == "scratch").to(device)
    if args.init == "warm":
        warm_start(model, args)
    mel = MelExtractor(m["audio_rate"], m["win_seconds"], a["n_fft"], a["hop_length"], a["n_mels"]).to(device)

    last = [p for n, p in model.named_parameters() if p.requires_grad and "enc.image.backbone.layer4" in n]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and "enc.image.backbone.layer4" not in n]
    opt = torch.optim.Adam([{"params": rest, "lr": args.lr}, {"params": last, "lr": args.backbone_lr}],
                           weight_decay=cfg["train"]["weight_decay"])
    cw = torch.tensor(vis["class_weight"], dtype=torch.float32, device=device)
    vis_iter, eng_iter = cycle(vis["train"]), cycle(eng["train"])

    history, best, best_state = [], -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        use_vis = source_schedule(steps, len(vis["train"]), args.vision_passes)
        use_eng = source_schedule(steps, len(eng["train"]), args.engine_passes)
        tot = {"maf": 0.0, "vis": 0.0, "eng": 0.0}
        cnt = {"maf": 0, "vis": 0, "eng": 0}
        for step, b in enumerate(maf["train"]):
            opt.zero_grad()
            # MAFAULDA: vibration + sound -> fault + severity
            out = model(maf_inputs(b, mel, device))
            loss = args.w_fault * F.cross_entropy(out["fault_logits"], b["fault"].to(device))
            sev = b["severity"].to(device)
            if (sev >= 0).any():
                loss = loss + args.w_severity * F.cross_entropy(out["severity_logits"], sev, ignore_index=-1)
            loss.backward()
            tot["maf"] += loss.item()
            cnt["maf"] += 1
            if use_vis[step]:  # MVTec: image -> defect
                x, y, _ = next(vis_iter)
                loss = args.w_defect * F.cross_entropy(model({"image": x.to(device)})["defect_logits"],
                                                       y.to(device), weight=cw)
                loss.backward()
                tot["vis"] += loss.item()
                cnt["vis"] += 1
            if use_eng[step]:  # C-MAPSS: engine sensors -> RUL
                x, y = next(eng_iter)
                pred = model({"engine": x.to(device)})["rul_norm"]
                loss = args.w_rul * F.smooth_l1_loss(pred, y.to(device) / 125.0)
                loss.backward()
                tot["eng"] += loss.item()
                cnt["eng"] += 1
            opt.step()

        mv = eval_maf(model, maf["val"], mel, device, len(classes))
        ids, yv, pv = eval_vision(model, vis["val"], device)
        dv = vision_score(yv, pv)
        rv = rul_metrics(eng["yva"], eval_engine(model, eng["Xva"], device))
        comp = composite_score(mv["f1"], dv["f1"], rv["rmse"])
        history.append({"epoch": epoch, "loss_maf": tot["maf"] / max(cnt["maf"], 1),
                        "loss_vis": tot["vis"] / max(cnt["vis"], 1), "loss_eng": tot["eng"] / max(cnt["eng"], 1),
                        "val_fault_f1": mv["f1"], "val_fault_acc": mv["accuracy"], "val_defect_f1": dv["f1"],
                        "val_defect_auc": dv.get("roc_auc"), "val_rul_rmse": float(rv["rmse"]), "score": comp})
        print(f"epoch {epoch:02d} loss maf/vis/eng={history[-1]['loss_maf']:.3f}/{history[-1]['loss_vis']:.3f}/"
              f"{history[-1]['loss_eng']:.3f} | fault_f1={mv['f1']:.3f} defect_f1={dv['f1']:.3f} "
              f"rul_rmse={rv['rmse']:.1f} | score={comp:.3f} ({time.time() - t0:.0f}s)")
        if comp > best:
            best, best_state = comp, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    ids, yt, pt = eval_vision(model, vis["test"], device)
    sub = vis["index"].set_index("id").loc[ids].assign(p=pt)
    pred_rul = eval_engine(model, eng["Xte"], device)
    rul = {k: float(x) for k, x in rul_metrics(eng["true_rul"], pred_rul).items()}
    rul["nasa_score"] = nasa_score(eng["true_rul"], pred_rul)
    rul["late_fraction"] = float((pred_rul > eng["true_rul"]).mean())
    test = {"fault": eval_maf(model, maf["test"], mel, device, len(classes)),
            "fault_missing_vibration": eval_maf(model, maf["test"], mel, device, len(classes), drop=("vib",)),
            "fault_missing_sound": eval_maf(model, maf["test"], mel, device, len(classes), drop=("audio",)),
            "defect": vision_score(yt, pt), "rul": rul,
            "defect_by_type": {d: {"n": int(len(g)), "predicted_defect": float((g["p"] >= 0.5).mean())}
                               for d, g in sub.groupby("defect_type")}}
    f, d_ = test["fault"], test["defect"]
    print(f"\nTEST (init={args.init}, split={args.split}, seed={args.seed})\n"
          f"  fault : acc={f['accuracy']:.4f} macro_f1={f['f1']:.4f} file_acc={f['file_accuracy']:.4f}\n"
          f"          vibration hidden acc={test['fault_missing_vibration']['accuracy']:.4f} | "
          f"sound hidden acc={test['fault_missing_sound']['accuracy']:.4f}\n"
          f"  defect: acc={d_['accuracy']:.3f} macro_f1={d_['f1']:.3f} auc={d_.get('roc_auc', float('nan')):.3f} "
          f"defect_recall={d_['defect_recall']:.3f}\n"
          f"  RUL   : rmse={rul['rmse']:.2f} mae={rul['mae']:.2f} nasa={rul['nasa_score']:.0f}")

    tag = f"combined_{args.init}_{args.split}_seed{args.seed}"
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps({
        "args": vars(args), "device_used": str(device), "classes": classes, "history": history,
        "test": test, "params": model.encoder_params(),
        "rul_test_true": eng["true_rul"].tolist(), "rul_test_pred": pred_rul.tolist()}, indent=2))
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"model": best_state, "classes": classes, "engine_features": eng["features"],
                "engine_mean": eng["mean"], "engine_std": eng["std"], "vib_stats": maf["stats"],
                "categories": args.categories, "image_size": v["image_size"], "args": vars(args)},
               f"checkpoints/{tag}.pt")


if __name__ == "__main__":
    main()
