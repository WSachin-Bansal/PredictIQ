"""Train an LSTM to predict Remaining Useful Life (RUL) on NASA C-MAPSS.

    python -m predictiq.training.train_cmapss --subset FD001 --seed 1 --epochs 30

Expects in data/raw/cmapss/:  train_FD001.txt, test_FD001.txt, RUL_FD001.txt
Evaluation follows the official protocol: for every test engine we take the LAST window of
its (cut-off) trajectory and compare with the true RUL from RUL_FD001.txt.
Validation (for picking the best epoch) uses held-out TRAIN engines, never the test engines.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from predictiq.data.sensor import (CMAPSS_COLS, add_rul, apply_scaler, drop_constant,
                                   fit_scaler, load_cmapss, make_windows, split_by_unit)
from predictiq.models.encoders import SensorEncoder
from predictiq.training.metrics import rul_metrics


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric C-MAPSS score: late predictions (pred > true) are punished harder."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def last_windows(df: pd.DataFrame, cols: list[str], window: int):
    """One window per engine: its last `window` cycles (front-padded by repeating row 1)."""
    xs, units = [], []
    for unit, g in df.groupby("unit"):
        f = g[cols].to_numpy(dtype=np.float32)
        if len(f) < window:
            f = np.concatenate([np.repeat(f[:1], window - len(f), axis=0), f])
        xs.append(f[-window:])
        units.append(unit)
    return np.stack(xs), np.array(units)


def predict(model, x: np.ndarray, device, bs: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model(torch.tensor(x[i:i + bs]).to(device)).squeeze(-1).cpu())
    return torch.cat(out).numpy() * 125.0  # model works on RUL/125


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/cmapss")
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    s, max_rul = cfg["sensor"], cfg["rul"]["max_rul"]
    epochs = args.epochs or cfg["train"]["epochs"]
    bs = args.batch_size or cfg["train"]["batch_size"]
    lr = args.lr or cfg["train"]["lr"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"device: {device}")

    d = Path(args.data_dir)
    train = add_rul(load_cmapss(str(d / f"train_{args.subset}.txt")), max_rul)
    test = load_cmapss(str(d / f"test_{args.subset}.txt"))
    true_rul = np.minimum(np.loadtxt(d / f"RUL_{args.subset}.txt"), max_rul)

    feat_cols = drop_constant(train, [c for c in CMAPSS_COLS if c.startswith(("s", "op"))])
    tr_units, va_units = split_by_unit(train["unit"].to_numpy(), seed=args.seed)
    tr_df, va_df = train[train.unit.isin(tr_units)], train[train.unit.isin(va_units)]
    mean, std = fit_scaler(tr_df, feat_cols)  # TRAIN engines only
    tr_df, va_df = (apply_scaler(x, feat_cols, mean, std) for x in (tr_df, va_df))
    test = apply_scaler(test, feat_cols, mean, std)

    Xtr, ytr, _ = make_windows(tr_df, feat_cols, s["window"], s["stride"])
    Xva, yva, _ = make_windows(va_df, feat_cols, s["window"], s["stride"])
    Xte, te_units = last_windows(test, feat_cols, s["window"])
    assert len(Xte) == len(true_rul), "test engines and RUL file length differ"
    print(f"{args.subset}: {len(feat_cols)} features | engines train/val/test = "
          f"{len(tr_units)}/{len(va_units)}/{len(Xte)} | windows train/val = {len(Xtr)}/{len(Xva)}")

    model = nn.Sequential(SensorEncoder(len(feat_cols), s["hidden"], s["layers"], s["dropout"],
                                        s["embed_dim"]), nn.Linear(s["embed_dim"], 1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg["train"]["weight_decay"])
    tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=bs, shuffle=True)

    history, best_rmse, best_state = [], 1e9, None
    for ep in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.smooth_l1_loss(model(xb).squeeze(-1), yb / 125.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        val = rul_metrics(yva, predict(model, Xva, device))
        history.append({"epoch": ep, "train_loss": total / n, "val_rmse": val["rmse"],
                        "val_mae": val["mae"]})
        print(f"epoch {ep:02d} loss={total / n:.5f} val_rmse={val['rmse']:.2f} "
              f"val_mae={val['mae']:.2f} ({time.time() - t0:.0f}s)")
        if val["rmse"] < best_rmse:
            best_rmse = val["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    pred = predict(model, Xte, device)
    m = rul_metrics(true_rul, pred)
    m["nasa_score"] = nasa_score(true_rul, pred)
    m["late_fraction"] = float((pred > true_rul).mean())  # dangerous side: predicted too long
    print(f"\nTEST {args.subset} (seed {args.seed}): RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} "
          f"R2={m['r2']:.3f} NASA={m['nasa_score']:.0f} late={m['late_fraction']:.0%}")

    tag = f"rul_{args.subset}_seed{args.seed}"
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    res = {"args": vars(args), "features": feat_cols, "history": history,
           "n_engines": {"train": len(tr_units), "val": len(va_units), "test": len(Xte)},
           "test": {k: float(v) for k, v in m.items()},
           "test_engines": te_units.tolist(), "test_true": true_rul.tolist(),
           "test_pred": pred.tolist()}
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2))
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"model": best_state, "features": feat_cols, "mean": mean, "std": std,
                "window": s["window"], "max_rul": max_rul, "sensor_cfg": s},
               f"checkpoints/{tag}.pt")

    try:  # a quick picture: predicted vs true RUL for every test engine
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = np.argsort(true_rul)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(true_rul[order], label="true RUL", lw=2)
        ax.plot(pred[order], "o", ms=4, label="predicted RUL")
        ax.set(xlabel="test engines (sorted by true RUL)", ylabel="RUL (cycles)",
               title=f"{args.subset} test: RMSE {m['rmse']:.1f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{tag}.png", dpi=150)
    except ImportError:
        print("(matplotlib not installed - skipped plot)")


if __name__ == "__main__":
    main()
