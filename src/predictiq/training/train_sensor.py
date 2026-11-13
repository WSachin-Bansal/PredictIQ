"""Baseline M1: sensor-only RUL on NASA C-MAPSS.

Usage:
    python -m predictiq.training.train_sensor --data data/raw/train_FD001.txt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from predictiq.data.sensor import (CMAPSS_COLS, add_rul, apply_scaler, drop_constant,
                                   fit_scaler, load_cmapss, make_windows, split_by_unit)
from predictiq.models.encoders import SensorEncoder
from predictiq.training.metrics import rul_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    df = add_rul(load_cmapss(args.data), cfg["rul"]["max_rul"])
    feat_cols = drop_constant(df, [c for c in CMAPSS_COLS if c.startswith(("s", "op"))])
    tr_units, va_units = split_by_unit(df["unit"].to_numpy(), seed=cfg["seed"])
    tr_df, va_df = df[df.unit.isin(tr_units)], df[df.unit.isin(va_units)]

    mean, std = fit_scaler(tr_df, feat_cols)           # fit on TRAIN only
    tr_df, va_df = apply_scaler(tr_df, feat_cols, mean, std), apply_scaler(va_df, feat_cols, mean, std)

    s = cfg["sensor"]
    Xtr, ytr, _ = make_windows(tr_df, feat_cols, s["window"], s["stride"])
    Xva, yva, _ = make_windows(va_df, feat_cols, s["window"], s["stride"])
    print(f"features={len(feat_cols)} train windows={len(Xtr)} val windows={len(Xva)}")

    enc = SensorEncoder(len(feat_cols), s["hidden"], s["layers"], s["dropout"], s["embed_dim"])
    model = nn.Sequential(enc, nn.Linear(s["embed_dim"], 1))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=cfg["train"]["batch_size"], shuffle=True)
    xva, yva_t = torch.tensor(Xva), torch.tensor(yva)

    for ep in range(cfg["train"]["epochs"]):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            loss = nn.functional.smooth_l1_loss(model(xb).squeeze(-1) / 125, yb / 125)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(xva).squeeze(-1).numpy()
        m = rul_metrics(yva, pred)
        print(f"epoch {ep + 1:02d}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R2={m['r2']:.3f}")


if __name__ == "__main__":
    main()
