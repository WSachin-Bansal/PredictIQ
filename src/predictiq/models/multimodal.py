"""Fusion network and multi-task model (Eq. 1 and Eq. 2 of the proposal)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import AudioEncoder, SensorEncoder, VisionEncoder

MODALITIES = ("sensor", "vision", "audio")


class PredictIQ(nn.Module):
    """Sensor / vision / audio encoders -> fusion -> (fault, failure prob, RUL) heads.

    Any subset of modalities can be enabled (configs M1..M7). A missing modality at
    inference is replaced by a zero embedding; modality dropout during training teaches
    the network to cope with that.
    """

    def __init__(self, cfg: dict, modalities=MODALITIES):
        super().__init__()
        self.modalities = tuple(modalities)
        s, v, a, f = cfg["sensor"], cfg["vision"], cfg["audio"], cfg["fusion"]
        self.embed_dim = s["embed_dim"]
        self.enc = nn.ModuleDict()
        if "sensor" in self.modalities:
            self.enc["sensor"] = SensorEncoder(s["n_features"], s["hidden"], s["layers"],
                                               s["dropout"], s["embed_dim"])
        if "vision" in self.modalities:
            self.enc["vision"] = VisionEncoder(v["backbone"], v["pretrained"],
                                               v["embed_dim"], v["freeze_backbone"])
        if "audio" in self.modalities:
            self.enc["audio"] = AudioEncoder(a["embed_dim"])

        self.mode = f["mode"]
        self.modality_dropout = f["modality_dropout"]
        n = len(self.modalities)
        if self.mode == "weighted":
            self.w = nn.Parameter(torch.zeros(n))
        elif self.mode == "attention":
            self.attn = nn.Linear(self.embed_dim, 1)
        fused_dim = self.embed_dim * (n if self.mode == "concat" else 1)

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, f["hidden"]), nn.ReLU(), nn.Dropout(f["dropout"]),
            nn.Linear(f["hidden"], f["hidden"] // 2), nn.ReLU(),
        )
        h = f["hidden"] // 2
        self.fault_head = nn.Linear(h, f["n_classes"])
        self.failure_head = nn.Linear(h, 1)
        self.rul_head = nn.Linear(h, 1)

    def forward(self, batch: dict) -> dict:
        embs = []
        for m in self.modalities:
            x = batch.get(m)
            if x is None:  # missing modality at inference
                ref = next(v for v in batch.values() if v is not None)
                e = torch.zeros(ref.shape[0], self.embed_dim, device=ref.device)
            else:
                e = self.enc[m](x)
                if self.training and self.modality_dropout > 0:
                    keep = (torch.rand(e.shape[0], 1, device=e.device)
                            > self.modality_dropout).float()
                    e = e * keep
            embs.append(e)

        if self.mode == "concat":
            fused = torch.cat(embs, dim=-1)  # F_multi = [Fs; Fv; Fa]
        elif self.mode == "weighted":
            w = torch.softmax(self.w, 0)
            fused = sum(wi * e for wi, e in zip(w, embs))
        else:  # attention
            stack = torch.stack(embs, dim=1)                 # (B, n, D)
            a = torch.softmax(self.attn(stack), dim=1)       # (B, n, 1)
            fused = (a * stack).sum(dim=1)

        h = self.fusion(fused)
        return {
            "fault_logits": self.fault_head(h),
            "failure_logit": self.failure_head(h).squeeze(-1),
            "rul": self.rul_head(h).squeeze(-1),
        }
