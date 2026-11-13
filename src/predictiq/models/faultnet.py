"""Sensor / audio / fusion classifier for MAFAULDA (fault class + severity bucket)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import AudioEncoder, SensorCNN1D


class FaultNet(nn.Module):
    """Encoders -> concat -> MLP -> (fault, severity) heads.

    modalities: any non-empty subset of ("sensor", "audio").
    Inputs: {"sensor": (B, C, T) vibration, "audio": (B, 1, n_mels, frames) log-Mel}.
    A modality passed as None (or absent from the dict) is replaced by a zero embedding,
    which is how the missing-modality evaluation works. During training, modality
    dropout randomly zeroes whole modalities (never all of them for a sample).
    """

    def __init__(self, modalities=("sensor", "audio"), n_classes: int = 6,
                 n_severity: int = 3, n_vib_channels: int = 6, embed_dim: int = 128,
                 hidden: int = 256, dropout: float = 0.3, modality_dropout: float = 0.2):
        super().__init__()
        self.modalities = tuple(modalities)
        self.embed_dim = embed_dim
        self.modality_dropout = modality_dropout
        self.enc = nn.ModuleDict()
        if "sensor" in self.modalities:
            self.enc["sensor"] = SensorCNN1D(n_vib_channels, embed_dim)
        if "audio" in self.modalities:
            self.enc["audio"] = AudioEncoder(embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * len(self.modalities), hidden), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fault_head = nn.Linear(hidden, n_classes)
        self.severity_head = nn.Linear(hidden, n_severity)

    def forward(self, batch: dict) -> dict:
        present = [m for m in self.modalities if batch.get(m) is not None]
        if not present:
            raise ValueError("at least one modality must be provided")
        ref = batch[present[0]]
        embs = []
        for m in self.modalities:
            x = batch.get(m)
            if x is None:
                embs.append(torch.zeros(ref.shape[0], self.embed_dim, device=ref.device,
                                        dtype=ref.dtype))
            else:
                embs.append(self.enc[m](x))

        n = len(embs)
        if self.training and n > 1 and self.modality_dropout > 0:
            keep = (torch.rand(ref.shape[0], n, device=ref.device)
                    > self.modality_dropout).float()
            empty = (keep.sum(1) == 0).nonzero(as_tuple=True)[0]
            if len(empty):  # never drop every modality of a sample
                keep[empty, torch.randint(0, n, (len(empty),), device=ref.device)] = 1.0
            embs = [e * keep[:, i:i + 1] for i, e in enumerate(embs)]

        h = self.fusion(torch.cat(embs, dim=-1))
        return {"fault_logits": self.fault_head(h), "severity_logits": self.severity_head(h)}
