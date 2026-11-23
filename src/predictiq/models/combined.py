"""One model for every data source: vibration + sound + image + engine sensors -> shared fusion
layer -> four heads (fault, severity, defect, remaining life).

Every encoder turns its input into a 128-number embedding. The fusion layer reads the four
embeddings side by side; a source that is not present is replaced by zeros. During training
each batch comes from ONE dataset (MAFAULDA, MVTec or C-MAPSS), so each head is trained only
on the data that has its labels (see training/train_combined.py). No dataset contains all four
inputs for the same machine, so the sources never co-occur in training.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import AudioEncoder, SensorCNN1D, SensorEncoder, VisionEncoder

SLOTS = ("vib", "audio", "image", "engine")


class CombinedModel(nn.Module):
    """inputs: dict with any of
         vib    (B, 6, 10000)      standardised vibration window
         audio  (B, 1, 64, 32)     log-Mel spectrogram
         image  (B, 3, H, W)       ImageNet-normalised image
         engine (B, 50, F)         standardised C-MAPSS window
       Missing sources: leave the key out or pass None.
       Returns: fault_logits (B, n_faults), severity_logits (B, n_severity),
                defect_logits (B, 2), rul_norm (B,) = RUL / 125.
    """

    def __init__(self, n_faults: int = 10, n_severity: int = 3, n_vib_channels: int = 6,
                 n_engine_features: int = 17, embed_dim: int = 128, hidden: int = 256,
                 dropout: float = 0.3, modality_dropout: float = 0.2,
                 engine_hidden: int = 64, engine_layers: int = 2, engine_dropout: float = 0.2,
                 vision_backbone: str = "resnet18", pretrained: bool = True,
                 finetune_last_block: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_dropout = modality_dropout
        self.enc = nn.ModuleDict({
            "vib": SensorCNN1D(n_vib_channels, embed_dim),
            "audio": AudioEncoder(embed_dim),
            "image": VisionEncoder(vision_backbone, pretrained, embed_dim, freeze_backbone=True),
            "engine": SensorEncoder(n_engine_features, engine_hidden, engine_layers,
                                    engine_dropout, embed_dim),
        })
        if finetune_last_block and vision_backbone == "resnet18":
            for p in self.enc["image"].backbone.layer4.parameters():
                p.requires_grad = True
        self.fusion = nn.Sequential(nn.Linear(embed_dim * len(SLOTS), hidden), nn.ReLU(),
                                    nn.Dropout(dropout))
        self.fault_head = nn.Linear(hidden, n_faults)
        self.severity_head = nn.Linear(hidden, n_severity)
        self.defect_head = nn.Linear(hidden, 2)
        self.rul_head = nn.Linear(hidden, 1)

    def forward(self, inputs: dict) -> dict:
        present = [s for s in SLOTS if inputs.get(s) is not None]
        if not present:
            raise ValueError("at least one input source must be provided")
        ref = inputs[present[0]]
        embs = [self.enc[s](inputs[s]) if s in present else
                torch.zeros(ref.shape[0], self.embed_dim, device=ref.device, dtype=ref.dtype)
                for s in SLOTS]

        n = len(present)
        if self.training and n > 1 and self.modality_dropout > 0:
            # randomly hide whole sources (never all of them), like FaultNet does
            keep = torch.ones(ref.shape[0], len(SLOTS), device=ref.device)
            idx = [SLOTS.index(s) for s in present]
            k = (torch.rand(ref.shape[0], n, device=ref.device) > self.modality_dropout).float()
            empty = (k.sum(1) == 0).nonzero(as_tuple=True)[0]
            if len(empty):
                k[empty, torch.randint(0, n, (len(empty),), device=ref.device)] = 1.0
            keep[:, idx] = k
            embs = [e * keep[:, i:i + 1] for i, e in enumerate(embs)]

        h = self.fusion(torch.cat(embs, dim=-1))
        return {"fault_logits": self.fault_head(h), "severity_logits": self.severity_head(h),
                "defect_logits": self.defect_head(h), "rul_norm": self.rul_head(h).squeeze(-1)}

    def encoder_params(self) -> dict:
        """Learnable parameter count per block (for the dashboard and the docs)."""
        count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        out = {f"{s} encoder": count(self.enc[s]) for s in SLOTS}
        out["fusion"] = count(self.fusion)
        out["heads"] = sum(count(h) for h in (self.fault_head, self.severity_head,
                                              self.defect_head, self.rul_head))
        return out
