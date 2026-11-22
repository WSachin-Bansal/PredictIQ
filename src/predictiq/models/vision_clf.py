"""Good-vs-defect image classifier: pretrained ResNet18 encoder + linear head."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import VisionEncoder


class VisionClassifier(nn.Module):
    """(B, 3, H, W) -> (B, 2) logits [good, defect]. `encoder` is the same VisionEncoder the
    fused model will reuse later, so its embedding can be plugged into the fusion layer."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 embed_dim: int = 128, finetune_last_block: bool = True):
        super().__init__()
        self.encoder = VisionEncoder(backbone, pretrained, embed_dim, freeze_backbone=True)
        self.head = nn.Linear(embed_dim, 2)
        if finetune_last_block and backbone == "resnet18":
            for p in self.encoder.backbone.layer4.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
