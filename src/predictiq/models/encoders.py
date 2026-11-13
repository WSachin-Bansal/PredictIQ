"""Modality encoders. Each maps its input to a fixed-size embedding (B, embed_dim)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class SensorEncoder(nn.Module):
    """2-layer LSTM -> last hidden state -> dense embedding. (B, T, F) -> (B, D)."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2, embed_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(nn.Linear(hidden, embed_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.proj(self.drop(out[:, -1]))


class VisionEncoder(nn.Module):
    """Pretrained CNN backbone with a new embedding head. (B, 3, H, W) -> (B, D)."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 embed_dim: int = 128, freeze_backbone: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        if backbone == "resnet18":
            net = torchvision.models.resnet18(weights=weights)
            feat = net.fc.in_features
            net.fc = nn.Identity()
        elif backbone == "mobilenet_v3_small":
            net = torchvision.models.mobilenet_v3_small(weights=weights)
            feat = net.classifier[0].in_features
            net.classifier = nn.Identity()
        else:
            raise ValueError(f"unknown backbone {backbone}")
        self.backbone = net
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Sequential(nn.Linear(feat, embed_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class AudioEncoder(nn.Module):
    """Small 2D CNN over log-Mel spectrograms. (B, 1, M, T) -> (B, D)."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(64, embed_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.features(x)).flatten(1))


class SensorCNN1D(nn.Module):
    """1D CNN for raw multichannel vibration. (B, C, T) -> (B, D).

    Designed for ~10k samples per window (e.g. 1 s at 10 kHz); the four pooling stages
    shrink time by 128x before global average pooling, so any T >= ~256 works.
    """

    def __init__(self, in_channels: int = 6, embed_dim: int = 128):
        super().__init__()

        def block(i, o, k, pool):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o),
                                 nn.ReLU(), nn.MaxPool1d(pool))

        self.features = nn.Sequential(
            block(in_channels, 32, 15, 4), block(32, 64, 9, 4),
            block(64, 128, 5, 4), block(128, 128, 3, 2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Linear(128, embed_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.features(x)).flatten(1))
