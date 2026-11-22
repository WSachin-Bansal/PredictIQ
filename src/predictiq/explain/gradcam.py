"""Grad-CAM for the ResNet18 vision classifier: which image regions drove the decision."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gradcam(model, x: torch.Tensor, target: int | None = None):
    """x: (1, 3, H, W) normalised image. Returns (heatmap (H, W) in [0, 1], probs (2,), target).

    The heatmap is for `target` (default: the predicted class). Uses the last conv block
    (backbone.layer4): channel weights = mean gradient, map = ReLU(sum of weighted channels).
    """
    model.eval()
    layer = model.encoder.backbone.layer4
    store = {}
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("f", o))
    try:
        x = x.clone().requires_grad_(True)       # keeps the graph alive even if layers are frozen
        logits = model(x)
        probs = logits.softmax(-1)[0].detach()
        t = int(logits.argmax(-1)) if target is None else target
        grads = torch.autograd.grad(logits[0, t], store["f"])[0]
    finally:
        h.remove()
    w = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["f"]).sum(1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy(), probs.cpu().numpy(), t
