"""Multi-task loss: L_total = l1*L_fault + l2*L_failure + l3*L_RUL."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(self, w_fault: float = 1.0, w_failure: float = 1.0, w_rul: float = 0.5,
                 rul_scale: float = 125.0):
        super().__init__()
        self.w = (w_fault, w_failure, w_rul)
        self.rul_scale = rul_scale  # normalise RUL so its loss is comparable to the others

    def forward(self, out: dict, target: dict) -> dict:
        l_fault = F.cross_entropy(out["fault_logits"], target["fault"])
        l_fail = F.binary_cross_entropy_with_logits(out["failure_logit"], target["failure"])
        l_rul = F.smooth_l1_loss(out["rul"] / self.rul_scale, target["rul"] / self.rul_scale)
        total = self.w[0] * l_fault + self.w[1] * l_fail + self.w[2] * l_rul
        return {"total": total, "fault": l_fault, "failure": l_fail, "rul": l_rul}
