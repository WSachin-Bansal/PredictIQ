"""Small pure-python/numpy helpers for train_combined.py (kept torch-free so they can be tested)."""
from __future__ import annotations

import math


def source_schedule(total_steps: int, batches_per_pass: int, passes: float) -> list[bool]:
    """Which of `total_steps` steps should also draw a batch from a smaller dataset, so that it
    is traversed about `passes` times (one batch at most per step, spread evenly)."""
    need = passes * batches_per_pass
    r = min(1.0, need / max(total_steps, 1))
    return [math.floor((i + 1) * r) > math.floor(i * r) for i in range(total_steps)]


def composite_score(fault_f1: float, defect_f1: float, rul_rmse: float, rmse_ref: float = 50.0) -> float:
    """One number for picking the best epoch: fault macro-F1 + defect macro-F1 + (1 - RMSE/50),
    each term in [0, 1]. Higher is better."""
    return fault_f1 + defect_f1 + (1.0 - min(rul_rmse / rmse_ref, 1.0))


def remap_state(state: dict, rules: list[tuple[str, str]]) -> dict:
    """Keep only keys that start with an old prefix and rename them to the new prefix."""
    out = {}
    for k, v in state.items():
        for old, new in rules:
            if k.startswith(old):
                out[new + k[len(old):]] = v
                break
    return out
