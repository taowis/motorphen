"""Losses, data windows and optimiser setup for E1."""

import numpy as np
import torch

from motorphen.model import token_features


def l1(a, b, w=None):
    d = (a - b).abs()
    if w is None:
        return d.mean()
    w = w[..., None].expand_as(d).float()
    return (d * w).sum() / w.sum().clamp_min(1.0)


def compute_losses(model, angles, mask1, mask2):
    """The five equally weighted losses of the paper, on the sin-cos features.

    angles (B, J, T, 3) radians; mask1, mask2 (B, J, T) two different masks."""
    target = token_features(angles)[..., :6]
    pred = model(angles, mask1)[..., :6]
    pred2 = model(angles, mask2)[..., :6]
    parts = {
        "final": l1(pred[:, :, -1], target[:, :, -1]),
        "masked": l1(pred, target, mask1),
        "visible": l1(pred, target, ~mask1),
        "velocity": l1(pred[:, :, 1:] - pred[:, :, :-1], target[:, :, 1:] - target[:, :, :-1]),
        "context": l1(pred[:, :, -1], pred2[:, :, -1]),
    }
    return sum(parts.values()), parts


def window_starts(starts, lengths, T=7):
    """Absolute frame index of every window start, for segments given by start/length."""
    out = [np.arange(s, s + n - T + 1) for s, n in zip(starts, lengths) if n >= T]
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def gather_windows(all_angles, idx, T=7):
    """all_angles (N, J, 3) tensor in radians; idx (B,) window starts -> (B, J, T, 3)."""
    frames = idx[:, None] + torch.arange(T, device=idx.device)[None, :]
    return all_angles[frames].permute(0, 2, 1, 3)


def make_optimizer(model, lr=2e-4, weight_decay=0.05):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim < 2 or "norm" in name or name.endswith(".bias") or "mask" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95))
