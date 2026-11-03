"""Two-pass inference: joint scoring by tiled occlusion (Pass 1) and masked
reconstruction of the flagged joints (Pass 2). See docs/e1-implementation-spec.md."""

import numpy as np
import torch

from motorphen.c3d_io import JOINTS
from motorphen.model import angles_from_pred, rot_yxz, wrap

SCORED = ["neck", "pelvis", "l_hip", "r_hip", "l_knee", "r_knee"]
SCORED_IDX = [JOINTS.index(j) for j in SCORED]
AXIS_WEIGHTS = torch.tensor([0.6, 0.25, 0.15])   # sagittal, frontal, transverse
LONG_AXIS = torch.tensor([0.0, 0.0, 1.0])         # child segment long axis in the joint frame


@torch.no_grad()
def predict_last(model, seq, hide=(), T=7, batch=1024):
    """Last-frame predictions for every window of a trial.

    seq: (N, J, 3) tensor of angles in radians on the model's device.
    hide: joint indices hidden from the encoder at every frame.
    Returns (N - T + 1, J, 3) predicted angles."""
    N, J, _ = seq.shape
    W = N - T + 1
    out = []
    for b in range(0, W, batch):
        starts = torch.arange(b, min(W, b + batch), device=seq.device)
        frames = starts[:, None] + torch.arange(T, device=seq.device)[None, :]
        a = seq[frames].permute(0, 2, 1, 3)               # (B, J, T, 3)
        m = torch.zeros(a.shape[:3], dtype=torch.bool, device=seq.device)
        if len(hide):
            m[:, list(hide), :] = True
        out.append(angles_from_pred(model(a, m))[:, :, -1])
    return torch.cat(out)


def badness(base_j, tile_j, rom_j):
    """Per-frame badness B_j(t) (paper eqs. 1-5) for one joint.

    base_j, tile_j: (W, 3) predicted angles; rom_j: (3,) range of motion, radians."""
    d = wrap(tile_j - base_j).abs()
    c_rom = (AXIS_WEIGHTS.to(d) * (d / rom_j.to(d)).clamp(0, 1)).sum(-1)
    v_base = rot_yxz(base_j) @ LONG_AXIS.to(d)
    v_tile = rot_yxz(tile_j) @ LONG_AXIS.to(d)
    cos = torch.nn.functional.cosine_similarity(v_base, v_tile, dim=-1)
    e_geom = (1 - cos) / 2
    return e_geom * (0.5 + 0.5 * c_rom)


def pass1(model, seq, rom, peak_q=0.95):
    """Trial-level scores for the scored joints. rom: (J, 3) tensor, radians.

    Returns dict joint index -> score, and the baseline predictions."""
    base = predict_last(model, seq)
    scores = {}
    for j in SCORED_IDX:
        tile = predict_last(model, seq, hide=[j])
        b = badness(base[:, j], tile[:, j], rom[j])
        scores[j] = float(torch.quantile(b.float().cpu(), peak_q))
    return scores, base


def select_joints(scores, floor=None, k=2):
    """Top k joints by score, keeping only those above their noise floor (if given)."""
    ranked = sorted(scores, key=scores.get, reverse=True)
    if floor is not None:
        ranked = [j for j in ranked if scores[j] > floor[j]]
    return ranked[:k]


def reconstruct(model, seq, flagged, T=7):
    """Pass 2: hide the flagged joints, keep the reconstructed last frame of each
    window for all joints. The first T - 1 frames keep the input."""
    out = seq.clone()
    if len(flagged) == 0:
        return out
    out[T - 1:] = predict_last(model, seq, hide=flagged, T=T)
    return out


def range_of_motion(angles_rad, lo=1, hi=99):
    """(N, J, 3) -> (J, 3) percentile range, used as ROM_i."""
    a = np.asarray(angles_rad)
    r = np.nanpercentile(a, hi, axis=0) - np.nanpercentile(a, lo, axis=0)
    return np.maximum(r, np.deg2rad(1.0))
