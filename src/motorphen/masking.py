"""Training masks: curriculum from random joint dropout to structured patterns."""

import itertools

import torch

from motorphen.c3d_io import JOINTS

LIMBS = [
    ["l_shoulder", "l_elbow"],
    ["r_shoulder", "r_elbow"],
    ["l_hip", "l_knee", "l_ankle"],
    ["r_hip", "r_knee", "r_ankle"],
    ["neck", "pelvis"],
]


def _pattern_table(joints=JOINTS):
    J = len(joints)
    rows = {"limb": [], "single": [], "pair": []}
    for limb in LIMBS:
        r = torch.zeros(J, dtype=torch.bool)
        r[[joints.index(j) for j in limb]] = True
        rows["limb"].append(r)
    for j in range(J):
        r = torch.zeros(J, dtype=torch.bool)
        r[j] = True
        rows["single"].append(r)
    for a, b in itertools.combinations(range(J), 2):
        r = torch.zeros(J, dtype=torch.bool)
        r[[a, b]] = True
        rows["pair"].append(r)
    return [torch.stack(rows[k]) for k in ("limb", "single", "pair")]


PATTERNS = _pattern_table()


def structured_prob(epoch):
    """Share of structured masks by epoch (read from Fig. 3 of the paper)."""
    if epoch < 5:
        return 0.0
    if epoch < 60:
        return 0.9 * (epoch - 5) / 55
    return 0.9


def _span(B, T, gen):
    u = torch.rand(B, generator=gen)
    length = torch.randint(2, min(6, T - 1) + 1, (B,), generator=gen)
    length = torch.where(u < 0.25, torch.ones_like(length), length)
    length = torch.where(u >= 0.75, torch.full_like(length, T), length)
    start = (torch.rand(B, generator=gen) * (T - length + 1)).floor().long()
    t = torch.arange(T)[None, :]
    return (t >= start[:, None]) & (t < (start + length)[:, None]), start


def sample_masks(B, J, T, epoch, gen=None, p_random=0.5):
    """Boolean masks (B, J, T); True = hidden."""
    # random joint dropout, independent per frame
    rand = torch.rand(B, T, J, generator=gen) < p_random
    # structured: category then pattern, independent per frame
    cat = torch.randint(0, 3, (B, T), generator=gen)
    struct = torch.zeros(B, T, J, dtype=torch.bool)
    for c, table in enumerate(PATTERNS):
        idx = torch.randint(0, table.shape[0], (B, T), generator=gen)
        struct = torch.where((cat == c)[..., None], table[idx], struct)
    use_struct = torch.rand(B, generator=gen) < structured_prob(epoch)
    frames = torch.where(use_struct[:, None, None], struct, rand)
    # temporal coherence: frames inside the span reuse the span's first-frame mask
    inspan, start = _span(B, T, gen)
    base = frames[torch.arange(B), start]
    frames = torch.where(inspan[..., None], base[:, None, :], frames)
    m = frames.permute(0, 2, 1).contiguous()          # (B, J, T)
    # never all hidden, never nothing hidden
    full = m.flatten(1).all(1)
    m[full, :, -1] = False
    empty = ~m.flatten(1).any(1)
    if empty.any():
        j = torch.randint(0, J, (int(empty.sum()),), generator=gen)
        m[empty.nonzero(as_tuple=True)[0], j, -1] = True
    return m


def joint_mask(B, J, T, joints):
    """Hide the given joint indices at every frame (inference tiles and Pass 2)."""
    m = torch.zeros(B, J, T, dtype=torch.bool)
    m[:, joints, :] = True
    return m
