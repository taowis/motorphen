"""Gap filling, resampling, trimming and features."""

import numpy as np
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation


def fill_gaps(x, max_gap):
    """Linearly interpolate NaN runs of at most max_gap frames along axis 0.

    Longer runs stay NaN. x has shape (frames, ...)."""
    x = x.copy()
    flat = x.reshape(x.shape[0], -1)
    t = np.arange(flat.shape[0])
    for k in range(flat.shape[1]):
        col = flat[:, k]
        bad = np.isnan(col)
        if not bad.any() or bad.all():
            continue
        # find runs of NaN
        edges = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        good = ~bad
        for s, e in zip(starts, ends):
            if e - s <= max_gap and s > 0 and e < len(col):
                col[s:e] = np.interp(t[s:e], t[good], col[good])
    return flat.reshape(x.shape)


def walking_span(events, n_frames):
    """First to last foot strike (either side), clipped to the data."""
    strikes = events.get("Left Foot Strike", []) + events.get("Right Foot Strike", [])
    strikes = [f for f in strikes if 0 <= f < n_frames]
    if len(strikes) < 2:
        return None
    return min(strikes), max(strikes) + 1


def resample(x, rate_in, rate_out):
    """Polyphase resampling along axis 0. Input must have no NaN."""
    from fractions import Fraction
    fr = Fraction(rate_out / rate_in).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator, axis=0, padtype="line")


def contiguous_segments(x, min_len):
    """Split (frames, ...) into runs of frames with no NaN, each >= min_len."""
    ok = ~np.isnan(x.reshape(x.shape[0], -1)).any(axis=1)
    edges = np.diff(np.concatenate([[0], ok.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]


def features(angles_deg):
    """Per-joint 12-d features from (..., 3) Cardan angles in degrees.

    sin and cos of the three angles (6) plus the 6D rotation representation
    (first two columns of the rotation matrix, 6). The rotation is built from
    the angles with an intrinsic Y-X-Z sequence (flexion, ab/adduction,
    rotation), the Plug-in Gait convention. Returns (..., 12)."""
    a = np.deg2rad(angles_deg)
    sc = np.concatenate([np.sin(a), np.cos(a)], axis=-1)
    shape = a.shape[:-1]
    flat = a.reshape(-1, 3)
    # angle columns are (flexion about Y, abduction about X, rotation about Z)
    rot = Rotation.from_euler("YXZ", flat[:, [0, 1, 2]]).as_matrix()
    six = rot[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    return np.concatenate([sc, six.reshape(*shape, 6)], axis=-1)


def windows(x, length, step=1):
    """Sliding windows over axis 0: returns (n, length, ...)."""
    n = (x.shape[0] - length) // step + 1
    if n <= 0:
        return np.empty((0, length) + x.shape[1:], dtype=x.dtype)
    idx = np.arange(length)[None, :] + step * np.arange(n)[:, None]
    return x[idx]
