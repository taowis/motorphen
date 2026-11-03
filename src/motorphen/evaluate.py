"""Gait-cycle normalisation, RMSE to the normative mean and equivalence testing
(paper section 3.2)."""

import numpy as np

from motorphen.c3d_io import JOINTS

# angles reported in the paper: (name, joint, axis); axis 0 = flexion, 1 = ab/adduction
REPORTED = [
    ("pelvis_flex", "pelvis", 0),
    ("r_hip_abd", "r_hip", 1),
    ("r_hip_flex", "r_hip", 0),
    ("r_knee_flex", "r_knee", 0),
]


def cycles(series_deg, strikes, n=100):
    """Cut a 1-D angle series at successive foot strikes and resample each
    cycle to n points (unwrapped before interpolation). Returns (cycles, n)."""
    strikes = sorted(strikes)
    out = []
    for a, b in zip(strikes[:-1], strikes[1:]):
        if b - a < 10 or b > len(series_deg):
            continue
        seg = np.rad2deg(np.unwrap(np.deg2rad(series_deg[a:b + 1])))
        x = np.linspace(0, 1, len(seg))
        out.append(np.interp(np.linspace(0, 1, n), x, seg))
    return np.array(out).reshape(-1, n)


def reported_cycles(angles_deg, strikes, n=100):
    """dict name -> (cycles, n) for the four reported angles of one trial."""
    return {name: cycles(angles_deg[:, JOINTS.index(j), ax], strikes, n)
            for name, j, ax in REPORTED}


def rmse(traj, mu):
    d = (np.asarray(traj) - np.asarray(mu) + 180) % 360 - 180
    return float(np.sqrt(np.mean(d ** 2)))


def bootstrap_equivalence(diff, delta=1.5, n_boot=20000, ci=0.90, seed=0):
    """Mean paired difference, its bootstrap CI and whether it lies in [-delta, delta]."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return dict(mean=float(diff.mean()), ci_low=float(lo), ci_high=float(hi),
                equivalent=bool(lo > -delta and hi < delta), n=len(diff))
