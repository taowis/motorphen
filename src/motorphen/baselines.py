"""Simple side-scoring methods to compare against the masked autoencoder (E2)."""

import numpy as np
from sklearn.decomposition import PCA

from motorphen.c3d_io import JOINTS
from motorphen.evaluate import cycles

# per side: (joint, axis) of hip flexion, hip ab/adduction, knee flexion
SIDE_ANGLES = {
    "left": [("l_hip", 0), ("l_hip", 1), ("l_knee", 0)],
    "right": [("r_hip", 0), ("r_hip", 1), ("r_knee", 0)],
}
SIDE_JOINTS = {"left": ["l_hip", "l_knee"], "right": ["r_hip", "r_knee"]}


def side_cycles(trials, side, n=100):
    """trials: list of (angles_deg (N, J, 3), strikes for that side).
    Returns (n_angles, n_cycles, n) cycles pooled over trials."""
    out = []
    for joint, ax in SIDE_ANGLES[side]:
        cs = [cycles(a[:, JOINTS.index(joint), ax], st, n) for a, st in trials]
        out.append(np.concatenate(cs) if cs else np.empty((0, n)))
    return out


def normative_reference(trials_by_side, n=100):
    """Mean and SD cycle per side angle from healthy trials (both sides pooled
    by mirroring the side label). Returns dict side -> list of (mu, sd)."""
    ref = {}
    for side in ("left", "right"):
        cs = side_cycles(trials_by_side[side], side, n)
        ref[side] = [(c.mean(0), c.std(0)) for c in cs]
    return ref


def zscore_side(subject_trials, side, ref):
    cs = side_cycles(subject_trials, side)
    vals = []
    for c, (mu, sd) in zip(cs, ref[side]):
        if len(c):
            vals.append(np.mean(np.abs((c.mean(0) - mu) / np.maximum(sd, 1e-3))))
    return float(np.mean(vals)) if vals else np.nan


def rmse_side(subject_trials, side, ref):
    cs = side_cycles(subject_trials, side)
    vals = [np.sqrt(np.mean((c.mean(0) - mu) ** 2)) for c, (mu, _) in zip(cs, ref[side]) if len(c)]
    return float(np.mean(vals)) if vals else np.nan


def peak_knee_flexion(subject_trials, side):
    """Mean over cycles of the peak knee flexion (degrees)."""
    knee = side_cycles(subject_trials, side)[2]
    return float(knee.max(axis=1).mean()) if len(knee) else np.nan


class PCABaseline:
    """PCA on per-frame joint angle vectors of healthy subjects; per-joint
    reconstruction error as the deviation measure."""

    def __init__(self, var=0.95):
        self.var = var

    def fit(self, angles_deg):
        x = self._flat(angles_deg)
        self.mean_, self.sd_ = x.mean(0), x.std(0) + 1e-6
        self.pca = PCA(n_components=self.var, svd_solver="full").fit((x - self.mean_) / self.sd_)
        return self

    @staticmethod
    def _flat(a):
        a = np.asarray(a, dtype=float)
        return np.concatenate([np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a))], axis=-1).reshape(len(a), -1)

    def joint_error(self, angles_deg):
        """(N, J, 3) -> (J,) mean squared reconstruction error per joint."""
        x = (self._flat(angles_deg) - self.mean_) / self.sd_
        r = self.pca.inverse_transform(self.pca.transform(x))
        err = ((x - r) ** 2).reshape(len(x), len(JOINTS), 6).mean(axis=(0, 2))
        return err

    def side_score(self, subject_angles, side):
        e = np.mean([self.joint_error(a) for a in subject_angles], axis=0)
        return float(np.mean([e[JOINTS.index(j)] for j in SIDE_JOINTS[side]]))
