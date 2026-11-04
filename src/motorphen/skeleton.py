"""Stick figures from Plug-in Gait segment frames, with joint angles swapped in.

Plug-in Gait writes each segment's origin and three axis points (O, A, L, P) to
the C3D file. The segment frames give the original pose. A reconstructed pose
is built by forward kinematics: each modelled joint's relative rotation is
replaced by the one implied by its reconstructed Cardan angles, while segment
lengths, joint centres and unmodelled joints (spine, ankles) keep their
original values.

The angle conventions below were fitted to the segment frames of the Van
Criekinge 2023 files and reproduce the relative rotations exactly (pelvis,
hips, knees, elbows) or up to a fixed offset (neck, shoulders), which the
update rule in `relative` allows for.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from motorphen.c3d_io import JOINTS

# joint: (Euler order, angle permutation, signs, side of the fixed offset)
# side "R": relative = C @ g(angles); side "L": relative = g(angles) @ C
CONVENTION = {
    "pelvis": ("ZXY", (2, 1, 0), (1, -1, -1), "R"),    # transposed, see g()
    "l_hip": ("YXZ", (0, 1, 2), (-1, -1, -1), "R"),
    "r_hip": ("YXZ", (0, 1, 2), (-1, 1, 1), "R"),
    "l_knee": ("YXZ", (0, 1, 2), (1, -1, -1), "R"),
    "r_knee": ("YXZ", (0, 1, 2), (1, 1, 1), "R"),
    "l_elbow": ("YXZ", (0, 1, 2), (1, 1, 1), "R"),
    "r_elbow": ("YXZ", (0, 1, 2), (1, 1, 1), "R"),
    "neck": ("ZXY", (2, 1, 0), (1, -1, 1), "R"),
    "l_shoulder": ("XYZ", (1, 0, 2), (1, 1, 1), "L"),
    "r_shoulder": ("XYZ", (1, 0, 2), (1, 1, 1), "R"),
}

# child segment: (parent segment, joint whose angles rotate it, or None)
TREE = {
    "TRX": ("PEL", None),
    "HED": ("TRX", "neck"),
    "LHU": ("TRX", "l_shoulder"), "RHU": ("TRX", "r_shoulder"),
    "LRA": ("LHU", "l_elbow"), "RRA": ("RHU", "r_elbow"),
    "LHN": ("LRA", None), "RHN": ("RRA", None),
    "LFE": ("PEL", "l_hip"), "RFE": ("PEL", "r_hip"),
    "LTI": ("LFE", "l_knee"), "RTI": ("RFE", "r_knee"),
    "LFO": ("LTI", None), "RFO": ("RTI", None),
}
SEGMENTS = ["PEL"] + list(TREE)

# drawn points: joint centre between a parent and child, or a segment origin
POINTS = {
    "neck": ("TRX", "HED"), "head": "HEDO",
    "l_sho": ("TRX", "LHU"), "r_sho": ("TRX", "RHU"),
    "l_elb": ("LHU", "LRA"), "r_elb": ("RHU", "RRA"),
    "l_wri": ("LRA", "LHN"), "r_wri": ("RRA", "RHN"),
    "l_hip": ("PEL", "LFE"), "r_hip": ("PEL", "RFE"),
    "l_kne": ("LFE", "LTI"), "r_kne": ("RFE", "RTI"),
    "l_ank": ("LTI", "LFO"), "r_ank": ("RTI", "RFO"),
    "l_toe": "LFOO", "r_toe": "RFOO",
}
EDGES = [("head", "neck"), ("neck", "l_sho"), ("neck", "r_sho"), ("l_sho", "l_elb"),
         ("l_elb", "l_wri"), ("r_sho", "r_elb"), ("r_elb", "r_wri"), ("neck", "mid_hip"),
         ("l_hip", "r_hip"), ("l_hip", "l_kne"), ("l_kne", "l_ank"), ("l_ank", "l_toe"),
         ("r_hip", "r_kne"), ("r_kne", "r_ank"), ("r_ank", "r_toe")]
# edges drawn in red when a joint is flagged
FLAG_EDGES = {
    "neck": [("head", "neck")], "pelvis": [("l_hip", "r_hip"), ("neck", "mid_hip")],
    "l_hip": [("l_hip", "l_kne")], "r_hip": [("r_hip", "r_kne")],
    "l_knee": [("l_kne", "l_ank")], "r_knee": [("r_kne", "r_ank")],
}


def g(joint, angles):
    """Rotation implied by Cardan angles (..., 3) in radians for one joint."""
    order, perm, signs, _ = CONVENTION[joint]
    a = np.asarray(angles)[..., list(perm)] * np.asarray(signs)
    R = Rotation.from_euler(order, a.reshape(-1, 3)).as_matrix().reshape(a.shape[:-1] + (3, 3))
    return np.swapaxes(R, -1, -2) if joint == "pelvis" else R


def relative(joint, rel_orig, ang_orig, ang_rec):
    """Swap the reconstructed angles into an original relative rotation."""
    side = CONVENTION[joint][3]
    go, gr = g(joint, ang_orig), g(joint, ang_rec)
    if side == "R":
        return rel_orig @ np.swapaxes(go, -1, -2) @ gr
    return gr @ np.swapaxes(go, -1, -2) @ rel_orig


def segment_frames(points, labels):
    """Rotation (n, 3, 3) and origin (n, 3) of every segment from O, A, P points.

    points: (3, n_labels, n_frames) as read by ezc3d."""
    def pt(name):
        return points[:3, labels.index(name), :].T.astype(np.float64)

    out = {}
    for s in SEGMENTS:
        o, a, p = pt(s + "O"), pt(s + "A"), pt(s + "P")
        x = a - o
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        z = p - o
        z -= (z * x).sum(1, keepdims=True) * x
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        out[s] = (np.stack([x, np.cross(z, x), z], -1), o)
    return out


def joint_centre(frames, parent, child):
    """Point fixed in both segments, by least squares over frames.
    Returns its coordinates in the parent and the child frame."""
    Rp, Op = frames[parent]
    Rc, Oc = frames[child]
    ok = np.isfinite(Rp).all((1, 2)) & np.isfinite(Rc).all((1, 2))
    A = np.concatenate([Rp[ok], -Rc[ok]], axis=2).reshape(-1, 6)
    b = (Oc[ok] - Op[ok]).reshape(-1)
    # small ridge keeps hinge joints (knee, elbow) off their free axis
    lam = 1e-3 * len(b)
    x = np.linalg.solve(A.T @ A + lam * np.eye(6), A.T @ b)
    return x[:3], x[3:]


def fit(frames):
    """Joint centres for every parent-child pair, fitted on one trial."""
    return {c: joint_centre(frames, p, c) for c, (p, _) in TREE.items()}


def pose(frames, centres, t, angles_orig=None, angles_rec=None):
    """Global rotation and origin of every segment at frame t.

    With angles_orig and angles_rec ((12, 3) radians, JOINTS order), the
    modelled joints take their reconstructed rotations."""
    swap = angles_orig is not None
    R = {"PEL": frames["PEL"][0][t]}
    O = {"PEL": frames["PEL"][1][t]}
    if swap:
        j = JOINTS.index("pelvis")
        R["PEL"] = relative("pelvis", R["PEL"][None], angles_orig[j][None], angles_rec[j][None])[0]
    for c, (p, joint) in TREE.items():
        rel = frames[p][0][t].T @ frames[c][0][t]
        if swap and joint is not None:
            j = JOINTS.index(joint)
            rel = relative(joint, rel[None], angles_orig[j][None], angles_rec[j][None])[0]
        R[c] = R[p] @ rel
        a, b = centres[c]
        O[c] = O[p] + R[p] @ a - R[c] @ b
    return R, O


def points(R, O, centres):
    """Drawn points (name -> (3,)) for one pose."""
    out = {}
    for name, src in POINTS.items():
        if isinstance(src, tuple):
            p, c = src
            out[name] = O[p] + R[p] @ centres[c][0]
        else:
            out[name] = O[src[:-1]]
    out["mid_hip"] = (out["l_hip"] + out["r_hip"]) / 2
    return out
