"""Read Plug-in Gait joint angles and gait events from C3D files."""

from dataclasses import dataclass, field

import numpy as np

# The 12 joints used by Motta et al. Plug-in Gait gives mirrored left/right
# copies of the neck and pelvis angles; the left copy is used for both.
JOINTS = [
    "neck", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "pelvis",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]
LOWER_JOINTS = ["pelvis", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
PIG_LABELS = {
    "neck": "LNeckAngles",
    "l_shoulder": "LShoulderAngles",
    "r_shoulder": "RShoulderAngles",
    "l_elbow": "LElbowAngles",
    "r_elbow": "RElbowAngles",
    "pelvis": "LPelvisAngles",
    "l_hip": "LHipAngles",
    "r_hip": "RHipAngles",
    "l_knee": "LKneeAngles",
    "r_knee": "RKneeAngles",
    "l_ankle": "LAnkleAngles",
    "r_ankle": "RAnkleAngles",
}


@dataclass
class Trial:
    path: str
    rate: float
    first_frame: int            # absolute frame number of column 0
    angles: np.ndarray          # (frames, 12, 3) degrees, NaN where missing
    events: dict = field(default_factory=dict)  # e.g. {"Left Foot Strike": [frames]}

    @property
    def n_frames(self):
        return self.angles.shape[0]


def _labels(point_params):
    labels = list(point_params["LABELS"]["value"])
    i = 2
    while f"LABELS{i}" in point_params:
        labels += list(point_params[f"LABELS{i}"]["value"])
        i += 1
    # some files prefix every label with the subject name ("NAME:LHipAngles")
    return [lab.split(":", 1)[1] if ":" in lab else lab for lab in labels]


def read_trial(path):
    """Return a Trial, or None if the file has no gait events or no lower-limb angles."""
    import ezc3d  # only needed for reading C3D, not for training

    c = ezc3d.c3d(path)
    params = c["parameters"]
    labels = _labels(params["POINT"])
    if any(PIG_LABELS[j] not in labels for j in LOWER_JOINTS):
        return None
    ev = params.get("EVENT", {})
    if "LABELS" not in ev or "TIMES" not in ev or not ev["LABELS"]["value"]:
        return None

    rate = float(params["POINT"]["RATE"]["value"][0])
    first = int(params["TRIAL"]["ACTUAL_START_FIELD"]["value"][0])
    pts = c["data"]["points"]                      # (4, labels, frames)
    # joints without an output in this file (lower-body-only processing) stay NaN
    angles = np.full((pts.shape[2], len(JOINTS), 3), np.nan)
    for k, j in enumerate(JOINTS):
        if PIG_LABELS[j] in labels:
            angles[:, k, :] = pts[:3, labels.index(PIG_LABELS[j]), :].T
    # Plug-in Gait writes exact zeros on all three axes when a value is missing
    missing = np.all(angles == 0.0, axis=2)
    angles[missing] = np.nan

    events = {}
    times = np.asarray(ev["TIMES"]["value"])[1]
    for label, context, t in zip(ev["LABELS"]["value"], ev["CONTEXTS"]["value"], times):
        key = f"{context} {label}"
        frame = int(round(t * rate)) - first   # index into the angle array
        events.setdefault(key, []).append(frame)
    for k in events:
        events[k] = sorted(events[k])
    return Trial(path=path, rate=rate, first_frame=first, angles=angles, events=events)
