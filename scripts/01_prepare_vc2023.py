"""Extract joint angles from the Van Criekinge 2023 C3D files.

Keeps whole walking trials (files with gait events), fills gaps up to 0.1 s,
splits at longer gaps in the lower-limb angles, resamples 100 Hz to 30 Hz and
writes data/processed/vc2023.npz plus an inventory in results/e1/.

Upper-body angles are missing for subjects whose walking trials were processed
with a lower-body model; they are kept as NaN and flagged per segment
(has_upper), so analyses can use either all 12 joints or the lower limb only.

Run after scripts/00_download.sh and unzipping (see data/README.md).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

from motorphen.c3d_io import JOINTS, LOWER_JOINTS, read_trial
from motorphen.preprocess import contiguous_segments, fill_gaps, resample

ROOT = "data/raw/vancriekinge2023/extracted"
GROUPS = {"healthy": f"{ROOT}/138_HealthyPiG_10.05/SUBJ*", "stroke": f"{ROOT}/50_StrokePiG/TVC*"}
TARGET_HZ = 30.0
MAX_GAP_S = 0.1
MIN_SEG_S = 1.0
LOWER = [JOINTS.index(j) for j in LOWER_JOINTS]
UPPER = [k for k in range(len(JOINTS)) if k not in LOWER]


def process(x, rate):
    """Yield (angles_30hz, has_upper) for each usable segment of one trial."""
    x = fill_gaps(x, int(MAX_GAP_S * rate))
    for s, e in contiguous_segments(x[:, LOWER], int(MIN_SEG_S * rate)):
        seg = x[s:e]
        lower = resample(seg[:, LOWER], rate, TARGET_HZ)
        out = np.full((lower.shape[0], len(JOINTS), 3), np.nan)
        out[:, LOWER] = lower
        has_upper = bool(np.isfinite(seg[:, UPPER]).all())
        if has_upper:
            out[:, UPPER] = resample(seg[:, UPPER], rate, TARGET_HZ)
        yield out.astype(np.float32), has_upper


def main():
    angles, rows, inventory = [], [], []
    offset = 0
    for group, pattern in GROUPS.items():
        for sdir in sorted(d for d in glob.glob(pattern) if os.path.isdir(d)):
            subject = os.path.basename(sdir)
            files = sorted(glob.glob(os.path.join(sdir, "*.c3d")))
            n_walk = n_seg = n_frames = n_upper = 0
            skipped = []
            for f in files:
                tr = read_trial(f)
                if tr is None:
                    skipped.append(os.path.basename(f))
                    continue
                n_walk += 1
                for y, has_upper in process(tr.angles, tr.rate):
                    angles.append(y)
                    rows.append(dict(subject=subject, group=group, trial=os.path.basename(f),
                                     start=offset, length=len(y), has_upper=has_upper,
                                     source_rate=tr.rate))
                    offset += len(y)
                    n_seg += 1
                    n_frames += len(y)
                    n_upper += len(y) if has_upper else 0
            inventory.append(dict(subject=subject, group=group, c3d_files=len(files),
                                  walking_trials=n_walk, segments=n_seg, frames_30hz=n_frames,
                                  frames_with_upper_body=n_upper, skipped=";".join(skipped)))
            print(subject, group, n_walk, n_seg, n_frames, n_upper, file=sys.stderr)

    os.makedirs("data/processed", exist_ok=True)
    seg = pd.DataFrame(rows)
    np.savez_compressed("data/processed/vc2023.npz", angles=np.concatenate(angles),
                        joints=np.array(JOINTS), **{c: seg[c].to_numpy() for c in seg.columns})
    inv = pd.DataFrame(inventory)
    inv["full_body"] = inv.frames_with_upper_body > 0
    inv.to_csv("results/e1/inventory_vc2023.tsv", sep="\t", index=False)
    summary = inv.groupby(["group", "full_body"]).agg(
        subjects=("subject", "count"), walking_trials=("walking_trials", "sum"),
        frames_30hz=("frames_30hz", "sum"), median_frames=("frames_30hz", "median"))
    summary.to_csv("results/e1/inventory_vc2023_summary.tsv", sep="\t")
    print(summary)
    print("subjects with no usable walking:", inv.loc[inv.frames_30hz == 0, "subject"].tolist())


if __name__ == "__main__":
    main()
