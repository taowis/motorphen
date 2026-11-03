"""Nested training subsets of the E1 split for the E3d learning curve.

Writes results/e3/lc_n25/split.tsv and results/e3/lc_n50/split.tsv (same
validation and test subjects as E1) and prints the epochs needed for an equal
number of optimiser steps. See docs/e3-plan.md (E3d).
"""

import os

import numpy as np
import pandas as pd

from motorphen.train import window_starts

SEED = 20261105
SIZES = [25, 50]


def main():
    split = pd.read_csv("results/e1/split.tsv", sep="\t").set_index("subject")["split"]
    z = np.load("data/processed/vc2023.npz", allow_pickle=True)
    seg = pd.DataFrame({k: z[k] for k in ["subject", "start", "length", "has_upper"]})
    seg = seg[seg.has_upper]
    train = sorted(split[split == "train"].index)
    order = list(np.random.default_rng(SEED).permutation(train))

    def n_windows(subjects):
        g = seg[seg.subject.isin(subjects)]
        return len(window_starts(g.start.to_numpy(), g.length.to_numpy(), 7))

    full = n_windows(train)
    for n in SIZES:
        keep = set(order[:n])
        s = split.copy()
        s[(s == "train") & ~s.index.isin(keep)] = "unused"
        os.makedirs(f"results/e3/lc_n{n}", exist_ok=True)
        s.rename("split").to_csv(f"results/e3/lc_n{n}/split.tsv", sep="\t")
        w = n_windows(keep)
        print(f"n={n}: {w} windows, epochs for equal steps = {round(250 * full / w)}")


if __name__ == "__main__":
    main()
