"""Subject splits for E3b (age range of the normative training data).

See docs/e3-plan.md. Writes results/e3/young/split.tsv and
results/e3/allages/split.tsv (train / val / test_young / test_old) and
results/e3/age_test_set.tsv. Subjects whose MAT-to-folder mapping was
inconsistent are excluded, since their age is not reliable.
"""

import os

import numpy as np
import pandas as pd

SEED = 20261104
N_TEST_OLD, N_TEST_YOUNG = 20, 16
N_VAL = 6   # training size = young pool minus validation, same for both models


def main():
    meta = pd.read_csv("data/processed/subject_meta.tsv", sep="\t")
    z = np.load("data/processed/vc2023.npz", allow_pickle=True)
    seg = pd.DataFrame({k: z[k] for k in ["subject", "group", "has_upper"]})
    full = set(seg[(seg.group == "healthy") & seg.has_upper].subject)
    h = meta[(meta.group == "healthy") & (meta.check != "inconsistent") & meta.subject.isin(full)]
    h = h.sort_values("subject").reset_index(drop=True)
    young = h[(h.Age >= 22) & (h.Age <= 57)]
    old = h[h.Age >= 60]
    rng = np.random.default_rng(SEED)
    test_old = set(rng.choice(old.subject, N_TEST_OLD, replace=False))
    test_young = set(rng.choice(young.subject, N_TEST_YOUNG, replace=False))
    test = test_old | test_young
    rest = h[~h.subject.isin(test)]

    pools = {"young": rest[(rest.Age >= 22) & (rest.Age <= 57)], "allages": rest}
    n_train = len(pools["young"]) - N_VAL
    print("pool sizes:", {k: len(v) for k, v in pools.items()}, "-> train", n_train, "val", N_VAL)
    os.makedirs("results/e3", exist_ok=True)
    for name, pool in pools.items():
        pick = rng.choice(pool.subject, n_train + N_VAL, replace=False)
        split = {s: "train" for s in pick[:n_train]}
        split.update({s: "val" for s in pick[n_train:]})
        split.update({s: "test_old" for s in test_old})
        split.update({s: "test_young" for s in test_young})
        out = pd.Series(split, name="split").rename_axis("subject").sort_index()
        os.makedirs(f"results/e3/{name}", exist_ok=True)
        out.to_csv(f"results/e3/{name}/split.tsv", sep="\t")
        ages = h.set_index("subject").Age
        tr = ages[[s for s, v in split.items() if v == "train"]]
        print(f"{name}: train n={len(tr)} age {tr.min():.0f}-{tr.max():.0f} (median {tr.median():.0f}), "
              f"aged 60+ in train: {(tr >= 60).sum()}")
    t = h[h.subject.isin(test)][["subject", "Age", "Male"]].copy()
    t["band"] = np.where(t.Age >= 60, "60+", "22-57")
    t.to_csv("results/e3/age_test_set.tsv", sep="\t", index=False)
    print(t.groupby("band").Age.agg(["count", "min", "max"]))


if __name__ == "__main__":
    main()
