"""E3a: inference choices without retraining (docs/e3-plan.md).

Usage: .venv/bin/python scripts/07_eval_e3a.py --tags seed1 seed2

For every validation and test healthy trial (E1 split) and every stroke trial,
per-frame badness is computed once for neck, pelvis, hips, knees and ankles.
Variants (peak statistic, ankles scored, K) are then derived from these.
Writes results/e3/e3a/.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from motorphen import memcap
from motorphen.c3d_io import JOINTS
from motorphen.inference import SCORED_IDX, frame_badness, peak
from motorphen.model import GaitMAE

OUT = "results/e3/e3a"
ANKLES = [JOINTS.index("l_ankle"), JOINTS.index("r_ankle")]
ALL8 = SCORED_IDX + ANKLES
PEAKS = ["q95", "max", "mean"]
KS = [1, 2, 3, None]


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def side_of(name):
    return "left" if name.startswith("l_") else "right" if name.startswith("r_") else "midline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["seed1", "seed2"])
    ap.add_argument("--max-mem-gb", type=float, default=4.0)
    args = ap.parse_args()
    memcap.start(args.max_mem_gb)
    os.makedirs(OUT, exist_ok=True)
    dev = device()

    z = np.load("data/processed/vc2023.npz", allow_pickle=True)
    seg = pd.DataFrame({k: z[k] for k in ["subject", "group", "start", "length", "has_upper"]})
    seg["segment"] = np.arange(len(seg))
    split = pd.read_csv("results/e1/split.tsv", sep="\t").set_index("subject")["split"]
    seg["split"] = seg.subject.map(split)
    meta = pd.read_csv("data/processed/subject_meta.tsv", sep="\t").set_index("subject")
    seg.loc[seg.group == "stroke", "split"] = "stroke"
    seg = seg[seg.has_upper & seg.split.isin(["val", "test", "stroke"])]
    rom = torch.tensor(np.deg2rad(pd.read_csv("results/e1/rom_deg.tsv", sep="\t", index_col=0)
                                  .loc[JOINTS].to_numpy()), dtype=torch.float32, device=dev)
    A = z["angles"]

    # 1. peak statistics per trial and joint, for each model
    rows = []
    for tag in args.tags:
        from torch.torch_version import TorchVersion
        with torch.serialization.safe_globals([TorchVersion]):
            ck = torch.load(f"checkpoints/e1_{tag}.pt", map_location="cpu", weights_only=True)
        model = GaitMAE(n_frames=int(ck["run"].get("window", 7))).to(dev).eval()
        model.load_state_dict(ck["model"])
        for r in seg.itertuples():
            x = torch.tensor(np.deg2rad(A[r.start:r.start + r.length].astype(np.float64)),
                             dtype=torch.float32, device=dev)
            b = frame_badness(model, x, rom, ALL8)
            for j, v in b.items():
                rows.append(dict(tag=tag, subject=r.subject, segment=r.segment, split=r.split,
                                 joint=JOINTS[j], **{p: peak(v, p) for p in PEAKS}))
            if dev.type == "mps":
                torch.mps.empty_cache()
        del model
    pk = pd.DataFrame(rows)
    pk.to_csv(f"{OUT}/trial_peaks.tsv", sep="\t", index=False)

    # 2. variants
    hf, sf, ss = [], [], []
    for tag in args.tags:
        for p in PEAKS:
            for ankles in (False, True):
                joints = [JOINTS[j] for j in (ALL8 if ankles else SCORED_IDX)]
                d = pk[(pk.tag == tag) & pk.joint.isin(joints)].pivot_table(
                    index=["subject", "segment", "split"], columns="joint", values=p).reset_index()
                floor = d[d.split == "val"][joints].quantile(0.99)
                above = d[joints].gt(floor)
                d["n_above"] = above.sum(axis=1)
                test = d[d.split == "test"]
                hf.append(dict(tag=tag, peak=p, ankles=ankles, trials=len(test),
                               flagged_share=float((test.n_above > 0).mean()),
                               subjects_flagged=int(test[test.n_above > 0].subject.nunique()),
                               subjects=int(test.subject.nunique()),
                               mean_joints_above=float(test.n_above.mean())))
                st = d[d.split == "stroke"].copy()
                st["paretic"] = st.subject.map(meta.paretic)
                st = st[st.paretic.isin(["left", "right"])]
                for k in KS:
                    pf = no = 0
                    for i, row in st.iterrows():
                        cand = [j for j in joints if above.loc[i, j]]
                        cand = sorted(cand, key=lambda j: row[j], reverse=True)
                        cand = cand if k is None else cand[:k]
                        sides = {side_of(j) for j in cand}
                        pf += row.paretic in sides and (({"left", "right"} - {row.paretic}) & sides) == set()
                        no += len(cand) == 0
                    sf.append(dict(tag=tag, peak=p, ankles=ankles, k="none" if k is None else k,
                                   trials=len(st), paretic_side_only=pf / len(st), no_flag=no / len(st)))
                sides = {"left": [j for j in joints if side_of(j) == "left"],
                         "right": [j for j in joints if side_of(j) == "right"]}
                st["left"] = st[sides["left"]].max(axis=1)
                st["right"] = st[sides["right"]].max(axis=1)
                subj = st.groupby("subject").agg(left=("left", "mean"), right=("right", "mean"),
                                                 paretic=("paretic", "first"))
                y = subj.paretic == "right"
                diff = subj.right - subj.left
                ss.append(dict(tag=tag, peak=p, ankles=ankles, n=len(subj),
                               accuracy=float(((diff > 0) == y).mean()), auroc=float(roc_auc_score(y, diff))))
    hf, sf, ss = pd.DataFrame(hf), pd.DataFrame(sf), pd.DataFrame(ss)
    hf.to_csv(f"{OUT}/healthy_flags.tsv", sep="\t", index=False)
    sf.to_csv(f"{OUT}/stroke_flags_by_k.tsv", sep="\t", index=False)
    ss.to_csv(f"{OUT}/stroke_side.tsv", sep="\t", index=False)
    pd.set_option("display.width", 160)
    print(hf.round(2).to_string(index=False))
    print(ss.round(2).to_string(index=False))
    print(sf[(sf.peak == "q95")].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
