"""E1 specificity: does the reconstruction leave held-out healthy gait unchanged?

Usage: .venv/bin/python scripts/03_eval_e1.py --tag seed1

Follows paper section 3.2: right-side pelvis flexion, hip ab/adduction, hip
flexion and knee flexion; cycles from right foot strike to right foot strike,
100 points; one mean trajectory per subject; RMSE to the training mean; paired
difference reconstructed minus original; bootstrap 90% CI against +/-1.5 deg.

Two selection rules are reported: joints above the noise floor (paper) and
always the top K = 2 (stricter, since the model then always edits two joints).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from motorphen import memcap
from motorphen.c3d_io import JOINTS
from motorphen.evaluate import REPORTED, bootstrap_equivalence, reported_cycles, rmse
from motorphen.inference import SCORED_IDX, pass1, range_of_motion, reconstruct, select_joints
from motorphen.model import GaitMAE

DATA = "data/processed/vc2023.npz"
EVENTS = "data/processed/vc2023_events.tsv"


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="seed1")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--floor-q", type=float, default=0.99)
    ap.add_argument("--max-mem-gb", type=float, default=4.0)
    ap.add_argument("--out", default="results/e1")
    ap.add_argument("--split-file", default=None, help="default: <out>/split.tsv")
    ap.add_argument("--ckpt", default=None, help="default: checkpoints/e1_<tag>.pt")
    ap.add_argument("--test-splits", nargs="+", default=["test"])
    args = ap.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    memcap.start(args.max_mem_gb)
    dev = device()

    z = np.load(DATA, allow_pickle=True)
    seg = pd.DataFrame({k: z[k] for k in ["subject", "group", "trial", "start", "length", "has_upper"]})
    seg["segment"] = np.arange(len(seg))
    split = pd.read_csv(args.split_file or f"{OUT}/split.tsv", sep="\t").set_index("subject")["split"]
    seg["split"] = seg.subject.map(split)
    ev = pd.read_csv(EVENTS, sep="\t")
    rstrikes = (ev[(ev.side == "Right") & (ev.event == "Foot Strike")]
                .groupby("segment").frame.apply(list).to_dict())
    angles_deg = z["angles"]
    healthy = seg[(seg.group == "healthy") & seg.has_upper & seg.split.notna()]

    def trial(r):
        return angles_deg[r.start:r.start + r.length].astype(np.float64)

    # ROM and normative mean from the training subjects
    train = healthy[healthy.split == "train"]
    train_angles = np.concatenate([trial(r) for r in train.itertuples()])
    rom = torch.tensor(range_of_motion(np.deg2rad(train_angles)), dtype=torch.float32)
    pooled = {name: [] for name, _, _ in REPORTED}
    for r in train.itertuples():
        for name, c in reported_cycles(trial(r), rstrikes.get(r.segment, [])).items():
            pooled[name].append(c)
    mu = {k: np.concatenate(v).mean(0) for k, v in pooled.items()}
    sd = {k: np.concatenate(v).std(0) for k, v in pooled.items()}
    pd.DataFrame({**{f"{k}_mean": v for k, v in mu.items()}, **{f"{k}_sd": v for k, v in sd.items()}}) \
        .to_csv(f"{OUT}/normative_mean.tsv", sep="\t", index_label="cycle_pct")
    pd.DataFrame(np.rad2deg(rom.numpy()), index=JOINTS, columns=["flex", "abd", "rot"]) \
        .to_csv(f"{OUT}/rom_deg.tsv", sep="\t", index_label="joint")

    # older checkpoints stored torch.__version__ as a TorchVersion object; allow
    # only that type and keep weights_only loading
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        ck = torch.load(args.ckpt or f"checkpoints/e1_{args.tag}.pt", map_location="cpu", weights_only=True)
    model = GaitMAE().to(dev).eval()
    model.load_state_dict(ck["model"])
    rom_d = rom.to(dev)

    def seq_of(r):
        return torch.tensor(np.deg2rad(trial(r)), dtype=torch.float32, device=dev)

    # noise floor from validation subjects
    val_scores = []
    for r in healthy[healthy.split == "val"].itertuples():
        s, _ = pass1(model, seq_of(r), rom_d)
        val_scores.append(s)
    val_df = pd.DataFrame(val_scores)
    floor = {j: float(val_df[j].quantile(args.floor_q)) for j in SCORED_IDX}
    pd.DataFrame({"joint": [JOINTS[j] for j in SCORED_IDX], "noise_floor": [floor[j] for j in SCORED_IDX]}) \
        .to_csv(f"{OUT}/noise_floor_{args.tag}.tsv", sep="\t", index=False)

    # held-out healthy subjects
    flags_rows, per_subject = [], {}
    for r in healthy[healthy.split.isin(args.test_splits)].itertuples():
        x = seq_of(r)
        scores, _ = pass1(model, x, rom_d)
        sel = {"floor": select_joints(scores, floor, args.k), "topk": select_joints(scores, None, args.k)}
        flags_rows.append(dict(subject=r.subject, segment=r.segment, split=r.split,
                               **{f"score_{JOINTS[j]}": scores[j] for j in SCORED_IDX},
                               flagged_floor=";".join(JOINTS[j] for j in sel["floor"]),
                               flagged_topk=";".join(JOINTS[j] for j in sel["topk"])))
        versions = {"original": trial(r)}
        for rule, joints in sel.items():
            versions[rule] = np.rad2deg(reconstruct(model, x, joints).cpu().numpy())
        st = rstrikes.get(r.segment, [])
        d = per_subject.setdefault((r.subject, r.split), {})
        for v, a in versions.items():
            for name, c in reported_cycles(a, st).items():
                d.setdefault((v, name), []).append(c)

    rows = []
    for (subj, sp), d in per_subject.items():
        for name, _, _ in REPORTED:
            row = dict(subject=subj, split=sp, angle=name)
            for v in ("original", "floor", "topk"):
                c = np.concatenate(d[(v, name)])
                row[f"n_cycles"] = len(c)
                row[f"rmse_{v}"] = rmse(c.mean(0), mu[name]) if len(c) else np.nan
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/specificity_{args.tag}.tsv", sep="\t", index=False)
    pd.DataFrame(flags_rows).to_csv(f"{OUT}/flags_test_healthy_{args.tag}.tsv", sep="\t", index=False)

    summary = {"tag": args.tag, "k": args.k, "floor_quantile": args.floor_q,
               "n_test_subjects": int(res.subject.nunique()), "results": {}}
    for sp in args.test_splits:
        prefix = "" if len(args.test_splits) == 1 else f"{sp}/"
        for rule in ("floor", "topk"):
            for name, _, _ in REPORTED:
                sub = res[(res.angle == name) & (res.split == sp)].dropna()
                summary["results"][f"{prefix}{rule}/{name}"] = bootstrap_equivalence(
                    sub[f"rmse_{rule}"] - sub.rmse_original)
    with open(f"{OUT}/specificity_summary_{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=1)
    for k, v in summary["results"].items():
        print(f"{k:<24} mean diff {v['mean']:+.2f} deg  90% CI [{v['ci_low']:+.2f}, {v['ci_high']:+.2f}]"
              f"  equivalent={v['equivalent']}  n={v['n']}")


if __name__ == "__main__":
    main()
