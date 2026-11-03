"""E2: paretic-side localisation in stroke survivors (docs/e2-plan.md).

Usage: .venv/bin/python scripts/05_eval_e2.py --tags seed1 seed2

Needs results/e1/rom_deg.tsv and results/e1/noise_floor_<tag>.tsv from
scripts/03_eval_e1.py, and data/processed/subject_meta.tsv from
scripts/04_subject_meta.py.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from motorphen import memcap
from motorphen.baselines import (SIDE_ANGLES, PCABaseline, normative_reference, peak_knee_flexion,
                                 rmse_side, side_cycles, zscore_side)
from motorphen.c3d_io import JOINTS
from motorphen.inference import SCORED_IDX, pass1, reconstruct, select_joints
from motorphen.model import GaitMAE

DATA = "data/processed/vc2023.npz"
EVENTS = "data/processed/vc2023_events.tsv"
META = "data/processed/subject_meta.tsv"
OUT = "results/e2"
SIDE_IDX = {"left": [JOINTS.index("l_hip"), JOINTS.index("l_knee")],
            "right": [JOINTS.index("r_hip"), JOINTS.index("r_knee")]}


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(tag, dev):
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        ck = torch.load(f"checkpoints/e1_{tag}.pt", map_location="cpu", weights_only=True)
    m = GaitMAE().to(dev).eval()
    m.load_state_dict(ck["model"])
    return m


def metrics(df, method):
    y = (df.paretic == "right").to_numpy()
    diff = (df[f"{method}_right"] - df[f"{method}_left"]).to_numpy()
    ok = ~np.isnan(diff)
    y, diff = y[ok], diff[ok]
    correct = (diff > 0) == y
    lo, hi = proportion_confint(correct.sum(), len(correct), method="wilson")
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], diff[i]))
    return dict(n=int(len(y)), accuracy=float(correct.mean()), acc_ci=[float(lo), float(hi)],
                auroc=float(roc_auc_score(y, diff)),
                auroc_ci=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]), correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["seed1", "seed2"])
    ap.add_argument("--max-mem-gb", type=float, default=4.0)
    args = ap.parse_args()
    memcap.start(args.max_mem_gb)
    os.makedirs(OUT, exist_ok=True)
    dev = device()

    z = np.load(DATA, allow_pickle=True)
    seg = pd.DataFrame({k: z[k] for k in ["subject", "group", "trial", "start", "length", "has_upper"]})
    seg["segment"] = np.arange(len(seg))
    split = pd.read_csv("results/e1/split.tsv", sep="\t").set_index("subject")["split"]
    seg["split"] = seg.subject.map(split)
    meta = pd.read_csv(META, sep="\t").set_index("subject")
    ev = pd.read_csv(EVENTS, sep="\t")
    strikes = {side: ev[(ev.side == side.capitalize()) & (ev.event == "Foot Strike")]
               .groupby("segment").frame.apply(list).to_dict() for side in ("left", "right")}
    A = z["angles"]

    def trial(r):
        return A[r.start:r.start + r.length].astype(np.float64)

    train = seg[(seg.group == "healthy") & seg.has_upper & (seg.split == "train")]
    stroke = seg[(seg.group == "stroke") & seg.has_upper]
    stroke = stroke[stroke.subject.map(meta.paretic).isin(["left", "right"])]

    # normative references and PCA from training subjects only
    ref = normative_reference({s: [(trial(r), strikes[s].get(r.segment, [])) for r in train.itertuples()]
                               for s in ("left", "right")})
    pca = PCABaseline().fit(np.concatenate([trial(r) for r in train.itertuples()]))
    rom = torch.tensor(np.deg2rad(pd.read_csv("results/e1/rom_deg.tsv", sep="\t", index_col=0)
                                  .loc[JOINTS].to_numpy()), dtype=torch.float32, device=dev)

    rows = []
    for subj, g in stroke.groupby("subject"):
        by_side = {s: [(trial(r), strikes[s].get(r.segment, [])) for r in g.itertuples()] for s in ("left", "right")}
        row = dict(subject=subj, paretic=meta.loc[subj, "paretic"], n_trials=len(g))
        for s in ("left", "right"):
            row[f"zscore_{s}"] = zscore_side(by_side[s], s, ref)
            row[f"rmse_{s}"] = rmse_side(by_side[s], s, ref)
            row[f"pca_{s}"] = pca.side_score([trial(r) for r in g.itertuples()], s)
            row[f"kneepeak_{s}"] = -peak_knee_flexion(by_side[s], s)   # lower peak = more likely paretic
        rows.append(row)
    base = pd.DataFrame(rows).set_index("subject")

    summary = {"n_subjects": int(len(base)), "paretic_right": int((base.paretic == "right").sum()),
               "methods": {}, "mcnemar_vs_model": {}, "flags": {}, "correction": {}}
    correct = {}
    for m in ("zscore", "rmse", "pca", "kneepeak"):
        summary["methods"][m], correct[m] = metrics(base, m)

    flag_rows, corr_rows = [], []
    for tag in args.tags:
        model = load_model(tag, dev)
        fl = pd.read_csv(f"results/e1/noise_floor_{tag}.tsv", sep="\t").set_index("joint").noise_floor
        floor = {JOINTS.index(j): v for j, v in fl.items()}
        side_scores = {}
        for subj, g in stroke.groupby("subject"):
            par = meta.loc[subj, "paretic"]
            acc = {"left": [], "right": []}
            orig_c, rec_c = [], []
            for r in g.itertuples():
                x = torch.tensor(np.deg2rad(trial(r)), dtype=torch.float32, device=dev)
                scores, _ = pass1(model, x, rom)
                for s in ("left", "right"):
                    acc[s].append(max(scores[j] for j in SIDE_IDX[s]))
                flagged = select_joints(scores, floor, 2)
                names = [JOINTS[j] for j in flagged]
                pside = "r_" if par == "right" else "l_"
                nside = "l_" if par == "right" else "r_"
                flag_rows.append(dict(tag=tag, subject=subj, segment=r.segment, paretic=par,
                                      flagged=";".join(names),
                                      paretic_flag=any(n.startswith(pside) for n in names),
                                      nonparetic_flag=any(n.startswith(nside) for n in names),
                                      **{f"score_{JOINTS[j]}": scores[j] for j in SCORED_IDX}))
                rec = np.rad2deg(reconstruct(model, x, flagged).cpu().numpy())
                st = strikes[par].get(r.segment, [])
                orig_c.append((trial(r), st))
                rec_c.append((rec, st))
            side_scores[subj] = {s: float(np.mean(v)) for s, v in acc.items()}
            o = side_cycles(orig_c, par)
            c = side_cycles(rec_c, par)
            for (joint, ax), co, cr, (mu, _) in zip(SIDE_ANGLES[par], o, c, ref[par]):
                if len(co):
                    corr_rows.append(dict(tag=tag, subject=subj, angle=f"{joint}_{ax}",
                                          rmse_original=float(np.sqrt(np.mean((co.mean(0) - mu) ** 2))),
                                          rmse_reconstructed=float(np.sqrt(np.mean((cr.mean(0) - mu) ** 2)))))
        for s in ("left", "right"):
            base[f"model_{tag}_{s}"] = [side_scores[i][s] for i in base.index]
        summary["methods"][f"model_{tag}"], correct[f"model_{tag}"] = metrics(base, f"model_{tag}")
        for m in ("zscore", "rmse", "pca", "kneepeak"):
            a, b = correct[f"model_{tag}"], correct[m]
            table = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
            summary["mcnemar_vs_model"][f"{tag}_vs_{m}"] = dict(table=table, p=float(mcnemar(table, exact=True).pvalue))

    flags = pd.DataFrame(flag_rows)
    corr = pd.DataFrame(corr_rows)
    for tag in args.tags:
        f = flags[flags.tag == tag]
        summary["flags"][tag] = dict(trials=int(len(f)), any=float((f.flagged != "").mean()),
                                     paretic_side=float(f.paretic_flag.mean()),
                                     nonparetic_side=float(f.nonparetic_flag.mean()),
                                     paretic_only=float((f.paretic_flag & ~f.nonparetic_flag).mean()))
        for ang, c in corr[corr.tag == tag].groupby("angle"):
            d = c.rmse_reconstructed - c.rmse_original
            summary["correction"][f"{tag}/{ang}"] = dict(
                n=int(len(c)), rmse_original=float(c.rmse_original.mean()),
                rmse_reconstructed=float(c.rmse_reconstructed.mean()), mean_diff=float(d.mean()),
                p_wilcoxon=float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0)

    base.to_csv(f"{OUT}/side_scores.tsv", sep="\t")
    flags.to_csv(f"{OUT}/flags_stroke.tsv", sep="\t", index=False)
    corr.to_csv(f"{OUT}/correction.tsv", sep="\t", index=False)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    for m, v in summary["methods"].items():
        print(f"{m:<14} acc {v['accuracy']:.2f} [{v['acc_ci'][0]:.2f}, {v['acc_ci'][1]:.2f}]  "
              f"AUROC {v['auroc']:.2f} [{v['auroc_ci'][0]:.2f}, {v['auroc_ci'][1]:.2f}]  n={v['n']}")
    for k, v in summary["mcnemar_vs_model"].items():
        print(f"McNemar {k:<20} p={v['p']:.3f} table={v['table']}")
    for k, v in summary["flags"].items():
        print("flags", k, {a: round(b, 2) for a, b in v.items()})
    for k, v in summary["correction"].items():
        print(f"correction {k:<18} {v['rmse_original']:.2f} -> {v['rmse_reconstructed']:.2f}  p={v['p_wilcoxon']:.3g}")


if __name__ == "__main__":
    main()
