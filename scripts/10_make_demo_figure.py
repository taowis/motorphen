"""Stick-figure demo of flagged joints and their reconstruction (README figure).

Usage: .venv/bin/python scripts/10_make_demo_figure.py [--tag seed1] [--seed 0]

Two held-out healthy adults and six stroke survivors are drawn at random with a
fixed seed, not chosen for the result. For each, the first full-body walking
segment is scored and reconstructed as in E1 and E2 (at most two joints above
the noise floor), and six frames of one gait cycle are drawn: original pose in
grey, reconstruction in blue, flagged joints in red. Poses come from the
Plug-in Gait segment frames in the C3D files (src/motorphen/skeleton.py).

Writes docs/images/motorphen-demo.png and results/demo/demo_cases.tsv.
"""

import argparse
import importlib.util
import os

import ezc3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from motorphen import memcap, skeleton
from motorphen.c3d_io import JOINTS, _labels, read_trial
from motorphen.inference import pass1, reconstruct, select_joints
from motorphen.model import GaitMAE

DATA = "data/processed/vc2023.npz"
EVENTS = "data/processed/vc2023_events.tsv"
META = "data/processed/subject_meta.tsv"
ROOT = "data/raw/vancriekinge2023/extracted"
DIRS = {"healthy": "138_HealthyPiG_10.05", "stroke": "50_StrokePiG"}
GAP = 1.0   # metres between snapshots
GREY, BLUE, RED = "#b4b4b4", "#1537b5", "#d7191c"
NAMES = {"neck": "neck", "pelvis": "pelvis", "l_hip": "left hip", "r_hip": "right hip",
         "l_knee": "left knee", "r_knee": "right knee"}

spec = importlib.util.spec_from_file_location("prep", "scripts/01_prepare_vc2023.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def source_segment(path, target):
    """Re-run the preprocessing on one C3D file and return the source frame
    bounds of the segment equal to `target` (30 Hz angles)."""
    tr = read_trial(path)
    for y, _, (s, e) in prep.process(tr.angles, tr.rate):
        if y.shape == target.shape and np.allclose(y, target, equal_nan=True, atol=1e-4):
            return tr.rate, s
    raise RuntimeError(f"segment not found in {path}")


def draw(ax, P, color, lw, x0, flagged_edges=()):
    for a, b in skeleton.EDGES:
        c = RED if (a, b) in flagged_edges else color
        ax.plot([P[a][0] + x0, P[b][0] + x0], [P[a][1], P[b][1]], color=c, lw=lw,
                solid_capstyle="round", zorder=3 if c == RED else 2)
    xy = np.array([P[k] for k in skeleton.POINTS])
    ax.scatter(xy[:, 0] + x0, xy[:, 1], s=6, color=color, zorder=4, linewidths=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="seed1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--azimuth", type=float, default=35.0, help="view, degrees from the side")
    ap.add_argument("--out", default="docs/images/motorphen-demo.png")
    args = ap.parse_args()
    memcap.start(4.0)
    dev = torch.device("cpu")

    z = np.load(DATA, allow_pickle=True)
    seg = pd.DataFrame({k: z[k] for k in ["subject", "group", "trial", "start", "length", "has_upper"]})
    seg["segment"] = np.arange(len(seg))
    split = pd.read_csv("results/e1/split.tsv", sep="\t").set_index("subject")["split"]
    meta = pd.read_csv(META, sep="\t").set_index("subject")
    ev = pd.read_csv(EVENTS, sep="\t")
    strikes = ev[ev.event == "Foot Strike"].groupby("segment").frame.apply(sorted).to_dict()

    ck = torch.load(f"checkpoints/e1_{args.tag}.pt", map_location="cpu", weights_only=False)
    model = GaitMAE().eval()
    model.load_state_dict(ck["model"])
    rom = torch.tensor(np.deg2rad(pd.read_csv("results/e1/rom_deg.tsv", sep="\t", index_col=0)
                                  .loc[JOINTS].to_numpy()), dtype=torch.float32)
    fl = pd.read_csv(f"results/e1/noise_floor_{args.tag}.tsv", sep="\t").set_index("joint").noise_floor
    floor = {JOINTS.index(j): v for j, v in fl.items()}

    full = seg[seg.has_upper]
    healthy = sorted(full[(full.group == "healthy") & (full.subject.map(split) == "test")].subject.unique())
    stroke = sorted(full[(full.group == "stroke") &
                         full.subject.map(meta.paretic).isin(["left", "right"])].subject.unique())
    rng = np.random.default_rng(args.seed)
    chosen = ([("healthy", s) for s in sorted(rng.choice(healthy, 2, replace=False))] +
              [("stroke", s) for s in sorted(rng.choice(stroke, 6, replace=False))])

    fig, axes = plt.subplots(4, 2, figsize=(16, 11.6))
    rows = []
    for ax, (group, subj) in zip(axes.T.ravel(), chosen):
        r = full[full.subject == subj].sort_values("segment").iloc[0]
        A = z["angles"][r.start:r.start + r.length].astype(np.float64)
        x = torch.tensor(np.deg2rad(A), dtype=torch.float32)
        scores, _ = pass1(model, x, rom)
        flagged = select_joints(scores, floor, 2)
        rec = reconstruct(model, x, flagged).numpy().astype(np.float64)
        names = [JOINTS[j] for j in flagged]

        path = os.path.join(ROOT, DIRS[group], subj, r.trial)
        rate, s0 = source_segment(path, z["angles"][r.start:r.start + r.length])
        c = ezc3d.c3d(path)
        frames = skeleton.segment_frames(c["data"]["points"], _labels(c["parameters"]["POINT"]))
        centres = skeleton.fit(frames)

        T = model.T
        st = [k for k in strikes.get(r.segment, []) if k >= T - 1]
        k0, k1 = (st[0], st[2]) if len(st) >= 3 else (T - 1, min(T - 1 + 30, r.length - 1))
        ks = np.linspace(k0, k1, 6).round().astype(int)

        o = frames["PEL"][1][s0:s0 + int(r.length * rate / 30)]
        d = o[-1] - o[0]
        d[2] = 0
        d /= np.linalg.norm(d)
        lat = np.cross([0, 0, 1.0], d)
        az = np.deg2rad(args.azimuth)
        screen = np.cos(az) * d + np.sin(az) * lat

        red = [e for n in names for e in skeleton.FLAG_EDGES.get(n, [])]
        for i, k in enumerate(ks):
            t = s0 + int(round(k * rate / 30))
            P0 = skeleton.points(*skeleton.pose(frames, centres, t), centres)
            P1 = skeleton.points(*skeleton.pose(frames, centres, t, np.deg2rad(A[k]), rec[k]), centres)
            mid = P0["mid_hip"] @ screen
            proj = lambda P: {n: np.array([(p @ screen - mid) / 1000, p[2] / 1000]) for n, p in P.items()}
            draw(ax, proj(P0), GREY, 1.6, i * GAP)
            draw(ax, proj(P1), BLUE, 1.6, i * GAP, red)
        if group == "healthy":
            title = f"{subj}  ·  healthy adult, held out"
        else:
            title = f"{subj}  ·  stroke, {meta.loc[subj, 'paretic']} side paretic"
        ax.set_title(title, fontsize=11, fontweight="bold")
        label = ", ".join(NAMES[n] for n in names) if names else "none"
        ax.text(0.5, -0.06, f"flagged: {label}", transform=ax.transAxes, ha="center", fontsize=10.5)
        ax.set_xlim(-0.55, 5 * GAP + 0.55)
        ax.set_ylim(-0.05, 1.95)
        ax.set_aspect("equal")
        ax.axis("off")
        rows.append(dict(group=group, subject=subj, segment=int(r.segment), trial=r.trial,
                         paretic=meta.paretic.get(subj, ""), flagged=";".join(names),
                         frames_30hz=";".join(map(str, ks))))

    handles = [plt.Line2D([], [], color=c, lw=2) for c in (GREY, BLUE, RED)]
    fig.legend(handles, ["original", "normative reconstruction", "flagged joint"],
               loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.06, wspace=0.04, hspace=0.32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    os.makedirs("results/demo", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/demo/demo_cases.tsv", sep="\t", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
