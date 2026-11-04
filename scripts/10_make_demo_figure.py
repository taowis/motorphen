"""Stick-figure demo of flagged joints and their reconstruction (README figure).

Usage: .venv/bin/python scripts/10_make_demo_figure.py [--tag seed1] [--seed 0]
       .venv/bin/python scripts/10_make_demo_figure.py --select-paretic-only
       .venv/bin/python scripts/10_make_demo_figure.py --animate TVC27 TVC04

Two held-out healthy adults and six stroke survivors are drawn at random with a
fixed seed, not chosen for the result. For each, the first full-body walking
segment is scored and reconstructed as in E1 and E2 (at most two joints above
the noise floor), and six frames of one gait cycle are drawn: original pose in
grey, reconstruction in blue, flagged joints in red. Poses come from the
Plug-in Gait segment frames in the C3D files (src/motorphen/skeleton.py).

Writes docs/images/motorphen-demo.png and results/demo/demo_cases.tsv.

With --select-paretic-only, one stroke survivor is chosen instead of drawn: the
one with the highest share of trials (run 1, results/e2/flags_stroke.tsv) in
which every flagged joint is on the paretic side, ties broken by subject ID.
Writes docs/images/motorphen-demo-selected.png and
results/demo/demo_selected.tsv.

With --animate, the whole first walking segment of each named participant is
drawn frame by frame at half speed (15 frames per second for 30 Hz data) to
docs/images/motorphen-anim-<subject>.gif. Each GIF says whether the participant
is the selected example or comes from the random draw.
"""

import argparse
import importlib.util
import os

import ezc3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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


def paretic_only_subject(tag):
    f = pd.read_csv("results/e2/flags_stroke.tsv", sep="\t")
    f = f[f.tag == tag].fillna({"flagged": ""})
    side = np.where(f.paretic == "right", "r_", "l_")
    f["only"] = [fl != "" and all(n.startswith(s) for n in fl.split(";"))
                 for fl, s in zip(f.flagged, side)]
    share = f.groupby("subject").only.mean().reset_index()
    best = share.sort_values(["only", "subject"], ascending=[False, True]).iloc[0]
    print(f"paretic-only trials: {int(f.only.sum())} of {len(f)}; chosen {best.subject} ({best.only:.2f})")
    return best.subject


def draw(ax, P, color, lw, x0, flagged_edges=()):
    for a, b in skeleton.EDGES:
        c = RED if (a, b) in flagged_edges else color
        ax.plot([P[a][0] + x0, P[b][0] + x0], [P[a][1], P[b][1]], color=c, lw=lw,
                solid_capstyle="round", zorder=3 if c == RED else 2)
    xy = np.array([P[k] for k in skeleton.POINTS])
    ax.scatter(xy[:, 0] + x0, xy[:, 1], s=6, color=color, zorder=4, linewidths=0)


def animate(cs, title, flagged, out, T, fps=15):
    """GIF of one walking segment, from the first reconstructed frame to the end."""
    fig, ax = plt.subplots(figsize=(4.4, 5.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.14)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in (GREY, BLUE, RED)]

    def frame(k):
        ax.cla()
        P0, P1 = cs["poses"](k)
        ax.plot([-0.9, 0.9], [0, 0], color="#dddddd", lw=1, zorder=1)
        draw(ax, P0, GREY, 2.2, 0)
        draw(ax, P1, BLUE, 2.2, 0, cs["red"])
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.08, 1.9)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.text(0.5, -0.02, flagged, transform=ax.transAxes, ha="center", va="top", fontsize=9.5)
        ax.text(0.99, 0.99, f"{(k - T + 1) / 30:.2f} s, half speed", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color="#777777")

    fig.legend(handles, ["original", "reconstruction", "flagged"], loc="lower center",
               ncol=3, frameon=False, fontsize=8.5)
    anim = FuncAnimation(fig, frame, frames=range(T - 1, cs["n"]), interval=1000 / fps)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print(out, os.path.getsize(out) // 1024, "kB,", cs["n"] - T + 1, "frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="seed1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--azimuth", type=float, default=35.0, help="view, degrees from the side")
    ap.add_argument("--select-paretic-only", action="store_true")
    ap.add_argument("--animate", nargs="+", metavar="SUBJECT",
                    help="write a GIF of the first walking segment of each subject")
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
    if args.animate:
        chosen, axes = [], np.array([])
    elif args.select_paretic_only:
        chosen = [("stroke", paretic_only_subject(args.tag))]
        out, cases = "docs/images/motorphen-demo-selected.png", "results/demo/demo_selected.tsv"
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.1))
        axes = np.array([ax])
    else:
        rng = np.random.default_rng(args.seed)
        chosen = ([("healthy", s) for s in sorted(rng.choice(healthy, 2, replace=False))] +
                  [("stroke", s) for s in sorted(rng.choice(stroke, 6, replace=False))])
        out, cases = "docs/images/motorphen-demo.png", "results/demo/demo_cases.tsv"
        fig, axes = plt.subplots(4, 2, figsize=(16, 11.6))
    def case(group, subj):
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

        o = frames["PEL"][1][s0:s0 + int(r.length * rate / 30)]
        d = o[-1] - o[0]
        d[2] = 0
        d /= np.linalg.norm(d)
        lat = np.cross([0, 0, 1.0], d)
        az = np.deg2rad(args.azimuth)
        screen = np.cos(az) * d + np.sin(az) * lat

        def poses(k):
            """Projected original and reconstructed points at 30 Hz frame k, in metres,
            centred on the original mid-hip."""
            t = s0 + int(round(k * rate / 30))
            P0 = skeleton.points(*skeleton.pose(frames, centres, t), centres)
            P1 = skeleton.points(*skeleton.pose(frames, centres, t, np.deg2rad(A[k]), rec[k]), centres)
            mid = P0["mid_hip"] @ screen
            proj = lambda P: {n: np.array([(p @ screen - mid) / 1000, p[2] / 1000]) for n, p in P.items()}
            return proj(P0), proj(P1)

        red = [e for n in names for e in skeleton.FLAG_EDGES.get(n, [])]
        return dict(r=r, names=names, red=red, poses=poses, n=int(r.length))

    def title_of(group, subj):
        if group == "healthy":
            return f"{subj}  ·  healthy adult, held out"
        return f"{subj}  ·  stroke, {meta.loc[subj, 'paretic']} side paretic"

    def flagged_text(names):
        return "flagged: " + (", ".join(NAMES[n] for n in names) if names else "none")

    if args.animate:
        rng = np.random.default_rng(args.seed)
        rng.choice(healthy, 2, replace=False)
        random_stroke = set(rng.choice(stroke, 6, replace=False))
        selected = paretic_only_subject(args.tag)
        for subj in args.animate:
            group = "stroke" if subj in stroke else "healthy"
            cs = case(group, subj)
            if subj == selected:
                origin = "selected example, not random"
            elif subj in random_stroke:
                origin = "from the random draw"
            else:
                origin = "chosen by hand"
            animate(cs, f"{title_of(group, subj)}\n{origin}", flagged_text(cs["names"]),
                    f"docs/images/motorphen-anim-{subj}.gif", model.T)
        return

    rows = []
    for ax, (group, subj) in zip(axes.T.ravel(), chosen):
        cs = case(group, subj)
        r, names, red = cs["r"], cs["names"], cs["red"]
        T = model.T
        st = [k for k in strikes.get(r.segment, []) if k >= T - 1]
        k0, k1 = (st[0], st[2]) if len(st) >= 3 else (T - 1, min(T - 1 + 30, r.length - 1))
        ks = np.linspace(k0, k1, 6).round().astype(int)
        for i, k in enumerate(ks):
            P0, P1 = cs["poses"](k)
            draw(ax, P0, GREY, 1.6, i * GAP)
            draw(ax, P1, BLUE, 1.6, i * GAP, red)
        title = title_of(group, subj)
        if args.select_paretic_only:
            title += "  ·  selected example, not random"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(0.5, -0.06, flagged_text(names), transform=ax.transAxes, ha="center", fontsize=10.5)
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
    if args.select_paretic_only:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.17)
    else:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.06, wspace=0.04, hspace=0.32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    os.makedirs("results/demo", exist_ok=True)
    pd.DataFrame(rows).to_csv(cases, sep="\t", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
