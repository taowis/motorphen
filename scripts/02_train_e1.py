"""Train the E1 masked autoencoder on able-bodied subjects (Van Criekinge 2023).

Usage: .venv/bin/python scripts/02_train_e1.py [--epochs N] [--seed S]

Writes the subject split and training log to results/e1/ and the checkpoint to
checkpoints/ (not committed). Settings follow docs/e1-implementation-spec.md.
"""

import argparse
import os

# MPS raises an out-of-memory error instead of taking the whole machine
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")
import hashlib
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from motorphen import memcap
from motorphen.masking import sample_masks
from motorphen.model import GaitMAE
from motorphen.train import compute_losses, gather_windows, make_optimizer, window_starts

DATA = "data/processed/vc2023.npz"
SPLIT_SEED = 20261103
N_TRAIN, N_VAL = 110, 8
J = 12


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_commit():
    """git commit of the code: from git, or from the COMMIT file written by
    scripts/remote/sync.sh when the copy is not a git checkout."""
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    if os.path.exists("COMMIT"):
        return open("COMMIT").read().strip()
    return ""


def ckpt_name(out, tag):
    """e1_seed1 for the E1 runs; e3_young_seed1 etc. for runs written to results/e3/<name>."""
    name = os.path.basename(os.path.normpath(out))
    return f"e1_{tag}" if name == "e1" else f"{name}_{tag}"


def make_split(seg):
    full = sorted(seg.loc[(seg.group == "healthy") & seg.has_upper, "subject"].unique())
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(full)
    split = {s: "train" for s in order[:N_TRAIN]}
    split.update({s: "val" for s in order[N_TRAIN:N_TRAIN + N_VAL]})
    split.update({s: "test" for s in order[N_TRAIN + N_VAL:]})
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--micro-batch", type=int, default=16,
                    help="gradient accumulation chunk; effective batch stays --batch")
    ap.add_argument("--max-batches", type=int, default=0, help="stop each epoch early (testing)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--window", type=int, default=7, help="frames per window (paper: 7)")
    ap.add_argument("--split-file", default=None,
                    help="TSV with subject and split (train/val/...); default: the E1 split")
    ap.add_argument("--out", default="results/e1", help="directory for split, log and run record")
    ap.add_argument("--max-mem-gb", type=float, default=4.0,
                    help="stop the run if process RSS plus MPS memory exceeds this")
    args = ap.parse_args()
    memcap.start(args.max_mem_gb)

    torch.manual_seed(args.seed)
    T = args.window
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    z = np.load(DATA, allow_pickle=True)
    seg = pd.DataFrame({k: z[k] for k in ["subject", "group", "trial", "start", "length", "has_upper"]})
    os.makedirs(args.out, exist_ok=True)
    if args.split_file:
        split = pd.read_csv(args.split_file, sep="\t").set_index("subject")["split"].to_dict()
    else:
        split = make_split(seg)
        pd.Series(split, name="split").rename_axis("subject").to_csv(f"{args.out}/split.tsv", sep="\t")

    seg["split"] = seg.subject.map(split)
    usable = seg[seg.has_upper & seg.split.notna()]
    starts = {s: window_starts(g.start.to_numpy(), g.length.to_numpy(), T)
              for s, g in usable.groupby("split")}
    all_angles = torch.tensor(np.deg2rad(z["angles"]), dtype=torch.float32, device=device)
    train_idx = torch.tensor(starts["train"], device=device)
    val_idx = torch.tensor(starts["val"], device=device)

    model = GaitMAE(n_joints=J, n_frames=T).to(device)
    opt = make_optimizer(model)
    n_params = sum(p.numel() for p in model.parameters())
    tag = args.tag or f"seed{args.seed}"
    run = dict(tag=tag, seed=args.seed, epochs=args.epochs, batch=args.batch,
               micro_batch=args.micro_batch, max_mem_gb=args.max_mem_gb, device=str(device),
               n_params=n_params, train_windows=len(train_idx), val_windows=len(val_idx),
               window=T, split_file=args.split_file or "E1 split", out=args.out,
               split_seed=SPLIT_SEED, data=DATA, data_md5=md5(DATA), torch=str(torch.__version__),
               python=platform.python_version(),
               git_commit=code_commit(),
               started=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps(run, indent=1))

    # fixed validation masks so val loss is comparable across epochs
    g_val = torch.Generator().manual_seed(0)
    val_m1 = sample_masks(len(val_idx), J, T, 250, gen=g_val).to(device)
    val_m2 = sample_masks(len(val_idx), J, T, 250, gen=g_val).to(device)

    g = torch.Generator().manual_seed(args.seed)
    log = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=g).to(device)]
        totals, n = {}, 0
        for b in range(0, len(perm), args.batch):
            if args.max_batches and b // args.batch >= args.max_batches:
                break
            idx = perm[b:b + args.batch]
            opt.zero_grad(set_to_none=True)
            for c in range(0, len(idx), args.micro_batch):
                sub = idx[c:c + args.micro_batch]
                a = gather_windows(all_angles, sub, T)
                m1 = sample_masks(len(sub), J, T, epoch, gen=g).to(device)
                m2 = sample_masks(len(sub), J, T, epoch, gen=g).to(device)
                loss, parts = compute_losses(model, a, m1, m2)
                (loss * len(sub) / len(idx)).backward()
                for k, v in parts.items():
                    totals[k] = totals.get(k, 0.0) + v.item() * len(sub)
                n += len(sub)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for b in range(0, len(val_idx), args.micro_batch):
                sl = slice(b, b + args.micro_batch)
                a = gather_windows(all_angles, val_idx[sl], T)
                vl += compute_losses(model, a, val_m1[sl], val_m2[sl])[0].item() * a.shape[0]
            vl /= len(val_idx)
        row = dict(epoch=epoch, **{k: v / n for k, v in totals.items()},
                   train_total=sum(totals.values()) / n, val_total=vl, minutes=(time.time() - t0) / 60)
        log.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model": model.state_dict(), "run": run}, f"checkpoints/{ckpt_name(args.out, tag)}.pt")
    pd.DataFrame(log).to_csv(f"{args.out}/train_log_{tag}.tsv", sep="\t", index=False)
    if device.type == "cuda":
        run["gpu"] = torch.cuda.get_device_name(0)
        run["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    run["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run["minutes"] = (time.time() - t0) / 60
    with open(f"{args.out}/run_{tag}.json", "w") as f:
        json.dump(run, f, indent=1)


if __name__ == "__main__":
    main()
