"""Paired comparison of AUROCs for the paretic side in E2: each model seed against the
clinical rule (B4, lower peak knee flexion in swing) and the other baselines.

Two methods on the same 50 subjects: paired bootstrap over subjects (the AUROC difference
is computed on the same resample for both scores) and DeLong's test for correlated ROC
curves. Reads results/e2/side_scores.tsv only; no model is run.

Usage: .venv/bin/python scripts/09_compare_auroc_e2.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

SRC = Path("results/e2/side_scores.tsv")
OUT = Path("results/e2")
N_BOOT = 10000
MODELS = ["model_seed1", "model_seed2"]
BASELINES = ["kneepeak", "zscore", "rmse", "pca"]


def midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(y, scores):
    """AUROCs and their covariance for several scores on the same subjects (DeLong 1988,
    fast form of Sun and Xu 2014). y is boolean, scores is k x n."""
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([midrank(r) for r in pos])
    ty = np.array([midrank(r) for r in neg])
    tz = np.array([midrank(r) for r in np.hstack([pos, neg])])
    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01).reshape(k, k) / m + np.cov(v10).reshape(k, k) / n
    return auc, cov


def compare(y, a, b, rng):
    """AUROC(b) - AUROC(a): bootstrap CI and P, DeLong z and P."""
    auc_a, auc_b = roc_auc_score(y, a), roc_auc_score(y, b)
    diffs = []
    n = len(y)
    while len(diffs) < N_BOOT:
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    diffs = np.array(diffs)
    p_boot = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    auc, cov = delong(y, np.vstack([a, b]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (auc[1] - auc[0]) / np.sqrt(var)
    return dict(auroc_a=float(auc_a), auroc_b=float(auc_b), diff=float(auc_b - auc_a),
                boot_ci=[float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
                p_boot=float(p_boot), delong_z=float(z), p_delong=float(2 * norm.sf(abs(z))),
                delong_auc_check=[float(auc[0]), float(auc[1])])


def main():
    df = pd.read_csv(SRC, sep="\t")
    y = (df.paretic == "right").to_numpy()
    side = {m: (df[f"{m}_right"] - df[f"{m}_left"]).to_numpy() for m in MODELS + BASELINES}
    assert all(not np.isnan(v).any() for v in side.values()), "missing side scores"
    rng = np.random.default_rng(20261104)
    rows = []
    for model in MODELS:
        for base in BASELINES:
            r = compare(y, side[model], side[base], rng)
            rows.append(dict(a=model, b=base, **r))
    r = compare(y, side["model_seed1"], side["model_seed2"], rng)
    rows.append(dict(a="model_seed1", b="model_seed2", **r))

    tab = pd.DataFrame([{k: v for k, v in r.items() if k != "delong_auc_check"} for r in rows])
    tab["boot_ci_low"] = [r["boot_ci"][0] for r in rows]
    tab["boot_ci_high"] = [r["boot_ci"][1] for r in rows]
    tab = tab.drop(columns="boot_ci")
    tab.to_csv(OUT / "auroc_comparison.tsv", sep="\t", index=False, float_format="%.4f")
    (OUT / "auroc_comparison.json").write_text(json.dumps(
        dict(n_subjects=int(len(y)), paretic_right=int(y.sum()), n_boot=N_BOOT, rng_seed=20261104,
             rows=rows), indent=1))
    for r in rows:
        print(f"{r['b']:<12} - {r['a']:<12} AUROC {r['auroc_b']:.2f} vs {r['auroc_a']:.2f}  "
              f"diff {r['diff']:+.2f} [{r['boot_ci'][0]:+.2f}, {r['boot_ci'][1]:+.2f}]  "
              f"P boot {r['p_boot']:.3f}  DeLong z {r['delong_z']:+.2f} P {r['p_delong']:.3f}")


if __name__ == "__main__":
    main()
