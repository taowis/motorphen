"""Subject characteristics (age, sex, lesion side, clinical scores) from the
post-processed MAT files, matched to the C3D subject folders.

Only Data.Sub(i).sub_char is read (h5py, MATLAB 7.3 / HDF5), not the gait data.
The MAT subject index is matched to a folder by body mass and height against
the C3D PROCESSING parameters; the match is checked, not assumed.

Writes data/processed/subject_meta.tsv and results/e2/meta_mapping_check.tsv.
"""

import glob
import os

import ezc3d
import h5py
import numpy as np
import pandas as pd

ROOT = "data/raw/vancriekinge2023"
MATS = {"healthy": f"{ROOT}/MAT_normalizedData_AbleBodiedAdults_v06-03-23.mat",
        "stroke": f"{ROOT}/MAT_normalizedData_PostStrokeAdults_v27-02-23.mat"}
FOLDERS = {"healthy": f"{ROOT}/extracted/138_HealthyPiG_10.05/SUBJ*",
           "stroke": f"{ROOT}/extracted/50_StrokePiG/TVC*"}


def _value(f, obj):
    """Scalar or string from an HDF5 dataset or reference."""
    if isinstance(obj, h5py.Reference):
        obj = f[obj]
    if isinstance(obj, h5py.Dataset):
        a = obj[()]
        if obj.attrs.get("MATLAB_class", b"") == b"char":
            return "".join(chr(c) for c in np.ravel(a))
        a = np.ravel(a)
        if a.dtype == object:
            return _value(f, a[0]) if len(a) else np.nan
        return float(a[0]) if a.size else np.nan
    return np.nan


def read_sub_char(path):
    rows = []
    with h5py.File(path, "r") as f:
        sub = f["Data"]["Sub"]
        refs = np.ravel(sub["sub_char"][()])
        for i, ref in enumerate(refs):
            g = f[ref]
            row = {"mat_index": i + 1}
            for k in g.keys():
                row[k] = _value(f, g[k])
            rows.append(row)
    return pd.DataFrame(rows)


def c3d_anthropometrics(pattern):
    rows = []
    for d in sorted(x for x in glob.glob(pattern) if os.path.isdir(x)):
        for c in sorted(glob.glob(os.path.join(d, "*.c3d"))):
            p = ezc3d.c3d(c)["parameters"].get("PROCESSING", {})
            if "Bodymass" in p and "Height" in p:
                rows.append(dict(subject=os.path.basename(d), bodymass=float(p["Bodymass"]["value"][0]),
                                 height=float(p["Height"]["value"][0])))
                break
    return pd.DataFrame(rows)


def match(meta, anth):
    out = []
    for r in meta.itertuples():
        d = (np.abs(anth.bodymass - r.Weight) / 1.0) + (np.abs(anth.height - r.Height) / 10.0)
        k = int(d.argmin())
        second = float(np.sort(d)[1]) if len(d) > 1 else np.inf
        out.append(dict(mat_index=r.mat_index, subject=anth.subject.iloc[k],
                        d_mass=float(anth.bodymass.iloc[k] - r.Weight),
                        d_height=float(anth.height.iloc[k] - r.Height),
                        next_best=second))
    return pd.DataFrame(out)


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results/e2", exist_ok=True)
    tables, checks = [], []
    for group, mat in MATS.items():
        meta = read_sub_char(mat)
        print(group, "sub_char fields:", [c for c in meta.columns if c != "mat_index"])
        anth = c3d_anthropometrics(FOLDERS[group])
        m = match(meta, anth)
        m["group"] = group
        m["ok"] = (m.d_mass.abs() < 0.5) & (m.d_height.abs() < 5) & ~m.subject.duplicated(keep=False)
        checks.append(m)
        t = meta.merge(m[["mat_index", "subject", "ok"]], on="mat_index")
        t["group"] = group
        tables.append(t)
        print(group, f"{int(m.ok.sum())}/{len(m)} matched unambiguously")
    meta = pd.concat(tables, ignore_index=True)
    if "LesionLeft" in meta:
        meta["paretic"] = np.where(meta.group != "stroke", "",
                                   np.where(meta.LesionLeft == 1, "right",
                                            np.where(meta.LesionLeft == 0, "left", "")))
    meta.loc[~meta.ok, "paretic"] = ""
    meta.to_csv("data/processed/subject_meta.tsv", sep="\t", index=False)
    pd.concat(checks).to_csv("results/e2/meta_mapping_check.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
