"""Subject characteristics (age, sex, lesion side, clinical scores) from the
post-processed MAT files, matched to the C3D subject folders.

Only Sub(i).sub_char is read (h5py, MATLAB 7.3 / HDF5), not the gait data.
MAT subject i is mapped to the i-th folder in numeric order, and every pair is
checked against the C3D PROCESSING body mass and height (see match()).

Writes data/processed/subject_meta.tsv and results/e2/meta_mapping_check.tsv.
"""

import glob
import os
import re

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
        # the description file says Data.Sub; in the files the top-level variable is Sub
        sub = f["Sub"] if "Sub" in f else f["Data"]["Sub"]
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


def folder_number(name):
    return int(re.sub(r"\D", "", name))


def match(meta, pattern, anth):
    """MAT subject i is taken to be the i-th folder in numeric order (SUBJ1..138,
    TVC03..TVC60); each pair is then checked against the C3D anthropometrics.

    check = verified (mass within 0.5 kg and height within 5 mm),
    inconsistent, or not_checkable (no PROCESSING parameters in the C3D files)."""
    folders = sorted((os.path.basename(d) for d in glob.glob(pattern) if os.path.isdir(d)), key=folder_number)
    if len(folders) != len(meta):
        raise SystemExit(f"{len(folders)} folders but {len(meta)} MAT subjects")
    a = anth.set_index("subject")
    out = []
    for r, subject in zip(meta.itertuples(), folders):
        row = dict(mat_index=r.mat_index, subject=subject, d_mass=np.nan, d_height=np.nan, check="not_checkable")
        if subject in a.index:
            row["d_mass"] = float(a.bodymass[subject] - r.Weight)
            row["d_height"] = float(a.height[subject] - r.Height)
            ok = abs(row["d_mass"]) <= 0.5 + 1e-6 and abs(row["d_height"]) <= 5
            row["check"] = "verified" if ok else "inconsistent"
        out.append(row)
    return pd.DataFrame(out)


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results/e2", exist_ok=True)
    tables, checks = [], []
    for group, mat in MATS.items():
        meta = read_sub_char(mat)
        print(group, "sub_char fields:", [c for c in meta.columns if c != "mat_index"])
        anth = c3d_anthropometrics(FOLDERS[group])
        m = match(meta, FOLDERS[group], anth)
        m["group"] = group
        checks.append(m)
        t = meta.merge(m[["mat_index", "subject", "check"]], on="mat_index")
        t["group"] = group
        tables.append(t)
        print(group, m.check.value_counts().to_dict())
    meta = pd.concat(tables, ignore_index=True)
    if "LesionLeft" in meta:
        meta["paretic"] = np.where(meta.group != "stroke", "",
                                   np.where(meta.LesionLeft == 1, "right",
                                            np.where(meta.LesionLeft == 0, "left", "")))
    meta.loc[meta.check == "inconsistent", "paretic"] = ""
    meta.to_csv("data/processed/subject_meta.tsv", sep="\t", index=False)
    pd.concat(checks).to_csv("results/e2/meta_mapping_check.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
