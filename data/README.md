# Data

Raw downloads live in `data/raw/` and are not committed. `data/MANIFEST.tsv`
lists every file with its exact URL, version, access date, md5 and size. To get
everything back:

    ./scripts/00_download.sh
    ./scripts/00_unzip.sh
    .venv/bin/python scripts/01_prepare_vc2023.py

The script checks each md5 and stops on a mismatch.

## Van Criekinge et al. 2023 (healthy adults and stroke survivors)

Van Criekinge T, Saeys W, Truijen S, Vereeck L, Sloot LH, Hallemans A. A
full-body motion capture gait dataset of 138 able-bodied adults across the life
span and 50 stroke survivors. Sci Data 2023;10:852. doi:10.1038/s41597-023-02767-y

- Figshare collection: doi:10.6084/m9.figshare.c.6503791.v1. Every file in the
  collection is released under CC0; the DataCite record of the collection lists
  CC BY 4.0, so the dataset is cited in any work that uses it (checked 2026-11-04).
- Used: the two C3D archives (138_HealthyPiG.zip, 50_StrokePiG.zip), two
  description spreadsheets, and the two post-processed Excel files, which hold
  the subject characteristics (age, sex, lesion side, FAC, POMA, TIS) that the
  C3D files lack (it turned out they do not hold them either).
- The two post-processed MAT files (6.2 GB and 2.9 GB) were added on
  2026-11-03 because subject characteristics (sub_char: age, sex, lesion side,
  FAC, POMA, TIS) are stored only there. Only sub_char is read from them.
- The C3D archives are unzipped to `data/raw/vancriekinge2023/extracted/`
  (stroke subjects come as one zip per subject inside 50_StrokePiG.zip).
- Vicon, 100 Hz, full-body Plug-in Gait, barefoot overground walking at
  self-selected speed. Ages 21-86 (healthy) and 19-85 (stroke).
- Accessed 2026-11-03.

## Shida et al. 2023 (Parkinson's disease, ON and OFF medication)

Shida TKF, Costa TM, de Oliveira CEN, et al. A public data set of walking
full-body kinematics and kinetics in individuals with Parkinson's disease.
Front Neurosci 2023;17:992585. doi:10.3389/fnins.2023.992585

- Figshare: doi:10.6084/m9.figshare.14896881.v4 (CC BY 4.0)
- Used: C3Dfiles.zip, PDGinfo.xlsx (demographics, UPDRS-II/III, Hoehn and Yahr,
  freezing of gait), Supplementary_material.docx. Not downloaded: Gait cycles.zip
  (116 MB, cycle-normalised ASCII; can be added later).
- Accessed 2026-11-03.

## Considered, not used for training

Serrao M et al. Dataset on gait patterns in degenerative neurological diseases.
Data Brief 2017;16:806-816. doi:10.1016/j.dib.2017.12.022. Only per-subject
summary parameters and sagittal range of motion are shared, no time series.
