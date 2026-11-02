# Evaluation plan: normative gait model (after Motta et al., GenGait)

GenGait code, weights and data are not released, so the evaluation is done on an
independent reimplementation trained and tested on public motion capture data.
Results speak to the method as described in the paper, not to the authors'
exact software.

## Data

| Dataset | Use | Content | Licence |
|---|---|---|---|
| Van Criekinge et al. 2023, Sci Data 10:852, doi:10.1038/s41597-023-02767-y; Figshare collection doi:10.6084/m9.figshare.c.6503791.v1 | Normative training and healthy test; stroke test set | 138 able-bodied adults aged 21-86, 50 stroke survivors; Vicon 100 Hz, full-body Plug-in Gait, C3D plus stride-normalised MAT/Excel | CC0 |
| Shida et al. 2023, Front Neurosci, doi:10.3389/fnins.2023.992585; Figshare doi:10.6084/m9.figshare.14896881.v4 | Parkinson's disease ON vs OFF medication | 26 people with PD, ON and OFF; C3D, UPDRS-II/III, Hoehn and Yahr, freezing status | CC BY 4.0 |

The stroke set gives a real-patient check the original study lacked, with a known
ground truth (paretic side). The PD set has no healthy controls from the same lab,
so it is used only for within-subject ON vs OFF comparisons.

Not usable for model training: Serrao et al. 2017 (Data in Brief 16:806, ataxia,
HSP, PD) shares only per-subject range-of-motion summaries, not time series.

## Representation

Continuous Plug-in Gait joint angles from C3D, downsampled from 100 Hz to 30 Hz to
match the markerless setting. Joints as in the paper (neck/thorax, shoulders,
elbows, pelvis, hips, knees, ankles). The paper also uses a 6D rotation feature;
whether it can be derived from the Plug-in Gait segment outputs is checked first.

## Experiments

E1. Reimplementation. Masked autoencoder with the published settings (7-frame
window, 8 encoder and 2 decoder layers, 288-d, 12 heads). Healthy held-out
subjects: reconstruction should not move normative gait (equivalence, delta 1.5 deg).

E2. Baselines. Per-joint z-score against an age-matched normative band; PCA
reconstruction; mean trajectory. Compare with the model on paretic-side
localisation in stroke (subject-level AUROC).

E3. Design choices questioned in the letter:
- window length 7 vs 30 frames vs a full cycle
- fixed K = 2 vs a threshold on the anomaly score
- ankle scored vs excluded
- normative training on ages 22-57 only vs the full 21-86 range: false positive
  rate in healthy adults over 60

E4. Statistics. Trial-level Wilcoxon (as in the paper) vs subject-level tests and
mixed-effects models, to show how much repeated trials per person change the p-values.

E5. Parkinson's disease. Within-subject anomaly scores OFF vs ON; association
with UPDRS-III.

All splits are by subject. Every run records seed, data version and code commit.

## Compute

Model is small (a few million parameters); training on the local Mac (Apple
silicon, 17 GB RAM) should be feasible. Environment: Python venv with PyTorch,
ezc3d for C3D, numpy/pandas/scipy/statsmodels.
