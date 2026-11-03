# E2 results: paretic-side localisation in 50 stroke survivors

Plan: docs/e2-plan.md (written before these results). Models: E1 seeds 1 and 2,
trained on 110 able-bodied adults only. Run on the Mac, 2026-11-03,
scripts/05_eval_e2.py; subject characteristics from scripts/04_subject_meta.py.

## Data and ground truth

- 50 stroke survivors, 147 walking trials with full-body angles.
- Paretic side = contralateral to the lesion (`LesionLeft` in the MAT file):
  left 32, right 18. Always answering "left" would be 64% accurate, so
  accuracy is read against 0.64 and AUROC is the main measure.
- MAT subject i was mapped to the i-th TVC folder in numeric order. Checked
  against the C3D body mass and height: 47 verified (height identical, mass
  within 0.5 kg), 3 not checkable (no anthropometrics in their C3D files),
  none inconsistent (`meta_mapping_check.tsv`).

## Which side is paretic?

One value per subject; accuracy with Wilson 95% CI, AUROC of right-minus-left
score with bootstrap 95% CI (2,000 resamples).

| Method | Accuracy | AUROC |
|---|---|---|
| Model, seed 1 | 0.52 [0.39, 0.65] | 0.54 [0.38, 0.70] |
| Model, seed 2 | 0.66 [0.52, 0.78] | 0.62 [0.46, 0.78] |
| B1 z-score to normative band | 0.54 [0.40, 0.67] | 0.43 [0.27, 0.61] |
| B2 RMSE to normative mean | 0.56 [0.42, 0.69] | 0.47 [0.29, 0.64] |
| B3 PCA reconstruction error | 0.56 [0.42, 0.69] | 0.54 [0.38, 0.70] |
| B4 lower peak knee flexion in swing | 0.80 [0.67, 0.89] | 0.84 [0.70, 0.95] |

- Neither model seed separates the paretic side better than chance (AUROC CIs
  include 0.5), and the two seeds differ by 0.08 in AUROC and 0.14 in accuracy.
- The one-line clinical rule (B4) does clearly better. Paired exact McNemar:
  B4 vs seed 1 p = 0.020, B4 vs seed 2 p = 0.23.
- The three deviation-from-normal baselines also fail, with AUROC below 0.5
  for B1 and B2: the non-paretic leg deviates from normal gait as much as or
  more than the paretic leg in these angles. A plausible reading is
  compensation by the non-paretic side; this was not tested here.
- Restricting to the 47 verified subjects changes no number by more than 0.02.

## Flags under the paper's rule (noise floor, K = 2)

| | seed 1 | seed 2 |
|---|---|---|
| Trials with any flag | 94% | 90% |
| Flag on the paretic side | 74% | 78% |
| Flag on the non-paretic side | 72% | 65% |
| Paretic side only | 21% | 25% |

The method flags stroke gait as abnormal almost always, but on both legs at
similar rates.

## Correction (paper's sensitivity outcome, one value per subject)

RMSE of the paretic-side angle's mean cycle to the normative mean, original
then reconstructed; Wilcoxon signed-rank over subjects (left-paretic n = 32,
right-paretic n = 18).

| Paretic side, angle | seed 1 | seed 2 |
|---|---|---|
| Left hip flexion | 12.75 -> 9.94, p < 0.001 | 12.75 -> 9.32, p < 0.001 |
| Left hip ab/adduction | 5.54 -> 5.23, p = 0.33 | 5.54 -> 5.11, p = 0.27 |
| Left knee flexion | 14.71 -> 12.20, p = 0.001 | 14.71 -> 13.34, p = 0.99 |
| Right hip flexion | 10.99 -> 9.94, p = 0.25 | 10.99 -> 9.50, p = 0.027 |
| Right hip ab/adduction | 4.55 -> 4.86, p = 0.13 | 4.55 -> 5.19, p = 0.30 |
| Right knee flexion | 13.27 -> 11.32, p = 0.024 | 13.27 -> 11.55, p = 0.030 |

Reconstruction moves paretic-side hip and knee flexion towards the normative
mean, as the paper reported for simulated gait. Read with E1: the same
operation also moves healthy knees towards the mean when forced, so a lower
RMSE to the mean is not by itself evidence of correct localisation.

## Files

`side_scores.tsv` (per subject, every method and side), `flags_stroke.tsv`
(per trial scores and flags), `correction.tsv`, `summary.json`,
`meta_mapping_check.tsv`.
