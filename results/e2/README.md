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

## Paired AUROC comparison (added 2026-11-04)

scripts/09_compare_auroc_e2.py on the Mac (about 4 min), from `side_scores.tsv` only.
Difference in AUROC (second minus first) on the same 50 subjects: paired bootstrap over
subjects (10,000 resamples, generator seed 20261104), 95% CI and two-sided P; and
DeLong's test for correlated ROC curves.

| Comparison | AUROC | Difference [95% CI] | P bootstrap | P DeLong |
|---|---|---|---|---|
| B4 clinical rule vs model seed 1 | 0.84 vs 0.54 | +0.30 [+0.06, +0.53] | 0.014 | 0.013 |
| B4 clinical rule vs model seed 2 | 0.84 vs 0.62 | +0.22 [-0.01, +0.44] | 0.061 | 0.060 |
| B1 z-score vs model seed 1 | 0.43 vs 0.54 | -0.11 [-0.29, +0.08] | 0.25 | 0.25 |
| B1 z-score vs model seed 2 | 0.43 vs 0.62 | -0.19 [-0.40, +0.02] | 0.077 | 0.084 |
| B2 RMSE vs model seed 1 | 0.47 vs 0.54 | -0.07 [-0.25, +0.11] | 0.44 | 0.42 |
| B2 RMSE vs model seed 2 | 0.47 vs 0.62 | -0.16 [-0.37, +0.06] | 0.17 | 0.16 |
| B3 PCA vs model seed 1 | 0.54 vs 0.54 | -0.00 [-0.22, +0.22] | 0.99 | 0.98 |
| B3 PCA vs model seed 2 | 0.54 vs 0.62 | -0.09 [-0.31, +0.14] | 0.46 | 0.46 |
| Model seed 2 vs seed 1 | 0.62 vs 0.54 | +0.08 [-0.08, +0.25] | 0.33 | 0.33 |

- The two methods agree. The clinical rule is better than seed 1 (P = 0.014) and
  better than seed 2 by 0.22 with P = 0.06; with 50 subjects the second difference is
  not significant at 0.05. This matches the McNemar results on accuracy above.
- The two seeds do not differ significantly from each other, and the model does not
  differ significantly from the deviation-from-normal baselines B1 to B3.
- Files: `auroc_comparison.tsv`, `auroc_comparison.json`. Test: tests/test_auroc_compare.py.

## Files

`side_scores.tsv` (per subject, every method and side), `flags_stroke.tsv`
(per trial scores and flags), `correction.tsv`, `summary.json`,
`meta_mapping_check.tsv`.
