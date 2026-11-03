# Summary of E1 and E2 (2026-11-03)

Independent evaluation of the GenGait method (Motta et al., Artificial
Intelligence in Medicine 2027, doi:10.1016/j.artmed.2026.103545). Details:
results/e1/README.md and results/e2/README.md.

## Setup

GenGait's code and data are not public, so the method was reimplemented from
the paper: a transformer masked autoencoder (8 + 2 layers, d = 288, 7-frame
windows, 12 joints) with its two-pass inference and badness score. It was
trained on 110 able-bodied adults from a public motion capture dataset (Van
Criekinge et al., Sci Data 2023), with two training seeds. Details the paper
leaves open (noise-floor calibration, peak statistic, ROM values, axis
weights) are our choices, listed in docs/e1-implementation-spec.md.

## E1: held-out healthy adults (n = 17)

- Under the paper's selection rule, reconstruction left all four reported
  angles (pelvis flexion, right hip ab/adduction, right hip flexion, right
  knee flexion) within the +/-1.5 deg equivalence margin in both seeds, as the
  paper reported.
- When two joints are always corrected, the healthy right knee is pulled
  towards the population mean by 1.3 to 1.6 deg, outside the margin. An
  RMSE-to-mean outcome scores this as an improvement.
- Detection flagged at least one joint in 31 to 44% of healthy trials (7 to 8
  of 17 people). This rate depends on our noise-floor choice, and the
  equivalence test does not show it.

## E2: real stroke survivors (n = 50, paretic side known)

- The model did not localise the paretic side better than chance: AUROC 0.54
  [0.38, 0.70] and 0.62 [0.46, 0.78] for the two seeds; accuracy 0.52 to 0.66
  against a majority-class rate of 0.64.
- A one-line clinical rule, lower peak knee flexion in swing, reached AUROC
  0.84 [0.70, 0.95] and accuracy 0.80.
- z-score, RMSE-to-mean and PCA baselines also failed (AUROC 0.43 to 0.54).
- 90 to 94% of stroke trials were flagged, at similar rates on the paretic
  (74 to 78%) and non-paretic (65 to 72%) sides.
- Reconstruction still lowered paretic-side hip and knee flexion RMSE to the
  normative mean (several p < 0.05), which, given E1, is not evidence of
  correct localisation.

## Reading

The method reproduces its reported behaviour on healthy gait, but on real
hemiparetic gait it flags both legs, does not identify the affected side and
is outperformed by a simple clinical rule. Lower RMSE to the population mean
is not a valid outcome for joint-level detection.

## Caveats

- Independent reimplementation, not the authors' software; Vicon motion
  capture rather than a markerless system.
- Small samples (17 and 50 people), wide confidence intervals, noticeable
  seed-to-seed differences.
- Several implementation details are our choices.
- E3 (window length, K, ankle exclusion, normative age range) not yet run.
