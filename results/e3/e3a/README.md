# E3a results: inference choices (no retraining)

Plan: docs/e3-plan.md. Models: E1 seeds 1 and 2. Healthy: 17 held-out subjects
(48 trials), noise floor from the 8 validation subjects (99th percentile of
trial scores, recalibrated for each variant). Stroke: 50 subjects (147
trials). scripts/07_eval_e3a.py on the Mac, 2026-11-03.

The default row (95th percentile, ankles not scored, K = 2) reproduces the E1
and E2 numbers exactly (44% / 31% healthy flags; AUROC 0.54 / 0.62; 21% / 25%
paretic-side-only flags).

## Healthy trials flagged (any joint above the noise floor)

| Peak statistic | Ankles | seed 1 | seed 2 |
|---|---|---|---|
| 95th percentile | no | 44% (8/17 subjects) | 31% (7/17) |
| 95th percentile | yes | 60% (11/17) | 62% (12/17) |
| maximum | no | 40% (7/17) | 31% (8/17) |
| maximum | yes | 44% (8/17) | 44% (11/17) |
| mean | no | 35% (8/17) | 21% (5/17) |
| mean | yes | 60% (12/17) | 58% (11/17) |

## Paretic side in stroke (AUROC of right-minus-left side score)

| Peak statistic | Ankles | seed 1 | seed 2 |
|---|---|---|---|
| 95th percentile | no | 0.54 | 0.62 |
| 95th percentile | yes | 0.52 | 0.58 |
| maximum | no | 0.41 | 0.51 |
| maximum | yes | 0.44 | 0.51 |
| mean | no | 0.54 | 0.58 |
| mean | yes | 0.50 | 0.56 |

For comparison, the clinical rule (lower peak knee flexion in swing) reached
0.84 (results/e2).

## Stroke trials flagged on the paretic side only, by K (95th percentile)

| K | Ankles no, seed 1 / 2 | Ankles yes, seed 1 / 2 |
|---|---|---|
| 1 | 41% / 52% | 47% / 57% |
| 2 | 21% / 25% | 27% / 29% |
| 3 | 10% / 16% | 12% / 12% |
| no cap | 6% / 15% | 7% / 11% |

## Reading

- Scoring the ankles adds healthy flags (about 60% of trials) and does not
  improve localisation, so the exclusion of the ankles is not why the model
  misses the paretic side.
- No peak statistic gives an AUROC clearly above 0.5; the differences between
  statistics are within the seed-to-seed difference.
- With K = 1 the single top joint is on the paretic side alone in about half
  of the stroke trials; larger K mostly adds joints on the other side.
- None of these inference choices explains the E2 result.

## Files

`trial_peaks.tsv` (per trial and joint: 95th percentile, max and mean of
B_j(t)), `healthy_flags.tsv`, `stroke_side.tsv`, `stroke_flags_by_k.tsv`.
