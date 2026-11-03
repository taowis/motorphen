# E3d results: learning curve (amount of normative training data)

Plan: docs/e3-plan.md (E3d), written before these results. Nested random
subsets of the E1 training set (25 inside 50 inside 110 subjects); E1
validation (8) and test (17 healthy) subjects; 50 stroke survivors. Equal
optimiser steps for every size (1226, 570 and 250 epochs), curriculum
stretched in proportion. Two seeds per size. The 110-subject point is E1.
Training on the GPU server (about 55 min per run, commit 640404c); evaluation
on the Mac (scripts/03_eval_e1.py, scripts/05_eval_e2.py).

| Training subjects | Stroke AUROC, seed 1 / 2 | Mean | Healthy trials flagged, seed 1 / 2 | Equivalent angles (paper's rule) |
|---|---|---|---|---|
| 25 | 0.40 [0.25, 0.56] / 0.42 [0.26, 0.58] | 0.41 | 50% / 52% | 4/4, 4/4 |
| 50 | 0.48 [0.31, 0.65] / 0.46 [0.30, 0.62] | 0.47 | 35% / 58% | 4/4, 3/4 |
| 110 | 0.54 [0.38, 0.70] / 0.62 [0.46, 0.78] | 0.58 | 44% / 31% | 4/4, 4/4 |

Clinical rule (lower peak knee flexion in swing): AUROC 0.84 [0.70, 0.95].

## Reading

- Stroke AUROC rises with training size, by about 0.17 from 25 to 110
  subjects, which is more than the seed-to-seed difference at any size (0.02
  to 0.08). By the rule fixed in the plan, the amount of normative data does
  matter.
- At 110 subjects every CI still includes 0.5, and the rule-based baseline is
  far ahead. Three points do not support an extrapolation of how much data
  would be needed.
- Healthy flag rates show no trend (31 to 58%); their variation is larger
  than any size effect.
- Which subjects are in the training set matters more than the seed: the
  E3b models, also trained on about 50 subjects but a different set, gave
  AUROC 0.24 to 0.36, against 0.46 to 0.48 here. Single runs should not be
  read as the performance of a training size.
- Forced correction (K = 2 always) moves the healthy right knee by -1.1 to
  -1.6 deg at every size.

## Files

`lc_n25/` and `lc_n50/` (split, run records, training logs, specificity,
flags, noise floor, `e2/`); E1 files for the 110-subject point;
`server_logs/` (run_lc.out, monitor log, training logs).
