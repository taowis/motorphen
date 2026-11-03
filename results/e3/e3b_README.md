# E3b results: age range of the normative training data

Plan: docs/e3-plan.md (E3b), written before these results. Two models, each
trained on 52 able-bodied adults (6 validation), two seeds each, on the GPU
server (RTX 5090, about 23 min per run, commit a9aec6f):

- young: all remaining subjects aged 22 to 57 (the paper's range);
- allages: the same number drawn from all remaining subjects (ages 21 to 83,
  10 of them aged 60 or over).

Test: 20 held-out healthy adults aged 64 to 85 and 16 aged 23 to 55
(`age_test_set.tsv`); 50 stroke survivors. Evaluated on the Mac with
scripts/03_eval_e1.py and scripts/05_eval_e2.py (outputs in `young/` and
`allages/`).

## Healthy trials flagged (paper's rule: above the noise floor, K = 2)

| Model | seed | 60+ trials flagged | 22-57 trials flagged |
|---|---|---|---|
| young | 1 | 70% | 60% |
| young | 2 | 73% | 79% |
| allages | 1 | 36% | 63% |
| allages | 2 | 57% | 74% |

Young minus all-ages, mean per-subject share of flagged trials, bootstrap 95%
CI over subjects:

| seed | 60+ band | 22-57 band |
|---|---|---|
| 1 | +0.34 [+0.17, +0.53] | -0.09 [-0.30, +0.13] |
| 2 | +0.16 [+0.02, +0.33] | +0.08 [-0.12, +0.29] |

As stated in advance: the young-only reference flags more healthy older
adults, and the two models are similar in the younger band. The size of the
effect differs between seeds (0.34 vs 0.16).

With 52 training subjects, flag rates in healthy adults (36 to 79%) are well
above the E1 rates with 110 subjects (31 to 44%); E3d tests training size
directly.

## Specificity (equivalence within 1.5 deg, paper's rule)

Most angle and band combinations stay within the margin; right hip flexion
in the 22-57 band falls outside it in three of four runs (mean -0.84 to
-0.98 deg) and right hip ab/adduction in one run per model. Forced correction
(K = 2 always) moves the right knee by -0.70 to -1.73 deg. Details in
`*/specificity_summary_seed*.json`.

## Stroke survivors (paretic side)

| Model | seed 1 AUROC | seed 2 AUROC |
|---|---|---|
| E1 model (110 subjects) | 0.54 | 0.62 |
| young (52) | 0.34 [0.19, 0.52] | 0.36 [0.20, 0.51] |
| allages (52) | 0.24 [0.11, 0.39] | 0.36 [0.21, 0.51] |
| clinical rule, lower peak knee flexion | 0.84 | 0.84 |

With the smaller training sets the model points to the non-paretic side more
often than to the paretic side (AUROC below 0.5; for allages seed 1 the CI
excludes 0.5). Flags appear on nearly every stroke trial (99 to 100%), on the
non-paretic side (70 to 74%) as often as or more often than on the paretic
side (63 to 69%). An age-matched reference does not help: stroke survivors
are 64 years old on average, and the all-ages model does no better.

## Reading

The age range of the normative reference affects false flags in older
healthy adults, which matters for any clinical use in older patients, but it
does not explain the failure to localise the paretic side.

## Files

`age_test_set.tsv`, `young/` and `allages/` (split, run records, training
logs, specificity, flags, noise floor, ROM, normative mean, `e2/`),
`server_logs/` (run_e3b.out, monitor log, training logs).
