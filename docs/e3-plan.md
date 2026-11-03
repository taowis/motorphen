# E3 plan: do the design choices explain the E1/E2 results?

Written 2026-11-03, after E1 and E2 and before any E3 result.

## E3a: inference choices (no retraining)

Models: E1 seeds 1 and 2. Same test sets as E1 (17 held-out healthy) and E2
(50 stroke).

| Choice | Variants | Paper |
|---|---|---|
| Number of corrected joints | K = 1, 2, 3; no cap (all joints above the noise floor) | K = 2 |
| Ankles | scored and usable in the side rule, or not | not scored |
| Peak statistic | 95th percentile, maximum, mean | "peak statistic" |

Outcomes per variant: healthy flag rate (share of trials with any flag);
E2 side AUROC and accuracy; E2 share of trials flagged on the paretic side
only. K only changes flags and correction, not the side scores.

## E3b: age range of the normative training data (retraining)

Subjects: 124 able-bodied adults with full-body angles and an age that passed
the mapping check (11 with inconsistent anthropometrics excluded).

- Held-out test set, fixed first: 20 subjects aged 60 or over and 16 aged 22
  to 57, drawn at random within each band (seed 20261104).
- Young model: trained on the remaining subjects aged 22 to 57 (the paper's
  range), all of them: 6 validation, the rest training.
- All-ages model: the same numbers drawn at random
  from all remaining subjects regardless of age, so training size is equal.
- Two seeds each, E1 settings otherwise (7-frame window, batch 256, 250 epochs),
  on the GPU server.

Outcomes:
- Healthy flag rate by test age band (under 60 / 60 and over), per model;
  difference with a bootstrap 95% CI over subjects.
- E1 equivalence test separately in each age band.
- E2 side AUROC and flags in the 50 stroke survivors (mean age 64) with each
  model.
- Expectation stated in advance: if a young-only reference contributes to the
  false flags, the young model should flag more trials in the 60+ band than
  the all-ages model, with the 22 to 57 band similar for both.

## E3c: window length (retraining), decided after a speed test

The 7-frame window (0.23 s) covers about a quarter of a gait cycle. Candidate
windows: 15 and 31 frames (about 0.5 s and 1 s, one cycle). Tokens grow from
84 to 180 and 372, and attention cost grows with the square, so the number of
runs is set after measuring speed and memory on the server. Same E1 split and
outcomes as E1 and E2.

## Comparison rule

As before, a difference between variants is only called a difference when it
is larger than the seed-to-seed difference of the reference setting.
