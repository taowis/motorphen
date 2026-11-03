# E2 plan: paretic-side localisation in stroke survivors

Written 2026-11-03, before any E2 result was seen.

## Question

GenGait was tested only on healthy people imitating abnormal gait. Does the
method point to the right side in real hemiparetic gait, and does it do better
than simple methods that need no deep learning?

## Data

- 50 stroke survivors (Van Criekinge et al. 2023), all walking trials with
  full-body angles (151 trials). Ground truth: paretic side is contralateral to
  the lesion (`sub_char.LesionLeft` in the post-processed MAT file).
- Models: E1 seeds 1 and 2, trained on the 110 able-bodied training subjects
  only. Noise floor and ROM as in E1 (validation and training subjects).
- Normative references for the baselines come from the same 110 training
  subjects. No stroke data is used for fitting anything.

## Mapping MAT subjects to C3D folders

The MAT structure numbers subjects 1..50 (and 1..138). The mapping to the
TVC and SUBJ folders is checked, not assumed: body mass and height in
`sub_char` must match the C3D `PROCESSING` parameters of the matched folder.
Any mismatch is reported and the subject excluded.

## Scores per side

For each trial, Pass 1 gives scores for left and right hip and knee. Side score
= max(hip, knee) for that side; subject side score = mean over the subject's
trials. Predicted paretic side = the side with the higher score.

## Baselines (same side rule)

- B1 z-score: subject mean cycle (right and left foot strikes) of hip flexion,
  hip ab/adduction and knee flexion compared with the normative mean and SD;
  side score = mean absolute z over the cycle and the three angles.
- B2 RMSE to the normative mean, same angles.
- B3 PCA: PCA on per-frame joint angle vectors (12 joints x 3) of training
  subjects, components kept to 95% of variance; side score = mean
  reconstruction error of that side's hip and knee angles.
- B4 clinical rule: smaller peak knee flexion in swing = paretic side.

## Outcomes

- Accuracy over 50 subjects with a Wilson 95% CI; AUROC of (right minus left
  score) for paretic = right, bootstrap 95% CI over subjects (2,000).
- Paired comparison of the model with each baseline: exact McNemar test on
  correct/incorrect per subject.
- Flag rate under the paper's rule (noise floor, K = 2): share of trials with a
  flagged joint on the paretic side, on the non-paretic side, and none.
- Correction, as in the paper but with one value per subject: RMSE of the
  paretic-side hip and knee angles to the normative mean, original against
  reconstructed, Wilcoxon signed-rank over subjects.
- Seeds 1 and 2 are reported side by side; a difference between methods is
  only called a difference if it is larger than the seed-to-seed difference.

## Not in E2

Age matching (E3), window length, K and ankle ablations (E3), Parkinson's
disease (E5).

## Changes after the plan was written (2026-11-03)

- Subject mapping: matching on body mass and height alone was ambiguous, so
  MAT subject i is mapped to the i-th folder in numeric order and each pair is
  checked (47 stroke subjects verified, 3 not checkable and kept; results
  without them are reported). The plan's exclusion rule applies to
  inconsistent pairs; there were none in the stroke group.
- The majority-class rate (64% left-paretic) is reported next to accuracy.
- Trials: 147 (trials with full-body angles of the 50 subjects).
