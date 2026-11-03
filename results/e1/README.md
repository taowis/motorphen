# E1 results: specificity on held-out healthy adults

Independent reimplementation of the GenGait method (docs/e1-implementation-spec.md),
trained on 110 able-bodied adults from Van Criekinge et al. 2023, tested on 17
held-out able-bodied adults (48 walking trials). Two training seeds.

## Runs

| | seed 1 | seed 2 |
|---|---|---|
| Hardware | GPU server, RTX 5090 | same |
| Code | commit 95f0d35 (server_logs/run_e1.out; `git_commit` in the run JSON is empty because the server copy was not a git checkout, fixed in #11) | same |
| Environment | PyTorch 2.11.0+cu128, Python 3.12.13 | same |
| Data | vc2023.npz md5 fac569287d6aceefac10c59eaedfe0ab | same |
| Training | 250 epochs, batch 256, 53.1 min, peak GPU memory 9.55 GB | 250 epochs, batch 256, 53.2 min |
| Evaluation | scripts/03_eval_e1.py on the Mac, 2026-11-03 | same |

Training logs: `train_log_seed*.tsv` and `server_logs/`. No pause was triggered
by the monitor (host memory stayed above 50% free; the sequencer's basecall
server stayed idle).

## Reconstruction leaves healthy gait unchanged (paper's rule)

Mean paired difference in RMSE to the normative mean, reconstructed minus
original, degrees, with bootstrap 90% CI; equivalence margin 1.5 deg; one value
per subject (n = 17).

| Angle | seed 1 | seed 2 |
|---|---|---|
| Pelvis flexion | +0.03 [-0.01, +0.08] | +0.01 [-0.02, +0.04] |
| Right hip ab/adduction | +0.04 [+0.02, +0.06] | +0.00 [-0.00, +0.01] |
| Right hip flexion | +0.03 [-0.00, +0.08] | +0.00 [-0.03, +0.03] |
| Right knee flexion | -0.57 [-1.12, -0.12] | -0.01 [-0.01, +0.00] |

All within the margin for both seeds, as in the paper (which reported -0.30,
-0.32, -0.82 and +0.24 deg on 10 subjects).

## When two joints are always corrected

| Angle | seed 1 | seed 2 |
|---|---|---|
| Right knee flexion | -1.60 [-2.36, -0.85], not equivalent | -1.31 [-1.98, -0.62], not equivalent |

The other three angles stay within 0.15 deg. A negative difference means the
reconstructed knee is closer to the population mean than the person's own
knee: forced correction pulls healthy individual variation towards the mean,
and an RMSE-to-mean outcome scores that as an improvement.

## Detection flags healthy trials

Share of held-out healthy trials with at least one joint flagged under the
paper's rule (score above the noise floor, at most K = 2):

| | seed 1 | seed 2 |
|---|---|---|
| Trials | 21 / 48 (44%) | 15 / 48 (31%) |
| Subjects with any flag | 8 / 17 | 7 / 17 |

The noise floor is our choice (99th percentile of trial scores in 8
validation subjects; the paper does not say how it was calibrated), so these
rates depend on it. The seed-to-seed difference (44% vs 31%) is the noise
level for this number. The equivalence test above does not show these flags
because the flagged joints barely change the four reported angles.

## Files

- `specificity_seed*.tsv`: per subject and angle, RMSE original / paper rule / top-K.
- `specificity_summary_seed*.json`: equivalence tests.
- `flags_test_healthy_seed*.tsv`: per trial scores and flagged joints.
- `noise_floor_seed*.tsv`, `rom_deg.tsv`, `normative_mean.tsv`: calibration and references.
- `run_seed*.json`, `train_log_seed*.tsv`, `server_logs/`: run records.
- `benchmark_2026-11-03/`: speed test before the runs.
