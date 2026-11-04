# E3c results: window length

Plan: docs/e3-plan.md (E3c), written before these results. The 7-frame window
(0.23 s) of the paper covers about a quarter of a gait cycle; 15 and 31 frames
are about 0.5 s and 1 s. E1 split (110 training, 8 validation, 17 test healthy
subjects); 50 stroke survivors. E1 settings otherwise (batch 256, micro-batch
64, 250 epochs), two seeds per window. The 7-frame point is E1.

Training on the GPU server (RTX 5090, PyTorch 2.11.0+cu128, commit 9b688b2,
data md5 fac569287d6aceefac10c59eaedfe0ab), 2026-11-03 23:19 to 2026-11-04
10:39; 15- and 31-frame runs in two parallel streams. 3.7 to 3.9 h per run,
7.6 h for 31 frames seed 1 while it shared the GPU; peak GPU memory 5.2 GB
(15 frames) and 10.6 GB (31 frames). No monitor pause; GPU temperature at most
69 C. Evaluation on the Mac (scripts/03_eval_e1.py, scripts/05_eval_e2.py),
2026-11-04.

Evaluation change: the first stroke evaluation with 15 frames ran out of MPS
memory under the 0.3 watermark cap. `predict_last` now sets its batch by window
length (1024 windows at 7 frames, unchanged; 104 at 15; 11 at 31). Batching
does not change the predictions.

| Window | Stroke AUROC, seed 1 / 2 | Mean | Healthy trials flagged, seed 1 / 2 | Equivalent angles (paper's rule) | Forced K = 2, right knee (deg) |
|---|---|---|---|---|---|
| 7 | 0.54 [0.38, 0.70] / 0.62 [0.46, 0.78] | 0.58 | 44% / 31% | 4/4, 4/4 | -1.60 / -1.31 |
| 15 | 0.73 [0.58, 0.88] / 0.58 [0.43, 0.75] | 0.66 | 19% / 52% | 4/4, 4/4 | -0.52 / -1.08 |
| 31 | 0.56 [0.36, 0.74] / 0.41 [0.23, 0.60] | 0.49 | 21% / 33% | 4/4, 4/4 | -0.90 / -0.95 |

Clinical rule (lower peak knee flexion in swing): AUROC 0.84 [0.70, 0.95].
Exact McNemar, clinical rule against the model: 15 frames p = 0.21 / 0.029;
31 frames p = 0.003 / 0.001.

Stroke flags (any / paretic side / non-paretic side / paretic side only):
15 frames 94-98% / 77-86% / 67-69% / 26-28%; 31 frames 88-93% / 62-72% /
61-78% / 14-19%.

## Reading

- Stroke AUROC does not rise with window length: mean 0.58, 0.66 and 0.49 for
  7, 15 and 31 frames. The 15-frame gain (+0.08) equals the seed-to-seed
  difference at 7 frames (0.08) and is smaller than the one at 15 frames
  (0.15). By the comparison rule fixed in the plan, window length makes no
  difference to localisation.
- One 15-frame run has a CI that excludes 0.5 (0.73); the other seed does not
  reproduce it. A single run should not be read as an effect.
- The one-variable clinical rule stays ahead at every window, by a clear
  margin with 31 frames.
- Healthy flag rates vary between seeds (19 to 52%) as much as between
  windows; no trend.
- Forced correction moves the healthy right knee less with longer windows
  (about -0.5 to -1.1 deg against -1.3 to -1.6 at 7 frames), and with 31 frames
  the stroke flags stay on both legs at similar rates.
- The short window of the paper does not explain the failure to localise the
  paretic side.

## Files

`e3c_w15/`, `e3c_w31/` (split, run records, training logs, specificity,
flags, noise floor, ROM, normative mean, `e2/`), `server_logs/e3c/`
(run_e3c.out, monitor log with GPU temperature and power, training logs),
`e3c_bench/` (speed test before the runs).
