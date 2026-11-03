# E1 run notes

## 2026-11-03 12:24, first training run froze the machine

- Run: `scripts/02_train_e1.py` with the paper's batch of 256 on Apple MPS,
  two forward passes per step (context-invariance loss).
- Within about 80 seconds free memory fell from 35% to 3% and swap grew by
  about 7 GB (log of the memory monitor that was running for another project).
  The process was killed (exit 137) and the desktop froze. Nothing was lost:
  only results/e1/split.tsv had been written; no checkpoint existed yet.
- Cause: activation memory. 84 tokens x 288 dimensions through 10 layers, twice
  per step, keeps roughly 70 to 100 MB per sample for the backward pass
  (measured on CPU: micro-batch 32 peaks at 2.6 GB, 64 exceeds 3 GB). A batch
  of 256 needs well over 15 GB.
- Changes: gradient accumulation (`--micro-batch 16`, effective batch still 256),
  an in-process memory cap (`src/motorphen/memcap.py`, default 4 GB of RSS plus
  MPS memory, exits with code 3), and MPS watermark ratios 0.3 / 0.25.
- Capped test after the fix: 20 batches of 256 in about 2 minutes, no cap hit.
  Speed is about 4.8 s per batch, 101 batches per epoch (25,760 training
  windows), so 250 epochs would take about 35 hours on this laptop.

## 2026-11-03 13:42 to 15:29, seeds 1 and 2 on the GPU server

- Both seeds completed (exit 0), about 53 minutes each, no monitor pause.
- After the run the monitor did not exit: the remote shell of the ssh command
  used to launch the job stayed alive (orphaned) and its command line contained
  the monitor's match pattern `scripts/02_train_e1.py`. The shell was ended by
  hand at 15:33 and the monitor exited. The pattern is now anchored to the
  python command (`^\.venv/bin/python scripts/02_train_e1\.py`).
