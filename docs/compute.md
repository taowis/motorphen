# Compute

## Local (development)

Mac, Apple silicon, 17 GB unified memory, PyTorch 2.15.0 with MPS
(`requirements.txt`). Fine for tests and short runs; 250-epoch training takes
about 35 hours here (docs/e1-run-notes.md).

## GPU workstation (training)

Lab workstation, AMD Ryzen 9 9950X (16 cores), 60 GB RAM, one NVIDIA RTX 5090
(32 GB), driver 570.153.02 (CUDA 12.8). Shared with a nanopore sequencer
(MinKNOW, dorado basecall server), so check the GPU is idle before a run.

- Project directory: `~/Projects/motorphen-dev` on the workstation.
- Code and processed data are copied with `scripts/remote/sync.sh`, which
  refuses to run with uncommitted code and writes the git commit to `COMMIT`.
- Environment: `scripts/remote/setup_env.sh` builds `.venv` with uv from
  `requirements-gpu.txt`. PyTorch is 2.11.0+cu128 there (the newest build the
  driver supports) against 2.15.0 on the Mac. `results/env_freeze.txt` on the
  workstation records the exact packages; it is copied back with each run.
- Host and user are passed as `REMOTE=user@host`; they are not written in the repo.

Benchmark, 2026-11-03 (20 batches of 256, E1 model): about 0.13 s per batch,
about 15 s per epoch, so about one hour for 250 epochs; peak GPU memory 9.6 GB
at batch 256 and 5.0 GB at micro-batch 128. Run with a borrowed PyTorch
2.11.0+cu128 install before the project environment existed.
