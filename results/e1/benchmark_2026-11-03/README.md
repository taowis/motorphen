# Benchmark, 2026-11-03

20 batches of 256 windows, one epoch cut short, on the GPU server (RTX 5090),
at micro-batch 128 and 256. Run before the project environment existed, with a
borrowed PyTorch 2.11.0+cu128 install and code copied by hand (so `git_commit`
is empty; the code matched commit 1f2c24a plus the CUDA change merged in #5).
Peak GPU memory 4.96 GB and 9.55 GB; about 2.9 s per run including validation.
Checkpoints from these runs were not kept.
