#!/usr/bin/env bash
# E3c speed and memory test: 20 batches at window 15 and 31 on the GPU server.
# Run from the project directory: setsid -f scripts/remote/bench_e3c.sh > logs/bench_e3c.out 2>&1 < /dev/null
set -uo pipefail
cd "$(dirname "$0")/../.."
for w in 15 31; do
  echo "window $w start $(date '+%F %T')"
  .venv/bin/python scripts/02_train_e1.py --window "$w" --epochs 1 --max-batches 20 --micro-batch 64 \
    --max-mem-gb 12 --out "results/e3/e3c_bench" --tag "w$w" > "logs/bench_e3c_w$w.log" 2>&1
  echo "window $w done $(date '+%F %T') exit $?"
done
touch logs/bench_e3c.done
