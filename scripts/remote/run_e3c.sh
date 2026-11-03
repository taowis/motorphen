#!/usr/bin/env bash
# E3c: windows of 15 and 31 frames, two seeds each, E1 split. The two windows
# run side by side (about 5 and 11 GB of GPU memory), seeds one after the other.
# Run from the project directory: setsid -f scripts/remote/run_e3c.sh > logs/run_e3c.out 2>&1 < /dev/null
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
rm -f logs/run_e3c.done
echo "start $(date '+%F %T') commit $(cat COMMIT)"
stream() {
  local w=$1
  for seed in 1 2; do
    echo "w$w seed $seed start $(date '+%F %T')"
    .venv/bin/python scripts/02_train_e1.py --window "$w" --seed "$seed" --micro-batch 64 --max-mem-gb 12 \
      --split-file results/e1/split.tsv --out "results/e3/e3c_w$w" > "logs/train_e3c_w${w}_seed$seed.log" 2>&1
    echo "w$w seed $seed done $(date '+%F %T') exit $?"
  done
}
stream 15 &
stream 31 &
wait
touch logs/run_e3c.done
echo "all done $(date '+%F %T')"
