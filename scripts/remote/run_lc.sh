#!/usr/bin/env bash
# E3d learning curve: 25 and 50 training subjects, two seeds each, equal steps.
# Run from the project directory: setsid -f scripts/remote/run_lc.sh > logs/run_lc.out 2>&1 < /dev/null
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
rm -f logs/run_lc.done
echo "start $(date '+%F %T') commit $(cat COMMIT)"
for spec in "25 1226" "50 570"; do
  set -- $spec
  n=$1; epochs=$2
  for seed in 1 2; do
    echo "n$n seed $seed start $(date '+%F %T') epochs $epochs"
    .venv/bin/python scripts/02_train_e1.py --seed "$seed" --epochs "$epochs" --micro-batch 256 --max-mem-gb 12 \
      --split-file "results/e3/lc_n$n/split.tsv" --out "results/e3/lc_n$n" \
      > "logs/train_lc_n${n}_seed$seed.log" 2>&1
    echo "n$n seed $seed done $(date '+%F %T') exit $?"
  done
done
touch logs/run_lc.done
echo "all done $(date '+%F %T')"
