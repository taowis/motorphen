#!/usr/bin/env bash
# E3b: train the young (22-57) and all-ages normative models, two seeds each,
# one after the other. Run from the project directory under nohup:
#   setsid nohup scripts/remote/run_e3b.sh > logs/run_e3b.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
rm -f logs/run_e3b.done
echo "start $(date '+%F %T') commit $(cat COMMIT)"
for model in young allages; do
  for seed in 1 2; do
    echo "$model seed $seed start $(date '+%F %T')"
    .venv/bin/python scripts/02_train_e1.py --seed "$seed" --micro-batch 256 --max-mem-gb 12 \
      --split-file "results/e3/$model/split.tsv" --out "results/e3/$model" \
      > "logs/train_e3b_${model}_seed$seed.log" 2>&1
    echo "$model seed $seed done $(date '+%F %T') exit $?"
  done
done
touch logs/run_e3b.done
echo "all done $(date '+%F %T')"
