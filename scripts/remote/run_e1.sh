#!/usr/bin/env bash
# Train E1 for seeds 1 and 2, one after the other, on the GPU server.
# Run from the project directory under nohup:
#   nohup scripts/remote/run_e1.sh > logs/run_e1.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
echo "start $(date '+%F %T') commit $(cat COMMIT)"
for seed in 1 2; do
  echo "seed $seed start $(date '+%F %T')"
  .venv/bin/python scripts/02_train_e1.py --seed "$seed" --micro-batch 256 --max-mem-gb 12 \
    > "logs/train_seed$seed.log" 2>&1
  rc=$?
  echo "seed $seed done $(date '+%F %T') exit $rc"
done
touch logs/run_e1.done
echo "all done $(date '+%F %T')"
