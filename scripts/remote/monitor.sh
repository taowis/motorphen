#!/usr/bin/env bash
# Log host memory and GPU state every 30 s while training runs on the shared
# GPU server. Pauses training (SIGSTOP) when host memory is short or when the
# sequencer's basecall server starts using more GPU memory, resumes after.
# Run from the project directory: nohup scripts/remote/monitor.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
LOG=logs/monitor.log
PATTERN='scripts/02_train_e1.py'
CRIT=${CRIT_PCT:-12}; RESUME=${RESUME_PCT:-30}; BASECALL_MARGIN_MIB=${BASECALL_MARGIN_MIB:-1024}
basecall_mib() {
  nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '/dorado/ {s += $2} END {print s + 0}'
}
base=$(basecall_mib)
paused=0
idle=0
echo "$(date '+%F %T') monitor start, basecall GPU memory baseline ${base} MiB" >> "$LOG"
while true; do
  pids=$(pgrep -f "$PATTERN" || true)
  if [[ -z "$pids" && -e logs/run_e1.done ]]; then
    echo "$(date '+%F %T') training finished, monitor exits" >> "$LOG"; exit 0
  fi
  if [[ -z "$pids" ]]; then idle=$((idle + 1)); else idle=0; fi
  if (( idle >= 20 )); then
    echo "$(date '+%F %T') no training process for 10 min, monitor exits" >> "$LOG"; exit 0
  fi
  avail=$(free | awk '/Mem:/ {printf "%d", $7 * 100 / $2}')
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ')
  bc=$(basecall_mib)
  state=ok
  reason=""
  if (( avail < CRIT )); then reason="host memory ${avail}% free"; fi
  if (( bc > base + BASECALL_MARGIN_MIB )); then reason="${reason:+$reason; }basecalling active (${bc} MiB)"; fi
  if [[ -n "$reason" && -n "$pids" && $paused -eq 0 ]]; then
    kill -STOP $pids; paused=1; state=PAUSE
  elif [[ -z "$reason" && $paused -eq 1 && $avail -ge $RESUME ]]; then
    kill -CONT $pids 2>/dev/null || true; paused=0; state=RESUME
  elif (( avail < 25 )); then
    state=WARN
  fi
  echo "$(date '+%F %T') $state host_free=${avail}% gpu_util,used,total=${gpu} basecall=${bc}MiB paused=$paused ${reason}" >> "$LOG"
  sleep 30
done
