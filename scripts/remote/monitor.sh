#!/usr/bin/env bash
# Log host memory and GPU state every 30 s while training runs on the shared
# GPU server. Pauses training (SIGSTOP) when host memory is short, when the
# sequencer's basecall server starts using more GPU memory, or when the GPU is
# hotter than TEMP_PAUSE (85 C); resumes when all are clear and the GPU has
# cooled below TEMP_RESUME (75 C). Logs GPU temperature and power each cycle.
# Run from the project directory: DONE=logs/run_e1.done nohup scripts/remote/monitor.sh > /dev/null 2>&1 &
# DONE is the marker file the run script writes when it finishes.
set -u
cd "$(dirname "$0")/../.."
LOG=${LOG:-logs/monitor.log}
DONE=${DONE:-logs/run_e1.done}
# anchored to the python command: a shell whose command line merely mentions the
# script (e.g. an ssh launch command) must not count as a running job
PATTERN=${PATTERN:-'^\.venv/bin/python scripts/02_train_e1\.py'}
CRIT=${CRIT_PCT:-12}; RESUME=${RESUME_PCT:-30}; BASECALL_MARGIN_MIB=${BASECALL_MARGIN_MIB:-1024}
TEMP_PAUSE=${TEMP_PAUSE:-85}; TEMP_RESUME=${TEMP_RESUME:-75}; INTERVAL=${INTERVAL:-30}
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
  if [[ -z "$pids" && -e "$DONE" ]]; then
    echo "$(date '+%F %T') training finished, monitor exits" >> "$LOG"; exit 0
  fi
  if [[ -z "$pids" ]]; then idle=$((idle + 1)); else idle=0; fi
  if (( idle >= 20 )); then
    echo "$(date '+%F %T') no training process for 10 min, monitor exits" >> "$LOG"; exit 0
  fi
  avail=$(free | awk '/Mem:/ {printf "%d", $7 * 100 / $2}')
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ')
  bc=$(basecall_mib)
  read -r temp power < <(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits \
    | head -1 | awk -F', ' '{printf "%d %.0f", $1, $2}')
  temp=${temp:-0}; power=${power:-0}
  state=ok
  reason=""
  if (( avail < CRIT )); then reason="host memory ${avail}% free"; fi
  if (( bc > base + BASECALL_MARGIN_MIB )); then reason="${reason:+$reason; }basecalling active (${bc} MiB)"; fi
  if (( temp > TEMP_PAUSE )); then reason="${reason:+$reason; }GPU ${temp} C above ${TEMP_PAUSE} C"
  elif (( paused == 1 && temp >= TEMP_RESUME )); then reason="${reason:+$reason; }GPU cooling, ${temp} C"; fi
  if [[ -n "$reason" && -n "$pids" && $paused -eq 0 ]]; then
    kill -STOP $pids; paused=1; state=PAUSE
  elif [[ -z "$reason" && $paused -eq 1 && $avail -ge $RESUME ]]; then
    kill -CONT $pids 2>/dev/null || true; paused=0; state=RESUME
  elif (( avail < 25 )); then
    state=WARN
  fi
  echo "$(date '+%F %T') $state host_free=${avail}% gpu_util,used,total=${gpu} temp=${temp}C power=${power}W basecall=${bc}MiB paused=$paused ${reason}" >> "$LOG"
  sleep "$INTERVAL"
done
