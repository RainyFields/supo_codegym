#!/bin/bash
# Keep one arm alive across queue reclamations (both ark queues stop pods on a ~4 h grid).
# Submits on H100 first; if the trial is still `queued` after PEND_TIMEOUT_MIN, stops it and
# resubmits on A100 (user rule: H100 first, A100 fallback). Any non-success exit without a DONE
# marker is resubmitted (verl resumes from the last HDFS checkpoint). Run under setsid/tmux.
#   bash scripts/merlin/resubmit_loop.sh supo [run_tag]
set -uo pipefail
ARM=${1:?grpo|supo}; RUN_TAG=${2:-}
MAX_RESUBMITS=${MAX_RESUBMITS:-12}
PEND_TIMEOUT_MIN=${PEND_TIMEOUT_MIN:-60}
POLL_S=${POLL_S:-300}
M=${MERLIN_CLI:-$HOME/.merlin-cli/bin/merlin-cli}; CP=i18n-tt
DIR=$(cd "$(dirname "$0")/../.." && pwd)
RUNS=/mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym/job-runs/${ARM}${RUN_TAG}
GPU=${GPU_FIRST:-h100}
n=0
while [ $n -lt $MAX_RESUBMITS ]; do
  [ -f "$RUNS/DONE" ] && { echo "[loop] DONE marker present, exiting"; exit 0; }
  SPEC=$DIR/jobs/${ARM}${RUN_TAG}_${GPU}.json
  python3 $DIR/scripts/merlin/make_job_spec.py --arm $ARM --gpu $GPU --run_tag "$RUN_TAG" > "$SPEC"
  echo "[loop] submit #$n on $GPU $(TZ=America/Los_Angeles date)"
  OUT=$(bash $DIR/scripts/merlin/submit.sh "$SPEC") || { echo "[loop] submit failed"; sleep $POLL_S; n=$((n+1)); continue; }
  SID=$(echo "$OUT" | sed -n 's/.*sid=\([0-9a-f]*\) .*/\1/p' | tail -1)
  echo "[loop] sid=$SID"
  t_sub=$(date +%s)
  while true; do
    sleep $POLL_S
    J=$($M --control-plane $CP job-v2 runs get --json "{\"sid\":\"$SID\"}" 2>/dev/null)
    ST=$(echo "$J" | python3 -c "import sys,json; raw=sys.stdin.read(); i=raw.find('{'); d=json.loads(raw[i:])['data']; print(d['status'], d.get('runtime_context',{}).get('trial_status',''))" 2>/dev/null)
    status=${ST%% *}; trial=${ST##* }
    echo "[loop] $(TZ=America/Los_Angeles date '+%H:%M') status=$status trial=$trial"
    if [ "$trial" = queued ] && [ $(( ($(date +%s) - t_sub) / 60 )) -ge $PEND_TIMEOUT_MIN ]; then
      echo "[loop] pending > ${PEND_TIMEOUT_MIN}min on $GPU -> stop and fall back"
      $M --control-plane $CP job-v2 runs stop --json "{\"sid\":\"$SID\"}" >/dev/null 2>&1
      GPU=$([ "$GPU" = h100 ] && echo a100 || echo h100)
      break
    fi
    case "$status" in
      success) [ -f "$RUNS/DONE" ] && { echo "[loop] finished"; exit 0; }; break ;;
      failed|stopped) echo "[loop] run ended ($status) -> resubmit"; GPU=${GPU_FIRST:-h100}; break ;;
    esac
  done
  n=$((n+1))
done
echo "[loop] gave up after $n resubmits"; exit 1
