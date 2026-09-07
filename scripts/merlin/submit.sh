#!/bin/bash
# Submit one arm as a Merlin batch job (dry-run first), record it in jobs/JOBS.tsv.
#   bash scripts/merlin/submit.sh <spec.json>            # dry-run + create
#   DRY_RUN=1 bash scripts/merlin/submit.sh <spec.json>  # dry-run only
set -euo pipefail
SPEC=${1:?spec json}
M=${MERLIN_CLI:-$HOME/.merlin-cli/bin/merlin-cli}
CP=${CONTROL_PLANE:-i18n-tt}
LEDGER=$(dirname "$0")/../../jobs/JOBS.tsv
[ -f "$LEDGER" ] || printf "submitted_sf\tname\tsid\turl\tgpu\tspec\tnote\n" > "$LEDGER"
echo "[submit] dry-run $SPEC"
$M --control-plane $CP job-v2 runs create --from-file "$SPEC" --dry-run
[ "${DRY_RUN:-0}" = 1 ] && exit 0
OUT=$($M --control-plane $CP job-v2 runs create --from-file "$SPEC")
echo "$OUT" | tail -c 2000
SID=$(echo "$OUT" | python3 -c "import sys,json; raw=sys.stdin.read(); i=raw.find('{'); d=json.loads(raw[i:]); print(d.get('sid') or d['data']['sid'])")
NAME=$(python3 -c "import json,sys; print(json.load(open('$SPEC'))['name'])")
GPU=$(python3 -c "import json,sys; print(json.load(open('$SPEC'))['resource_config']['arnold_resource_config']['roles'][0]['gpu_type'])")
URL="https://ml.tiktok-row.net/development/instance/jobs/$SID"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M')" "$NAME" "$SID" "$URL" "$GPU" "$SPEC" "" >> "$LEDGER"
echo "[submit] sid=$SID url=$URL (ledger: $LEDGER)"
echo "[submit] monitor: $M --control-plane $CP job-v2 runs get --json '{\"sid\":\"$SID\"}' | python3 -c \"import sys,json;d=json.load(sys.stdin)['data'];print(d['status'], d['runtime_context'].get('trial_status'))\""
