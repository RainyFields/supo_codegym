#!/bin/bash
# Checkpoint sync for Merlin pods: verl saves to pod-local disk (fast); this script mirrors each
# completed checkpoint to HDFS and restores the latest complete one before training.
# Why: writing a 113 GB FSDP checkpoint through the HDFS fuse mount ran at ~9 MB/s aggregate and
# tripped the 30-min gloo collective timeout (smoke run 5dd1a81ef5db89a8).
# Usage (sourced): ckpt_restore ; ckpt_sync_loop & ; ... ; ckpt_drain
# Requires: CKPT_DIR (local, verl default_local_dir), HDFS_CKPT (fuse path), HDFS_CKPT_URI (hdfs:// path)
set -uo pipefail
H=${HDFS_BIN:-/opt/tiger/yarn_deploy/hadoop/bin/hdfs}
export HADOOP_OPTS="-Djava.net.preferIPv4Stack=false -Djava.net.preferIPv6Addresses=true" HADOOP_CLIENT_OPTS="-Djava.net.preferIPv4Stack=false -Djava.net.preferIPv6Addresses=true"
KEEP_HDFS=${KEEP_HDFS:-2}
PUT_PAR=${PUT_PAR:-8}
_hdfs_ok() { [ -x "$H" ] && timeout 120 "$H" dfs -ls "$HDFS_CKPT_URI" >/dev/null 2>&1 || timeout 120 "$H" dfs -mkdir -p "$HDFS_CKPT_URI" >/dev/null 2>&1; }
_log() { echo "[ckpt-sync] $* $(date '+%H:%M:%S')"; }
# delete a path under $HDFS_CKPT: hdfs CLI first (fuse rm -rf of directory trees is unreliable), fuse fallback
_rm_retry() {
  local p=$1 i uri="${HDFS_CKPT_URI}${1#"$HDFS_CKPT"}"
  [ -e "$p" ] || return 0
  if [ "${SYNC_MODE:-cli}" = cli ] && [ -x "$H" ] && [ "$uri" != "$p" ]; then
    timeout 300 "$H" dfs -rm -r -skipTrash "$uri" >/dev/null 2>&1; sleep 2; [ -e "$p" ] || return 0
  fi
  for i in 1 2 3 4 5; do rm -rf "$p" 2>/dev/null; [ -e "$p" ] || return 0; sleep 5; done
  _log "WARN: could not remove $p"; return 1
}
_fuse_copy() { mkdir -p "$2" && cp -r "$1/." "$2/"; }   # copy contents; safe when dst already exists

# copy one local global_step dir to HDFS (parallel per-file puts via CLI; fuse cp fallback)
_upload_dir() {  # $1 = step N
  local n=$1 src="$CKPT_DIR/global_step_$1" dst="$HDFS_CKPT/global_step_$1" uri="$HDFS_CKPT_URI/global_step_$1"
  [ -d "$src" ] || { _log "src vanished: $src"; return 1; }
  local t0=$(date +%s) bytes=$(du -sb "$src" | cut -f1)
  _rm_retry "$dst"
  if [ "${SYNC_MODE:-cli}" = cli ]; then
    # per-file put with 3 attempts alternating the JVM IP-stack flags (datanodes answer on IPv4 or
    # IPv6 and the JVM picks one stack; "Protocol family unavailable" is the symptom). First error kept.
    export _PUT_SRC="$src" _PUT_URI="$uri" _PUT_ERR="/tmp/supo_put_err.$$"; rm -f "$_PUT_ERR"
    ( cd "$src" && find . -type d | sed 's|^\./||' | grep -v '^\.$' | sed "s|^|$uri/|" | xargs -r "$H" dfs -mkdir -p >/dev/null 2>&1
      find . -type f | sed 's|^\./||' | xargs -r -P $PUT_PAR -I{} bash -c '
        f="$1"; H="$2"; ok=0
        for flags in "-Djava.net.preferIPv4Stack=false -Djava.net.preferIPv6Addresses=true" "-Djava.net.preferIPv4Stack=true" "-Djava.net.preferIPv4Stack=false -Djava.net.preferIPv6Addresses=true"; do
          HADOOP_OPTS="$flags" HADOOP_CLIENT_OPTS="$flags" "$H" dfs -put -f "$_PUT_SRC/$f" "$_PUT_URI/$f" >/tmp/supo_put_$$.log 2>&1 && { ok=1; break; }
          [ -s "$_PUT_ERR" ] || grep -v "lock\|WARN\|^\s*at " /tmp/supo_put_$$.log | tail -2 > "$_PUT_ERR"
        done; rm -f /tmp/supo_put_$$.log; [ $ok = 1 ] || echo "PUTFAIL $f"' _ {} "$H" ) | grep -c PUTFAIL | grep -q '^0$' || { _log "cli upload had failures ($(head -c 300 "$_PUT_ERR" 2>/dev/null | tr '\n' ' ')), falling back to fuse cp"; _fuse_copy "$src" "$dst"; }
  else
    _fuse_copy "$src" "$dst"
  fi
  local nsrc=$(find "$src" -type f | wc -l) ndst=$(find "$dst" -type f 2>/dev/null | wc -l) bdst=$(du -sb "$dst" 2>/dev/null | cut -f1)
  if [ "$nsrc" = "$ndst" ] && [ "$bytes" = "$bdst" ]; then
    touch "$dst/.COMPLETE"; echo "$n" > "$HDFS_CKPT/latest_synced.txt"
    _log "uploaded global_step_$n: $((bytes/1000000)) MB in $(( $(date +%s)-t0 )) s ($(( bytes/1000000/($(date +%s)-t0+1) )) MB/s, $nsrc files)"
    return 0
  fi
  _log "VERIFY FAILED global_step_$n: files $nsrc vs $ndst, bytes $bytes vs $bdst"; return 1
}

_prune_hdfs() {
  ls -d "$HDFS_CKPT"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | head -n -$KEEP_HDFS | while read -r n; do
    [ -f "$HDFS_CKPT/global_step_$n/.COMPLETE" ] && { _log "pruning HDFS global_step_$n"; _rm_retry "$HDFS_CKPT/global_step_$n"; }
  done
}

ckpt_restore() {
  mkdir -p "$CKPT_DIR" "$HDFS_CKPT"
  SYNC_MODE=${SYNC_MODE_FORCE:-cli}; [ "$SYNC_MODE" = cli ] && ! _hdfs_ok && { SYNC_MODE=fuse; _log "hdfs CLI unavailable -> fuse mode"; }; export SYNC_MODE
  local n=$(cat "$HDFS_CKPT/latest_synced.txt" 2>/dev/null || echo 0)
  # fall back to the newest .COMPLETE dir if the pointer is missing
  [ "$n" -gt 0 ] 2>/dev/null || n=$(ls -d "$HDFS_CKPT"/global_step_*/.COMPLETE 2>/dev/null | sed 's|.*global_step_\([0-9]*\)/.*|\1|' | sort -n | tail -1)
  [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null || { _log "no HDFS checkpoint to restore (fresh start)"; return 0; }
  local src="$HDFS_CKPT/global_step_$n" uri="$HDFS_CKPT_URI/global_step_$n" dst="$CKPT_DIR/global_step_$n" t0=$(date +%s)
  [ -f "$src/.COMPLETE" ] || { _log "global_step_$n has no .COMPLETE marker; fresh start"; return 0; }
  _log "restoring global_step_$n from HDFS ($(du -sh "$src" | cut -f1))..."
  rm -rf "$dst"; mkdir -p "$dst"
  if [ "$SYNC_MODE" = cli ]; then
    ( cd "$src" && find . -type d | sed 's|^\./||' | grep -v '^\.$' | xargs -r -I{} mkdir -p "$dst/{}"
      find . -type f ! -name .COMPLETE | sed 's|^\./||' | xargs -r -P $PUT_PAR -I{} sh -c "$H dfs -get -f \"$uri/{}\" \"$dst/{}\" >/dev/null 2>&1 || echo GETFAIL {}" ) | grep -q GETFAIL && { _log "cli restore failed, fuse cp"; _fuse_copy "$src" "$dst"; }
  else
    _fuse_copy "$src" "$dst"
  fi
  rm -f "$dst/.COMPLETE"
  local bsrc=$(find "$src" -type f ! -name .COMPLETE -printf '%s\n' | awk '{s+=$1}END{print s}') bdst=$(find "$dst" -type f -printf '%s\n' | awk '{s+=$1}END{print s}')
  [ "$bsrc" = "$bdst" ] || { _log "RESTORE VERIFY FAILED ($bsrc vs $bdst bytes) -> fresh start"; rm -rf "$dst"; return 0; }
  echo "$n" > "$CKPT_DIR/latest_checkpointed_iteration.txt"
  _log "restored global_step_$n in $(( $(date +%s)-t0 )) s -> verl resume_mode=auto will pick it up"
}

ckpt_sync_loop() {  # background; stops when $SYNC_STOP_FILE exists and everything is uploaded
  local last=$(cat "$HDFS_CKPT/latest_synced.txt" 2>/dev/null || echo 0)
  while true; do
    local n=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    if [ "$n" -gt "$last" ] 2>/dev/null && [ -d "$CKPT_DIR/global_step_$n" ]; then
      sleep 20   # let verl finish writing data.pt / tq state after the tracker file
      if _upload_dir "$n"; then
        last=$n; _prune_hdfs
        # free pod-local disk (shared nodes had only ~250 GB free): the verified HDFS copy is the
        # source of truth; a restart restores it via ckpt_restore. verl's keep=1 pruning tolerates
        # already-missing older dirs.
        [ "${LOCAL_DELETE_AFTER_SYNC:-1}" = 1 ] && { rm -rf "$CKPT_DIR/global_step_$n" && _log "removed local global_step_$n after verified upload"; }
      fi
    elif [ -f "${SYNC_STOP_FILE:-/tmp/supo_sync_stop}" ]; then
      _log "sync loop exiting (last synced global_step_$last)"; return 0
    else
      sleep 30
    fi
  done
}

ckpt_drain() {  # call after training: signal the loop and wait for it
  touch "${SYNC_STOP_FILE:-/tmp/supo_sync_stop}"
  [ -n "${SYNC_PID:-}" ] && { _log "waiting for sync loop pid $SYNC_PID"; wait "$SYNC_PID" 2>/dev/null; }
  _log "drain done: HDFS latest_synced=$(cat "$HDFS_CKPT/latest_synced.txt" 2>/dev/null || echo none)"
}
