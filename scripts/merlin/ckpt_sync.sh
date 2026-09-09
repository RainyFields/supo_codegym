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
# Multi-node (NNODES>1): every node mirrors only the shards on its own disk into the shared step dir and
# touches .COMPLETE.<rank>; the step is complete when all NNODES markers exist. NODE_RANK/NNODES from env.
_is_complete() { local d=$1; if [ "${NNODES:-1}" -gt 1 ]; then [ "$(ls "$d"/.COMPLETE.* 2>/dev/null | wc -l)" -ge "${NNODES}" ]; else [ -f "$d/.COMPLETE" ]; fi; }
_complete_steps() { local d; for d in "$HDFS_CKPT"/global_step_*; do [ -d "$d" ] && _is_complete "$d" && basename "$d" | sed 's/global_step_//'; done | sort -n; }
_mine() { # keep files this node needs: everything non-sharded + shards whose rank // N_GPUS == NODE_RANK
  local f=$1 r; case "$f" in *_rank_*.pt) r=${f##*_rank_}; r=${r%.pt};; *) return 0;; esac; [[ "$r" =~ ^[0-9]+$ ]] || return 0; [ $(( r / ${N_GPUS:-8} )) = "${NODE_RANK:-0}" ]; }
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
_fuse_copy_mine() { local f; (cd "$1" && find . -type f ! -name '.COMPLETE*' | sed 's|^\./||') | while read -r f; do _mine "$f" || continue; mkdir -p "$2/$(dirname "$f")"; cp "$1/$f" "$2/$f"; done; }

# copy one local global_step dir to HDFS (parallel per-file puts via CLI; fuse cp fallback)
_upload_dir() {  # $1 = step N
  local n=$1 src="$CKPT_DIR/global_step_$1" dst="$HDFS_CKPT/global_step_$1" uri="$HDFS_CKPT_URI/global_step_$1"
  [ -d "$src" ] || { _log "src vanished: $src"; return 1; }
  local t0=$(date +%s) bytes=$(find "$src" -type f -printf '%s\n' | awk '{s+=$1}END{print s+0}') nsrc=$(find "$src" -type f | wc -l)   # captured BEFORE the copy: verl (keep=1) may rotate $src away right after
  _log "uploading global_step_$n: $((bytes/1000000)) MB, $nsrc files, mode ${SYNC_MODE:-cli}, node ${NODE_RANK:-0}/${NNODES:-1}"
  # Never copy over an existing/partial dst: fuse cannot rm -rf directory trees and overwriting files
  # through fuse is very slow (run #5). Move a leftover aside (rename works), copy into a fresh dir.
  if [ -e "$dst" ] && [ "${NNODES:-1}" -le 1 ]; then
    _rm_retry "$dst"
    [ -e "$dst" ] && { mv "$dst" "$dst.stale.$(date +%s)" 2>/dev/null && _log "moved leftover $dst aside" || { _log "cannot clear $dst"; return 1; }; }
  fi
  local final="$dst"
  if [ "${NNODES:-1}" -gt 1 ]; then mkdir -p "$dst"; else dst="$dst.tmp"; rm -rf "$dst" 2>/dev/null; mv "$dst" "$dst.stale.$(date +%s)" 2>/dev/null; fi
  if [ "${SYNC_MODE:-cli}" = cli ]; then
    # per-file put with 3 attempts alternating the JVM IP-stack flags (datanodes answer on IPv4 or
    # IPv6 and the JVM picks one stack; "Protocol family unavailable" is the symptom). First error kept.
    export _PUT_SRC="$src" _PUT_URI="$uri" _PUT_ERR="/tmp/supo_put_err.$$"; rm -f "$_PUT_ERR"
    ( cd "$src" && find . -type d | sed 's|^\./||' | grep -v '^\.$' | sed "s|^|$uri/|" | xargs -r "$H" dfs -mkdir -p >/dev/null 2>&1
      find . -type f | sed 's|^\./||' | xargs -r -P $PUT_PAR -I{} bash -c '
        f="$1"; H="$2"; ok=0
        for flags in "-Djava.net.preferIPv4Stack=false -Djava.net.preferIPv6Addresses=true" "-Djava.net.preferIPv4Stack=true" "-Djava.net.preferIPv4Stack=false -Djava.net.preferIPv6Addresses=true"; do
          HADOOP_OPTS="$flags" HADOOP_CLIENT_OPTS="$flags" timeout ${PUT_TIMEOUT:-900} "$H" dfs -put -f "$_PUT_SRC/$f" "$_PUT_URI/$f" >/tmp/supo_put_$$.log 2>&1 && { ok=1; break; }
          [ -s "$_PUT_ERR" ] || grep -v "lock\|WARN\|^\s*at " /tmp/supo_put_$$.log | tail -2 > "$_PUT_ERR"
        done; rm -f /tmp/supo_put_$$.log; [ $ok = 1 ] || echo "PUTFAIL $f"' _ {} "$H" ) | grep -c PUTFAIL | grep -q '^0$' || { _log "cli upload had failures ($(head -c 300 "$_PUT_ERR" 2>/dev/null | tr '\n' ' ')), falling back to fuse cp"; _fuse_copy "$src" "$dst"; }
  else
    _fuse_copy "$src" "$dst"
  fi
  local ndst bdst
  if [ "${NNODES:-1}" -gt 1 ]; then
    ndst=0; bdst=0; for f in $(cd "$src" 2>/dev/null && find . -type f | sed 's|^\./||'); do [ -f "$dst/$f" ] && { ndst=$((ndst+1)); bdst=$((bdst + $(stat -c %s "$dst/$f"))); }; done
  else
    ndst=$(find "$dst" -type f 2>/dev/null | wc -l); bdst=$(find "$dst" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1}END{print s+0}')
  fi
  if [ "$nsrc" = "$ndst" ] && [ "$bytes" = "$bdst" ]; then
    if [ "${NNODES:-1}" -gt 1 ]; then
      touch "$dst/.COMPLETE.${NODE_RANK:-0}"
      _is_complete "$dst" && echo "$n" > "$HDFS_CKPT/latest_synced.txt"
    else
      mv "$dst" "$final" || { _log "rename $dst -> $final FAILED"; return 1; }; dst="$final"
      touch "$dst/.COMPLETE"; echo "$n" > "$HDFS_CKPT/latest_synced.txt"
    fi
    _log "uploaded global_step_$n: $((bytes/1000000)) MB in $(( $(date +%s)-t0 )) s ($(( bytes/1000000/($(date +%s)-t0+1) )) MB/s, $nsrc files)"
    return 0
  fi
  _log "VERIFY FAILED global_step_$n: files $nsrc vs $ndst, bytes $bytes vs $bdst"; return 1
}

_prune_hdfs() {
  # keep the newest KEEP_HDFS complete checkpoints; also drop incomplete dirs that are older than the
  # newest complete one (failed/partial mirrors otherwise accumulate ~113 GB per save; run #4 left
  # 7 of them). The newest dir is never touched here (it may be mid-upload).
  local latest_complete=$(_complete_steps | tail -1)
  local newest=$(ls -d "$HDFS_CKPT"/global_step_* 2>/dev/null | grep -E 'global_step_[0-9]+$' | sed 's/.*global_step_//' | sort -n | tail -1)
  # stale/tmp leftovers (fuse cannot rm -rf them): try the CLI, ignore failures
  for d in "$HDFS_CKPT"/global_step_*.stale.* "$HDFS_CKPT"/global_step_*.tmp; do [ -e "$d" ] && _rm_retry "$d" >/dev/null 2>&1; done
  ls -d "$HDFS_CKPT"/global_step_* 2>/dev/null | grep -E 'global_step_[0-9]+$' | sed 's/.*global_step_//' | sort -n | while read -r n; do
    [ "$n" = "$newest" ] && continue
    if _is_complete "$HDFS_CKPT/global_step_$n"; then
      _complete_steps | head -n -$KEEP_HDFS | grep -qx "$n" && { _log "pruning HDFS global_step_$n (complete, beyond keep=$KEEP_HDFS)"; _rm_retry "$HDFS_CKPT/global_step_$n"; }
    elif [ -n "$latest_complete" ] && [ "$n" -lt "$latest_complete" ]; then
      _log "pruning HDFS global_step_$n (incomplete, older than complete $latest_complete)"; _rm_retry "$HDFS_CKPT/global_step_$n"
    fi
  done
}

ckpt_restore() {
  mkdir -p "$CKPT_DIR" "$HDFS_CKPT"
  SYNC_MODE=${SYNC_MODE_FORCE:-cli}; [ "$SYNC_MODE" = cli ] && ! _hdfs_ok && { SYNC_MODE=fuse; _log "hdfs CLI unavailable -> fuse mode"; }; export SYNC_MODE
  local n=$(cat "$HDFS_CKPT/latest_synced.txt" 2>/dev/null || echo 0)
  # fall back to the newest .COMPLETE dir if the pointer is missing
  [ "$n" -gt 0 ] 2>/dev/null || n=$(_complete_steps | tail -1)
  [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null || { _log "no HDFS checkpoint to restore (fresh start)"; return 0; }
  local src="$HDFS_CKPT/global_step_$n" uri="$HDFS_CKPT_URI/global_step_$n" dst="$CKPT_DIR/global_step_$n" t0=$(date +%s)
  _is_complete "$src" || { _log "global_step_$n is not complete (markers: $(ls -a "$src" | grep -c COMPLETE)); fresh start"; return 0; }
  _log "restoring global_step_$n from HDFS ($(du -sh "$src" | cut -f1))..."
  rm -rf "$dst"; mkdir -p "$dst"
  if [ "$SYNC_MODE" = cli ]; then
    ( cd "$src" && find . -type d | sed 's|^\./||' | grep -v '^\.$' | xargs -r -I{} mkdir -p "$dst/{}"
      find . -type f ! -name '.COMPLETE*' | sed 's|^\./||' | while read -r f; do _mine "$f" && echo "$f"; done | xargs -r -P $PUT_PAR -I{} sh -c "$H dfs -get -f \"$uri/{}\" \"$dst/{}\" >/dev/null 2>&1 || echo GETFAIL {}" ) | grep -q GETFAIL && { _log "cli restore failed, fuse cp"; _fuse_copy_mine "$src" "$dst"; }
  else
    _fuse_copy_mine "$src" "$dst"
  fi
  rm -f "$dst"/.COMPLETE*
  local bsrc=$( (cd "$src" && find . -type f ! -name '.COMPLETE*' | sed 's|^\./||') | while read -r f; do _mine "$f" && stat -c %s "$src/$f"; done | awk '{s+=$1}END{print s+0}') bdst=$(find "$dst" -type f -printf '%s\n' | awk '{s+=$1}END{print s+0}')
  [ "$bsrc" = "$bdst" ] || { _log "RESTORE VERIFY FAILED ($bsrc vs $bdst bytes) -> fresh start"; rm -rf "$dst"; return 0; }
  echo "$n" > "$CKPT_DIR/latest_checkpointed_iteration.txt"
  _log "restored global_step_$n in $(( $(date +%s)-t0 )) s -> verl resume_mode=auto will pick it up"
}

ckpt_sync_loop() {  # background; stops when $SYNC_STOP_FILE exists and everything is uploaded
  local last=$(cat "$HDFS_CKPT/latest_synced.txt" 2>/dev/null || echo 0)
  while true; do
    local n
    if [ "${NNODES:-1}" -gt 1 ] && [ "${NODE_RANK:-0}" != 0 ]; then n=$(cat "$HDFS_CKPT/tracker_step.txt" 2>/dev/null || echo 0)
    else n=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0); [ "${NNODES:-1}" -gt 1 ] && [ "$n" -gt 0 ] 2>/dev/null && echo "$n" > "$HDFS_CKPT/tracker_step.txt"; fi
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

ckpt_drain() {  # call after training: signal the loop and wait for it, then make one explicit final attempt
  touch "${SYNC_STOP_FILE:-/tmp/supo_sync_stop}"
  [ -n "${SYNC_PID:-}" ] && { _log "waiting for sync loop pid $SYNC_PID"; wait "$SYNC_PID" 2>/dev/null; }
  # Smoke 4430907959c555c2 (2 nodes) exited DONE ~1 min after saving global_step_2 with nothing on HDFS and no
  # upload lines logged: never rely on the loop having seen the last save. Re-derive the newest local step and
  # upload it here if this node's copy is not marked complete (single-node: not yet the latest synced step).
  local n last
  if [ "${NNODES:-1}" -gt 1 ] && [ "${NODE_RANK:-0}" != 0 ]; then n=$(cat "$HDFS_CKPT/tracker_step.txt" 2>/dev/null || echo 0)
  else n=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0); fi
  last=$(cat "$HDFS_CKPT/latest_synced.txt" 2>/dev/null || echo 0)
  if [ "$n" -gt 0 ] 2>/dev/null && [ -d "$CKPT_DIR/global_step_$n" ]; then
    local need=0
    if [ "${NNODES:-1}" -gt 1 ]; then [ -f "$HDFS_CKPT/global_step_$n/.COMPLETE.${NODE_RANK:-0}" ] || need=1
    else [ "$n" -gt "$last" ] 2>/dev/null && need=1; fi
    if [ $need = 1 ]; then
      _log "drain: final upload attempt for global_step_$n (latest_synced=$last)"
      [ "${NNODES:-1}" -gt 1 ] && [ "${NODE_RANK:-0}" = 0 ] && echo "$n" > "$HDFS_CKPT/tracker_step.txt"
      _upload_dir "$n" || { _log "drain: retrying global_step_$n once"; _upload_dir "$n"; } || _log "drain: global_step_$n NOT mirrored"
    else _log "drain: global_step_$n already mirrored"; fi
  else _log "drain: no local checkpoint to mirror (tracker step=$n)"; fi
  _log "drain done: HDFS latest_synced=$(cat "$HDFS_CKPT/latest_synced.txt" 2>/dev/null || echo none)"
}
