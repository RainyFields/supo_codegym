#!/bin/bash
# Local regression test for scripts/merlin/ckpt_sync.sh _upload_dir / _prune_hdfs. No HDFS needed: a fake
# `hdfs` shim maps hdfs://fake/<path> to a local dir and reproduces the ark-pod behaviour where `dfs -put`
# lands the file but exits non-zero with empty stderr (FAKE_PUT_RC=1). Optionally drops one file
# (FAKE_DROP=<basename>) to exercise the fuse fill.
#   bash scripts/merlin/tests/test_ckpt_sync_upload.sh [path/to/ckpt_sync.sh]   (default: the repo's)
set -uo pipefail
SYNC=${1:-$(cd "$(dirname "$0")/.." && pwd)/ckpt_sync.sh}
T=$(mktemp -d "${TMPDIR:-/tmp}/cksync.XXXXXX"); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/hdfs"
cat > "$T/bin/hdfs" <<'EOF'
#!/bin/bash
# fake `hdfs dfs ...`: hdfs://fake/X -> $FAKE_ROOT/X
map() { echo "$FAKE_ROOT/${1#hdfs://fake/}"; }
shift  # dfs
case "$1" in
  -ls) [ -e "$(map "$2")" ];;
  -mkdir) shift; [ "$1" = -p ] && shift; for u in "$@"; do mkdir -p "$(map "$u")"; done;;
  -put) shift; [ "$1" = -f ] && shift; src=$1; dst=$(map "$2")
        [ "$(basename "$src")" = "${FAKE_DROP:-}" ] && exit 1
        mkdir -p "$(dirname "$dst")" && cp "$src" "$dst"; exit "${FAKE_PUT_RC:-0}";;
  -get) shift; [ "$1" = -f ] && shift; cp "$(map "$1")" "$2";;
  -rm) shift; while [ "${1:0:1}" = - ]; do shift; done; rm -rf "$(map "$1")";;
  *) echo "fake hdfs: unsupported $*" >&2; exit 2;;
esac
EOF
chmod +x "$T/bin/hdfs"
export HDFS_BIN=$T/bin/hdfs FAKE_ROOT=$T/hdfs PUT_PAR=4
fail=0; ok() { echo "  PASS: $*"; }; bad() { echo "  FAIL: $*"; fail=1; }

mk_step() {  # $1 dir, $2 step, $3 n_gpus: 3 shards per rank under actor/ + data.pt + hf config (like verl)
  local d="$1/global_step_$2" r; mkdir -p "$d/actor/huggingface"; echo cfg > "$d/actor/huggingface/config.json"; head -c 7316 /dev/urandom > "$d/data.pt"
  for r in $(seq 0 $(($3-1))); do for k in model optim extra_state; do head -c $((20000+r)) /dev/urandom > "$d/actor/${k}_world_size_$3_rank_$r.pt"; done; done
}
fresh() {  # new exp dirs; $1 = tag
  export CKPT_DIR=$T/local_$1 HDFS_CKPT=$T/hdfs/$1 HDFS_CKPT_URI=hdfs://fake/$1; rm -rf "$CKPT_DIR" "$HDFS_CKPT"; mkdir -p "$CKPT_DIR" "$HDFS_CKPT"
}
run_upload() {  # $1 step; env passed through; prints the sync log
  ( source "$SYNC"; export SYNC_MODE=cli; _upload_dir "$1" ) 2>&1
}
nfiles() { find "$1" -type f ! -name '.COMPLETE*' 2>/dev/null | wc -l; }

echo "== single node, puts land but exit 1 (ark-pod behaviour): $SYNC"
fresh s1; mk_step "$CKPT_DIR" 10 2; unset NNODES NODE_RANK; export N_GPUS=2 FAKE_PUT_RC=1
log=$(run_upload 10); echo "$log" | sed 's/^/    /'
d=$HDFS_CKPT/global_step_10
[ -f "$d/.COMPLETE" ] && ok ".COMPLETE written" || bad "no .COMPLETE"
[ "$(cat "$HDFS_CKPT/latest_synced.txt" 2>/dev/null)" = 10 ] && ok "latest_synced=10" || bad "latest_synced=$(cat "$HDFS_CKPT/latest_synced.txt" 2>/dev/null)"
[ -e "$d/global_step_10.tmp" ] && bad "nested global_step_10.tmp copy exists (double upload)" || ok "no nested .tmp copy"
[ "$(nfiles "$d")" = "$(nfiles "$CKPT_DIR/global_step_10")" ] && ok "file count matches src ($(nfiles "$d"))" || bad "file count $(nfiles "$d") vs src $(nfiles "$CKPT_DIR/global_step_10")"
( cd "$CKPT_DIR/global_step_10" && find . -type f -exec cmp -s {} "$d/{}" \; -o -print | grep -q . ) && bad "content mismatch" || ok "every file byte-identical"
echo "$log" | grep -q "fuse fill copied [1-9]" && bad "fuse fill re-copied files although the puts landed" || ok "no fuse re-copy of landed files"

echo "== single node, one put genuinely fails (file not landed) -> fuse fill of that file only"
fresh s2; mk_step "$CKPT_DIR" 20 2; export FAKE_PUT_RC=0 FAKE_DROP=optim_world_size_2_rank_1.pt
log=$(run_upload 20); echo "$log" | sed 's/^/    /'
d=$HDFS_CKPT/global_step_20
echo "$log" | grep -q "fuse fill copied 1 file" && ok "fuse fill copied exactly 1 file" || bad "fuse fill did not copy exactly 1 file"
[ -f "$d/.COMPLETE" ] && [ "$(nfiles "$d")" = "$(nfiles "$CKPT_DIR/global_step_20")" ] && ok "complete + count matches" || bad "incomplete or count mismatch"
unset FAKE_DROP

echo "== two nodes, shared dir, each mirrors its own shards; complete after both"
fresh m1; export NNODES=2 N_GPUS=2 FAKE_PUT_RC=1
mk_step "$CKPT_DIR" 30 4   # node 0 disk: all 4 ranks written here for simplicity; _mine filters per node
export NODE_RANK=0; log0=$(run_upload 30); echo "$log0" | sed 's/^/    /'
d=$HDFS_CKPT/global_step_30
[ -f "$d/.COMPLETE.0" ] && [ ! -f "$HDFS_CKPT/latest_synced.txt" ] && ok "node 0 marker only, not yet complete" || bad "node 0 state wrong"
export NODE_RANK=1; log1=$(run_upload 30); echo "$log1" | sed 's/^/    /'
[ -f "$d/.COMPLETE.1" ] && [ "$(cat "$HDFS_CKPT/latest_synced.txt" 2>/dev/null)" = 30 ] && ok "complete after node 1, latest_synced=30" || bad "multi-node completion wrong"
[ -e "$d/global_step_30.tmp" ] && bad "nested .tmp in multi-node dir" || ok "no nested .tmp"
unset NNODES NODE_RANK

echo "== prune sweep removes the nested .tmp left by the old code inside a completed dir"
fresh p1; mkdir -p "$HDFS_CKPT/global_step_40/actor" "$HDFS_CKPT/global_step_40/global_step_40.tmp/actor" "$HDFS_CKPT/global_step_50/actor"
touch "$HDFS_CKPT/global_step_40/.COMPLETE" "$HDFS_CKPT/global_step_50/.COMPLETE"; echo 50 > "$HDFS_CKPT/latest_synced.txt"
( source "$SYNC"; export SYNC_MODE=cli KEEP_HDFS=2; _prune_hdfs ) 2>&1 | sed 's/^/    /'
[ -e "$HDFS_CKPT/global_step_40/global_step_40.tmp" ] && bad "nested .tmp survived the sweep" || ok "nested .tmp removed"
[ -d "$HDFS_CKPT/global_step_40" ] && [ -d "$HDFS_CKPT/global_step_50" ] && ok "both complete steps kept (keep=2)" || bad "a kept step was pruned"

[ $fail = 0 ] && echo "ALL PASS" || { echo "SOME FAILED"; exit 1; }
