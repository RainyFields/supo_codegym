#!/bin/bash
# Merlin/Arnold batched-job entrypoint for the SUPO CodeGym replication (pattern: ARCO
# job_entrypoint.sh). Restores python+venv, the byted-wandb overlay and the repo snapshot
# from HDFS job-assets at the SAME absolute paths as the devbox, stages the base model to
# pod-local disk, runs a GPU preflight, then launches scripts/train_codegym.sh. verl writes
# checkpoints directly to HDFS ($PROJECT_ROOT/checkpoints/<exp>) with resume_mode=auto, so a
# restart of this entrypoint resumes the run.
#
# Required env (job env_map): ARM (grpo|supo). Optional: RUN_TAG, TOTAL_STEPS, SAVE_FREQ,
# TEST_FREQ, TRAIN_BS, N_ROLLOUT, MINI_BS, GPU_MEM_UTIL, EXTRA_ARGS, MIN_GPUS (default 8).
set -uo pipefail
XD=/home/tiger/xiaoxuan
PROJECT_ROOT=/mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym
ASSETS=$PROJECT_ROOT/job-assets
ARM=${ARM:?ARM unset (grpo|supo)}
RUN_NAME=${ARM}${RUN_TAG:-}
RUNS=$PROJECT_ROOT/job-runs/$RUN_NAME
PY=$XD/envs/supo/bin/python
export TZ=America/Los_Angeles
echo "[job] node=$(hostname) arm=$ARM run=$RUN_NAME $(date)"

ls /mnt/hdfs/mlsys/users/xiaoxuan >/dev/null 2>&1 || { echo "[job] FATAL: HDFS fuse not available"; exit 41; }
mkdir -p "$RUNS" "$XD/envs" "$XD/external" /tmp/supo_tmp
# Multi-node needs an IPv4 node address: on IPv6-only A100 pods (smokes 8ffc47df8d1b6184 / 48bd7be4dd69201b, host
# fdbd:dc61:1a:459::12) ray joins over "[v6]:port" but the trainer then hangs after vLLM init with 0% GPU util.
# Fail fast before the 62 GB model fetch so a resubmit can land on a different node.
if [ "${NNODES:-1}" -gt 1 ]; then
  _v4=$(for c in "${MY_HOST_IP:-}" "${BYTED_HOST_IP:-}" $(hostname -I 2>/dev/null); do case "$c" in [0-9]*.[0-9]*.[0-9]*.[0-9]*) [ "${c#127.}" = "$c" ] && { echo "$c"; break; };; esac; done)
  [ -n "$_v4" ] || { echo "[job] FATAL: no IPv4 address on this pod (MY_HOST_IP=${MY_HOST_IP:-?}; hostname -I: $(hostname -I 2>/dev/null)); IPv6-only multi-node is unsupported -> exit 48"; echo "rc=48 no-ipv4 $(date)" >> "$RUNS/FAILED"; touch "$RUNS/FAILED.${ARNOLD_TRIAL_ID:-0}"; exit 48; }
fi
export TMPDIR=/tmp/supo_tmp    # bad pod TMPDIR breaks vLLM zmq ipc binds (FoldAgent lesson)
ulimit -n 65536 || true

# ── python 3.12 + venv (absolute paths inside the tarball) ────────────────
if [ ! -x "$PY" ]; then
  echo "[job] restoring venv ($(date))..."
  tar xzf "$ASSETS/envs-supo.tar.gz" -C / || { echo "[job] FATAL: venv untar failed"; exit 43; }
fi
# ── flash-attn 2.8.3 for torch 2.11+cu129 (pods run driver R535 = CUDA 12.9 max; no
#    prebuilt cu12/torch2.11 wheel exists). Install the cached wheel from HDFS, else build it
#    once here with the staged CUDA 12.9 toolchain (96 cores ≈ 10-15 min) and cache it. ─────
FA_WHEEL_DIR=$ASSETS/wheels/cu129torch2.11; mkdir -p "$FA_WHEEL_DIR"
FA_WHEEL=$(ls "$FA_WHEEL_DIR"/flash_attn-2.8.3*.whl 2>/dev/null | head -1)
if ! $PY -c "import flash_attn, flash_attn_2_cuda" 2>/dev/null; then
  if [ -z "$FA_WHEEL" ]; then
    echo "[job] building flash-attn wheel ($(date))..."
    mkdir -p /tmp/fa_build && cd /tmp/fa_build
    tar xzf "$ASSETS/cuda-12.9-toolchain.tar.gz" -C /tmp/fa_build && tar xzf "$ASSETS/flash_attn-2.8.3.tar.gz"
    ( export CUDA_HOME=/tmp/fa_build/cuda-12.9/usr/local/cuda-12.9 PATH=/tmp/fa_build/cuda-12.9/usr/local/cuda-12.9/bin:$XD/envs/supo/bin:$PATH \
             FLASH_ATTN_CUDA_ARCHS="80;90" MAX_JOBS=${FA_MAX_JOBS:-64} NVCC_THREADS=2 FLASH_ATTENTION_FORCE_BUILD=TRUE
      cd flash_attn-2.8.3 && $PY -m pip wheel . --no-deps --no-build-isolation -w /tmp/fa_build/out 2>&1 | grep -v "^\s*$" | tail -30 )
    FA_WHEEL=$(ls /tmp/fa_build/out/flash_attn-2.8.3*.whl 2>/dev/null | head -1)
    [ -n "$FA_WHEEL" ] || { echo "[job] FATAL: flash-attn build failed"; exit 46; }
    cp "$FA_WHEEL" "$FA_WHEEL_DIR/" && FA_WHEEL="$FA_WHEEL_DIR/$(basename "$FA_WHEEL")" && echo "[job] cached $FA_WHEEL ($(date))"
    cd /
  fi
  $PY -m pip install --no-deps --force-reinstall "$FA_WHEEL" 2>&1 | tail -1 || { echo "[job] FATAL: flash-attn install failed"; exit 46; }
fi
# ── byted-wandb overlay -> merlin tracking ──────────────────────────────────
if [ ! -d "$XD/envs/byted-wandb-overlay/wandb" ]; then
  tar xzf "$ASSETS/byted-wandb-overlay.tar.gz" -C "$XD/envs" || echo "[job] WARN: overlay untar failed (wandb would go offline)"
fi
# ── repo snapshot: supo_codegym + patched verl (always fresh) ──────────────
echo "[job] restoring repo ($(date))..."
rm -rf "$XD/supo_codegym" "$XD/external/verl"
tar xzf "$ASSETS/supo-repo.tar.gz" -C / || { echo "[job] FATAL: repo untar failed"; exit 44; }

# ── base model -> pod-local disk (fast, repeated loads) ─────────────────────
# MODEL_SRC: HDFS copy (default Qwen3.5-9B). MODEL_HF_ID: fetch from Hugging Face instead (pods reach
# huggingface.co at ~115 MB/s, probe 01031c55137557b8) — used for Qwen2.5-32B-Instruct (not on HDFS).
MODEL_SRC=${MODEL_SRC:-/mnt/hdfs/mlsys/models/Qwen3.5-9B}
MODEL_LOCAL=/tmp/models/$(basename "${MODEL_HF_ID:-$MODEL_SRC}")
NSHARD_MIN=${MODEL_MIN_SHARDS:-4}
if [ ! -f "$MODEL_LOCAL/model.safetensors.index.json" ] || [ "$(ls "$MODEL_LOCAL"/*.safetensors 2>/dev/null | wc -l)" -lt "$NSHARD_MIN" ]; then
  echo "[job] staging model to $MODEL_LOCAL ($(date))..."
  if [ -n "${MODEL_PARTS:-}" ] && [ -f "$ASSETS/$MODEL_PARTS/MANIFEST" ]; then
    # chunked tar of the model dir on HDFS (job-assets/<MODEL_PARTS>/*.part*, MANIFEST="<n> <md5>"): fast fuse read
    echo "[job] model from HDFS chunks $MODEL_PARTS ($(cat $ASSETS/$MODEL_PARTS/MANIFEST))"
    mkdir -p /tmp/models && cat $(ls $ASSETS/$MODEL_PARTS/*.part[0-9]* | sort) | tar xf - -C /tmp/models || { echo "[job] FATAL: chunk restore failed"; exit 45; }
  elif [ -n "${MODEL_HF_ID:-}" ]; then
    # per-file curl with resume + stall cutoff: huggingface_hub's snapshot_download ran at 520 MB/s then hung
    # for good at 31 GB (probe c6934f8f43e5eb2c); curl --speed-time aborts a stalled transfer and we retry.
    mkdir -p "$MODEL_LOCAL"; HFB="https://huggingface.co/$MODEL_HF_ID/resolve/main"
    curl -sL --retry 5 -o "$MODEL_LOCAL/model.safetensors.index.json" "$HFB/model.safetensors.index.json" || { echo "[job] FATAL: HF index fetch failed"; exit 45; }
    FILES="$($PY -c "import json;print(' '.join(sorted(set(json.load(open('$MODEL_LOCAL/model.safetensors.index.json'))['weight_map'].values()))))") config.json generation_config.json tokenizer_config.json tokenizer.json merges.txt vocab.json"
    echo "[job] HF fetch of $(echo $FILES | wc -w) files via curl ($(date))"
    _hf_get() { local f=$1 t; for t in 1 2 3 4 5 6; do curl -sL -C - --retry 3 --speed-time 60 --speed-limit 2000000 --max-time 1800 -o "$MODEL_LOCAL/$f" "$HFB/$f" && return 0; echo "[job] retry $t for $f"; sleep 5; done; return 1; }
    export -f _hf_get; export MODEL_LOCAL HFB
    echo $FILES | tr ' ' '\n' | xargs -P ${HF_PAR:-6} -I{} bash -c '_hf_get "$1" || { echo "[job] HF FETCH FAILED $1"; echo FAIL >> /tmp/hf_fail; }' _ {}
    [ -f /tmp/hf_fail ] && { echo "[job] FATAL: HF download failed"; exit 45; }
    $PY -c "
import json,os,sys; idx=json.load(open('$MODEL_LOCAL/model.safetensors.index.json')); miss=[f for f in set(idx['weight_map'].values()) if not os.path.exists('$MODEL_LOCAL/'+f) or os.path.getsize('$MODEL_LOCAL/'+f)<1e6]; print('[job] HF fetch done, missing shards:', miss); sys.exit(1 if miss else 0)" || exit 45
  else
    mkdir -p "$MODEL_LOCAL" && cp "$MODEL_SRC"/* "$MODEL_LOCAL"/ || { echo "[job] FATAL: model staging failed"; exit 45; }
  fi
fi
echo "[job] model staged $(du -sh $MODEL_LOCAL | cut -f1) $(date)"

# ── GPU preflight (bad node / driver -> exit 42 for reschedule) ─────────────
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | head -3
export MIN_GPUS=${MIN_GPUS:-8}
$PY - <<'PYEOF'
import os, sys, socket, torch
n = torch.cuda.device_count()
print(f"[job] preflight node={socket.gethostname()} torch={torch.__version__} cuda={torch.version.cuda} gpus={n} driver_api={torch.cuda.driver_version() if hasattr(torch.cuda, 'driver_version') else '?'}", flush=True)
need = int(os.environ.get("MIN_GPUS", "8"))
if n < need:
    sys.exit(f"[job] preflight FAILED: only {n} GPUs visible (need {need})")
torch.zeros(1, device="cuda:0") @ torch.zeros(1, device="cuda:0")
from vllm.platforms import current_platform
if not current_platform.is_cuda():
    sys.exit(f"[job] preflight FAILED: vLLM platform {current_platform.device_type!r}")
import flash_attn, fla, supo.agent_loop  # noqa
from flash_attn import flash_attn_varlen_func  # cu129 build must load
print("[job] preflight OK", flush=True)
PYEOF
rc=$?; [ $rc -ne 0 ] && { echo "[job] exiting 42 for reschedule"; exit 42; }

# ── checkpoints: local save + background HDFS mirror (see scripts/merlin/ckpt_sync.sh) ─────
case "$ARM" in supo) TAG=4kx8 ;; grpo) TAG=32k ;; *) TAG=$ARM ;; esac   # must match train_codegym.sh
export MODEL_TAG=${MODEL_TAG:-qwen35-9b}
export EXP_NAME=${EXP_NAME:-${ARM}_codegym_${MODEL_TAG}_${TAG}${RUN_TAG:-}}
export PROJECT_DIR=$XD/supo_codegym; source "$XD/supo_codegym/scripts/train_env.sh"   # before ray start (multi-node)
export CKPT_DIR=/tmp/supo_ckpt/$EXP_NAME
export HDFS_CKPT=$PROJECT_ROOT/checkpoints/$EXP_NAME
export HDFS_CKPT_URI=hdfs://harunava/home/byte_arnold_va_ssd/mlsys/users/xiaoxuan/supo_codegym/checkpoints/$EXP_NAME
export SYNC_STOP_FILE=/tmp/supo_sync_stop; rm -f $SYNC_STOP_FILE
export NNODES=${NNODES:-1} NODE_RANK=${ARNOLD_ID:-0}
SYNC_LOG=$RUNS/ckpt_sync.log; [ "$NNODES" -gt 1 ] && SYNC_LOG=$RUNS/ckpt_sync.$NODE_RANK.log
# The sync loop and the drain both append to the log; HDFS allows one writer per file, so appends from a second
# process were dropped (no drain lines ever reached HDFS). Write locally and copy the whole file over every minute.
SYNC_LOG_HDFS=$SYNC_LOG; SYNC_LOG=/tmp/supo_ckpt_sync.$NODE_RANK.log; : > "$SYNC_LOG"
_sync_log_push() { cp -f "$SYNC_LOG" "$SYNC_LOG_HDFS.tmp" 2>/dev/null && mv -f "$SYNC_LOG_HDFS.tmp" "$SYNC_LOG_HDFS" 2>/dev/null; }
( while true; do sleep 60; _sync_log_push; done ) & SYNC_LOG_PUSH_PID=$!
df -h /tmp | tail -1 | awk '{print "[job] /tmp disk: size="$2" used="$3" avail="$4}' | tee -a "$SYNC_LOG"
# Measured in run #4: the pod's fuse mount copies a 113 GB checkpoint in ~15 min (~125 MB/s) while the
# hdfs CLI puts stalled ~66 min on IPv4/IPv6 datanode connections before failing -> mirror via fuse.
# run #5 (Sep 8): the fuse copy crawled at ~20 MB/s on other nodes -> CLI puts first (per-file
# timeout + IPv4/IPv6 retries, 55 MB/s when they work), fuse only as fallback.
export SYNC_MODE_FORCE=${SYNC_MODE_FORCE:-cli} PUT_TIMEOUT=${PUT_TIMEOUT:-900}
source "$XD/supo_codegym/scripts/merlin/ckpt_sync.sh"
ckpt_restore >> "$SYNC_LOG" 2>&1; tail -2 "$SYNC_LOG"
ckpt_sync_loop >> "$SYNC_LOG" 2>&1 &
SYNC_PID=$!

# ── host stats every 30 s (memory / load / process counts / top RSS) -> $RUNS/host_stats.log ────
( while true; do
    { echo "=== $(date '+%m-%d %H:%M:%S') load=$(cut -d' ' -f1-3 /proc/loadavg) procs=$(ls /proc | grep -c '^[0-9]') py=$(pgrep -c python)";
      free -g | awk 'NR==2{print "mem_total="$2"G used="$3"G free="$4"G avail="$7"G"}';
      if [ -f /sys/fs/cgroup/memory.max ]; then echo "cgroup_v2 max=$(cat /sys/fs/cgroup/memory.max) current=$(cat /sys/fs/cgroup/memory.current) $(grep -E 'oom_kill ' /sys/fs/cgroup/memory.events | tr '\n' ' ')";
      elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then echo "cgroup_v1 limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) usage=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes) $(grep oom_kill /sys/fs/cgroup/memory/memory.oom_control | tr '\n' ' ')"; fi;
      ps -eo rss=,comm= --sort=-rss 2>/dev/null | head -4 | awk '{printf "%s %.1fG; ", $2, $1/1048576}'; echo;
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | paste -sd' ' | cut -c1-200;
      dmesg -T 2>/dev/null | grep -i -E 'killed process|out of memory' | tail -2; } >> "$RUNS/host_stats.log" 2>&1
    sleep 30; done ) &
HOST_STATS_PID=$!

# ── multi-node ray bootstrap (NNODES>1): rank 0 = head, others join; trainer runs on rank 0 only ──
NNODES=${NNODES:-1}; NODE_RANK=${ARNOLD_ID:-0}; export NNODES
if [ "$NNODES" -gt 1 ]; then
  # Prefer an IPv4 address: some A100 pods expose MY_HOST_IP as a bare IPv6 (smoke 8ffc47df8d1b6184) and
  # `ray start --address=v6:port` rejects it ("Invalid address format"). IPv6 fallback is bracketed for --address.
  _pick_ip() { local c; for c in "${MY_HOST_IP:-}" "${BYTED_HOST_IP:-}" $(hostname -I 2>/dev/null); do case "$c" in [0-9]*.[0-9]*.[0-9]*.[0-9]*) [ "${c#127.}" = "$c" ] && { echo "$c"; return; };; esac; done
                for c in "${MY_HOST_IP:-}" "${BYTED_HOST_IP:-}" $(hostname -I 2>/dev/null); do [ -n "$c" ] && { echo "$c"; return; }; done; }
  MY_IP=$(_pick_ip); MY_IP_ADDR=$MY_IP; case "$MY_IP" in *:*) MY_IP_ADDR="[$MY_IP]";; esac
  HEAD_FILE=$RUNS/ray_head_${ARNOLD_TRIAL_ID:-$$}.txt
  echo "[job] multi-node: NNODES=$NNODES rank=$NODE_RANK ip=$MY_IP hosts=${ARNOLD_WORKER_HOSTS:-?}"
  if [ "$NODE_RANK" = 0 ]; then
    RAY_LOG=$RUNS/ray_head_${ARNOLD_TRIAL_ID:-$$}.log; $XD/envs/supo/bin/ray stop --force >/dev/null 2>&1; sleep 2
    # try the allocated worker-0 port first, then the other allocated MERLIN_INTERNAL ports, then a free ephemeral one
    CANDS="${ARNOLD_WORKER_0_PORT:-} $(env | grep -o 'ARNOLD_MERLIN_INTERNAL_[0-9]*_CURRENT_PORT=[0-9]*' | cut -d= -f2 | tr '\n' ' ') $(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1])')"
    # Ray's default worker-port range is 10002-19999 and collides with Arnold-allocated ports (10188 in smoke #4:
    # "port ... is used by other components"); pin worker ports high. Two tries per port: the GCS connect check
    # (5 s) can miss a slow start on a loaded node.
    RAY_PORT=""; for pt in $CANDS; do for try in 1 2; do
      echo "[job] ray head attempt on port $pt try $try ($(date '+%H:%M:%S'))" >> "$RAY_LOG"
      if $XD/envs/supo/bin/ray start --head --node-ip-address="$MY_IP" --port="$pt" --num-gpus="$N_GPUS" --min-worker-port=30000 --max-worker-port=39999 --disable-usage-stats >> "$RAY_LOG" 2>&1; then RAY_PORT=$pt; break 2; fi
      $XD/envs/supo/bin/ray stop --force >/dev/null 2>&1; sleep 8
    done; done
    [ -n "$RAY_PORT" ] || { echo "[job] FATAL: ray head failed on all ports; last output:"; tail -15 "$RAY_LOG"; exit 47; }
    echo "$MY_IP_ADDR:$RAY_PORT" > "$HEAD_FILE"; export RAY_ADDRESS="$MY_IP_ADDR:$RAY_PORT"
    for i in $(seq 1 90); do n=$($XD/envs/supo/bin/python -c "import ray; ray.init(address='$RAY_ADDRESS', ignore_reinit_error=True, logging_level='ERROR'); print(len([x for x in ray.nodes() if x['Alive']]))" 2>/dev/null); [ "${n:-0}" -ge "$NNODES" ] && break; sleep 20; done
    echo "[job] ray cluster: ${n:-0}/$NNODES nodes alive ($(date))"; [ "${n:-0}" -ge "$NNODES" ] || { echo "[job] FATAL: ray workers did not join"; exit 47; }
  else
    for i in $(seq 1 90); do [ -s "$HEAD_FILE" ] && break; sleep 20; done
    HEAD=$(cat "$HEAD_FILE" 2>/dev/null); [ -n "$HEAD" ] || { echo "[job] FATAL: no ray head published"; exit 47; }
    echo "[job] joining ray head $HEAD ($(date))"
    $XD/envs/supo/bin/ray stop --force >/dev/null 2>&1; sleep 2
    ok=0; for try in 1 2 3; do $XD/envs/supo/bin/ray start --address="$HEAD" --node-ip-address="$MY_IP" --num-gpus="$N_GPUS" --min-worker-port=30000 --max-worker-port=39999 --disable-usage-stats >> "$RUNS/ray_worker_${NODE_RANK}_${ARNOLD_TRIAL_ID:-$$}.log" 2>&1 && { ok=1; break; }; $XD/envs/supo/bin/ray stop --force >/dev/null 2>&1; sleep 10; done
    [ $ok = 1 ] || { echo "[job] FATAL: ray worker failed to join; last output:"; tail -15 "$RUNS/ray_worker_${NODE_RANK}_${ARNOLD_TRIAL_ID:-$$}.log"; exit 47; }
    # stay up until rank 0 finishes (DONE/FAILED marker) or the head disappears; keep mirroring our shards
    while [ ! -f "$RUNS/DONE" ] && [ ! -f "$RUNS/FAILED.$ARNOLD_TRIAL_ID" ] && $XD/envs/supo/bin/ray status --address="$HEAD" >/dev/null 2>&1; do sleep 60; done
    echo "[job] rank $NODE_RANK: head finished ($(date)); draining mirror"
    kill $HOST_STATS_PID 2>/dev/null; ckpt_drain >> "$SYNC_LOG" 2>&1; tail -2 "$SYNC_LOG"; kill $SYNC_LOG_PUSH_PID 2>/dev/null; _sync_log_push
    $XD/envs/supo/bin/ray stop >/dev/null 2>&1; exit 0
  fi
fi

# ── train ───────────────────────────────────────────────────────────────────
export MODEL_PATH=$MODEL_LOCAL
export PROJECT_DIR=$XD/supo_codegym VERL_DIR=$XD/external/verl VENV=$XD/envs/supo
export RUN_TAG=${RUN_TAG:-}
LOG=$RUNS/train_$(date +%Y%m%d_%H%M%S).log
echo "[job] launching training, log=$LOG"
bash "$XD/supo_codegym/scripts/train_codegym.sh" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
kill $HOST_STATS_PID 2>/dev/null
ckpt_drain >> "$SYNC_LOG" 2>&1   # NOT piped: a pipe forks a subshell that cannot `wait` on SYNC_PID
tail -3 "$SYNC_LOG"; kill $SYNC_LOG_PUSH_PID 2>/dev/null; _sync_log_push
if [ $rc -eq 0 ]; then touch "$RUNS/DONE"; else echo "rc=$rc $(date)" >> "$RUNS/FAILED"; touch "$RUNS/FAILED.${ARNOLD_TRIAL_ID:-0}"; fi
[ "${NNODES:-1}" -gt 1 ] && $XD/envs/supo/bin/ray stop >/dev/null 2>&1
echo "[job] done rc=$rc $(date)"
exit $rc
