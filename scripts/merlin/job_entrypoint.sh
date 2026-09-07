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
MODEL_SRC=/mnt/hdfs/mlsys/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/models/Qwen3.5-9B
if [ ! -f "$MODEL_LOCAL/model.safetensors.index.json" ] || [ "$(ls "$MODEL_LOCAL"/*.safetensors 2>/dev/null | wc -l)" -lt 4 ]; then
  echo "[job] staging model to $MODEL_LOCAL ($(date))..."
  mkdir -p "$MODEL_LOCAL" && cp "$MODEL_SRC"/* "$MODEL_LOCAL"/ || { echo "[job] FATAL: model staging failed"; exit 45; }
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

# ── train ───────────────────────────────────────────────────────────────────
export MODEL_PATH=$MODEL_LOCAL
export PROJECT_DIR=$XD/supo_codegym VERL_DIR=$XD/external/verl VENV=$XD/envs/supo
export RUN_TAG=${RUN_TAG:-}
LOG=$RUNS/train_$(date +%Y%m%d_%H%M%S).log
echo "[job] launching training, log=$LOG"
bash "$XD/supo_codegym/scripts/train_codegym.sh" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -eq 0 ]; then touch "$RUNS/DONE"; else echo "rc=$rc $(date)" >> "$RUNS/FAILED"; fi
echo "[job] done rc=$rc $(date)"
exit $rc
