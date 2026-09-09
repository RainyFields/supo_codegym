#!/bin/bash
# Process environment shared by the trainer and by the ray daemons. Sourced by train_codegym.sh AND by
# scripts/merlin/job_entrypoint.sh BEFORE `ray start` in multi-node jobs: ray workers inherit the raylet's
# environment, not the driver's (smoke #5: wandb "No API key" because the byted-wandb overlay was missing).
# Requires: EXP_NAME, PROJECT_DIR (defaults below), optional OVERLAY.
OVERLAY=${OVERLAY:-/home/tiger/xiaoxuan/envs/byted-wandb-overlay}
PROJECT_DIR=${PROJECT_DIR:-/home/tiger/xiaoxuan/supo_codegym}
case ":${PYTHONPATH:-}:" in *":$OVERLAY:"*) ;; *) export PYTHONPATH="$OVERLAY:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}";; esac
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXEC_TIMEOUT:-3600}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export CODEGYM_SPAWN_CONCURRENCY=${CODEGYM_SPAWN_CONCURRENCY:-16}
export WANDB_RUN_ID=${WANDB_RUN_ID:-${EXP_NAME:?EXP_NAME unset}}
export WANDB_RESUME=${WANDB_RESUME:-allow}
export WANDB_PROJECT=${WANDB_PROJECT:-supo_codegym}
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export CODEGYM_STEP_TIMEOUT=${CODEGYM_STEP_TIMEOUT:-10}
