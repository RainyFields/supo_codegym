#!/bin/bash
# Train Table-1 CodeGym arms of SUPO (arXiv 2510.06727) with Qwen3.5-9B on verl.
#   ARM=grpo  -> row 1: vanilla multi-turn GRPO, 32K working context, no summarization
#   ARM=supo  -> row 4: SUPO, 4K working context, up to 7 summaries (32K effective)
# Paper hyper-parameters (Sec. 5.1): B=128 prompts, G=8 rollouts, 1 epoch over 12,800
# problems (=100 steps), lr 1e-6 constant, no KL / entropy loss, eps_low .20 eps_high .28,
# summarization threshold L = 0.95 W, H = 100 steps. Mini-batch (32 prompts -> 4 updates
# per step) and per-turn token caps are not given in the paper (see docs/REPLICATION_PLAN.md).
set -euo pipefail

ARM=${ARM:?set ARM=grpo|supo}
PROJECT_DIR=${PROJECT_DIR:-/home/tiger/xiaoxuan/supo_codegym}
VERL_DIR=${VERL_DIR:-/home/tiger/xiaoxuan/external/verl}
VENV=${VENV:-/home/tiger/xiaoxuan/envs/supo}
OVERLAY=${OVERLAY:-/home/tiger/xiaoxuan/envs/byted-wandb-overlay}   # byted-wandb -> merlin tracking
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym}
MODEL_PATH=${MODEL_PATH:-/mnt/hdfs/mlsys/models/Qwen3.5-9B}
DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data}
N_GPUS=${N_GPUS:-8}
NNODES=${NNODES:-1}
TOTAL_STEPS=${TOTAL_STEPS:-100}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
TRAIN_BS=${TRAIN_BS:-128}
N_ROLLOUT=${N_ROLLOUT:-8}
MINI_BS=${MINI_BS:-32}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.7}
AGENT_WORKERS=${AGENT_WORKERS:-8}
RUN_TAG=${RUN_TAG:-}
LOGGER=${LOGGER:-"['console','wandb']"}
EXTRA_ARGS=${EXTRA_ARGS:-}

case "$ARM" in
  supo) WORKING_CONTEXT=4096;  MAX_SUMMARIES=7; PROMPT_LEN=4096; RESP_LEN=5120;  MAX_MODEL_LEN=10240; TAG=4kx8 ;;
  grpo) WORKING_CONTEXT=32768; MAX_SUMMARIES=0; PROMPT_LEN=2048; RESP_LEN=30720; MAX_MODEL_LEN=34816; TAG=32k
        # 35K-token sequences OOM the FSDP update (74 GB in the logits/entropy backward, run 48d248234f287e0c):
        # fused linear+log-prob kernels (verl monkey_patch supports qwen3_5) never materialize full logits.
        USE_FUSED=${USE_FUSED:-True}; OPT_OFFLOAD=${OPT_OFFLOAD:-True} ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac
EXP_NAME=${EXP_NAME:-${ARM}_codegym_qwen35-9b_${TAG}${RUN_TAG}}
CKPT_DIR=${CKPT_DIR:-$PROJECT_ROOT/checkpoints/$EXP_NAME}
DUMP_DIR=${DUMP_DIR:-$PROJECT_ROOT/rollouts/$EXP_NAME}
VAL_DUMP_DIR=${VAL_DUMP_DIR:-$PROJECT_ROOT/outputs/$EXP_NAME/val}
mkdir -p "$CKPT_DIR" "$DUMP_DIR" "$VAL_DUMP_DIR"

# max tokens per micro-batch: one full sequence (prompt+response) per GPU with dynamic bsz
SEQ_LEN=$((PROMPT_LEN + RESP_LEN))
PPO_MAX_TOKENS=${PPO_MAX_TOKENS:-$SEQ_LEN}
LOGP_MAX_TOKENS=${LOGP_MAX_TOKENS:-$((2 * SEQ_LEN))}

export PYTHONPATH="$OVERLAY:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
# vLLM's executor kills the engine when one execute_model RPC exceeds this (default 300 s); a
# host-side stall (sandbox spawn burst) must not take the rollout engine down. Ray workers inherit it.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXEC_TIMEOUT:-3600}
export CODEGYM_SPAWN_CONCURRENCY=${CODEGYM_SPAWN_CONCURRENCY:-16}
export WANDB_RUN_ID=${WANDB_RUN_ID:-$EXP_NAME}
export WANDB_RESUME=${WANDB_RESUME:-allow}
export WANDB_PROJECT=${WANDB_PROJECT:-supo_codegym}
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export CODEGYM_STEP_TIMEOUT=${CODEGYM_STEP_TIMEOUT:-10}

echo "[train] ARM=$ARM EXP=$EXP_NAME W=$WORKING_CONTEXT S=$MAX_SUMMARIES steps=$TOTAL_STEPS ckpt=$CKPT_DIR $(TZ=America/Los_Angeles date)"
cd "$VERL_DIR"
exec "$VENV/bin/python" -m verl.trainer.main_ppo ${HYDRA_EXTRA:-} \
  data.train_files="$DATA_DIR/codegym_train.parquet" \
  data.val_files="$DATA_DIR/codegym_eval.parquet" \
  data.train_batch_size=$TRAIN_BS \
  data.max_prompt_length=$PROMPT_LEN \
  data.max_response_length=$RESP_LEN \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.return_raw_chat=True \
  data.shuffle=True \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  algorithm.adv_estimator=supo \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BS \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKENS \
  actor_rollout_ref.actor.clip_ratio_low=0.20 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.calculate_entropy=True \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPT_OFFLOAD:-False} \
  actor_rollout_ref.model.use_fused_kernels=${USE_FUSED:-False} \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=${FUSED_BACKEND:-triton} \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE:-1} \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
  actor_rollout_ref.rollout.prompt_length=$PROMPT_LEN \
  actor_rollout_ref.rollout.response_length=$RESP_LEN \
  actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.n=$N_ROLLOUT \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOGP_MAX_TOKENS \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.agent.default_agent_loop=supo_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_DIR/configs/agent.yaml" \
  actor_rollout_ref.rollout.agent.num_workers=$AGENT_WORKERS \
  "+actor_rollout_ref.rollout.custom={working_context: $WORKING_CONTEXT, max_summaries: $MAX_SUMMARIES, summary_ratio: 0.95, max_steps: 100, turn_max_tokens: 1024, summary_max_tokens: 1024, env_code_path: $DATA_DIR/codegym_env_codes.parquet, dump_dir: $DUMP_DIR, dump_every: 64}" \
  trainer.logger="$LOGGER" \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=$NNODES \
  trainer.total_training_steps=$TOTAL_STEPS \
  trainer.total_epochs=1 \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.val_before_train=$VAL_BEFORE_TRAIN \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.resume_mode=auto \
  trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP:-1} \
  trainer.validation_data_dir="$VAL_DUMP_DIR" \
  trainer.log_val_generations=8 \
  trainer.balance_batch=True \
  trainer.critic_warmup=0 \
  $EXTRA_ARGS
