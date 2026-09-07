# SUPO Table-1 (CodeGym) replication plan — Qwen3.5-9B on verl

Paper: Lu, Sun, Du, Ling, Yao, Liu, Chen. *Scaling LLM Multi-turn RL with End-to-end
Summarization-based Context Management* (arXiv 2510.06727, Oct 2025; ACL 2026 as *Beyond the
Context Window: Scaling Agentic RL via End-to-end Optimized Context Compression*). No official
code was released (checked arXiv, ACL Anthology, HF, authors' GitHub, 2026-09-07).

Goal: reproduce Table 1 rows 1 (GRPO, 32K working context) and 4 (SUPO, 4K working context ×
up to 8 trajectories = 32K effective) on CodeGym, replacing Qwen2.5-32B-Instruct with
**Qwen3.5-9B** (hybrid Gated-DeltaNet/attention, non-thinking mode).

| Paper (Qwen2.5-32B-Instruct) | Working | Effective | Acc. before | Acc. after | tool calls |
|---|---|---|---|---|---|
| GRPO | 32K | 32K | 32.0% | 44.5% | 52.1 |
| SUPO | 4K | 32K (4K×8) | 32.8% | 47.7% (+3.2) | 54.7 |

## Configuration (paper Sec. 5.1 → ours)

| Item | Paper | This replication | Notes |
|---|---|---|---|
| Policy | Qwen2.5-32B-Instruct | Qwen3.5-9B, `enable_thinking=False` | user choice; empty `<think></think>` block is part of the prompt |
| Env | CodeGym (12,800 train / 128 eval) | HF `VanishD/CodeGym` (community reproduction), 12,800 train / 128 eval | exact lists unreleased; see `scripts/prepare_data.py` |
| Eval split | different seeds, more turns on avg. | 128 held-out seeds, ref. solution 40–90 calls (mean 59.8 vs train 28.1) | |
| Prompt filter | – | initial prompt ≤ 1536 tokens (81 of 80,130 tasks dropped) | needed for the 4K arm, applied to both arms |
| Batch / group | B=128, G=8 | 128 / 8 | 1 epoch = 100 steps |
| Mini-batch | not stated | 32 prompts (×8 = 256 trajectories) → 4 updates/step | assumption |
| LR | 1e-6 constant | 1e-6 constant, no warmup, wd 0 | |
| Clip | ε_low .20, ε_high .28 | same (`clip_ratio_low/high`) | |
| KL / entropy | none | none | |
| Loss agg. | token-level (eq. 2) | `loss_agg_mode=token-mean` | |
| Advantage | eq. (3): per-rollout group stats, broadcast to trajectories | `compute_supo_advantage` (gen_uid de-dup) | identical to GRPO when 1 traj/rollout |
| Overlong mask | mask rollouts w/o final answer within H or S | `response_mask=0` for those rows (trainer patch) | still count in group mean/std |
| Dummy padding | pad N to multiple of B_mini | pad to multiple of world size (8), mask 0 | smaller last mini-batch is fine in verl |
| H (max steps) | 100 | 100 LLM calls incl. summaries | |
| L (summary threshold) | 0.95 W | 0.95 W (3891 tokens for W=4096) | `context_limit` |
| S (max summaries) | 7 (CodeGym) | 7 | GRPO arm: 0 |
| Working context W | 4K (SUPO) / 32K (GRPO) | 4096 / 32768 tokens (prompt+response) | |
| Per-turn max tokens (L_A) | not stated | 1024 (action turn), 1024 (summary) | assumption |
| Summary prompt v_sum | Appendix B.2 (CodeGym) | verbatim, sent as a `user` turn | Qwen3.5 template forbids mid-conversation `system` turns |
| Next-traj prompt | Appendix C.1.1 ("We are in the following stage…") | original user prompt + `\n\nWe are in the following stage of solving the problem:\n<summary>` | |
| Discard last round | Alg. 2 line 11 | yes (snapshot/restore before the action turn) | env state keeps the executed action |
| Observation role | "Tool:" | `user` turn (plain text) | avoids `<tool_response>` wrapping/synthetic tool-call rendering |
| Rollout sampling | T=1 (assumed) | temperature 1.0, top-p 1 | |
| Eval | pass@1 on 128 | greedy, 1 sample, every 10 steps + before training | |
| Rollout infra | vLLM async agent loop | verl `rollout.mode=async` (agent loop) + **synchronous** on-policy training | user requirement |
| Checkpoints | – | every 10 steps to HDFS, keep 2, `resume_mode=auto` | 4-h queue reclamation |

## Compute plan

* 1 node × 8 GPUs (H100 first, A100 fallback), FSDP2 actor, vLLM TP=1 (8 replicas), no ref model.
* Qwen3.5-9B KV cache is cheap (8 full-attention layers × 4 KV heads × 256 dim ≈ 32 KB/token), so
  32K-context rollouts are inexpensive; training on 32K sequences uses `use_dynamic_bsz` with one
  sequence per micro-batch and gradient checkpointing (`SP_SIZE=2` available if OOM).
* Per step: 1024 rollouts; SUPO produces up to 8× as many (shorter) trajectory rows.
* Estimated 10–20 min/step → 100 steps ≈ 1–1.5 days per arm on H100 (A100 ~2×).

## Infra layout (house conventions, COMMON_SETUP.md)

* code: `~/xiaoxuan/supo_codegym` (this repo), patched verl: `~/xiaoxuan/external/verl` (branch `supo`,
  patch file `supo/verl_patches/0001-supo-multi-trajectory.patch` on upstream `d040717`)
* venv: `~/xiaoxuan/envs/supo` (py3.12, torch 2.11+cu130, vllm 0.24, transformers 5.9, flash-attn 2.8.3, fla 0.5.2)
* byted-wandb overlay: `~/xiaoxuan/envs/byted-wandb-overlay` (PYTHONPATH-prepended → merlin tracking)
* HDFS `$PROJECT_ROOT=/mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym`: `data/raw`, `job-assets`,
  `job-runs/<arm>`, `checkpoints/<exp>`, `rollouts/<exp>` (jsonl dumps), `outputs/<exp>/val`
* base model: `/mnt/hdfs/mlsys/models/Qwen3.5-9B` (staged to pod `/tmp/models` at job start)
* jobs: Merlin batch jobs (`scripts/merlin/*`), group 765 ark-eng-algorithm, cluster 4, image
  `modelchef-gpu:1.0.0.38`; W&B project `supo_codegym`, runs `grpo_codegym_qwen35-9b_32k`,
  `supo_codegym_qwen35-9b_4kx8`.

## Deviations / risks (keep updated)

1. Model: 9B hybrid-attention Qwen3.5 instead of 32B Qwen2.5 — absolute numbers will differ; the
   target is the *ordering* and the mechanism (SUPO ≥ GRPO at 8× shorter working context, similar
   or more tool calls).
2. Dataset split re-created from the public dataset (paper's lists unreleased).
3. Mini-batch size, per-turn token caps, observation role, summary-turn role: assumptions above.
4. CUDA 13 wheels (torch 2.11+cu130) need driver ≥ 580 on the pod — verified only at the first
   GPU shakeout; fallback would be rebuilding the venv on an older verl (v0.9.0) stack.
5. Early stop after 10 consecutive turns without a function call (masked as overlong, as the
   paper would at H) to save compute.
