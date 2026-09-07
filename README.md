# supo_codegym — SUPO Table-1 CodeGym replication with Qwen3.5-9B on verl

Replicates rows 1 (GRPO, 32K context) and 4 (SUPO, 4K×8 summarization) of Table 1 in
*Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management*
(arXiv 2510.06727), with **Qwen3.5-9B** instead of Qwen2.5-32B-Instruct. Design, config
table and deviations: `docs/REPLICATION_PLAN.md`. The paper released no code; the
summarization/multi-trajectory mechanics follow the paper (Alg. 1–2, Sec. 4.2) with the
authors' FoldAgent re-implementation (`~/xiaoxuan/external/FoldAgent`) as a structural reference.

## Layout

```
supo/agent_loop.py        SupoAgentLoop (verl agent loop): Alg. 2 rollout, GRPO when max_summaries=0
supo/codegym_env.py       in-process CodeGym env executor (sandbox subprocess per rollout) + FC parser
supo/prompts.py           v_sum summary prompt (App. B.2) + continuation template (App. C.1.1)
supo/verl_patches/        patch applied to ~/xiaoxuan/external/verl (branch supo): multi-trajectory
                          agent loops, overlong mask + dummy padding, `supo` advantage estimator
scripts/prepare_data.py   builds data/codegym_{train,eval}.parquet + codegym_env_codes.parquet
scripts/train_codegym.sh  ARM=grpo|supo training launcher (verl main_ppo, all paper hyper-parameters)
scripts/analyze_timing.py per-phase time-distribution table from a run log (bottleneck analysis)
scripts/merlin/           batch-job entrypoint, asset staging, spec generator, submit, resubmit loop
configs/agent.yaml        registers `supo_agent` for verl
tests/                    CPU tests: env replay, agent loop with a scripted LLM, advantage/padding
jobs/                     generated job specs + JOBS.tsv ledger
```

## Environment

* venv `~/xiaoxuan/envs/supo` (py3.12; verl HEAD `d040717` + patch; torch 2.11+cu130, vLLM 0.24,
  transformers 5.9, flash-attn 2.8.3, flash-linear-attention 0.5.2, gymnasium) — built with
  `UV_PROJECT_ENVIRONMENT=~/xiaoxuan/envs/supo uv sync --frozen --python 3.12 --extra fsdp --extra vllm`
  inside `~/xiaoxuan/external/verl`, then `uv pip install -e ~/xiaoxuan/supo_codegym gymnasium`.
* byted-wandb overlay `~/xiaoxuan/envs/byted-wandb-overlay` (prepended to PYTHONPATH by the train
  script) → merlin tracking project `supo_codegym`.

## Run

```bash
# CPU tests
~/xiaoxuan/envs/supo/bin/python tests/test_codegym_env.py --n 60
~/xiaoxuan/envs/supo/bin/python tests/test_agent_loop.py
~/xiaoxuan/envs/supo/bin/python tests/test_advantage_and_padding.py

# data (already built; deterministic)
~/xiaoxuan/envs/supo/bin/python scripts/prepare_data.py

# batch job (Merlin, group 765 ark-eng-algorithm): stage assets once, then submit
bash scripts/merlin/stage_assets.sh                      # repo snapshot -> HDFS job-assets
python3 scripts/merlin/make_job_spec.py --arm supo --gpu h100 > jobs/supo_h100.json
bash scripts/merlin/submit.sh jobs/supo_h100.json        # dry-run + create (needs user OK)

# time distribution of a run
python3 scripts/analyze_timing.py /mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym/job-runs/supo/train_*.log
```

Artifacts live on HDFS under `/mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym/` (checkpoints,
rollout dumps, validation dumps, job logs).
