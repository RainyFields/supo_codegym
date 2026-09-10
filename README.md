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

* venv `~/xiaoxuan/envs/supo` (py3.12; verl HEAD `d040717` + patch; **torch 2.11+cu129**, vLLM 0.24+cu129,
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

### CUDA note (2026-09-07): pods run driver R535 → cu129, not cu130
Both ark queues (H100 535.129.03, A100 535.161.08, Debian 12 image, `/usr/local/cuda-12.9/compat`
libcuda 575.57.08) expose CUDA driver API **12.9** — CUDA 13 forward-compat needs R570+, so verl
HEAD's default cu130 stack fails at `torch.cuda` init ("driver too old", job 510a802886e902f6).
The venv was rebuilt in place (old one kept at `~/xiaoxuan/envs/supo-cu130`):
```
cp -a envs/supo envs/supo-cu129
uv pip install --python envs/supo-cu129/bin/python --reinstall --index-url https://download.pytorch.org/whl/cu129 \
   --extra-index-url https://bytedpypi.byted.org/simple/ --index-strategy unsafe-best-match \
   torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129
uv pip install --no-deps vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl   # GitHub release asset
uv pip install cupy-cuda12x==14.0.1 'cuda-python==12.9.*'
uv pip uninstall cupy-cuda13x cuda-toolkit nvidia-*-cu13 nvidia-cublas ... (all unsuffixed cu13 libs)
uv pip install --reinstall <every remaining nvidia-*-cu12 pkg>   # cu13 uninstall clobbers shared .so names
uv pip uninstall flash_attn                                       # wheelhouse wheel links libcudart.so.13
```
flash-attn 2.8.3 has no cu12/torch-2.11 wheel anywhere (official, verl wheelhouse, mjun0812) → the job
entrypoint builds it once in the pod (96 cores, `FLASH_ATTN_CUDA_ARCHS=80;90`, staged CUDA 12.9 nvcc
toolchain `job-assets/cuda-12.9-toolchain.tar.gz`) and caches it at `job-assets/wheels/cu129torch2.11/`.
Devbox build is impractical: this workspace is cgroup-capped at 8 CPUs / 32 GB (nvcc OOM-kills).

**Update 2026-09-08 (probe cad65f3e2949d9dd):** the CUDA 13.0 forward-compat `libcuda` (580.126.20, from
`/usr/local/cuda-13.0/compat` on the devbox, staged as `job-assets/cuda-13.0-compat.tar.gz`) DOES initialize on
the R535 pods — `cuInit` returns 0 with driver API 13000 and the H100 enumerated — despite NVIDIA's support
table listing R570+ only. So cu130 is probably usable here by exporting `LD_LIBRARY_PATH=<compat dir>` in the
entrypoint (untested beyond cuInit: no torch/NCCL/vLLM run yet). The cu130 venv is kept at
`~/xiaoxuan/envs/supo-cu130`; spec `jobs/compat13probe_h100.json` is the probe.

**Update 2026-09-08 13:45 (probes 1dd4f29519f37e88 / afdd17b9e6019e0d): the cu130 failure was the job IMAGE TAG.**
Batch jobs use `modelchef-gpu:1.0.0.38` (only `/usr/local/cuda-12.9` + CUDA 12.9 compat libcuda 575.57.08); the
Workbench devbox template is `modelchef-gpu:1.0.0.54` (`/usr/local/cuda-13.0` + CUDA 13.0 compat libcuda
580.126.20). On the same R535 H100 node, the original cu130 venv runs unmodified on image 1.0.0.54
(`torch.zeros(1,device='cuda')+1` OK, bf16 matmul 188 TFLOP/s), and also on 1.0.0.38 if the 13.0 compat dir is
staged and put on LD_LIBRARY_PATH (202 TFLOP/s). NCCL / flash-attn / vLLM verdicts: see probes #5 in jobs/JOBS.tsv.
To move a job to cu130: set `image_url` to `...modelchef-gpu:1.0.0.54` and restore `envs-supo-cu130` (23 chunks
under `job-assets/cu130_parts/`).

**Probes #6 (6bd014dd035ba227 img 1.0.0.38+staged compat, d2f7c982364f950c img 1.0.0.54): full cu130 stack PASSES on
R535** — torch matmul 203–212 TFLOP/s, cuSOLVER/cuDNN, flash-attn 2.8.3 cu13 dense+varlen, NCCL init+all_reduce,
vLLM 0.24 cu130 serving Qwen3.5-9B (≈280–300 tok/s for 8 prompts). Only blemish: SIGSEGV (exit 139) inside
`dist.destroy_process_group()` AFTER a successful all_reduce, on both images (cu129 does not show it) — a
teardown crash, not a training one; test in isolation before relying on cu130 for long runs.

## Reports

- `docs/report/` — SUPO Table-1 replication on CodeGym with Qwen3.5-9B (GRPO-32K vs SUPO-4K×8): `REPORT.pdf`, `REPORT.md`, figures + data.
- `docs/report_thinking_32b/` — thinking-mode arms (Qwen3.5-9B) and the Qwen2.5-32B-Instruct GRPO-32K arm: `REPORT.pdf`, `REPORT.md`, `report.html`, `assets/` (every figure paired with its data; see `assets/README.md`). Rebuild with `python scripts/make_report_thinking_32b.py`.
- Job ledger: `jobs/JOBS.tsv`.
