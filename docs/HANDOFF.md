# SUPO × CodeGym replication — thinking arms + Qwen2.5-32B — HANDOFF (2026-09-16, PDT)

Status: **ALL RUNS COMPLETE, nothing running, everything committed and pushed.** This file supersedes the
chronological log in `2026-09-07_supo-codegym-gpu-shakeout_HANDOFF.md` (keep that one for incident detail).

## 1. What was asked and what was delivered
| ask | deliverable | where |
|---|---|---|
| SUPO Table-1 rows 1+4 (GRPO-32K vs SUPO-4K×8) on CodeGym with Qwen3.5-9B | done, report by the other session | `docs/report/` |
| enable thinking (packed + trained) for both arms | done | `docs/report_thinking_32b/` |
| Qwen2.5-32B-Instruct GRPO-32K arm on 2×8 H100, compare vs 9B | done | same report |
| compare with the CodeGym paper (arXiv 2509.17325) | done (protocol differences explained) | report §5 |
| check thinking cut-offs (Qwen prefix-unstable template?) | done: template ruled out; per-turn cap is the cause | report §3.3 / fig7 |
| push to GitHub | done | https://github.com/RainyFields/supo_codegym (master 72f1457) |

Report page (private artifact, same content as the PDF): https://claude.ai/code/artifact/a9c4f1e7-c716-42bf-9ce1-436214e0e7b6
Rebuild: `python scripts/make_report_thinking_32b.py && python scripts/render_report_thinking_32b_html.py`, then `~/.local/bin/tectonic docs/report_thinking_32b/REPORT.tex`.

## 2. Results (greedy pass@1, 128 held-out CodeGym tasks, 100 steps)
| arm | val@0 | final | peak | mean 80–100 |
|---|---|---|---|---|
| Qwen3.5-9B GRPO-32K | .766 | .867 | .875 @60 | .867 |
| Qwen3.5-9B SUPO-4K×8 | .633 | .758 | .828 @70 | .786 |
| Qwen3.5-9B GRPO-32K + thinking | .742 | .859 | .859 @100 | .805 |
| Qwen3.5-9B SUPO-4K×8 + thinking | .656 | .695 | .805 @70 | .753 |
| Qwen2.5-32B-Instruct GRPO-32K | .523 | .836 | .836 @100 | .820 |

Findings: thinking gives no gain at this scale (GRPO .859 vs .867; SUPO .695 vs .758). GRPO+think dips at steps
20–60 because 3–5 % of turns overrun the 4,096-token per-turn cap (never emit `</think>`, no action, ~4K of the
30,720 response budget burned); those turns sit in ~1/3 of episodes whose accuracy is .18–.46 vs .88–.96 for
clean episodes; by step 100 the policy thinks shorter (cut-offs 1.1 %) and val recovers. The Qwen template's
dropping of earlier `<think>` blocks on re-render does NOT apply: verl's continuous-token builder appends only
rendered deltas and every generation is fed the raw token stream (`supo/agent_loop.py:_generate`,
`verl/utils/tokenizer/continuous_token.py:_merge_context_token_ids`). 32B: starts 24 pts below 9B, gains +31,
ends 3 pts behind; final policies agree on 100/128 tasks. CodeGym paper's 30.1→81.0 is on a set filtered to
≤25 % base accuracy (972 evals, T=0.7, 256 calls) — levels are not comparable, trends are.

## 3. Artifacts on HDFS (`/mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym`)
| exp | complete checkpoints (global_step) | val dumps | job sid |
|---|---|---|---|
| grpo_codegym_qwen35-9b_32k | 35, **95** (100 is a stub) | outputs/…/val/{0..100}.jsonl (+35) | 74961c9b3ac5b42c |
| supo_codegym_qwen35-9b_4kx8 | **40** only (final weights lost; mirror stalled) | 0..100 | d934b2c9b712729b |
| grpo_codegym_qwen35-9b_32k_think | 90, **100** | 0..100 | 28b2b92f75a4a1ad |
| supo_codegym_qwen35-9b_4kx8_think | 90, **100** | 0..100 | 65f260ed2acb8e5d |
| grpo_codegym_qwen25-32b_32k | 80, **100** (2-node: .COMPLETE.0 + .COMPLETE.1) | 0..100 | a6ce27053bf73eda |
Rollout dumps: `rollouts/<exp>/train_step<N>_pid*.jsonl` (assistant text has `<think>` tags stripped by the
loop's `skip_special_tokens` decode; the val `output` field keeps them). wandb (byted-wandb → Merlin tracking):
https://ml.tiktok-row.net/experiment/tracking/detail?Id=project_20260907_b5e8830b, run names = exp names.
Job ledger with every smoke and its verdict: `jobs/JOBS.tsv`. Model weights for 32B: HF per-file curl in the
entrypoint (`MODEL_HF_ID`); chunked copy also at `job-assets/q32b_parts/` (126 parts, MANIFEST).

## 4. Infrastructure state (all in repo, assets on HDFS `job-assets/`)
- Stack: verl HEAD d040717 + SUPO patch + `dense_common` fused-CE dtype fix (f23ff6eb); cu129 venv tarball;
  image modelchef-gpu:1.0.0.38 (R535 driver; 1.0.0.54 has CUDA 13 compat if cu130 is ever wanted).
- Multi-node entrypoint (2×8): IPv4-first ray IP (bracketed v6 fallback), exit 48 on IPv6-only pods, per-trial
  `DONE.<trial>` markers (stale-DONE bug), ray worker ports 30000–39999, `train_env.sh` sourced before ray.
- Checkpoint mirror (`scripts/merlin/ckpt_sync.sh`): per-node readiness from the node's own `*_rank_*.pt`
  shards (driver actor can land on any node), explicit final upload in the drain, pod-local sync log pushed
  to HDFS every 60 s (`job-runs/<arm>/ckpt_sync.<rank>.log`), integer byte sums (mawk prints 1e11 as sci).
  Validated live: 197 GB/node in ~7 min (fuse fallback, 400–460 MB/s); hdfs CLI puts fail fast on those pods.
- Queue facts: ark H100 2-node requests waited 36 min–13 h; ark A100 pool currently hands out the
  fdbd:dc61:1a:* rack = IPv6-only (fdbd:dc61:8:* rack has IPv4 and works; needs NCCL_NET_GDR_LEVEL=LOC).
- Compliance env vars (HAS_TT_DATA=False etc.) were added to every spec by the other session (719cb74).

## 5. Open items / how to pick up
1. Direct comparison with the CodeGym paper: build a second eval split with their filters (10–256-call oracle
   band + base-model accuracy ≤25 % screen, T=0.7/top-p 0.95, 256-call budget) via `scripts/prepare_data.py`
   flags, then score the saved checkpoints (95/100 above) with `trainer.val_only`-style runs.
2. Thinking at 32K needs a stop condition that closes thinking (separate think/answer budgets or a larger
   per-turn cap with the overlong mask) before re-running the thinking arms.
3. SUPO-4K×8 (non-thinking) final weights are lost (only step 40). A rerun with the current mirror would
   recover them if post-hoc eval on final weights is needed (~11 h on 1×8 H100).
4. Low priority: why hdfs CLI puts fail instantly on the 32B pods (empty error); IPv6 preflight may be
   over-strict now that the stale-DONE bug is fixed (smoke #12's "hang" was that bug, not IPv6).
5. PDF pages were never rasterized on the devbox (no poppler); open `REPORT.pdf` from GitHub once to eyeball.

## 6. Resume commands
```
cd ~/xiaoxuan/supo_codegym && git pull                      # repo (remote: RainyFields/supo_codegym)
tail -50 ~/xiaoxuan/handoffs/2026-09-07_supo-codegym-gpu-shakeout_HANDOFF.md   # incident log
cat jobs/JOBS.tsv | column -t -s$'\t' | tail -20             # every job + verdict
python scripts/merlin/make_job_spec.py --arm grpo --gpu h100 --nodes 2 --extra_env "MODEL_HF_ID=Qwen/Qwen2.5-32B-Instruct,MODEL_TAG=qwen25-32b,MODEL_MIN_SHARDS=17,SP_SIZE=4,ROLLOUT_TP=4,MAX_MODEL_LEN_OVERRIDE=32768" --save_freq 20 > jobs/<name>.json
bash /tmp/supo_stage/stage_small.sh                          # re-stage repo tarball + entrypoint after ANY script change (devbox-local script; recreate from the handoff if /tmp is gone)
~/.merlin-cli/bin/merlin-cli --control-plane i18n-tt job-v2 runs create --from-file jobs/<name>.json
```

## 7. Addendum — SUPO-4K×8 rerun to recover final weights (Sep 16, 15:45)
User asked to recover the lost SUPO final weights. Approach: continue from the original run's complete
`global_step_40` (the same lineage run 5 resumed from; verl restores the dataloader position) under a NEW exp
`supo_codegym_qwen35-9b_4kx8_rerun` so the original dumps stay intact. The devbox hdfs CLI could not copy the
113 GB checkpoint ("Protocol family unavailable" on every JVM flag), so instead `ckpt_restore` gained
`RESTORE_FROM_EXP=<exp>` (commit 26244a4): when the rerun's own checkpoint dir has no complete step, the pod
restores from `checkpoints/<exp>/` directly (bash dynamic scoping; the mirror loop still writes only to the
rerun dir, so the source is never pruned). Job: `jobs/supo_rerun_h100.json` (1×8 H100, SAVE_FREQ 10, THINK 0),
sid **3d645925cb1c719f** (also in `/tmp/supo_stage/supo_rerun_sid.txt`, `jobs/JOBS.tsv`). Expect ~40 min
restore + ~7 h for steps 41–100; final weights at `checkpoints/supo_codegym_qwen35-9b_4kx8_rerun/global_step_100`
with `.COMPLETE`; val in `outputs/supo_codegym_qwen35-9b_4kx8_rerun/val/`; logs `job-runs/supo_rerun/`.
