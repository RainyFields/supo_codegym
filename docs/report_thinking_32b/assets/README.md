# Assets for the thinking + 32B report

All files were produced by `scripts/make_report_thinking_32b.py` on 2026-09-16 11:30 PDT from
`$PROJECT_ROOT = /mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym`. Validation numbers come from the greedy validation dumps
`outputs/<exp>/val/<step>.jsonl` (one JSON record per held-out task; fields used: score, finished, overlong,
num_tool_calls, num_summaries, num_trajs, num_invalid_calls, rollout_response_tokens, num_steps; files with <100
records are skipped). Arms -> exp names: {"9B GRPO-32K": "grpo_codegym_qwen35-9b_32k", "9B SUPO-4Kx8": "supo_codegym_qwen35-9b_4kx8", "9B GRPO-32K+think": "grpo_codegym_qwen35-9b_32k_think", "9B SUPO-4Kx8+think": "supo_codegym_qwen35-9b_4kx8_think", "32B GRPO-32K": "grpo_codegym_qwen25-32b_32k"}.

| file | backs | contents / schema | provenance |
|---|---|---|---|
| fig1_val_acc_all_arms.{png,pdf,csv} | Fig. 1 | arm, step, acc (mean score over n tasks), se = sqrt(acc(1-acc)/n), n | val dumps |
| fig2_overlong_calls_tokens.{png,pdf,csv} | Fig. 2 | arm, step, overlong (fraction of tasks hitting the episode/context cap), tool_calls (mean per task), response_tokens (mean generated tokens per task), summaries (mean per task), invalid_calls, finished | val dumps |
| fig3_thinking_stats.{png,pdf,csv} | Fig. 3 | arm, step, n_tasks (128), assistant_turns (all turns over the 128 greedy validation episodes), frac_turns_with_think, frac_think_unterminated (thinking turn cut before </think>), think_tokens_mean/median/p90/max (Qwen3.5-9B tokenizer count of the text before </think>, opening tag removed), think_tokens_total_per_task | outputs/<exp>/val/<step>.jsonl field `output` (decoded generation incl. chat-template role text), turns split on "\nassistant\n" |
| fig4_training_curves.{png,pdf,csv} | Fig. 4 | arm, step, and the batch-level trainer metrics ['critic/score/mean', 'actor/pg_loss', 'actor/entropy', 'actor/grad_norm', 'response_length/mean', 'response_length/clip_ratio', 'training/num_turns/mean', 'timing_s/step', 'timing_s/gen', 'timing_s/update_actor', 'perf/throughput'] (verl metric names; response_length in tokens; timing in seconds) | job-runs/<arm>/train_*.log: 'step:N - key:value' lines, ray/ANSI prefixes stripped, last occurrence per step wins |
| fig5_gain_over_base.{png,pdf,csv} | Fig. 5 | arm, step, acc, base (= acc at step 0), gain = acc - base | val dumps |
| fig7_think_cutoffs.{png,pdf,csv} | Fig. 7, Table 5 | arm, step, n_episodes, turns (sum of num_steps), cut_off_turns (sum over episodes of max(0, num_steps - output.count("</think>"))), frac_turns_cut, frac_episodes_with_cut, n_episodes_with_cut, overlong_given_cut / overlong_given_none (mean overlong flag over episodes with >= 1 / 0 cut-off turns), acc_given_cut / acc_given_none (same for score), resp_tokens_per_turn (mean of rollout_response_tokens / num_steps) | val dumps of grpo_codegym_qwen35-9b_32k_think, fields output, num_steps, overlong, score, rollout_response_tokens; single-segment SUPO+think check in ../summary.json |
| fig6_task_agreement.{png,pdf,json} | Fig. 6 | 2x2 contingency of solved/unsolved per task for 32B@last vs 9B@last; matched_tasks; tasks unsolved by 9B at step 0 and how many of those each final policy solves | val dumps, tasks matched by md5 of the prompt text after stripping role markers/<think>/whitespace |
| val_metrics.csv | Tables 2-3 | arm, step, n, se + the nine per-task mean fields above | val dumps |
| ../summary.json | all tables | per-arm first/last/peak/mean-80-100 accuracy, overlong, tool calls, gain, train-log path and median step time; thinking stats; agreement; paper reference numbers | derived from the above |
| ../trajectories.json | Sec. 5 | three sampled training rollouts of the 32B arm (step 0 failure, step 100 success and failure; record minus messages + pretty rendering) and the first four turns of one greedy validation episode of the 9B thinking arm at its last checkpoint showing the think segments | rollouts dumps; val dumps |
| ../tables.md | Tables 2-3 | markdown tables quoted in the report | derived |
