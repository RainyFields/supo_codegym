#!/usr/bin/env python
"""Build the CodeGym train/eval split used for the SUPO Table-1 replication.

Paper (Sec. 5.1): 12,800 training problems from CodeGym; a held-out evaluation
set that (i) comes from different seed coding problems than the training set and
(ii) needs more function-calling turns on average. The paper's exact lists were
never released, so this script re-creates the split deterministically from the
public dataset (HF VanishD/CodeGym, task_en_instruction_en_env + envs_en):

  * one env file per seed problem (code_id); tasks are (env, task-config) pairs
  * eval: 128 tasks drawn from 128 distinct held-out seeds whose reference
    solution needs 40..90 tool calls (harder than the training average of ~30)
  * train: 12,800 tasks from the remaining seeds, reference length <= 80 calls
    (the rollout budget is H=100 LLM calls), at most 2 tasks per seed
  * both: initial prompt <= MAX_PROMPT_TOKENS Qwen3.5 tokens (needed for the 4K
    working-context SUPO arm; applied identically to the GRPO arm)

Outputs (parquet, verl RLHFDataset schema + `ability` env string):
  data/codegym_train.parquet, data/codegym_eval.parquet, data/codegym_env_codes.parquet
"""

import argparse
import collections
import glob
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=os.path.expanduser("~/xiaoxuan/external/codegym_hf"))
    ap.add_argument("--tokenizer", default="/mnt/hdfs/mlsys/models/Qwen3.5-9B")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--n_train", type=int, default=12800)
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--train_max_rounds", type=int, default=80)
    ap.add_argument("--eval_min_rounds", type=int, default=40)
    ap.add_argument("--eval_max_rounds", type=int, default=90)
    ap.add_argument("--max_prompt_tokens", type=int, default=1536)
    ap.add_argument("--max_tasks_per_seed", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261007)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)

    files = sorted(glob.glob(os.path.join(args.raw_dir, "task_en_instruction_en_env", "*.parquet")))
    cols = ["ability", "code_id", "data_source", "extra_info", "prompt", "reward_model"]
    tasks = pa.concat_tables([pq.read_table(f, columns=cols) for f in files]).to_pylist()
    print(f"raw tasks: {len(tasks)}")

    env_tab = pq.read_table(os.path.join(args.raw_dir, "envs_en", "train-00000-of-00001.parquet")).to_pylist()
    env_codes = {f"{r['source']}__{r['env_name']}": r["env_code"] for r in env_tab}

    # prompt-length filter (system + user rendered through the chat template, thinking off)
    def n_prompt_tokens(prompt):
        msgs = [{"role": m["role"], "content": m["content"]} for m in prompt]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if hasattr(ids, "keys"):  # transformers>=5 returns a BatchEncoding
            ids = ids["input_ids"]
        return len(ids)

    kept = []
    n_bad_env = n_long = 0
    for t in tasks:
        env_key = t["ability"].split("@")[1]
        if env_key not in env_codes:
            n_bad_env += 1
            continue
        t["env_key"] = env_key
        t["ref_rounds"] = int(t["extra_info"]["solve_fc_round"])
        t["prompt_tokens"] = n_prompt_tokens(t["prompt"])
        if t["prompt_tokens"] > args.max_prompt_tokens:
            n_long += 1
            continue
        kept.append(t)
    print(f"kept {len(kept)} (missing env {n_bad_env}, prompt too long {n_long})")

    by_seed = collections.defaultdict(list)
    for t in kept:
        by_seed[t["code_id"]].append(t)
    seeds = sorted(by_seed)
    rng.shuffle(seeds)

    # eval: distinct held-out seeds with a reference trajectory in [eval_min, eval_max]
    eval_tasks, eval_seeds = [], set()
    for s in seeds:
        cands = [t for t in by_seed[s] if args.eval_min_rounds <= t["ref_rounds"] <= args.eval_max_rounds]
        if cands:
            eval_tasks.append(rng.choice(cands))
            eval_seeds.add(s)
        if len(eval_tasks) >= args.n_eval:
            break
    assert len(eval_tasks) == args.n_eval, len(eval_tasks)

    # train: remaining seeds, <= max_tasks_per_seed each, ref length <= train_max_rounds
    pool = []
    for s in seeds:
        if s in eval_seeds:
            continue
        cands = [t for t in by_seed[s] if t["ref_rounds"] <= args.train_max_rounds]
        rng.shuffle(cands)
        pool.extend(cands[: args.max_tasks_per_seed])
    rng.shuffle(pool)
    train_tasks = pool[: args.n_train]
    assert len(train_tasks) == args.n_train, len(train_tasks)

    def to_rows(ts, split):
        rows = []
        for i, t in enumerate(ts):
            rows.append(
                {
                    "data_source": "codegym_v1",
                    "prompt": [{"role": m["role"], "content": m["content"]} for m in t["prompt"]],
                    "ability": t["ability"],
                    "reward_model": {"style": "agent_env", "ground_truth": ""},
                    "extra_info": {
                        "split": split,
                        "index": i,
                        "code_id": t["code_id"],
                        "env_key": t["env_key"],
                        "ref_rounds": t["ref_rounds"],
                        "prompt_tokens": t["prompt_tokens"],
                    },
                    "agent_name": "supo_agent",
                }
            )
        return rows

    os.makedirs(args.out_dir, exist_ok=True)
    for name, ts in [("codegym_train", train_tasks), ("codegym_eval", eval_tasks)]:
        rows = to_rows(ts, name.split("_")[1])
        pq.write_table(pa.Table.from_pylist(rows), os.path.join(args.out_dir, f"{name}.parquet"))
        rr = [t["ref_rounds"] for t in ts]
        pt = [t["prompt_tokens"] for t in ts]
        print(
            f"{name}: n={len(ts)} seeds={len(set(t['code_id'] for t in ts))} "
            f"ref_rounds mean={sum(rr)/len(rr):.1f} median={sorted(rr)[len(rr)//2]} "
            f"prompt_tokens mean={sum(pt)/len(pt):.0f} max={max(pt)}"
        )
    need = sorted(set(t["env_key"] for t in train_tasks + eval_tasks))
    pq.write_table(
        pa.Table.from_pylist([{"env_key": k, "env_code": env_codes[k]} for k in need]),
        os.path.join(args.out_dir, "codegym_env_codes.parquet"),
    )
    print(f"env codes: {len(need)}")
    with open(os.path.join(args.out_dir, "split_manifest.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "eval_code_ids": sorted(eval_seeds),
                "n_train": len(train_tasks),
                "n_eval": len(eval_tasks),
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
