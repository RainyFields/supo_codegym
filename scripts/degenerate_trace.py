import asyncio, json, sys, os
sys.path.insert(0, 'tests'); sys.path.insert(0, '.')
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from test_agent_loop import make_loop, FakeServer
from supo.codegym_env import parse_env_str
tok = AutoTokenizer.from_pretrained('/mnt/hdfs/mlsys/models/Qwen3.5-9B')
tasks = pq.read_table('data/codegym_train.parquet', columns=["ability","prompt","extra_info"]).to_pylist()
task = [t for t in tasks if 'Codeforces_5537_I__MinPathCostEnv' in t['ability']][0]
print("TASK ability:", task['ability'][:160])
print("TASK user prompt (first 500 chars):", task['prompt'][-1]['content'][:500].replace('\n',' | '))
actions = task['extra_info'].get('std_traj') or []
print("reference solution: %d actions, first 3: %s" % (len(actions), [str(a)[:80] for a in actions[:3]]))
server = FakeServer(tok, actions if actions else ['{"name":"Observe","parameters":{}}'])
messages = [{"role": m["role"], "content": m["content"]} for m in task["prompt"]]
async def main():
    loop = make_loop(tok, server, working_context=4096, max_summaries=7, env_code_path='data/codegym_env_codes.parquet')
    outs = await loop.run({"temperature": 1.0, "max_tokens": 1024, "logprobs": 1}, raw_prompt=messages, ability=task["ability"], index=0, validate=False, global_steps=3)
    outs = outs if isinstance(outs, list) else [outs]
    print("\n=== DEGENERATE ROLLOUT OUTPUT: %d row(s) ===" % len(outs))
    for o in outs:
        print(" prompt_ids: %d tokens (ends with %r)" % (len(o.prompt_ids), tok.decode(o.prompt_ids[-12:])))
        print(" response_ids:", o.response_ids, "->", repr(tok.decode(o.response_ids)))
        print(" response_mask:", o.response_mask, " response_logprobs:", o.response_logprobs)
        print(" reward_score:", o.reward_score, " num_turns:", o.num_turns)
        print(" extra_fields:", {k: v for k, v in o.extra_fields.items() if k != 'reward_extra_info'})
        print(" reward_extra_info:", o.extra_fields['reward_extra_info'])
    print(" LLM calls made by the policy:", len(server.calls))
asyncio.run(main())
