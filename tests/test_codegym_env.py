"""CPU test: replay reference trajectories through the in-process CodeGym executor.

Run: ~/xiaoxuan/envs/supo/bin/python tests/test_codegym_env.py [--n 20]
Expect: every replayed reference trajectory ends finished with reward 1.
"""
import argparse
import asyncio
import glob
import ast
import json
import os
import random
import time

import pyarrow as pa
import pyarrow.parquet as pq

from supo.codegym_env import CodeGymEnv, EnvCodeRegistry, create_env, parse_env_str, parse_function_call


async def replay(registry, task):
    env = await create_env(registry, task["ability"])
    try:
        raw = task["extra_info"]["std_traj"]
        actions = ast.literal_eval(raw) if isinstance(raw, str) else raw
        for a in actions:
            res = await env.step(a if isinstance(a, str) else json.dumps(a))
            if res.error:
                return False, 0.0, f"env error: {res.observation}"
            if await env.finished():
                break
        fin = await env.finished()
        rew = await env.reward()
        return fin, rew, res.observation[:80]
    finally:
        env.close()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=os.path.expanduser("~/xiaoxuan/external/codegym_hf"))
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    f = sorted(glob.glob(os.path.join(args.raw_dir, "task_en_instruction_en_env", "*.parquet")))[0]
    tasks = pq.read_table(f, columns=["ability", "extra_info"]).slice(0, 2000).to_pylist()
    env_tab = pq.read_table(os.path.join(args.raw_dir, "envs_en", "train-00000-of-00001.parquet")).to_pylist()
    env_codes = {f"{r['source']}__{r['env_name']}": r["env_code"] for r in env_tab}
    rng = random.Random(0)
    sample = rng.sample(tasks, args.n)
    keys = sorted({parse_env_str(t["ability"])[0] for t in sample})
    tmp = "/tmp/claude-1000/-home-tiger/96ae2643-1290-46b4-ac36-b983e9e3de4c/scratchpad/env_codes_test.parquet"
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    pq.write_table(pa.Table.from_pylist([{"env_key": k, "env_code": env_codes[k]} for k in keys]), tmp)
    registry = EnvCodeRegistry.get(tmp)

    t0 = time.time()
    results = await asyncio.gather(*[replay(registry, t) for t in sample], return_exceptions=True)
    dt = time.time() - t0
    ok = 0
    for t, r in zip(sample, results):
        if isinstance(r, Exception):
            print("EXC", t["ability"][:60], repr(r)[:200])
            continue
        fin, rew, obs = r
        ok += int(fin and rew > 0)
        if not (fin and rew > 0):
            print("FAIL", t["ability"][:70], fin, rew, obs)
    print(f"replayed {len(sample)} reference trajectories concurrently in {dt:.1f}s: {ok}/{len(sample)} finished with reward 1")

    # parser checks
    p = parse_function_call('Let me observe.\n<|FunctionCallBegin|>[{"name":"Observe","parameters":{}}]<|FunctionCallEnd|>')
    assert p.name == "Observe" and json.loads(p.action_json) == {"name": "Observe", "parameters": {}}, p
    p = parse_function_call("no call here")
    assert p.error == "no_call", p
    p = parse_function_call('<|FunctionCallBegin|>[{"name":"Done","parameters":{"answer": 3}}]<|FunctionCallEnd|> trailing')
    assert p.name == "Done" and json.loads(p.action_json)["parameters"]["answer"] == 3
    p = parse_function_call('<|FunctionCallBegin|>{"name":"X" "parameters":{}}<|FunctionCallEnd|>')
    assert p.error and "parsed" in p.error, p
    # timeout path: an env whose step never returns
    hang_code = (
        "import gymnasium\nclass HangEnv(gymnasium.Env):\n"
        "    def __init__(self, env_str=None):\n        self._done=False; self._reward=0\n"
        "    @property\n    def finished(self): return self._done\n"
        "    @property\n    def reward(self): return float(self._reward)\n"
        "    @staticmethod\n    def from_env_str(s): return HangEnv(s)\n"
        "    def step(self, a):\n        import time\n        time.sleep(1000)\n"
    )
    env = CodeGymEnv(hang_code, "HangEnv", "HangEnv@{}")
    assert env.wait_started()[0]
    t0 = time.time()
    res = env.step_sync('{"name":"x","parameters":{}}', timeout=2.0)
    assert res.error and not env.alive and time.time() - t0 < 5, res
    print("parser + timeout checks OK")
    assert ok == len(sample), "some reference replays failed"


if __name__ == "__main__":
    asyncio.run(main())
