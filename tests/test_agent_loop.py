"""CPU test of SupoAgentLoop with a scripted fake LLM server.

The fake policy replays a CodeGym reference solution one function call per turn
and answers the summarization instruction with a <summary> block, so the whole
SUPO rollout path (context threshold -> discard last round -> summarize -> new
trajectory -> ... -> Done, reward 1) is exercised without a GPU.

Run: ~/xiaoxuan/envs/supo/bin/python tests/test_agent_loop.py
"""
import ast
import asyncio
import glob
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, DictConfigWrap
from verl.workers.rollout.replica import TokenOutput

from supo.agent_loop import SupoAgentLoop
from supo.codegym_env import parse_env_str
from supo.prompts import FC_BEGIN, FC_END, SUMMARY_PROMPT

TOK_DIR = os.path.expanduser("~/xiaoxuan/external/qwen35_tokenizer")
RAW = os.path.expanduser("~/xiaoxuan/external/codegym_hf")
SCRATCH = "/tmp/claude-1000/-home-tiger/96ae2643-1290-46b4-ac36-b983e9e3de4c/scratchpad"


class FakeServer:
    """Scripted policy: replay reference actions; summarize on request."""

    def __init__(self, tok, actions, stop_after=None):
        self.tok = tok
        self.actions = actions
        self.i = 0
        self.calls = []
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.stop_after = stop_after

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
        prompt_text = self.tok.decode(prompt_ids)
        if SUMMARY_PROMPT[-60:] in prompt_text:
            text = f"<summary>Replayed {self.i} reference actions so far; continue from action {self.i}.</summary>"
        elif self.stop_after is not None and self.i >= self.stop_after:
            text = "I think we are done, but I will not call any function."
        else:
            a = self.actions[min(self.i, len(self.actions) - 1)]
            self.i += 1
            text = f"Next I call the function.\n{FC_BEGIN}[{a}]{FC_END}"
        ids = self.tok.encode(text, add_special_tokens=False) + [self.im_end]
        ids = ids[: sampling_params["max_tokens"]]
        self.calls.append((len(prompt_ids), text[:40]))
        return TokenOutput(token_ids=ids, log_probs=[-0.1] * len(ids), num_preempted=0)


def make_loop(tok, server, working_context, max_summaries, env_code_path, prompt_length=4096, response_length=5120):
    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": prompt_length,
                    "response_length": response_length,
                    "custom": {
                        "working_context": working_context,
                        "max_summaries": max_summaries,
                        "summary_ratio": 0.95,
                        "max_steps": 100,
                        "turn_max_tokens": 256,
                        "summary_max_tokens": 256,
                        "env_code_path": env_code_path,
                        "dump_dir": os.path.join(SCRATCH, "dumps"),
                        "dump_every": 1,
                    },
                }
            },
            "data": {"apply_chat_template_kwargs": {"enable_thinking": False}},
        }
    )
    return SupoAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=server,
        tokenizer=tok,
        processor=None,
        dataset_cls=None,
        data_config=DictConfigWrap(cfg.data),
        hf_model_type="qwen3_5",
        tools=None,
    )


def check_masks(tok, out: AgentLoopOutput, messages):
    assert len(out.response_ids) == len(out.response_mask)
    gen_ids = [t for t, m in zip(out.response_ids, out.response_mask) if m == 1]
    ctx_ids = [t for t, m in zip(out.response_ids, out.response_mask) if m == 0]
    gen_text = tok.decode(gen_ids)
    ctx_text = tok.decode(ctx_ids)
    assistant_texts = [m["content"] for m in messages if m["role"] == "assistant"]
    user_texts = [m["content"] for m in messages[2:] if m["role"] == "user"]
    for a in assistant_texts:
        assert a in gen_text, ("assistant text not covered by mask=1 tokens", a[:60])
    for u in user_texts:
        assert u[:40] in ctx_text, ("observation not in mask=0 tokens", u[:60])
    assert FC_BEGIN not in ctx_text, "function-call tokens must be model-generated (mask=1)"
    return len(gen_ids), len(ctx_ids)


async def main():
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    f = sorted(glob.glob(os.path.join(RAW, "task_en_instruction_en_env", "*.parquet")))[0]
    tasks = pq.read_table(f, columns=["ability", "prompt", "extra_info"]).slice(0, 300).to_pylist()
    env_tab = pq.read_table(os.path.join(RAW, "envs_en", "train-00000-of-00001.parquet")).to_pylist()
    env_codes = {f"{r['source']}__{r['env_name']}": r["env_code"] for r in env_tab}
    # pick a task with a ~25-40 call reference solution
    task = next(t for t in tasks if 25 <= t["extra_info"]["solve_fc_round"] <= 40)
    actions = ast.literal_eval(task["extra_info"]["std_traj"])
    env_key = parse_env_str(task["ability"])[0]
    env_code_path = os.path.join(SCRATCH, "env_codes_agent_test.parquet")
    pq.write_table(pa.Table.from_pylist([{"env_key": env_key, "env_code": env_codes[env_key]}]), env_code_path)
    messages = [{"role": m["role"], "content": m["content"]} for m in task["prompt"]]
    _ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    prompt_len = len(_ids["input_ids"] if hasattr(_ids, "keys") else _ids)
    print(f"task {env_key} ref_calls={len(actions)} prompt_tokens={prompt_len}")
    import shutil
    shutil.rmtree(os.path.join(SCRATCH, "dumps"), ignore_errors=True)
    sp = dict(temperature=1.0, top_p=1.0, top_k=-1, logprobs=True)
    kwargs = dict(raw_prompt=messages, ability=task["ability"], index=0, validate=False, global_steps=3)

    # ---- SUPO: small working context forces several summarizations
    W = prompt_len + 500
    server = FakeServer(tok, actions)
    loop = make_loop(tok, server, working_context=W, max_summaries=7, env_code_path=env_code_path)
    outs = await loop.run(sp, **kwargs)
    assert isinstance(outs, list) and len(outs) >= 2, len(outs)
    n_gen_total = 0
    for i, o in enumerate(outs):
        ef = o.extra_fields
        assert ef["gen_uid"] == outs[0].extra_fields["gen_uid"] and ef["traj_idx"] == i and ef["num_trajs"] == len(outs)
        assert o.reward_score == 1.0 and ef["finished"] and not ef["overlong"], ef
        assert len(o.prompt_ids) + len(o.response_ids) <= W + 300, (len(o.prompt_ids), len(o.response_ids))
        assert o.response_logprobs is not None and len(o.response_logprobs) == len(o.response_ids)
        ptxt = tok.decode(o.prompt_ids)
        if i > 0:
            assert "We are in the following stage of solving the problem" in ptxt and "<summary>" not in ptxt
        else:
            assert "We are in the following stage" not in ptxt
    # per-trajectory mask checks need the messages: re-derive from the dump
    dump = max(glob.glob(os.path.join(SCRATCH, "dumps", "train_step3_*.jsonl")), key=os.path.getmtime)
    rec = json.loads(open(dump).read().strip().splitlines()[-1])
    assert rec["num_summaries"] == len(outs) - 1 and rec["reward"] == 1.0, rec["stop_reason"]
    for o, tr in zip(outs, rec["trajs"]):
        n_gen, n_ctx = check_masks(tok, o, tr["messages"])
        n_gen_total += n_gen
        if o is not outs[-1]:
            assert "<summary>" in tr["messages"][-1]["content"] and tr["messages"][-1]["role"] == "assistant"
            assert tr["messages"][-2]["content"] == SUMMARY_PROMPT
    print(f"SUPO ok: {len(outs)} trajectories, summaries={rec['num_summaries']}, tool_calls={rec['num_tool_calls']}, "
          f"steps={rec['steps']}, generated tokens={n_gen_total}")

    # ---- validation mode returns one output (the last trajectory)
    server = FakeServer(tok, actions)
    loop = make_loop(tok, server, working_context=W, max_summaries=7, env_code_path=env_code_path)
    out = await loop.run(sp, **{**kwargs, "validate": True})
    assert isinstance(out, AgentLoopOutput) and out.reward_score == 1.0 and out.extra_fields["num_trajs"] >= 2
    assert set(out.extra_fields["reward_extra_info"]) >= {"acc", "finished", "overlong", "num_tool_calls", "num_summaries"}
    print("validate mode ok")

    # ---- GRPO baseline (no summaries) with a large context: single trajectory, finished
    server = FakeServer(tok, actions)
    loop = make_loop(tok, server, working_context=32768, max_summaries=0, env_code_path=env_code_path,
                     prompt_length=2048, response_length=30720)
    outs = await loop.run(sp, **kwargs)
    assert len(outs) == 1 and outs[0].reward_score == 1.0 and not outs[0].extra_fields["overlong"]
    print(f"GRPO ok: 1 trajectory, response tokens={len(outs[0].response_ids)}")

    # ---- GRPO with a tiny context: overlong, unfinished, reward 0
    server = FakeServer(tok, actions)
    loop = make_loop(tok, server, working_context=W, max_summaries=0, env_code_path=env_code_path)
    outs = await loop.run(sp, **kwargs)
    ef = outs[0].extra_fields
    assert len(outs) == 1 and ef["overlong"] and not ef["finished"] and outs[0].reward_score == 0.0, ef
    assert len(outs[0].prompt_ids) + len(outs[0].response_ids) <= W + 300
    print(f"GRPO overlong ok: stop_reason={ef['stop_reason']}")

    # ---- SUPO with max_summaries exhausted -> overlong
    server = FakeServer(tok, actions)
    loop = make_loop(tok, server, working_context=W, max_summaries=1, env_code_path=env_code_path)
    outs = await loop.run(sp, **kwargs)
    ef = outs[-1].extra_fields
    assert len(outs) == 2 and ef["overlong"] and ef["stop_reason"] == "max_summaries", ef
    print("SUPO max_summaries overlong ok")

    # ---- policy that stops calling functions -> no_call_limit, masked
    server = FakeServer(tok, actions, stop_after=3)
    loop = make_loop(tok, server, working_context=32768, max_summaries=0, env_code_path=env_code_path,
                     prompt_length=2048, response_length=30720)
    outs = await loop.run(sp, **kwargs)
    ef = outs[0].extra_fields
    assert ef["stop_reason"] == "no_call_limit" and ef["overlong"], ef
    print("no-call limit ok")
    print("ALL AGENT LOOP TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
