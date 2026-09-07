"""SUPO / multi-turn GRPO agent loop for CodeGym, on verl's AgentLoop API.

Implements Algorithm 2 of the SUPO paper (rollout with summarization-based
context management). With `max_summaries == 0` it degenerates to the vanilla
multi-turn GRPO baseline (Table 1 row 1); with `max_summaries == 7` and a 4K
working context it is the SUPO arm (Table 1 row 4).

Rollout contract (see paper Sec. 3.2 / 4.2):
  * a rollout = up to S+1 trajectories; trajectory i>1 starts from the original
    prompt plus the LLM summary of trajectory i-1; every trajectory shares the
    rollout's terminal reward (advantage broadcast happens in the trainer)
  * when the context (prompt + response) reaches L = summary_ratio * W after an
    (action, observation) pair, that pair is discarded, the summarization
    instruction v_sum is appended and the model writes the summary, which ends
    the trajectory (the summary tokens are trained)
  * rollouts that do not call `Done` within H steps or S summaries are
    "overlong" and are masked out of the policy loss by the trainer

Config (actor_rollout_ref.rollout.custom):
  working_context     W  (4096 for SUPO, 32768 for GRPO)
  max_summaries       S  (7 for SUPO, 0 for GRPO)
  summary_ratio       L/W (0.95)
  max_steps           H  (100 LLM calls, summaries included)
  turn_max_tokens     max new tokens per action turn (1024)
  summary_max_tokens  max new tokens per summary turn (1024)
  env_code_path       parquet with (env_key, env_code) built by scripts/prepare_data.py
  dump_dir            optional dir for jsonl rollout dumps
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Optional, Union

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

from .codegym_env import EnvCodeRegistry, CodeGymEnv, create_env, parse_function_call
from .prompts import CONTINUATION_TEMPLATE, NO_CALL_OBSERVATION, SUMMARY_PROMPT

logger = logging.getLogger(__name__)

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
MAX_SUMMARY_CHARS = 6000


def extract_summary(text: str) -> str:
    m = _SUMMARY_RE.findall(text)
    s = m[-1] if m else _THINK_RE.sub("", text)
    s = s.strip()
    if len(s) > MAX_SUMMARY_CHARS:
        s = s[:MAX_SUMMARY_CHARS] + " ..."
    return s


class Trajectory:
    """Token/message state of one trajectory (prompt + interleaved LLM/observation tokens)."""

    def __init__(self, prompt_ids: list[int], messages: list[dict[str, str]]):
        self.prompt_ids = list(prompt_ids)
        self.token_ids = list(prompt_ids)
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.messages = list(messages)
        self.assistant_turns = 0

    @property
    def length(self) -> int:
        return len(self.token_ids)

    @property
    def response_ids(self) -> list[int]:
        return self.token_ids[len(self.prompt_ids):]

    def snapshot(self):
        return (
            list(self.token_ids),
            list(self.response_mask),
            list(self.response_logprobs),
            list(self.messages),
            self.assistant_turns,
        )

    def restore(self, snap) -> None:
        self.token_ids, self.response_mask, self.response_logprobs, self.messages, self.assistant_turns = (
            list(snap[0]),
            list(snap[1]),
            list(snap[2]),
            list(snap[3]),
            snap[4],
        )


@register("supo_agent")
class SupoAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.rollout_config.get("custom") or {}
        self.working_context = int(cfg.get("working_context", 32768))
        self.max_summaries = int(cfg.get("max_summaries", 0))
        self.summary_ratio = float(cfg.get("summary_ratio", 0.95))
        self.max_steps = int(cfg.get("max_steps", 100))
        self.turn_max_tokens = int(cfg.get("turn_max_tokens", 1024))
        self.summary_max_tokens = int(cfg.get("summary_max_tokens", 1024))
        self.max_consecutive_no_call = int(cfg.get("max_consecutive_no_call", 10))
        self.env_code_path = cfg.get("env_code_path")
        self.dump_dir = cfg.get("dump_dir")
        self.dump_every = int(cfg.get("dump_every", 64))
        assert self.env_code_path, "rollout.custom.env_code_path is required"
        self.prompt_length = int(self.rollout_config.prompt_length)
        self.response_length = int(self.rollout_config.response_length)
        # summarization threshold L (SUPO) or hard context cap (GRPO)
        self.context_limit = (
            int(self.working_context * self.summary_ratio) if self.max_summaries > 0 else self.working_context
        )

    # ------------------------------------------------------------------ helpers
    async def _generate(self, traj: Trajectory, sampling_params: dict, request_id: str, metrics: dict):
        with simple_timer("generate_sequences", metrics):
            out = await self.server_manager.generate(
                request_id=request_id, prompt_ids=traj.token_ids, sampling_params=sampling_params
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = out.num_preempted if out.num_preempted is not None else -1
        elif out.num_preempted:
            metrics["num_preempted"] += out.num_preempted
        merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
            traj.token_ids,
            out.token_ids,
            traj.response_mask,
            traj.response_logprobs if (traj.response_logprobs or out.log_probs) else None,
            assistant_logprobs=out.log_probs if out.log_probs else None,
        )
        traj.token_ids = merge_result.token_ids
        traj.response_mask = response_mask
        if response_logprobs is not None:
            traj.response_logprobs = response_logprobs
        traj.assistant_turns += 1
        text = self.tokenizer.decode(out.token_ids, skip_special_tokens=True)
        traj.messages.append({"role": "assistant", "content": text})
        return text, len(out.token_ids)

    async def _append_user(self, traj: Trajectory, content: str, metrics: dict) -> None:
        prev = list(traj.messages)
        traj.messages.append({"role": "user", "content": content})
        merge_result, response_mask, response_logprobs = await self.ct_merge_context_msg(
            prev, traj.messages, traj.token_ids, traj.response_mask,
            traj.response_logprobs if traj.response_logprobs else None,
        )
        traj.token_ids = merge_result.token_ids
        traj.response_mask = response_mask
        if response_logprobs is not None:
            traj.response_logprobs = response_logprobs

    def _gen_budget(self, traj: Trajectory, want: int) -> int:
        """Max new tokens so that the trajectory stays within W and response_length."""
        return min(want, self.working_context - traj.length, self.response_length - len(traj.response_mask))

    # --------------------------------------------------------------------- run
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
        validate = bool(kwargs.get("validate", False))
        global_step = int(kwargs.get("global_steps", -1))
        messages = [dict(m) for m in kwargs["raw_prompt"]]
        env_str = str(kwargs["ability"])
        sample_index = kwargs.get("index", -1)
        gen_uid = uuid.uuid4().hex
        request_id = uuid.uuid4().hex
        metrics: dict[str, Any] = {}
        t_start = time.time()

        registry = EnvCodeRegistry.get(self.env_code_path)
        env: Optional[CodeGymEnv] = None
        try:
            with simple_timer("tool_calls", metrics):
                env = await create_env(registry, env_str)
            state = await self._rollout(env, messages, sampling_params, request_id, metrics)
        except Exception as e:  # noqa: BLE001 - a broken env/rollout must not kill training
            logger.exception("rollout failed for %s: %s", env_str[:80], e)
            state = None
        finally:
            if env is not None:
                env.close()

        if state is None:
            # degenerate rollout: single empty trajectory, reward 0, masked as overlong
            prompt_ids = await self.ct_build_initial_tokens(messages)
            traj = Trajectory(prompt_ids, messages)
            state = dict(trajs=[traj], reward=0.0, finished=False, overlong=True, n_tool_calls=0,
                         n_summaries=0, n_invalid=0, steps=0, stop_reason="exception")

        trajs: list[Trajectory] = state["trajs"]
        reward = float(state["reward"])
        n_trajs = len(trajs)
        rollout_tokens = sum(len(t.response_mask) for t in trajs)
        reward_extra_info = {
            "acc": reward,
            "finished": float(state["finished"]),
            "overlong": float(state["overlong"]),
            "num_tool_calls": float(state["n_tool_calls"]),
            "num_summaries": float(state["n_summaries"]),
            "num_trajs": float(n_trajs),
            "num_steps": float(state["steps"]),
            "num_invalid_calls": float(state["n_invalid"]),
            "rollout_response_tokens": float(rollout_tokens),
        }
        outputs = []
        for i, t in enumerate(trajs):
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=t.prompt_ids,
                    response_ids=t.response_ids[: self.response_length],
                    response_mask=t.response_mask[: self.response_length],
                    response_logprobs=(
                        t.response_logprobs[: self.response_length] if t.response_logprobs else None
                    ),
                    multi_modal_data={},
                    reward_score=reward,
                    num_turns=2 * t.assistant_turns + 1,
                    metrics=AgentLoopMetrics(
                        generate_sequences=metrics.get("generate_sequences", 0.0),
                        tool_calls=metrics.get("tool_calls", 0.0),
                        compute_score=0.0,
                        num_preempted=metrics.get("num_preempted", -1),
                    ),
                    extra_fields={
                        "gen_uid": gen_uid,
                        "traj_idx": i,
                        "num_trajs": n_trajs,
                        "overlong": bool(state["overlong"]),
                        "finished": bool(state["finished"]),
                        "stop_reason": state["stop_reason"],
                        "num_tool_calls": int(state["n_tool_calls"]),
                        "num_summaries": int(state["n_summaries"]),
                        "reward_extra_info": dict(reward_extra_info),
                    },
                )
            )

        if self.dump_dir and isinstance(sample_index, (int,)) and (validate or sample_index % self.dump_every == 0):
            self._dump(trajs, state, env_str, validate, global_step, sample_index, gen_uid, time.time() - t_start)

        if validate:
            # one row per rollout keeps _validate()'s 1:1 batch union; the last
            # trajectory carries the final answer.
            return outputs[-1]
        return outputs

    # ----------------------------------------------------------------- rollout
    async def _rollout(self, env: CodeGymEnv, messages, sampling_params, request_id, metrics) -> dict:
        assert len(messages) >= 2 and messages[0]["role"] == "system" and messages[-1]["role"] == "user", messages
        sys_msg, user_msg = messages[0], messages[-1]
        base_user = user_msg["content"]

        prompt_ids = await self.ct_build_initial_tokens([sys_msg, user_msg])
        traj = Trajectory(prompt_ids, [sys_msg, user_msg])
        trajs = [traj]

        steps = n_summaries = n_tool_calls = n_invalid = consecutive_no_call = 0
        finished = overlong = False
        reward = 0.0
        stop_reason = "max_steps"

        while steps < self.max_steps:
            max_tokens = self._gen_budget(traj, self.turn_max_tokens)
            if max_tokens < 16:
                overlong, stop_reason = True, "context_full"
                break
            snap = traj.snapshot()
            steps += 1
            text, _ = await self._generate(traj, {**sampling_params, "max_tokens": max_tokens}, request_id, metrics)

            call = parse_function_call(text)
            if call.error == "no_call":
                consecutive_no_call += 1
                observation = NO_CALL_OBSERVATION
                if consecutive_no_call >= self.max_consecutive_no_call:
                    overlong, stop_reason = True, "no_call_limit"
                    break
            elif call.error:
                consecutive_no_call = 0
                n_invalid += 1
                observation = f"Error: {call.error}"
            else:
                consecutive_no_call = 0
                n_tool_calls += 1
                with simple_timer("tool_calls", metrics):
                    res = await env.step(call.action_json)
                observation = res.observation
                if res.error:
                    overlong, stop_reason = True, "env_error"
                    break
                with simple_timer("tool_calls", metrics):
                    done = await env.finished()
                if done:
                    finished, stop_reason = True, "done"
                    with simple_timer("tool_calls", metrics):
                        reward = float(await env.reward() > 0)
                    break

            await self._append_user(traj, observation, metrics)

            if traj.length >= self.context_limit:
                if self.max_summaries == 0:
                    overlong, stop_reason = True, "context_full"
                    break
                if n_summaries >= self.max_summaries:
                    overlong, stop_reason = True, "max_summaries"
                    break
                # discard the last (action, observation) pair, then summarize
                traj.restore(snap)
                await self._append_user(traj, SUMMARY_PROMPT, metrics)
                # the summary turn may overrun W by |v_sum| + L_A (paper Sec. 4.2, "fine control of
                # context length"); it is only bounded by the training response_length.
                s_max = min(self.summary_max_tokens, self.response_length - len(traj.response_mask))
                if s_max < 16:
                    overlong, stop_reason = True, "context_full"
                    break
                steps += 1
                summary_text, _ = await self._generate(
                    traj, {**sampling_params, "max_tokens": s_max}, request_id, metrics
                )
                n_summaries += 1
                summary = extract_summary(summary_text)
                new_user = {"role": "user", "content": CONTINUATION_TEMPLATE.format(user_prompt=base_user, summary=summary)}
                new_prompt_ids = await self.ct_build_initial_tokens([sys_msg, new_user])
                traj = Trajectory(new_prompt_ids, [sys_msg, new_user])
                trajs.append(traj)

        if not finished:
            overlong = True
        return dict(trajs=trajs, reward=reward, finished=finished, overlong=overlong, n_tool_calls=n_tool_calls,
                    n_summaries=n_summaries, n_invalid=n_invalid, steps=steps, stop_reason=stop_reason)

    # -------------------------------------------------------------------- dump
    def _dump(self, trajs, state, env_str, validate, global_step, sample_index, gen_uid, elapsed) -> None:
        try:
            os.makedirs(self.dump_dir, exist_ok=True)
            path = os.path.join(self.dump_dir, f"{'val' if validate else 'train'}_step{global_step}_pid{os.getpid()}.jsonl")
            rec = {
                "gen_uid": gen_uid, "index": int(sample_index), "env": env_str[:200], "reward": state["reward"],
                "finished": state["finished"], "overlong": state["overlong"], "stop_reason": state["stop_reason"],
                "num_tool_calls": state["n_tool_calls"], "num_summaries": state["n_summaries"], "steps": state["steps"],
                "elapsed_s": round(elapsed, 1),
                "trajs": [
                    {"prompt_tokens": len(t.prompt_ids), "response_tokens": len(t.response_mask),
                     "messages": t.messages} for t in trajs
                ],
            }
            with open(path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.warning("dump failed: %s", e)
