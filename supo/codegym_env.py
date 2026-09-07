"""In-process CodeGym environment executor.

CodeGym environments are LLM-generated `gymnasium.Env` subclasses (one python
file per environment) whose `step(action_json)` returns `(ok, observation_str)`
and which expose `finished` / `reward` properties. The official repo runs them
behind an HTTP server (StigLidu/CodeGym online_server). Here each rollout owns
one sandboxed child process (spawned, not forked) that hosts the env instance,
so a hung or crashing environment can be killed without affecting the agent
loop, and no server has to be deployed next to the training job.

Action format seen by the environment (same as the official `OnlineFcGymEnv`):
a JSON string of a single dict `{"name": <function>, "parameters": {...}}`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

STEP_TIMEOUT_S = float(os.environ.get("CODEGYM_STEP_TIMEOUT", "10"))
START_TIMEOUT_S = float(os.environ.get("CODEGYM_START_TIMEOUT", "30"))
MAX_OBS_CHARS = int(os.environ.get("CODEGYM_MAX_OBS_CHARS", "4096"))


# ----------------------------------------------------------------------------
# env code registry
# ----------------------------------------------------------------------------
class EnvCodeRegistry:
    """Maps env_key ("<source>__<EnvClass>") -> python source of the env."""

    _instances: dict[str, "EnvCodeRegistry"] = {}
    _lock = threading.Lock()

    def __init__(self, path: str):
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["env_key", "env_code"])
        keys = table.column("env_key").to_pylist()
        codes = table.column("env_code").to_pylist()
        self.codes: dict[str, str] = dict(zip(keys, codes))
        logger.info("EnvCodeRegistry loaded %d envs from %s", len(self.codes), path)

    @classmethod
    def get(cls, path: str) -> "EnvCodeRegistry":
        with cls._lock:
            if path not in cls._instances:
                cls._instances[path] = EnvCodeRegistry(path)
            return cls._instances[path]

    def __contains__(self, key: str) -> bool:
        return key in self.codes

    def __getitem__(self, key: str) -> str:
        return self.codes[key]


def parse_env_str(env_str: str) -> tuple[str, str, str]:
    """'codegym_v1@<source>__<EnvClass>@<json>' -> (env_key, class_name, class_env_str).

    `class_env_str` is what the env class's `from_env_str` expects: '<EnvClass>@<json>'.
    """
    prefix, rest = env_str.split("@", 1)
    if prefix != "codegym_v1":
        raise ValueError(f"unsupported env prefix {prefix!r}")
    env_key, task_json = rest.split("@", 1)
    class_name = env_key.split("__")[-1]
    return env_key, class_name, f"{class_name}@{task_json}"


# ----------------------------------------------------------------------------
# child process
# ----------------------------------------------------------------------------
def _env_child_main(conn, env_code: str, class_name: str, class_env_str: str) -> None:  # pragma: no cover
    """Runs inside the sandbox process. Protocol: recv (cmd, arg) -> send (ok, payload)."""
    import io
    import contextlib

    env = None
    try:
        ns: dict[str, Any] = {"__name__": f"codegym_env_{class_name}"}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(env_code, f"<codegym:{class_name}>", "exec"), ns)
            cls = ns[class_name]
            env = cls.from_env_str(class_env_str)
            if env is None:
                raise RuntimeError("from_env_str returned None")
        conn.send((True, "started"))
    except Exception as e:  # noqa: BLE001
        conn.send((False, f"{type(e).__name__}: {e}"))
        return

    while True:
        try:
            cmd, arg = conn.recv()
        except (EOFError, OSError):
            return
        try:
            if cmd == "step":
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    res = env.step(arg)
                if isinstance(res, tuple) and len(res) == 2:
                    ok, obs = res
                else:  # some envs may return only the message
                    ok, obs = True, res
                conn.send((True, (bool(ok), str(obs))))
            elif cmd == "finished":
                conn.send((True, bool(env.finished)))
            elif cmd == "reward":
                conn.send((True, float(env.reward)))
            elif cmd == "close":
                conn.send((True, None))
                return
            else:
                conn.send((False, f"unknown cmd {cmd}"))
        except Exception as e:  # noqa: BLE001
            conn.send((False, f"{type(e).__name__}: {e}"))


# ----------------------------------------------------------------------------
# parent-side handle
# ----------------------------------------------------------------------------
@dataclass
class StepResult:
    ok: bool
    observation: str
    error: bool = False  # transport/timeout error (env is dead afterwards)


class CodeGymEnv:
    """One environment instance living in a dedicated child process."""

    _ctx = mp.get_context("spawn")
    _executor: Optional[ThreadPoolExecutor] = None
    _executor_lock = threading.Lock()

    def __init__(self, env_code: str, class_name: str, class_env_str: str):
        self.class_name = class_name
        self.class_env_str = class_env_str
        self._parent_conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_env_child_main, args=(child_conn, env_code, class_name, class_env_str), daemon=True
        )
        self._proc.start()
        child_conn.close()
        self.alive = True
        self.num_steps = 0
        self.last_error: Optional[str] = None

    @classmethod
    def executor(cls) -> ThreadPoolExecutor:
        with cls._executor_lock:
            if cls._executor is None:
                cls._executor = ThreadPoolExecutor(max_workers=int(os.environ.get("CODEGYM_EXEC_THREADS", "128")))
            return cls._executor

    # -- sync primitives ---------------------------------------------------
    def _call(self, cmd: str, arg: Any, timeout: float) -> tuple[bool, Any]:
        if not self.alive:
            return False, f"env process dead: {self.last_error}"
        try:
            self._parent_conn.send((cmd, arg))
            if self._parent_conn.poll(timeout):
                return self._parent_conn.recv()
            self.last_error = f"timeout after {timeout}s on {cmd}"
        except (EOFError, OSError, BrokenPipeError) as e:
            self.last_error = f"{type(e).__name__}: {e}"
        self.kill()
        return False, self.last_error

    def wait_started(self, timeout: float = START_TIMEOUT_S) -> tuple[bool, str]:
        try:
            if self._parent_conn.poll(timeout):
                ok, msg = self._parent_conn.recv()
                if not ok:
                    self.last_error = msg
                    self.kill()
                return ok, msg
            self.last_error = f"start timeout after {timeout}s"
        except (EOFError, OSError) as e:
            self.last_error = f"{type(e).__name__}: {e}"
        self.kill()
        return False, self.last_error

    def step_sync(self, action_json: str, timeout: float = STEP_TIMEOUT_S) -> StepResult:
        self.num_steps += 1
        ok, payload = self._call("step", action_json, timeout)
        if not ok:
            return StepResult(ok=False, observation=f"Error occurred: {payload}", error=not self.alive)
        status, obs = payload
        if len(obs) > MAX_OBS_CHARS:
            obs = obs[:MAX_OBS_CHARS] + f"... [observation truncated to {MAX_OBS_CHARS} chars]"
        return StepResult(ok=bool(status), observation=obs)

    def finished_sync(self) -> bool:
        ok, payload = self._call("finished", None, STEP_TIMEOUT_S)
        return bool(payload) if ok else True

    def reward_sync(self) -> float:
        ok, payload = self._call("reward", None, STEP_TIMEOUT_S)
        return float(payload) if ok else 0.0

    def kill(self) -> None:
        self.alive = False
        try:
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=2)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._parent_conn.close()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self.alive:
            try:
                self._call("close", None, 2.0)
            except Exception:  # noqa: BLE001
                pass
        self.kill()

    # -- async wrappers ----------------------------------------------------
    async def step(self, action_json: str) -> StepResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor(), self.step_sync, action_json)

    async def finished(self) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor(), self.finished_sync)

    async def reward(self) -> float:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor(), self.reward_sync)


async def create_env(registry: EnvCodeRegistry, env_str: str) -> CodeGymEnv:
    env_key, class_name, class_env_str = parse_env_str(env_str)
    env = CodeGymEnv(registry[env_key], class_name, class_env_str)
    loop = asyncio.get_running_loop()
    ok, msg = await loop.run_in_executor(CodeGymEnv.executor(), env.wait_started)
    if not ok:
        raise RuntimeError(f"env start failed for {env_key}: {msg}")
    return env


# ----------------------------------------------------------------------------
# function-call parsing (model output -> env action)
# ----------------------------------------------------------------------------
_FC_RE = re.compile(r"<\|FunctionCallBegin\|>(.*?)<\|FunctionCallEnd\|>", re.DOTALL)


@dataclass
class ParsedCall:
    name: Optional[str]
    action_json: Optional[str]  # json string of {"name":..., "parameters":...}
    error: Optional[str]


def parse_function_call(text: str) -> ParsedCall:
    """Extract the LAST <|FunctionCallBegin|>[{...}]<|FunctionCallEnd|> block.

    Mirrors the official OnlineFcGymEnv preprocessing: strip surrounding list
    brackets, then hand a single-dict JSON string to env.step. Malformed calls
    become an error observation (the env is not stepped).
    """
    matches = _FC_RE.findall(text)
    if not matches:
        # tolerate a missing end tag at the very end of the generation
        tail = text.rsplit("<|FunctionCallBegin|>", 1)
        if len(tail) == 2 and tail[1].strip():
            matches = [tail[1]]
        else:
            return ParsedCall(None, None, "no_call")
    body = matches[-1].strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        return ParsedCall(None, None, f"The action cannot be parsed in json format: {e}")
    if isinstance(obj, list):
        if len(obj) == 0:
            return ParsedCall(None, None, "The function call list is empty.")
        obj = obj[0]
    if not isinstance(obj, dict) or "name" not in obj:
        return ParsedCall(None, None, "The function call must be a dict with `name` and `parameters`.")
    obj.setdefault("parameters", {})
    if obj["parameters"] is None:
        obj["parameters"] = {}
    return ParsedCall(str(obj["name"]), json.dumps(obj, ensure_ascii=False), None)
