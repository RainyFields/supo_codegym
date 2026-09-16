#!/usr/bin/env python
"""Build the report assets for the thinking-mode arms and the Qwen2.5-32B-Instruct arm.

  python scripts/make_report_thinking_32b.py [--out docs/report_thinking_32b]

Inputs (all under $PROJECT_ROOT, default /mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym):
  outputs/<exp>/val/<step>.jsonl     greedy validation dumps (one record per held-out task)
  job-runs/<arm>/train_*.log         trainer stdout (ray-prefixed "step:N - key:value - ..." lines)
  rollouts/<exp>/train_step<N>_pid*.jsonl   sampled training rollouts (full message lists)
Outputs: <out>/assets/*.{png,pdf,csv,json} + <out>/assets/README.md, <out>/tables.md,
<out>/summary.json, <out>/trajectories.json.  Every number quoted in REPORT.tex/REPORT.md
comes from these files.
"""
import argparse, csv, glob, hashlib, json, os, re, statistics, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/.claude/skills/scientific-figure-making/assets"))
from figstyle import apply_publication_style, PALETTE, finalize_figure  # house style

ROOT = os.environ.get("PROJECT_ROOT", "/mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym")
ARMS = {  # label -> (exp name, job-runs dir, train log (largest / explicit), colour)
    "9B GRPO-32K":        ("grpo_codegym_qwen35-9b_32k",        "grpo",       "train_20260908_111515.log", PALETTE["blue_main"]),
    "9B SUPO-4Kx8":       ("supo_codegym_qwen35-9b_4kx8",       "supo",       None,                        PALETTE["teal"]),
    "9B GRPO-32K+think":  ("grpo_codegym_qwen35-9b_32k_think",  "grpo_think", None,                        PALETTE["blue_secondary"]),
    "9B SUPO-4Kx8+think": ("supo_codegym_qwen35-9b_4kx8_think", "supo_think", None,                        PALETTE["violet"]),
    "32B GRPO-32K":       ("grpo_codegym_qwen25-32b_32k",       "grpo",       "train_20260909_150629.log", PALETTE["red_strong"]),
}
STYLE = {"9B GRPO-32K": "-", "9B SUPO-4Kx8": "-", "9B GRPO-32K+think": "--", "9B SUPO-4Kx8+think": "--", "32B GRPO-32K": "-"}
VAL_KEYS = ["score", "finished", "overlong", "num_tool_calls", "num_summaries", "num_trajs",
            "num_invalid_calls", "rollout_response_tokens", "num_steps"]
LOG_KEYS = ["critic/score/mean", "actor/pg_loss", "actor/entropy", "actor/grad_norm", "response_length/mean",
            "response_length/clip_ratio", "training/num_turns/mean", "timing_s/step", "timing_s/gen",
            "timing_s/update_actor", "perf/throughput"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


# ---------------------------------------------------------------- validation dumps
def task_key(inp: str) -> str:
    """Model-independent task identity: the prompt text with chat-template role markers,
    thinking scaffolds and whitespace removed (uids differ between runs)."""
    s = THINK_RE.sub("", inp)
    s = re.sub(r"<\|im_start\|>|<\|im_end\|>|^(system|user|assistant)\n", "", s, flags=re.M)
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.md5(s.encode()).hexdigest()


def load_val(exp):
    per_step, per_task = {}, {}
    for f in glob.glob(f"{ROOT}/outputs/{exp}/val/*.jsonl"):
        step = int(os.path.basename(f).split(".")[0])
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if len(recs) < 100:
            continue
        n = len(recs)
        m = lambda k: sum(float(r.get(k, 0)) for r in recs) / n
        row = {"step": step, "n": n, **{k: m(k) for k in VAL_KEYS}}
        p = row["score"]; row["se"] = (p * (1 - p) / n) ** 0.5
        per_step[step] = row
        per_task[step] = {task_key(r["input"]): float(r.get("score", 0)) for r in recs}
        VAL_OUTPUTS[(exp, step)] = [(r.get("output", ""), float(r.get("score", 0))) for r in recs]
        VAL_RECS[(exp, step)] = [{k: r.get(k) for k in ("output", "score", "overlong", "num_steps", "rollout_response_tokens", "num_trajs")} for r in recs]
    return [per_step[s] for s in sorted(per_step)], per_task


VAL_OUTPUTS = {}   # (exp, step) -> [(decoded generation incl. chat-template role text, score)]
VAL_RECS = {}      # (exp, step) -> per-record fields for the cut-off analysis
TURN_SPLIT = re.compile(r"\nassistant\n")


def split_think(turn: str, first_turn_open: bool = False):
    """One assistant turn of the decoded validation output -> (think_body or None, answer_text).
    Qwen3.5 chat template: the prompt ends with '<think>\n' when thinking is enabled (so the first
    generated turn has no opening tag: first_turn_open=True) and with '<think>\n\n</think>\n\n' when
    disabled; later turns are generated with their own tags. A turn is a thinking turn iff the text
    before '</think>' (or the whole turn when the think block is opened but never closed = cut by the
    per-turn token cap) is non-empty after stripping."""
    i = turn.find("</think>")
    if i >= 0:
        body = turn[:i].replace("<think>", "").strip(); rest = turn[i + len("</think>"):]
    elif turn.lstrip().startswith("<think>") or first_turn_open:
        body = turn.replace("<think>", "").strip(); rest = ""
    else:
        body = ""; rest = turn
    return (body if body else None), rest


def think_stats_val(exp, step, think_arm=True):
    recs = VAL_OUTPUTS.get((exp, step), [])
    turns = present = unterminated = 0; bodies = []
    for out, _ in recs:
        for ti, t in enumerate(TURN_SPLIT.split(out)):
            if not t.strip():
                continue
            turns += 1
            body, _rest = split_think(t, first_turn_open=(think_arm and ti == 0))
            if body:
                present += 1; bodies.append(body)
                if "</think>" not in t: unterminated += 1
    toks = count_tokens(bodies)
    return {"n_tasks": len(recs), "assistant_turns": turns, "frac_turns_with_think": present / turns if turns else None,
            "frac_think_unterminated": unterminated / present if present else None,
            "think_tokens_mean": statistics.mean(toks) if toks else None, "think_tokens_median": statistics.median(toks) if toks else None,
            "think_tokens_p90": sorted(toks)[int(0.9 * (len(toks) - 1))] if toks else None, "think_tokens_max": max(toks) if toks else None,
            "think_tokens_total_per_task": sum(toks) / len(recs) if recs else None}


# ---------------------------------------------------------------- trainer logs
def parse_log(path):
    """step -> {metric: value} from 'step:N - k:v - k:v ...' lines (ray/ANSI prefixes stripped;
    the last occurrence of a step wins, i.e. a resumed run overrides the pre-resume line)."""
    out = {}
    for line in open(path, errors="ignore"):
        line = ANSI.sub("", line)
        i = line.find("step:")
        if i < 0 or " - " not in line:
            continue
        body = line[i:].strip()
        if not re.match(r"step:\d+ - ", body):
            continue
        d = {}
        for tok in body.split(" - "):
            k, _, v = tok.partition(":")
            try:
                d[k.strip()] = float(v)
            except ValueError:
                pass
        if "step" in d:
            out[int(d["step"])] = d
    return out


def pick_log(arm_dir, explicit):
    if explicit:
        return f"{ROOT}/job-runs/{arm_dir}/{explicit}"
    logs = glob.glob(f"{ROOT}/job-runs/{arm_dir}/train_*.log")
    return max(logs, key=os.path.getsize) if logs else None


# ---------------------------------------------------------------- rollouts / thinking
def load_rollouts(exp, steps):
    out = {}
    for s in steps:
        recs = []
        for f in sorted(glob.glob(f"{ROOT}/rollouts/{exp}/train_step{s}_pid*.jsonl")):
            recs += [json.loads(l) for l in open(f) if l.strip()]
        if recs:
            out[s] = recs
    return out


TOK_PY = os.path.expanduser("~/xiaoxuan/envs/supo/bin/python")   # venv with transformers (system python has matplotlib only)
def count_tokens(texts):
    """Qwen3.5-9B tokenizer token counts for a list of strings (batched through the training venv)."""
    if not texts:
        return []
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(texts, fh); tmp = fh.name
    code = ("import json,sys; from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('/mnt/hdfs/mlsys/models/Qwen3.5-9B'); "
            "xs=json.load(open(sys.argv[1])); print(json.dumps([len(t.encode(x, add_special_tokens=False)) for x in xs]))")
    out = subprocess.run([TOK_PY, "-c", code, tmp], capture_output=True, text=True, check=True).stdout
    os.unlink(tmp)
    return json.loads(out.strip().splitlines()[-1])


def think_stats(recs):
    """Per assistant turn: thinking present iff the assistant message content has a non-empty
    <think>...</think> block (after stripping whitespace); think tokens = Qwen3.5-9B tokenizer
    count of the block body. Turns = all assistant messages across all trajectories/summaries."""
    turns = present = 0; bodies = []; resp = []
    for r in recs:
        for t in r.get("trajs", []):
            resp.append(t.get("response_tokens", 0))
            for msg in t.get("messages", []):
                if msg.get("role") != "assistant":
                    continue
                turns += 1
                m = THINK_RE.search(msg.get("content", "") or "")
                body = m.group(1).strip() if m else ""
                if body:
                    present += 1; bodies.append(body)
    toks = count_tokens(bodies)
    return {"n_rollouts": len(recs), "assistant_turns": turns, "frac_turns_with_think": present / turns if turns else None,
            "think_tokens_mean": statistics.mean(toks) if toks else None,
            "think_tokens_median": statistics.median(toks) if toks else None,
            "think_tokens_max": max(toks) if toks else None,
            "response_tokens_per_traj_mean": statistics.mean(resp) if resp else None,
            "reward_mean": statistics.mean(float(r.get("reward", 0)) for r in recs)}


def pretty_traj(rec, max_obs=300, max_calls=14):
    """Compact human-readable rendering of one rollout record."""
    env = rec.get("env", ""); parts = env.split("@", 2)
    out = [f"env: {parts[1] if len(parts) > 1 else env}", f"config: {parts[2][:200] if len(parts) > 2 else ''}",
           f"reward={rec.get('reward')} finished={rec.get('finished')} overlong={rec.get('overlong')} "
           f"stop={rec.get('stop_reason')} tool_calls={rec.get('num_tool_calls')} summaries={rec.get('num_summaries')} steps={rec.get('steps')}"]
    trajs = rec.get("trajs", [])
    for ti, t in enumerate(trajs):
        msgs = t.get("messages", [])
        out.append(f"--- trajectory {ti + 1}/{len(trajs)} (prompt {t.get('prompt_tokens')} tok, response {t.get('response_tokens')} tok)")
        if ti == 0:
            user = next((m for m in msgs if m.get("role") == "user"), None)
            if user:
                out.append("task: " + re.sub(r"\s+", " ", user["content"])[:600])
        calls = 0; shown = 0
        for m in msgs:
            if m.get("role") == "assistant":
                c = m.get("content", "") or ""
                th = THINK_RE.search(c)
                if th and th.group(1).strip():
                    out.append("  <think> " + re.sub(r"\s+", " ", th.group(1).strip())[:400] + " </think>")
                    c = THINK_RE.sub("", c)
                c = re.sub(r"\s+", " ", c).strip()
                calls += 1
                if shown < max_calls or m is msgs[-1]:
                    out.append(f"  A{calls}: {c[:420]}"); shown += 1
                elif shown == max_calls:
                    out.append("  ... (assistant turns elided) ..."); shown += 1
            elif m.get("role") == "user" and m is not msgs[1] and shown <= max_calls:
                c = re.sub(r"\s+", " ", m.get("content", "") or "").strip()
                out.append(f"  obs: {c[:max_obs]}")
    return "\n".join(out)


# ---------------------------------------------------------------- figures
def fig_val_acc(data, out):
    apply_publication_style(font_size=13, axes_linewidth=1.8)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    rows = []
    for arm, (rws, _) in data.items():
        if not rws: continue
        x = [r["step"] for r in rws]; y = [r["score"] for r in rws]; se = [r["se"] for r in rws]
        c = ARMS[arm][3]
        ax.fill_between(x, [a - b for a, b in zip(y, se)], [a + b for a, b in zip(y, se)], color=c, alpha=0.12, lw=0)
        ax.plot(x, y, STYLE[arm], color=c, lw=2.4, marker="o", ms=5, label=arm)
        rows += [{"arm": arm, "step": r["step"], "acc": r["score"], "se": r["se"], "n": r["n"]} for r in rws]
    ax.set_xlabel("training step"); ax.set_ylabel("validation accuracy (128 held-out tasks)")
    ax.set_ylim(0.45, 0.95); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=10, ncol=2, loc="lower right")
    finalize_figure(fig, f"{out}/fig1_val_acc_all_arms")
    with open(f"{out}/fig1_val_acc_all_arms.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "step", "acc", "se", "n"]); w.writeheader(); w.writerows(rows)


def fig_overlong_calls(data, out):
    apply_publication_style(font_size=13, axes_linewidth=1.8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    rows = []
    for arm, (rws, _) in data.items():
        if not rws: continue
        x = [r["step"] for r in rws]; c = ARMS[arm][3]
        axes[0].plot(x, [r["overlong"] for r in rws], STYLE[arm], color=c, lw=2.2, marker="o", ms=4, label=arm)
        axes[1].plot(x, [r["num_tool_calls"] for r in rws], STYLE[arm], color=c, lw=2.2, marker="o", ms=4)
        axes[2].plot(x, [r["rollout_response_tokens"] for r in rws], STYLE[arm], color=c, lw=2.2, marker="o", ms=4)
        rows += [{"arm": arm, "step": r["step"], "overlong": r["overlong"], "tool_calls": r["num_tool_calls"],
                  "response_tokens": r["rollout_response_tokens"], "summaries": r["num_summaries"],
                  "invalid_calls": r["num_invalid_calls"], "finished": r["finished"]} for r in rws]
    for ax, yl in zip(axes, ["overlong fraction", "mean tool calls / task", "mean response tokens / task"]):
        ax.set_xlabel("training step"); ax.set_ylabel(yl); ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)
    finalize_figure(fig, f"{out}/fig2_overlong_calls_tokens")
    with open(f"{out}/fig2_overlong_calls_tokens.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def fig_think(stats, out):
    apply_publication_style(font_size=13, axes_linewidth=1.8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    rows = []
    for arm, per_step in stats.items():
        steps = sorted(per_step); c = ARMS[arm][3]
        f = [per_step[s]["frac_turns_with_think"] or 0 for s in steps]
        mt = [per_step[s]["think_tokens_mean"] or 0 for s in steps]
        md = [per_step[s]["think_tokens_median"] or 0 for s in steps]
        axes[0].plot(steps, f, STYLE.get(arm, "-"), color=c, lw=2.2, marker="o", ms=4, label=arm)
        axes[1].plot(steps, mt, "-", color=c, lw=2.2, marker="o", ms=4, label=arm + " (mean)")
        axes[1].plot(steps, md, "--", color=c, lw=1.6, marker="s", ms=3, label=arm + " (median)")
        axes[2].plot(steps, [per_step[s]["think_tokens_total_per_task"] or 0 for s in steps], STYLE.get(arm, "-"), color=c, lw=2.2, marker="o", ms=4, label=arm)
        rows += [{"arm": arm, "step": s, **per_step[s]} for s in steps]
    axes[0].set_ylabel("fraction of assistant turns with thinking"); axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("think tokens per thinking turn"); axes[2].set_ylabel("think tokens per task (sum over turns)")
    for ax in axes: ax.set_xlabel("training step"); ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9); axes[1].legend(frameon=False, fontsize=8)
    finalize_figure(fig, f"{out}/fig3_thinking_stats")
    with open(f"{out}/fig3_thinking_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def fig_train(logs, out):
    apply_publication_style(font_size=12, axes_linewidth=1.8)
    keys = [("critic/score/mean", "batch mean reward"), ("actor/pg_loss", "policy-gradient loss"),
            ("actor/entropy", "policy entropy"), ("response_length/mean", "mean response length (tokens)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5)); axes = axes.ravel()
    rows = []
    for arm, d in logs.items():
        steps = sorted(d); c = ARMS[arm][3]
        for ax, (k, yl) in zip(axes, keys):
            xs = [s for s in steps if k in d[s]]
            ax.plot(xs, [d[s][k] for s in xs], STYLE[arm], color=c, lw=1.8, alpha=0.9, label=arm)
            ax.set_ylabel(yl); ax.set_xlabel("training step"); ax.grid(alpha=0.3)
        rows += [{"arm": arm, "step": s, **{k: d[s].get(k) for k in LOG_KEYS}} for s in steps]
    axes[0].legend(frameon=False, fontsize=9)
    finalize_figure(fig, f"{out}/fig4_training_curves")
    with open(f"{out}/fig4_training_curves.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "step"] + LOG_KEYS); w.writeheader(); w.writerows(rows)


def fig_gain(data, out):
    apply_publication_style(font_size=13, axes_linewidth=1.8)
    fig, ax = plt.subplots(figsize=(7.5, 4.6)); rows = []
    for arm in ["9B GRPO-32K", "32B GRPO-32K", "9B GRPO-32K+think"]:
        rws = data[arm][0]
        if not rws: continue
        base = rws[0]["score"]; x = [r["step"] for r in rws]; g = [r["score"] - base for r in rws]
        ax.plot(x, g, STYLE[arm], color=ARMS[arm][3], lw=2.4, marker="o", ms=5, label=f"{arm} (base {base:.3f})")
        rows += [{"arm": arm, "step": r["step"], "acc": r["score"], "base": base, "gain": r["score"] - base} for r in rws]
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("training step"); ax.set_ylabel("accuracy gain over step 0")
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=10)
    finalize_figure(fig, f"{out}/fig5_gain_over_base")
    with open(f"{out}/fig5_gain_over_base.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "step", "acc", "base", "gain"]); w.writeheader(); w.writerows(rows)


def cutoff_stats(exp, step, single_segment_only=False):
    """Thinking cut-offs by the per-turn cap. With enable_thinking the Qwen3.5 generation prompt ends in
    '<think>\n', so every assistant turn starts inside a think block; a turn whose thinking never closed is
    one with no '</think>'. Per episode: cut_off_turns = num_steps - output.count('</think>') (clipped at 0)."""
    recs = VAL_RECS.get((exp, step), [])
    if single_segment_only:
        recs = [r for r in recs if int(r.get("num_trajs") or 1) == 1]
    if not recs:
        return None
    cut = [max(0, int(r["num_steps"]) - str(r["output"]).count("</think>")) for r in recs]
    has = [c >= 1 for c in cut]
    turns = sum(int(r["num_steps"]) for r in recs)
    m = lambda xs: (sum(xs) / len(xs)) if xs else None
    ov = [float(r["overlong"]) for r in recs]; ac = [float(r["score"]) for r in recs]
    return {"n_episodes": len(recs), "turns": turns, "cut_off_turns": sum(cut), "frac_turns_cut": sum(cut) / turns if turns else None,
            "frac_episodes_with_cut": m([1.0 if h else 0.0 for h in has]), "n_episodes_with_cut": sum(has),
            "overlong_given_cut": m([o for o, h in zip(ov, has) if h]), "overlong_given_none": m([o for o, h in zip(ov, has) if not h]),
            "acc_given_cut": m([a for a, h in zip(ac, has) if h]), "acc_given_none": m([a for a, h in zip(ac, has) if not h]),
            "resp_tokens_per_turn": m([float(r["rollout_response_tokens"]) / max(1, int(r["num_steps"])) for r in recs])}


def fig_cutoff(stats, out):
    apply_publication_style(font_size=13, axes_linewidth=1.8)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    steps = sorted(stats); c = ARMS["9B GRPO-32K+think"][3]
    axes[0].plot(steps, [100 * stats[s]["frac_turns_cut"] for s in steps], "-", color=c, lw=2.2, marker="o", ms=4, label="cut-off turns / all turns (%)")
    axes[0].plot(steps, [100 * stats[s]["frac_episodes_with_cut"] for s in steps], "--", color=PALETTE["red_strong"], lw=2.2, marker="s", ms=4, label="episodes with >=1 cut-off (%)")
    axes[0].set_ylabel("percent"); axes[0].set_ylim(0, None); axes[0].legend(frameon=False, fontsize=9)
    for key, lab, col, ls in [("overlong_given_cut", "overlong | cut-off", PALETTE["red_strong"], "-"), ("overlong_given_none", "overlong | none", PALETTE["red_strong"], "--"),
                              ("acc_given_cut", "accuracy | cut-off", c, "-"), ("acc_given_none", "accuracy | none", c, "--")]:
        axes[1].plot(steps, [stats[s][key] if stats[s][key] is not None else float("nan") for s in steps], ls, color=col, lw=2.2, marker="o", ms=4, label=lab)
    axes[1].set_ylabel("fraction of episodes"); axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    for ax in axes: ax.set_xlabel("training step"); ax.grid(alpha=0.3)
    finalize_figure(fig, f"{out}/fig7_think_cutoffs")
    rows = [{"arm": "9B GRPO-32K+think", "step": s, **stats[s]} for s in steps]
    with open(f"{out}/fig7_think_cutoffs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def fig_agreement(data, out):
    """Per-task solved/unsolved contingency between 32B@last and 9B@last (and 9B@0), matched by task_key."""
    a9 = data["9B GRPO-32K"][1]; a32 = data["32B GRPO-32K"][1]
    if not a9 or not a32:
        return None
    s9, s32 = max(a9), max(a32)
    keys = set(a9[s9]) & set(a32[s32])
    cells = {"both": 0, "only_9B": 0, "only_32B": 0, "neither": 0}
    for k in keys:
        x, y = a9[s9][k] > 0.5, a32[s32][k] > 0.5
        cells["both" if x and y else "only_9B" if x else "only_32B" if y else "neither"] += 1
    base9 = {k: a9[0][k] > 0.5 for k in keys if 0 in a9 and k in a9[0]}
    hard = [k for k in keys if k in base9 and not base9[k]]
    hard_solved = {"9B_last": sum(a9[s9][k] > 0.5 for k in hard), "32B_last": sum(a32[s32][k] > 0.5 for k in hard), "n_hard": len(hard)}
    res = {"matched_tasks": len(keys), "step_9B": s9, "step_32B": s32, "contingency": cells, "tasks_unsolved_by_9B_at_step0": hard_solved,
           "matching_key": "md5 of the validation prompt text after removing chat-template role markers, <think> scaffolds and whitespace"}
    apply_publication_style(font_size=13, axes_linewidth=1.8)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    mat = [[cells["both"], cells["only_32B"]], [cells["only_9B"], cells["neither"]]]
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(max(r) for r in mat))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i][j]), ha="center", va="center", fontsize=18, color="white" if mat[i][j] > 0.6 * max(max(r) for r in mat) else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"9B solved (@{s9})", "9B failed"]); ax.set_yticks([0, 1]); ax.set_yticklabels([f"32B solved (@{s32})", "32B failed"])
    ax.set_title(f"per-task agreement, {len(keys)} matched tasks", fontsize=12)
    finalize_figure(fig, f"{out}/fig6_task_agreement")
    json.dump(res, open(f"{out}/fig6_task_agreement.json", "w"), indent=2)
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "report_thinking_32b"))
    a = ap.parse_args(); out = os.path.abspath(a.out); assets = f"{out}/assets"; os.makedirs(assets, exist_ok=True)

    data = {arm: load_val(exp) for arm, (exp, *_) in ARMS.items()}
    # validation table (csv + md)
    steps = sorted({r["step"] for rws, _ in data.values() for r in rws})
    with open(f"{assets}/val_metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm", "step", "n", "se"] + VAL_KEYS)
        for arm, (rws, _) in data.items():
            for r in rws: w.writerow([arm, r["step"], r["n"], round(r["se"], 4)] + [round(r[k], 4) for k in VAL_KEYS])
    md = ["| arm | " + " | ".join(str(s) for s in steps) + " | final | peak | mean 80-100 |", "|---|" + "---|" * (len(steps) + 3)]
    summ = {}
    for arm, (rws, _) in data.items():
        d = {r["step"]: r for r in rws}
        if not rws: continue
        last = rws[-1]; best = max(rws, key=lambda r: r["score"]); tail = [r["score"] for r in rws if r["step"] >= 80]
        summ[arm] = {"exp": ARMS[arm][0], "steps": [r["step"] for r in rws], "acc_first": rws[0]["score"], "acc_last": last["score"], "last_step": last["step"],
                     "acc_peak": best["score"], "peak_step": best["step"], "mean_80_100": sum(tail) / len(tail) if tail else None,
                     "overlong_first": rws[0]["overlong"], "overlong_last": last["overlong"], "tool_calls_last": last["num_tool_calls"],
                     "summaries_last": last["num_summaries"], "gain": last["score"] - rws[0]["score"], "n": last["n"]}
        md.append(f"| {arm} | " + " | ".join(f"{d[s]['score']:.3f}" if s in d else "–" for s in steps) +
                  f" | **{last['score']:.3f}** | {best['score']:.3f} @{best['step']} | {summ[arm]['mean_80_100']:.3f} |")
    md2 = ["| arm | overlong @0 | overlong @100 | tool calls @0 | tool calls @100 | summaries @100 | resp tokens @100 |", "|---|---|---|---|---|---|---|"]
    for arm, (rws, _) in data.items():
        if not rws: continue
        f, l = rws[0], rws[-1]
        md2.append(f"| {arm} | {f['overlong']:.2f} | {l['overlong']:.2f} | {f['num_tool_calls']:.1f} | {l['num_tool_calls']:.1f} | {l['num_summaries']:.2f} | {l['rollout_response_tokens']:.0f} |")

    # trainer logs
    logs = {}
    for arm, (exp, adir, explicit, _) in ARMS.items():
        p = pick_log(adir, explicit)
        if p and os.path.exists(p):
            logs[arm] = parse_log(p); summ.setdefault(arm, {})["train_log"] = os.path.relpath(p, ROOT); summ[arm]["train_steps_logged"] = len(logs[arm])
            st = [logs[arm][s].get("timing_s/step") for s in logs[arm] if logs[arm][s].get("timing_s/step")]
            summ[arm]["step_time_s_median"] = statistics.median(st) if st else None

    # thinking statistics + example trajectories
    think = {}
    for arm in ["9B GRPO-32K+think", "9B SUPO-4Kx8+think", "9B GRPO-32K", "9B SUPO-4Kx8"]:   # last two = non-thinking controls
        exp = ARMS[arm][0]
        think[arm] = {s: think_stats_val(exp, s, think_arm="think" in arm) for s in sorted(st for (e, st) in VAL_OUTPUTS if e == exp)}
    think_ctrl = {arm: think.pop(arm) for arm in ["9B GRPO-32K", "9B SUPO-4Kx8"]}
    ro32 = load_rollouts(ARMS["32B GRPO-32K"][0], [0, 100])
    ro_th = {}
    traj = {}
    def pick(recs, want_reward):
        c = [r for r in recs if (float(r.get("reward", 0)) > 0.5) == want_reward]
        return c[0] if c else None
    if 0 in ro32: traj["32B_step0_failure"] = pick(ro32[0], False)
    if 100 in ro32:
        traj["32B_step100_success"] = pick(ro32[100], True); traj["32B_step100_failure"] = pick(ro32[100], False)
    traj_txt = {k: pretty_traj(v) for k, v in traj.items() if v}
    meta = {k: {kk: vv for kk, vv in v.items() if kk != "trajs"} for k, v in traj.items() if v}
    # thinking example: first solved task of the 9B thinking arm at its last checkpoint, first 4 turns of the greedy validation output
    exp_t = ARMS["9B GRPO-32K+think"][0]; st_t = max(s for (e, s) in VAL_OUTPUTS if e == exp_t)
    solved = [o for o, sc in VAL_OUTPUTS[(exp_t, st_t)] if sc > 0.5]
    if solved:
        lines = [f"source: greedy validation output, {exp_t} @ step {st_t}, first solved task (score=1)"]
        for ti, t in enumerate(TURN_SPLIT.split(solved[0])[:4]):
            body, rest = split_think(t, first_turn_open=(ti == 0))
            if body: lines.append(f"  turn {ti + 1} <think> " + re.sub(r"\s+", " ", body)[:500] + (" ..." if len(body) > 500 else "") + " </think>")
            ans = re.sub(r"\s+", " ", rest.split("\nuser\n")[0]).strip()
            lines.append(f"  turn {ti + 1} answer: " + ans[:400] + (" ..." if len(ans) > 400 else ""))
            obs = rest.split("\nuser\n")[1] if "\nuser\n" in rest else ""
            if obs: lines.append("  obs: " + re.sub(r"\s+", " ", obs)[:200])
        traj_txt["9B_think_val_step_last"] = "\n".join(lines); meta["9B_think_val_step_last"] = {"exp": exp_t, "step": st_t, "score": 1.0}
    json.dump({k: {"record": meta[k], "pretty": traj_txt[k]} for k in traj_txt}, open(f"{out}/trajectories.json", "w"), indent=1)

    # thinking cut-offs by the per-turn cap (GRPO+think; SUPO+think single-segment episodes as a check)
    exp_g = ARMS["9B GRPO-32K+think"][0]; exp_s = ARMS["9B SUPO-4Kx8+think"][0]
    cutoff = {st: cutoff_stats(exp_g, st) for (e, st) in sorted(VAL_RECS) if e == exp_g}
    cutoff = {st: v for st, v in cutoff.items() if v}
    cutoff_supo1 = {st: cutoff_stats(exp_s, st, single_segment_only=True) for (e, st) in sorted(VAL_RECS) if e == exp_s}
    cutoff_supo1 = {st: v for st, v in cutoff_supo1.items() if v}
    # figures
    fig_val_acc(data, assets); fig_overlong_calls(data, assets); fig_think(think, assets); fig_train(logs, assets); fig_gain(data, assets)
    agree = fig_agreement(data, assets); fig_cutoff(cutoff, assets)
    md3 = ["| step | cut-off turns / all turns | episodes with a cut-off | overlong \\| cut-off | overlong \\| none | acc \\| cut-off | acc \\| none | resp tokens / turn |", "|---|---|---|---|---|---|---|---|"]
    f2 = lambda v: "–" if v is None else f"{v:.2f}"
    for st, v in cutoff.items():
        md3.append(f"| {st} | {v['cut_off_turns']}/{v['turns']} ({100 * v['frac_turns_cut']:.1f}%) | {v['n_episodes_with_cut']}/{v['n_episodes']} ({100 * v['frac_episodes_with_cut']:.0f}%) | {f2(v['overlong_given_cut'])} | {f2(v['overlong_given_none'])} | {f2(v['acc_given_cut'])} | {f2(v['acc_given_none'])} | {v['resp_tokens_per_turn']:.0f} |")

    json.dump({"arms": summ, "thinking": think, "thinking_control_nonthinking_9B_GRPO": think_ctrl, "agreement": agree,
               "think_cutoffs_grpo_think": cutoff, "think_cutoffs_supo_think_single_segment": cutoff_supo1,
               "paper_2509_17325": {"model": "Qwen2.5-32B-Instruct", "in_domain_base": 30.1, "in_domain_full": 75.0, "in_domain_filter": 81.0,
                                    "eval": "972 evals / 500 unseen envs, 10-256 calls, base acc <= 25%, T=0.7 top-p 0.95, Tmax 256, prompt 5120 + response 24576"}},
              open(f"{out}/summary.json", "w"), indent=2)
    open(f"{out}/tables.md", "w").write("## Validation accuracy (greedy, 128 held-out tasks)\n" + "\n".join(md) + "\n\n## Behaviour at first/last checkpoint\n" + "\n".join(md2) +
                                        "\n\n## Thinking cut-offs by the per-turn cap (9B GRPO-32K+think, greedy validation)\n" + "\n".join(md3) + "\n")
    readme = f"""# Assets for the thinking + 32B report

All files were produced by `scripts/make_report_thinking_32b.py` on {os.popen('date "+%Y-%m-%d %H:%M %Z"').read().strip()} from
`$PROJECT_ROOT = {ROOT}`. Validation numbers come from the greedy validation dumps
`outputs/<exp>/val/<step>.jsonl` (one JSON record per held-out task; fields used: score, finished, overlong,
num_tool_calls, num_summaries, num_trajs, num_invalid_calls, rollout_response_tokens, num_steps; files with <100
records are skipped). Arms -> exp names: {json.dumps({k: v[0] for k, v in ARMS.items()})}.

| file | backs | contents / schema | provenance |
|---|---|---|---|
| fig1_val_acc_all_arms.{{png,pdf,csv}} | Fig. 1 | arm, step, acc (mean score over n tasks), se = sqrt(acc(1-acc)/n), n | val dumps |
| fig2_overlong_calls_tokens.{{png,pdf,csv}} | Fig. 2 | arm, step, overlong (fraction of tasks hitting the episode/context cap), tool_calls (mean per task), response_tokens (mean generated tokens per task), summaries (mean per task), invalid_calls, finished | val dumps |
| fig3_thinking_stats.{{png,pdf,csv}} | Fig. 3 | arm, step, n_tasks (128), assistant_turns (all turns over the 128 greedy validation episodes), frac_turns_with_think, frac_think_unterminated (thinking turn cut before </think>), think_tokens_mean/median/p90/max (Qwen3.5-9B tokenizer count of the text before </think>, opening tag removed), think_tokens_total_per_task | outputs/<exp>/val/<step>.jsonl field `output` (decoded generation incl. chat-template role text), turns split on "\\nassistant\\n" |
| fig4_training_curves.{{png,pdf,csv}} | Fig. 4 | arm, step, and the batch-level trainer metrics {LOG_KEYS} (verl metric names; response_length in tokens; timing in seconds) | job-runs/<arm>/train_*.log: 'step:N - key:value' lines, ray/ANSI prefixes stripped, last occurrence per step wins |
| fig5_gain_over_base.{{png,pdf,csv}} | Fig. 5 | arm, step, acc, base (= acc at step 0), gain = acc - base | val dumps |
| fig7_think_cutoffs.{{png,pdf,csv}} | Fig. 7, Table 5 | arm, step, n_episodes, turns (sum of num_steps), cut_off_turns (sum over episodes of max(0, num_steps - output.count("</think>"))), frac_turns_cut, frac_episodes_with_cut, n_episodes_with_cut, overlong_given_cut / overlong_given_none (mean overlong flag over episodes with >= 1 / 0 cut-off turns), acc_given_cut / acc_given_none (same for score), resp_tokens_per_turn (mean of rollout_response_tokens / num_steps) | val dumps of grpo_codegym_qwen35-9b_32k_think, fields output, num_steps, overlong, score, rollout_response_tokens; single-segment SUPO+think check in ../summary.json |
| fig6_task_agreement.{{png,pdf,json}} | Fig. 6 | 2x2 contingency of solved/unsolved per task for 32B@last vs 9B@last; matched_tasks; tasks unsolved by 9B at step 0 and how many of those each final policy solves | val dumps, tasks matched by md5 of the prompt text after stripping role markers/<think>/whitespace |
| val_metrics.csv | Tables 2-3 | arm, step, n, se + the nine per-task mean fields above | val dumps |
| ../summary.json | all tables | per-arm first/last/peak/mean-80-100 accuracy, overlong, tool calls, gain, train-log path and median step time; thinking stats; agreement; paper reference numbers | derived from the above |
| ../trajectories.json | Sec. 5 | three sampled training rollouts of the 32B arm (step 0 failure, step 100 success and failure; record minus messages + pretty rendering) and the first four turns of one greedy validation episode of the 9B thinking arm at its last checkpoint showing the think segments | rollouts dumps; val dumps |
| ../tables.md | Tables 2-3 | markdown tables quoted in the report | derived |
"""
    open(f"{assets}/README.md", "w").write(readme)
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("acc_first", "acc_last", "acc_peak", "peak_step", "mean_80_100", "gain", "step_time_s_median", "train_steps_logged")} for k, v in summ.items()}, indent=1))
    print("agreement:", json.dumps(agree)); print("trajectories:", list(traj_txt)); print("think steps:", {k: sorted(v) for k, v in think.items()})


if __name__ == "__main__":
    main()
