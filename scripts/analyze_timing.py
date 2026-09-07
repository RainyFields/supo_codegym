#!/usr/bin/env python
"""Time-distribution table for a verl run: where does each training step spend its time?

verl logs one metrics line per step to the console (and to wandb/merlin tracking), e.g.
  step:3 - timing_s/gen:412.1 - timing_s/old_log_prob:38.2 - timing_s/update_actor:95.0 - ...
This script parses those lines from one or more log files and prints, per phase, the mean /
median seconds and the share of the total step time, plus the agent-loop breakdown inside
generation (LLM generate vs env tool calls, slowest rollout) so the bottleneck is explicit.

  python scripts/analyze_timing.py /mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym/job-runs/supo/train_*.log
  python scripts/analyze_timing.py --last 20 --markdown docs/timing_supo.md <log>
"""
import argparse
import glob
import re
import statistics as st
import sys

LINE_RE = re.compile(r"^step:(\d+)\s+-\s+(.*)$")
KV_RE = re.compile(r"([A-Za-z0-9_/@.\-]+):(-?[0-9.]+(?:e-?\d+)?)")

# phase key -> label; order = display order
PHASES = [
    ("timing_s/gen", "rollout generation (agent loop, all rollouts)"),
    ("timing_s/reward", "reward extraction"),
    ("timing_s/old_log_prob", "old log-prob recompute (actor fwd)"),
    ("timing_s/ref", "ref log-prob (unused: no KL)"),
    ("timing_s/adv", "advantage computation"),
    ("timing_s/update_actor", "actor update (PPO mini-batches)"),
    ("timing_s/save_checkpoint", "checkpoint save"),
    ("timing_s/testing", "validation (eval set)"),
    ("timing_s/step", "TOTAL step"),
]
AGENT = [
    ("agent_loop/generate_sequences/mean", "per-rollout LLM generate time (mean)"),
    ("agent_loop/generate_sequences/max", "per-rollout LLM generate time (max)"),
    ("agent_loop/tool_calls/mean", "per-rollout env tool-call time (mean)"),
    ("agent_loop/tool_calls/max", "per-rollout env tool-call time (max)"),
    ("agent_loop/slowest/generate_sequences", "slowest rollout: generate"),
    ("agent_loop/slowest/tool_calls", "slowest rollout: tool calls"),
    ("agent_loop/slowest/num_preempted", "slowest rollout: vLLM preemptions"),
]
ROLLOUT_STATS = [
    ("supo/trajs_per_rollout", "trajectories per rollout"),
    ("supo/num_summaries_mean", "summaries per rollout"),
    ("supo/num_tool_calls_mean", "tool calls per rollout"),
    ("supo/overlong_mean", "overlong (masked) rollout fraction"),
    ("supo/finished_mean", "finished rollout fraction"),
    ("critic/score/mean", "train reward mean"),
    ("response_length/mean", "response tokens per row (mean)"),
]


def parse(paths):
    steps = {}
    for path in paths:
        for line in open(path, errors="replace"):
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            step = int(m.group(1))
            kv = {k: float(v) for k, v in KV_RE.findall(m.group(2))}
            steps.setdefault(step, {}).update(kv)
    return steps


def table(rows, header):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    out = ["| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(header)) + " |",
           "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    out += ["| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--last", type=int, default=0, help="only the last N steps")
    ap.add_argument("--markdown", default=None, help="also write the tables to this file")
    args = ap.parse_args()
    paths = [p for g in args.logs for p in sorted(glob.glob(g))]
    steps = parse(paths)
    if not steps:
        sys.exit("no `step:N - ...` metric lines found")
    keys = sorted(steps)
    if args.last:
        keys = keys[-args.last:]
    rows = [steps[k] for k in keys]
    total = [r.get("timing_s/step") for r in rows if "timing_s/step" in r]
    mean_total = st.mean(total) if total else float("nan")

    out = [f"# Time distribution — {len(keys)} steps ({keys[0]}..{keys[-1]}), mean step {mean_total:.0f}s ({mean_total/60:.1f} min)\n"]
    prow = []
    for key, label in PHASES:
        vals = [r[key] for r in rows if key in r]
        if not vals:
            continue
        share = 100 * st.mean(vals) / mean_total if mean_total else float("nan")
        prow.append([label, f"{st.mean(vals):.1f}", f"{st.median(vals):.1f}", f"{max(vals):.1f}", f"{share:.1f}%", len(vals)])
    out.append(table(prow, ["phase", "mean s", "median s", "max s", "share of step", "n"]))
    arow = []
    for key, label in AGENT:
        vals = [r[key] for r in rows if key in r]
        if vals:
            arow.append([label, f"{st.mean(vals):.1f}", f"{st.median(vals):.1f}", f"{max(vals):.1f}"])
    if arow:
        out.append("\n## Inside rollout generation (per-rollout agent-loop timers)\n")
        out.append(table(arow, ["quantity", "mean", "median", "max"]))
    srow = []
    for key, label in ROLLOUT_STATS:
        vals = [r[key] for r in rows if key in r]
        if vals:
            srow.append([label, f"{st.mean(vals):.3f}", f"{vals[-1]:.3f}"])
    if srow:
        out.append("\n## Rollout statistics\n")
        out.append(table(srow, ["quantity", "mean over steps", "last step"]))
    gen = [r.get("timing_s/gen", 0) for r in rows]
    upd = [r.get("timing_s/update_actor", 0) for r in rows]
    if total:
        g, u = st.mean(gen) / mean_total, st.mean(upd) / mean_total
        out.append(f"\nBottleneck: {'rollout generation' if g >= u else 'actor update'} "
                   f"({100*max(g,u):.0f}% of step time). Generation is bounded by the slowest rollout "
                   f"(see agent_loop/slowest/*): long-horizon rollouts (many turns) dominate wall-clock, not tokens.")
    text = "\n".join(out)
    print(text)
    if args.markdown:
        open(args.markdown, "w").write(text + "\n")


if __name__ == "__main__":
    main()
