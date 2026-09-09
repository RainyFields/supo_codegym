#!/usr/bin/env python
"""Build the replication report assets from the validation dumps + timing tables.
  python scripts/make_report.py [--out docs/report]
Reads $PROJECT_ROOT/outputs/<exp>/val/<step>.jsonl for both arms, writes
  <out>/val_curves.png, <out>/tool_calls.png, <out>/val_table.md, <out>/val_metrics.csv,
  <out>/summary.json  (all numbers the report quotes come from these files).
"""
import argparse, csv, glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("PROJECT_ROOT", "/mnt/hdfs/mlsys/users/xiaoxuan/supo_codegym")
ARMS = {"GRPO 32K": "grpo_codegym_qwen35-9b_32k", "SUPO 4K×8": "supo_codegym_qwen35-9b_4kx8"}
PAPER = {"GRPO 32K": (32.0, 44.5, 52.1), "SUPO 4K×8": (32.8, 47.7, 54.7)}  # acc before, after, tool calls (Table 1)

def load(exp):
    rows = {}
    for f in glob.glob(f"{ROOT}/outputs/{exp}/val/*.jsonl"):
        step = int(os.path.basename(f).split(".")[0])
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if len(recs) < 100: continue
        n = len(recs)
        m = lambda k: sum(float(r.get(k, 0)) for r in recs) / n
        rows[step] = dict(step=step, n=n, acc=m("score"), finished=m("finished"), overlong=m("overlong"),
                          tool_calls=m("num_tool_calls"), summaries=m("num_summaries"), trajs=m("num_trajs"),
                          invalid=m("num_invalid_calls"), resp_tokens=m("rollout_response_tokens"), steps=m("num_steps"))
    return [rows[s] for s in sorted(rows)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "report"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    data = {arm: load(exp) for arm, exp in ARMS.items()}
    # table + csv
    keys = ["step", "acc", "tool_calls", "summaries", "trajs", "overlong", "finished", "invalid", "resp_tokens"]
    with open(os.path.join(a.out, "val_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm"] + keys)
        for arm, rows in data.items():
            for r in rows: w.writerow([arm] + [round(r[k], 4) if isinstance(r[k], float) else r[k] for k in keys])
    steps = sorted({r["step"] for rows in data.values() for r in rows})
    lines = ["| step | " + " | ".join(steps and [str(s) for s in steps]) + " |", "|---|" + "---|" * len(steps)]
    for arm, rows in data.items():
        d = {r["step"]: r for r in rows}
        lines.append(f"| {arm} acc | " + " | ".join(f"{d[s]['acc']:.3f}" if s in d else "–" for s in steps) + " |")
    for arm, rows in data.items():
        d = {r["step"]: r for r in rows}
        lines.append(f"| {arm} tool calls | " + " | ".join(f"{d[s]['tool_calls']:.1f}" if s in d else "–" for s in steps) + " |")
    open(os.path.join(a.out, "val_table.md"), "w").write("\n".join(lines) + "\n")
    # summary
    summ = {}
    for arm, rows in data.items():
        if not rows: continue
        first, last = rows[0], rows[-1]; best = max(rows, key=lambda r: r["acc"])
        summ[arm] = dict(steps_evaluated=[r["step"] for r in rows], acc_first=first["acc"], acc_last=last["acc"], last_step=last["step"],
                         acc_best=best["acc"], best_step=best["step"], acc_mean_last3=sum(r["acc"] for r in rows[-3:]) / len(rows[-3:]),
                         tool_calls_first=first["tool_calls"], tool_calls_last=last["tool_calls"], summaries_last=last["summaries"],
                         trajs_last=last["trajs"], overlong_first=first["overlong"], overlong_last=last["overlong"],
                         resp_tokens_last=last["resp_tokens"], paper_before=PAPER[arm][0] / 100, paper_after=PAPER[arm][1] / 100, paper_tool_calls=PAPER[arm][2])
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    # figures
    colors = {"GRPO 32K": "#8A5A12", "SUPO 4K×8": "#0F6E78"}
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    for arm, rows in data.items():
        ax.plot([r["step"] for r in rows], [r["acc"] for r in rows], marker="o", ms=4, lw=1.6, color=colors[arm], label=arm)
    ax.set_xlabel("training step (128 prompts × 8 rollouts each)"); ax.set_ylabel("greedy pass@1, 128 held-out seeds")
    ax.set_ylim(0.55, 0.95); ax.grid(alpha=.25); ax.legend(frameon=False); ax.set_title("CodeGym validation accuracy, Qwen3.5-9B", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "val_curves.png")); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), dpi=160)
    for arm, rows in data.items():
        x = [r["step"] for r in rows]
        axs[0].plot(x, [r["tool_calls"] for r in rows], marker="o", ms=3, color=colors[arm], label=arm)
        axs[1].plot(x, [r["overlong"] for r in rows], marker="o", ms=3, color=colors[arm], label=arm)
        axs[2].plot(x, [r["summaries"] for r in rows], marker="o", ms=3, color=colors[arm], label=arm)
    for ax, t in zip(axs, ["tool calls per episode", "fraction of episodes overlong", "summaries per episode"]):
        ax.set_title(t, fontsize=9); ax.set_xlabel("step"); ax.grid(alpha=.25)
    axs[0].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(a.out, "tool_calls.png")); plt.close(fig)
    print(open(os.path.join(a.out, "val_table.md")).read()); print(json.dumps(summ, indent=1)[:1500])

if __name__ == "__main__":
    main()
