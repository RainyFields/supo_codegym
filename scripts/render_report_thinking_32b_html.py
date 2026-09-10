#!/usr/bin/env python
"""Render docs/report_thinking_32b/REPORT.md to a self-contained report.html (images embedded as data URIs).
Supports the markdown subset used in the report: #/##/**bold**/*italic*/`code`/links/bullets/tables/images/fenced code."""
import base64, html, os, re, sys
D = os.path.join(os.path.dirname(__file__), "..", "docs", "report_thinking_32b")
md = open(os.path.join(D, "REPORT.md")).read()

def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<em>\1</em>", t)
    t = re.sub(r"(https?://[^\s<)]+)", r'<a href="\1">\1</a>', t)
    return t

def img_uri(path):
    p = os.path.join(D, path); b = base64.b64encode(open(p, "rb").read()).decode()
    return f"data:image/png;base64,{b}"

out = []; lines = md.split("\n"); i = 0; para = []
def flush():
    global para
    if para:
        txt = " ".join(para).strip()
        if txt.startswith("*") and txt.endswith("*") and txt.count("*") == 2:
            out.append(f"<p class='caption'>{inline(txt[1:-1])}</p>")
        else:
            out.append(f"<p>{inline(txt)}</p>")
        para = []
while i < len(lines):
    l = lines[i]
    if l.startswith("```"):
        flush(); j = i + 1; buf = []
        while j < len(lines) and not lines[j].startswith("```"): buf.append(lines[j]); j += 1
        out.append("<pre class='traj'>" + html.escape("\n".join(buf)) + "</pre>"); i = j + 1; continue
    if l.startswith("# "): flush(); out.append(f"<h1>{inline(l[2:])}</h1>"); i += 1; continue
    if l.startswith("## "): flush(); sid = re.sub(r"[^a-z0-9]+", "-", l[3:].lower()).strip("-"); out.append(f"<h2 id='{sid}'>{inline(l[3:])}</h2>"); i += 1; continue
    m = re.match(r"!\[(.*?)\]\((.*?)\)", l.strip())
    if m: flush(); out.append(f"<figure><img src='{img_uri(m.group(2))}' alt='{html.escape(m.group(1))}'></figure>"); i += 1; continue
    if l.startswith("|"):
        flush(); rows = []
        while i < len(lines) and lines[i].startswith("|"): rows.append(lines[i]); i += 1
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows if not re.match(r"^\|[\s\-:|]+\|$", r.strip())]
        t = "<div class='tablewrap'><table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells[0]) + "</tr></thead><tbody>"
        for r in cells[1:]: t += "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
        out.append(t + "</tbody></table></div>"); continue
    if l.startswith("- "):
        flush(); items = []
        while i < len(lines) and lines[i].startswith("- "): items.append(lines[i][2:]); i += 1
        out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>"); continue
    if not l.strip(): flush(); i += 1; continue
    para.append(l); i += 1
flush()
body = "\n".join(out)
toc = "".join(f"<a href='#{m.group(1)}'>{html.escape(m.group(2))}</a>" for m in re.finditer(r"<h2 id='([^']+)'>(.*?)</h2>", body))
page = f"""<title>SUPO CodeGym: Thinking &amp; 32B</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#FAFAF7;--ink:#1C2430;--muted:#5C6B7A;--rule:#D9DDE3;--accent:#0F4D92;--accent2:#42949E;--red:#B64342;--code:#F1F3F6;--tile:#FFFFFF;--shadow:0 1px 2px rgba(20,30,50,.06)}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#12171F;--ink:#E6E9EE;--muted:#9AA6B4;--rule:#2A323D;--accent:#7FB0E6;--accent2:#6DBCC5;--red:#E07B78;--code:#1B222C;--tile:#182029;--shadow:none}}}}
:root[data-theme="dark"]{{--bg:#12171F;--ink:#E6E9EE;--muted:#9AA6B4;--rule:#2A323D;--accent:#7FB0E6;--accent2:#6DBCC5;--red:#E07B78;--code:#1B222C;--tile:#182029;--shadow:none}}
body{{background:var(--bg);color:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif;font-size:16px;line-height:1.55;margin:0}}
main{{max-width:900px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-family:"Source Serif 4",Georgia,serif;font-weight:600;font-size:2.1rem;line-height:1.15;margin:0 0 .4em;text-wrap:balance}}
h2{{font-family:"Source Serif 4",Georgia,serif;font-weight:600;font-size:1.35rem;margin:2.2em 0 .6em;padding-top:.6em;border-top:1px solid var(--rule)}}
p{{max-width:70ch;margin:.7em 0}} p.caption{{font-size:.9rem;color:var(--muted);max-width:none;border-left:3px solid var(--accent2);padding-left:12px;margin:.4em 0 1.6em}}
ul{{max-width:76ch;padding-left:1.2em}} li{{margin:.35em 0}}
code{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.86em;background:var(--code);padding:1px 4px;border-radius:3px}}
pre.traj{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.78rem;line-height:1.45;background:var(--code);border:1px solid var(--rule);padding:12px 14px;overflow-x:auto;white-space:pre-wrap;word-break:break-word;margin:.6em 0 1.4em}}
figure{{margin:1.2em 0 .4em}} figure img{{width:100%;height:auto;display:block;background:#fff;border:1px solid var(--rule)}}
.tablewrap{{overflow-x:auto;margin:1em 0 1.4em}} table{{border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums;min-width:100%}}
th,td{{padding:6px 9px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}} th{{font-weight:600;color:var(--muted);font-size:.8rem;letter-spacing:.03em;text-transform:uppercase;border-bottom:2px solid var(--rule)}}
td:first-child{{white-space:nowrap}}
nav{{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:.85rem;margin:1em 0 1.5em}} nav a{{color:var(--accent);text-decoration:none}} nav a:hover,nav a:focus{{text-decoration:underline}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:1.2em 0 .4em}}
.tile{{background:var(--tile);border:1px solid var(--rule);padding:12px 14px;box-shadow:var(--shadow)}} .tile b{{display:block;font-family:"Source Serif 4",Georgia,serif;font-size:1.6rem;font-weight:600;line-height:1.1}} .tile span{{font-size:.82rem;color:var(--muted)}}
.tile.red b{{color:var(--red)}} .tile.blue b{{color:var(--accent)}} .tile.teal b{{color:var(--accent2)}}
a{{color:var(--accent)}} a:focus{{outline:2px solid var(--accent2);outline-offset:2px}}
@media (prefers-reduced-motion: reduce){{*{{scroll-behavior:auto}}}}
</style>
<main>
<div class="tiles">
<div class="tile blue"><b>.867 → .859</b><span>9B GRPO-32K, final accuracy without → with thinking (n=128)</span></div>
<div class="tile teal"><b>.758 → .695</b><span>9B SUPO-4K×8, final accuracy without → with thinking</span></div>
<div class="tile red"><b>.523 → .836</b><span>Qwen2.5-32B-Instruct GRPO-32K, step 0 → step 100 (+.312)</span></div>
<div class="tile"><b>95 / 128</b><span>tasks solved by both final GRPO policies (16 only 9B, 12 only 32B, 5 neither)</span></div>
</div>
<nav>{toc}</nav>
{body}
</main>
"""
open(os.path.join(D, "report.html"), "w").write(page)
print("report.html", len(page) // 1024, "KB")
