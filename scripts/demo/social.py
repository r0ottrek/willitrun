"""docs/social-preview.png: the 1280x640 card GitHub and LinkedIn show for links."""

import html
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from willitrun.report import big  # noqa: E402

logo = "\n".join(big("will it run"))
rows = [
    ("gpt-oss 120B (MoE)", "MXFP4", "62 GB", "34% on GPU", 6, "26 tok/s", "#8fbf7f"),
    ("Qwen3 32B", "Q4_K_M", "22 GB", "on GPU", 6, "31 tok/s", "#8fbf7f"),
    ("Qwen3 30B-A3B (MoE)", "Q5_K_M", "22 GB", "on GPU", 9, "167 tok/s", "#8fbf7f"),
    ("Llama 3.3 70B", "Q4_K_M", "44 GB", "51% on GPU", 1, "2.3 tok/s", "#e06c5f"),
]
table = "\n".join(
    f"{html.escape(n):<22}{q:<8}{m:>6}  {w:<12}<span style='color:{c}'>{'■' * b}</span>"
    f"<span style='color:#555'>{'·' * (10 - b)}</span> {s:>9}"
    for n, q, m, w, b, s, c in rows)
doc = f"""<!doctype html><meta charset=utf-8><style>
body{{margin:0;width:1280px;height:640px;background:#161616;color:#cfcfcf;
font-family:'DejaVu Sans Mono',monospace;display:flex;flex-direction:column;justify-content:center;padding:0 90px;box-sizing:border-box}}
pre{{margin:0;font:inherit}}
.logo{{color:#6fb8c4;font-size:46px;line-height:1.05;font-weight:700}}
.tag{{font-size:28px;margin:30px 0 8px}}
.sub{{font-size:20px;color:#7a7a7a;margin-bottom:38px}}
.t{{font-size:19px;line-height:1.6}}</style>
<pre class=logo>{logo}</pre>
<div class=tag>Which local LLMs will run on your machine, and how fast.</div>
<div class=sub>Reads your GPU, RAM and bandwidth. Context-aware, MoE-aware, one command.</div>
<pre class=t>{table}</pre>"""
out = os.path.join(ROOT, "docs", "social-preview.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(doc)
print("wrote", out)
