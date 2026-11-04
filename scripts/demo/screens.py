"""Rebuild the screenshots in docs/ from real willitrun output.

The machine is fixed (Ryzen 7 7700X, RTX 3090, 64 GB DDR5) so the images
don't depend on whoever runs this. Needs Node with Playwright for the PNGs:

    python scripts/demo/screens.py && node scripts/demo/render.mjs
"""

import contextlib
import html
import io
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ["WILLITRUN_HOME"] = tempfile.mkdtemp()   # no user models in the pictures
os.environ["COLUMNS"] = "88"

from willitrun import cli, hardware, report  # noqa: E402


def demo_machine() -> hardware.Machine:
    spec = hardware.lookup_gpu("RTX 3090")
    return hardware.Machine("Linux", "AMD Ryzen 7 7700X 8-Core Processor", 64, 83.2,
                            [hardware.Gpu("RTX 3090", spec.vram_gb, spec.bw_gbs)])


hardware.detect = demo_machine

COLORS = {"1": "font-weight:700;color:#ededed", "2": "color:#7a7a7a", "32": "color:#8fbf7f",
          "33": "color:#d7a85f", "31": "color:#e06c5f", "36": "color:#6fb8c4"}


def capture(args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.main(args)
    report.set_color(True)
    return buf.getvalue()


def to_html(text):
    out, depth = [], 0
    for part in re.split(r"(\033\[[0-9;]*m)", text):
        m = re.fullmatch(r"\033\[([0-9;]*)m", part)
        if not m:
            out.append(html.escape(part))
        elif m.group(1) == "0":
            out.append("</span>" * depth)
            depth = 0
        else:
            out.append(f'<span style="{COLORS.get(m.group(1), "")}">')
            depth += 1
    return "".join(out)


def page(cmd, args, name):
    report.set_color(True)
    body = to_html(capture(args).rstrip("\n"))
    doc = f"""<!doctype html><meta charset=utf-8><style>
body{{margin:0;background:#161616}}
pre{{margin:0;padding:22px 26px 26px;font:15px/1.42 'DejaVu Sans Mono',monospace;color:#cfcfcf;
display:inline-block;background:#161616}}
.p{{color:#7a7a7a}}.cursor{{background:#cfcfcf;color:#161616}}</style>
<pre><span class=p>~ $</span> {html.escape(cmd)}\n{body}\n<span class=p>~ $</span> <span class=cursor> </span></pre>"""
    path = os.path.join(ROOT, "docs", f"{name}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    print("wrote", path)


report.set_color(True)
page("willitrun", ["--no-upgrade"], "terminal")
page("willitrun -m gpt-oss-120b --ctx 32k", ["-m", "gpt-oss-120b", "--ctx", "32k"], "model-detail")
page("willitrun --gpu \"RTX 5090\"", ["--gpu", "RTX 5090"], "what-if")
