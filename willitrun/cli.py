from __future__ import annotations

import argparse
import json
import sys

from . import __version__, hardware, report
from .catalog import CATALOG, find_model
from .estimate import USABLE, best_fit

EPILOG = """examples:
  willitrun                         check this machine against popular models
  willitrun -m qwen3-32b            every quant of one model, plus run commands
  willitrun --ctx 32768             plan for long documents / coding agents
  willitrun --hf mistralai/Devstral-Small-2505   any model on Hugging Face
  willitrun --gpu "RTX 3090" --ram 64            try hardware before you buy it
  willitrun --gpu "M4 Max" --ram 128             same, for a Mac
"""


def parse_ctx(text: str) -> int:
    t = text.lower().strip()
    mult = 1
    if t.endswith("k"):
        t, mult = t[:-1], 1024
    return int(float(t) * mult)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="willitrun",
        description="Which open LLMs will run well on this machine, at what quantization and speed?",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = p.add_argument_group("what to check")
    what.add_argument("-m", "--model", help="show every quant of one catalog model")
    what.add_argument("--hf", metavar="ORG/REPO", help="read any Hugging Face model's config and check it")
    what.add_argument("-c", "--ctx", type=parse_ctx, default=8192,
                      help="context length to plan for, e.g. 8k, 32k (default 8k)")
    what.add_argument("--kv-cache", choices=["f16", "q8_0", "q4_0"], default="f16",
                      help="KV cache precision (q8_0 halves its memory)")
    what.add_argument("--tag", choices=["chat", "reasoning", "coding", "vision"], help="only these models")
    what.add_argument("--min-speed", type=float, default=USABLE,
                      help=f"tokens/sec you consider usable (default {USABLE:.0f})")
    hw = p.add_argument_group("hardware (override detection, or try hardware you don't own)")
    hw.add_argument("--gpu", help='GPU or Apple chip, e.g. "RTX 4090", "M3 Max", "none"')
    hw.add_argument("--gpu-count", type=int, default=1, help="number of identical GPUs")
    hw.add_argument("--vram", type=float, help="VRAM per GPU in GB")
    hw.add_argument("--ram", type=float, help="system RAM (or unified memory) in GB")
    hw.add_argument("--gpu-bw", type=float, help="GPU memory bandwidth in GB/s")
    hw.add_argument("--ram-bw", type=float, help="system RAM bandwidth in GB/s (DDR4 ~40, DDR5 ~60-90)")
    out = p.add_argument_group("output")
    out.add_argument("--all", action="store_true", help="list models that don't fit, too")
    out.add_argument("--no-upgrade", action="store_true", help="hide the upgrade advisor")
    out.add_argument("--json", action="store_true", help="machine-readable output")
    out.add_argument("--no-color", action="store_true")
    out.add_argument("--list-gpus", action="store_true", help="show the GPU bandwidth table")
    out.add_argument("--help-apple", action="store_true", help="how to let a Mac's GPU use more memory")
    out.add_argument("--version", action="version", version=f"willitrun {__version__}")
    a = p.parse_args(argv)

    if a.no_color:
        report.set_color(False)
    if a.list_gpus:
        print("\n".join(report.gpu_list()))
        return 0

    env = hardware.env_overrides()
    machine = hardware.manual(a.gpu, a.ram, a.vram, a.gpu_bw or env.get("gpu_bw"),
                              a.ram_bw or env.get("ram_bw"), a.gpu_count)

    if a.help_apple:
        print(report.apple_tip(machine) if machine.apple_chip else
              "This isn't an Apple Silicon Mac. Try: willitrun --gpu 'M4 Max' --ram 64 --help-apple")
        return 0

    catalog = CATALOG
    target = None
    if a.hf:
        from .hf import fetch_spec
        target = fetch_spec(a.hf)
    elif a.model:
        target = find_model(a.model)
        if not target:
            print(f"No catalog model matches '{a.model}'. Known: "
                  + ", ".join(m.id for m in CATALOG), file=sys.stderr)
            print("For anything else use --hf org/repo.", file=sys.stderr)
            return 2

    if a.json:
        models = [target] if target else [m for m in catalog if not a.tag or a.tag in m.tags]
        fits = [best_fit(m, machine, a.ctx, a.kv_cache, a.min_speed) for m in models]
        print(json.dumps(report.to_json(machine, fits), indent=2))
        return 0

    if target:
        lines = report.header(machine, a.ctx, a.kv_cache) + [""] + \
            report.detail(target, machine, a.ctx, a.kv_cache, a.min_speed)
        fit = best_fit(target, machine, a.ctx, a.kv_cache, a.min_speed)
        if fit.placement == "unified-raise" and machine.apple_chip:
            lines += ["", report.c(report.apple_tip(machine), "33")]
    else:
        lines = report.overview(machine, catalog, a.ctx, a.kv_cache, a.min_speed,
                                a.tag, a.all, not a.no_upgrade)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
