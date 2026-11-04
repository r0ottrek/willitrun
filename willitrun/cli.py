from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__, hardware, report, user_models
from .catalog import find_models
from .estimate import USABLE, best_fit

EPILOG = """examples:
  willitrun                         check this machine against popular models
  willitrun -m qwen3-30b-a3b        every quant of one model, plus the command to run it
  willitrun --ctx 32k               plan for long documents and coding agents
  willitrun --hf mistralai/Devstral-Small-2505   check any Hugging Face model once
  willitrun --gpu "RTX 3090" --ram 64            try hardware before you buy it
  willitrun --gpu "M4 Max" --ram 128             same, for a Mac

your own models:
  willitrun add org/repo            save a Hugging Face model to your list
  willitrun add ./config.json --params 8.2       or a local model's config.json
  willitrun models                  show the list (built-in and yours)
  willitrun remove <id>             take one of yours off the list
"""


def parse_ctx(text: str) -> int:
    t = text.lower().strip()
    mult = 1
    if t.endswith("k"):
        t, mult = t[:-1], 1024
    try:
        val = int(float(t) * mult)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' isn't a context length. Try 8192 or 32k.")
    if val < 256:
        raise argparse.ArgumentTypeError("context must be at least 256 tokens")
    return val


def positive(text: str) -> float:
    try:
        val = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' isn't a number")
    if val <= 0:
        raise argparse.ArgumentTypeError("must be more than 0")
    return val


def setup_terminal(force_ascii: bool, no_color: bool) -> None:
    """Make colors and box-drawing characters work, or fall back cleanly.

    Windows consoles need VT processing switched on for ANSI colors, and a
    redirected stdout on Windows defaults to a code page without box-drawing
    characters.
    """
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = k32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if k32.GetConsoleMode(handle, ctypes.byref(mode)):
                k32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except (AttributeError, OSError):
            report.set_color(False)
    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    enc = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
    try:
        "┏━█■·›…".encode(enc)
        unicode_ok = True
    except (LookupError, UnicodeEncodeError):
        unicode_ok = False
    report.set_unicode(unicode_ok and not force_ascii and not os.environ.get("WILLITRUN_ASCII"))
    if no_color:
        report.set_color(False)


def manage(argv: list[str]) -> int:
    """willitrun add / remove / models."""
    cmd = argv[0]
    p = argparse.ArgumentParser(prog=f"willitrun {cmd}")
    if cmd == "add":
        p.add_argument("source", help="Hugging Face repo (org/name) or a path to a config.json")
        p.add_argument("--params", type=positive, metavar="B",
                       help="total parameters in billions (needed for a local config.json)")
        p.add_argument("--name", help="display name")
        p.add_argument("--id", dest="model_id", help="short id to use with -m")
        p.add_argument("--tags", default="chat", help="comma list: chat,reasoning,coding,vision")
        p.add_argument("--ollama", help="Ollama tag, for the run command")
        p.add_argument("--gguf", help="Hugging Face GGUF repo, for the llama.cpp command")
    elif cmd == "remove":
        p.add_argument("model_id")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--ascii", action="store_true")
    a = p.parse_args(argv[1:])
    setup_terminal(a.ascii, a.no_color)

    if cmd == "models":
        catalog, added = user_models.full_catalog()
        print("\n".join([""] + report.model_list(catalog, added, user_models.models_path()) + [""]))
        return 0
    if cmd == "remove":
        if user_models.remove(a.model_id):
            print(f"Removed {a.model_id}.")
            return 0
        print(f"'{a.model_id}' isn't one of your added models. See: willitrun models", file=sys.stderr)
        return 2

    from dataclasses import replace
    from .hf import fetch_spec, spec_from_file
    if os.path.exists(a.source):
        path = os.path.join(a.source, "config.json") if os.path.isdir(a.source) else a.source
        spec = spec_from_file(path, a.params)
    elif "/" in a.source:
        spec = fetch_spec(a.source)
    else:
        print(f"'{a.source}' is neither a file nor a Hugging Face repo like org/name.", file=sys.stderr)
        return 2
    tags = tuple(t.strip() for t in a.tags.split(",") if t.strip()) or ("chat",)
    spec = replace(spec, id=(a.model_id or spec.id), name=(a.name or spec.name), tags=tags,
                   ollama=a.ollama or spec.ollama, gguf=a.gguf or spec.gguf)
    path, replaced = user_models.add(spec)
    moe = f", {spec.active:.1f}B active (MoE)" if spec.is_moe else ""
    print(f"{'Updated' if replaced else 'Added'} {spec.name}: {spec.params_b:.1f}B params{moe}, "
          f"{spec.layers} layers, up to {spec.max_ctx:,} context.")
    print(f"It's in every scan now. Details: willitrun -m {spec.id}")
    print(report.dim(f"Saved to {path}"))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("add", "remove", "models"):
        return manage(argv)

    p = argparse.ArgumentParser(
        prog="willitrun",
        description="Which open LLMs will run well on this machine, at what quantization and speed?",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = p.add_argument_group("what to check")
    what.add_argument("-m", "--model", help="one model in detail (id or name; see: willitrun models)")
    what.add_argument("--hf", metavar="ORG/REPO", help="check any Hugging Face model once, without saving it")
    what.add_argument("-c", "--ctx", type=parse_ctx, default=8192,
                      help="context length to plan for, e.g. 8k, 32k (default 8k)")
    what.add_argument("--kv-cache", choices=["f16", "q8_0", "q4_0"], default="f16",
                      help="KV cache precision (q8_0 halves its memory)")
    what.add_argument("--tag", choices=["chat", "reasoning", "coding", "vision"], help="only these models")
    what.add_argument("--min-speed", type=positive, default=USABLE,
                      help=f"tokens/sec you consider usable (default {USABLE:.0f})")
    hw = p.add_argument_group("hardware (override detection, or try hardware you don't own)")
    hw.add_argument("--gpu", help='GPU or Apple chip, e.g. "RTX 4090", "M3 Max", "none"')
    hw.add_argument("--gpu-count", type=int, default=1, help="number of identical GPUs")
    hw.add_argument("--vram", type=positive, help="VRAM per GPU in GB")
    hw.add_argument("--ram", type=positive, help="system RAM (or unified memory) in GB")
    hw.add_argument("--gpu-bw", type=positive, help="GPU memory bandwidth in GB/s")
    hw.add_argument("--ram-bw", type=positive, help="system RAM bandwidth in GB/s (DDR4 ~50, DDR5 ~80)")
    out = p.add_argument_group("output")
    out.add_argument("--all", action="store_true", help="list models that don't fit one per line")
    out.add_argument("--no-upgrade", action="store_true", help="hide the upgrade advisor")
    out.add_argument("--json", action="store_true", help="machine-readable output")
    out.add_argument("--no-color", action="store_true")
    out.add_argument("--ascii", action="store_true", help="plain ASCII, no box-drawing characters")
    out.add_argument("--list-gpus", action="store_true", help="show the GPU bandwidth table")
    out.add_argument("--help-apple", action="store_true", help="how to let a Mac's GPU use more memory")
    out.add_argument("--version", action="version", version=f"willitrun {__version__}")
    a = p.parse_args(argv)
    if a.gpu_count < 1:
        p.error("--gpu-count must be at least 1")
    setup_terminal(a.ascii, a.no_color or a.json)

    if a.list_gpus:
        print("\n".join([""] + report.gpu_list() + [""]))
        return 0

    env = hardware.env_overrides()
    try:
        machine = hardware.manual(a.gpu, a.ram, a.vram, a.gpu_bw or env.get("gpu_bw"),
                                  a.ram_bw or env.get("ram_bw"), a.gpu_count)
    except hardware.HardwareError as e:
        p.error(str(e))

    if a.help_apple:
        print("\n".join(report.apple_tip(machine)) if machine.apple_chip else
              "This isn't an Apple Silicon Mac. Try: willitrun --gpu 'M4 Max' --ram 64 --help-apple")
        return 0

    catalog, added = user_models.full_catalog()
    target = None
    if a.hf:
        from .hf import fetch_spec
        target = fetch_spec(a.hf)
    elif a.model:
        hits = find_models(a.model, catalog)
        if len(hits) > 1:
            print(f"'{a.model}' matches {len(hits)} models: " + ", ".join(m.id for m in hits), file=sys.stderr)
            print("Use one of those ids with -m.", file=sys.stderr)
            return 2
        if not hits:
            print(f"No model matches '{a.model}'. See the list with: willitrun models", file=sys.stderr)
            print("For anything else: willitrun add org/repo, or --hf org/repo to check it once.", file=sys.stderr)
            return 2
        target = hits[0]

    if a.json:
        models = [target] if target else [m for m in catalog if not a.tag or a.tag in m.tags]
        fits = [best_fit(m, machine, a.ctx, a.kv_cache, a.min_speed) for m in models]
        print(json.dumps(report.to_json(machine, fits), indent=2))
        return 0

    if target:
        lines = report.single(machine, target, a.ctx, a.kv_cache, a.min_speed)
    else:
        lines = report.overview(machine, catalog, a.ctx, a.kv_cache, a.min_speed,
                                a.tag, a.all, not a.no_upgrade, added)
    print("\n".join(line.rstrip() for line in lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
