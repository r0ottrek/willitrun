"""Terminal output.

Layout: a box-drawing logo, a memory map of the machine, the best picks,
every model grouped by how well it runs (with speed bars), what an upgrade
would unlock, and a few next commands. Falls back to plain ASCII when the
terminal can't show box-drawing characters.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
import textwrap
from dataclasses import replace

from . import __version__, hardware
from .catalog import ModelSpec
from .estimate import (GOOD, Fit, all_fits, best_fit, cpu_budget_gb, estimate, gpu_budget_gb,
                       quality_score, USABLE)
from .hardware import Machine

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
_UNICODE = True


def set_color(on: bool) -> None:
    global _COLOR
    _COLOR = on


def set_unicode(on: bool) -> None:
    global _UNICODE
    _UNICODE = on


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(t: str) -> str: return c(t, "1")
def dim(t: str) -> str: return c(t, "2")
def cyan(t: str) -> str: return c(t, "36")
def yellow(t: str) -> str: return c(t, "33")


def _strip(t: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", t)


def vlen(t: str) -> int:
    return len(_strip(t))


def pad(text: str, width: int) -> str:
    """ljust that ignores ANSI codes."""
    return text + " " * max(0, width - vlen(text))


def rpad(text: str, width: int) -> str:
    return " " * max(0, width - vlen(text)) + text


def fit_text(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: width - 1] + ("…" if _UNICODE else ".")


def width() -> int:
    return max(80, min(88, shutil.get_terminal_size((88, 24)).columns))


def ch(uni: str, ascii_: str) -> str:
    return uni if _UNICODE else ascii_


# --- building blocks -----------------------------------------------------------

# Three-row box-drawing letters, in the style of figlet's "future" font.
GLYPHS = {
    "w": ["╻ ╻", "┃╻┃", "┗┻┛"], "i": ["╻", "┃", "╹"], "l": ["╻  ", "┃  ", "┗━╸"],
    "t": ["╺┳╸", " ┃ ", " ╹ "], "r": ["┏━┓", "┣┳┛", "╹┗╸"], "u": ["╻ ╻", "┃ ┃", "┗━┛"],
    "n": ["┏┓╻", "┃┗┫", "╹ ╹"], " ": ["  ", "  ", "  "],
}


def big(text: str) -> list[str]:
    return ["".join(GLYPHS[x][row] for x in text) for row in range(3)]


def logo(side: list[str]) -> list[str]:
    if _UNICODE:
        art = big("will it run")
    else:
        art = ["will it run?", "------------", ""]
    w = max(len(a) for a in art)
    return [f"  {cyan(bold(a.ljust(w)))}   {s}" for a, s in zip(art, side)]


def section(title: str, right: str = "") -> list[str]:
    W = width()
    head = f"  {bold(title)}"
    if right:
        head += " " * max(2, W - 2 - vlen(head) - vlen(right)) + dim(right)
    return [head, "  " + dim(ch("─", "-") * (W - 4))]


SPEED_BAR = 10


def speed_bar(f: Fit) -> str:
    """Log scale: 1 tok/s is empty, 300 tok/s is full. Reading speed is ~5-10 tok/s."""
    if not f.runs:
        return " " * SPEED_BAR
    n = max(1, min(SPEED_BAR, round(SPEED_BAR * math.log(max(f.tok_s, 1)) / math.log(300))))
    color = {"great": "32", "good": "32", "usable": "33", "slow": "31"}[f.verdict]
    return c(ch("■", "#") * n, color) + dim(ch("·", ".") * (SPEED_BAR - n))


def speed(f: Fit) -> str:
    if not f.runs:
        return ""
    return f"{f.tok_s:.0f} tok/s" if f.tok_s >= 10 else f"{f.tok_s:.1f} tok/s"


def gb(x: float) -> str:
    return f"{x:.0f} GB" if x >= 10 else f"{x:.1f} GB"


# --- machine --------------------------------------------------------------------

def short_cpu(name: str) -> str:
    """'Intel(R) Core(TM) i7-12700K CPU @ 3.60GHz' -> 'Intel Core i7-12700K'."""
    n = re.sub(r"\((?:R|TM)\)|\bCPU\b|\bProcessor\b|@.*$|\bw/.*$|\b\d+-Core\b", " ", name, flags=re.I)
    return re.sub(r"\s+", " ", n).strip() or name


def machine_line(machine: Machine) -> str:
    if machine.unified:
        return f"{short_cpu(machine.cpu)} {ch('·', '-')} {machine.ram_gb:.0f} GB unified memory"
    parts = [short_cpu(machine.cpu)]
    names: dict[str, int] = {}
    for g in machine.gpus:
        names[g.name] = names.get(g.name, 0) + 1
    parts += [f"{n}x {name}" if n > 1 else name for name, n in names.items()]
    parts.append(f"{machine.ram_gb:.0f} GB RAM")
    return f" {ch('·', '-')} ".join(parts)


def memory_map(machine: Machine, ctx: int, kv_type: str) -> list[str]:
    """Each pool drawn on one shared GB scale, so size and speed compare at a glance."""
    W = width()
    kv = "" if kv_type == "f16" else f", {kv_type} KV cache"
    out = section("Memory", f"planning for {ctx:,} tokens of context{kv}")
    pools = []
    if machine.unified:
        pools.append(("GPU", "unified", gpu_budget_gb(machine), machine.ram_gb, machine.ram_bw_gbs))
    else:
        if machine.gpus:
            pools.append(("GPU", "VRAM", gpu_budget_gb(machine), machine.vram_gb, machine.gpu_bw_gbs or 0))
        pools.append(("RAM", "system", cpu_budget_gb(machine), machine.ram_gb, machine.ram_bw_gbs))
    scale = max(p[3] for p in pools) or 1
    bar_w = W - 4 - 4 - 30
    for label, kind, usable, total, bw in pools:
        n_use = round(bar_w * usable / scale)
        n_tot = max(n_use, round(bar_w * total / scale))
        bar = cyan(ch("█", "#") * n_use) + dim(ch("░", "-") * (n_tot - n_use))
        info = f"{gb(usable)} usable  {bw:>4.0f} GB/s"
        out.append(f"  {bold(label)}  {pad(bar, bar_w)}  {rpad(info, 24)}")
    if machine.unified:
        out.append(dim(f"       the GPU may use {gb(gpu_budget_gb(machine))} of {gb(machine.ram_gb)}; "
                       "macOS and apps keep the rest" if machine.apple_chip else
                       f"       the GPU share of {gb(machine.ram_gb)} is set by BIOS or driver"))
    elif machine.gpus:
        out.append(dim("       what doesn't fit on the GPU spills into RAM, which is much slower"))
    for n in machine.notes:
        out += note(n)
    return out


def note(text: str) -> list[str]:
    lines = textwrap.wrap(text, width() - 6)
    return [f"  {yellow('!') if i == 0 else ' '} {line}" for i, line in enumerate(lines)]


# --- overview -------------------------------------------------------------------

LABELS = (("General", "chat"), ("Reasoning", "reasoning"), ("Coding", "coding"), ("Vision", "vision"))


def picks(fits: list[Fit], min_speed: float = USABLE) -> list[str]:
    out = section("Best picks", "the model to start with, by task")
    good = [f for f in fits if f.runs and f.tok_s >= min_speed]
    if not good:
        out.append("  Nothing runs at a usable speed here. Try a shorter --ctx, or see Upgrades below.")
        return out
    chosen: dict[str, tuple[Fit, list[str]]] = {}
    for label, tag in LABELS:
        pool = [f for f in good if tag in f.model.tags]
        # Recommend setups that need no tweaking and no heavy quantization first.
        pool = [f for f in pool if f.rank >= 2 and f.placement != "unified-raise"] or pool
        pool = [f for f in pool if f.tok_s >= GOOD] or pool   # pleasant, not just possible
        if pool:
            best = max(pool, key=quality_score)
            chosen.setdefault(best.model.id, (best, []))[1].append(label.lower())
    fast = [f for f in good if f.tok_s >= 40 and "chat" in f.model.tags and f.model.id not in chosen]
    rows = [(f, labels) for f, labels in chosen.values()]
    if fast:
        rows.append((max(fast, key=quality_score), ["fast replies"]))
    for f, labels in rows:
        label = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]
        out.append(f"  {cyan(ch('●', '*'))} {label[0].upper() + label[1:]}")
        out.append(f"  {dim(ch('└─', '`-'))} " + row(f, bold, 24))
    return out


def row(f: Fit, style=lambda t: t, name_w: int = 27) -> str:
    """name, quant, memory, where, speed bar: 77 columns with the default name width."""
    return (f"{pad(style(fit_text(f.model.name, name_w - 1)), name_w)}{pad(f.quant, 7)}{rpad(gb(f.total_gb), 6)} "
            f"{pad(f.where, 16)}{speed_bar(f)}{rpad(speed(f), 10)}")


def model_rows(fits: list[Fit], added: set[str]) -> list[str]:
    out = []
    for f in fits:
        star = " *" if f.model.id in added else ""
        f2 = f if not star else replace(f, model=replace(f.model, name=fit_text(f.model.name, 23) + star))
        out.append("  " + row(f2))
    return out


def table(fits: list[Fit], show_all: bool, added: set[str]) -> list[str]:
    rows = sorted(fits, key=lambda f: -f.model.params_b)
    well = [f for f in rows if f.runs and f.tok_s >= GOOD]
    slow = [f for f in rows if f.runs and f.tok_s < GOOD]
    nope = [f for f in rows if not f.runs]
    head = dim(f"  {'model':<27}{'quant':<7}{'needs':>6} {'where':<16}speed")
    out: list[str] = []
    if well:
        out += section("Runs well", f"{len(well)} models at {GOOD:.0f}+ tokens/sec") + [head] + model_rows(well, added) + [""]
    if slow:
        out += section("Runs, slowly", "fine for batch jobs, not for chat") + [head] + model_rows(slow, added) + [""]
    if nope:
        out += section("Won't fit", "even at the smallest quant")
        if show_all:
            out += [dim(f"  {pad(fit_text(f.model.name, 25), 27)}{f.quant:<7}{rpad(gb(f.total_gb), 6)}  needed") for f in nope]
        else:
            items = [f"{f.model.name} ({gb(f.total_gb)})" for f in nope]
            line = "  "
            for i, item in enumerate(items):
                item += ", " if i < len(items) - 1 else ""
                if vlen(line) + len(item) > width() - 2:
                    out.append(dim(line.rstrip()))
                    line = "  "
                line += item
            out.append(dim(line.rstrip()))
        out.append("")
    if added:
        out.append(dim("  * added by you (willitrun models)"))
        out.append("")
    return out


# --- upgrade advisor ------------------------------------------------------------

RAM_STEPS = [16, 32, 48, 64, 96, 128, 192, 256]
APPLE_STEPS = [16, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512]
GPU_OPTIONS = ["RTX 5060 Ti", "RTX 3090", "RTX 4090", "RTX 5090"]


def _scenario(machine: Machine, ram: float | None = None, gpu: str | None = None) -> Machine:
    m = replace(machine, gpus=[replace(g) for g in machine.gpus], notes=[])
    if ram:
        m.ram_gb = ram
        if m.unified:
            m.unified_gpu_limit_gb = (hardware.apple_gpu_limit(ram) if m.apple_chip
                                      else ram * hardware.unified_share(m.cpu))
            if m.apple_chip:
                m.ram_bw_gbs = hardware.apple_bandwidth(m.apple_chip, ram) or m.ram_bw_gbs
    if gpu:
        spec = hardware.lookup_gpu(gpu)
        m.gpus = [hardware.Gpu(gpu, spec.vram_gb, spec.bw_gbs) for _ in machine.gpus or [None]]
    return m


def _gains(before: dict[str, Fit], machine: Machine, ctx: int, kv: str, min_speed: float,
           catalog: list[ModelSpec]) -> list[Fit]:
    gains = []
    for m in catalog:
        new = best_fit(m, machine, ctx, kv, min_speed)
        old = before[m.id]
        if new.runs and new.tok_s >= min_speed and (not old.runs or old.tok_s < min_speed):
            gains.append(new)
        elif new.runs and old.runs and old.tok_s >= min_speed and new.tok_s >= 2 * old.tok_s and new.tok_s >= GOOD:
            gains.append(new)
    gains.sort(key=quality_score, reverse=True)
    return gains[:3]


def upgrades(machine: Machine, fits: list[Fit], ctx: int, kv: str, min_speed: float,
             catalog: list[ModelSpec]) -> list[str]:
    before = {f.model.id: f for f in fits}
    out = section("Upgrades", "what more memory would unlock, before you pay for it")
    options: list[tuple[str, list[Fit]]] = []
    if machine.apple_chip:
        top = hardware.APPLE_MAX_RAM.get(machine.apple_chip.lower(), 512)
        for s in [s for s in APPLE_STEPS if machine.ram_gb + 1 < s <= top][:3]:
            options.append((f"{machine.apple_chip} with {s} GB",
                            _gains(before, _scenario(machine, ram=s), ctx, kv, min_speed, catalog)))
        if not options:
            out.append(dim(f"  This is the most memory a {machine.apple_chip} comes with."))
    elif machine.unified:
        out.append(dim("  Memory on this machine is soldered, so there is no upgrade to price here."))
    else:
        for s in [s for s in RAM_STEPS if s > machine.ram_gb + 1][:2]:
            options.append((f"{s} GB system RAM", _gains(before, _scenario(machine, ram=s), ctx, kv, min_speed, catalog)))
        count = max(1, len(machine.gpus))
        for gpu in GPU_OPTIONS:
            spec = hardware.lookup_gpu(gpu)
            if machine.gpus and spec.vram_gb * count <= machine.vram_gb + 0.5 and (machine.gpu_bw_gbs or 0) >= spec.bw_gbs:
                continue
            label = f"{count}x {gpu}" if count > 1 else gpu
            options.append((f"Swap to {label} ({spec.vram_gb * count:.0f} GB)" if machine.gpus
                            else f"Add an {gpu} ({spec.vram_gb:.0f} GB)",
                            _gains(before, _scenario(machine, gpu=gpu), ctx, kv, min_speed, catalog)))
    shown = 0
    for label, gains in options:
        if not gains:
            continue
        shown += 1
        out.append(f"  {cyan(ch('●', '*'))} {bold(label)}")
        for i, f in enumerate(gains):
            branch = dim(ch("└─", "`-") if i == len(gains) - 1 else ch("├─", "|-"))
            out.append(f"  {branch} {row(f, name_w=24)}")
    nothing = [label for label, gains in options if not gains]
    if nothing:
        text = f"No change worth paying for: {'; '.join(nothing)}."
        out += [dim("  " + line) for line in textwrap.wrap(text, width() - 4)]
    if not shown and not nothing and not (machine.unified and not options):
        out.append(dim("  The next steps up don't unlock anything new."))
    return out + [""]


# --- single model -----------------------------------------------------------------

def longest_context(m: ModelSpec, quant: str, rank: int, machine: Machine, kv: str) -> int:
    lo, hi = 0, m.max_ctx
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate(m, quant, rank, machine, mid, kv).placement in ("gpu", "unified"):
            lo = mid
        else:
            hi = mid - 1
    return lo


def run_commands(f: Fit, ctx: int, kv: str = "f16") -> list[str]:
    m = f.model
    expert_offload = f.placement == "split" and m.is_moe and m.shared_b is not None
    ollama, llama = [], []
    if m.ollama:
        env = "" if kv == "f16" else f"OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE={kv} "
        ollama.append(f"  {dim('$')} {env}ollama run {m.ollama}")
        if env:
            ollama.append(dim("    (set those two variables where the Ollama server runs, then restart it)"))
        if not m.native and f.quant != "Q4_K_M":
            ollama.append(dim(f"    Ollama's default tag is Q4_K_M. For {f.quant}, pick that tag on ollama.com/library."))
        ollama.append(dim(f"    then type: /set parameter num_ctx {ctx}"))
        if expert_offload:
            ollama.append(dim("    Ollama can't park only the experts in RAM, so expect it to be slower than this."))
    if m.gguf:
        cmd = f"llama-server -hf {m.gguf}" + ("" if m.native else f":{f.quant}") + f" -c {ctx}"
        if expert_offload:
            cmd += f" -ngl 999 --n-cpu-moe {m.layers - f.gpu_layers}"
        elif f.placement == "split":
            cmd += f" -ngl {f.gpu_layers}"
        elif f.placement == "cpu":
            cmd += " -ngl 0"
        else:
            cmd += " -ngl 999"
        if kv != "f16":
            cmd += f" -ctk {kv} -ctv {kv}"
        llama.append(f"  {dim('$')} {cmd} -fa on")
        if expert_offload:
            llama.append(dim("    --n-cpu-moe keeps attention on the GPU and parks only the experts in RAM,"))
            llama.append(dim("    which is what the speed above assumes."))
    out = llama + ollama if expert_offload else ollama + llama
    if not out:
        out.append(dim("  No Ollama tag or GGUF repo is known for this model. Search Hugging Face for"))
        out.append(dim(f"  \"{m.name} GGUF\" and load the {f.quant} file in llama.cpp or LM Studio."))
    return out


def detail(m: ModelSpec, machine: Machine, ctx: int, kv: str, min_speed: float) -> list[str]:
    fits = all_fits(m, machine, ctx, kv)
    best = best_fit(m, machine, ctx, kv, min_speed)
    eff_ctx = min(ctx, m.max_ctx)
    about = [f"{m.params_b:.1f}B params"]
    if m.is_moe:
        about.append(f"{m.active:.1f}B active per token")
    about.append(f"{m.max_ctx // 1024}k context" if m.max_ctx >= 1024 else f"{m.max_ctx} context")
    out = section(m.name, f" {ch('·', '-')} ".join(about))
    out.append(dim(f"  {'quant':<8}{'weights':>7}{'cache':>8}{'total':>8}  {'where':<16}speed"))
    for f in fits:
        mark = cyan(" " + ch("◂", "<")) if f is best and f.runs else ""
        out.append(f"  {pad(bold(f.quant) if f is best else f.quant, 8)}{rpad(gb(f.weights_gb), 7)}"
                   f"{rpad(gb(f.kv_gb), 8)}{rpad(gb(f.total_gb), 8)}  {pad(f.where if f.runs else dim(f.where), 16)}"
                   f"{speed_bar(f)}{rpad(speed(f), 10)}{mark}")
    out.append(dim(f"  cache = KV cache for {eff_ctx:,} tokens. Speeds are for a short chat."))
    out.append("")
    if best.runs:
        facts = []
        if abs(best.tok_s_full_ctx - best.tok_s) / best.tok_s > 0.05:
            facts.append(f"With the context window full, {best.quant} slows to about {best.tok_s_full_ctx:.0f} tok/s.")
        if best.placement in ("gpu", "unified"):
            longest = longest_context(m, best.quant, best.rank, machine, kv)
            facts.append(f"The longest context that stays fully on the GPU at {best.quant} is {longest:,} tokens.")
        for fact in facts:
            out += ["  " + line for line in textwrap.wrap(fact, width() - 4)]
        if facts:
            out.append("")
        out += section("Run it", f"{best.quant}, {best.where}")
        out += run_commands(best, eff_ctx, kv)
        out.append("")
    else:
        smallest = min(fits, key=lambda f: f.total_gb)
        have = gpu_budget_gb(machine) + (0 if machine.unified else cpu_budget_gb(machine))
        out += ["  " + line for line in textwrap.wrap(
            f"It needs about {gb(smallest.total_gb)} at {smallest.quant} with this context, "
            f"and about {gb(have)} is usable here.", width() - 4)]
        out.append("")
    for n in sorted(set(best.notes)):
        if "--help-apple" in n and machine.apple_chip:
            continue  # the full tip is printed right after
        out += note(n)
    return out


APPLE_TIP = [
    "macOS lets the GPU use about two thirds of memory (three quarters above 36 GB). "
    "To raise that until the next restart:",
    "  sudo sysctl iogpu.wired_limit_mb={mb}",
    "That leaves about 8 GB for macOS, so quit other apps first.",
]


def apple_tip(machine: Machine) -> list[str]:
    mb = int(max(machine.ram_gb - 8, machine.ram_gb * 0.75) * 1024)   # sysctl takes MiB
    return (note(APPLE_TIP[0]) + [f"    {APPLE_TIP[1].strip().format(mb=mb)}"]
            + ["    " + line for line in textwrap.wrap(APPLE_TIP[2], width() - 6)])


def hints(lines: list[tuple[str, str]]) -> list[str]:
    arrow = dim(ch("›", ">"))
    w = max(len(a) for a, _ in lines) + 3
    return [f"  {arrow} {bold(pad(a, w))}{dim(b)}" for a, b in lines]


def banner(machine: Machine) -> list[str]:
    return [""] + logo([dim(f"v{__version__}"), "which local LLMs will run here, and how fast",
                        dim(fit_text(machine_line(machine), width() - 36))]) + [""]


def overview(machine: Machine, catalog: list[ModelSpec], ctx: int, kv: str, min_speed: float,
             tag: str | None, show_all: bool, show_upgrades: bool, added: set[str] = frozenset()) -> list[str]:
    models = [m for m in catalog if not tag or tag in m.tags]
    fits = [best_fit(m, machine, ctx, kv, min_speed) for m in models]
    out = banner(machine) + memory_map(machine, ctx, kv) + [""] + picks(fits, min_speed) + [""] + table(fits, show_all, added)
    if any(f.placement == "unified-raise" for f in fits) and machine.apple_chip:
        out += apple_tip(machine) + [""]
    if show_upgrades:
        out += upgrades(machine, fits, ctx, kv, min_speed, models)
    out += hints([
        ("-m <model>", "every quant, longest context, and how to run it"),
        ("--ctx 32k", "plan for long documents and coding agents"),
        ("add <org/repo>", "add any Hugging Face model to your list"),
        ("--gpu \"RTX 5090\"", "try hardware before you buy it"),
    ])
    out.append(dim("  Estimates, usually within 25%. Speeds are for writing replies."))
    out.append("")
    return out


def single(machine: Machine, m: ModelSpec, ctx: int, kv: str, min_speed: float) -> list[str]:
    out = banner(machine) + memory_map(machine, ctx, kv) + [""] + detail(m, machine, ctx, kv, min_speed)
    best = best_fit(m, machine, ctx, kv, min_speed)
    if best.placement == "unified-raise" and machine.apple_chip:
        out += apple_tip(machine) + [""]
    return out


def to_json(machine: Machine, fits: list[Fit]) -> dict:
    return {
        "version": __version__,
        "machine": {
            "description": machine_line(machine), "os": machine.os, "cpu": machine.cpu,
            "ram_gb": round(machine.ram_gb, 1), "ram_bandwidth_gbs": machine.ram_bw_gbs,
            "unified_memory": machine.unified,
            "gpu_usable_gb": round(gpu_budget_gb(machine), 1),
            "gpus": [{"name": g.name, "vram_gb": round(g.vram_gb, 1), "bandwidth_gbs": g.bw_gbs}
                     for g in machine.gpus],
            "notes": machine.notes,
        },
        "models": [{
            "id": f.model.id, "name": f.model.name, "quant": f.quant, "context": f.ctx,
            "weights_gb": round(f.weights_gb, 2), "kv_cache_gb": round(f.kv_gb, 2),
            "total_gb": round(f.total_gb, 2), "placement": f.placement,
            "gpu_layers": f.gpu_layers, "tokens_per_sec": round(f.tok_s, 1),
            "tokens_per_sec_full_context": round(f.tok_s_full_ctx, 1),
            "verdict": f.verdict, "notes": f.notes,
        } for f in fits],
    }


def gpu_list() -> list[str]:
    out = section("Known GPUs and chips", "memory and bandwidth used for estimates")
    for name, spec in sorted(hardware.GPU_TABLE.items(), key=lambda kv: -kv[1].bw_gbs):
        kind = "unified" if spec.unified else "VRAM"
        out.append(f"  {name:<22}{spec.vram_gb:>5.0f} GB {kind:<8}{spec.bw_gbs:>6.0f} GB/s")
    for chip, bw in hardware.APPLE_BW.items():
        out.append(f"  {'apple ' + chip:<22}{'your RAM':>16}{bw:>6.0f} GB/s")
    return out


def model_list(catalog: list[ModelSpec], added: set[str], path: str) -> list[str]:
    out = section("Models", f"{len(catalog)} in your list")
    for m in sorted(catalog, key=lambda m: (m.id not in added, m.name.lower())):
        mark = cyan("*") if m.id in added else " "
        size = f"{m.params_b:.0f}B, {m.active:.0f}B active" if m.is_moe else f"{m.params_b:.1f}B"
        tags = fit_text(", ".join(m.tags), width() - 2 - 22 - 26 - 18 - 1)
        out.append(f" {mark}{pad(fit_text(m.id, 21), 22)}{pad(fit_text(m.name, 25), 26)}{pad(size, 18)}{dim(tags)}")
    out.append("")
    out.append(dim("  * added by you, saved in"))
    out.append(dim(f"    {path}"))
    return out
