"""Terminal output."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import replace

from . import hardware
from .catalog import CATALOG, ModelSpec
from .estimate import (Fit, all_fits, best_fit, cpu_budget_gb, gpu_budget_gb,
                       quality_score, estimate, USABLE)
from .hardware import Machine

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def set_color(on: bool) -> None:
    global _COLOR
    _COLOR = on


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(t: str) -> str: return c(t, "1")
def dim(t: str) -> str: return c(t, "2")


VERDICT_COLOR = {"great": "32", "good": "32", "usable": "33", "slow": "31", "won't fit": "2"}


def verdict(f: Fit) -> str:
    return c(f.verdict, VERDICT_COLOR[f.verdict])


def speed(f: Fit) -> str:
    if not f.runs:
        return "-"
    s = f.tok_s
    return f"~{s:.0f} tok/s" if s >= 10 else f"~{s:.1f} tok/s"


def pad(text: str, width: int) -> str:
    """ljust that ignores ANSI codes."""
    return text + " " * max(0, width - len(_strip(text)))


def _strip(t: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", t)


def header(machine: Machine, ctx: int, kv_type: str) -> list[str]:
    out = [bold("Will it run? ") + machine.describe()]
    if machine.unified:
        line = (f"GPU can use ~{gpu_budget_gb(machine):.0f} GB of it · "
                f"{machine.ram_bw_gbs:.0f} GB/s")
    elif machine.gpus:
        line = (f"GPU memory usable ~{gpu_budget_gb(machine):.0f} GB at "
                f"{machine.gpu_bw_gbs:.0f} GB/s · spill-over RAM ~{cpu_budget_gb(machine):.0f} GB "
                f"at ~{machine.ram_bw_gbs:.0f} GB/s")
    else:
        line = (f"No GPU found, CPU only · RAM usable ~{cpu_budget_gb(machine):.0f} GB "
                f"at ~{machine.ram_bw_gbs:.0f} GB/s")
    kv = "" if kv_type == "f16" else f" ({kv_type} KV cache)"
    out.append(dim(f"{line} · context {ctx:,} tokens{kv}"))
    for n in machine.notes:
        out.append(c("! " + n, "33"))
    return out


def picks(fits: list[Fit]) -> list[str]:
    good = [f for f in fits if f.runs and f.tok_s >= USABLE]
    if not good:
        return [bold("Best picks"), "  Nothing runs at a usable speed. Try a smaller --ctx or see the upgrade advisor."]
    out = [bold("Best picks for this machine")]
    seen: dict[str, list[str]] = {}
    chosen: dict[str, Fit] = {}
    for label, tag in (("General", "chat"), ("Reasoning", "reasoning"), ("Coding", "coding"), ("Vision", "vision")):
        pool = [f for f in good if tag in f.model.tags]
        # A pick should be pleasant to use, not just possible.
        pool = [f for f in pool if f.tok_s >= 15] or pool
        if pool:
            best = max(pool, key=quality_score)
            seen.setdefault(best.model.id, []).append(label)
            chosen[best.model.id] = best
    for mid, labels in seen.items():
        best = chosen[mid]
        label = " & ".join(labels) if len(labels) < 3 else ", ".join(labels[:-1]) + " & " + labels[-1]
        out.append(f"  {pad(label, 30)}{pad(best.model.name, 30)}{pad(best.quant, 8)}{pad(speed(best), 13)}{best.where}")
    fast = [f for f in good if f.tok_s >= 40 and f.model.id not in seen]
    if fast:
        best = max(fast, key=quality_score)
        out.append(f"  {pad('Fastest good option', 30)}{pad(best.model.name, 30)}{pad(best.quant, 8)}{pad(speed(best), 13)}{best.where}")
    return out


def table(fits: list[Fit], show_all: bool) -> list[str]:
    rows = sorted(fits, key=lambda f: (not f.runs, -f.model.params_b))
    runs = [f for f in rows if f.runs]
    out = [bold(f"{'Model':<32}{'Quant':<8}{'Needs':>9}  {'Where':<23}{'Speed':<13}Verdict")]
    for f in runs:
        out.append(f"{f.model.name:<32}{f.quant:<8}{f.total_gb:>6.1f} GB  {f.where:<23}"
                   f"{speed(f):<13}{verdict(f)}")
    nope = [f for f in rows if not f.runs]
    if nope:
        if show_all:
            for f in nope:
                out.append(dim(f"{f.model.name:<32}{f.quant:<8}{f.total_gb:>6.1f} GB  won't fit"))
        else:
            names = ", ".join(f"{f.model.name} ({f.total_gb:.0f} GB)" for f in nope)
            out.append(dim(f"Won't fit even at the smallest quant: {names}"))
    return out


# --- upgrade advisor ------------------------------------------------------

RAM_STEPS = [16, 32, 48, 64, 96, 128, 192, 256]
APPLE_STEPS = [16, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512]
GPU_OPTIONS = ["RTX 5060 Ti", "RTX 3090", "RTX 4090", "RTX 5090"]


def _scenario(machine: Machine, ram: float | None = None, gpu: str | None = None) -> Machine:
    m = replace(machine, gpus=[replace(g) for g in machine.gpus], notes=[])
    if ram:
        m.ram_gb = ram
        if m.unified:
            m.unified_gpu_limit_gb = (hardware.apple_gpu_limit(ram) if m.apple_chip else ram * 0.75)
    if gpu:
        spec = hardware.lookup_gpu(gpu)
        m.gpus = [hardware.Gpu(gpu, spec.vram_gb, spec.bw_gbs)]
    return m


def _gains(before: dict[str, Fit], machine: Machine, ctx: int, kv: str, min_speed: float,
           catalog: list[ModelSpec]) -> list[str]:
    gains = []
    for m in catalog:
        new = best_fit(m, machine, ctx, kv, min_speed)
        old = before[m.id]
        if new.runs and new.tok_s >= min_speed and (not old.runs or old.tok_s < min_speed):
            gains.append(new)
        elif new.runs and old.runs and old.tok_s >= min_speed and new.tok_s >= 2 * old.tok_s and new.tok_s >= 15:
            gains.append(new)
    gains.sort(key=quality_score, reverse=True)
    return [f"{f.model.name} {f.quant} {speed(f)}" for f in gains[:4]]


def upgrades(machine: Machine, fits: list[Fit], ctx: int, kv: str, min_speed: float,
             catalog: list[ModelSpec]) -> list[str]:
    before = {f.model.id: f for f in fits}
    out = [bold("Upgrade advisor") + dim("  (what more memory actually buys you, before you pay for it)")]
    lines = []
    if machine.unified:
        steps = [s for s in APPLE_STEPS if s > machine.ram_gb + 1][:3]
        for s in steps:
            g = _gains(before, _scenario(machine, ram=s), ctx, kv, min_speed, catalog)
            if g:
                lines.append(f"  Same chip with {s} GB: " + ", ".join(g))
    else:
        steps = [s for s in RAM_STEPS if s > machine.ram_gb + 1][:2]
        for s in steps:
            g = _gains(before, _scenario(machine, ram=s), ctx, kv, min_speed, catalog)
            label = f"  {s} GB system RAM: "
            lines.append(label + (", ".join(g) if g else dim("nothing new at a usable speed")))
        for gpu in GPU_OPTIONS:
            spec = hardware.lookup_gpu(gpu)
            if spec.vram_gb <= machine.vram_gb and (machine.gpu_bw_gbs or 0) >= spec.bw_gbs:
                continue
            g = _gains(before, _scenario(machine, gpu=gpu), ctx, kv, min_speed, catalog)
            if g:
                lines.append(f"  Swap to {gpu} ({spec.vram_gb:.0f} GB): " + ", ".join(g))
    if not lines:
        lines.append("  The next steps up don't unlock anything new in the catalog.")
    return out + lines


# --- single-model detail ------------------------------------------------------

def longest_context(m: ModelSpec, quant: str, rank: int, machine: Machine, kv: str) -> int:
    lo, hi = 0, m.max_ctx
    while lo < hi:
        mid = (lo + hi + 1) // 2
        f = estimate(m, quant, rank, machine, mid, kv)
        if f.placement in ("gpu", "unified"):
            lo = mid
        else:
            hi = mid - 1
    return lo


def run_commands(f: Fit, ctx: int) -> list[str]:
    m = f.model
    out = []
    if m.ollama:
        tag = m.ollama
        if not m.native and f.quant != "Q4_K_M":
            tag += dim(f"   (default is Q4_K_M; pick a {f.quant} tag on ollama.com/library)")
        out.append(f"  ollama run {tag}")
        if ctx != 2048:
            out.append(dim(f"  then /set parameter num_ctx {ctx}"))
    if m.gguf:
        cmd = f"  llama-server -hf {m.gguf}"
        if not m.native:
            cmd += f":{f.quant}"
        cmd += f" -c {ctx}"
        if f.placement == "split":
            if m.is_moe:
                cmd += f" -ngl 999 --n-cpu-moe {m.layers - f.gpu_layers}"
            else:
                cmd += f" -ngl {f.gpu_layers}"
        elif f.placement != "cpu":
            cmd += " -ngl 999"
        else:
            cmd += " -ngl 0"
        cmd += " -fa on"
        out.append(cmd)
    return out


def detail(m: ModelSpec, machine: Machine, ctx: int, kv: str, min_speed: float) -> list[str]:
    fits = all_fits(m, machine, ctx, kv)
    best = best_fit(m, machine, ctx, kv, min_speed)
    eff_ctx = min(ctx, m.max_ctx)
    moe = f", {m.active:.1f}B active per token (MoE)" if m.is_moe else ""
    out = [bold(m.name) + dim(f"  {m.params_b:.1f}B params{moe}, up to {m.max_ctx:,} context")]
    out.append(bold(f"{'Quant':<9}{'Weights':>9}{'KV cache':>10}{'Total':>9}  {'Where':<23}{'Speed':<13}"
                    f"{'@ full ctx':<12}Verdict"))
    for f in fits:
        mark = c(" <- pick", "36") if f.quant == best.quant and f.runs else ""
        full = f"~{f.tok_s_full_ctx:.0f}" if f.runs and f.tok_s_full_ctx >= 10 else (
            f"~{f.tok_s_full_ctx:.1f}" if f.runs else "-")
        out.append(f"{f.quant:<9}{f.weights_gb:>6.1f} GB{f.kv_gb:>7.1f} GB{f.total_gb:>6.1f} GB  "
                   f"{f.where:<23}{speed(f):<13}{full:<12}{verdict(f)}{mark}")
    out.append(dim(f"KV cache is for {eff_ctx:,} tokens of context. Speeds are for a short chat; "
                   "'@ full ctx' is with the context window full."))
    if best.runs:
        out.append("")
        if best.placement in ("gpu", "unified"):
            longest = longest_context(m, best.quant, best.rank, machine, kv)
            out.append(f"Longest context that stays fully on the GPU at {best.quant}: {longest:,} tokens")
        out.append(bold("Run it"))
        out.extend(run_commands(best, eff_ctx))
        if best.placement == "split" and m.is_moe:
            out.append(dim("  --n-cpu-moe keeps attention on the GPU and parks the experts in RAM, "
                           "which is much faster than splitting whole layers."))
    for n in sorted(set(best.notes)):
        out.append(c("! " + n, "33"))
    if not best.runs:
        out.append("")
        smallest = min(fits, key=lambda f: f.total_gb)
        out.append(f"Needs about {smallest.total_gb:.0f} GB at {smallest.quant} with this context. "
                   f"You have ~{gpu_budget_gb(machine) + (0 if machine.unified else cpu_budget_gb(machine)):.0f} GB usable.")
    return out


APPLE_TIP = (
    "macOS caps how much unified memory the GPU may use (about 75%). You can raise it until reboot:\n"
    "  sudo sysctl iogpu.wired_limit_mb={mb}\n"
    "That leaves ~8 GB for macOS. Quit other apps first."
)


def apple_tip(machine: Machine) -> str:
    mb = int(max(machine.ram_gb - 8, machine.ram_gb * 0.75) * 1000)
    return APPLE_TIP.format(mb=mb)


def overview(machine: Machine, catalog: list[ModelSpec], ctx: int, kv: str, min_speed: float,
             tag: str | None, show_all: bool, show_upgrades: bool) -> list[str]:
    models = [m for m in catalog if not tag or tag in m.tags]
    fits = [best_fit(m, machine, ctx, kv, min_speed) for m in models]
    out = header(machine, ctx, kv) + [""] + picks(fits) + [""] + table(fits, show_all)
    if any(f.placement == "unified-raise" for f in fits) and machine.apple_chip:
        out += ["", c(apple_tip(machine), "33")]
    if show_upgrades:
        out += [""] + upgrades(machine, fits, ctx, kv, min_speed, models)
    out += ["", dim("Estimates, +/-25%. Details for one model: willitrun -m <name>. "
                    "Any Hugging Face model: willitrun --hf org/repo")]
    return out


def to_json(machine: Machine, fits: list[Fit]) -> dict:
    return {
        "machine": {
            "description": machine.describe(), "os": machine.os, "cpu": machine.cpu,
            "ram_gb": round(machine.ram_gb, 1), "unified_memory": machine.unified,
            "gpu_usable_gb": round(gpu_budget_gb(machine), 1),
            "gpus": [{"name": g.name, "vram_gb": round(g.vram_gb, 1), "bandwidth_gbs": g.bw_gbs}
                     for g in machine.gpus],
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
    out = [bold(f"{'GPU / chip':<22}{'Memory':>8}{'Bandwidth':>12}")]
    for name, spec in sorted(hardware.GPU_TABLE.items(), key=lambda kv: -kv[1].bw_gbs):
        out.append(f"{name:<22}{spec.vram_gb:>5.0f} GB{spec.bw_gbs:>8.0f} GB/s")
    for chip, bw in hardware.APPLE_BW.items():
        out.append(f"{'apple ' + chip:<22}{'(RAM)':>8}{bw:>8.0f} GB/s")
    return out
