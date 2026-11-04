"""Memory and speed estimates.

Memory = weights + KV cache (grows with context) + runtime buffers.
Speed  = memory bandwidth / bytes read per generated token.

Both are first-principles estimates, tuned against published llama.cpp
benchmarks (see tests/test_calibration.py). Expect +/- 25% on speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .catalog import QUANTS, QUANT_BY_NAME, RECOMMENDED_MIN_RANK, ModelSpec, Quant
from .hardware import Machine

# Sizes are GiB (what "24 GB" on a card means); bandwidth is decimal GB/s.
GIB_BYTES = 2 ** 30
GIB = GIB_BYTES / 1e9

KV_BYTES = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}

# Calibration constants (fitted to llama.cpp token-generation benchmarks).
EFF_GPU = 0.70          # discrete GPU: share of peak bandwidth actually achieved
EFF_UNIFIED = 0.65      # Apple Silicon / Strix Halo / DGX Spark
EFF_ULTRA = 0.50        # two-die Apple Ultra chips scale worse
EFF_CPU = 0.65
MOE_PENALTY = 0.75      # expert routing and scattered reads cost extra
PER_TOKEN_MS_GPU = 1.0  # kernel launch / sampling overhead per token
PER_TOKEN_MS_CPU = 2.0
GPU_RESERVE_GB = 0.7    # driver context + display, per GPU
SPEED_AT_TOKENS = 1024  # table speeds assume a short conversation

GREAT, GOOD, USABLE = 30.0, 15.0, 8.0


@dataclass
class Fit:
    model: ModelSpec
    quant: str
    rank: int
    ctx: int
    weights_gb: float
    kv_gb: float
    overhead_gb: float
    placement: str            # gpu | split | cpu | unified | unified-raise | none
    gpu_fraction: float       # share of layers on the GPU
    gpu_layers: int
    tok_s: float
    tok_s_full_ctx: float
    notes: list[str] = field(default_factory=list)

    @property
    def total_gb(self) -> float:
        return self.weights_gb + self.kv_gb + self.overhead_gb

    @property
    def runs(self) -> bool:
        return self.placement != "none"

    @property
    def verdict(self) -> str:
        if not self.runs:
            return "won't fit"
        if self.tok_s >= GREAT:
            return "great"
        if self.tok_s >= GOOD:
            return "good"
        if self.tok_s >= USABLE:
            return "usable"
        return "slow"

    @property
    def where(self) -> str:
        return {
            "gpu": "on GPU",
            "unified": "unified memory",
            "unified-raise": "raise GPU limit",
            "split": f"{round(self.gpu_fraction * 100)}% on GPU",
            "cpu": "on CPU",
            "none": "won't fit",
        }[self.placement]


def kv_cache_gb(m: ModelSpec, ctx: int, kv_type: str = "f16") -> float:
    per_elem = KV_BYTES[kv_type]
    if m.kv_bytes_per_token:
        return ctx * m.kv_bytes_per_token / 2.0 * per_elem / GIB_BYTES
    per_layer_token = 2 * m.kv_heads * m.head_dim * per_elem  # K and V
    if m.window:
        win, frac = m.window
        local = m.layers * frac
        glob = m.layers - local
        tokens = glob * ctx + local * min(ctx, win)
    else:
        tokens = m.layers * ctx
    return tokens * per_layer_token / GIB_BYTES


def weights_gb(m: ModelSpec, q: Quant | str) -> float:
    name = q if isinstance(q, str) else q.name
    if name in m.native:
        return m.native[name] * 1e9 / GIB_BYTES   # file sizes are listed in decimal GB
    quant = QUANT_BY_NAME[name]
    return m.params_b * 1e9 * quant.bpw / 8 / GIB_BYTES


def quants_for(m: ModelSpec) -> list[tuple[str, int]]:
    if m.native:
        return [(name, 4) for name in m.native]   # native MXFP4 ~ Q5 quality
    return [(q.name, q.rank) for q in QUANTS if q.name != "F16" or m.params_b <= 4]


def _eff(machine: Machine) -> float:
    if machine.unified:
        chip = (machine.apple_chip or "").lower()
        return EFF_ULTRA if "ultra" in chip else EFF_UNIFIED
    return EFF_GPU


def _ms_per_token(gpu_gb: float, cpu_gb: float, machine: Machine, moe: bool) -> float:
    penalty = MOE_PENALTY if moe else 1.0
    ms = 0.0
    if gpu_gb > 0:
        bw = machine.ram_bw_gbs if machine.unified else (machine.gpu_bw_gbs or 400)
        ms += gpu_gb * GIB / (bw * _eff(machine) * penalty) * 1000 + PER_TOKEN_MS_GPU
    if cpu_gb > 0:
        ms += cpu_gb * GIB / (machine.ram_bw_gbs * EFF_CPU * penalty) * 1000 + PER_TOKEN_MS_CPU
    return ms


def cpu_budget_gb(machine: Machine) -> float:
    return machine.ram_gb - max(4.0, 0.15 * machine.ram_gb)


def gpu_budget_gb(machine: Machine) -> float:
    if machine.unified:
        return (machine.unified_gpu_limit_gb or machine.ram_gb * 0.7) - 1.0
    return max(0.0, sum(max(0.0, g.vram_gb - GPU_RESERVE_GB) for g in machine.gpus))


def estimate(m: ModelSpec, quant: str, rank: int, machine: Machine,
             ctx: int = 8192, kv_type: str = "f16") -> Fit:
    notes: list[str] = []
    if ctx > m.max_ctx:
        notes.append(f"context capped at the model's max of {m.max_ctx:,}")
        ctx = m.max_ctx
    w = weights_gb(m, quant)
    kv = kv_cache_gb(m, ctx, kv_type)
    overhead = 0.5 + 0.03 * w
    total = w + kv + overhead
    bpp = w / m.params_b                       # GB per billion params at this quant
    kv_read_short = kv_cache_gb(m, min(ctx, SPEED_AT_TOKENS), kv_type)
    moe_split = m.is_moe and m.shared_b is not None
    shared_w = (m.shared_b or 0) * bpp
    expert_w = w - shared_w                    # all routed experts
    expert_active = (m.active - (m.shared_b or 0)) * bpp

    def speed(frac_gpu: float, kv_read: float) -> float:
        if moe_split and 0 < frac_gpu < 1:
            # Expert offload: attention + shared weights + KV stay on the GPU,
            # only the routed experts of some layers live in system RAM.
            gpu = shared_w + kv_read + expert_active * frac_gpu
            cpu = expert_active * (1 - frac_gpu)
        else:
            per_tok = w * m.active / m.params_b + kv_read
            gpu, cpu = per_tok * frac_gpu, per_tok * (1 - frac_gpu)
        return 1000 / _ms_per_token(gpu, cpu, machine, m.is_moe)

    gpu_cap = gpu_budget_gb(machine)
    cpu_cap = cpu_budget_gb(machine)
    has_gpu = bool(machine.gpus)

    def fit(placement: str, frac: float) -> Fit:
        return Fit(m, quant, rank, ctx, w, kv, overhead, placement, frac,
                   math.floor(frac * m.layers), speed(frac, kv_read_short), speed(frac, kv), notes)

    if machine.unified:
        if total <= gpu_cap:
            return fit("unified", 1.0)
        if total <= machine.ram_gb - 6:
            if machine.apple_chip:
                notes.append("raise the macOS GPU memory limit first (see --help-apple)")
            else:
                notes.append("needs a bigger GPU memory share in BIOS / driver settings")
            return fit("unified-raise", 1.0)
        return fit("none", 0.0)

    if has_gpu and total <= gpu_cap:
        return fit("gpu", 1.0)
    if has_gpu and gpu_cap > overhead + 1:
        if moe_split and gpu_cap > overhead + kv + shared_w + 0.5:
            frac = (gpu_cap - overhead - kv - shared_w) / expert_w
            cpu_need = (1 - frac) * expert_w
        else:
            frac = (gpu_cap - overhead) / (w + kv)
            cpu_need = (1 - frac) * (w + kv)
        frac = max(0.0, min(1.0, frac))
        if cpu_need <= cpu_cap:
            return fit("split", frac)
        return fit("none", 0.0)
    if total <= cpu_cap:
        return fit("cpu", 0.0)
    return fit("none", 0.0)


def all_fits(m: ModelSpec, machine: Machine, ctx: int, kv_type: str) -> list[Fit]:
    return [estimate(m, q, r, machine, ctx, kv_type) for q, r in quants_for(m)]


def best_fit(m: ModelSpec, machine: Machine, ctx: int = 8192, kv_type: str = "f16",
             min_speed: float = USABLE) -> Fit:
    """Pick the setup most people would actually want.

    1. Fully on GPU at Q4 or better and fast enough: take the best quality.
    2. Otherwise any placement at Q3 or better that is fast enough.
    3. Otherwise whatever runs fastest (it will be marked slow).
    """
    fits = all_fits(m, machine, ctx, kv_type)
    runnable = [f for f in fits if f.runs]
    if not runnable:
        return min(fits, key=lambda f: f.total_gb)
    full = [f for f in runnable if f.placement in ("gpu", "unified")
            and RECOMMENDED_MIN_RANK <= f.rank <= 5 and f.tok_s >= min_speed]
    if full:
        # Start from Q4_K_M and step up in quality only while it stays fast:
        # within 75% of Q4's speed, or above 'great' regardless.
        base = min(full, key=lambda f: f.rank)
        floor = min(GREAT, 0.75 * base.tok_s)
        return max((f for f in full if f.tok_s >= floor), key=lambda f: f.rank)
    # Partial offload or a raised memory limit: Q4_K_M is the sensible
    # default; extra bits would cost speed we can't spare.
    ok = [f for f in runnable if 2 <= f.rank <= RECOMMENDED_MIN_RANK and f.tok_s >= min_speed]
    if ok:
        return max(ok, key=lambda f: (f.placement in ("gpu", "unified", "unified-raise"), f.rank))
    # Nothing is fast enough: show the sensible default (Q4_K_M, or the
    # closest quant that fits) so the verdict reflects a setup worth running.
    sane = [f for f in runnable if f.rank <= RECOMMENDED_MIN_RANK] or runnable
    return max(sane, key=lambda f: f.rank)


def quality_score(f: Fit) -> float:
    """Ranking heuristic for 'best picks': bigger, newer, less-quantized wins."""
    qmul = {1: 0.70, 2: 0.85, 3: 0.95, 4: 0.98}.get(f.rank, 1.0)
    recency = 1.0 if f.model.year >= 2025 else 0.8
    return math.log2(1 + f.model.effective_b) * qmul * recency


def min_memory_gb(m: ModelSpec, quant: str, ctx: int, kv_type: str = "f16") -> float:
    w = weights_gb(m, quant)
    kv = kv_cache_gb(m, min(ctx, m.max_ctx), kv_type)
    return w + kv + 0.5 + 0.03 * w
