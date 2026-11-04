"""Hardware detection and a lookup table of memory bandwidth.

Decode speed for a local LLM is limited mostly by memory bandwidth, not
compute: every generated token streams the active weights through the chip
once. So we need two numbers per memory pool: how big it is and how fast it is.
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field, replace

# Memory sizes are in GiB throughout, because that is what "24 GB" on a GPU
# box or "36 GB" on a Mac really means. Bandwidth stays in decimal GB/s.
GB = 2 ** 30


@dataclass(frozen=True)
class GpuSpec:
    vram_gb: float
    bw_gbs: float
    unified: bool = False


# name fragment (lowercase) -> spec. Longer fragments are matched first.
GPU_TABLE: dict[str, GpuSpec] = {
    # NVIDIA GeForce, desktop
    "rtx 5090": GpuSpec(32, 1792), "rtx 5080": GpuSpec(16, 960),
    "rtx 5070 ti": GpuSpec(16, 896), "rtx 5070": GpuSpec(12, 672),
    "rtx 5060 ti": GpuSpec(16, 448), "rtx 5060": GpuSpec(8, 448),
    "rtx 4090": GpuSpec(24, 1008), "rtx 4080 super": GpuSpec(16, 736),
    "rtx 4080": GpuSpec(16, 717), "rtx 4070 ti super": GpuSpec(16, 672),
    "rtx 4070 ti": GpuSpec(12, 504), "rtx 4070 super": GpuSpec(12, 504),
    "rtx 4070": GpuSpec(12, 504), "rtx 4060 ti": GpuSpec(16, 288),
    "rtx 4060": GpuSpec(8, 272), "rtx 3090 ti": GpuSpec(24, 1008),
    "rtx 3090": GpuSpec(24, 936), "rtx 3080 ti": GpuSpec(12, 912),
    "rtx 3080": GpuSpec(10, 760), "rtx 3080 12gb": GpuSpec(12, 912), "rtx 3070 ti": GpuSpec(8, 608),
    "rtx 3070": GpuSpec(8, 448), "rtx 3060 ti": GpuSpec(8, 448),
    "rtx 3060": GpuSpec(12, 360), "rtx 2080 ti": GpuSpec(11, 616),
    # NVIDIA laptop parts share names with desktop cards but are much slower
    "rtx 5090 laptop": GpuSpec(24, 896), "rtx 5080 laptop": GpuSpec(16, 896),
    "rtx 5070 ti laptop": GpuSpec(12, 672), "rtx 5070 laptop": GpuSpec(8, 384),
    "rtx 5060 laptop": GpuSpec(8, 384),
    "rtx 4090 laptop": GpuSpec(16, 576), "rtx 4080 laptop": GpuSpec(12, 432),
    "rtx 4070 laptop": GpuSpec(8, 256), "rtx 4060 laptop": GpuSpec(8, 256),
    "rtx 4050 laptop": GpuSpec(6, 192), "rtx 3080 laptop": GpuSpec(16, 448),
    "rtx 3070 laptop": GpuSpec(8, 448), "rtx 3060 laptop": GpuSpec(6, 336),
    # NVIDIA workstation / datacenter
    "rtx pro 6000": GpuSpec(96, 1792), "rtx 6000 ada": GpuSpec(48, 960),
    "rtx a6000": GpuSpec(48, 768), "a100 80gb": GpuSpec(80, 2039),
    "a100": GpuSpec(40, 1555), "h100": GpuSpec(80, 3350), "h100 pcie": GpuSpec(80, 2000),
    "l40s": GpuSpec(48, 864), "l40": GpuSpec(48, 864),
    "a10": GpuSpec(24, 600), "l4": GpuSpec(24, 300), "t4": GpuSpec(16, 320),
    "v100": GpuSpec(16, 900), "p40": GpuSpec(24, 346),
    "gb10": GpuSpec(128, 273, unified=True), "dgx spark": GpuSpec(128, 273, unified=True),
    # AMD Radeon
    "rx 9070 xt": GpuSpec(16, 640), "rx 9070": GpuSpec(16, 640),
    "rx 7900 xtx": GpuSpec(24, 960), "rx 7900 xt": GpuSpec(20, 800),
    "rx 7900 gre": GpuSpec(16, 576), "rx 7800 xt": GpuSpec(16, 624),
    "rx 7700 xt": GpuSpec(12, 432), "rx 7600": GpuSpec(8, 288),
    "rx 6900 xt": GpuSpec(16, 512), "rx 6800 xt": GpuSpec(16, 512),
    "pro w7900": GpuSpec(48, 864), "mi300x": GpuSpec(192, 5300),
    "ryzen ai max": GpuSpec(128, 256, unified=True),
    # Intel Arc
    "arc b580": GpuSpec(12, 456), "arc a770": GpuSpec(16, 560),
}

# Apple Silicon unified memory bandwidth, GB/s. Some chips come in two
# bandwidth bins; the RAM size tells them apart (see apple_bandwidth).
APPLE_BW = {
    "m1": 68, "m1 pro": 200, "m1 max": 400, "m1 ultra": 800,
    "m2": 100, "m2 pro": 200, "m2 max": 400, "m2 ultra": 800,
    "m3": 100, "m3 pro": 150, "m3 max": 400, "m3 ultra": 819,
    "m4": 120, "m4 pro": 273, "m4 max": 546,
    "m5": 153,
}

# Largest memory each chip was sold with, so the upgrade advisor only
# suggests Macs that exist.
APPLE_MAX_RAM = {
    "m1": 16, "m1 pro": 32, "m1 max": 64, "m1 ultra": 128,
    "m2": 24, "m2 pro": 32, "m2 max": 96, "m2 ultra": 192,
    "m3": 24, "m3 pro": 36, "m3 max": 128, "m3 ultra": 512,
    "m4": 32, "m4 pro": 64, "m4 max": 128, "m5": 32,
}


def apple_bandwidth(chip: str, ram_gb: float) -> float | None:
    key = chip.lower().replace("apple", "").strip()
    if key == "m3 max" and ram_gb <= 36:
        return 300  # 14-core M3 Max
    if key == "m4 max" and ram_gb <= 36:
        return 410  # 14-core M4 Max
    return APPLE_BW.get(key)


# "rtx 4060" must not match "RTX 4060 Ti", nor "rx 7900 xt" match "RX 7900 XTX".
SUFFIX_GUARD = r"(?![a-z0-9])(?!\s+(?:ti|super|xt|xtx|gre)\b)"
LAPTOP_MARKERS = re.compile(r"\b(laptop gpu|laptop|mobile|with max-q design|max-q|max-p)\b")


def lookup_gpu(name: str, vram_gb: float | None = None) -> GpuSpec | None:
    """Find a GPU by name. Matches whole words, so "A1000" is not an "A100".

    When several entries match (rocm-smi says "RX 7900 XT/7900 XTX", or the
    3080 that came with 10 or 12 GB), the detected VRAM picks the right one.
    """
    n = name.lower()
    laptop = bool(LAPTOP_MARKERS.search(n))
    n = LAPTOP_MARKERS.sub(" ", n)
    n = re.sub(r"\b(geforce|nvidia|amd|radeon|graphics|\(r\)|\(tm\))\b", " ", n)
    n = re.sub(r"[\s/\[\]]+", " ", n).strip() + (" laptop" if laptop else "")
    hits = []
    for frag in sorted(GPU_TABLE, key=len, reverse=True):
        if "laptop" in frag and not laptop:
            continue
        words = frag.split()
        forms = [frag]
        if len(words) > 1 and words[0] in ("rtx", "rx", "arc") and words[0] in n.split():
            forms.append(" ".join(words[1:]))   # "RX 7900 XT/7900 XTX" names two cards
        if any(re.search(r"(?<![a-z0-9])" + re.escape(f) + SUFFIX_GUARD, n) for f in forms):
            hits.append(frag)
    if not hits:
        return None
    if laptop and any("laptop" in h for h in hits):
        hits = [h for h in hits if "laptop" in h]   # a laptop part with its own entry
    if vram_gb:  # same name, more memory: "rtx 3080" -> "rtx 3080 12gb"
        hits += [f"{h} {round(vram_gb)}gb" for h in hits if f"{h} {round(vram_gb)}gb" in GPU_TABLE]
    if vram_gb and len(hits) > 1:
        return GPU_TABLE[min(hits, key=lambda f: (abs(GPU_TABLE[f].vram_gb - vram_gb), -len(f)))]
    return GPU_TABLE[hits[0]]


@dataclass
class Gpu:
    name: str
    vram_gb: float
    bw_gbs: float | None
    bw_known: bool = True


@dataclass
class Machine:
    os: str
    cpu: str
    ram_gb: float
    ram_bw_gbs: float
    gpus: list[Gpu] = field(default_factory=list)
    unified: bool = False          # Apple Silicon, Strix Halo, DGX Spark
    unified_gpu_limit_gb: float | None = None
    apple_chip: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def vram_gb(self) -> float:
        return sum(g.vram_gb for g in self.gpus)

    @property
    def gpu_bw_gbs(self) -> float | None:
        # Layers split across GPUs run one after another, so the effective
        # bandwidth is the slowest card's.
        bws = [g.bw_gbs for g in self.gpus if g.bw_gbs]
        return min(bws) if bws else None

    def describe(self) -> str:
        if self.unified:
            return f"{self.cpu}, {self.ram_gb:.0f} GB unified memory"
        parts = [self.cpu, f"{self.ram_gb:.0f} GB RAM"]
        for g in self.gpus:
            parts.append(f"{g.name} ({g.vram_gb:.0f} GB)")
        return ", ".join(parts)

    def to_cpu_only(self) -> None:
        self.gpus, self.unified, self.unified_gpu_limit_gb, self.apple_chip = [], False, None, None
        self.ram_bw_gbs = DEFAULT_RAM_BW


def _run(cmd: list[str], timeout: float = 5) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _ram_gb() -> float:
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            return int(out.strip()) / GB
    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024 / GB
        except OSError:
            pass
    if system == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return stat.ullTotalPhys / GB
    return 16.0


def _cpu_name() -> str:
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except (ImportError, OSError):
            pass
    return platform.processor() or platform.machine() or "Unknown CPU"


def _nvidia_gpus() -> list[Gpu]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    gpus: list[Gpu] = []
    for line in (out or "").strip().splitlines():
        name, _, mem = line.rpartition(",")
        name = name.strip()
        try:
            vram = float(mem.strip()) / 1024  # MiB -> GiB
        except ValueError:
            vram = None  # "[N/A]" on some unified-memory parts
        spec = lookup_gpu(name, vram)
        if vram is None:
            vram = spec.vram_gb if spec else 0.0
        gpus.append(Gpu(name, vram, spec.bw_gbs if spec else None, spec is not None))
    return gpus


def _amd_gpus_linux() -> list[Gpu]:
    gpus: list[Gpu] = []
    names: list[str] = []
    out = _run(["rocm-smi", "--showproductname", "--json"])
    if out:
        try:
            for card in json.loads(out).values():
                names.append(card.get("Card Series") or card.get("Card series") or "AMD GPU")
        except (ValueError, AttributeError):
            pass
    for i, path in enumerate(sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"))):
        try:
            with open(path) as f:
                vram = int(f.read().strip()) / GB
        except (OSError, ValueError):
            continue
        if vram < 2:  # integrated GPU carve-out, not a real card
            continue
        name = names[i] if i < len(names) else "AMD GPU"
        spec = lookup_gpu(name, vram)
        gpus.append(Gpu(name, vram, spec.bw_gbs if spec else None, spec is not None))
    return gpus


def apple_gpu_limit(ram_gb: float) -> float:
    """macOS lets the GPU wire about 2/3 of RAM (3/4 above 36 GB) by default."""
    return ram_gb * (0.75 if ram_gb > 36.5 else 0.67)


DEFAULT_RAM_BW = 80.0   # dual-channel DDR5-5000ish; DDR4 desktops are closer to 50
UNIFIED_SHARE = 0.75    # Strix Halo: GPU share of RAM, set in BIOS / driver
SPARK_SHARE = 0.9       # DGX Spark's GPU can use nearly all of its memory


def unified_share(name: str) -> float:
    n = name.lower()
    return SPARK_SHARE if ("gb10" in n or "spark" in n) else UNIFIED_SHARE


def _ram_speed() -> tuple[float, str] | None:
    """Peak RAM bandwidth from DIMM speed, when the OS will tell us.

    Consumer boards run two 64-bit channels, so we count at most two DIMMs.
    """
    system = platform.system()
    speeds: list[int] = []
    kind = ""
    if system == "Linux":
        out = _run(["dmidecode", "-t", "17"])  # needs root; silently skipped otherwise
        for line in (out or "").splitlines():
            line = line.strip()
            m = re.match(r"Configured (?:Memory|Clock) Speed: (\d+) MT/s", line)
            if m:
                speeds.append(int(m.group(1)))
            m = re.match(r"Type: (DDR\d|LPDDR\d\w?)", line)
            if m:
                kind = m.group(1)
    elif system == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_PhysicalMemory | ForEach-Object { $_.ConfiguredClockSpeed }"])
        speeds = [int(x) for x in (out or "").split() if x.isdigit() and int(x) > 0]
    if not speeds:
        return None
    channels = min(len(speeds), 2)
    return min(speeds) * 8 * channels / 1000, kind


def detect() -> Machine:
    system = platform.system()
    ram = _ram_gb()
    cpu = _cpu_name()
    m = Machine(os=system, cpu=cpu, ram_gb=ram, ram_bw_gbs=DEFAULT_RAM_BW)

    chip = re.search(r"Apple (M\d(?: (?:Pro|Max|Ultra))?)", cpu)
    if system == "Darwin" and chip:
        return apple_machine(chip.group(1), ram, m)

    unified = lookup_gpu(cpu)
    if unified and unified.unified:  # e.g. Ryzen AI Max (Strix Halo)
        m.unified = True
        m.ram_bw_gbs = unified.bw_gbs
        m.unified_gpu_limit_gb = ram * UNIFIED_SHARE
        m.gpus = [Gpu(f"{cpu} iGPU", 0, unified.bw_gbs)]
        m.notes.append("Unified memory: the GPU share is set in BIOS/driver; assumed 75% of RAM.")
        return m

    m.gpus = _nvidia_gpus()
    if any(lookup_gpu(g.name) and lookup_gpu(g.name).unified for g in m.gpus):  # DGX Spark
        spec = lookup_gpu(m.gpus[0].name)
        m.unified = True
        m.ram_bw_gbs = spec.bw_gbs
        m.unified_gpu_limit_gb = ram * SPARK_SHARE
        m.gpus = [Gpu(m.gpus[0].name, 0, spec.bw_gbs)]
        return m
    if not m.gpus and system == "Linux":
        m.gpus = _amd_gpus_linux()
    speed = _ram_speed()
    if speed:
        m.ram_bw_gbs = speed[0]
    else:
        m.notes.append(f"RAM speed unknown, assumed {DEFAULT_RAM_BW:.0f} GB/s (DDR5). "
                       "For DDR4 add --ram-bw 50.")
    for g in m.gpus:
        if not g.bw_known:
            g.bw_gbs = 400.0
            m.notes.append(f"Unknown bandwidth for '{g.name}', assumed 400 GB/s. Pass --gpu-bw to fix.")
    return m


def apple_machine(chip: str, ram: float, base: Machine | None = None) -> Machine:
    bw = apple_bandwidth(chip, ram)
    m = base or Machine(os="Darwin", cpu=f"Apple {chip}", ram_gb=ram, ram_bw_gbs=bw or 100.0)
    if bw is None:
        bw = 100.0 * (4 if "max" in chip.lower() else 2 if "pro" in chip.lower() else 8
                      if "ultra" in chip.lower() else 1.5)
        m.notes.append(f"Bandwidth for the {chip} isn't in the table yet, assumed {bw:.0f} GB/s. "
                       "Pass --gpu-bw to fix.")
    m.cpu = f"Apple {chip}"
    m.unified = True
    m.apple_chip = chip
    m.ram_bw_gbs = bw
    m.unified_gpu_limit_gb = apple_gpu_limit(ram)
    m.gpus = [Gpu(f"Apple {chip} GPU", 0, bw)]
    return m


class HardwareError(ValueError):
    """A flag combination that can't describe a real machine."""


def manual(gpu: str | None, ram_gb: float | None, vram_gb: float | None,
           gpu_bw: float | None, ram_bw: float | None, gpu_count: int = 1,
           base: Machine | None = None) -> Machine:
    """Build a hypothetical (or corrected) machine from CLI flags."""
    m = base or detect()
    if ram_gb:
        m.ram_gb = ram_gb
    if gpu:
        if gpu.lower() in ("none", "cpu"):
            if vram_gb or gpu_count != 1:
                raise HardwareError("--vram and --gpu-count don't apply with --gpu none.")
            m.to_cpu_only()
            m.notes = [] if ram_bw else [f"RAM speed assumed {DEFAULT_RAM_BW:.0f} GB/s (DDR5). "
                                         "For DDR4 add --ram-bw 50."]
        else:
            chip = re.fullmatch(r"(?:apple )?(m\d(?: (?:pro|max|ultra))?)", gpu.strip().lower())
            spec = None if chip else lookup_gpu(gpu, vram_gb)
            if chip or (spec and spec.unified):
                if gpu_count != 1:
                    raise HardwareError("--gpu-count only applies to separate graphics cards.")
                if vram_gb:
                    raise HardwareError("Unified-memory machines have no separate VRAM. Use --ram.")
            if chip:
                words = chip.group(1).split()
                name = " ".join([words[0].upper()] + [w.capitalize() for w in words[1:]])
                m.notes = []
                m = apple_machine(name, m.ram_gb, m)
            elif spec and spec.unified:
                m.to_cpu_only()
                m.notes = []
                m.unified = True
                m.ram_bw_gbs = spec.bw_gbs
                if not ram_gb:
                    m.ram_gb = spec.vram_gb
                m.unified_gpu_limit_gb = m.ram_gb * unified_share(gpu)
                m.gpus = [Gpu(gpu, 0, spec.bw_gbs)]
                m.cpu = gpu
            else:
                if not spec and not vram_gb:
                    raise HardwareError(f"'{gpu}' isn't in the GPU table (see --list-gpus). "
                                        "Add --vram and --gpu-bw for it.")
                if m.unified or m.apple_chip:
                    # A Mac or APU with a hypothetical card: RAM is ordinary system RAM now.
                    cpu = m.cpu
                    m.to_cpu_only()
                    m.cpu = cpu
                    m.notes = []
                vram = vram_gb or spec.vram_gb
                bw = gpu_bw or (spec.bw_gbs if spec else 400.0)
                m.gpus = [Gpu(gpu, vram, bw, spec is not None) for _ in range(gpu_count)]
                m.notes = [n for n in m.notes if "Unknown bandwidth" not in n]
                if not spec and not gpu_bw:
                    m.notes.append(f"Unknown GPU '{gpu}', assumed 400 GB/s. Pass --gpu-bw to fix.")
    else:
        if gpu_count != 1 and m.gpus and not m.unified:
            m.gpus = [replace(m.gpus[0]) for _ in range(gpu_count)]
        elif gpu_count != 1:
            raise HardwareError("--gpu-count needs a graphics card. Add --gpu \"<name>\".")
        if vram_gb and (not m.gpus or m.unified):
            raise HardwareError("--vram needs a graphics card. Add --gpu \"<name>\" as well.")
    if vram_gb and m.gpus and not m.unified:
        for g in m.gpus:
            g.vram_gb = vram_gb
    if gpu_bw:
        for g in m.gpus:
            g.bw_gbs = gpu_bw
            g.bw_known = True
        if m.unified:
            m.ram_bw_gbs = gpu_bw
        m.notes = [n for n in m.notes if "assumed" not in n or "RAM speed" in n]
    if ram_bw and not m.unified:
        m.ram_bw_gbs = ram_bw
    if ram_bw or m.unified:
        m.notes = [n for n in m.notes if not n.startswith("RAM speed")]
    if m.unified and m.apple_chip and ram_gb:
        m.unified_gpu_limit_gb = apple_gpu_limit(m.ram_gb)
        bw = apple_bandwidth(m.apple_chip, m.ram_gb)
        if bw and not gpu_bw:
            m.ram_bw_gbs = bw
            for g in m.gpus:
                g.bw_gbs = bw
    elif m.unified and ram_gb:
        m.unified_gpu_limit_gb = m.ram_gb * unified_share(m.cpu)
    return m


def env_overrides() -> dict[str, float]:
    """WILLITRUN_GPU_BW etc. let people pin corrections without retyping flags."""
    out = {}
    for key in ("GPU_BW", "RAM_BW"):
        val = os.environ.get(f"WILLITRUN_{key}")
        if val:
            try:
                out[key.lower()] = float(val)
            except ValueError:
                pass
    return out
