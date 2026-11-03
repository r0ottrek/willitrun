"""Speed estimates checked against published llama.cpp token-generation numbers.

Reference figures are typical community results (llama.cpp `llama-bench` tg128
threads and model release write-ups). Hardware varies, so each check allows
+/-30%. If you have a machine in this list and a very different number, please
open an issue with your `llama-bench` output.
"""

import pytest

from willitrun.catalog import CATALOG_BY_ID
from willitrun.estimate import estimate
from willitrun.hardware import Gpu, Machine, apple_machine, lookup_gpu


def nvidia(name, count=1, ram=64):
    spec = lookup_gpu(name)
    return Machine("Linux", "test", ram, 80, [Gpu(name, spec.vram_gb, spec.bw_gbs) for _ in range(count)])


def cpu_only(ram=64, bw=80):
    return Machine("Linux", "test", ram, bw, [])


CASES = [
    # (machine, model id, quant, reference tok/s)
    (lambda: nvidia("RTX 4090"), "llama3.1-8b", "Q4_K_M", 130),
    (lambda: nvidia("RTX 3090"), "llama3.1-8b", "Q4_K_M", 112),
    (lambda: nvidia("RTX 4090"), "qwen3-30b-a3b", "Q4_K_M", 175),
    (lambda: nvidia("RTX 3090", count=2), "llama3.3-70b", "Q4_K_M", 17),
    (lambda: apple_machine("M2 Ultra", 192), "llama3.1-8b", "Q4_K_M", 80),
    (lambda: apple_machine("M1 Max", 64), "llama3.1-8b", "Q4_K_M", 50),
    (lambda: apple_machine("M3 Max", 128), "llama3.3-70b", "Q4_K_M", 7),
    (lambda: cpu_only(), "llama3.1-8b", "Q4_K_M", 11),
]


@pytest.mark.parametrize("machine,model,quant,ref", CASES)
def test_speed_close_to_benchmarks(machine, model, quant, ref):
    f = estimate(CATALOG_BY_ID[model], quant, 3, machine(), ctx=4096)
    assert f.runs
    assert ref * 0.7 <= f.tok_s <= ref * 1.3, f"{model} {quant}: estimated {f.tok_s:.1f}, reference {ref}"
