from willitrun.catalog import CATALOG, CATALOG_BY_ID, find_model
from willitrun.estimate import best_fit, estimate, kv_cache_gb, weights_gb
from willitrun.hardware import Gpu, Machine, apple_machine, lookup_gpu


def rig(gpu="RTX 4090", ram=64):
    spec = lookup_gpu(gpu)
    return Machine("Linux", "test", ram, 80, [Gpu(gpu, spec.vram_gb, spec.bw_gbs)])


def test_llama8b_kv_cache_matches_known_size():
    # 32 layers * 2 (K,V) * 8 heads * 128 dims * 2 bytes = 128 KiB per token -> 1 GiB at 8k
    assert kv_cache_gb(CATALOG_BY_ID["llama3.1-8b"], 8192) == 1.0


def test_sliding_window_keeps_gemma_cache_small():
    gemma = CATALOG_BY_ID["gemma3-27b"]
    full_attention = gemma.layers * 2 * gemma.kv_heads * gemma.head_dim * 2 * 131072 / 1e9
    assert kv_cache_gb(gemma, 131072) < full_attention / 4


def test_kv_quant_shrinks_cache():
    m = CATALOG_BY_ID["qwen3-32b"]
    assert kv_cache_gb(m, 32768, "q8_0") < kv_cache_gb(m, 32768, "f16") * 0.6


def test_native_size_used_for_gpt_oss():
    # the 63.4 GB (decimal) file is 59 GiB
    assert abs(weights_gb(CATALOG_BY_ID["gpt-oss-120b"], "MXFP4") - 59.05) < 0.01


def test_q4_weights_match_real_file_size():
    # bartowski's Llama 3.1 8B Q4_K_M file is 4.92 GB = 4.58 GiB
    assert 4.45 < weights_gb(CATALOG_BY_ID["llama3.1-8b"], "Q4_K_M") < 4.7


def test_8b_fits_on_24gb_card():
    f = best_fit(CATALOG_BY_ID["llama3.1-8b"], rig())
    assert f.placement == "gpu" and f.verdict == "great"


def test_70b_spills_to_ram_on_24gb_card():
    f = estimate(CATALOG_BY_ID["llama3.3-70b"], "Q4_K_M", 3, rig())
    assert f.placement == "split" and 0.3 < f.gpu_fraction < 0.6 and f.tok_s < 5


def test_moe_expert_offload_beats_plain_split():
    m = CATALOG_BY_ID["gpt-oss-120b"]
    f = estimate(m, "MXFP4", 4, rig())
    assert f.placement == "split"
    # 117B parameters, but only 5B are read per token and attention stays on
    # the GPU, so it should be far faster than a dense 70B on the same rig.
    dense = estimate(CATALOG_BY_ID["llama3.3-70b"], "Q4_K_M", 3, rig())
    assert f.tok_s > 5 * dense.tok_s


def test_nothing_fits_tiny_machine():
    f = best_fit(CATALOG_BY_ID["deepseek-r1-671b"], rig("RTX 3060", ram=32))
    assert not f.runs and f.verdict == "won't fit"


def test_long_context_costs_memory():
    m = CATALOG_BY_ID["qwen3-32b"]
    short = estimate(m, "Q4_K_M", 3, rig(), ctx=4096)
    long = estimate(m, "Q4_K_M", 3, rig(), ctx=32768)
    assert long.kv_gb > 7 * short.kv_gb
    assert short.placement == "gpu" and long.placement == "split"


def test_context_capped_at_model_max():
    f = estimate(CATALOG_BY_ID["phi4-14b"], "Q4_K_M", 3, rig(), ctx=100000)
    assert f.ctx == 16384 and any("capped" in n for n in f.notes)


def test_apple_limit_and_raise_hint():
    mac = apple_machine("M4 Max", 64)
    f = estimate(CATALOG_BY_ID["llama3.3-70b"], "Q4_K_M", 3, mac, ctx=32768)
    assert f.placement == "unified-raise"


def test_ultra_is_less_efficient_per_gb_s():
    m = CATALOG_BY_ID["llama3.1-8b"]
    ultra = estimate(m, "Q4_K_M", 3, apple_machine("M2 Ultra", 192))
    maxchip = estimate(m, "Q4_K_M", 3, apple_machine("M2 Max", 96))
    assert ultra.tok_s < 2 * maxchip.tok_s


def test_pick_prefers_full_gpu_over_bigger_quant_split():
    f = best_fit(CATALOG_BY_ID["qwen3-32b"], rig())
    assert f.placement == "gpu" and f.quant == "Q4_K_M"


def test_every_model_has_sane_numbers():
    for m in CATALOG:
        assert m.layers > 0 and m.params_b > 0 and m.max_ctx >= 8192
        if m.is_moe:
            assert m.shared_b is not None and m.shared_b < m.active


def test_find_model_fuzzy():
    assert find_model("gpt-oss-20b").id == "gpt-oss-20b"
    assert find_model("Qwen3 32B").id == "qwen3-32b"
    assert find_model("nonexistent") is None
    assert find_model("phi4") is None   # ambiguous: phi4-mini and phi4-14b
    assert find_model("phi4-14b").id == "phi4-14b"


def test_laptop_gpus_not_confused_with_desktop():
    assert lookup_gpu("NVIDIA GeForce RTX 4090 Laptop GPU").bw_gbs == 576
    assert lookup_gpu("NVIDIA GeForce RTX 4090").bw_gbs == 1008
    assert lookup_gpu("NVIDIA GeForce RTX 4070 Ti SUPER").vram_gb == 16


def test_gpu_names_match_whole_words():
    assert lookup_gpu("NVIDIA RTX A1000") is None          # not an A100
    assert lookup_gpu("NVIDIA L40").bw_gbs == 864           # not an L4
    assert lookup_gpu("NVIDIA H100 PCIe").bw_gbs == 2000
    assert lookup_gpu("NVIDIA GeForce RTX 4060 Ti", 8).bw_gbs == 288
    assert lookup_gpu("NVIDIA GeForce RTX 3080", 12).bw_gbs == 912
    assert lookup_gpu("Navi 31 [Radeon RX 7900 XT/7900 XTX]", 24).bw_gbs == 960
    assert lookup_gpu("NVIDIA GeForce RTX 3080 with Max-Q Design").bw_gbs == 448
    assert lookup_gpu("AMD RYZEN AI MAX+ 395 w/ Radeon 8060S").unified


def test_moe_on_cpu_only_uses_ram_bandwidth():
    cpu = Machine("Linux", "test", 64, 80, [])
    moe = estimate(CATALOG_BY_ID["qwen3-30b-a3b"], "Q4_K_M", 3, cpu)
    assert moe.placement == "cpu" and moe.tok_s < 25


def test_hypothetical_card_on_a_mac_uses_normal_ram_speed():
    from willitrun.hardware import manual, DEFAULT_RAM_BW
    mac = apple_machine("M4 Max", 64)
    m = manual("RTX 3090", 64, None, None, None, base=mac)
    assert not m.unified and m.apple_chip is None and m.ram_bw_gbs == DEFAULT_RAM_BW
    cpu = manual("none", None, None, None, None, base=apple_machine("M4 Max", 64))
    assert cpu.ram_bw_gbs == DEFAULT_RAM_BW and not cpu.gpus


def test_unknown_gpu_needs_vram():
    import pytest
    from willitrun.hardware import HardwareError, manual
    with pytest.raises(HardwareError):
        manual("RTX 9999", 64, None, None, None, base=rig())
    m = manual("RTX 9999", 64, 20, 700, None, base=rig())
    assert m.vram_gb == 20 and m.gpu_bw_gbs == 700


def test_36gb_macs_get_the_slower_bin():
    assert apple_machine("M4 Max", 36).ram_bw_gbs == 410
    assert apple_machine("M4 Max", 48).ram_bw_gbs == 546
