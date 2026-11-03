from willitrun.catalog import CATALOG, CATALOG_BY_ID, find_model
from willitrun.estimate import best_fit, estimate, kv_cache_gb, weights_gb
from willitrun.hardware import Gpu, Machine, apple_machine, lookup_gpu


def rig(gpu="RTX 4090", ram=64):
    spec = lookup_gpu(gpu)
    return Machine("Linux", "test", ram, 80, [Gpu(gpu, spec.vram_gb, spec.bw_gbs)])


def test_llama8b_kv_cache_matches_known_size():
    # 32 layers * 2 (K,V) * 8 heads * 128 dims * 2 bytes = 128 KiB per token
    assert abs(kv_cache_gb(CATALOG_BY_ID["llama3.1-8b"], 8192) - 1.07) < 0.01


def test_sliding_window_keeps_gemma_cache_small():
    gemma = CATALOG_BY_ID["gemma3-27b"]
    full_attention = gemma.layers * 2 * gemma.kv_heads * gemma.head_dim * 2 * 131072 / 1e9
    assert kv_cache_gb(gemma, 131072) < full_attention / 4


def test_kv_quant_shrinks_cache():
    m = CATALOG_BY_ID["qwen3-32b"]
    assert kv_cache_gb(m, 32768, "q8_0") < kv_cache_gb(m, 32768, "f16") * 0.6


def test_native_size_used_for_gpt_oss():
    assert weights_gb(CATALOG_BY_ID["gpt-oss-120b"], "MXFP4") == 63.4


def test_q4_weights_are_about_60_percent_of_params():
    assert 4.7 < weights_gb(CATALOG_BY_ID["llama3.1-8b"], "Q4_K_M") < 5.0


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
    f = estimate(CATALOG_BY_ID["llama3.3-70b"], "Q4_K_M", 3, mac)
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


def test_laptop_gpus_not_confused_with_desktop():
    assert lookup_gpu("NVIDIA GeForce RTX 4090 Laptop GPU").bw_gbs == 576
    assert lookup_gpu("NVIDIA GeForce RTX 4090").bw_gbs == 1008
    assert lookup_gpu("NVIDIA GeForce RTX 4070 Ti SUPER").vram_gb == 16
