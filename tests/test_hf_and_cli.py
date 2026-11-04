import json

import os

import pytest

from willitrun import report
from willitrun.cli import main
from willitrun.hf import spec_from_config

QWEN3_30B_A3B = {
    "hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 32,
    "num_key_value_heads": 4, "head_dim": 128, "max_position_embeddings": 40960,
    "vocab_size": 151936, "intermediate_size": 6144, "num_experts": 128,
    "num_experts_per_tok": 8, "moe_intermediate_size": 768,
}

GEMMA3_TEXT = {"text_config": {
    "hidden_size": 5376, "num_hidden_layers": 62, "num_attention_heads": 32,
    "num_key_value_heads": 16, "head_dim": 128, "max_position_embeddings": 131072,
    "sliding_window": 1024, "sliding_window_pattern": 6,
}}


def test_moe_config_gives_active_params():
    spec = spec_from_config("Qwen/Qwen3-30B-A3B", QWEN3_30B_A3B, 30.5e9)
    assert spec.is_moe
    assert 2.5 < spec.active < 4.0
    assert spec.kv_heads == 4 and spec.layers == 48


DEEPSEEK_R1 = {
    "hidden_size": 7168, "num_hidden_layers": 61, "num_attention_heads": 128,
    "num_key_value_heads": 128, "max_position_embeddings": 163840, "vocab_size": 129280,
    "intermediate_size": 18432, "n_routed_experts": 256, "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048, "first_k_dense_replace": 3, "n_shared_experts": 1,
    "kv_lora_rank": 512, "qk_rope_head_dim": 64, "num_nextn_predict_layers": 1,
}


def test_deepseek_layout_gives_about_37b_active():
    spec = spec_from_config("deepseek-ai/DeepSeek-R1", DEEPSEEK_R1, 684.5e9)
    assert 33 < spec.active < 45, spec.active
    assert 10 < spec.shared_b < spec.active
    assert spec.kv_bytes_per_token == 61 * 576 * 2


def test_odd_configs_fail_cleanly(tmp_path):
    from willitrun.hf import spec_from_file
    gpt2 = {"n_embd": 768, "n_layer": 12, "n_head": 12, "n_positions": 1024, "vocab_size": 50257}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(gpt2))
    assert spec_from_file(str(p), 0.124).layers == 12
    p.write_text("[1, 2]")
    with pytest.raises(SystemExit):
        spec_from_file(str(p))
    p.write_text("<html>")
    with pytest.raises(SystemExit):
        spec_from_file(str(p))


def test_nested_text_config_and_sliding_window():
    spec = spec_from_config("google/gemma-3-27b-it", GEMMA3_TEXT, 27.4e9)
    assert spec.window == (1024, 5 / 6)
    assert not spec.is_moe


def test_param_count_fallback_is_in_the_right_ballpark():
    spec = spec_from_config("Qwen/Qwen3-30B-A3B", QWEN3_30B_A3B, None)
    assert 25 < spec.params_b < 36


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("WILLITRUN_HOME", str(tmp_path / "home"))
    yield
    report.set_unicode(True)
    report.set_color(False)


def test_cli_overview_runs(capsys):
    assert main(["--gpu", "RTX 4090", "--ram", "64", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "Best picks" in out and "Upgrades" in out and "Runs well" in out


def test_cli_model_detail_has_commands(capsys):
    assert main(["--gpu", "RTX 3060", "--ram", "32", "-m", "qwen3-30b-a3b", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "--n-cpu-moe" in out and "ollama run qwen3:30b" in out


def test_cli_json(capsys):
    assert main(["--gpu", "M4 Max", "--ram", "64", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["machine"]["unified_memory"] is True
    assert any(m["id"] == "gpt-oss-20b" and m["verdict"] == "great" for m in data["models"])


def test_cli_ctx_suffix(capsys):
    assert main(["--gpu", "RTX 4090", "-m", "llama3.1-8b", "-c", "32k", "--no-color"]) == 0
    assert "32,768" in capsys.readouterr().out


def test_add_local_config_then_it_shows_everywhere(tmp_path, capsys):
    cfg = tmp_path / "my-model" / "config.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps(QWEN3_30B_A3B))
    assert main(["add", str(cfg), "--params", "30.5", "--id", "mine", "--name", "My Model",
                 "--tags", "coding"]) == 0
    assert "Added My Model" in capsys.readouterr().out
    assert main(["models", "--no-color"]) == 0
    assert "mine" in capsys.readouterr().out
    assert main(["--gpu", "RTX 4090", "--ram", "64", "--no-color"]) == 0
    assert "My Model *" in capsys.readouterr().out
    assert main(["-m", "mine", "--gpu", "RTX 4090", "--no-color"]) == 0
    assert "My Model" in capsys.readouterr().out
    assert main(["remove", "mine"]) == 0
    assert main(["remove", "mine"]) == 2


def test_add_rejects_garbage(tmp_path, capsys):
    bad = tmp_path / "config.json"
    bad.write_text("{}")
    with pytest.raises(SystemExit):
        main(["add", str(bad), "--params", "7"])
    assert main(["add", "not-a-repo"]) == 2


def test_broken_user_file_gives_clear_error(tmp_path, monkeypatch):
    home = tmp_path / "h"
    home.mkdir()
    (home / "models.json").write_text("[{oops")
    monkeypatch.setenv("WILLITRUN_HOME", str(home))
    with pytest.raises(SystemExit) as e:
        main(["--gpu", "RTX 4090", "--no-color"])
    assert "models file" in str(e.value)


def test_ascii_mode_has_no_box_drawing(capsys):
    assert main(["--gpu", "RTX 3060", "--ram", "32", "--ascii", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert out.isascii(), sorted({ch for ch in out if not ch.isascii()})


def test_every_screen_fits_80_columns(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    for args in (["--gpu", "RTX 3090", "--ram", "64"], ["--gpu", "M3 Ultra", "--ram", "512"],
                 ["--gpu", "none", "--ram", "16"], ["-m", "gpt-oss-120b", "--gpu", "RTX 3060", "--ram", "32"],
                 ["--list-gpus"], ["models"], ["--gpu", "M4 Max", "--ram", "64", "-m", "llama3.3-70b"],
                 ["--gpu", "Ryzen AI Max", "--ram", "128", "--ctx", "128k", "--all"],
                 ["--gpu", "RTX 3060", "-m", "deepseek-r1-671b"]):
        assert main(args + ["--no-color"]) == 0
        for line in capsys.readouterr().out.splitlines():
            if line.lstrip().startswith("$ ") or line.endswith("models.json"):
                continue  # commands stay on one line so they can be copied
            assert len(line) <= 80, line


def test_bad_flags_are_rejected():
    for args in (["--ctx", "lots"], ["--ctx", "0"], ["--ram", "-4"], ["--gpu-count", "0"]):
        with pytest.raises(SystemExit) as e:
            main(args)
        assert e.value.code == 2


def test_kv_cache_choice_reaches_the_run_command(capsys):
    assert main(["--gpu", "RTX 4090", "-m", "qwen3-32b", "--ctx", "32k", "--kv-cache", "q8_0", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "-ctk q8_0 -ctv q8_0" in out and "OLLAMA_KV_CACHE_TYPE=q8_0" in out


def test_upgrade_advisor_only_offers_macs_that_exist(capsys):
    assert main(["--gpu", "M4 Pro", "--ram", "48", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "with 96 GB" not in out and "with 128 GB" not in out


def test_impossible_hardware_flags_are_explained(capsys):
    for args in (["--gpu", "RTX 9999"], ["--gpu", "M4 Max", "--gpu-count", "2"], ["--gpu", "none", "--vram", "24"]):
        with pytest.raises(SystemExit) as e:
            main(args + ["--no-color"])
        assert e.value.code == 2
    assert "--vram" in capsys.readouterr().err


def test_ambiguous_model_name(capsys):
    assert main(["--gpu", "RTX 4090", "-m", "phi4", "--no-color"]) == 2
    assert "phi4-mini" in capsys.readouterr().err
