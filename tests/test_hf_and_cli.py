import json

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


def test_nested_text_config_and_sliding_window():
    spec = spec_from_config("google/gemma-3-27b-it", GEMMA3_TEXT, 27.4e9)
    assert spec.window == (1024, 5 / 6)
    assert not spec.is_moe


def test_param_count_fallback_is_in_the_right_ballpark():
    spec = spec_from_config("Qwen/Qwen3-30B-A3B", QWEN3_30B_A3B, None)
    assert 25 < spec.params_b < 36


def test_cli_overview_runs(capsys):
    assert main(["--gpu", "RTX 4090", "--ram", "64", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "Best picks" in out and "Upgrade advisor" in out


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
