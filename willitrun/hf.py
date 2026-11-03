"""Build a ModelSpec for any Hugging Face model from its config.json.

This is what keeps the tool current: new models come out weekly, and the
built-in catalog can't keep up. `willitrun --hf Qwen/Qwen3-8B` reads the
architecture straight from the source.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .catalog import ModelSpec

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def _get(url: str) -> dict:
    headers = {"User-Agent": "willitrun"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def spec_from_config(repo: str, cfg: dict, total_params: float | None) -> ModelSpec:
    # Multimodal configs nest the language model under text_config.
    text = cfg.get("text_config") or cfg.get("llm_config") or cfg
    hidden = text["hidden_size"]
    layers = text["num_hidden_layers"]
    heads = text["num_attention_heads"]
    kv_heads = text.get("num_key_value_heads") or heads
    head_dim = text.get("head_dim") or hidden // heads
    max_ctx = text.get("max_position_embeddings") or 32768

    vocab = text.get("vocab_size", 32000)
    inter = text.get("intermediate_size", 4 * hidden)
    experts = (text.get("num_local_experts") or text.get("num_experts")
               or text.get("n_routed_experts") or 0)
    per_tok = text.get("num_experts_per_tok") or text.get("num_experts_per_token") or 0
    moe_inter = text.get("moe_intermediate_size") or inter

    if total_params is None:  # fall back to counting from the shapes
        attn = hidden * head_dim * (heads + 2 * kv_heads) + heads * head_dim * hidden
        mlp = 3 * hidden * (moe_inter * experts if experts else inter)
        total_params = layers * (attn + mlp) + 2 * vocab * hidden

    active = shared = None
    if experts and per_tok:
        expert_params = layers * experts * 3 * hidden * moe_inter
        active = total_params - expert_params * (1 - per_tok / experts)
        active = max(active, total_params * 0.02)
        shared = max(total_params - expert_params, 0.0)

    window = None
    sw = text.get("sliding_window")
    layer_types = text.get("layer_types")
    if sw and layer_types:
        local = sum(1 for t in layer_types if "sliding" in str(t) or "chunk" in str(t))
        if local:
            window = (int(sw), local / len(layer_types))
    elif sw and text.get("sliding_window_pattern"):
        pat = int(text["sliding_window_pattern"])
        window = (int(sw), (pat - 1) / pat)

    kv_bytes = None
    if text.get("kv_lora_rank"):  # DeepSeek-style multi-head latent attention
        kv_bytes = layers * (text["kv_lora_rank"] + text.get("qk_rope_head_dim", 64)) * 2

    name = repo.split("/")[-1]
    return ModelSpec(
        id=name.lower(), name=name, params_b=total_params / 1e9, layers=layers,
        kv_heads=kv_heads, head_dim=head_dim, max_ctx=int(max_ctx), year=2026,
        tags=("chat",), active_b=active / 1e9 if active else None,
        shared_b=shared / 1e9 if shared is not None else None, window=window,
        kv_bytes_per_token=kv_bytes, gguf=None,
    )


def fetch_spec(repo: str) -> ModelSpec:
    try:
        cfg = _get(f"{HF}/{repo}/resolve/main/config.json")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise SystemExit(f"{repo} is gated. Accept its license on Hugging Face and set HF_TOKEN.")
        raise SystemExit(f"Couldn't read config.json for {repo} (HTTP {e.code}).")
    except urllib.error.URLError as e:
        raise SystemExit(f"Couldn't reach Hugging Face: {e.reason}")
    total = None
    try:
        info = _get(f"{HF}/api/models/{repo}")
        total = (info.get("safetensors") or {}).get("total")
    except (urllib.error.URLError, ValueError):
        pass
    return spec_from_config(repo, cfg, float(total) if total else None)
