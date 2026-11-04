"""Build a ModelSpec for any Hugging Face model from its config.json.

This is what keeps the tool current: new models come out weekly, and the
built-in catalog can't keep up. `willitrun --hf Qwen/Qwen3-8B` reads the
architecture straight from the source.
"""

from __future__ import annotations

import datetime
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
    hidden = text.get("hidden_size") or text["n_embd"]
    layers = text.get("num_hidden_layers") or text["n_layer"]
    heads = text.get("num_attention_heads") or text["n_head"]
    kv_heads = text.get("num_key_value_heads") or heads
    head_dim = text.get("head_dim") or hidden // heads
    max_ctx = text.get("max_position_embeddings") or text.get("n_positions") or 32768

    vocab = text.get("vocab_size", 32000)
    inter = text.get("intermediate_size", 4 * hidden)
    experts = (text.get("num_local_experts") or text.get("num_experts")
               or text.get("n_routed_experts") or 0)
    per_tok = text.get("num_experts_per_tok") or text.get("num_experts_per_token") or 0
    moe_inter = text.get("moe_intermediate_size") or inter

    # DeepSeek / Kimi / GLM style: the first k layers are dense, and every MoE
    # layer also has always-on shared experts.
    dense_first = int(text.get("first_k_dense_replace") or 0) if experts else 0
    moe_layers = layers - dense_first
    # Multi-token-prediction layers ship in the weights (and the parameter count)
    # but aren't used for normal generation.
    moe_layers_stored = moe_layers + (int(text.get("num_nextn_predict_layers") or 0) if experts else 0)
    shared_experts = int(text.get("n_shared_experts") or 0)
    routed = moe_layers_stored * experts * 3 * hidden * moe_inter   # all routed expert weights

    if total_params is None:  # fall back to counting from the shapes
        attn = hidden * head_dim * (heads + 2 * kv_heads) + heads * head_dim * hidden
        dense_mlp = 3 * hidden * inter
        always = moe_layers * shared_experts * 3 * hidden * moe_inter
        total_params = (layers * attn + dense_first * dense_mlp + always + routed
                        + (0 if experts else layers * dense_mlp) + 2 * vocab * hidden)

    active = shared = None
    if experts and per_tok:
        routed = min(routed, total_params * 0.98)
        active = total_params - routed * (1 - per_tok / experts)
        shared = total_params - routed

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
        kv_heads=kv_heads, head_dim=head_dim, max_ctx=int(max_ctx), year=datetime.date.today().year,
        tags=("chat",), active_b=active / 1e9 if active else None,
        shared_b=shared / 1e9 if shared is not None else None, window=window,
        kv_bytes_per_token=kv_bytes, gguf=None,
    )


def fetch_spec(repo: str) -> ModelSpec:
    try:
        cfg = _get(f"{HF}/{repo}/resolve/main/config.json")
    except ValueError:
        raise SystemExit(f"{repo}/config.json isn't JSON. The repo may be private or not a model.")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise SystemExit(f"{repo} is gated. Accept its license on Hugging Face and set HF_TOKEN.")
        raise SystemExit(f"Couldn't read config.json for {repo} (HTTP {e.code}).")
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(f"Couldn't reach Hugging Face: {getattr(e, 'reason', e)}")
    if not isinstance(cfg, dict):
        raise SystemExit(f"{repo}/config.json isn't a model config.")
    total = None
    try:
        info = _get(f"{HF}/api/models/{repo}")
        total = (info.get("safetensors") or {}).get("total")
    except (urllib.error.URLError, OSError, ValueError, AttributeError):
        pass
    try:
        return spec_from_config(repo, cfg, float(total) if total else None)
    except (KeyError, TypeError, AttributeError) as e:
        raise SystemExit(f"{repo}'s config.json has no {e}. It may not be a text model this tool understands.")


def spec_from_file(path: str, params_b: float | None = None) -> ModelSpec:
    """Same as fetch_spec, for a config.json on disk (local or private models)."""
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(f"Couldn't read {path}: {e}")
    name = os.path.basename(os.path.dirname(os.path.abspath(path))) or "local-model"
    try:
        return spec_from_config(name, cfg, params_b * 1e9 if params_b else None)
    except (KeyError, TypeError, AttributeError) as e:
        raise SystemExit(f"{path} doesn't look like a model config.json (missing {e}).")
