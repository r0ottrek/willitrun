"""Model catalog and quantization formats.

The built-in models are in models.json. People add their own with
`willitrun add org/repo`, which saves them to a user models file (see
user_models.py).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields


@dataclass(frozen=True)
class Quant:
    name: str
    bpw: float          # effective bits per weight, including scales/metadata
    rank: int           # higher = better quality
    note: str


# Effective bits-per-weight for llama.cpp GGUF formats (measured on real files,
# so they include block scales and the few tensors kept at higher precision).
QUANTS: list[Quant] = [
    Quant("Q2_K", 3.00, 1, "big quality loss, last resort"),
    Quant("Q3_K_M", 3.91, 2, "noticeable quality loss"),
    Quant("Q4_K_M", 4.85, 3, "the usual sweet spot"),
    Quant("Q5_K_M", 5.69, 4, "very close to full quality"),
    Quant("Q6_K", 6.56, 5, "practically lossless"),
    Quant("Q8_0", 8.50, 6, "lossless for chat use"),
    Quant("F16", 16.0, 7, "full precision, rarely worth the memory"),
]
QUANT_BY_NAME = {q.name: q for q in QUANTS}
RECOMMENDED_MIN_RANK = QUANT_BY_NAME["Q4_K_M"].rank


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    params_b: float                 # total parameters, billions
    layers: int
    kv_heads: int
    head_dim: int
    max_ctx: int
    year: int
    tags: tuple[str, ...] = ("chat",)
    active_b: float | None = None   # MoE: parameters used per token
    # MoE: parameters outside the routed experts (attention, embeddings,
    # shared experts). These are read for every token, so they belong on the GPU.
    shared_b: float | None = None
    # Sliding-window / chunked attention: (window tokens, fraction of layers using it).
    # Gemma 3: 1024 on 5 of 6 layers. gpt-oss: 128 on half. Llama 4: 8192 chunks on 3 of 4.
    window: tuple[int, float] | None = None
    # Models with compressed KV (DeepSeek MLA) give bytes/token at f16 directly
    kv_bytes_per_token: int | None = None
    # Models that ship in one native format (gpt-oss MXFP4): quant name -> file GB
    native: dict[str, float] = field(default_factory=dict)
    ollama: str | None = None
    gguf: str | None = None         # Hugging Face GGUF repo for llama.cpp -hf

    @property
    def is_moe(self) -> bool:
        return self.active_b is not None and self.active_b < self.params_b

    @property
    def active(self) -> float:
        return self.active_b if self.active_b is not None else self.params_b

    @property
    def effective_b(self) -> float:
        """Rough 'dense-equivalent' size used only for ranking picks.

        A common rule of thumb puts an MoE's quality near the geometric mean of
        its total and active parameter counts.
        """
        if self.is_moe:
            return (self.params_b * self.active) ** 0.5
        return self.params_b


def spec_from_dict(d: dict) -> ModelSpec:
    d = dict(d)
    d.pop("source", None)
    if "tags" in d:
        d["tags"] = tuple(d["tags"])
    if d.get("window"):
        d["window"] = (int(d["window"][0]), float(d["window"][1]))
    return ModelSpec(**d)


def spec_to_dict(m: ModelSpec) -> dict:
    out = {}
    for f in fields(m):
        v = getattr(m, f.name)
        if v in (None, {}, ()) or (f.name == "tags" and v == ("chat",)):
            continue
        out[f.name] = list(v) if isinstance(v, tuple) else v
    return out


def _load_builtin() -> list[ModelSpec]:
    with open(os.path.join(os.path.dirname(__file__), "models.json"), encoding="utf-8") as f:
        return [spec_from_dict(d) for d in json.load(f)]


# The built-in list lives in models.json: one model per line, numbers copied
# from each model's config.json and model card.
CATALOG: list[ModelSpec] = _load_builtin()

CATALOG_BY_ID = {m.id: m for m in CATALOG}


def find_models(query: str, catalog: list[ModelSpec] | None = None) -> list[ModelSpec]:
    """Exact id first; otherwise every model whose id or name contains the query."""
    catalog = catalog or CATALOG
    q = query.lower().replace(" ", "").replace("_", "-")
    exact = [m for m in catalog if m.id == q or m.name.lower().replace(" ", "") == q]
    if exact:
        return exact[:1]
    return [m for m in catalog if q in m.id or q in m.name.lower().replace(" ", "")]


def find_model(query: str, catalog: list[ModelSpec] | None = None) -> ModelSpec | None:
    """The single match, or None if there are none or several."""
    hits = find_models(query, catalog)
    return hits[0] if len(hits) == 1 else None
