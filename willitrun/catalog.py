"""Model catalog and quantization formats.

Every number here comes from the model's published config.json (layer count,
KV heads, head size, attention windows) or its model card (parameter counts).
To add a model, add a ModelSpec below, or point the CLI at any Hugging Face
repo with --hf and it will read the config for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    # Sliding-window / chunked attention: (window tokens, fraction of layers using it)
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


GEMMA3_WINDOW = (1024, 5 / 6)       # 5 local layers for every global layer
GPT_OSS_WINDOW = (128, 0.5)          # alternating banded / full attention
LLAMA4_CHUNK = (8192, 0.75)          # chunked attention on 3 of 4 layers

CATALOG: list[ModelSpec] = [
    # --- Meta Llama
    ModelSpec("llama3.2-1b", "Llama 3.2 1B", 1.24, 16, 8, 64, 131072, 2024,
              ollama="llama3.2:1b", gguf="bartowski/Llama-3.2-1B-Instruct-GGUF"),
    ModelSpec("llama3.2-3b", "Llama 3.2 3B", 3.21, 28, 8, 128, 131072, 2024,
              ollama="llama3.2:3b", gguf="bartowski/Llama-3.2-3B-Instruct-GGUF"),
    ModelSpec("llama3.1-8b", "Llama 3.1 8B", 8.03, 32, 8, 128, 131072, 2024,
              ollama="llama3.1:8b", gguf="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
    ModelSpec("llama3.3-70b", "Llama 3.3 70B", 70.6, 80, 8, 128, 131072, 2024,
              ollama="llama3.3:70b", gguf="bartowski/Llama-3.3-70B-Instruct-GGUF"),
    ModelSpec("llama4-scout", "Llama 4 Scout (109B MoE)", 109.0, 48, 8, 128, 131072, 2025,
              tags=("chat", "vision"), active_b=17.0, shared_b=11.0, window=LLAMA4_CHUNK,
              ollama="llama4:scout", gguf="unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF"),
    # --- Qwen
    ModelSpec("qwen3-4b", "Qwen3 4B", 4.02, 36, 8, 128, 32768, 2025,
              tags=("chat", "reasoning"), ollama="qwen3:4b", gguf="unsloth/Qwen3-4B-GGUF"),
    ModelSpec("qwen3-8b", "Qwen3 8B", 8.19, 36, 8, 128, 32768, 2025,
              tags=("chat", "reasoning"), ollama="qwen3:8b", gguf="unsloth/Qwen3-8B-GGUF"),
    ModelSpec("qwen3-14b", "Qwen3 14B", 14.8, 40, 8, 128, 32768, 2025,
              tags=("chat", "reasoning"), ollama="qwen3:14b", gguf="unsloth/Qwen3-14B-GGUF"),
    ModelSpec("qwen3-32b", "Qwen3 32B", 32.8, 64, 8, 128, 32768, 2025,
              tags=("chat", "reasoning", "coding"), ollama="qwen3:32b", gguf="unsloth/Qwen3-32B-GGUF"),
    ModelSpec("qwen3-30b-a3b", "Qwen3 30B-A3B (MoE)", 30.5, 48, 4, 128, 32768, 2025,
              tags=("chat", "reasoning", "coding"), active_b=3.3, shared_b=1.5,
              ollama="qwen3:30b", gguf="unsloth/Qwen3-30B-A3B-GGUF"),
    ModelSpec("qwen3-235b-a22b", "Qwen3 235B-A22B (MoE)", 235.0, 94, 4, 128, 32768, 2025,
              tags=("chat", "reasoning", "coding"), active_b=22.0, shared_b=7.8,
              ollama="qwen3:235b", gguf="unsloth/Qwen3-235B-A22B-GGUF"),
    ModelSpec("qwen2.5-coder-7b", "Qwen2.5 Coder 7B", 7.62, 28, 4, 128, 32768, 2024,
              tags=("coding",), ollama="qwen2.5-coder:7b",
              gguf="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"),
    ModelSpec("qwen2.5-coder-32b", "Qwen2.5 Coder 32B", 32.8, 64, 8, 128, 32768, 2024,
              tags=("coding",), ollama="qwen2.5-coder:32b",
              gguf="bartowski/Qwen2.5-Coder-32B-Instruct-GGUF"),
    # --- Google Gemma
    ModelSpec("gemma3-4b", "Gemma 3 4B", 4.3, 34, 4, 256, 131072, 2025,
              tags=("chat", "vision"), window=GEMMA3_WINDOW,
              ollama="gemma3:4b", gguf="unsloth/gemma-3-4b-it-GGUF"),
    ModelSpec("gemma3-12b", "Gemma 3 12B", 12.2, 48, 8, 256, 131072, 2025,
              tags=("chat", "vision"), window=GEMMA3_WINDOW,
              ollama="gemma3:12b", gguf="unsloth/gemma-3-12b-it-GGUF"),
    ModelSpec("gemma3-27b", "Gemma 3 27B", 27.4, 62, 16, 128, 131072, 2025,
              tags=("chat", "vision"), window=GEMMA3_WINDOW,
              ollama="gemma3:27b", gguf="unsloth/gemma-3-27b-it-GGUF"),
    # --- Mistral
    ModelSpec("mistral-nemo-12b", "Mistral Nemo 12B", 12.2, 40, 8, 128, 131072, 2024,
              ollama="mistral-nemo:12b", gguf="bartowski/Mistral-Nemo-Instruct-2407-GGUF"),
    ModelSpec("mistral-small-24b", "Mistral Small 3.1 24B", 24.0, 40, 8, 128, 131072, 2025,
              tags=("chat", "vision"), ollama="mistral-small3.1:24b",
              gguf="unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF"),
    # --- Microsoft Phi
    ModelSpec("phi4-mini", "Phi-4 Mini 3.8B", 3.84, 32, 8, 128, 131072, 2025,
              tags=("chat", "reasoning"), ollama="phi4-mini:3.8b",
              gguf="unsloth/Phi-4-mini-instruct-GGUF"),
    ModelSpec("phi4-14b", "Phi-4 14B", 14.7, 40, 10, 128, 16384, 2024,
              tags=("chat", "reasoning"), ollama="phi4:14b", gguf="bartowski/phi-4-GGUF"),
    # --- OpenAI open weights (ship natively in MXFP4)
    ModelSpec("gpt-oss-20b", "gpt-oss 20B (MoE)", 20.9, 24, 8, 64, 131072, 2025,
              tags=("chat", "reasoning", "coding"), active_b=3.6, shared_b=1.2, window=GPT_OSS_WINDOW,
              native={"MXFP4": 12.8}, ollama="gpt-oss:20b", gguf="ggml-org/gpt-oss-20b-GGUF"),
    ModelSpec("gpt-oss-120b", "gpt-oss 120B (MoE)", 116.8, 36, 8, 64, 131072, 2025,
              tags=("chat", "reasoning", "coding"), active_b=5.1, shared_b=1.5, window=GPT_OSS_WINDOW,
              native={"MXFP4": 63.4}, ollama="gpt-oss:120b", gguf="ggml-org/gpt-oss-120b-GGUF"),
    # --- DeepSeek
    ModelSpec("r1-distill-qwen-14b", "DeepSeek R1 Distill Qwen 14B", 14.8, 48, 8, 128, 131072, 2025,
              tags=("reasoning",), ollama="deepseek-r1:14b",
              gguf="bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF"),
    ModelSpec("r1-distill-qwen-32b", "DeepSeek R1 Distill Qwen 32B", 32.8, 64, 8, 128, 131072, 2025,
              tags=("reasoning",), ollama="deepseek-r1:32b",
              gguf="bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF"),
    ModelSpec("deepseek-r1-671b", "DeepSeek R1 671B (MoE)", 671.0, 61, 1, 1, 131072, 2025,
              tags=("chat", "reasoning", "coding"), active_b=37.0, shared_b=16.6,
              kv_bytes_per_token=61 * (512 + 64) * 2,
              ollama="deepseek-r1:671b", gguf="unsloth/DeepSeek-R1-0528-GGUF"),
]

CATALOG_BY_ID = {m.id: m for m in CATALOG}


def find_model(query: str, catalog: list[ModelSpec] | None = None) -> ModelSpec | None:
    catalog = catalog or CATALOG
    q = query.lower().replace(" ", "").replace("_", "-")
    for m in catalog:
        if m.id == q:
            return m
    for m in catalog:
        if q in m.id or q in m.name.lower().replace(" ", ""):
            return m
    return None
