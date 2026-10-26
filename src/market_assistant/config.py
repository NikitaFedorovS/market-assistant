"""Project configuration: one Pydantic section per YAML file in ``configs/``.

``configs/data.yaml``       -> ``AppConfig.data``
``configs/train_lora.yaml`` -> ``AppConfig.train_lora``
``configs/router.yaml``     -> ``AppConfig.router``
``configs/rag.yaml``        -> ``AppConfig.rag``
``configs/serve.yaml``      -> ``AppConfig.serve``

Any field can be overridden with ``MA__<SECTION>__<FIELD>[__<SUBFIELD>]=<yaml value>``,
e.g. ``MA__SERVE__LLM__BASE_URL=http://gpu-box:8000/v1``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

ENV_PREFIX = "MA__"
SECTIONS = ("data", "train_lora", "router", "rag", "serve")
INTENTS: tuple[str, ...] = (
    "order_status",
    "return",
    "product_question",
    "faq",
    "chitchat",
    "human_handoff",
)
LANGS: tuple[str, ...] = ("ru", "uz", "mixed")


# --------------------------------------------------------------------------- data


class SynthConfig(BaseModel):
    """Synthetic "business" and dialog generation."""

    n_products: int = Field(5000, gt=0)
    n_orders: int = Field(3000, gt=0)
    dialogs_per_intent: int = Field(1500, gt=0)
    router_samples_per_intent: int = Field(3500, gt=0)
    seed: int = 42


class SFTConfig(BaseModel):
    """SFT dataset building: filters, mixing, splits."""

    min_chars: int = Field(2, ge=1)
    max_chars: int = Field(1200, gt=0)
    dedup_threshold: float = Field(0.85, gt=0, le=1)
    general_mix_ratio: float = Field(0.25, ge=0, lt=1)
    general_dataset: str | None = None  # HF instruct dataset with a `messages` column
    general_max_samples: int = Field(3000, ge=0)
    splits: tuple[float, float, float] = (0.9, 0.05, 0.05)
    # Optional OpenAI-compatible teacher LLM for persona rewriting; null = rule-based stylizer.
    teacher_base_url: str | None = None
    teacher_model: str | None = None
    seed: int = 42

    @field_validator("splits")
    @classmethod
    def _sum_to_one(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if abs(sum(v) - 1) > 1e-6:
            raise ValueError("splits must sum to 1")
        return v


class DataConfig(BaseModel):
    """Data paths and generation settings."""

    data_dir: str = "data"
    persona_path: str = "configs/persona.md"
    synth: SynthConfig = Field(default_factory=SynthConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_dir) / "raw"

    @property
    def processed_dir(self) -> Path:
        return Path(self.data_dir) / "processed"

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "market.db"

    @property
    def kb_path(self) -> Path:
        return Path(self.data_dir) / "kb" / "articles.jsonl"

    @property
    def catalog_path(self) -> Path:
        return Path(self.data_dir) / "catalog" / "products.jsonl"


# --------------------------------------------------------------------------- training


class LoraParams(BaseModel):
    """LoRA hyperparameters."""

    r: int = Field(16, gt=0)
    alpha: int = Field(32, gt=0)
    dropout: float = Field(0.05, ge=0, lt=1)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


class TrainLoraConfig(BaseModel):
    """Persona LoRA / QLoRA fine-tuning."""

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    backend: Literal["auto", "unsloth", "hf"] = "auto"
    load_in_4bit: bool = True
    max_seq_len: int = Field(2048, gt=0)
    lora: LoraParams = Field(default_factory=LoraParams)
    output_dir: str = "models/persona-lora"
    learning_rate: float = Field(1e-4, gt=0)
    num_epochs: float = Field(2, gt=0)
    batch_size: int = Field(4, gt=0)
    grad_accumulation: int = Field(4, gt=0)
    warmup_ratio: float = Field(0.03, ge=0, lt=1)
    eval_steps: int = Field(200, gt=0)
    save_steps: int = Field(200, gt=0)
    logging_steps: int = Field(10, gt=0)
    early_stopping_patience: int = Field(3, ge=0)
    gradient_checkpointing: bool = True
    # Pad batches to a multiple of N. A large value (e.g. 256) bounds the number of tensor shapes,
    # which keeps the Apple MPS allocator from fragmenting into swap; 8 suits CUDA tensor cores.
    pad_to_multiple_of: int | None = Field(8, gt=0)
    report_to: Literal["none", "mlflow", "wandb"] = "none"
    seed: int = 42


class RouterConfig(BaseModel):
    """ruBERT-tiny2 intent router."""

    base_model: str = "cointegrated/rubert-tiny2"
    output_dir: str = "models/router"
    max_len: int = Field(64, gt=0)
    learning_rate: float = Field(5e-5, gt=0)
    num_epochs: float = Field(3, gt=0)
    batch_size: int = Field(64, gt=0)
    confidence_threshold: float = Field(0.6, ge=0, le=1)
    seed: int = 42


# --------------------------------------------------------------------------- RAG


class RagConfig(BaseModel):
    """Hybrid retrieval settings."""

    embedder: str = "BAAI/bge-m3"
    reranker: str | None = "BAAI/bge-reranker-v2-m3"
    device: str | None = None  # null = auto (cuda > mps > cpu); "cpu" keeps the GPU free for the LLM
    index_dir: str = "data/index"
    collection: str = "kb"
    qdrant_url: str | None = None  # null = embedded local Qdrant in index_dir
    chunk_tokens: int = Field(300, gt=0)
    chunk_overlap: int = Field(50, ge=0)
    bm25_top_k: int = Field(50, gt=0)
    dense_top_k: int = Field(50, gt=0)
    rrf_k: int = Field(60, gt=0)
    final_top_k: int = Field(5, gt=0)
    context_token_budget: int = Field(1500, gt=0)
    compare_embedders: list[str] = Field(
        default_factory=lambda: [
            "cointegrated/rubert-tiny2",
            "intfloat/multilingual-e5-small",
            "BAAI/bge-m3",
        ]
    )


# --------------------------------------------------------------------------- serving


class LLMConfig(BaseModel):
    """LLM backend used by the agent."""

    backend: Literal["openai", "transformers", "mock"] = "openai"
    base_url: str = "http://localhost:8001/v1"  # vLLM OpenAI-compatible server
    api_key: str = "EMPTY"
    model: str = "persona"  # LoRA module name registered in vLLM
    local_model: str = "Qwen/Qwen2.5-0.5B-Instruct"  # transformers backend (dev on Mac/CPU)
    local_adapter: str | None = None
    temperature: float = Field(0.3, ge=0)
    max_tokens: int = Field(512, gt=0)
    timeout_s: float = Field(60, gt=0)


class AgentConfig(BaseModel):
    """Agent loop limits."""

    max_steps: int = Field(3, gt=0)
    confirm_tools: list[str] = Field(default_factory=lambda: ["create_return"])


class VLLMConfig(BaseModel):
    """Flags passed to ``vllm serve`` by scripts/run_vllm.sh."""

    model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    quantization: str | None = "awq"
    lora_path: str = "models/persona-lora"
    max_loras: int = 4
    max_lora_rank: int = 16
    max_model_len: int = 8192
    gpu_memory_utilization: float = Field(0.85, gt=0, le=1)
    max_num_seqs: int = 64
    port: int = 8001


class ImagesConfig(BaseModel):
    """Product background generation."""

    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    backend: Literal["diffusers", "comfyui"] = "diffusers"
    comfyui_url: str = "http://localhost:8188"
    steps: int = Field(25, gt=0)
    guidance_scale: float = Field(6.0, ge=0)
    size: int = Field(1024, gt=0)
    max_concurrent: int = Field(1, gt=0)
    unload_after_s: float = Field(300, ge=0)  # free VRAM when idle (shared GPU with the LLM)


class ApiConfig(BaseModel):
    """HTTP gateway limits."""

    max_input_chars: int = Field(2000, gt=0)
    max_history_turns: int = Field(10, gt=0)
    request_timeout_s: float = Field(90, gt=0)
    rate_limit_per_min: int = Field(60, gt=0)


class ServeConfig(BaseModel):
    """Serving section."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    images: ImagesConfig = Field(default_factory=ImagesConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)


class AppConfig(BaseModel):
    """Root config."""

    data: DataConfig = Field(default_factory=DataConfig)
    train_lora: TrainLoraConfig = Field(default_factory=TrainLoraConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)


def _set_nested(target: dict[str, Any], keys: list[str], value: Any) -> None:
    for key in keys[:-1]:
        nxt = target.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            target[key] = nxt
        target = nxt
    target[keys[-1]] = value


def apply_env_overrides(raw: dict[str, Any], environ: dict[str, str]) -> dict[str, Any]:
    """Overlay ``MA__SECTION__FIELD[__SUB]=value`` env vars (values parsed as YAML)."""
    for key, value in environ.items():
        if key.startswith(ENV_PREFIX):
            parts = key[len(ENV_PREFIX) :].lower().split("__")
            if len(parts) >= 2 and parts[0] in SECTIONS:
                _set_nested(raw, parts, yaml.safe_load(value))
    return raw


def load_config(config_dir: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load every ``configs/<section>.yaml`` that exists, apply env overrides, validate."""
    load_dotenv()
    config_dir = Path(config_dir or os.environ.get("MA_CONFIG_DIR", "configs"))
    raw: dict[str, Any] = {}
    for section in SECTIONS:
        path = config_dir / f"{section}.yaml"
        if path.exists():
            raw[section] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(apply_env_overrides(raw, dict(os.environ)))
