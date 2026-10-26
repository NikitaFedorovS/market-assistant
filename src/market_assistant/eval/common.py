"""Helpers shared by the eval scripts: building base vs fine-tuned systems, result tables."""

from __future__ import annotations

from typing import Any

from market_assistant.config import LLMConfig
from market_assistant.llm.client import ChatClient, OpenAIChatClient, TransformersChatClient


def build_systems(
    llm_cfg: LLMConfig,
    adapter: str | None,
    backend: str,
    base_model_name: str | None = None,
) -> dict[str, ChatClient]:
    """``{"base": ..., "finetuned": ...}`` clients.

    - ``transformers``: the local model without / with the LoRA adapter.
    - ``openai``: two model names on the same vLLM server — the base model and the LoRA module.
    """
    systems: dict[str, ChatClient] = {}
    if backend == "transformers":
        systems["base"] = TransformersChatClient(llm_cfg.model_copy(update={"local_adapter": None}))
        if adapter:
            systems["finetuned"] = TransformersChatClient(llm_cfg.model_copy(update={"local_adapter": adapter}))
    else:
        if base_model_name:
            systems["base"] = OpenAIChatClient(llm_cfg.model_copy(update={"model": base_model_name}))
        systems["finetuned"] = OpenAIChatClient(llm_cfg)
    return systems


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Render dict rows as a Markdown table (floats with 3 decimals)."""

    def cell(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join(lines)
