"""LLM backends behind one interface.

- ``OpenAIChatClient`` — vLLM's OpenAI-compatible server (production; tool calls parsed by vLLM).
- ``TransformersChatClient`` — local HF model (+ LoRA) for development on Mac/CPU; parses Qwen's
  ``<tool_call>{...}</tool_call>`` blocks itself.
- ``ScriptedClient`` — returns pre-defined responses (tests).
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from market_assistant.config import LLMConfig


@dataclass
class ToolCall:
    """A function call requested by the model."""

    name: str
    arguments: str  # raw JSON string, validated by the agent
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:10]}")

    def to_message(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class LLMResponse:
    """Assistant turn: text and/or tool calls."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatClient(Protocol):
    """Chat completion with optional tools."""

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse: ...


class OpenAIChatClient:
    """Client for vLLM (or any OpenAI-compatible endpoint)."""

    def __init__(self, cfg: LLMConfig) -> None:
        from openai import OpenAI

        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout_s)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            **kwargs,
        )
        msg = resp.choices[0].message
        calls = [ToolCall(tc.function.name, tc.function.arguments, tc.id) for tc in msg.tool_calls or []]
        return LLMResponse(content=msg.content or "", tool_calls=calls)


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_hermes_tool_calls(text: str) -> LLMResponse:
    """Split Qwen/Hermes-style output into text and ``<tool_call>`` JSON blocks."""
    calls = []
    for block in _TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(block)
            calls.append(ToolCall(obj["name"], json.dumps(obj.get("arguments", {}), ensure_ascii=False)))
        except (json.JSONDecodeError, KeyError):
            calls.append(ToolCall("__invalid__", block))
    return LLMResponse(content=_TOOL_CALL_RE.sub("", text).strip(), tool_calls=calls)


class TransformersChatClient:
    """Local generation with transformers (dev only — use vLLM for real traffic)."""

    def __init__(self, cfg: LLMConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from market_assistant.utils import resolve_device

        self.cfg = cfg
        self.device = resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.local_model)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(cfg.local_model, dtype=dtype)
        if cfg.local_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, cfg.local_adapter).merge_and_unload()
        self.model = model.to(self.device).eval()
        self._torch = torch

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        from market_assistant.train.chat_format import to_template_messages

        ids = self.tokenizer.apply_chat_template(
            to_template_messages(messages), tools=tools, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        ).to(self.device)  # fmt: skip
        with self._torch.inference_mode():
            out = self.model.generate(
                **ids,
                max_new_tokens=self.cfg.max_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=self.cfg.temperature or None,
                top_p=0.9 if self.cfg.temperature > 0 else None,
            )
        text = self.tokenizer.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
        return parse_hermes_tool_calls(text)


class ScriptedClient:
    """Test double: returns the given responses in order and records the requests."""

    def __init__(self, responses: Iterable[LLMResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.requests.append({"messages": messages, "tools": tools})
        if not self.responses:
            return LLMResponse(content="")
        return self.responses.pop(0)


def build_client(cfg: LLMConfig) -> ChatClient:
    """Factory by ``cfg.backend``."""
    if cfg.backend == "openai":
        return OpenAIChatClient(cfg)
    if cfg.backend == "transformers":
        return TransformersChatClient(cfg)
    return ScriptedClient([LLMResponse(content="(mock) Salom! Чем помочь?")] * 1000)
