"""Chat-template tokenization with an assistant-only loss mask (shared by training and eval)."""

from __future__ import annotations

import copy
import json
from typing import Any

from market_assistant.agent.tools import tool_schemas


def to_template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format messages for HF chat templates (tool-call arguments as dicts)."""
    out = []
    for m in messages:
        m = copy.deepcopy(m)
        for tc in m.get("tool_calls") or []:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                tc["function"]["arguments"] = json.loads(args)
        if m.get("tool_calls") and not m.get("content"):
            m["content"] = ""
        out.append(m)
    return out


def tokenize_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    uses_tools: bool,
    max_len: int,
) -> dict[str, list[int]]:
    """Render with the chat template and mark assistant tokens.

    Every assistant turn's span is located by rendering the conversation prefix with
    ``add_generation_prompt=True`` (start) and the prefix including the turn (end). Tokens whose
    character offset falls inside a span get ``assistant_masks == 1`` — the only ones in the loss.

    Returns:
        ``{"input_ids": [...], "assistant_masks": [...]}``, truncated to ``max_len``.
    """
    msgs = to_template_messages(messages)
    tools = tool_schemas() if uses_tools else None

    def render(part: list[dict[str, Any]], gen_prompt: bool = False) -> str:
        return tokenizer.apply_chat_template(part, tools=tools, tokenize=False, add_generation_prompt=gen_prompt)

    full = render(msgs)
    spans: list[tuple[int, int]] = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        start, end = len(render(msgs[:i], gen_prompt=True)), len(render(msgs[: i + 1]))
        if not full.startswith(render(msgs[: i + 1])):
            raise ValueError("chat template is not prefix-stable; cannot build assistant mask")
        spans.append((start, end))

    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    mask = [int(any(s <= off[0] < e for s, e in spans)) for off in enc["offset_mapping"]]
    return {"input_ids": enc["input_ids"][:max_len], "assistant_masks": mask[:max_len]}
