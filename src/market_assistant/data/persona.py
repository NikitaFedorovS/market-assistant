"""Persona system prompt and stylization of operator replies.

Two stylization modes:
- ``rule``: synthetic dialogs carry a persona rendering next to the dry operator reply; the stylizer
  picks it (offline, deterministic — the default for the demo).
- ``teacher``: an OpenAI-compatible teacher LLM rewrites the dry operator reply using ``persona.md``
  (the production path for real support logs).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты — Dilkash, ассистент маркетплейса Market. Говоришь тепло и коротко: по-русски на «ты», "
    "по-узбекски на «siz», можешь вставлять узбекские слова (Salom, rahmat, hop, zo'r). Не больше одного "
    "эмодзи. Сроки, статусы и цены — только из инструментов или контекста, ничего не выдумывай. Перед "
    "оформлением возврата всегда спрашивай подтверждение. В жалобах — без шуток, предлагай оператора. "
    "Отвечай на языке пользователя."
)


def build_system_prompt(context: list[dict[str, Any]] | None = None) -> str:
    """System prompt with optional RAG context (list of KB chunks with ``title``/``text``)."""
    if not context:
        return SYSTEM_PROMPT
    ctx = "\n\n".join(f"[{c.get('title', '')}]\n{c['text']}" for c in context)
    return f"{SYSTEM_PROMPT}\n\nКонтекст из базы знаний (отвечай только по нему):\n{ctx}"


class TeacherStylizer:
    """Rewrites a dry operator reply in the persona voice with an OpenAI-compatible LLM."""

    def __init__(self, base_url: str, model: str, persona_path: str | Path) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url)
        self.model = model
        self.guide = Path(persona_path).read_text(encoding="utf-8")

    def __call__(self, user_text: str, operator_reply: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.7,
            messages=[
                {"role": "system", "content": f"Перепиши ответ оператора голосом персоны.\n\n{self.guide}"},
                {
                    "role": "user",
                    "content": f"Сообщение клиента: {user_text}\nОтвет оператора: {operator_reply}\n\n"
                    "Сохрани все факты, числа и номера без изменений. Верни только новый ответ.",
                },
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def stylize_dialog(dialog: dict[str, Any], teacher: TeacherStylizer | None = None) -> dict[str, Any]:
    """Replace dry assistant replies with the persona version (rule-based or via teacher)."""
    messages = []
    last_user = ""
    for msg in dialog["messages"]:
        msg = dict(msg)
        if msg["role"] == "user":
            last_user = msg["content"]
        if msg["role"] == "assistant" and msg.get("content"):
            if teacher is not None:
                msg["content"] = teacher(last_user, msg["content"])
            elif msg.get("persona"):
                msg["content"] = msg["persona"]
        msg.pop("persona", None)
        messages.append(msg)
    return {**dialog, "messages": messages}
