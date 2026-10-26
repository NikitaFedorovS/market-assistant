"""Agent loop: router -> (RAG) -> LLM with tools -> guarded tool execution -> reply.

Guards that do not depend on the model behaving:
- irreversible tools (``create_return``) run only after an explicit user confirmation;
- at most ``agent.max_steps`` LLM calls per message;
- invalid tool calls are reported back to the model; tool crashes / exhausted steps -> operator handoff.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from market_assistant.agent.tools import MarketTools, ToolError, tool_schemas
from market_assistant.config import AgentConfig
from market_assistant.data.normalize_lang import detect_lang
from market_assistant.data.persona import build_system_prompt
from market_assistant.llm.client import ChatClient, ToolCall
from market_assistant.router.classifier import HANDOFF_INTENT, SHORT_PATH_INTENTS, RouteDecision, Router, wants_operator

logger = logging.getLogger(__name__)

RAG_INTENTS = {"faq", "return"}
CONFIRM_RE = re.compile(r"^\s*(да|ага|подтверждаю|оформляй|давай|ok|ок|ha|xo'p|hop|tasdiqlayman|mayli)\b", re.I)

CANNED = {
    "handoff": {"ru": "Hop, подключаю оператора 🙂 Обычно отвечают за пару минут.",
                "uz": "Xo'p, operatorni ulayapman 🙂 Odatda bir necha daqiqada javob berishadi."},
    "fallback": {"ru": "Что-то пошло не так с моей стороны 🙏 Подключаю оператора, он поможет.",
                 "uz": "Nimadir xato ketdi 🙏 Operatorni ulayapman, u yordam beradi."},
    "confirm": {"ru": "Подтверди, пожалуйста: оформить возврат по заказу {order_id}?",
                "uz": "Iltimos, tasdiqlang: {order_id} buyurtma bo'yicha qaytarishni rasmiylashtiraymi?"},
    "declined": {"ru": "Xo'p, ничего не оформляю. Если передумаешь — я тут 🙂",
                 "uz": "Xo'p, hech narsa rasmiylashtirmayman. Fikringiz o'zgarsa — yozing 🙂"},
}  # fmt: skip


@dataclass
class Session:
    """Per-conversation state kept by the gateway."""

    history: list[dict[str, Any]] = field(default_factory=list)
    pending: ToolCall | None = None  # irreversible call waiting for confirmation


@dataclass
class AgentResult:
    """What the gateway returns to the client."""

    reply: str
    intent: str
    handoff: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0


class Agent:
    """Orchestrates routing, retrieval, LLM calls and tools for one user message."""

    def __init__(
        self,
        llm: ChatClient,
        tools: MarketTools,
        cfg: AgentConfig,
        router: Router | None = None,
        retriever: Any | None = None,
        max_history: int = 10,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.cfg = cfg
        self.router = router
        self.retriever = retriever
        self.max_history = max_history
        self.schemas = tool_schemas()

    def _run_tool(self, call: ToolCall, events: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            result = self.tools.call(call.name, json.loads(call.arguments))
        except (ToolError, json.JSONDecodeError) as e:
            result = {"error": "invalid_call", "detail": str(e)[:300]}
        events.append({"type": "tool", "name": call.name, "arguments": call.arguments, "result": result})
        return result

    def handle(self, session: Session, text: str) -> AgentResult:
        """Process one user message and update ``session``."""
        t0 = time.perf_counter()
        lang = detect_lang(text)
        events: list[dict[str, Any]] = []
        if wants_operator(text):
            decision: RouteDecision | None = RouteDecision(HANDOFF_INTENT, 1.0, "rule")
        else:
            decision = self.router(text) if self.router else None
        intent = decision.intent if decision else "faq"
        if decision:
            events.append({"type": "route", "intent": intent, "confidence": round(decision.confidence, 3),
                           "source": decision.source})  # fmt: skip

        def done(reply: str, handoff: bool = False) -> AgentResult:
            session.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
            session.history = session.history[-2 * self.max_history :]
            ms = (time.perf_counter() - t0) * 1000
            return AgentResult(reply, intent, handoff, events, round(ms, 1))

        # 1. pending irreversible action: execute only on explicit confirmation
        if session.pending is not None:
            call, session.pending = session.pending, None
            if not CONFIRM_RE.search(text):
                return done(CANNED["declined"][lang])
            messages = [*session.history, {"role": "user", "content": text}]
            result = self._run_tool(call, events)
            messages += [
                {"role": "assistant", "content": "", "tool_calls": [call.to_message()]},
                {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": json.dumps(result, ensure_ascii=False)},
            ]  # fmt: skip
            return done(self._final(messages, events, lang))

        # 2. explicit handoff never needs the LLM
        if intent == HANDOFF_INTENT:
            return done(CANNED["handoff"][lang], handoff=True)

        # 3. context for FAQ/return questions
        context = None
        if self.retriever is not None and intent in RAG_INTENTS:
            context = self.retriever.search(text)
            events.append({"type": "retrieve", "doc_ids": [c["doc_id"] for c in context]})
        system = {"role": "system", "content": build_system_prompt(context)}
        messages = [system, *session.history, {"role": "user", "content": text}]
        tools = None if intent in SHORT_PATH_INTENTS else self.schemas

        # 4. LLM <-> tools loop
        try:
            for _ in range(self.cfg.max_steps):
                resp = self.llm.chat(messages, tools)
                if not resp.tool_calls:
                    events.append({"type": "message"})
                    return done(resp.content)
                messages.append({"role": "assistant", "content": resp.content,
                                 "tool_calls": [c.to_message() for c in resp.tool_calls]})  # fmt: skip
                for call in resp.tool_calls:
                    if call.name in self.cfg.confirm_tools and not self._confirmed(session, text):
                        session.pending = call
                        events.append({"type": "confirm_required", "name": call.name})
                        order_id = json.loads(call.arguments).get("order_id", "?")
                        return done(CANNED["confirm"][lang].format(order_id=order_id))
                    result = self._run_tool(call, events)
                    messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                     "content": json.dumps(result, ensure_ascii=False)})  # fmt: skip
        except Exception:
            logger.exception("Agent failure")
            return done(CANNED["fallback"][lang], handoff=True)
        events.append({"type": "max_steps"})
        return done(CANNED["fallback"][lang], handoff=True)

    @staticmethod
    def _confirmed(session: Session, text: str) -> bool:
        """An irreversible tool may run only if the assistant just asked and the user said yes."""
        if not session.history or not CONFIRM_RE.search(text):
            return False
        last = session.history[-1]
        return last["role"] == "assistant" and last["content"].rstrip().endswith("?")

    def _final(self, messages: list[dict[str, Any]], events: list[dict[str, Any]], lang: str) -> str:
        """Let the model phrase the tool result (no further tools allowed)."""
        system = {"role": "system", "content": build_system_prompt()}
        resp = self.llm.chat([system, *messages], tools=None)
        events.append({"type": "message"})
        return resp.content or CANNED["fallback"][lang]
