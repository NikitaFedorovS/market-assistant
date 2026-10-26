"""Tools and the agent loop with a scripted LLM."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from market_assistant.agent.loop import CANNED, Agent, Session
from market_assistant.agent.tools import MarketTools, ToolError, tool_schemas
from market_assistant.config import AgentConfig
from market_assistant.llm.client import LLMResponse, ScriptedClient, ToolCall, parse_hermes_tool_calls
from market_assistant.router.classifier import RouteDecision


def _order(db: Any, status: str) -> str:
    con = sqlite3.connect(db)
    row = con.execute("SELECT order_id FROM orders WHERE status = ? LIMIT 1", (status,)).fetchone()
    con.close()
    return row[0]


def _eligible_order(tools: MarketTools) -> str:
    con = sqlite3.connect(tools.db_path)
    ids = [r[0] for r in con.execute("SELECT order_id FROM orders WHERE status='delivered'")]
    con.close()
    return next(i for i in ids if tools.check_return(i)["eligible"])


def test_tool_schemas_are_openai_format() -> None:
    names = {s["function"]["name"] for s in tool_schemas()}
    assert names == {"get_order_status", "create_return", "search_products"}
    params = next(s for s in tool_schemas() if s["function"]["name"] == "create_return")["function"]["parameters"]
    assert set(params["required"]) == {"order_id", "reason"}


def test_tools(tools: MarketTools, market: dict[str, Any]) -> None:
    oid = _order(market["db"], "in_transit")
    res = tools.call("get_order_status", {"order_id": oid})
    assert res["status"] == "in_transit" and res["status_ru"] == "в пути"
    assert tools.call("get_order_status", {"order_id": "999999"})["error"] == "not_found"
    with pytest.raises(ToolError):
        tools.call("get_order_status", {"order_id": "12"})
    with pytest.raises(ToolError):
        tools.call("rm_rf", {})

    found = tools.call("search_products", {"query": market["products"][0]["brand"]})
    assert found["items"] and all(it["price"] > 0 for it in found["items"])
    capped = tools.call("search_products", {"query": market["products"][0]["brand"], "max_price": 1})
    assert capped["items"] == []

    not_delivered = tools.call("create_return", {"order_id": oid, "reason": "передумал"})
    assert not_delivered["reason"] == "not_delivered"
    ok_id = _eligible_order(tools)
    created = tools.call("create_return", {"order_id": ok_id, "reason": "передумал"})
    assert created["status"] == "created"
    assert tools.call("create_return", {"order_id": ok_id, "reason": "ещё раз"})["reason"] == "already_requested"


def test_dry_run_does_not_persist(market: dict[str, Any]) -> None:
    from market_assistant.data.catalog import REFERENCE_DATE

    dry = MarketTools(market["db"], REFERENCE_DATE, dry_run=True)
    oid = _eligible_order(dry)
    dry.call("create_return", {"order_id": oid, "reason": "x y"})
    assert dry.check_return(oid)["eligible"]


def test_parse_hermes_tool_calls() -> None:
    r = parse_hermes_tool_calls(
        'Сейчас гляну\n<tool_call>\n{"name": "get_order_status", "arguments": {"order_id": "104562"}}\n</tool_call>'
    )
    assert r.content == "Сейчас гляну"
    assert r.tool_calls[0].name == "get_order_status"
    assert json.loads(r.tool_calls[0].arguments) == {"order_id": "104562"}
    assert parse_hermes_tool_calls("<tool_call>{broken</tool_call>").tool_calls == []  # no closing brace match
    assert parse_hermes_tool_calls("<tool_call>{bad json}</tool_call>").tool_calls[0].name == "__invalid__"


def call(name: str, **args: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name, json.dumps(args, ensure_ascii=False))])


def text(content: str) -> LLMResponse:
    return LLMResponse(content=content)


class FixedRouter:
    def __init__(self, intent: str) -> None:
        self.intent = intent

    def __call__(self, text: str) -> RouteDecision:
        return RouteDecision(self.intent, 0.99, "model")


def test_order_status_flow(tools: MarketTools, market: dict[str, Any]) -> None:
    oid = _order(market["db"], "in_transit")
    llm = ScriptedClient([call("get_order_status", order_id=oid), text(f"Заказ {oid} в пути 👀")])
    agent = Agent(llm, tools, AgentConfig(), router=FixedRouter("order_status"))
    res = agent.handle(Session(), f"где заказ {oid}?")
    assert res.reply == f"Заказ {oid} в пути 👀"
    tool_ev = next(e for e in res.events if e["type"] == "tool")
    assert tool_ev["result"]["status"] == "in_transit"
    # the tool result was passed back to the model, tools were offered
    second = llm.requests[1]
    assert second["messages"][-1]["role"] == "tool" and second["tools"]


def test_return_requires_confirmation(tools: MarketTools) -> None:
    oid = _eligible_order(tools)
    llm = ScriptedClient([call("create_return", order_id=oid, reason="не подошёл размер"), text("Готово ✅")])
    agent = Agent(llm, tools, AgentConfig(), router=FixedRouter("return"))
    session = Session()
    first = agent.handle(session, f"хочу вернуть {oid}, не подошёл размер")
    assert first.reply == CANNED["confirm"]["ru"].format(order_id=oid)
    assert session.pending is not None and tools.check_return(oid)["eligible"]  # nothing executed yet
    second = agent.handle(session, "да")
    assert second.reply == "Готово ✅"
    assert tools.check_return(oid)["reason"] == "already_requested"


def test_model_may_execute_after_its_own_confirmation_question(tools: MarketTools) -> None:
    oid = _eligible_order(tools)
    llm = ScriptedClient([text(f"Оформить возврат по заказу {oid}?"),
                          call("create_return", order_id=oid, reason="передумал"), text("Готово ✅")])  # fmt: skip
    agent = Agent(llm, tools, AgentConfig(), router=FixedRouter("return"))
    session = Session()
    agent.handle(session, f"верните {oid}, передумал")
    assert agent.handle(session, "да, оформляй").reply == "Готово ✅"
    assert session.pending is None


def test_return_declined(tools: MarketTools) -> None:
    oid = _eligible_order(tools)
    llm = ScriptedClient([call("create_return", order_id=oid, reason="передумал")])
    agent = Agent(llm, tools, AgentConfig(), router=FixedRouter("return"))
    session = Session()
    agent.handle(session, f"вернуть {oid}")
    assert agent.handle(session, "нет, не надо").reply == CANNED["declined"]["ru"]
    assert tools.check_return(oid)["eligible"]


def test_handoff_skips_llm(tools: MarketTools) -> None:
    llm = ScriptedClient([])
    res = Agent(llm, tools, AgentConfig(), router=FixedRouter("human_handoff")).handle(
        Session(), "operatorni chaqiring"
    )
    assert res.handoff and res.reply == CANNED["handoff"]["uz"] and llm.requests == []


def test_invalid_tool_call_is_reported_to_model(tools: MarketTools) -> None:
    llm = ScriptedClient([call("get_order_status", order_id="12"), text("Уточни номер заказа?")])
    res = Agent(llm, tools, AgentConfig(), router=FixedRouter("order_status")).handle(Session(), "где заказ 12")
    assert res.reply == "Уточни номер заказа?"
    tool_msg = llm.requests[1]["messages"][-1]
    assert json.loads(tool_msg["content"])["error"] == "invalid_call"


def test_max_steps_falls_back_to_operator(tools: MarketTools) -> None:
    llm = ScriptedClient([call("search_products", query="x")] * 5)
    res = Agent(llm, tools, AgentConfig(max_steps=2), router=FixedRouter("product_question")).handle(Session(), "x")
    assert res.handoff and len(llm.requests) == 2


def test_llm_crash_falls_back(tools: MarketTools) -> None:
    class Broken:
        def chat(self, *a: Any, **k: Any) -> LLMResponse:
            raise ConnectionError("vLLM down")

    res = Agent(Broken(), tools, AgentConfig(), router=FixedRouter("faq")).handle(Session(), "Как оформить рассрочку?")
    assert res.handoff and res.reply == CANNED["fallback"]["ru"]


def test_chitchat_has_no_tools_and_faq_gets_context(tools: MarketTools) -> None:
    class Retriever:
        def search(self, q: str) -> list[dict[str, Any]]:
            return [{"doc_id": "installments-ru", "title": "Рассрочка", "text": "На 3, 6, 12 месяцев."}]

    llm = ScriptedClient([text("Salom!"), text("На 3, 6 или 12 месяцев 🙂")])
    Agent(llm, tools, AgentConfig(), router=FixedRouter("chitchat")).handle(Session(), "привет")
    assert llm.requests[0]["tools"] is None
    res = Agent(llm, tools, AgentConfig(), router=FixedRouter("faq"), retriever=Retriever()).handle(
        Session(), "рассрочка на сколько?"
    )
    assert "12 месяцев" in llm.requests[1]["messages"][0]["content"]
    assert any(e["type"] == "retrieve" for e in res.events)


def test_history_is_capped(tools: MarketTools) -> None:
    llm = ScriptedClient([text("ok")] * 30)
    agent = Agent(llm, tools, AgentConfig(), router=FixedRouter("chitchat"), max_history=3)
    session = Session()
    for i in range(10):
        agent.handle(session, f"привет {i}")
    assert len(session.history) == 6


def test_regression_gate_catches_family_drop_hidden_by_overall() -> None:
    from market_assistant.eval.regression_eval import regressions

    base = {"sorting": 0.267, "tool_call": 0.5, "overall": 0.536}
    tuned = {"sorting": 0.067, "tool_call": 1.0, "overall": 0.691}
    failures = regressions(base, tuned, max_drop=0.02, max_family_drop=0.10)
    assert len(failures) == 1 and failures[0].startswith("sorting")
    assert regressions(base, base, 0.02, 0.10) == []


def test_operator_request_overrides_unsure_model(tools: MarketTools) -> None:
    llm = ScriptedClient([])
    agent = Agent(llm, tools, AgentConfig(), router=FixedRouter("faq"))  # model got it wrong
    res = agent.handle(Session(), "Позовите, пожалуйста, оператора")
    assert res.handoff and llm.requests == []
    assert res.events[0] == {"type": "route", "intent": "human_handoff", "confidence": 1.0, "source": "rule"}


def test_order_dates_match_status(market: dict[str, Any]) -> None:
    from market_assistant.data.catalog import REFERENCE_DATE

    con = sqlite3.connect(market["db"])
    rows = con.execute("SELECT status, eta, delivered_at FROM orders").fetchall()
    con.close()
    ref = REFERENCE_DATE.isoformat()
    for status, eta, delivered in rows:
        if status in {"created", "paid", "assembling", "in_transit"}:
            assert eta >= ref, (status, eta)
        if status == "delivered":
            assert delivered and delivered <= ref
