"""Synthetic data, PII masking, SFT filters, dedup and the end-to-end data pipeline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from market_assistant.config import DataConfig, load_config
from market_assistant.data.build_sft import (
    MinHashDeduper,
    broken_tool_call,
    build,
    builtin_general_examples,
    contradicts_policy,
    drop_reason,
)
from market_assistant.data.catalog import generate_kb
from market_assistant.data.dialogs import fmt_date, fmt_price
from market_assistant.data.persona import build_system_prompt, stylize_dialog
from market_assistant.data.pii import contains_pii, mask_pii
from market_assistant.data.policies import CATEGORIES, RETURN_DAYS
from market_assistant.data.synth import synthesize
from market_assistant.utils import read_jsonl


def test_kb_covers_regions_categories_and_languages() -> None:
    kb = generate_kb()
    assert {a["lang"] for a in kb} == {"ru", "uz"}
    assert len({a["doc_id"] for a in kb}) == len(kb)
    assert sum(a["topic"] == "return" for a in kb) == 2 * len(CATEGORIES)
    assert all(a["questions"] and a["text"] for a in kb)
    non_ret = next(a for a in kb if a["doc_id"] == "return-food-ru")
    assert "не подлежат" in non_ret["text"]


def test_formatting_helpers() -> None:
    assert fmt_date("2026-10-22", "ru") == "22 октября"
    assert fmt_date("2026-10-22", "uz") == "22-oktabr"
    assert fmt_price(1250000, "ru") == "1 250 000 сум"
    assert fmt_price(1250000, "uz") == "1 250 000 so'm"


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("мой номер +998 90 123 45 67", "PHONE"),
        ("telefonim 99 876 54 32", "PHONE"),
        ("карта 8600 1234 5678 9012", "CARD"),
        ("пишите на test.user@mail.uz", "EMAIL"),
        ("паспорт AB1234567", "PASSPORT"),
    ],
)
def test_mask_pii(text: str, label: str) -> None:
    masked, found = mask_pii(text)
    assert f"[{label}]" in masked and label in found
    assert not contains_pii(masked)


def test_order_numbers_and_prices_are_not_pii() -> None:
    text = "где мой заказ 104562? бюджет 1 500 000 сум"
    assert mask_pii(text) == (text, [])


def _dialog(reply: str, args: str = '{"order_id": "104562"}', user: str = "где заказ 104562?") -> dict[str, Any]:
    return {
        "id": "x",
        "intent": "order_status",
        "lang": "ru",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_order_status", "arguments": args}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "get_order_status", "content": "{}"},
            {"role": "assistant", "content": reply},
        ],
    }  # fmt: skip


def test_filters(data_cfg: DataConfig) -> None:
    assert drop_reason(_dialog("Заказ 104562 в пути, ожидаем 22 октября."), data_cfg) is None
    assert drop_reason(_dialog("Ок."), data_cfg) == "low_info"
    assert drop_reason(_dialog("в пути", args='{"order_id": "1234'), data_cfg) == "broken_tool_call"
    assert drop_reason(_dialog("в пути", args='{"order_id": "12"}'), data_cfg) == "broken_tool_call"  # schema
    assert drop_reason(_dialog("в пути", user="где заказ, тупой бот"), data_cfg) == "toxic"
    bad = "Вернуть товар можно в течение 30 дней."
    assert drop_reason(_dialog(bad), data_cfg) == "contradiction"


def test_contradiction_check_uses_policy() -> None:
    assert not contradicts_policy(f"Товар можно вернуть в течение {RETURN_DAYS} дней после получения.")
    assert contradicts_policy("Возврат возможен в течение 14 дней.")
    assert not contradicts_policy("Деньги вернутся на карту за 3–5 рабочих дня.")


def test_broken_tool_call_unknown_tool() -> None:
    d = _dialog("ok")
    d["messages"][1]["tool_calls"][0]["function"]["name"] = "drop_database"
    assert broken_tool_call(d)


def test_minhash_dedup() -> None:
    dd = MinHashDeduper(threshold=0.9)
    a = "где мой заказ 104562 статус в пути ожидаем 22 октября пункт выдачи номер 12 " * 3
    assert not dd.is_duplicate(a)
    assert dd.is_duplicate(a)
    assert not dd.is_duplicate("как оформить рассрочку на двенадцать месяцев без первого взноса " * 3)


def test_stylize_rule_mode_uses_persona_and_drops_field() -> None:
    d = {"messages": [{"role": "user", "content": "привет"},
                      {"role": "assistant", "content": "Здравствуйте.", "persona": "Salom! 👋"}]}  # fmt: skip
    out = stylize_dialog(d)
    assert out["messages"][1] == {"role": "assistant", "content": "Salom! 👋"}


def test_system_prompt_with_context() -> None:
    p = build_system_prompt([{"title": "Рассрочка", "text": "На 3, 6, 12 месяцев."}])
    assert "Dilkash" in p and "[Рассрочка]" in p and "12 месяцев" in p


def test_general_examples_are_correct() -> None:
    ex = builtin_general_examples(10, seed=0)
    q, a = ex[0]["messages"][1]["content"], ex[0]["messages"][2]["content"]
    x, y = (int(n) for n in q.removeprefix("Сколько будет ").removesuffix("?").split(" × "))
    assert str(x * y) in a
    assert json.loads(ex[4]["messages"][2]["content"])["name"]


def test_pipeline_end_to_end(data_cfg: DataConfig) -> None:
    counts = synthesize(data_cfg)
    assert counts["dialogs"] >= 6 * data_cfg.synth.dialogs_per_intent
    raw = list(read_jsonl(data_cfg.raw_dir / "dialogs.jsonl"))
    assert {d["intent"] for d in raw} == {"order_status", "return", "product_question", "faq", "chitchat",
                                          "human_handoff"}  # fmt: skip

    summary = build(data_cfg)
    assert summary["domain"] > 0 and summary["general"] > 0
    recs = [r for s in ("train", "val", "test") for r in read_jsonl(data_cfg.processed_dir / f"sft_{s}.jsonl")]
    assert len(recs) == sum(summary["splits"].values())
    for r in recs:
        assert r["messages"][0]["role"] == "system"
        for m in r["messages"]:
            assert "persona" not in m
            if m["role"] in {"user", "assistant"} and m.get("content"):
                assert not contains_pii(m["content"]), m["content"]
    noise = Counter(d.get("noise") for d in raw)
    assert summary["cleaning"].get("drop_toxic", 0) >= noise["toxic"]
    assert summary["cleaning"].get("drop_broken_tool_call", 0) >= noise["broken_json"]
    # tool results in training data come from the real tool implementation
    tool_msgs = [m for r in recs for m in r["messages"] if m["role"] == "tool"]
    assert tool_msgs and all(isinstance(json.loads(m["content"]), dict) for m in tool_msgs)


def test_repo_configs_load() -> None:
    cfg = load_config(Path(__file__).parents[1] / "configs")
    assert cfg.train_lora.base_model.startswith("Qwen/")
    assert cfg.serve.agent.confirm_tools == ["create_return"]


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MA__SERVE__LLM__BASE_URL", "http://gpu:9000/v1")
    monkeypatch.setenv("MA__TRAIN_LORA__LORA__R", "8")
    cfg = load_config(Path(__file__).parents[1] / "configs")
    assert cfg.serve.llm.base_url == "http://gpu:9000/v1"
    assert cfg.train_lora.lora.r == 8
