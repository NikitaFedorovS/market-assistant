"""Router fallback, threshold behaviour and split hygiene."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from market_assistant.data.normalize_lang import detect_lang
from market_assistant.router.classifier import IntentRouter, KeywordRouter, load_router
from market_assistant.train.train_router import split_router_data
from market_assistant.utils import read_jsonl, write_jsonl


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("позовите оператора", "human_handoff"),
        ("хочу вернуть заказ 104562", "return"),
        ("где мой заказ 104562?", "order_status"),
        ("Samsung televizor bormi?", "product_question"),
        ("привет", "chitchat"),
        ("сколько хранится посылка в пункте", "order_status"),
        ("Как оформить рассрочку?", "faq"),
    ],
)
def test_keyword_router(text: str, intent: str) -> None:
    assert KeywordRouter()(text).intent == intent


def test_load_router_falls_back_without_model(tmp_path: Path) -> None:
    assert isinstance(load_router(tmp_path / "nope", 0.6), KeywordRouter)


def test_low_confidence_goes_to_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    class FakeModel:
        config = type("C", (), {"id2label": {0: "chitchat", 1: "faq"}})()

        def __call__(self, **_: Any) -> Any:
            return type("O", (), {"logits": torch.tensor([[0.1, 0.0]])})()

    router = IntentRouter.__new__(IntentRouter)
    router._torch, router.device, router.threshold = torch, "cpu", 0.6
    router.tokenizer = lambda *a, **k: type("E", (dict,), {"to": lambda self, d: self})()
    router.model = FakeModel()
    decision = router("hmm")
    assert decision.intent == "faq" and decision.confidence < 0.6


def test_split_has_no_text_leakage(tmp_path: Path) -> None:
    rows = [{"text": f"фраза {i % 50}", "label": "faq"} for i in range(200)]
    write_jsonl(tmp_path / "router.jsonl", rows)
    split_router_data(tmp_path / "router.jsonl", tmp_path, seed=0)
    parts = {s: {r["text"] for r in read_jsonl(tmp_path / f"router_{s}.jsonl")} for s in ("train", "val", "test")}
    assert sum(len(p) for p in parts.values()) == 50
    assert not (parts["train"] & parts["test"]) and not (parts["train"] & parts["val"])


@pytest.mark.parametrize(
    ("text", "lang"),
    [("где мой заказ?", "ru"), ("buyurtmam qayerda?", "uz"), ("Salom, где заказ?", "ru"), ("12345", "ru")],
)
def test_detect_lang(text: str, lang: str) -> None:
    assert detect_lang(text) == lang
