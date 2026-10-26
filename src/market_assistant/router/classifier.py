"""Intent router: ruBERT-tiny2 classifier with a keyword fallback."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

# Paths the gateway may take without the full RAG + agent pipeline.
SHORT_PATH_INTENTS = {"chitchat"}
HANDOFF_INTENT = "human_handoff"


@dataclass
class RouteDecision:
    """Router output."""

    intent: str
    confidence: float
    source: str  # "model" | "keywords"


class Router(Protocol):
    """Anything that maps a user message to an intent."""

    def __call__(self, text: str) -> RouteDecision: ...


_HANDOFF_RE = re.compile(r"оператор|живым человеком|живой человек|operator|tirik odam|odam bilan|жалоб", re.I)


def wants_operator(text: str) -> bool:
    """Explicit request for a human. Checked before the model: a missed handoff is the costliest error."""
    return bool(_HANDOFF_RE.search(text))


_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    ("human_handoff", _HANDOFF_RE),
    ("return", re.compile(r"возврат|вернуть|qaytar", re.I)),
    ("order_status", re.compile(r"заказ|посылк|buyurtma|posilka", re.I)),
    ("product_question", re.compile(r"есть ли|ищу|покажи|купить|bormi|kerak|sotib ol", re.I)),
    ("chitchat", re.compile(r"^(привет|salom|спасибо|rahmat|как дела|qalaysiz)\W*$", re.I)),
]


class KeywordRouter:
    """Rule-based fallback when no trained model is available."""

    def __call__(self, text: str) -> RouteDecision:
        for intent, pattern in _KEYWORDS:
            if pattern.search(text):
                return RouteDecision(intent, 0.5, "keywords")
        return RouteDecision("faq", 0.3, "keywords")


class IntentRouter:
    """Fine-tuned sequence classifier (CPU-friendly, ~5–10 ms per message)."""

    def __init__(self, model_dir: str | Path, threshold: float, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
        self.device = device
        self.threshold = threshold

    def __call__(self, text: str) -> RouteDecision:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(self.device)
        with self._torch.inference_mode():
            probs = self.model(**enc).logits.softmax(-1)[0]
        conf, idx = probs.max(-1)
        intent = self.model.config.id2label[int(idx)]
        if float(conf) < self.threshold:
            # unsure -> take the full pipeline (RAG + agent) rather than a shortcut
            intent = "faq"
        return RouteDecision(intent, float(conf), "model")


def load_router(model_dir: str | Path, threshold: float) -> Router:
    """Trained router if present, else the keyword fallback."""
    if (Path(model_dir) / "config.json").exists():
        return IntentRouter(model_dir, threshold)
    logger.warning("No router model at %s — using keyword fallback", model_dir)
    return KeywordRouter()
