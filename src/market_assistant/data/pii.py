"""PII masking for support texts (regex; an optional NER model can be plugged in)."""

from __future__ import annotations

import re
from collections.abc import Callable

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    # payment cards: 16 digits, optionally grouped by 4 (Uzcard 8600, Humo 9860, Visa, MC)
    ("CARD", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
    # Uzbek phones: +998 XX XXX XX XX, 998XXXXXXXXX, or local XX XXX XX XX
    ("PHONE", re.compile(r"(?:\+?998[ -]?)?\(?\d{2}\)?[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}\b")),
    ("PASSPORT", re.compile(r"\b[A-Z]{2}\s?\d{7}\b")),
]

NerFn = Callable[[str], list[tuple[int, int, str]]]  # returns (start, end, label) spans


def mask_pii(text: str, ner: NerFn | None = None) -> tuple[str, list[str]]:
    """Replace PII with ``[LABEL]`` placeholders.

    Order numbers (5–7 digits) are intentionally *not* masked — the agent needs them.

    Returns:
        Masked text and the list of labels found.
    """
    found: list[str] = []
    for label, pattern in _PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        found += [label] * n
    if ner is not None:
        for start, end, label in sorted(ner(text), reverse=True):
            text = text[:start] + f"[{label}]" + text[end:]
            found.append(label)
    return text, found


def contains_pii(text: str) -> bool:
    """True if any regex PII pattern matches."""
    return any(p.search(text) for _, p in _PATTERNS)
