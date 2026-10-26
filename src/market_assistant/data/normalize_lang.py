"""Cheap language detection for ru / uz / mixed messages."""

from __future__ import annotations

_UZ_MARKERS = ("salom", "rahmat", "buyurtma", "qanday", "qachon", "bormi", "kerak", "o'", "g'", "yo'q", "ha ")


def detect_lang(text: str) -> str:
    """``ru`` if the text is mostly Cyrillic (Uzbek words aside), else ``uz``."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "ru"
    cyr = sum("Ѐ" <= ch <= "ӿ" for ch in letters) / len(letters)
    if cyr > 0.5:
        return "ru"
    lower = f"{text.lower()} "
    return "uz" if any(m in lower for m in _UZ_MARKERS) or cyr < 0.1 else "ru"
