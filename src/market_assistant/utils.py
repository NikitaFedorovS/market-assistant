"""Shared helpers: logging, device selection, JSONL IO, token estimates."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging (level from arg or ``$LOG_LEVEL``)."""
    logging.basicConfig(
        level=(level or os.environ.get("LOG_LEVEL", "INFO")).upper(),
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )


def resolve_device(preference: str = "auto") -> str:
    """cuda > mps > cpu unless a device is given explicitly."""
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records to JSONL (creating parent dirs); returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: str | Path, obj: Any) -> None:
    """Pretty-print JSON to a file (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token for Latin, ~3 for Cyrillic) without a tokenizer."""
    return max(1, len(text) // 3)
