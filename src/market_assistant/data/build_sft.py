"""Build the SFT dataset for the persona LoRA.

raw dialogs -> persona stylization -> PII masking -> filters -> near-dup removal
-> system prompt (+ RAG context) -> general-instruction mix -> train/val/test JSONL.

CLI: ``python -m market_assistant.data.build_sft``
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

import typer

from market_assistant.agent.tools import TOOL_ARGS
from market_assistant.config import DataConfig, load_config
from market_assistant.data.persona import SYSTEM_PROMPT, TeacherStylizer, build_system_prompt, stylize_dialog
from market_assistant.data.pii import mask_pii
from market_assistant.data.policies import RETURN_DAYS
from market_assistant.utils import read_jsonl, setup_logging, write_json, write_jsonl

logger = logging.getLogger(__name__)

LOW_INFO = {"ок", "ок.", "ok", "ok.", "минуту", "минуту.", "да", "нет", "+", "xo'p"}
TOXIC_WORDS = re.compile(r"\b(туп\w*|идиот\w*|дебил\w*|ahmoq\w*|tentak\w*)\b", re.IGNORECASE)
RETURN_PERIOD_RE = re.compile(r"(?:вернуть|возврат\w*|qaytar\w*)[^.!?\n]*?(\d+)\s*(?:дн|kun)", re.IGNORECASE)
GENERAL_SYSTEM = "You are a helpful assistant."


# --------------------------------------------------------------------------- filters


def broken_tool_call(dialog: dict[str, Any]) -> bool:
    """True if any tool call has invalid JSON, unknown tool or arguments failing the schema."""
    for m in dialog["messages"]:
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            if fn["name"] not in TOOL_ARGS:
                return True
            try:
                TOOL_ARGS[fn["name"]].model_validate(json.loads(fn["arguments"]))
            except (json.JSONDecodeError, ValueError):
                return True
    return False


def contradicts_policy(text: str) -> bool:
    """True if a return-period number in ``text`` differs from the policy."""
    return any(int(n) != RETURN_DAYS for n in RETURN_PERIOD_RE.findall(text))


def drop_reason(dialog: dict[str, Any], cfg: DataConfig) -> str | None:
    """Return the first filter that rejects the dialog, or ``None`` if it is clean."""
    if broken_tool_call(dialog):
        return "broken_tool_call"
    texts = [m.get("content") or "" for m in dialog["messages"] if m["role"] in {"user", "assistant"}]
    if any(TOXIC_WORDS.search(t) for t in texts):
        return "toxic"
    replies = [m["content"] for m in dialog["messages"] if m["role"] == "assistant" and m.get("content")]
    if not replies:
        return "no_reply"
    for r in replies:
        if r.strip().lower() in LOW_INFO or len(r.strip()) < cfg.sft.min_chars:
            return "low_info"
        if len(r) > cfg.sft.max_chars:
            return "too_long"
        if contradicts_policy(r):
            return "contradiction"
    return None


# --------------------------------------------------------------------------- near-dup (MinHash + LSH)


class MinHashDeduper:
    """Near-duplicate detector on character 5-gram shingles (MinHash, 64 perms, 16 LSH bands)."""

    def __init__(self, threshold: float, num_perm: int = 64, bands: int = 16, seed: int = 1) -> None:
        self.threshold = threshold
        self.num_perm, self.bands, self.rows = num_perm, bands, num_perm // bands
        rng = random.Random(seed)
        self._salts = [rng.getrandbits(64).to_bytes(8, "little") for _ in range(num_perm)]
        self._buckets: dict[tuple[int, bytes], list[int]] = {}
        self._signatures: list[list[int]] = []

    def _signature(self, text: str) -> list[int]:
        shingles = {text[i : i + 5] for i in range(max(1, len(text) - 4))}
        return [
            min(int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8, key=salt).digest(), "little")
                for s in shingles)
            for salt in self._salts
        ]  # fmt: skip

    def is_duplicate(self, text: str) -> bool:
        """Check ``text`` against everything seen so far; remember it if new."""
        sig = self._signature(text)
        candidates: set[int] = set()
        keys = []
        for b in range(self.bands):
            key = (b, repr(sig[b * self.rows : (b + 1) * self.rows]).encode())
            keys.append(key)
            candidates.update(self._buckets.get(key, []))
        for c in candidates:
            sim = sum(x == y for x, y in zip(sig, self._signatures[c], strict=True)) / self.num_perm
            if sim >= self.threshold:
                return True
        idx = len(self._signatures)
        self._signatures.append(sig)
        for key in keys:
            self._buckets.setdefault(key, []).append(idx)
        return False


def dialog_fingerprint(dialog: dict[str, Any]) -> str:
    """Text used for dedup: all turns incl. tool calls/results, normalized."""
    parts = []
    for m in dialog["messages"]:
        parts.append(m.get("content") or "")
        parts += [tc["function"]["arguments"] for tc in m.get("tool_calls") or []]
    return re.sub(r"\s+", " ", " ".join(parts).lower())


# --------------------------------------------------------------------------- general mix


def builtin_general_examples(n: int, seed: int) -> list[dict[str, Any]]:
    """Small procedural set of general tasks (arithmetic, reasoning, formatting, JSON extraction).

    A stand-in for a real instruct dataset (``sft.general_dataset``): its job is to keep the model's
    general skills exercised so the persona LoRA doesn't erode them.
    """
    rng = random.Random(seed)
    names = ["Азиз", "Мадина", "Тимур", "Нигора", "Олег", "Дильноза"]
    items = ["яблок", "книг", "ручек", "тетрадей"]
    out = []
    for i in range(n):
        kind = i % 5
        if kind == 0:
            a, b = rng.randint(12, 99), rng.randint(12, 99)
            q, ans = f"Сколько будет {a} × {b}?", f"{a} × {b} = {a * b}."
        elif kind == 1:
            name, item = rng.choice(names), rng.choice(items)
            x, y, z = rng.randint(5, 30), rng.randint(1, 4), rng.randint(1, 5)
            q = f"У {name} было {x} {item}. Каждый из {y} друзей взял по {z}. Сколько осталось?"
            ans = f"Друзья взяли {y} × {z} = {y * z}. Осталось {x} − {y * z} = {x - y * z}."
        elif kind == 2:
            price, disc = rng.randrange(100_000, 900_000, 10_000), rng.choice([10, 15, 20, 25])
            q = f"Товар стоит {price} сум, скидка {disc}%. Какая итоговая цена?"
            ans = f"Скидка: {price} × {disc}% = {price * disc // 100} сум. Итог: {price - price * disc // 100} сум."
        elif kind == 3:
            nums = rng.sample(range(1, 100), 6)
            q = f"Отсортируй по возрастанию: {', '.join(map(str, nums))}"
            ans = ", ".join(map(str, sorted(nums)))
        else:
            name, city, age = rng.choice(names), rng.choice(["Ташкент", "Самарканд", "Бухара"]), rng.randint(18, 70)
            q = f"Извлеки JSON с полями name, city, age: «{name}, {age} лет, живёт в городе {city}.»"
            ans = json.dumps({"name": name, "city": city, "age": age}, ensure_ascii=False)
        out.append({
            "id": f"g{i:05d}", "intent": "general", "lang": "ru", "source": "general",
            "messages": [{"role": "system", "content": GENERAL_SYSTEM}, {"role": "user", "content": q},
                         {"role": "assistant", "content": ans}],
        })  # fmt: skip
    return out


def hf_general_examples(name: str, n: int, seed: int) -> list[dict[str, Any]]:
    """Sample ``n`` chats from a HF dataset with a ``messages`` column."""
    from datasets import load_dataset

    ds = load_dataset(name, split="train").shuffle(seed=seed).select(range(n))
    return [
        {"id": f"g{i:05d}", "intent": "general", "lang": "mixed", "source": "general", "messages": row["messages"]}
        for i, row in enumerate(ds)
    ]


# --------------------------------------------------------------------------- pipeline


def to_sft_record(dialog: dict[str, Any], kb_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Prepend the persona system prompt (with RAG context for FAQ dialogs)."""
    context = [kb_by_id[i] for i in dialog.get("context_doc_ids", []) if i in kb_by_id]
    system = {"role": "system", "content": build_system_prompt(context) if context else SYSTEM_PROMPT}
    return {
        "id": dialog["id"],
        "intent": dialog["intent"],
        "lang": dialog["lang"],
        "source": "domain",
        "uses_tools": any(m.get("tool_calls") for m in dialog["messages"]),
        "messages": [system, *dialog["messages"]],
    }


def clean_dialogs(
    dialogs: Iterable[dict[str, Any]], cfg: DataConfig, teacher: TeacherStylizer | None = None
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Stylize, mask PII, filter and dedup. Returns clean dialogs and drop statistics."""
    stats: Counter[str] = Counter()
    dedup = MinHashDeduper(cfg.sft.dedup_threshold)
    out = []
    for d in dialogs:
        stats["total"] += 1
        d = stylize_dialog(d, teacher)
        for m in d["messages"]:
            if m["role"] in {"user", "assistant"} and m.get("content"):
                m["content"], found = mask_pii(m["content"])
                stats["pii_masked"] += len(found)
        reason = drop_reason(d, cfg)
        if reason is None and dedup.is_duplicate(dialog_fingerprint(d)):
            reason = "near_duplicate"
        if reason:
            stats[f"drop_{reason}"] += 1
            continue
        out.append(d)
    stats["kept"] = len(out)
    return out, stats


def split_records(records: list[dict[str, Any]], ratios: tuple[float, float, float],
                  seed: int) -> dict[str, list[dict[str, Any]]]:  # fmt: skip
    """Deterministic shuffle + split."""
    recs = records.copy()
    random.Random(seed).shuffle(recs)
    n_val, n_test = int(len(recs) * ratios[1]), int(len(recs) * ratios[2])
    return {"test": recs[:n_test], "val": recs[n_test : n_test + n_val], "train": recs[n_test + n_val :]}


def build(cfg: DataConfig) -> dict[str, Any]:
    """Run the pipeline; writes ``processed/sft_{train,val,test}.jsonl`` and ``sft_stats.json``."""
    sc = cfg.sft
    teacher = (
        TeacherStylizer(sc.teacher_base_url, sc.teacher_model, cfg.persona_path)
        if sc.teacher_base_url and sc.teacher_model
        else None
    )
    kb_by_id = {a["doc_id"]: a for a in read_jsonl(cfg.kb_path)}
    dialogs, stats = clean_dialogs(read_jsonl(cfg.raw_dir / "dialogs.jsonl"), cfg, teacher)
    domain = [to_sft_record(d, kb_by_id) for d in dialogs]

    n_general = round(len(domain) * sc.general_mix_ratio / (1 - sc.general_mix_ratio))
    n_general = min(n_general, sc.general_max_samples) if sc.general_max_samples else n_general
    general = (
        hf_general_examples(sc.general_dataset, n_general, sc.seed)
        if sc.general_dataset
        else builtin_general_examples(n_general, sc.seed)
    )
    splits = split_records(domain + general, sc.splits, sc.seed)
    counts = {name: write_jsonl(cfg.processed_dir / f"sft_{name}.jsonl", recs) for name, recs in splits.items()}
    summary = {
        "cleaning": dict(stats),
        "domain": len(domain),
        "general": len(general),
        "splits": counts,
        "by_intent": dict(Counter(r["intent"] for r in domain + general)),
        "by_lang": dict(Counter(r["lang"] for r in domain)),
        "stylizer": "teacher" if teacher else "rule",
    }
    write_json(cfg.processed_dir / "sft_stats.json", summary)
    logger.info("SFT build: %s", summary)
    return summary


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(config_dir: str = typer.Option("configs", help="Directory with YAML configs")) -> None:
    """Build SFT splits."""
    setup_logging()
    build(load_config(config_dir).data)


if __name__ == "__main__":
    app()
