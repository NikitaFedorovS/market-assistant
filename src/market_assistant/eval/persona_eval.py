"""Persona quality: base vs LoRA on held-out dialogs.

Two scorers:
- ``heuristic`` (offline proxy): persona markers — Uzbek interjections in Russian replies, informal
  «ты», ≤1 emoji, brevity — plus "no invented numbers" w.r.t. the reference.
- ``judge`` (optional): an OpenAI-compatible LLM rates style / coherence / factuality 1–5 and picks
  the better of the two answers (win-rate). LLM-judge bias is known: back it with a blind human sample.

CLI: ``python -m market_assistant.eval.persona_eval --backend transformers --adapter models/persona-lora``
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import typer

from market_assistant.config import load_config
from market_assistant.eval.common import build_systems, markdown_table
from market_assistant.llm.client import ChatClient
from market_assistant.utils import read_jsonl, setup_logging, write_json, write_jsonl

logger = logging.getLogger(__name__)

UZ_WORDS = re.compile(r"\b(salom|rahmat|hop|zo'r|xo'p|bir daqiqa|arzimaydi)\b", re.I)
TY_FORM = re.compile(r"\b(ты|тебя|тебе|твой|твоя|твои|подскажи|напиши|проверь|подтверждаешь)\b", re.I)
VY_FORM = re.compile(r"\b(вы|вас|вам|ваш|ваша|ваши|подскажите|напишите|проверьте)\b", re.I)
EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")
NUM = re.compile(r"\d+")


def heuristic_style(reply: str, reference: str, lang: str) -> dict[str, float]:
    """Proxy persona scores in [0, 1]."""
    sentences = [s for s in re.split(r"[.!?]+", reply) if s.strip()]
    ru_like = lang != "uz"
    s = {
        "uz_flavour": float(bool(UZ_WORDS.search(reply))) if ru_like else 1.0,
        "informal": float(bool(TY_FORM.search(reply)) or not VY_FORM.search(reply)) if ru_like else 1.0,
        "emoji_ok": float(len(EMOJI.findall(reply)) <= 1),
        "brevity": float(1 <= len(sentences) <= 4),
        "no_invented_numbers": float(set(NUM.findall(reply)) <= set(NUM.findall(reference)) | {"1", "2"}),
    }
    s["style_score"] = sum(s.values()) / len(s)
    return s


JUDGE_PROMPT = """Ты оцениваешь ответы ассистента маркетплейса с персоной Dilkash (тепло, на «ты», узбекские словечки,
коротко, без выдуманных фактов). Контекст диалога и эталонный ответ даны для проверки фактов.

Диалог:
{dialog}

Эталон: {reference}
Ответ A: {a}
Ответ B: {b}

Верни JSON: {{"a": {{"style": 1-5, "coherence": 1-5, "facts": 1-5}}, "b": {{...}}, "winner": "a"|"b"|"tie"}}"""


def judge_pair(judge: ChatClient, dialog: list[dict[str, Any]], reference: str, a: str, b: str) -> dict[str, Any]:
    """Ask the judge LLM to score two answers (order randomised by the caller)."""
    text = "\n".join(f"{m['role']}: {m.get('content') or ''}" for m in dialog if m["role"] in {"user", "assistant"})
    prompt = JUDGE_PROMPT.format(dialog=text, reference=reference, a=a, b=b)
    resp = judge.chat([{"role": "user", "content": prompt}])
    match = re.search(r"\{.*\}", resp.content, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def eval_items(path: Path, limit: int) -> list[dict[str, Any]]:
    """Held-out domain dialogs cut right before the first *text* assistant reply."""
    items = []
    for rec in read_jsonl(path):
        if rec["source"] != "domain":
            continue
        msgs = rec["messages"]
        idx = next((i for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("content")), None)
        if idx is None:
            continue
        items.append({"id": rec["id"], "lang": rec["lang"], "intent": rec["intent"], "prompt": msgs[:idx],
                      "reference": msgs[idx]["content"]})  # fmt: skip
        if len(items) >= limit:
            break
    return items


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config_dir: str = typer.Option("configs", help="Directory with YAML configs"),
    backend: str = typer.Option("transformers", help="transformers | openai"),
    adapter: str | None = typer.Option(None, help="LoRA adapter dir (transformers backend)"),
    base_model_name: str | None = typer.Option(None, help="Base model name on the vLLM server (openai backend)"),
    limit: int = typer.Option(100, help="Number of held-out dialogs"),
    judge_base_url: str | None = typer.Option(None, help="OpenAI-compatible judge endpoint (optional)"),
    judge_model: str | None = typer.Option(None, help="Judge model name"),
) -> None:
    """Generate with base and fine-tuned models and score the persona."""
    import random

    from market_assistant.llm.client import OpenAIChatClient

    setup_logging()
    cfg = load_config(config_dir)
    items = eval_items(cfg.data.processed_dir / "sft_test.jsonl", limit)
    systems = build_systems(cfg.serve.llm, adapter, backend, base_model_name)
    outputs: dict[str, list[str]] = {name: [client.chat(it["prompt"]).content for it in items]
                                     for name, client in systems.items()}  # fmt: skip

    summary: dict[str, Any] = {}
    for name, replies in outputs.items():
        scores = [heuristic_style(r, it["reference"], it["lang"]) for r, it in zip(replies, items, strict=True)]
        summary[name] = {k: sum(s[k] for s in scores) / len(scores) for k in scores[0]}

    if judge_base_url and judge_model and {"base", "finetuned"} <= outputs.keys():
        judge = OpenAIChatClient(cfg.serve.llm.model_copy(update={"base_url": judge_base_url, "model": judge_model}))
        rng, wins, judged = random.Random(0), {"base": 0, "finetuned": 0, "tie": 0}, 0
        for i, it in enumerate(items):
            pair = [("base", outputs["base"][i]), ("finetuned", outputs["finetuned"][i])]
            rng.shuffle(pair)
            verdict = judge_pair(judge, it["prompt"], it["reference"], pair[0][1], pair[1][1])
            winner = verdict.get("winner")
            wins[{"a": pair[0][0], "b": pair[1][0]}.get(winner, "tie")] += 1
            judged += 1
        summary["judge_win_rate_finetuned"] = wins["finetuned"] / max(judged, 1)
        summary["judge_counts"] = wins

    out = Path("results")
    write_json(out / "persona_eval.json", summary)
    write_jsonl(out / "persona_outputs.jsonl", [
        {"id": it["id"], "lang": it["lang"], "reference": it["reference"], **{n: o[i] for n, o in outputs.items()}}
        for i, it in enumerate(items)
    ])  # fmt: skip
    metric_names = [k for k in next(iter(summary.values())) if isinstance(k, str)]
    rows = [{"metric": m, **{n: summary[n][m] for n in outputs}} for m in metric_names]
    table = markdown_table(rows, ["metric", *outputs])
    (out / "persona_eval.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table + "\n")


if __name__ == "__main__":
    app()
