"""Regression suite: did the persona LoRA hurt general skills or tool calling?

Task families (held out — generated with a different seed than the training mix):
arithmetic, word problems, sorting, JSON extraction, short-answer instruction following, and
tool-call correctness on unseen domain prompts. Scored for the base and the fine-tuned model.

CLI: ``python -m market_assistant.eval.regression_eval --backend transformers --adapter models/persona-lora``
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from market_assistant.agent.tools import tool_schemas
from market_assistant.config import load_config
from market_assistant.data.persona import SYSTEM_PROMPT
from market_assistant.eval.common import build_systems, markdown_table
from market_assistant.llm.client import ChatClient, LLMResponse
from market_assistant.utils import setup_logging, write_json

logger = logging.getLogger(__name__)

Check = Callable[[LLMResponse], bool]


def _numbers(text: str) -> list[int]:
    return [int(n.replace(" ", "")) for n in re.findall(r"-?\d[\d ]*\d|-?\d", text)]


def build_tasks(n_per_family: int = 20, seed: int = 1234) -> list[dict[str, Any]]:
    """Deterministic task list: ``{family, messages, tools, check}``."""
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    general = [{"role": "system", "content": "You are a helpful assistant."}]

    def add(family: str, prompt: str, check: Check, system: list[dict[str, Any]] | None = None,
            tools: bool = False) -> None:  # fmt: skip
        tasks.append({"family": family, "messages": [*(system or general), {"role": "user", "content": prompt}],
                      "tools": tool_schemas() if tools else None, "check": check})  # fmt: skip

    for _ in range(n_per_family):
        a, b = rng.randint(11, 97), rng.randint(11, 97)
        add("arithmetic", f"Сколько будет {a} × {b}? Ответь числом.", lambda r, v=a * b: v in _numbers(r.content))

        x, y, z = rng.randint(20, 60), rng.randint(2, 5), rng.randint(2, 6)
        add("word_problem", f"В коробке {x} конфет. {y} детей взяли по {z}. Сколько осталось? Дай итоговое число.",
            lambda r, v=x - y * z: bool(_numbers(r.content)) and _numbers(r.content)[-1] == v)  # fmt: skip

        nums = rng.sample(range(1, 200), 5)
        target = ", ".join(map(str, sorted(nums)))
        add("sorting", f"Отсортируй по возрастанию и выведи через запятую: {', '.join(map(str, nums))}",
            lambda r, t=target: t in r.content.replace("，", ","))  # fmt: skip

        name, age = rng.choice(["Азиз", "Лола", "Сардор", "Камила"]), rng.randint(18, 80)

        def json_ok(r: LLMResponse, n: str = name, a_: int = age) -> bool:
            m = re.search(r"\{.*\}", r.content, re.DOTALL)
            try:
                obj = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                return False
            return obj.get("name") == n and int(obj.get("age", -1)) == a_

        add("json", f"Верни только JSON с полями name и age: «Меня зовут {name}, мне {age} лет».", json_ok)

    capitals = [("Узбекистана", "Ташкент"), ("Казахстана", "Астана"), ("Франции", "Париж"), ("Японии", "Токио"),
                ("Италии", "Рим")]  # fmt: skip
    for country, capital in capitals * (n_per_family // len(capitals) + 1):
        add("instruction", f"Ответь одним словом: столица {country}?",
            lambda r, c=capital: c.lower() in r.content.lower() and len(r.content.split()) <= 3)  # fmt: skip

    persona_sys = [{"role": "system", "content": SYSTEM_PROMPT}]
    for _ in range(n_per_family):
        oid = str(rng.randint(100000, 120000))
        add("tool_call", rng.choice([f"подскажи, где заказ {oid}", f"{oid} buyurtma qayerda?", f"статус {oid}?"]),
            lambda r, o=oid: any(c.name == "get_order_status" and json.loads(c.arguments or "{}").get("order_id") == o
                                 for c in r.tool_calls),
            system=persona_sys, tools=True)  # fmt: skip
        q = rng.choice(["есть наушники JBL?", "ищу кроссовки Nike до 1 500 000 сум", "Samsung televizor bormi?"])
        add("tool_call", q, lambda r: any(c.name == "search_products" for c in r.tool_calls), system=persona_sys,
            tools=True)  # fmt: skip
    return tasks


def score(client: ChatClient, tasks: list[dict[str, Any]]) -> dict[str, float]:
    """Accuracy per family + overall."""
    hits: dict[str, list[bool]] = defaultdict(list)
    for t in tasks:
        try:
            ok = bool(t["check"](client.chat(t["messages"], t["tools"])))
        except Exception:
            ok = False
        hits[t["family"]].append(ok)
    acc = {fam: sum(v) / len(v) for fam, v in hits.items()}
    acc["overall"] = sum(sum(v) for v in hits.values()) / sum(len(v) for v in hits.values())
    return acc


def regressions(base: dict[str, float], tuned: dict[str, float], max_drop: float,
                max_family_drop: float) -> list[str]:  # fmt: skip
    """Human-readable list of gate violations (empty = passed)."""
    out = []
    for family, b in base.items():
        limit = max_drop if family == "overall" else max_family_drop
        if b - tuned.get(family, 0.0) > limit:
            out.append(f"{family}: {b:.3f} -> {tuned.get(family, 0.0):.3f} (allowed drop {limit:.0%})")
    return out


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config_dir: str = typer.Option("configs", help="Directory with YAML configs"),
    backend: str = typer.Option("transformers", help="transformers | openai"),
    adapter: str | None = typer.Option(None, help="LoRA adapter dir (transformers backend)"),
    base_model_name: str | None = typer.Option(None, help="Base model name on the vLLM server (openai backend)"),
    n_per_family: int = typer.Option(20, help="Tasks per family"),
    max_drop: float = typer.Option(0.02, help="Allowed overall accuracy drop before failing"),
    max_family_drop: float = typer.Option(0.10, help="Allowed drop in any single task family before failing"),
) -> None:
    """Score base vs fine-tuned; exit 1 on an overall drop > ``max_drop`` or any family drop > ``max_family_drop``.

    The per-family gate matters: gains in one family (e.g. tool calling) can hide a regression in another
    (e.g. sorting) inside the overall number.
    """
    setup_logging()
    cfg = load_config(config_dir)
    tasks = build_tasks(n_per_family)
    results = {name: score(client, tasks) for name, client in build_systems(cfg.serve.llm, adapter, backend,
                                                                             base_model_name).items()}  # fmt: skip
    families = list(next(iter(results.values())))
    rows = [{"family": f, **{name: res[f] for name, res in results.items()}} for f in families]
    table = markdown_table(rows, ["family", *results])
    write_json(Path("results") / "regression_eval.json", results)
    Path("results/regression_eval.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table + "\n")
    if {"base", "finetuned"} <= results.keys():
        failures = regressions(results["base"], results["finetuned"], max_drop, max_family_drop)
        write_json(Path("results") / "regression_gate.json", {"passed": not failures, "failures": failures})
        for f in failures:
            print(f"REGRESSION: {f}")
        if failures:
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
