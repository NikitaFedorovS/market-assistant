"""Compare retrieval setups on the KB question set: Recall@1/5 and MRR (document level).

Setups: BM25, dense with each embedder in ``rag.compare_embedders``, hybrid (BM25 + dense, RRF)
and hybrid + reranker with the production embedder.

CLI: ``python -m market_assistant.eval.rag_eval``
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import typer

from market_assistant.config import AppConfig, load_config
from market_assistant.eval.common import markdown_table
from market_assistant.rag.index import Embedder
from market_assistant.rag.retrieve import HybridRetriever, Mode
from market_assistant.utils import read_jsonl, setup_logging, write_json

logger = logging.getLogger(__name__)


def ranking_metrics(ranked_doc_ids: list[list[str]], gold: list[str]) -> dict[str, float]:
    """Recall@1, Recall@5 and MRR@10 for one relevant document per query."""
    r1 = r5 = mrr = 0.0
    for ranked, g in zip(ranked_doc_ids, gold, strict=True):
        dedup = list(dict.fromkeys(ranked))  # several chunks of one doc count once
        if dedup[:1] == [g]:
            r1 += 1
        if g in dedup[:5]:
            r5 += 1
        if g in dedup[:10]:
            mrr += 1 / (dedup.index(g) + 1)
    n = max(len(gold), 1)
    return {"recall@1": r1 / n, "recall@5": r5 / n, "mrr": mrr / n}


def run_setup(retriever: HybridRetriever, questions: list[dict[str, Any]], mode: Mode) -> dict[str, float]:
    t0 = time.perf_counter()
    ranked = [[c["doc_id"] for c in retriever.search(q["question"], mode=mode, top_k=10, budget=False)]
              for q in questions]  # fmt: skip
    metrics = ranking_metrics(ranked, [q["doc_id"] for q in questions])
    metrics["ms_per_query"] = (time.perf_counter() - t0) * 1000 / max(len(questions), 1)
    return metrics


def evaluate(cfg: AppConfig, embedders: list[str] | None = None, with_reranker: bool = True) -> list[dict[str, Any]]:
    """Run all setups; writes ``results/rag_eval.{json,md}``."""
    questions = list(read_jsonl(cfg.data.processed_dir / "rag_eval.jsonl"))
    embedders = embedders or cfg.rag.compare_embedders
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(embedders):
        logger.info("Embedder %s", name)
        retr = HybridRetriever(cfg.rag.model_copy(update={"embedder": name}), Embedder(name, cfg.rag.device), use_reranker=False,
                               in_memory=True)  # fmt: skip
        if i == 0:
            rows.append({"setup": "BM25", **run_setup(retr, questions, "bm25")})
        rows.append({"setup": f"dense: {name}", **run_setup(retr, questions, "dense")})
        rows.append({"setup": f"hybrid: BM25 + {name}", **run_setup(retr, questions, "hybrid")})
        retr.close()
    if with_reranker and cfg.rag.reranker:
        retr = HybridRetriever(cfg.rag, in_memory=True)
        rows.append({"setup": f"hybrid + rerank: {cfg.rag.embedder}", **run_setup(retr, questions, "hybrid")})
        retr.close()
    out = Path("results")
    write_json(out / "rag_eval.json", {"n_questions": len(questions), "rows": rows})
    table = markdown_table(rows, ["setup", "recall@1", "recall@5", "mrr", "ms_per_query"])
    (out / "rag_eval.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table + "\n")
    return rows


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config_dir: str = typer.Option("configs", help="Directory with YAML configs"),
    embedders: list[str] | None = typer.Option(None, "--embedder", help="Override rag.compare_embedders"),
    no_reranker: bool = typer.Option(False, help="Skip the reranker setup"),
) -> None:
    """Evaluate retrieval."""
    setup_logging()
    evaluate(load_config(config_dir), embedders, not no_reranker)


if __name__ == "__main__":
    app()
