"""Chunking, fusion, token budget and retrieval on a mini index (fake embedder, in-memory Qdrant)."""

from __future__ import annotations

from typing import Any

from market_assistant.config import AppConfig
from market_assistant.eval.rag_eval import ranking_metrics
from market_assistant.rag.index import chunk_article
from market_assistant.rag.retrieve import HybridRetriever, apply_token_budget, rrf
from market_assistant.utils import write_jsonl


def test_chunking_with_overlap() -> None:
    text = " ".join(f"Предложение номер {i} про доставку и возврат." for i in range(40))
    chunks = chunk_article({"doc_id": "d", "lang": "ru", "topic": "t", "title": "T", "text": text}, 60, 15)
    assert len(chunks) > 1
    assert chunks[0]["chunk_id"] == "d#0"
    assert chunks[0]["text"].split(". ")[-1] in chunks[1]["text"]  # overlap carried over
    short = chunk_article({"doc_id": "s", "lang": "ru", "topic": "t", "title": "T", "text": "Коротко."}, 60, 15)
    assert len(short) == 1


def test_rrf_prefers_items_ranked_high_in_both() -> None:
    scores = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert max(scores, key=scores.get) in {"a", "b"}
    assert scores["a"] > scores["c"] and scores["b"] > scores["d"]


def test_token_budget_keeps_first() -> None:
    chunks = [{"text": "x" * 3000}, {"text": "y" * 30}]
    assert apply_token_budget(chunks, 100) == chunks[:1]
    assert len(apply_token_budget([{"text": "a" * 30}] * 5, 25)) == 2


def test_ranking_metrics() -> None:
    m = ranking_metrics([["a", "a", "b"], ["c", "b"], ["x"]], ["a", "b", "z"])
    assert m["recall@1"] == 1 / 3 and m["recall@5"] == 2 / 3
    assert abs(m["mrr"] - (1 + 0.5) / 3) < 1e-9


def test_hybrid_retrieval(app_cfg: AppConfig, market: dict[str, Any], fake_embedder: Any) -> None:
    chunks = [c for a in market["kb"] for c in chunk_article(a, 300, 50)]
    write_jsonl(f"{app_cfg.rag.index_dir}/chunks.jsonl", chunks)
    retr = HybridRetriever(app_cfg.rag, embedder=fake_embedder, use_reranker=False, in_memory=True)
    for mode in ("bm25", "dense", "hybrid"):
        hits = retr.search("Сколько идёт доставка в Самарканд?", mode=mode)
        assert hits, mode
    top = retr.search("Сколько идёт доставка в Самарканд?")[0]
    assert top["doc_id"].startswith("delivery-") and top["lang"] == "ru"
    assert retr.search("Muddatli to'lovni qanday rasmiylashtiraman?")[0]["doc_id"] == "installments-uz"
    retr.close()
