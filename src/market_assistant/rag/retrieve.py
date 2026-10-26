"""Hybrid retrieval: BM25 + dense (Qdrant) -> Reciprocal Rank Fusion -> reranker -> token budget."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Literal

from market_assistant.config import RagConfig
from market_assistant.rag.index import Embedder, qdrant_client, upsert_chunks
from market_assistant.utils import approx_tokens, read_jsonl

logger = logging.getLogger(__name__)

Mode = Literal["hybrid", "bm25", "dense"]
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def bm25_tokens(text: str) -> list[str]:
    """Lowercased word tokens, with a crude 5-char stem for morphology-rich ru/uz."""
    return [t[:5] for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


def rrf(rankings: list[list[str]], k: int) -> dict[str, float]:
    """Reciprocal Rank Fusion: ``score(d) = Σ 1 / (k + rank)``."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def apply_token_budget(chunks: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Keep top chunks while their total size fits the budget (always keeps the first one)."""
    out, used = [], 0
    for c in chunks:
        cost = approx_tokens(c["text"])
        if out and used + cost > budget:
            break
        out.append(c)
        used += cost
    return out


class HybridRetriever:
    """Retrieves KB chunks for a query."""

    def __init__(
        self,
        cfg: RagConfig,
        embedder: Embedder | None = None,
        use_reranker: bool = True,
        in_memory: bool = False,
    ) -> None:
        """Load chunks, BM25 and the dense index.

        Args:
            cfg: RAG config.
            embedder: pre-loaded embedder (defaults to ``cfg.embedder``).
            use_reranker: apply ``cfg.reranker`` cross-encoder after fusion.
            in_memory: build a throwaway in-memory Qdrant collection (eval / tests).
        """
        from rank_bm25 import BM25Okapi

        self.cfg = cfg
        self.chunks = list(read_jsonl(Path(cfg.index_dir) / "chunks.jsonl"))
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([bm25_tokens(f"{c['title']} {c['text']}") for c in self.chunks])
        self.embedder = embedder or Embedder(cfg.embedder, cfg.device)
        self.client = qdrant_client(cfg, in_memory=in_memory)
        if in_memory:
            upsert_chunks(self.client, cfg.collection, self.embedder, self.chunks)
        self.reranker = None
        if use_reranker and cfg.reranker:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(cfg.reranker, device=cfg.device)

    def _bm25(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(bm25_tokens(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.chunks[i]["chunk_id"] for i in order if scores[i] > 0]

    def _dense(self, query: str, k: int) -> list[str]:
        vec = self.embedder.encode([query], is_query=True)[0]
        hits = self.client.query_points(self.cfg.collection, query=vec, limit=k).points
        return [h.payload["chunk_id"] for h in hits]

    def search(self, query: str, mode: Mode = "hybrid", top_k: int | None = None,
               budget: bool = True) -> list[dict[str, Any]]:  # fmt: skip
        """Return ranked chunks (dicts with ``score``), trimmed to the context token budget."""
        cfg = self.cfg
        top_k = top_k or cfg.final_top_k
        if mode == "bm25":
            ranked = self._bm25(query, cfg.bm25_top_k)
        elif mode == "dense":
            ranked = self._dense(query, cfg.dense_top_k)
        else:
            fused = rrf([self._bm25(query, cfg.bm25_top_k), self._dense(query, cfg.dense_top_k)], cfg.rrf_k)
            ranked = sorted(fused, key=lambda c: -fused[c])
        candidates = [dict(self.by_id[c]) for c in ranked[: max(top_k * 4, 20)]]
        if self.reranker is not None and candidates:
            scores = self.reranker.predict([(query, f"{c['title']}. {c['text']}") for c in candidates])
            for c, s in zip(candidates, scores, strict=True):
                c["score"] = float(s)
            candidates.sort(key=lambda c: -c["score"])
        result = candidates[:top_k]
        return apply_token_budget(result, cfg.context_token_budget) if budget else result

    def close(self) -> None:
        """Release the Qdrant client (embedded mode holds a file lock)."""
        self.client.close()
