"""Build the RAG index: chunk KB articles, embed into Qdrant, keep chunks for BM25.

CLI: ``python -m market_assistant.rag.index``
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any

import typer

from market_assistant.config import RagConfig, load_config
from market_assistant.utils import approx_tokens, read_jsonl, setup_logging, write_jsonl

logger = logging.getLogger(__name__)

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_article(article: dict[str, Any], max_tokens: int, overlap: int) -> list[dict[str, Any]]:
    """Split an article into sentence-aligned chunks of ~``max_tokens`` with ~``overlap`` carry-over."""
    sentences = [s for s in _SENT_RE.split(article["text"]) if s.strip()]
    chunks: list[list[str]] = [[]]
    for s in sentences:
        if chunks[-1] and approx_tokens(" ".join(chunks[-1] + [s])) > max_tokens:
            tail: list[str] = []
            for prev in reversed(chunks[-1]):  # carry trailing sentences as overlap
                if approx_tokens(" ".join([prev, *tail])) > overlap:
                    break
                tail.insert(0, prev)
            chunks.append(tail)
        chunks[-1].append(s)
    return [
        {
            "chunk_id": f"{article['doc_id']}#{i}",
            "doc_id": article["doc_id"],
            "lang": article["lang"],
            "topic": article["topic"],
            "title": article["title"],
            "text": " ".join(parts),
        }
        for i, parts in enumerate(chunks)
        if parts
    ]


class Embedder:
    """SentenceTransformer wrapper that adds the query/passage prefixes some models need (e5)."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.e5 = "e5" in model_name.lower()

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], is_query: bool) -> list[list[float]]:
        if self.e5:
            texts = [("query: " if is_query else "passage: ") + t for t in texts]
        vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()


def qdrant_client(cfg: RagConfig, in_memory: bool = False) -> Any:
    """Qdrant server (``qdrant_url``), embedded on-disk mode, or in-memory."""
    from qdrant_client import QdrantClient

    if in_memory:
        return QdrantClient(location=":memory:")
    if cfg.qdrant_url:
        return QdrantClient(url=cfg.qdrant_url)
    return QdrantClient(path=str(Path(cfg.index_dir) / "qdrant"))


def upsert_chunks(client: Any, collection: str, embedder: Embedder, chunks: list[dict[str, Any]]) -> None:
    """(Re)create the collection and upload chunk embeddings with payload."""
    from qdrant_client.models import Distance, PointStruct, VectorParams

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection, vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE))
    vectors = embedder.encode([f"{c['title']}. {c['text']}" for c in chunks], is_query=False)
    client.upsert(
        collection,
        points=[
            PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["chunk_id"])), vector=v, payload=c)
            for c, v in zip(chunks, vectors, strict=True)
        ],
    )


def build_index(cfg: RagConfig, kb_path: Path) -> int:
    """Chunk the KB, write ``chunks.jsonl`` and fill the Qdrant collection. Returns chunk count."""
    chunks = [c for art in read_jsonl(kb_path) for c in chunk_article(art, cfg.chunk_tokens, cfg.chunk_overlap)]
    write_jsonl(Path(cfg.index_dir) / "chunks.jsonl", chunks)
    embedder = Embedder(cfg.embedder, cfg.device)
    client = qdrant_client(cfg)
    upsert_chunks(client, cfg.collection, embedder, chunks)
    client.close()
    logger.info("Indexed %d chunks with %s into %s", len(chunks), cfg.embedder, cfg.index_dir)
    return len(chunks)


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(config_dir: str = typer.Option("configs", help="Directory with YAML configs")) -> None:
    """Build the RAG index."""
    setup_logging()
    cfg = load_config(config_dir)
    build_index(cfg.rag, cfg.data.kb_path)


if __name__ == "__main__":
    app()
