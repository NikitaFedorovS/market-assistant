"""Shared fixtures: a small synthetic marketplace in a temp dir."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import pytest

from market_assistant.agent.tools import MarketTools
from market_assistant.config import AppConfig, DataConfig, SynthConfig
from market_assistant.data.catalog import REFERENCE_DATE, create_db, generate_kb, generate_products


@pytest.fixture
def data_cfg(tmp_path: Path) -> DataConfig:
    return DataConfig(
        data_dir=str(tmp_path / "data"),
        synth=SynthConfig(n_products=300, n_orders=400, dialogs_per_intent=40, router_samples_per_intent=30),
    )


@pytest.fixture
def market(data_cfg: DataConfig) -> dict[str, Any]:
    """Products + DB + KB, no dialogs."""
    rng = random.Random(0)
    products = generate_products(data_cfg.synth.n_products, rng)
    create_db(data_cfg.db_path, products, data_cfg.synth.n_orders, rng)
    return {"products": products, "kb": generate_kb(), "db": data_cfg.db_path}


@pytest.fixture
def tools(market: dict[str, Any]) -> MarketTools:
    return MarketTools(market["db"], REFERENCE_DATE)


@pytest.fixture
def app_cfg(data_cfg: DataConfig, tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data": data_cfg.model_dump(),
            "rag": {"index_dir": str(tmp_path / "index"), "reranker": None, "embedder": "fake"},
            "router": {"output_dir": str(tmp_path / "no-router")},
        }
    )


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors (no model download)."""

    name = "fake"
    dim = 64

    def encode(self, texts: list[str], is_query: bool) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                v[int(hashlib.md5(w[:5].encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
