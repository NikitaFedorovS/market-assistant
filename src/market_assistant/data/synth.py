"""Generate the synthetic marketplace: catalog, orders DB, knowledge base, dialogs, router data.

CLI: ``python -m market_assistant.data.synth``
"""

from __future__ import annotations

import logging
import random
from collections import Counter

import typer

from market_assistant.config import DataConfig, load_config
from market_assistant.data.catalog import REFERENCE_DATE, create_db, generate_kb, generate_products
from market_assistant.data.dialogs import DialogGenerator, rag_eval_set, router_samples
from market_assistant.utils import setup_logging, write_jsonl

logger = logging.getLogger(__name__)


def synthesize(cfg: DataConfig, dialogs_per_intent: int | None = None) -> dict[str, int]:
    """Write every synthetic artifact under ``cfg.data_dir``; returns counts."""
    sc = cfg.synth
    rng = random.Random(sc.seed)
    products = generate_products(sc.n_products, rng)
    write_jsonl(cfg.catalog_path, products)
    create_db(cfg.db_path, products, sc.n_orders, rng)
    kb = generate_kb()
    write_jsonl(cfg.kb_path, kb)

    gen = DialogGenerator(cfg.db_path, kb, products, REFERENCE_DATE, seed=sc.seed)
    dialogs = gen.generate(dialogs_per_intent or sc.dialogs_per_intent)
    write_jsonl(cfg.raw_dir / "dialogs.jsonl", dialogs)
    router = router_samples(dialogs, kb, sc.router_samples_per_intent, seed=sc.seed)
    write_jsonl(cfg.raw_dir / "router.jsonl", router)
    rag_eval = rag_eval_set(kb)
    write_jsonl(cfg.processed_dir / "rag_eval.jsonl", rag_eval)

    noise = Counter(d.get("noise", "clean") for d in dialogs)
    logger.info("Products: %d | KB articles: %d | dialogs: %d | noise: %s", len(products), len(kb),
                len(dialogs), dict(noise))  # fmt: skip
    logger.info("Router samples: %d | RAG eval questions: %d", len(router), len(rag_eval))
    return {"products": len(products), "kb": len(kb), "dialogs": len(dialogs), "router": len(router)}


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config_dir: str = typer.Option("configs", help="Directory with YAML configs"),
    dialogs_per_intent: int | None = typer.Option(None, help="Override synth.dialogs_per_intent"),
) -> None:
    """Generate synthetic data."""
    setup_logging()
    synthesize(load_config(config_dir).data, dialogs_per_intent)


if __name__ == "__main__":
    app()
