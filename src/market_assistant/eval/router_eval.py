"""Router quality and CPU latency on the held-out split.

CLI: ``python -m market_assistant.eval.router_eval``
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import typer

from market_assistant.config import INTENTS, load_config
from market_assistant.router.classifier import IntentRouter
from market_assistant.utils import read_jsonl, setup_logging, write_json


def evaluate(model_dir: str, test_path: Path, threshold: float) -> dict[str, Any]:
    """Macro-F1, per-class report, confusion matrix and p50/p95 latency (CPU, batch=1)."""
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    router = IntentRouter(model_dir, threshold=0.0, device="cpu")  # raw argmax for quality metrics
    rows = list(read_jsonl(test_path))
    preds, latencies = [], []
    for r in rows:
        t0 = time.perf_counter()
        preds.append(router(r["text"]).intent)
        latencies.append((time.perf_counter() - t0) * 1000)
    gold = [r["label"] for r in rows]
    labels = list(INTENTS)
    result = {
        "n": len(rows),
        "macro_f1": f1_score(gold, preds, average="macro"),
        "report": classification_report(gold, preds, labels=labels, output_dict=True, zero_division=0),
        "confusion": {"labels": labels, "matrix": confusion_matrix(gold, preds, labels=labels).tolist()},
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": sorted(latencies)[int(len(latencies) * 0.95)],
        "threshold_in_prod": threshold,
    }
    write_json(Path("results") / "router_eval.json", result)
    print(f"macro-F1 {result['macro_f1']:.3f} | p50 {result['latency_ms_p50']:.1f} ms | "
          f"p95 {result['latency_ms_p95']:.1f} ms (CPU)")  # fmt: skip
    return result


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(config_dir: str = typer.Option("configs", help="Directory with YAML configs")) -> None:
    """Evaluate the router."""
    setup_logging()
    cfg = load_config(config_dir)
    evaluate(cfg.router.output_dir, cfg.data.processed_dir / "router_test.jsonl", cfg.router.confidence_threshold)


if __name__ == "__main__":
    app()
