"""Fine-tune ruBERT-tiny2 as the intent router.

CLI: ``python -m market_assistant.train.train_router``
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import typer

from market_assistant.config import INTENTS, AppConfig, load_config
from market_assistant.utils import read_jsonl, resolve_device, setup_logging, write_json, write_jsonl

logger = logging.getLogger(__name__)

LABEL2ID = {label: i for i, label in enumerate(INTENTS)}
ID2LABEL = dict(enumerate(INTENTS))


def split_router_data(raw_path: Path, out_dir: Path, seed: int) -> dict[str, list[dict[str, Any]]]:
    """Deduplicate texts and split 80/10/10 (a text never appears in two splits)."""
    seen: set[str] = set()
    rows = []
    for r in read_jsonl(raw_path):
        if r["text"] not in seen:
            seen.add(r["text"])
            rows.append(r)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    splits = {"test": rows[: n // 10], "val": rows[n // 10 : n // 5], "train": rows[n // 5 :]}
    for name, part in splits.items():
        write_jsonl(out_dir / f"router_{name}.jsonl", part)
    return splits


def train_router(cfg: AppConfig, max_steps: int | None = None) -> dict[str, Any]:
    """Train and save the classifier to ``router.output_dir``; returns test metrics."""
    import numpy as np
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    rc = cfg.router
    splits = split_router_data(cfg.data.raw_dir / "router.jsonl", cfg.data.processed_dir, rc.seed)
    logger.info("Router splits: %s", {k: len(v) for k, v in splits.items()})
    tokenizer = AutoTokenizer.from_pretrained(rc.base_model)

    def encode(rows: list[dict[str, Any]]) -> Any:
        ds = Dataset.from_list([{"text": r["text"], "label": LABEL2ID[r["label"]]} for r in rows])
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=rc.max_len), batched=True)

    train_ds, val_ds, test_ds = (encode(splits[s]) for s in ("train", "val", "test"))
    model = AutoModelForSequenceClassification.from_pretrained(
        rc.base_model, num_labels=len(INTENTS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    def metrics(pred: Any) -> dict[str, float]:
        y_hat = np.argmax(pred.predictions, axis=-1)
        return {"macro_f1": float(f1_score(pred.label_ids, y_hat, average="macro"))}

    args = TrainingArguments(
        output_dir=str(Path(rc.output_dir) / "checkpoints"),
        learning_rate=rc.learning_rate,
        num_train_epochs=rc.num_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=rc.batch_size,
        per_device_eval_batch_size=rc.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to=[],
        seed=rc.seed,
        dataloader_pin_memory=resolve_device() == "cuda",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    test = trainer.evaluate(test_ds, metric_key_prefix="test")
    trainer.save_model(rc.output_dir)
    tokenizer.save_pretrained(rc.output_dir)
    write_json(Path(rc.output_dir) / "router_metrics.json", test)
    logger.info("Router test: %s", test)
    return test


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config_dir: str = typer.Option("configs", help="Directory with YAML configs"),
    max_steps: int | None = typer.Option(None, help="Stop after N steps (quick run)"),
) -> None:
    """Train the intent router."""
    setup_logging()
    train_router(load_config(config_dir), max_steps)


if __name__ == "__main__":
    app()
