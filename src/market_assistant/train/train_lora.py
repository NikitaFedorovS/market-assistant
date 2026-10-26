"""Persona LoRA / QLoRA fine-tuning of an instruct LLM (Qwen2.5) with TRL.

Backends:
- ``unsloth`` — FastLanguageModel (4-bit, fused kernels); CUDA only, fastest.
- ``hf`` — transformers + PEFT (+ bitsandbytes 4-bit on CUDA); also runs on MPS/CPU with small models.

Loss is computed on assistant tokens only (see :mod:`market_assistant.train.chat_format`).

CLI: ``python -m market_assistant.train.train_lora``
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import typer

from market_assistant.config import AppConfig, TrainLoraConfig, load_config
from market_assistant.train.chat_format import tokenize_chat
from market_assistant.utils import read_jsonl, resolve_device, setup_logging, write_json

logger = logging.getLogger(__name__)


def pick_backend(cfg: TrainLoraConfig, device: str) -> str:
    """``unsloth`` if requested/available on CUDA, else ``hf``."""
    has_unsloth = importlib.util.find_spec("unsloth") is not None
    if cfg.backend == "unsloth" and not (has_unsloth and device == "cuda"):
        raise RuntimeError("backend=unsloth needs the `unsloth` package and a CUDA GPU")
    if cfg.backend == "auto":
        return "unsloth" if has_unsloth and device == "cuda" else "hf"
    return cfg.backend


def load_model_and_tokenizer(cfg: TrainLoraConfig, device: str, backend: str) -> tuple[Any, Any]:
    """Load the base model (4-bit when configured and possible) with LoRA adapters attached."""
    lc = cfg.lora
    if backend == "unsloth":
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]

        model, tokenizer = FastLanguageModel.from_pretrained(
            cfg.base_model, max_seq_length=cfg.max_seq_len, load_in_4bit=cfg.load_in_4bit
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lc.r,
            lora_alpha=lc.alpha,
            lora_dropout=lc.dropout,
            target_modules=lc.target_modules,
            use_gradient_checkpointing="unsloth" if cfg.gradient_checkpointing else False,
            random_state=cfg.seed,
        )
        return model, tokenizer

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    quantize = cfg.load_in_4bit and device == "cuda"
    if cfg.load_in_4bit and not quantize:
        logger.warning("4-bit loading needs CUDA (bitsandbytes); loading full precision on %s", device)
    kwargs: dict[str, Any] = {"dtype": torch.bfloat16 if device == "cuda" else torch.float32}
    if quantize:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    if quantize:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        task_type="CAUSAL_LM", r=lc.r, lora_alpha=lc.alpha, lora_dropout=lc.dropout, target_modules=lc.target_modules
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


def load_tokenized(path: Path, tokenizer: Any, max_len: int, limit: int | None) -> Any:
    """Read an SFT split and tokenize it with assistant masks."""
    from datasets import Dataset

    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `make data` first")
    rows = []
    for i, rec in enumerate(read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        tok = tokenize_chat(tokenizer, rec["messages"], rec.get("uses_tools", False), max_len)
        if 1 in tok["assistant_masks"]:  # nothing to learn if truncation removed every answer
            rows.append(tok)
    return Dataset.from_list(rows)


def train(
    cfg: AppConfig,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Fine-tune and save only the LoRA adapter to ``train_lora.output_dir``."""
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    tc = cfg.train_lora
    device = resolve_device()
    backend = pick_backend(tc, device)
    logger.info("Base: %s | device: %s | backend: %s | 4-bit: %s", tc.base_model, device, backend, tc.load_in_4bit)
    model, tokenizer = load_model_and_tokenizer(tc, device, backend)

    data_dir = cfg.data.processed_dir
    train_ds = load_tokenized(data_dir / "sft_train.jsonl", tokenizer, tc.max_seq_len, max_train_samples)
    val_ds = load_tokenized(data_dir / "sft_val.jsonl", tokenizer, tc.max_seq_len, max_eval_samples)
    logger.info("Train: %d | val: %d", len(train_ds), len(val_ds))

    use_cuda = device == "cuda"
    args = SFTConfig(
        output_dir=tc.output_dir,
        max_length=tc.max_seq_len,
        learning_rate=tc.learning_rate,
        num_train_epochs=tc.num_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=tc.batch_size,
        per_device_eval_batch_size=tc.batch_size,
        gradient_accumulation_steps=tc.grad_accumulation,
        warmup_steps=tc.warmup_ratio,  # float in [0, 1) = ratio of total steps (transformers 5)
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=tc.eval_steps,
        save_strategy="steps",
        save_steps=tc.save_steps,
        save_total_limit=2,
        logging_steps=tc.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_cuda,
        report_to=[] if tc.report_to == "none" else [tc.report_to],
        seed=tc.seed,
        dataloader_pin_memory=use_cuda,
        pad_to_multiple_of=tc.pad_to_multiple_of,
    )
    callbacks = [EarlyStoppingCallback(tc.early_stopping_patience)] if tc.early_stopping_patience else []
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    result = trainer.train()
    final = trainer.evaluate()

    out = Path(tc.output_dir)
    trainer.model.save_pretrained(out)  # adapter only
    tokenizer.save_pretrained(out)
    summary = {"base_model": tc.base_model, "backend": backend, "train": result.metrics, "eval": final}
    write_json(out / "train_metrics.json", summary)
    logger.info("Saved adapter to %s | eval_loss=%.4f", out, final.get("eval_loss", float("nan")))
    return summary


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config_dir: str = typer.Option("configs", help="Directory with YAML configs"),
    max_train_samples: int | None = typer.Option(None, help="Use first N train dialogs"),
    max_eval_samples: int | None = typer.Option(None, help="Use first N val dialogs"),
    max_steps: int | None = typer.Option(None, help="Stop after N optimizer steps"),
) -> None:
    """Train the persona LoRA."""
    setup_logging()
    train(load_config(config_dir), max_train_samples, max_eval_samples, max_steps)


if __name__ == "__main__":
    app()
