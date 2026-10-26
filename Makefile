PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python
ARGS   ?=

.PHONY: install install-gpu synth data train train-router index eval eval-rag eval-router eval-llm serve serve-local images load-test test lint format clean

install:                ## local dev (Mac/CPU): everything except GPU-only packages
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements/dev.txt
	$(BIN)/pip install -e .

install-gpu: install    ## + unsloth (CUDA host)
	$(BIN)/pip install unsloth

synth:                  ## synthetic catalog, orders DB, KB, dialogs, router data
	$(PY) -m market_assistant.data.synth $(ARGS)

data: synth             ## stylize, PII, filters, dedup, general mix, splits
	$(PY) -m market_assistant.data.build_sft $(ARGS)

train:                  ## persona QLoRA (Qwen2.5-7B on a 24 GB GPU)
	$(PY) -m market_assistant.train.train_lora $(ARGS)

train-router:           ## ruBERT-tiny2 intent router
	$(PY) -m market_assistant.train.train_router $(ARGS)

index:                  ## RAG index (Qdrant + BM25 chunks)
	$(PY) -m market_assistant.rag.index $(ARGS)

eval-rag:
	$(PY) -m market_assistant.eval.rag_eval $(ARGS)

eval-router:
	$(PY) -m market_assistant.eval.router_eval $(ARGS)

eval-llm:               ## persona + regression (pass ARGS="--backend transformers --adapter models/persona-lora")
	$(PY) -m market_assistant.eval.persona_eval $(ARGS)
	$(PY) -m market_assistant.eval.regression_eval $(ARGS)

eval: eval-rag eval-router

serve:                  ## GPU host: vLLM + API + Qdrant + monitoring
	docker compose up -d --build

serve-local:            ## API only, LLM from serve.llm (e.g. MA__SERVE__LLM__BACKEND=transformers)
	$(BIN)/uvicorn app.main:app --host 0.0.0.0 --port $${PORT:-8000}

images:                 ## image service on the GPU host
	docker compose --profile images up -d --build images

load-test:
	$(BIN)/locust -f scripts/load_test.py --host $${HOST:-http://localhost:8000} -u 50 -r 5 -t 5m --headless

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src app tests scripts
	$(BIN)/black --check src app tests scripts

format:
	$(BIN)/ruff check --fix src app tests scripts
	$(BIN)/black src app tests scripts

clean:
	rm -rf .pytest_cache .ruff_cache build src/*.egg-info
	find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +
