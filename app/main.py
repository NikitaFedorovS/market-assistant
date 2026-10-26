"""Chat gateway: router + RAG + agent over the LLM backend (vLLM in production).

Run: ``uvicorn app.main:app --host 0.0.0.0 --port 8000``
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from market_assistant.agent.loop import Agent, AgentResult, Session
from market_assistant.agent.tools import MarketTools
from market_assistant.config import AppConfig, load_config
from market_assistant.data.catalog import REFERENCE_DATE
from market_assistant.llm.client import build_client
from market_assistant.router.classifier import load_router
from market_assistant.utils import setup_logging

logger = logging.getLogger("market_assistant.api")

REQUESTS = Counter("assistant_requests_total", "Chat requests", ["intent"])
HANDOFFS = Counter("assistant_handoffs_total", "Conversations handed to an operator")
TOOL_CALLS = Counter("assistant_tool_calls_total", "Tool calls", ["name", "status"])
ERRORS = Counter("assistant_errors_total", "Failed requests", ["kind"])
LATENCY = Histogram("assistant_latency_seconds", "End-to-end latency", buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16))

MAX_SESSIONS = 10_000


class ChatRequest(BaseModel):
    """One user message."""

    message: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    """Assistant reply + trace of what the agent did."""

    session_id: str
    reply: str
    intent: str
    handoff: bool
    events: list[dict[str, Any]]
    latency_ms: float


class SessionStore:
    """In-memory LRU of sessions (swap for Redis when running several replicas)."""

    def __init__(self, capacity: int = MAX_SESSIONS) -> None:
        self._data: OrderedDict[str, Session] = OrderedDict()
        self.capacity = capacity

    def get(self, session_id: str) -> Session:
        session = self._data.pop(session_id, None) or Session()
        self._data[session_id] = session
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)
        return session


class RateLimiter:
    """Sliding-window limit per key."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            return False
        hits.append(now)
        return True


def build_agent(cfg: AppConfig) -> tuple[Agent, dict[str, str]]:
    """Wire LLM, tools, router and retriever from config. Returns the agent and component status."""
    status: dict[str, str] = {}
    today = date.fromisoformat(os.environ["MA_TODAY"]) if os.environ.get("MA_TODAY") else REFERENCE_DATE
    tools = MarketTools(cfg.data.db_path, today)
    status["db"] = "ok" if cfg.data.db_path.exists() else "missing"
    router = load_router(cfg.router.output_dir, cfg.router.confidence_threshold)
    status["router"] = type(router).__name__
    retriever = None
    if (Path(cfg.rag.index_dir) / "chunks.jsonl").exists():
        from market_assistant.rag.retrieve import HybridRetriever

        retriever = HybridRetriever(cfg.rag)
        status["rag"] = "ok"
    else:
        status["rag"] = "missing index"
    llm = build_client(cfg.serve.llm)
    status["llm"] = (
        f"{cfg.serve.llm.backend}:{cfg.serve.llm.model if cfg.serve.llm.backend == 'openai' else cfg.serve.llm.local_model}"
    )
    agent = Agent(llm, tools, cfg.serve.agent, router, retriever, cfg.serve.api.max_history_turns)
    return agent, status


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load everything once at startup."""
    setup_logging()
    cfg = load_config()
    app.state.cfg = cfg
    app.state.sessions = SessionStore()
    app.state.limiter = RateLimiter(cfg.serve.api.rate_limit_per_min)
    # vLLM batches concurrent requests itself; a local transformers model must be used serially.
    app.state.slots = asyncio.Semaphore(1 if cfg.serve.llm.backend == "transformers" else 64)
    app.state.agent, app.state.status = None, {}
    try:
        app.state.agent, app.state.status = await asyncio.to_thread(build_agent, cfg)
        logger.info("Agent ready: %s", app.state.status)
    except Exception as e:
        logger.exception("Startup failed")
        app.state.status = {"error": str(e)}
    yield


app = FastAPI(title="Market assistant", version="0.1.0", lifespan=lifespan)


def _record(result: AgentResult, seconds: float) -> None:
    REQUESTS.labels(result.intent).inc()
    LATENCY.observe(seconds)
    if result.handoff:
        HANDOFFS.inc()
    for ev in result.events:
        if ev["type"] == "tool":
            TOOL_CALLS.labels(ev["name"], "error" if "error" in ev["result"] else "ok").inc()


async def _handle(request: Request, req: ChatRequest) -> tuple[str, AgentResult]:
    state = request.app.state
    if state.agent is None:
        raise HTTPException(503, "Assistant is not ready")
    api = state.cfg.serve.api
    if len(req.message) > api.max_input_chars or not req.message.strip():
        raise HTTPException(422, f"message must be 1..{api.max_input_chars} non-blank chars")
    session_id = req.session_id or uuid.uuid4().hex
    client_key = request.client.host if request.client else "unknown"
    if not state.limiter.allow(f"{client_key}:{session_id}"):
        raise HTTPException(429, "Too many requests")
    session = state.sessions.get(session_id)
    t0 = time.perf_counter()
    try:
        async with state.slots:
            result = await asyncio.wait_for(
                asyncio.to_thread(state.agent.handle, session, req.message), timeout=api.request_timeout_s
            )
    except asyncio.TimeoutError as e:  # builtin TimeoutError alias only from 3.11
        ERRORS.labels("timeout").inc()
        raise HTTPException(504, "Assistant timed out") from e
    _record(result, time.perf_counter() - t0)
    return session_id, result


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, stream: bool = False) -> Any:
    """Send a message. With ``?stream=true`` the agent trace and reply come as Server-Sent Events."""
    session_id, result = await _handle(request, req)
    if not stream:
        return ChatResponse(session_id=session_id, **result.__dict__)

    async def events() -> AsyncIterator[str]:
        for ev in result.events:
            yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
        payload = {"session_id": session_id, "reply": result.reply, "intent": result.intent, "handoff": result.handoff}
        yield f"event: reply\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Component status (LLM reachability is checked for the vLLM backend)."""
    state = request.app.state
    status = dict(state.status)
    cfg = state.cfg.serve.llm
    if state.agent is not None and cfg.backend == "openai":
        import httpx

        try:
            async with httpx.AsyncClient(timeout=2) as client:
                r = await client.get(f"{cfg.base_url.rstrip('/')}/models")
            status["llm_reachable"] = r.status_code == 200
        except httpx.HTTPError:
            status["llm_reachable"] = False
    ok = state.agent is not None and status.get("llm_reachable", True)
    return {"status": "ok" if ok else "degraded", "components": status}


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus metrics."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
