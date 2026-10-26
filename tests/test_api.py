"""Gateway and image-service APIs with stubbed dependencies."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.images_main as images_main
import app.main as main
from market_assistant.agent.loop import Agent
from market_assistant.agent.tools import MarketTools
from market_assistant.config import AgentConfig
from market_assistant.images.service import composite
from market_assistant.llm.client import LLMResponse, ScriptedClient, ToolCall
from market_assistant.router.classifier import KeywordRouter


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tools: MarketTools) -> Iterator[TestClient]:
    responses = [LLMResponse(tool_calls=[ToolCall("search_products", json.dumps({"query": "Samsung"}))]),
                 LLMResponse(content="Zo'r выбор! Нашёл варианты.")] + [LLMResponse(content="Salom!")] * 50  # fmt: skip
    agent = Agent(ScriptedClient(responses), tools, AgentConfig(), router=KeywordRouter())
    monkeypatch.setattr(main, "build_agent", lambda cfg: (agent, {"llm": "mock"}))
    with TestClient(main.app) as c:
        c.app.state.cfg.serve.llm.backend = "mock"
        yield c


def test_chat_with_tool_trace(client: TestClient) -> None:
    resp = client.post("/chat", json={"message": "ищу Samsung телевизор, есть?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "Zo'r выбор! Нашёл варианты." and body["intent"] == "product_question"
    assert any(e["type"] == "tool" and e["name"] == "search_products" for e in body["events"])
    again = client.post("/chat", json={"message": "спасибо", "session_id": body["session_id"]}).json()
    assert again["session_id"] == body["session_id"]


def test_chat_stream_sse(client: TestClient) -> None:
    with client.stream("POST", "/chat?stream=true", json={"message": "привет"}) as resp:
        text = "".join(resp.iter_text())
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "event: route" in text and "event: reply" in text and text.rstrip().endswith("data: {}")


@pytest.mark.parametrize("message", ["", "   ", "x" * 5000])
def test_chat_validation(client: TestClient, message: str) -> None:
    assert client.post("/chat", json={"message": message}).status_code == 422


def test_handoff_and_metrics(client: TestClient) -> None:
    body = client.post("/chat", json={"message": "позовите оператора"}).json()
    assert body["handoff"] is True
    metrics = client.get("/metrics").text
    assert "assistant_handoffs_total" in metrics and 'assistant_requests_total{intent="human_handoff"}' in metrics


def test_rate_limit(client: TestClient) -> None:
    client.app.state.limiter = main.RateLimiter(per_minute=2)
    codes = [client.post("/chat", json={"message": "привет", "session_id": "s1"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["components"]["llm"] == "mock"


def test_not_ready_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(cfg: Any) -> Any:
        raise RuntimeError("no db")

    monkeypatch.setattr(main, "build_agent", boom)
    with TestClient(main.app) as c:
        assert c.post("/chat", json={"message": "привет"}).status_code == 503
        assert c.get("/health").json()["status"] == "degraded"


# --------------------------------------------------------------------------- images


def _png(size: tuple[int, int] = (64, 64), mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_composite_keeps_background_size() -> None:
    product = Image.new("RGBA", (300, 200), (255, 0, 0, 255))
    out = composite(product, Image.new("RGB", (512, 512), (240, 240, 240)))
    assert out.size == (512, 512) and out.mode == "RGB"
    assert out.getpixel((256, 400))[0] > 200  # product placed in the lower middle


def test_images_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubService:
        def background(self, photo: bytes, style: str, seed: int) -> bytes:
            return _png((32, 32))

        def maybe_unload(self) -> bool:
            return False

    monkeypatch.setattr(images_main, "build_service", StubService)
    with TestClient(images_main.app) as c:
        assert "studio" in c.get("/styles").json()["styles"]
        resp = c.post("/images/background", files={"image": ("p.png", _png(), "image/png")}, data={"style": "kitchen"})
        assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"
        bad = c.post("/images/background", files={"image": ("p.png", _png(), "image/png")}, data={"style": "mars"})
        assert bad.status_code == 422


def test_image_service_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    from market_assistant.config import ImagesConfig
    from market_assistant.images import service as svc

    monkeypatch.setattr(svc, "cut_out", lambda img: img.convert("RGBA"))
    service = svc.ImageService(ImagesConfig(size=128, unload_after_s=0))
    monkeypatch.setattr(service, "_generate_diffusers", lambda prompt, seed: Image.new("RGB", (128, 128), "white"))
    png = service.background(_png(), style="studio")
    assert Image.open(io.BytesIO(png)).size == (128, 128)
    with pytest.raises(ValueError):
        service.background(_png(), style="unknown")
