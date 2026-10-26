"""Image service: generated backgrounds for seller product photos.

Run: ``uvicorn app.images_main:app --host 0.0.0.0 --port 8002``
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from market_assistant.config import load_config
from market_assistant.images.service import STYLES, ImageService
from market_assistant.utils import setup_logging

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def build_service() -> Any:
    """Create the service from config (patched in tests)."""
    return ImageService(load_config().serve.images)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the service; a background task frees VRAM when the pipeline sits idle."""
    setup_logging()
    app.state.service = build_service()

    async def janitor() -> None:
        while True:
            await asyncio.sleep(30)
            await asyncio.to_thread(app.state.service.maybe_unload)

    task = asyncio.create_task(janitor())
    yield
    task.cancel()


app = FastAPI(title="Market image service", version="0.1.0", lifespan=lifespan)


@app.get("/styles")
async def styles() -> dict[str, list[str]]:
    """Available background styles."""
    return {"styles": sorted(STYLES)}


@app.post("/images/background", response_class=Response)
async def background(
    request: Request,
    image: UploadFile = File(...),
    style: str = Form("studio"),
    seed: int = Form(0),
) -> Response:
    """Upload a product photo, get a PNG of it on a generated background."""
    data = await image.read()
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(422, "image must be 1 byte .. 10 MB")
    if style not in STYLES:
        raise HTTPException(422, f"unknown style; choose from {sorted(STYLES)}")
    try:
        png = await asyncio.to_thread(request.app.state.service.background, data, style, seed)
    except ImportError as e:
        raise HTTPException(503, f"image backend not installed: {e}") from e
    return Response(png, media_type="image/png")
