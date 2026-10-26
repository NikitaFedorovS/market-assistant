"""Product-photo background generation: cut out the product (rembg), generate a scene, composite.

Backends:
- ``diffusers`` — SDXL / Flux pipeline loaded in-process (lazy; unloaded after idle to free VRAM for the LLM);
- ``comfyui`` — a running ComfyUI server driven through its HTTP API.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import uuid
from typing import Any

from PIL import Image, ImageFilter

from market_assistant.config import ImagesConfig

logger = logging.getLogger(__name__)

STYLES: dict[str, str] = {
    "studio": "clean professional product photography background, soft studio light, light grey seamless backdrop",
    "kitchen": "modern bright kitchen interior, marble countertop, morning sunlight, shallow depth of field",
    "living_room": "cozy scandinavian living room, wooden table, warm natural light, blurred background",
    "outdoor": "outdoor park scene, green grass, golden hour light, bokeh",
    "festive": "festive background with soft bokeh lights, elegant, holiday mood",
}
NEGATIVE = "text, watermark, logo, people, hands, low quality, blurry, distorted"


def cut_out(image: Image.Image) -> Image.Image:
    """Remove the background (RGBA output). Needs the optional ``rembg`` package."""
    from rembg import remove  # type: ignore[import-not-found]

    return remove(image.convert("RGBA"))


def composite(product: Image.Image, background: Image.Image, scale: float = 0.6) -> Image.Image:
    """Place the RGBA product centred near the bottom with a soft shadow."""
    bg = background.convert("RGBA")
    w = int(bg.width * scale)
    product = product.copy()
    product.thumbnail((w, w))
    x = (bg.width - product.width) // 2
    y = int(bg.height * 0.92) - product.height
    alpha = product.split()[-1]
    shadow = Image.new("RGBA", product.size, (0, 0, 0, 90))
    shadow.putalpha(alpha.point(lambda a: int(a * 0.35)).filter(ImageFilter.GaussianBlur(12)))
    bg.alpha_composite(shadow, (x + 12, y + 16))
    bg.alpha_composite(product, (x, y))
    return bg.convert("RGB")


class ImageService:
    """Thread-safe wrapper with lazy loading, a concurrency cap and idle unloading."""

    def __init__(self, cfg: ImagesConfig) -> None:
        self.cfg = cfg
        self._pipe: Any = None
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(cfg.max_concurrent)
        self._last_used = 0.0

    # ---------------------------------------------------------------- diffusers

    def _load(self) -> Any:
        import torch
        from diffusers import AutoPipelineForText2Image  # type: ignore[import-not-found]

        from market_assistant.utils import resolve_device

        device = resolve_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info("Loading %s on %s", self.cfg.model, device)
        pipe = AutoPipelineForText2Image.from_pretrained(self.cfg.model, torch_dtype=dtype)
        if device == "cuda":
            pipe.enable_model_cpu_offload()  # keeps peak VRAM low next to vLLM
        else:
            pipe = pipe.to(device)
        return pipe

    def maybe_unload(self) -> bool:
        """Free the pipeline if idle longer than ``unload_after_s``. Returns True if unloaded."""
        with self._lock:
            if self._pipe is None or time.monotonic() - self._last_used < self.cfg.unload_after_s:
                return False
            self._pipe = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
        logger.info("Image pipeline unloaded (idle)")
        return True

    def _generate_diffusers(self, prompt: str, seed: int) -> Image.Image:
        import torch

        with self._lock:
            if self._pipe is None:
                self._pipe = self._load()
            pipe = self._pipe
        gen = torch.Generator("cpu").manual_seed(seed)
        out = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=self.cfg.steps,
            guidance_scale=self.cfg.guidance_scale,
            width=self.cfg.size,
            height=self.cfg.size,
            generator=gen,
        )
        return out.images[0]

    # ---------------------------------------------------------------- comfyui

    def _generate_comfyui(self, prompt: str, seed: int) -> Image.Image:
        """Queue a minimal SDXL txt2img workflow in ComfyUI and fetch the result."""
        import httpx

        workflow = comfy_workflow(self.cfg.model, prompt, NEGATIVE, seed, self.cfg.steps, self.cfg.guidance_scale,
                                  self.cfg.size)  # fmt: skip
        base = self.cfg.comfyui_url.rstrip("/")
        with httpx.Client(timeout=30) as client:
            prompt_id = client.post(f"{base}/prompt", json={"prompt": workflow, "client_id": uuid.uuid4().hex}).json()[
                "prompt_id"
            ]
            for _ in range(300):
                hist = client.get(f"{base}/history/{prompt_id}").json()
                if prompt_id in hist:
                    img = next(iter(hist[prompt_id]["outputs"].values()))["images"][0]
                    data = client.get(f"{base}/view", params=img).content
                    return Image.open(io.BytesIO(data))
                time.sleep(1)
        raise TimeoutError("ComfyUI did not finish in time")

    # ---------------------------------------------------------------- public

    def background(self, photo: bytes, style: str = "studio", seed: int = 0, prompt_extra: str = "") -> bytes:
        """Return a PNG of the product on a generated background."""
        if style not in STYLES:
            raise ValueError(f"unknown style {style!r}; choose from {sorted(STYLES)}")
        prompt = (
            f"{STYLES[style]}, empty space in the center for a product{', ' + prompt_extra if prompt_extra else ''}"
        )
        product = cut_out(Image.open(io.BytesIO(photo)))
        with self._slots:
            gen = self._generate_comfyui if self.cfg.backend == "comfyui" else self._generate_diffusers
            bg = gen(prompt, seed)
            self._last_used = time.monotonic()
        buf = io.BytesIO()
        composite(product, bg).save(buf, format="PNG")
        return buf.getvalue()


def comfy_workflow(
    ckpt: str, prompt: str, negative: str, seed: int, steps: int, cfg: float, size: int
) -> dict[str, Any]:
    """Minimal ComfyUI API-format graph: checkpoint -> CLIP encode -> KSampler -> VAE decode -> save."""
    name = ckpt.split("/")[-1] + ".safetensors"
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": name}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": size, "height": size, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                                                   "latent_image": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg,
                                                   "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "market_bg"}},
    }  # fmt: skip
    return graph
