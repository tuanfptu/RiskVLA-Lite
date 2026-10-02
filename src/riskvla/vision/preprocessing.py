"""Conservative frame normalization before the model-owned processor."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageOps


def load_rgb_image(source: str | Path | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert("RGB")
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def constrain_pixels(image: Image.Image, max_pixels: int | None) -> Image.Image:
    """Downscale while preserving aspect ratio; never upscale."""
    if max_pixels is None:
        return image
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive or None")
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / pixels)
    size = (
        max(1, int(math.floor(image.width * scale))),
        max(1, int(math.floor(image.height * scale))),
    )
    return image.resize(size, resample=Image.Resampling.LANCZOS)


def prepare_frames(
    frames: Iterable[str | Path | Image.Image],
    *,
    max_pixels: int | None = None,
) -> list[Image.Image]:
    prepared = [
        constrain_pixels(load_rgb_image(frame), max_pixels=max_pixels) for frame in frames
    ]
    if not prepared:
        raise ValueError("At least one frame is required")
    return prepared
