"""SaSPA source geometry and Canny conditioning."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageOps


def resolve_resize_mode(mode: str, device_kind: str) -> str:
    """Use official aspect geometry unless XLA needs one static shape."""
    value = mode.strip().lower()
    if value == "auto":
        return "letterbox" if device_kind == "xla" else "official"
    if value not in {"official", "center_crop", "letterbox"}:
        raise ValueError(
            "SaSPA source_resize must be auto, official, center_crop, or "
            f"letterbox. Got {mode!r}."
        )

    return value


def resize_official(
    image: Image.Image,
    shorter_side: int = 512,
    max_pixels: int = 1_200_000,
) -> Image.Image:
    """Match the official shortest-side, multiple-of-64 resize helper."""
    if shorter_side < 64 or shorter_side % 64 != 0:
        raise ValueError("SaSPA generation_size must be a multiple of 64.")
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = float(shorter_side) / min(width, height)
    resized_width = width * scale
    resized_height = height * scale
    if resized_width * resized_height > max_pixels:
        scale *= math.sqrt(max_pixels / (resized_width * resized_height))
        resized_width = width * scale
        resized_height = height * scale
    output_width = max(64, int(round(resized_width / 64.0)) * 64)
    output_height = max(64, int(round(resized_height / 64.0)) * 64)
    method = Image.Resampling.LANCZOS if scale > 1.0 else Image.Resampling.BOX

    return rgb.resize((output_width, output_height), method)


def prepare_image(
    image: Image.Image,
    generation_size: int,
    mode: str,
) -> Image.Image:
    """Prepare a source or same-class reference image for generation."""
    if generation_size < 64 or generation_size % 64 != 0:
        raise ValueError("SaSPA generation_size must be a multiple of 64.")
    if mode == "official":
        return resize_official(image, shorter_side=generation_size)
    if mode == "center_crop":
        return ImageOps.fit(
            image.convert("RGB"),
            (generation_size, generation_size),
            method=Image.Resampling.LANCZOS,
        )
    if mode != "letterbox":
        raise ValueError(f"Unsupported SaSPA resize mode: {mode!r}.")
    contained = ImageOps.contain(
        image.convert("RGB"),
        (generation_size, generation_size),
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new(
        "RGB",
        (generation_size, generation_size),
        (127, 127, 127),
    )
    canvas.paste(
        contained,
        (
            (generation_size - contained.width) // 2,
            (generation_size - contained.height) // 2,
        ),
    )

    return canvas


def make_canny_image(
    image: Image.Image,
    low_threshold: int = 120,
    high_threshold: int = 200,
) -> Image.Image:
    """Create the three-channel Canny map used by official SaSPA."""
    if not 0 <= low_threshold < high_threshold <= 255:
        raise ValueError(
            "SaSPA Canny thresholds must satisfy "
            "0 <= low < high <= 255."
        )
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "SaSPA generation requires opencv-python-headless."
        ) from error
    edges = cv2.Canny(
        np.asarray(image.convert("RGB"), dtype=np.uint8),
        low_threshold,
        high_threshold,
    )

    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2), mode="RGB")
