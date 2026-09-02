"""Framework-independent image composition for DiffuseMix.

The diffusion editor is intentionally kept out of this module.  This makes the
paper's concatenation and fractal-blending stages usable and testable without
installing PyTorch/XLA or Diffusers in the JAX training environment.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

PAPER_PROMPTS = (
    "autumn",
    "snowy",
    "sunset",
    "watercolor art",
    "rainbow",
    "aurora",
    "mosaic",
    "ukiyo-e",
    "a sketch with crayon",
)

OFFICIAL_RELEASE_PROMPTS = (
    "Autumn",
    "snowy",
    "watercolor art",
    "sunset",
    "rainbow",
    "aurora",
    "mosaic",
    "ukiyo-e",
    "a sketch with crayon",
)

DEFAULT_PROMPTS = PAPER_PROMPTS

PAPER_MASKS = (
    "generated_left",
    "generated_right",
    "generated_top",
    "generated_bottom",
)

OFFICIAL_CODE_MASKS = (
    "generated_right",
    "generated_bottom",
)


def build_instruction(
    prompt: str,
    template: str = "A transformed version of image into {prompt}",
) -> str:
    """Insert one filter-like prompt into the paper's instruction template."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("DiffuseMix prompt must not be empty.")
    if "{prompt}" not in template:
        raise ValueError("prompt_template must contain the token '{prompt}'.")

    return template.format(
        prompt=prompt,
    )


def _rgb_array(
    image: Image.Image,
) -> np.ndarray:
    """Convert a PIL image to an RGB float array in [0, 255]."""
    return np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    )


def _quantize_rgb(
    array: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Convert a float RGB array using paper or release-code semantics."""
    if mode == "round":
        array = np.rint(
            array,
        )
    elif mode != "truncate":
        raise ValueError(
            "quantization must be 'round' or 'truncate'. "
            f"Got {mode}."
        )

    return np.clip(
        array,
        0,
        255,
    ).astype(
        np.uint8,
    )


def _hard_mask(
    height: int,
    width: int,
    mask_name: str,
) -> np.ndarray:
    """Build a binary mask whose ones select the generated image."""
    mask = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    if mask_name == "generated_left":
        mask[:, : width // 2] = 1.0
    elif mask_name == "generated_right":
        mask[:, width // 2 :] = 1.0
    elif mask_name == "generated_top":
        mask[: height // 2, :] = 1.0
    elif mask_name == "generated_bottom":
        mask[height // 2 :, :] = 1.0
    else:
        raise ValueError(
            f"Unsupported DiffuseMix mask: {mask_name}. "
            f"Expected one of {PAPER_MASKS}."
        )

    return mask


def _soften_mask(
    mask: np.ndarray,
    mask_name: str,
    seam_width: int,
) -> np.ndarray:
    """Replace the hard center boundary with a linear transition."""
    if seam_width <= 0:
        return mask

    height, width = mask.shape
    axis_size = width if mask_name.endswith(("left", "right")) else height
    seam_width = min(
        seam_width,
        axis_size,
    )
    seam_start = (axis_size - seam_width) // 2
    seam_stop = seam_start + seam_width

    increasing = mask_name in {
        "generated_right",
        "generated_bottom",
    }
    values = np.linspace(
        0.0 if increasing else 1.0,
        1.0 if increasing else 0.0,
        seam_width,
        dtype=np.float32,
    )

    softened = mask.copy()
    if mask_name.endswith(("left", "right")):
        softened[:, seam_start:seam_stop] = values[np.newaxis, :]
    else:
        softened[seam_start:seam_stop, :] = values[:, np.newaxis]

    return softened


def merge_original_and_generated(
    original: Image.Image,
    generated: Image.Image,
    mask_name: str,
    seam_width: int = 0,
    quantization: str = "round",
) -> Image.Image:
    """Form the half-original, half-generated hybrid image."""
    original = original.convert("RGB")
    generated = generated.convert("RGB")
    if generated.size != original.size:
        generated = generated.resize(
            original.size,
            Image.Resampling.LANCZOS,
        )

    width, height = original.size
    mask = _hard_mask(
        height=height,
        width=width,
        mask_name=mask_name,
    )
    mask = _soften_mask(
        mask=mask,
        mask_name=mask_name,
        seam_width=seam_width,
    )[:, :, np.newaxis]

    original_array = _rgb_array(
        original,
    )
    generated_array = _rgb_array(
        generated,
    )
    hybrid = (
        (1.0 - mask) * original_array
        + mask * generated_array
    )

    return Image.fromarray(
        _quantize_rgb(
            hybrid,
            mode=quantization,
        ),
    )


def blend_fractal(
    hybrid: Image.Image,
    fractal: Image.Image,
    fractal_alpha: float = 0.2,
    quantization: str = "round",
    resize_resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Blend a fractal with the hybrid using the paper's lambda equation."""
    if not np.isfinite(
        fractal_alpha,
    ) or fractal_alpha < 0.0 or fractal_alpha > 1.0:
        raise ValueError(
            "fractal_alpha must be in [0, 1]. "
            f"Got {fractal_alpha}."
        )

    hybrid = hybrid.convert("RGB")
    fractal = fractal.convert("RGB").resize(
        hybrid.size,
        resize_resample,
    )
    hybrid_array = _rgb_array(
        hybrid,
    )
    fractal_array = _rgb_array(
        fractal,
    )
    blended = (
        (1.0 - fractal_alpha) * hybrid_array
        + fractal_alpha * fractal_array
    )

    return Image.fromarray(
        _quantize_rgb(
            blended,
            mode=quantization,
        ),
    )


def compose_diffusemix(
    original: Image.Image,
    generated: Image.Image,
    fractal: Image.Image,
    mask_name: str,
    fractal_alpha: float = 0.2,
    seam_width: int = 0,
    quantization: str = "round",
    resize_resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Run the concatenation and fractal stages of DiffuseMix."""
    hybrid = merge_original_and_generated(
        original=original,
        generated=generated,
        mask_name=mask_name,
        seam_width=seam_width,
        quantization=quantization,
    )

    return blend_fractal(
        hybrid=hybrid,
        fractal=fractal,
        fractal_alpha=fractal_alpha,
        quantization=quantization,
        resize_resample=resize_resample,
    )


def is_near_black(
    image: Image.Image,
    channel_threshold: int = 8,
    pixel_fraction: float = 0.98,
) -> bool:
    """Return whether almost all pixels are effectively black."""
    if channel_threshold < 0 or channel_threshold > 255:
        raise ValueError("channel_threshold must be in [0, 255].")
    if not np.isfinite(
        pixel_fraction,
    ) or pixel_fraction < 0.0 or pixel_fraction > 1.0:
        raise ValueError("pixel_fraction must be in [0, 1].")

    image_array = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )
    dark_pixels = np.max(
        image_array,
        axis=-1,
    ) <= channel_threshold

    return bool(
        np.mean(
            dark_pixels,
        ) >= pixel_fraction
    )


def masks_for_mode(
    mode: str,
) -> Sequence[str]:
    """Return paper masks or the narrower behavior in the official code."""
    normalized = mode.lower().replace("-", "_")
    if normalized == "paper":
        return PAPER_MASKS
    if normalized == "official_code":
        return OFFICIAL_CODE_MASKS

    raise ValueError(
        "mask_mode must be 'paper' or 'official_code'. "
        f"Got {mode}."
    )
