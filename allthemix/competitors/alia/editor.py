"""Stable Diffusion Img2Img editor matching the released ALIA CUB setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from allthemix.competitors.generative.torch_runtime import (
    TorchRuntime,
    resolve_huggingface_snapshot,
)

DEFAULT_EDITOR_MODEL = "runwayml/stable-diffusion-v1-5"


@dataclass(frozen=True)
class EditorConfig:
    """ALIA Img2Img settings reported by the official CUB release."""

    model_id: str = DEFAULT_EDITOR_MODEL
    model_revision: str = ""
    device: str = "auto"
    dtype: str = "auto"
    strength: float = 0.6
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    disable_safety_checker: bool = True
    attention_slicing: bool = False
    xla_cache_dir: str = ""


class StableDiffusionImg2ImgEditor:
    """Generate language-guided edits without sharing a process with JAX."""

    def __init__(self, config: EditorConfig) -> None:
        if not 0.0 < config.strength <= 1.0:
            raise ValueError(
                f"ALIA Img2Img strength must be in (0, 1]. Got {config.strength}."
            )
        if config.guidance_scale <= 0.0 or config.num_inference_steps < 1:
            raise ValueError(
                "ALIA guidance_scale and num_inference_steps must be positive."
            )
        self.config = config
        self.runtime = TorchRuntime(
            device=config.device,
            dtype=config.dtype,
            xla_cache_dir=config.xla_cache_dir,
        )
        self.model_source, self.resolved_model_commit = (
            resolve_huggingface_snapshot(
                model_id=config.model_id,
                revision=config.model_revision,
            )
        )
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        """Load the immutable Stable Diffusion Img2Img pipeline."""
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
        except ImportError as error:
            raise RuntimeError(
                "ALIA editing requires diffusers and transformers."
            ) from error
        kwargs: dict[str, Any] = {
            "torch_dtype": self.runtime.dtype,
            "use_safetensors": True,
        }
        if self.config.disable_safety_checker:
            kwargs["safety_checker"] = None
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_source,
            **kwargs,
        ).to(self.runtime.device)
        pipeline.set_progress_bar_config(disable=True)
        if self.config.attention_slicing:
            pipeline.enable_attention_slicing()

        return pipeline

    def edit(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
        """Generate one deterministic edit from a source image and prompt."""
        with self.runtime.inference_context():
            result = self.pipeline(
                prompt=prompt,
                image=image.convert("RGB"),
                strength=self.config.strength,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                num_images_per_prompt=1,
                generator=self.runtime.cpu_generator(seed),
            )
        self.runtime.mark_step()

        return result.images[0].convert("RGB")

    def synchronize(self) -> None:
        """Wait for pending XLA work before publishing the artifact."""
        self.runtime.synchronize()
