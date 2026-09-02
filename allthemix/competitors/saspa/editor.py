"""BLIP-Diffusion-ControlNet editor used by the official SaSPA method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from allthemix.competitors.generative.torch_runtime import (
    TorchRuntime,
    resolve_huggingface_snapshot,
)

DEFAULT_MODEL_ID = "Salesforce/blipdiffusion-controlnet"
DEFAULT_NEGATIVE_PROMPT = (
    "over-exposure, under-exposure, saturated, duplicate, out of frame, "
    "lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, "
    "mutilated, ugly, bad anatomy, bad proportions, deformed, blurry"
)


@dataclass(frozen=True)
class EditorConfig:
    """Immutable SaSPA foundation-model settings."""

    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = ""
    device: str = "auto"
    dtype: str = "auto"
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    xla_cache_dir: str = ""


class SaSPAEditor:
    """Generate one subject- and structure-preserving synthetic image."""

    def __init__(self, config: EditorConfig) -> None:
        if config.guidance_scale <= 0.0 or config.num_inference_steps < 1:
            raise ValueError(
                "SaSPA guidance_scale and inference steps must be positive."
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
                include_pytorch_bin=True,
            )
        )
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        """Load the last Diffusers release containing BLIP ControlNet."""
        try:
            from diffusers.pipelines import BlipDiffusionControlNetPipeline
        except ImportError as error:
            raise RuntimeError(
                "SaSPA requires diffusers==0.32.2 as in the official "
                "environment; newer releases removed "
                "BlipDiffusionControlNetPipeline."
            ) from error
        pipeline = BlipDiffusionControlNetPipeline.from_pretrained(
            self.model_source,
            torch_dtype=self.runtime.dtype,
        ).to(self.runtime.device)
        pipeline.set_progress_bar_config(disable=True)

        return pipeline

    def edit(
        self,
        reference_image: Image.Image,
        conditioning_image: Image.Image,
        prompt: str,
        superclass: str,
        seed: int,
        width: int,
        height: int,
    ) -> Image.Image:
        """Run the official BLIP subject plus Canny structure call."""
        with self.runtime.inference_context():
            result = self.pipeline(
                prompt=prompt,
                reference_image=reference_image.convert("RGB"),
                # The misspelling is part of Diffusers' public BLIP API.
                condtioning_image=conditioning_image.convert("RGB"),
                source_subject_category=superclass,
                target_subject_category=superclass,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                generator=self.runtime.cpu_generator(seed),
                neg_prompt=self.config.negative_prompt,
                width=width,
                height=height,
                # Diffusers cannot convert a bfloat16 tensor directly to
                # NumPy while producing its default PIL output. Keep the
                # normalized tensor on the backend and perform the narrow
                # output-boundary cast below instead.
                output_type="pt",
            )
        image = result.images[0].detach().float()
        # Include the narrow output cast in the same XLA graph, then perform
        # the blocking host transfer before NumPy conversion.
        self.runtime.mark_step()
        image = image.cpu().permute(1, 2, 0).numpy()
        pixels = np.clip(
            np.rint(image * 255.0),
            0,
            255,
        ).astype(np.uint8)

        return Image.fromarray(pixels).convert("RGB")

    def synchronize(self) -> None:
        """Wait for pending lazy XLA operations."""
        self.runtime.synchronize()
