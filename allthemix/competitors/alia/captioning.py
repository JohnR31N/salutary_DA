"""BLIP caption generation for ALIA prompt discovery."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from allthemix.competitors.generative.torch_runtime import (
    TorchRuntime,
    resolve_huggingface_snapshot,
)

DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"


@dataclass(frozen=True)
class CaptionerConfig:
    """BLIP inference settings used for dataset captioning."""

    model_id: str = DEFAULT_CAPTION_MODEL
    model_revision: str = ""
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 64
    xla_cache_dir: str = ""


class BlipCaptioner:
    """Lazy BLIP captioner compatible with CPU, CUDA, and PyTorch/XLA."""

    def __init__(self, config: CaptionerConfig) -> None:
        self.config = config
        self.runtime = TorchRuntime(
            device=config.device,
            dtype=config.dtype,
            xla_cache_dir=config.xla_cache_dir,
        )
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as error:
            raise RuntimeError(
                "ALIA captioning requires transformers with BLIP support."
            ) from error
        self.model_source, self.resolved_model_commit = (
            resolve_huggingface_snapshot(
                model_id=config.model_id,
                revision=config.model_revision,
            )
        )
        self.processor = BlipProcessor.from_pretrained(self.model_source)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_source,
            torch_dtype=self.runtime.dtype,
        ).to(self.runtime.device)
        self.model.eval()

    def caption(self, image: Image.Image) -> str:
        """Caption one RGB source image without updating model state."""
        inputs = self.processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )
        inputs = self.runtime.move_inputs(dict(inputs))
        with self.runtime.inference_context():
            tokens = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
            )
        self.runtime.mark_step()

        return self.processor.decode(
            tokens[0].detach().cpu(),
            skip_special_tokens=True,
        ).strip()

    def synchronize(self) -> None:
        """Wait for pending XLA work before publishing caption artifacts."""
        self.runtime.synchronize()
