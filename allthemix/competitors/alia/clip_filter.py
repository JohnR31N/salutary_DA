"""CLIP semantic scoring with ALIA's positive-vs-negative decision rule."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from allthemix.competitors.generative.torch_runtime import (
    TorchRuntime,
    resolve_huggingface_snapshot,
)

DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"


@dataclass(frozen=True)
class ClipConfig:
    """CLIP semantic-filter inference settings."""

    model_id: str = DEFAULT_CLIP_MODEL
    model_revision: str = ""
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 16
    logit_scale: float = 100.0
    xla_cache_dir: str = ""


class ClipSemanticScorer:
    """Score generated edits against positive and negative text prompts."""

    def __init__(
        self,
        config: ClipConfig,
        positive_prompts: Sequence[str],
        negative_prompts: Sequence[str],
    ) -> None:
        if not positive_prompts or not negative_prompts:
            raise ValueError(
                "ALIA CLIP filtering requires positive and negative prompts."
            )
        if config.batch_size < 1 or config.logit_scale <= 0.0:
            raise ValueError("CLIP batch_size and logit_scale must be positive.")
        self.config = config
        self.positive_prompts = tuple(positive_prompts)
        self.negative_prompts = tuple(negative_prompts)
        self.runtime = TorchRuntime(
            device=config.device,
            dtype=config.dtype,
            xla_cache_dir=config.xla_cache_dir,
        )
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError(
                "ALIA semantic filtering requires transformers CLIP support."
            ) from error
        self.model_source, self.resolved_model_commit = (
            resolve_huggingface_snapshot(
                model_id=config.model_id,
                revision=config.model_revision,
            )
        )
        self.processor = CLIPProcessor.from_pretrained(self.model_source)
        self.model = CLIPModel.from_pretrained(
            self.model_source,
            torch_dtype=self.runtime.dtype,
        ).to(self.runtime.device)
        self.model.eval()
        self._text_features = self._encode_text()

    def _encode_text(self):
        """Encode and normalize all semantic candidates once."""
        texts = [*self.positive_prompts, *self.negative_prompts]
        inputs = self.processor(
            text=texts,
            padding=True,
            return_tensors="pt",
        )
        inputs = self.runtime.move_inputs(dict(inputs))
        with self.runtime.inference_context():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        self.runtime.mark_step()

        return features

    def score(self, images: Sequence[Image.Image]) -> list[dict[str, object]]:
        """Return the official argmax semantic decision for each image."""
        if not images:
            return []
        inputs = self.processor(
            images=[image.convert("RGB") for image in images],
            return_tensors="pt",
        )
        inputs = self.runtime.move_inputs(dict(inputs))
        with self.runtime.inference_context():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            )
            logits = (
                self.config.logit_scale
                * image_features
                @ self._text_features.T
            )
            probabilities = logits.softmax(dim=-1).float().cpu().numpy()
        self.runtime.mark_step()
        predictions = np.argmax(probabilities, axis=1)
        positive_count = len(self.positive_prompts)

        return [
            {
                "semantic_pass": bool(prediction < positive_count),
                "clip_prediction_index": int(prediction),
                "clip_prediction_text": (
                    [*self.positive_prompts, *self.negative_prompts][prediction]
                ),
                "clip_positive_probability": float(
                    np.sum(probability[:positive_count])
                ),
            }
            for prediction, probability in zip(predictions, probabilities)
        ]

    def synchronize(self) -> None:
        """Wait for pending XLA work before publishing scores."""
        self.runtime.synchronize()
