"""OpenAI CLIP RN50 semantic filter used by the official SaSPA code."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from allthemix.competitors.generative.torch_runtime import TorchRuntime


@dataclass(frozen=True)
class SaSPAClipConfig:
    """Immutable OpenAI CLIP inference settings."""

    model_name: str = "RN50"
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 16
    download_root: str = ""
    xla_cache_dir: str = ""


class SaSPAClipSemanticScorer:
    """Apply the exact RN50 superclass-vs-negative argmax decision."""

    def __init__(
        self,
        config: SaSPAClipConfig,
        positive_prompts: Sequence[str],
        negative_prompts: Sequence[str],
    ) -> None:
        if not positive_prompts or not negative_prompts:
            raise ValueError(
                "SaSPA CLIP filtering requires positive and negative prompts."
            )
        if config.batch_size < 1:
            raise ValueError("SaSPA CLIP batch_size must be positive.")
        self.config = config
        self.positive_prompts = tuple(positive_prompts)
        self.negative_prompts = tuple(negative_prompts)
        self.runtime = TorchRuntime(
            device=config.device,
            dtype=config.dtype,
            xla_cache_dir=config.xla_cache_dir,
        )
        try:
            import clip
        except ImportError as error:
            raise RuntimeError(
                "SaSPA semantic filtering requires openai-clip==1.0.1 and "
                "the matching torchvision build from requirements-xla.txt."
            ) from error
        if config.model_name not in clip.available_models():
            raise ValueError(
                f"Unknown OpenAI CLIP model {config.model_name!r}."
            )
        self._clip = clip
        self.model, self.preprocess = clip.load(
            config.model_name,
            device="cpu",
            jit=False,
            download_root=config.download_root or None,
        )
        self.model = self.model.to(
            device=self.runtime.device,
            dtype=self.runtime.dtype,
        )
        self.model.eval()
        model_url = str(getattr(clip, "_MODELS", {}).get(config.model_name, ""))
        self.resolved_model_commit = (
            "openai-url-sha256:"
            + hashlib.sha256(model_url.encode("utf-8")).hexdigest()
        )
        self._text_features = self._encode_text()

    def _encode_text(self):
        """Encode all official semantic candidates exactly once."""
        tokens = self._clip.tokenize(
            [*self.positive_prompts, *self.negative_prompts]
        ).to(self.runtime.device)
        with self.runtime.inference_context():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        self.runtime.mark_step()

        return features

    def score(self, images: Sequence[Image.Image]) -> list[dict[str, object]]:
        """Return the official first-prompt argmax decision for each image."""
        if not images:
            return []
        torch = self.runtime.torch
        image_tensor = torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images]
        ).to(device=self.runtime.device, dtype=self.runtime.dtype)
        with self.runtime.inference_context():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            )
            logits = (
                self.model.logit_scale.exp()
                * image_features
                @ self._text_features.T
            )
            probabilities = logits.softmax(dim=-1).float().cpu().numpy()
        self.runtime.mark_step()
        predictions = np.argmax(probabilities, axis=1)
        positive_count = len(self.positive_prompts)
        texts = [*self.positive_prompts, *self.negative_prompts]

        return [
            {
                "semantic_pass": bool(prediction < positive_count),
                "clip_prediction_index": int(prediction),
                "clip_prediction_text": texts[prediction],
                "clip_positive_probability": float(
                    np.sum(probability[:positive_count])
                ),
            }
            for prediction, probability in zip(predictions, probabilities)
        ]

    def synchronize(self) -> None:
        """Wait for pending lazy XLA work before publishing the manifest."""
        self.runtime.synchronize()
