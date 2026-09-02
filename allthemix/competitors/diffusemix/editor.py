"""Lazy Diffusers editor with CUDA, CPU, and PyTorch/XLA backends."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from allthemix.competitors.diffusemix.compat import import_torch_xla

DEFAULT_MODEL_ID = "timbrooks/instruct-pix2pix"


@dataclass(frozen=True)
class EditorConfig:
    """Inference settings matching the released DiffuseMix generator."""

    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = ""
    device: str = "auto"
    dtype: str = "auto"
    guidance_scale: float = 4.0
    image_guidance_scale: float = 1.5
    num_inference_steps: int = 100
    disable_safety_checker: bool = True
    attention_slicing: bool = False
    xla_cache_dir: str = ""


def _looks_like_xla_environment() -> bool:
    """Return whether environment variables explicitly select XLA/PJRT."""
    pjrt_device = os.environ.get(
        "PJRT_DEVICE",
        "",
    ).upper()

    return bool(
        pjrt_device
        or os.environ.get(
            "XRT_TPU_CONFIG",
        )
        or os.environ.get(
            "TPU_NAME",
        )
    )


class InstructPix2PixEditor:
    """Run InstructPix2Pix without importing it in the JAX process."""

    def __init__(
        self,
        config: EditorConfig,
    ) -> None:
        self.config = config
        self._torch = self._import_torch()
        self.device_kind = self._resolve_device_kind(
            config.device,
        )
        self._torch_xla = (
            import_torch_xla()
            if self.device_kind == "xla"
            else None
        )
        if self.device_kind == "xla" and config.xla_cache_dir:
            import torch_xla.runtime as xr

            cache_dir = os.path.abspath(
                os.path.expanduser(
                    config.xla_cache_dir,
                )
            )
            os.makedirs(
                cache_dir,
                exist_ok=True,
            )
            xr.initialize_cache(
                cache_dir,
                readonly=False,
            )
        self.device = self._resolve_device(
            self.device_kind,
        )
        self.dtype = self._resolve_dtype(
            config.dtype,
            self.device_kind,
        )
        self._xla_model = None
        if self.device_kind == "xla":
            import torch_xla.core.xla_model as xm

            self._xla_model = xm

        (
            self.model_source,
            self.resolved_model_commit,
        ) = self._resolve_model_source()
        self.pipeline = self._build_pipeline()

    def _resolve_model_source(self) -> tuple[str, str]:
        """Resolve a Hub branch/tag to one immutable cached snapshot."""
        local_path = Path(
            self.config.model_id,
        ).expanduser()
        if local_path.is_dir():
            resolved_path = local_path.resolve()
            digest = hashlib.sha256()
            for file_path in sorted(
                path
                for path in resolved_path.rglob(
                    "*",
                )
                if path.is_file()
            ):
                stat = file_path.stat()
                digest.update(
                    file_path.relative_to(
                        resolved_path,
                    ).as_posix().encode(
                        "utf-8",
                    )
                )
                digest.update(
                    f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode(
                        "ascii",
                    )
                )
            return (
                str(
                    resolved_path,
                ),
                f"local-metadata-sha256:{digest.hexdigest()}",
            )

        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "DiffuseMix generation requires huggingface_hub to resolve "
                "the editor checkpoint to an immutable snapshot."
            ) from error

        snapshot_path = Path(
            snapshot_download(
                repo_id=self.config.model_id,
                revision=self.config.model_revision or None,
                ignore_patterns=(
                    "*.bin",
                    "*.ckpt",
                    "*.h5",
                    "*.msgpack",
                    "*.onnx",
                    "*.pb",
                ),
            )
        ).resolve()
        commit = snapshot_path.name
        if snapshot_path.parent.name != "snapshots" or not commit:
            raise RuntimeError(
                "Could not identify the immutable Hugging Face snapshot "
                f"commit from: {snapshot_path}"
            )

        return str(
            snapshot_path,
        ), commit

    @staticmethod
    def _import_torch():
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "DiffuseMix generation requires PyTorch. Install the "
                "generation environment described in docs/diffusemix.md."
            ) from error

        return torch

    def _resolve_device_kind(
        self,
        requested: str,
    ) -> str:
        requested = requested.lower()
        if requested not in {
            "auto",
            "xla",
            "cuda",
            "cpu",
        }:
            raise ValueError(
                "device must be one of auto, xla, cuda, or cpu. "
                f"Got {requested}."
            )
        if requested != "auto":
            return requested
        if _looks_like_xla_environment():
            return "xla"
        if self._torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def _resolve_device(
        self,
        device_kind: str,
    ):
        if device_kind == "xla":
            if self._torch_xla is None:
                raise RuntimeError(
                    "device=xla requires a validated PyTorch-XLA runtime."
                )

            device_fn = getattr(
                self._torch_xla,
                "device",
                None,
            )
            if device_fn is not None:
                return device_fn()

            import torch_xla.core.xla_model as xm

            return xm.xla_device()

        if device_kind == "cuda" and not self._torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda was requested but torch.cuda.is_available() is "
                "false."
            )

        return self._torch.device(
            device_kind,
        )

    def _resolve_dtype(
        self,
        requested: str,
        device_kind: str,
    ):
        requested = requested.lower()
        if requested == "auto":
            requested = {
                "xla": "bfloat16",
                "cuda": "float16",
                "cpu": "float32",
            }[device_kind]

        dtype_by_name = {
            "float32": self._torch.float32,
            "float16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
        }
        if requested not in dtype_by_name:
            raise ValueError(
                "dtype must be auto, float32, float16, or bfloat16. "
                f"Got {requested}."
            )
        if device_kind == "cpu" and requested == "float16":
            raise ValueError(
                "float16 Diffusers inference is not supported on CPU; use "
                "dtype=float32."
            )

        return dtype_by_name[requested]

    def _build_pipeline(self):
        try:
            from diffusers import (
                EulerAncestralDiscreteScheduler,
                StableDiffusionInstructPix2PixPipeline,
            )
        except ImportError as error:
            raise RuntimeError(
                "DiffuseMix generation requires diffusers and transformers. "
                "Install requirements-xla.txt in the PyTorch/XLA "
                "generation environment."
            ) from error

        pipeline_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
        }
        if self.config.disable_safety_checker:
            pipeline_kwargs["safety_checker"] = None

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_source,
            **pipeline_kwargs,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config,
        )
        pipeline = pipeline.to(
            self.device,
        )
        pipeline.set_progress_bar_config(
            disable=True,
        )
        if self.config.attention_slicing:
            pipeline.enable_attention_slicing()

        return pipeline

    def edit(
        self,
        image: Image.Image,
        instruction: str,
        seed: int,
    ) -> Image.Image:
        """Generate one edited image with a deterministic CPU RNG stream."""
        generator = self._torch.Generator(
            device="cpu",
        ).manual_seed(
            int(
                seed,
            )
        )
        # XLA needs ordinary tensor version counters for views such as CLIP's
        # position_ids slice. torch.inference_mode() removes those counters
        # and fails with "Cannot set version_counter for inference tensor".
        # no_grad still disables autograd and is also the context used by the
        # Diffusers pipeline itself.
        with self._inference_context():
            result = self.pipeline(
                prompt=instruction,
                image=image,
                num_images_per_prompt=1,
                guidance_scale=self.config.guidance_scale,
                image_guidance_scale=self.config.image_guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                generator=generator,
            )

        if self._xla_model is not None:
            self._xla_model.mark_step()

        return result.images[0].convert(
            "RGB",
        )

    def _inference_context(self):
        """Return a grad-disabled context compatible with this backend."""
        if self.device_kind == "xla":
            return self._torch.no_grad()

        return self._torch.inference_mode()

    def synchronize(self) -> None:
        """Wait for outstanding lazy XLA work before process exit."""
        if self._xla_model is not None:
            self._xla_model.mark_step()
            self._xla_model.wait_device_ops()
