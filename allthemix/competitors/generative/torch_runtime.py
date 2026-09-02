"""Lazy Torch runtime shared by offline foundation-model inference stages."""

from __future__ import annotations

import hashlib
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any


def looks_like_xla_environment() -> bool:
    """Return whether process environment variables explicitly select XLA."""
    return bool(
        os.environ.get("PJRT_DEVICE", "").upper()
        or os.environ.get("XRT_TPU_CONFIG")
        or os.environ.get("TPU_NAME")
    )


class TorchRuntime:
    """Resolve optional Torch devices without importing them in JAX runs."""

    def __init__(
        self,
        device: str = "auto",
        dtype: str = "auto",
        xla_cache_dir: str = "",
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Offline generative competitors require PyTorch in their "
                "separate generation environment."
            ) from error

        self.torch = torch
        self.device_kind = self._resolve_device_kind(device)
        self._torch_xla = None
        self._xla_model = None
        if self.device_kind == "xla":
            # This established compatibility gate gives a much clearer error
            # than an unresolved symbol from the binary extension.
            from allthemix.competitors.generative.torch_xla import (
                import_torch_xla,
            )

            self._torch_xla = import_torch_xla()
            if xla_cache_dir:
                import torch_xla.runtime as xr

                cache_dir = Path(xla_cache_dir).expanduser().resolve()
                cache_dir.mkdir(parents=True, exist_ok=True)
                xr.initialize_cache(str(cache_dir), readonly=False)
            import torch_xla.core.xla_model as xm

            self._xla_model = xm
        self.device = self._resolve_device()
        self.dtype = self._resolve_dtype(dtype)

    def _resolve_device_kind(self, requested: str) -> str:
        """Resolve auto, XLA, CUDA, or CPU into one backend name."""
        value = requested.lower()
        if value not in {"auto", "xla", "cuda", "cpu"}:
            raise ValueError(
                "device must be auto, xla, cuda, or cpu. "
                f"Got {requested!r}."
            )
        if value != "auto":
            return value
        if looks_like_xla_environment():
            return "xla"
        if self.torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def _resolve_device(self):
        """Create the selected torch.device or XLA device handle."""
        if self.device_kind == "xla":
            device_fn = getattr(self._torch_xla, "device", None)
            if device_fn is not None:
                return device_fn()

            return self._xla_model.xla_device()
        if self.device_kind == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda was requested but torch.cuda.is_available() is "
                "false."
            )

        return self.torch.device(self.device_kind)

    def _resolve_dtype(self, requested: str):
        """Choose a backend-appropriate floating-point inference dtype."""
        value = requested.lower()
        if value == "auto":
            value = {
                "xla": "bfloat16",
                "cuda": "float16",
                "cpu": "float32",
            }[self.device_kind]
        dtypes = {
            "float32": self.torch.float32,
            "float16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
        }
        if value not in dtypes:
            raise ValueError(
                "dtype must be auto, float32, float16, or bfloat16. "
                f"Got {requested!r}."
            )
        if self.device_kind == "cpu" and value == "float16":
            raise ValueError("CPU inference requires float32 or bfloat16.")

        return dtypes[value]

    def inference_context(self) -> AbstractContextManager[Any]:
        """Disable gradients while preserving XLA tensor version counters."""
        if self.device_kind == "xla":
            return self.torch.no_grad()

        return self.torch.inference_mode()

    def move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Move processor outputs and cast floating tensors consistently."""
        moved = {}
        for key, value in inputs.items():
            value = value.to(self.device)
            if getattr(value, "is_floating_point", lambda: False)():
                value = value.to(self.dtype)
            moved[key] = value

        return moved

    def cpu_generator(self, seed: int):
        """Create a deterministic generator accepted by Diffusers backends."""
        return self.torch.Generator(device="cpu").manual_seed(int(seed))

    def mark_step(self) -> None:
        """Materialize pending lazy XLA operations when applicable."""
        if self._xla_model is not None:
            self._xla_model.mark_step()

    def synchronize(self) -> None:
        """Wait for outstanding XLA work before publishing completion."""
        if self._xla_model is not None:
            self._xla_model.mark_step()
            self._xla_model.wait_device_ops()


def resolve_huggingface_snapshot(
    model_id: str,
    revision: str = "",
    include_pytorch_bin: bool = False,
) -> tuple[str, str]:
    """Resolve a local model or Hub revision to an immutable source."""
    local_path = Path(model_id).expanduser()
    if local_path.is_dir():
        root = local_path.resolve()
        digest = hashlib.sha256()
        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            stat = file_path.stat()
            digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))

        return str(root), f"local-metadata-sha256:{digest.hexdigest()}"

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Resolving offline foundation models requires huggingface_hub."
        ) from error
    ignore_patterns = [
        "*.ckpt",
        "*.h5",
        "*.msgpack",
        "*.onnx",
        "*.pb",
    ]
    if not include_pytorch_bin:
        ignore_patterns.insert(0, "*.bin")

    snapshot_path = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision or None,
            ignore_patterns=tuple(ignore_patterns),
        )
    ).resolve()
    commit = snapshot_path.name
    if snapshot_path.parent.name != "snapshots" or not commit:
        raise RuntimeError(
            f"Could not identify immutable Hugging Face snapshot: {snapshot_path}"
        )

    return str(snapshot_path), commit
