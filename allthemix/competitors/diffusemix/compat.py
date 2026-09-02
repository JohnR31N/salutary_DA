"""Compatibility exports for shared PyTorch/XLA dependency checks."""

from allthemix.competitors.generative.torch_xla import (
    import_torch_xla,
    validate_torch_xla_versions,
)

__all__ = [
    "import_torch_xla",
    "validate_torch_xla_versions",
]
