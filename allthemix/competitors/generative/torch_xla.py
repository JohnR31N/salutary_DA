"""Dependency checks shared by optional PyTorch/XLA generation stages."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from types import ModuleType


def _installed_distribution(
    distribution: str,
) -> tuple[str, str]:
    """Return one distribution's version and root with a useful error."""
    try:
        installed = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"PyTorch/XLA generation requires {distribution}. Install a "
            "pinned environment from requirements-xla.txt."
        ) from error

    return installed.version, str(installed.locate_file(""))


def _public_version(version: str) -> str:
    """Strip a wheel-local suffix while retaining prerelease identity."""
    return version.partition("+")[0]


def _validated_torch_xla_info() -> tuple[str, str, str, str]:
    """Return versions and roots after enforcing exact public versions."""
    torch_version, torch_root = _installed_distribution("torch")
    torch_xla_version, torch_xla_root = _installed_distribution("torch-xla")
    if _public_version(torch_version) != _public_version(torch_xla_version):
        raise RuntimeError(
            "Incompatible PyTorch/PyTorch-XLA binary pair: "
            f"torch={torch_version} from {torch_root}; "
            f"torch-xla={torch_xla_version} from {torch_xla_root}; "
            f"python={sys.executable}. Their public versions must match "
            "exactly. Create a clean virtual environment and install one "
            "pinned requirements-xla.txt environment; do not mix ~/.local and "
            "system packages."
        )

    return torch_version, torch_xla_version, torch_root, torch_xla_root


def validate_torch_xla_versions() -> tuple[str, str]:
    """Fail before loading ``_XLAC`` when Torch/XLA releases cannot match."""
    torch_version, torch_xla_version, _, _ = _validated_torch_xla_info()

    return torch_version, torch_xla_version


def import_torch_xla() -> ModuleType:
    """Import PyTorch/XLA and wrap binary-extension failures with context."""
    from allthemix.utils.backend_environment import validate_xla_environment

    validate_xla_environment()
    (
        torch_version,
        torch_xla_version,
        torch_root,
        torch_xla_root,
    ) = _validated_torch_xla_info()
    try:
        return importlib.import_module("torch_xla")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "PyTorch/XLA could not load its _XLAC extension even though the "
            "installed versions match: "
            f"torch={torch_version} from {torch_root}; "
            f"torch-xla={torch_xla_version} from {torch_xla_root}; "
            f"python={sys.executable}. This usually means wheels from "
            "different environments or C++ ABIs are being mixed. Reinstall "
            "requirements-xla.txt in a clean environment."
        ) from error
