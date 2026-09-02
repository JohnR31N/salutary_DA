"""Validate the isolated JAX and PyTorch/XLA runtime environments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_JAX_DISTRIBUTIONS = ("jax", "jaxlib")
_XLA_DISTRIBUTIONS = ("torch", "torch-xla", "libtpu")


def _optional_distribution_version(
    name: str,
) -> str | None:
    try:
        return importlib.metadata.version(
            name,
        )
    except importlib.metadata.PackageNotFoundError:
        return None


def _distribution_root(name: str) -> Path | None:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None

    return Path(distribution.locate_file("")).resolve()


def _public_version(version: str) -> str:
    return version.split("+", maxsplit=1)[0]


def _require_distributions(names: Iterable[str], backend: str) -> None:
    missing = [
        name
        for name in names
        if _optional_distribution_version(name) is None
    ]
    if missing:
        raise RuntimeError(
            f"The {backend} environment is incomplete. Missing: "
            f"{', '.join(missing)}. Install the matching requirements file "
            "with this environment's absolute Python interpreter."
        )


def _validate_expected_prefix(
    expected_prefix: str | Path,
    distributions: Iterable[str],
) -> None:
    expected = Path(expected_prefix).expanduser().resolve()
    actual = Path(sys.prefix).resolve()
    if actual != expected:
        raise RuntimeError(
            "The active interpreter does not belong to the requested virtual "
            f"environment: expected sys.prefix={expected}, got {actual}; "
            f"python={sys.executable}."
        )

    outside = []
    for name in distributions:
        root = _distribution_root(name)
        if root is not None and not root.is_relative_to(expected):
            outside.append(f"{name}={root}")

    if outside:
        raise RuntimeError(
            "Backend packages are leaking in from outside the virtual "
            f"environment {expected}: {', '.join(outside)}. Recreate the "
            "environment with system site packages disabled."
        )


def validate_jax_environment() -> None:
    """Reject a Torch/XLA installation before JAX initializes PJRT."""
    torch_xla_version = _optional_distribution_version(
        "torch-xla",
    )
    if torch_xla_version is None:
        return

    jax_version = _optional_distribution_version(
        "jax",
    )
    libtpu_version = _optional_distribution_version(
        "libtpu",
    )
    raise RuntimeError(
        "JAX classifier training cannot use an environment that also "
        "contains PyTorch/XLA. The two backends install incompatible TPU "
        "runtime plugins and can abort before Python reports an exception. "
        f"python={sys.executable}; jax={jax_version or 'missing'}; "
        f"torch-xla={torch_xla_version}; "
        f"libtpu={libtpu_version or 'missing'}. Activate the isolated JAX "
        "environment and install requirements-jax.txt there; keep "
        "requirements-xla.txt in a separate environment."
    )


def validate_xla_environment() -> None:
    """Reject JAX packages and mismatched Torch/XLA public versions."""
    jax_version = _optional_distribution_version("jax")
    jaxlib_version = _optional_distribution_version("jaxlib")
    if jax_version is not None or jaxlib_version is not None:
        raise RuntimeError(
            "PyTorch/XLA generation cannot use an environment that also "
            "contains JAX. The two backends install incompatible TPU runtime "
            f"plugins. python={sys.executable}; "
            f"jax={jax_version or 'missing'}; "
            f"jaxlib={jaxlib_version or 'missing'}; "
            "torch-xla="
            f"{_optional_distribution_version('torch-xla') or 'missing'}; "
            f"libtpu={_optional_distribution_version('libtpu') or 'missing'}."
        )

    torch_version = _optional_distribution_version("torch")
    torch_xla_version = _optional_distribution_version("torch-xla")
    if torch_version is None or torch_xla_version is None:
        return
    if _public_version(torch_version) != _public_version(torch_xla_version):
        raise RuntimeError(
            "PyTorch and PyTorch/XLA public versions must match exactly: "
            f"torch={torch_version}; torch-xla={torch_xla_version}."
        )


def validate_backend_environment(
    backend: str,
    *,
    expected_prefix: str | Path | None = None,
) -> dict[str, Any]:
    """Validate installed packages and return a serializable inventory."""
    normalized = backend.strip().lower()
    if normalized == "jax":
        validate_jax_environment()
        required = _JAX_DISTRIBUTIONS
    elif normalized == "xla":
        validate_xla_environment()
        required = _XLA_DISTRIBUTIONS
    else:
        raise ValueError(f"Unsupported backend: {backend!r}")

    _require_distributions(required, normalized)
    if expected_prefix is not None:
        _validate_expected_prefix(expected_prefix, required)

    all_names = (*_JAX_DISTRIBUTIONS, *_XLA_DISTRIBUTIONS)
    return {
        "backend": normalized,
        "python": sys.executable,
        "prefix": sys.prefix,
        "packages": {
            name: _optional_distribution_version(name)
            for name in all_names
        },
        "package_roots": {
            name: str(root) if root is not None else None
            for name in all_names
            for root in [_distribution_root(name)]
        },
    }


def probe_backend_runtime(
    backend: str,
    *,
    require_tpu: bool = False,
) -> dict[str, Any]:
    """Initialize the selected runtime in a short-lived process."""
    normalized = backend.strip().lower()
    if normalized == "jax":
        import jax
        import jax.numpy as jnp

        devices = jax.devices()
        value = jax.jit(lambda x: x + 1)(jnp.asarray(1.0))
        value.block_until_ready()
        platforms = sorted({device.platform for device in devices})
        if require_tpu and "tpu" not in platforms:
            raise RuntimeError(
                f"JAX initialized without a TPU: platforms={platforms}."
            )
        return {
            "device_count": len(devices),
            "devices": [str(device) for device in devices],
            "platforms": platforms,
        }

    if normalized == "xla":
        import torch
        import torch_xla

        if hasattr(torch_xla, "device"):
            device = torch_xla.device()
        else:
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
        value = (torch.ones(1, device=device) + 1).cpu().item()
        device_name = str(device)
        if require_tpu and not device_name.lower().startswith("xla"):
            raise RuntimeError(
                f"PyTorch/XLA initialized without an XLA device: {device}."
            )
        return {
            "device": device_name,
            "probe_value": value,
            "torch": torch.__version__,
            "torch_xla": torch_xla.__version__,
        }

    raise ValueError(f"Unsupported backend: {backend!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an isolated AllTheMix backend environment.",
    )
    parser.add_argument("--expected", choices=("jax", "xla"), required=True)
    parser.add_argument("--expected-prefix", default=None)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--require-tpu", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        report = validate_backend_environment(
            args.expected,
            expected_prefix=args.expected_prefix,
        )
        if args.runtime:
            report["runtime"] = probe_backend_runtime(
                args.expected,
                require_tpu=args.require_tpu,
            )
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"Backend environment check failed: {error}") from error

    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
