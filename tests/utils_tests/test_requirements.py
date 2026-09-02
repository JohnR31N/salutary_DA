"""Protect the dependency boundary between JAX and PyTorch/XLA."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _requirement_lines(filename: str) -> list[str]:
    return [
        line.split("#", 1)[0].strip()
        for line in (ROOT / filename).read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]


def _package_names(filename: str) -> set[str]:
    names = set()
    for line in _requirement_lines(filename):
        if line.startswith("-"):
            continue
        name = re.split(r"[<>=!~;\[]", line, maxsplit=1)[0]
        names.add(name.lower().replace("_", "-"))

    return names


def test_backend_requirements_share_only_the_common_layer() -> None:
    common = _package_names("requirements-common.txt")
    jax = _package_names("requirements-jax.txt")
    xla = _package_names("requirements-xla.txt")

    assert not common & {"jax", "jaxlib", "libtpu", "torch", "torch-xla"}
    assert {"jax", "jaxlib", "flax", "optax", "absl-py"} <= jax
    assert not jax & {"torch", "torch-xla"}
    assert {"torch", "torch-xla", "diffusers", "transformers"} <= xla
    assert not xla & {"jax", "jaxlib", "flax", "optax"}
    assert "-r requirements-common.txt" in _requirement_lines(
        "requirements-jax.txt"
    )
    assert "-r requirements-common.txt" in _requirement_lines(
        "requirements-xla.txt"
    )


def test_legacy_and_development_entries_target_jax() -> None:
    assert _requirement_lines("requirements.txt") == [
        "-r requirements-jax.txt"
    ]
    assert "-r requirements-jax.txt" in _requirement_lines(
        "requirements-dev.txt"
    )
    assert "pytest" in _package_names("requirements-dev.txt")


def test_xla_stack_pins_compatible_saspa_versions() -> None:
    lines = set(_requirement_lines("requirements-xla.txt"))

    assert "setuptools==80.9.0" in lines
    assert "torch==2.9.0" in lines
    assert "torch-xla[tpu]==2.9.0" in lines
    assert "torchvision==0.24.0" in lines
    assert "diffusers==0.32.2" in lines
    assert "transformers==4.48.3" in lines
    assert "openai-clip==1.0.1" in lines
