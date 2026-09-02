from __future__ import annotations

import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from allthemix.utils import backend_environment
from allthemix.utils.backend_environment import (
    validate_backend_environment,
    validate_jax_environment,
    validate_xla_environment,
)


def test_jax_environment_accepts_missing_torch_xla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def version(name: str) -> str:
        if name == "torch-xla":
            raise importlib.metadata.PackageNotFoundError(
                name,
            )

        return "0.6.2"

    monkeypatch.setattr(
        importlib.metadata,
        "version",
        version,
    )

    validate_jax_environment()


def test_jax_environment_rejects_torch_xla_libtpu_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "jax": "0.6.2",
        "torch-xla": "2.9.0",
        "libtpu": "0.0.21",
    }
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        versions.__getitem__,
    )

    with pytest.raises(
        RuntimeError,
        match=r"jax=0\.6\.2.*torch-xla=2\.9\.0.*libtpu=0\.0\.21",
    ):
        validate_jax_environment()


def test_xla_environment_accepts_matching_pair_without_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "torch": "2.9.0+cpu",
        "torch-xla": "2.9.0",
        "libtpu": "0.0.21",
    }

    def version(name: str) -> str:
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", version)

    validate_xla_environment()


def test_xla_environment_rejects_jax_libtpu_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "jax": "0.6.2",
        "jaxlib": "0.6.2",
        "torch": "2.9.0",
        "torch-xla": "2.9.0",
        "libtpu": "0.0.21",
    }
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        versions.__getitem__,
    )

    with pytest.raises(RuntimeError, match=r"also contains JAX"):
        validate_xla_environment()


def test_backend_environment_rejects_package_outside_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = tmp_path / "jax-env"
    environment.mkdir()
    outside = tmp_path / "user-site"
    outside.mkdir()

    versions = {
        "jax": "0.6.2",
        "jaxlib": "0.6.2",
    }

    def version(name: str) -> str:
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    def distribution(name: str) -> SimpleNamespace:
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return SimpleNamespace(locate_file=lambda _: outside)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(backend_environment.sys, "prefix", str(environment))

    with pytest.raises(RuntimeError, match="leaking in from outside"):
        validate_backend_environment(
            "jax",
            expected_prefix=environment,
        )
