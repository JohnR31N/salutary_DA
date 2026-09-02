from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from allthemix.competitors.diffusemix import compat


def _mock_versions(
    monkeypatch: pytest.MonkeyPatch,
    *,
    torch: str,
    torch_xla: str,
) -> None:
    versions = {
        "torch": torch,
        "torch-xla": torch_xla,
    }

    def distribution(
        name: str,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            version=versions[name],
            locate_file=lambda _: Path(
                "/venv/site-packages",
            ),
        )

    def version(name: str) -> str:
        if name in {"jax", "jaxlib"}:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        distribution,
    )
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        version,
    )


def test_validate_torch_xla_versions_accepts_matching_public_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_versions(
        monkeypatch,
        torch="2.9.0+cpu",
        torch_xla="2.9.0",
    )

    assert compat.validate_torch_xla_versions() == (
        "2.9.0+cpu",
        "2.9.0",
    )


def test_validate_torch_xla_versions_rejects_mismatched_public_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_versions(
        monkeypatch,
        torch="2.13.0",
        torch_xla="2.9.0",
    )

    with pytest.raises(
        RuntimeError,
        match=r"torch=2\.13\.0.*torch-xla=2\.9\.0",
    ):
        compat.validate_torch_xla_versions()


def test_validate_torch_xla_versions_rejects_patch_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_versions(
        monkeypatch,
        torch="2.9.1",
        torch_xla="2.9.0",
    )

    with pytest.raises(
        RuntimeError,
        match="public versions must match exactly",
    ):
        compat.validate_torch_xla_versions()


def test_validate_torch_xla_versions_reports_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(
        name: str,
    ) -> None:
        raise importlib.metadata.PackageNotFoundError(
            name,
        )

    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        missing_distribution,
    )

    with pytest.raises(
        RuntimeError,
        match="requirements-xla.txt",
    ):
        compat.validate_torch_xla_versions()


def test_import_torch_xla_wraps_binary_extension_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_versions(
        monkeypatch,
        torch="2.9.0",
        torch_xla="2.9.0",
    )

    def fail_import(
        name: str,
    ):
        assert name == "torch_xla"
        raise ImportError(
            "undefined symbol",
        )

    monkeypatch.setattr(
        importlib,
        "import_module",
        fail_import,
    )

    with pytest.raises(
        RuntimeError,
        match=r"different environments or C\+\+ ABIs",
    ):
        compat.import_torch_xla()


def test_import_torch_xla_rejects_jax_runtime_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_versions(
        monkeypatch,
        torch="2.9.0",
        torch_xla="2.9.0",
    )
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

    with pytest.raises(RuntimeError, match="also contains JAX"):
        compat.import_torch_xla()
