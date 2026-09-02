"""Tests for fail-closed formal-run provenance and workload accounting."""

from __future__ import annotations

import hashlib
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from allthemix.cli.formal_run_protocol import (
    FormalRunContext,
    canonical_json_bytes,
    resolved_config_payload,
    stl10_tfrecord_fingerprint,
    validate_pre_endpoint_workload,
)
from scripts.experiment_run.prepare_stl10_saliencymix_formal import (
    SOURCE_CONFIG_SHA256,
)


def test_resolved_config_payload_preserves_all_argument_values() -> None:
    """Canonical config serialization includes nulls and normalizes tuples."""

    args = Namespace(alpha=0.5, enabled=True, names=("a", "b"), unset=None)
    payload = resolved_config_payload(args)

    assert payload == {
        "schema_version": 1,
        "arguments": {
            "alpha": 0.5,
            "enabled": True,
            "names": ["a", "b"],
            "unset": None,
        },
    }
    assert canonical_json_bytes(payload).endswith(b"\n")


def test_registered_source_config_hash_matches_committed_shared_config() -> None:
    """The registered source hash is derived from the exact Git blob bytes."""

    repository = Path(__file__).resolve().parents[2]
    config_path = repository / "configs/stl10/preact_resnet18/saliencymix.yaml"
    blob_bytes = subprocess.check_output(
        [
            "git",
            "show",
            "HEAD:configs/stl10/preact_resnet18/saliencymix.yaml",
        ],
        cwd=repository,
    )

    assert hashlib.sha256(blob_bytes).hexdigest() == SOURCE_CONFIG_SHA256
    assert b"validation_split: 0.5\n" in blob_bytes
    assert "validation_split: 0.5" in config_path.read_text(encoding="utf-8")


def test_stl10_tfrecord_fingerprint_matches_registered_two_stage_hash(
    tmp_path: Path,
) -> None:
    """The Python fingerprint exactly hashes sha256sum-style input lines."""

    root = tmp_path / "stl10" / "1.0.0"
    root.mkdir(parents=True)
    train_path = root / "stl10-train.tfrecord-00000-of-00001"
    test_path = root / "stl10-test.tfrecord-00000-of-00001"
    train_path.write_bytes(b"train")
    test_path.write_bytes(b"test")
    digest = hashlib.sha256()
    for path in (train_path, test_path):
        file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{file_sha}  {path.resolve()}\n".encode())

    assert stl10_tfrecord_fingerprint(tmp_path) == digest.hexdigest()


def _formal_context(tmp_path: Path) -> FormalRunContext:
    """Build a minimal verified context for workload-only tests."""

    return FormalRunContext(
        protocol={
            "expected_workload": {
                "completed_epochs": 200,
                "optimizer_steps": 7_800,
                "train_batches_per_epoch": 39,
                "vdev_batches_per_epoch": 32,
                "vdev_examples_per_epoch": 4_000,
            },
            "artifacts": {
                "best_checkpoint": str(tmp_path / "checkpoints" / "best"),
            },
        },
        protocol_path=tmp_path / "protocol.json",
        protocol_sha256="protocol",
        resolved_config_path=tmp_path / "resolved.json",
        resolved_config_sha256="config",
    )


def _epoch_records() -> list[dict[str, float | int]]:
    """Create 200 finite records with a unique strict best at epoch 200."""

    return [
        {
            "epoch": epoch,
            "train_batches": 39,
            "vdev_batches": 32,
            "vdev_examples": 4_000,
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "vdev_loss": 1.0,
            "vdev_top1_accuracy": epoch / 1_000,
            "vdev_top5_accuracy": 0.8,
            "vdev_top1_error": 1.0 - epoch / 1_000,
            "vdev_top5_error": 0.2,
        }
        for epoch in range(1, 201)
    ]


def test_pre_endpoint_workload_uses_observed_counts_and_checkpoint(
    tmp_path: Path,
) -> None:
    """A complete exact workload closes before the endpoint is available."""

    checkpoint = tmp_path / "checkpoints" / "best"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state").write_bytes(b"checkpoint")
    records = _epoch_records()
    closure = validate_pre_endpoint_workload(
        _formal_context(tmp_path),
        epoch_records=records,
        terminal_optimizer_step=7_800,
        best_epoch=200,
        best_top1_error=float(records[-1]["vdev_top1_error"]),
        checkpoint_path=tmp_path / "checkpoints",
    )

    assert closure["train_batches"] == 7_800
    assert closure["vdev_batches"] == 6_400
    assert closure["vdev_examples"] == 800_000
    assert closure["best_epoch"] == 200


def test_pre_endpoint_workload_rejects_one_missing_vdev_batch(
    tmp_path: Path,
) -> None:
    """One under-counted Vdev epoch prevents endpoint construction."""

    checkpoint = tmp_path / "checkpoints" / "best"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state").write_bytes(b"checkpoint")
    records = _epoch_records()
    records[17]["vdev_batches"] = 31

    with pytest.raises(ValueError, match="epoch_18.vdev_batches"):
        validate_pre_endpoint_workload(
            _formal_context(tmp_path),
            epoch_records=records,
            terminal_optimizer_step=7_800,
            best_epoch=200,
            best_top1_error=float(records[-1]["vdev_top1_error"]),
            checkpoint_path=tmp_path / "checkpoints",
        )
