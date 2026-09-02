"""Synthetic end-to-end test for the formal SaliencyMix postflight."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from allthemix.cli.formal_run_protocol import (
    STL10_SALIENCYMIX_FORMAL_V1,
    canonical_json_bytes,
    sha256_directory,
    sha256_file,
    write_json_atomic,
)
from scripts.experiment_run.validate_stl10_saliencymix_formal import validate


def test_postflight_accepts_only_complete_registered_artifacts(
    tmp_path: Path,
) -> None:
    """A fully consistent synthetic run passes every postflight gate."""

    metrics_path = tmp_path / "metrics.csv"
    final_path = tmp_path / "metrics_final_test.csv"
    resolved_path = tmp_path / "resolved.json"
    completion_path = tmp_path / "completion.json"
    checkpoint_path = tmp_path / "checkpoints" / "best"
    checkpoint_path.mkdir(parents=True)
    (checkpoint_path / "state").write_bytes(b"state")
    (tmp_path / "wandb" / "wandb" / "run-test").mkdir(parents=True)
    write_json_atomic(resolved_path, {"schema_version": 1, "arguments": {}})

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "eval_loss",
        "eval_top1_accuracy",
        "eval_top5_accuracy",
        "eval_top1_error",
        "eval_top5_error",
        "best_top1_error",
        "best_epoch",
        "epoch_time",
    ]
    records = []
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, 201):
            error = 1.0 - epoch / 1_000
            row = {
                "epoch": epoch,
                "train_loss": 1.0,
                "train_accuracy": 0.5,
                "eval_loss": 1.0,
                "eval_top1_accuracy": 1.0 - error,
                "eval_top5_accuracy": 0.8,
                "eval_top1_error": error,
                "eval_top5_error": 0.2,
                "best_top1_error": error,
                "best_epoch": epoch,
                "epoch_time": 1.0,
            }
            writer.writerow(row)
            records.append(
                {
                    "epoch": epoch,
                    "train_batches": 39,
                    "vdev_batches": 32,
                    "vdev_examples": 4_000,
                    "train_loss": 1.0,
                    "train_accuracy": 0.5,
                    "vdev_loss": 1.0,
                    "vdev_top1_accuracy": 1.0 - error,
                    "vdev_top5_accuracy": 0.8,
                    "vdev_top1_error": error,
                    "vdev_top5_error": 0.2,
                }
            )
    endpoint_result = {
        "loss": 1.0,
        "top1_accuracy": 0.4,
        "top5_accuracy": 0.8,
        "top1_error": 0.6,
        "top5_error": 0.2,
    }
    with final_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "test_loss",
                "test_top1_accuracy",
                "test_top5_accuracy",
                "test_top1_error",
                "test_top5_error",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "test_loss": endpoint_result["loss"],
                "test_top1_accuracy": endpoint_result["top1_accuracy"],
                "test_top5_accuracy": endpoint_result["top5_accuracy"],
                "test_top1_error": endpoint_result["top1_error"],
                "test_top5_error": endpoint_result["top5_error"],
            }
        )

    expected = {
        "completed_epochs": 200,
        "endpoint_batches": 32,
        "endpoint_builder_calls": 1,
        "endpoint_evaluations": 1,
        "endpoint_examples": 4_000,
        "optimizer_steps": 7_800,
        "train_batches_per_epoch": 39,
        "vdev_batches_per_epoch": 32,
        "vdev_evaluations": 200,
        "vdev_examples_per_epoch": 4_000,
        "vdev_total_batches": 6_400,
    }
    protocol = {
        "schema_version": 1,
        "protocol_id": STL10_SALIENCYMIX_FORMAL_V1,
        "resolved_config_sha256": sha256_file(resolved_path),
        "expected_workload": expected,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "final_test_csv": str(final_path),
            "best_checkpoint": str(checkpoint_path),
            "wandb_dir": str(tmp_path / "wandb"),
        },
    }
    protocol_path = write_json_atomic(tmp_path / "protocol.json", protocol)
    checkpoint_sha = sha256_directory(checkpoint_path)
    completion = {
        "schema_version": 1,
        "status": "SUCCESS",
        "protocol_id": STL10_SALIENCYMIX_FORMAL_V1,
        "protocol_artifact_sha256": sha256_file(protocol_path),
        "resolved_config_sha256": sha256_file(resolved_path),
        "workload": {
            "completed_epochs": 200,
            "optimizer_steps": 7_800,
            "train_batches": 7_800,
            "vdev_evaluations": 200,
            "vdev_batches": 6_400,
            "vdev_examples": 800_000,
        },
        "terminal_optimizer_step": 7_800,
        "best_checkpoint_optimizer_step": 7_800,
        "epochs": records,
        "selection": {
            "best_epoch": 200,
            "best_top1_error": 0.8,
            "selected_checkpoint": str(checkpoint_path),
            "selected_checkpoint_sha256": checkpoint_sha,
        },
        "endpoint": {
            "endpoint_builder_calls": 1,
            "endpoint_evaluations": 1,
            "endpoint_batches": 32,
            "endpoint_examples": 4_000,
            "built_after_training_and_checkpoint_closure": True,
            "result": endpoint_result,
        },
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "final_test_csv": str(final_path),
        },
        "wandb": {
            "enabled": True,
            "mode": "online",
            "run_id": "run-id",
            "url": "https://wandb.example/run-id",
            "finish_completed": True,
        },
    }
    completion["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(completion)
    ).hexdigest()
    write_json_atomic(completion_path, completion)

    result = validate(protocol_path, completion_path)

    assert result["status"] == "SUCCESS"
    assert result["best_epoch"] == 200
    assert result["endpoint_result"] == endpoint_result
