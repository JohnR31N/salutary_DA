"""Fail-closed provenance and workload checks for registered formal runs."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STL10_SALIENCYMIX_FORMAL_V1 = "stl10_saliencymix_formal_v1"


@dataclass(frozen=True)
class FormalRunContext:
    """Verified immutable inputs for one formal training process."""

    protocol: dict[str, Any]
    protocol_path: Path
    protocol_sha256: str
    resolved_config_path: Path
    resolved_config_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON with one deterministic byte representation."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    """Hash one file without loading it wholly into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Hash a non-empty directory using relative names and file bytes."""

    resolved = Path(path).expanduser().resolve()
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {resolved}")
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.relative_to(resolved).as_posix().encode("utf-8"))
        with file_path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def stl10_tfrecord_fingerprint(data_dir: str | Path) -> str:
    """Reproduce the registered two-stage SHA-256 over STL-10 TFRecords."""

    root = Path(data_dir).expanduser().resolve() / "stl10" / "1.0.0"
    files = [
        *sorted(root.glob("stl10-train.tfrecord-*")),
        *sorted(root.glob("stl10-test.tfrecord-*")),
    ]
    if not files:
        raise FileNotFoundError(f"no registered STL-10 TFRecords found under {root}")
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(
            f"{sha256_file(file_path)}  {file_path}\n".encode()
        )
    return digest.hexdigest()


def resolved_config_payload(args: Namespace) -> dict[str, Any]:
    """Capture every post-parse argument in a JSON-compatible mapping."""

    arguments: dict[str, Any] = {}
    for key, value in vars(args).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            arguments[key] = value
        elif isinstance(value, (list, tuple)):
            arguments[key] = list(value)
        else:
            raise TypeError(
                f"resolved argument {key!r} is not JSON-compatible: "
                f"{type(value).__name__}"
            )
    return {
        "schema_version": 1,
        "arguments": arguments,
    }


def write_json_atomic(path: str | Path, value: Any) -> Path:
    """Write canonical JSON atomically and return the resolved path."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(resolved)
    return resolved


def _require_equal(name: str, observed: Any, expected: Any) -> None:
    """Raise with the exact mismatched field name."""

    if observed != expected:
        raise ValueError(
            f"formal protocol mismatch for {name}: "
            f"observed={observed!r}, expected={expected!r}"
        )


def _validate_clean_checkout(repository: Path, declared_commit: str) -> None:
    """Require the current worktree to be the declared clean commit."""

    observed_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()
    _require_equal("git_commit", observed_commit, declared_commit)
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        text=True,
    ).strip()
    if changed:
        raise RuntimeError("formal run requires a clean checkout:\n" + changed)


def _validate_locked_arguments(args: Namespace, method_name: str) -> None:
    """Enforce the registered STL-10 SaliencyMix training contract."""

    locked = {
        "dataset": (args.dataset, "stl10"),
        "model": (args.model, "preact_resnet18"),
        "method": (method_name, "saliencymix"),
        "resnet_stem_type": (args.resnet_stem_type, "cifar"),
        "preact_stem_bn_relu": (args.preact_stem_bn_relu, False),
        "preact_pytorch_default_init": (
            args.preact_pytorch_default_init,
            False,
        ),
        "batch_size": (args.batch_size, 128),
        "epochs": (args.epochs, 200),
        "max_train_steps": (args.max_train_steps, -1),
        "max_eval_steps": (args.max_eval_steps, -1),
        "seed": (args.seed, 0),
        "data_seed": (args.data_seed, 0),
        "validation_split": (args.validation_split, 0.5),
        "val_source": (args.val_source, "test"),
        "train_subset_fraction": (args.train_subset_fraction, 1.0),
        "val_select_split_fraction": (args.val_select_split_fraction, 0.0),
        "eval_on_test_each_epoch": (args.eval_on_test_each_epoch, False),
        "final_test": (args.final_test, True),
        "final_test_checkpoint": (args.final_test_checkpoint, "best"),
        "learning_rate": (args.learning_rate, 0.1),
        "momentum": (args.momentum, 0.9),
        "nesterov": (args.nesterov, False),
        "weight_decay": (args.weight_decay, 0.0005),
        "lr_schedule": (args.lr_schedule, "cosine"),
        "min_learning_rate": (args.min_learning_rate, 0.0),
        "warmup_epochs": (args.warmup_epochs, 0),
        "lr_decay_epochs": (args.lr_decay_epochs, [100, 150]),
        "lr_decay_rate": (args.lr_decay_rate, 0.1),
        "saliencymix_alpha": (args.saliencymix_alpha, 1.0),
        "saliencymix_prob": (args.saliencymix_prob, 0.5),
        "saliencymix_per_sample": (args.saliencymix_per_sample, False),
        "basic_aug": (args.basic_aug, False),
        "aug_recipe": (args.aug_recipe, "none"),
        "sal_basic_aug": (args.sal_basic_aug, True),
        "sal_aug_recipe": (args.sal_aug_recipe, "basic"),
        "shuffle_buffer_size": (args.shuffle_buffer_size, 10_000),
        "distributed": (args.distributed, True),
        "sync_batch_stats": (args.sync_batch_stats, True),
        "cross_device_shuffle": (args.cross_device_shuffle, False),
        "deterministic_data": (args.deterministic_data, True),
        "strict_determinism": (args.strict_determinism, False),
        "early_stop_enabled": (args.early_stop_enabled, False),
        "debug_train_source": (args.debug_train_source, "none"),
        "save_csv": (args.save_csv, True),
        "save_checkpoint": (args.save_checkpoint, True),
        "save_best_only": (args.save_best_only, True),
        "resume_checkpoint": (args.resume_checkpoint, ""),
        "pretrained_checkpoint": (args.pretrained_checkpoint, ""),
        "run_name": (
            args.run_name,
            "stl10_preact_resnet18_saliencymix",
        ),
        "output_name": (args.output_name, "stl10_saliencymix_seed0.csv"),
        "wandb": (args.wandb, True),
        "wandb_project": (args.wandb_project, "allthemix"),
        "wandb_mode": (args.wandb_mode, "online"),
    }
    mismatches = [
        name for name, (observed, expected) in locked.items() if observed != expected
    ]
    if mismatches:
        details = ", ".join(
            f"{name}={locked[name][0]!r} (required {locked[name][1]!r})"
            for name in mismatches
        )
        raise ValueError("STL-10 SaliencyMix formal protocol mismatch: " + details)


def prepare_formal_run(
    args: Namespace,
    *,
    method_name: str,
    local_device_count: int,
) -> FormalRunContext | None:
    """Load and verify an explicitly supplied formal-run protocol artifact."""

    if not args.run_protocol_path:
        return None
    protocol_path = Path(args.run_protocol_path).expanduser().resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(f"run protocol artifact does not exist: {protocol_path}")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    _require_equal("schema_version", protocol["schema_version"], 1)
    _require_equal("protocol_id", protocol["protocol_id"], STL10_SALIENCYMIX_FORMAL_V1)
    _validate_locked_arguments(args, method_name)
    _require_equal("local_device_count", local_device_count, 4)

    repository = Path(__file__).resolve().parents[2]
    inputs = protocol["inputs"]
    _validate_clean_checkout(repository, inputs["git_commit"])
    _require_equal(
        "wandb_run_name",
        args.wandb_run_name,
        f"stl10_saliencymix_formal_seed0_{inputs['git_commit'][:8]}",
    )

    source_config_path = (repository / inputs["source_config_path"]).resolve()
    _require_equal("config path", Path(args.config).resolve(), source_config_path)
    _require_equal(
        "source_config_sha256",
        sha256_file(source_config_path),
        inputs["source_config_sha256"],
    )

    saliency_path = (Path(args.saliency_dir).expanduser().resolve() /
                     "stl10_train_saliency.npy")
    _require_equal(
        "saliency path",
        saliency_path,
        Path(inputs["saliency_path"]).expanduser().resolve(),
    )
    _require_equal(
        "saliency_sha256",
        sha256_file(saliency_path),
        inputs["saliency_sha256"],
    )
    _require_equal(
        "data_fingerprint",
        stl10_tfrecord_fingerprint(args.data_dir),
        inputs["data_fingerprint"],
    )

    split = protocol["split"]
    split_path = (repository / split["implementation_path"]).resolve()
    _require_equal(
        "split_implementation_sha256",
        sha256_file(split_path),
        split["implementation_sha256"],
    )
    _require_equal("split_data_fingerprint", split["data_fingerprint"], inputs["data_fingerprint"])
    _require_equal("official_test_class_counts", split["official_test_class_counts"], [800] * 10)
    _require_equal("vdev_class_counts", split["vdev_class_counts"], [400] * 10)
    _require_equal("sealed_class_counts", split["sealed_class_counts"], [400] * 10)
    _require_equal("vdev_examples", split["vdev_examples"], 4_000)
    _require_equal("sealed_examples", split["sealed_examples"], 4_000)

    expected = protocol["expected_workload"]
    _require_equal(
        "expected_workload",
        expected,
        {
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
        },
    )

    artifacts = protocol["artifacts"]
    run_dir = protocol_path.parent
    expected_artifact_paths = {
        "resolved_config": run_dir / "resolved_config.json",
        "metrics_csv": run_dir / "metrics" / "stl10_saliencymix_seed0.csv",
        "final_test_csv": (
            run_dir / "metrics" / "stl10_saliencymix_seed0_final_test.csv"
        ),
        "best_checkpoint": (
            run_dir
            / "checkpoints"
            / "stl10_preact_resnet18_saliencymix"
            / "best"
        ),
        "completion": run_dir / "training_complete.json",
        "postflight": run_dir / "postflight.json",
        "wandb_dir": run_dir / "wandb",
    }
    for artifact_name, expected_path in expected_artifact_paths.items():
        _require_equal(
            f"artifact.{artifact_name}",
            Path(artifacts[artifact_name]).expanduser().resolve(),
            expected_path.resolve(),
        )
    _require_equal(
        "metrics_csv",
        (Path(args.output_dir) / args.output_name).resolve(),
        Path(artifacts["metrics_csv"]).resolve(),
    )
    expected_checkpoint = (
        Path(args.checkpoint_dir).resolve()
        / "stl10_preact_resnet18_saliencymix"
        / "best"
    )
    _require_equal(
        "best_checkpoint",
        expected_checkpoint,
        Path(artifacts["best_checkpoint"]).resolve(),
    )

    resolved_payload = resolved_config_payload(args)
    resolved_bytes = canonical_json_bytes(resolved_payload)
    resolved_sha256 = hashlib.sha256(resolved_bytes).hexdigest()
    _require_equal(
        "resolved_config_sha256",
        resolved_sha256,
        protocol["resolved_config_sha256"],
    )
    resolved_path = Path(artifacts["resolved_config"]).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"pre-registered resolved config does not exist: {resolved_path}"
        )
    if resolved_path.read_bytes() != resolved_bytes:
        raise ValueError("pre-registered resolved config bytes do not match runtime args")

    return FormalRunContext(
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
        resolved_config_path=resolved_path,
        resolved_config_sha256=resolved_sha256,
    )


def validate_pre_endpoint_workload(
    context: FormalRunContext,
    *,
    epoch_records: list[dict[str, Any]],
    terminal_optimizer_step: int,
    best_epoch: int,
    best_top1_error: float,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Validate all training and Vdev work before opening the endpoint split."""

    expected = context.protocol["expected_workload"]
    _require_equal("completed_epochs", len(epoch_records), expected["completed_epochs"])
    _require_equal("terminal_optimizer_step", terminal_optimizer_step, expected["optimizer_steps"])
    _require_equal(
        "epoch_order",
        [record["epoch"] for record in epoch_records],
        list(range(1, expected["completed_epochs"] + 1)),
    )
    for record in epoch_records:
        _require_equal(
            f"epoch_{record['epoch']}.train_batches",
            record["train_batches"],
            expected["train_batches_per_epoch"],
        )
        _require_equal(
            f"epoch_{record['epoch']}.vdev_batches",
            record["vdev_batches"],
            expected["vdev_batches_per_epoch"],
        )
        _require_equal(
            f"epoch_{record['epoch']}.vdev_examples",
            record["vdev_examples"],
            expected["vdev_examples_per_epoch"],
        )
        for metric_name in (
            "train_loss",
            "train_accuracy",
            "vdev_loss",
            "vdev_top1_accuracy",
            "vdev_top5_accuracy",
            "vdev_top1_error",
            "vdev_top5_error",
        ):
            if not math.isfinite(float(record[metric_name])):
                raise RuntimeError(
                    f"epoch {record['epoch']} has non-finite {metric_name}"
                )

    errors = [float(record["vdev_top1_error"]) for record in epoch_records]
    strict_best_error = float("inf")
    strict_best_epoch = -1
    for index, error in enumerate(errors, start=1):
        if error < strict_best_error:
            strict_best_error = error
            strict_best_epoch = index
    _require_equal("best_epoch", best_epoch, strict_best_epoch)
    _require_equal("best_top1_error", best_top1_error, strict_best_error)

    best_path = Path(checkpoint_path).expanduser().resolve() / "best"
    _require_equal(
        "selected_checkpoint_path",
        best_path,
        Path(context.protocol["artifacts"]["best_checkpoint"]).resolve(),
    )
    checkpoint_sha256 = sha256_directory(best_path)
    return {
        "completed_epochs": len(epoch_records),
        "optimizer_steps": terminal_optimizer_step,
        "train_batches": sum(record["train_batches"] for record in epoch_records),
        "vdev_evaluations": len(epoch_records),
        "vdev_batches": sum(record["vdev_batches"] for record in epoch_records),
        "vdev_examples": sum(record["vdev_examples"] for record in epoch_records),
        "best_epoch": best_epoch,
        "best_top1_error": best_top1_error,
        "selected_checkpoint": str(best_path),
        "selected_checkpoint_sha256": checkpoint_sha256,
    }


__all__ = [
    "STL10_SALIENCYMIX_FORMAL_V1",
    "FormalRunContext",
    "canonical_json_bytes",
    "prepare_formal_run",
    "resolved_config_payload",
    "sha256_directory",
    "sha256_file",
    "stl10_tfrecord_fingerprint",
    "validate_pre_endpoint_workload",
    "write_json_atomic",
]
