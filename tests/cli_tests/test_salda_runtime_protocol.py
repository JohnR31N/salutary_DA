from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from allthemix.cli.train import (
    SALDA_GA_METRIC_NAMES,
    SALDA_GA_NOOP_METRICS,
    _build_salda_endpoint_after_closure,
    _canonical_salda_json_sha256,
    _expected_salda_direction_counts,
    _expected_salda_direction_example_visits,
    _inject_salda_noop_epoch_metrics,
    _materialize_salda_validation,
    _namespace_extra_metrics,
    _salda_data_protocol_from_run_protocol,
    _salda_data_protocol_payload,
    _salda_direction_workload_by_epoch,
    _salda_directory_sha256,
    _salda_runtime_config_payload,
    _salda_timing_target_payload,
    _salda_timing_workload_closure_payload,
    _salda_training_recipe_from_runtime_config,
    _salda_validation_direction_config,
    _summarize_salda_actions,
    _summarize_salda_epoch_timing,
    _validate_salda_best_checkpoint_pre_endpoint,
    _validate_salda_checkout,
    _validate_salda_completion_schema,
    _validate_salda_completion_workload,
    _validate_salda_pre_endpoint_workload,
    _validate_stl10_direction_prerequisite,
    _wandb_extra_metrics,
)
from allthemix.utils.experiment_logger import append_epoch_result, write_csv_header
from salutary_da.checkpoint_selection import (
    STRICT_VDEV_TOP1_ERROR_RULE,
    should_replace_best_vdev_top1_error,
)
from salutary_da.protocol import (
    CIFAR100_INSTANTANEOUS_GA_PROTOCOL,
    STL10_INSTANTANEOUS_GA_PROTOCOL,
)


def _finite_vtest_result() -> dict[str, float]:
    """Return the exact finite endpoint metric mapping emitted by the trainer."""

    return {
        "loss": 1.0,
        "top1_accuracy": 0.5,
        "top5_accuracy": 0.9,
        "top1_error": 0.5,
        "top5_error": 0.1,
    }


def _salda_args(monkeypatch: pytest.MonkeyPatch, *extra: str):
    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            "configs/cifar100/preact_resnet18/salda_ga.yaml",
            "--salda_ga_git_commit",
            "a" * 40,
            "--salda_ga_stop_epoch",
            "10",
            "--final_test",
            "false",
            *extra,
        ],
    )
    return parse_args()


def _stl10_salda_args(monkeypatch: pytest.MonkeyPatch, *extra: str):
    """Parse one registered STL-10 GA short-run configuration."""

    from allthemix.cli.args import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            "configs/stl10/preact_resnet18/salda_ga.yaml",
            "--salda_ga_git_commit",
            "a" * 40,
            "--salda_ga_stop_epoch",
            "10",
            "--final_test",
            "false",
            *extra,
        ],
    )
    return parse_args()


def test_runtime_config_is_shared_with_shuffled_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = _salda_args(monkeypatch, "--salda_ga_mode", "soft_label")
    shuffled = _salda_args(
        monkeypatch,
        "--salda_ga_mode",
        "shuffled_soft_label",
    )
    direct_config = _salda_runtime_config_payload(direct, method_name="mixup")
    shuffled_config = _salda_runtime_config_payload(shuffled, method_name="mixup")
    assert direct_config == shuffled_config
    assert direct_config["gradient_alignment"]["policy_mode"] == "soft_label"
    assert direct_config["gradient_alignment"]["validation_direction_layout"] == (
        "single_complete_batch"
    )
    assert direct_config["gradient_alignment"]["validation_examples_per_device"] == (
        1_250
    )
    assert direct_config["gradient_alignment"]["validation_direction_mode"] == ("full")
    assert (
        direct_config["gradient_alignment"]["validation_direction_main_table_eligible"]
        is True
    )
    assert (
        direct_config["gradient_alignment"][
            "validation_examples_per_gradient_evaluation"
        ]
        == 5_000
    )
    assert direct_config["gradient_alignment"]["validation_reanchor_interval"] is (None)
    assert direct_config["gradient_alignment"]["validation_batch_seed_policy"] is (None)
    assert direct_config["gradient_alignment"]["validation_reanchor_examples"] is (None)
    assert "validation_chunk_size" not in direct_config["gradient_alignment"]
    recipe = _salda_training_recipe_from_runtime_config(direct_config)
    assert "gradient_alignment" not in recipe


def test_balanced_batch_aggregate_protocol_and_exact_visit_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seal the 500-row cached mean and its periodic 5000-row reanchor."""

    args = _salda_args(
        monkeypatch,
        "--salda_ga_mode",
        "soft_label",
        "--salda_ga_validation_direction_mode",
        "batch_aggregate",
    )
    direction = _salda_validation_direction_config(args)
    assert direction == {
        "validation_direction_mode": "batch_aggregate",
        "validation_direction_layout": "class_balanced_cached_gradient_mean",
        "validation_direction_pool_examples": 5_000,
        "validation_examples_per_gradient_evaluation": 500,
        "validation_examples_per_device": 125,
        "validation_direction_cycle_length": 10,
        "validation_direction_schedule": (
            "cyclic_component_replacement_with_exact_reanchor"
        ),
        "validation_direction_main_table_eligible": False,
        "validation_batch_size": 500,
        "validation_batch_seed_policy": "training_seed",
        "validation_reanchor_interval": 50,
        "validation_reanchor_examples": 5_000,
    }
    assert (
        _expected_salda_direction_example_visits(
            mode="full",
            updates=390,
            validation_batch_size=500,
            reanchor_interval=50,
        )
        == 1_950_000
    )
    assert _expected_salda_direction_counts(
        mode="full",
        updates=390,
        validation_batch_size=500,
        reanchor_interval=50,
    ) == (390, 0)
    assert (
        _expected_salda_direction_example_visits(
            mode="batch_aggregate",
            updates=390,
            validation_batch_size=500,
            reanchor_interval=50,
        )
        == 231_000
    )
    assert (
        _expected_salda_direction_example_visits(
            mode="batch_aggregate",
            updates=3_900,
            validation_batch_size=500,
            reanchor_interval=50,
        )
        == 2_301_000
    )
    assert _expected_salda_direction_counts(
        mode="batch_aggregate",
        updates=390,
        validation_batch_size=500,
        reanchor_interval=50,
    ) == (462, 8)
    workload = _salda_direction_workload_by_epoch(
        mode="batch_aggregate",
        epochs=10,
        updates_per_epoch=390,
        direction_active=True,
        validation_batch_size=500,
        reanchor_interval=50,
    )
    assert len(workload) == 10
    assert workload[-1]["epoch"] == 10
    assert [row["validation_exact_reanchors"] for row in workload] == [
        8,
        8,
        8,
        8,
        7,
        8,
        8,
        8,
        8,
        7,
    ]
    assert [row["direction_validation_example_visits"] for row in workload] == [
        231_000,
        231_000,
        231_000,
        231_000,
        226_500,
        231_000,
        231_000,
        231_000,
        231_000,
        226_500,
    ]
    assert [row["validation_gradient_evaluations"] for row in workload] == [
        462,
        462,
        462,
        462,
        453,
        462,
        462,
        462,
        462,
        453,
    ]
    assert sum(row["validation_exact_reanchors"] for row in workload) == 78
    assert sum(row["validation_gradient_evaluations"] for row in workload) == 4_602
    assert (
        sum(row["direction_validation_example_visits"] for row in workload) == 2_301_000
    )


def test_stl10_origin_impulse_scores_only_the_configured_epoch() -> None:
    """Count one 39-update GA window between twenty warmup and nine followups."""

    workload = _salda_direction_workload_by_epoch(
        mode="full",
        epochs=30,
        updates_per_epoch=39,
        direction_active=True,
        score_start_optimizer_step=20 * 39,
        score_stop_optimizer_step=21 * 39,
        validation_batch_size=400,
        reanchor_interval=50,
        validation_pool_examples=4_000,
    )
    assert [row["validation_gradient_evaluations"] for row in workload] == (
        [0] * 20 + [39] + [0] * 9
    )
    assert sum(
        row["direction_validation_example_visits"] for row in workload
    ) == 39 * 4_000


def test_salda_rejects_unregistered_seed_and_explicit_data_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="seed must be exactly one of 0 1 2"):
        _salda_args(
            monkeypatch,
            "--salda_ga_mode",
            "baseline",
            "--seed",
            "999",
        )
    with pytest.raises(ValueError, match="data_seed must be exactly -1"):
        _salda_args(
            monkeypatch,
            "--salda_ga_mode",
            "baseline",
            "--data_seed",
            "123",
        )


def test_data_protocol_closes_runtime_config_and_full_vdev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _salda_args(monkeypatch, "--salda_ga_mode", "soft_label")
    runtime_config = _salda_runtime_config_payload(args, method_name="mixup")
    fingerprint = "a" * 64
    data_protocol = _salda_data_protocol_payload(
        method_name="mixup",
        validation_fingerprint=fingerprint,
    )
    protocol = {
        **data_protocol,
        "runtime_config": runtime_config,
        "runtime_config_sha256": hashlib.sha256(
            json.dumps(
                runtime_config,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "data_protocol_sha256": hashlib.sha256(
            json.dumps(
                data_protocol,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }
    assert _salda_data_protocol_from_run_protocol(protocol) == data_protocol


def test_materialize_salda_validation_requires_full_balanced_5000() -> None:
    labels = np.repeat(np.arange(100, dtype=np.int32), 50)
    images = np.arange(5_000, dtype=np.float32).reshape(5_000, 1, 1, 1)
    batches = [
        (images[start : start + 128], labels[start : start + 128])
        for start in range(0, 5_000, 128)
    ]
    observed_images, observed_labels, fingerprint = _materialize_salda_validation(
        batches,
        num_classes=100,
    )
    np.testing.assert_array_equal(observed_images, images)
    np.testing.assert_array_equal(observed_labels, labels)
    assert len(fingerprint) == 64

    with pytest.raises(ValueError, match="50 examples per class"):
        _materialize_salda_validation(
            [(images, np.roll(labels, 1) + (np.arange(5_000) == 0))],
            num_classes=100,
        )


def test_stl10_runtime_protocol_uses_balanced_full_4000() -> None:
    """Materialize the exact STL-10 Vdev contract and reject imbalance."""

    labels = np.repeat(np.arange(10, dtype=np.int32), 400)
    images = np.arange(4_000, dtype=np.float32).reshape(4_000, 1, 1, 1)
    batches = [
        (images[start : start + 128], labels[start : start + 128])
        for start in range(0, 4_000, 128)
    ]
    observed_images, observed_labels, fingerprint = _materialize_salda_validation(
        batches,
        num_classes=10,
        expected_validation_examples=4_000,
        expected_examples_per_class=400,
    )
    np.testing.assert_array_equal(observed_images, images)
    np.testing.assert_array_equal(observed_labels, labels)
    assert len(fingerprint) == 64

    bad_labels = labels.copy()
    bad_labels[0] = 1
    with pytest.raises(ValueError, match="400 examples per class"):
        _materialize_salda_validation(
            [(images, bad_labels)],
            num_classes=10,
            expected_validation_examples=4_000,
            expected_examples_per_class=400,
        )


def test_stl10_runtime_config_and_data_seal_use_39_32_4000(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close every STL-10 primitive count through the shared protocol object."""

    args = _stl10_salda_args(monkeypatch, "--salda_ga_mode", "score_only")
    runtime = _salda_runtime_config_payload(
        args,
        method_name="mixup",
        dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    )
    assert runtime["validation"]["examples"] == 4_000
    assert runtime["validation"]["epoch_batches"] == 32
    assert (
        runtime["gradient_alignment"]["validation_examples_per_gradient_evaluation"]
        == 4_000
    )
    assert runtime["gradient_alignment"]["validation_examples_per_device"] == 1_000
    assert runtime["gradient_alignment"]["score_start_epoch"] == 0
    assert runtime["gradient_alignment"]["score_stop_epoch"] == -1
    assert runtime["gradient_alignment"]["action_start_epoch"] == 0
    assert runtime["gradient_alignment"]["action_stop_epoch"] == -1
    data_protocol = _salda_data_protocol_payload(
        method_name="mixup",
        validation_fingerprint="a" * 64,
        dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    )
    assert data_protocol["train_examples"] == 5_000
    assert data_protocol["steps_per_epoch"] == 39
    assert data_protocol["validation_examples"] == 4_000
    assert data_protocol["validation_class_counts"] == [400] * 10


def test_dataset_timing_targets_are_explicit_and_stl10_is_unregistered() -> None:
    """Do not apply the CIFAR-100 epoch target to STL-10 measurements."""

    assert CIFAR100_INSTANTANEOUS_GA_PROTOCOL.bounded_timing_epochs == (10,)
    assert STL10_INSTANTANEOUS_GA_PROTOCOL.bounded_timing_epochs == (10, 20, 30)
    assert CIFAR100_INSTANTANEOUS_GA_PROTOCOL.timing_median_seconds_at_most == 12.0
    assert CIFAR100_INSTANTANEOUS_GA_PROTOCOL.timing_p90_seconds_at_most == 12.5
    assert STL10_INSTANTANEOUS_GA_PROTOCOL.timing_median_seconds_at_most is None
    assert STL10_INSTANTANEOUS_GA_PROTOCOL.timing_p90_seconds_at_most is None
    stable_wall = {"median": 2.7, "p90": 2.9, "mean": 2.75, "count": 9}
    stl10_target = _salda_timing_target_payload(
        dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
        complete_timing_workload=True,
        stable_wall=stable_wall,
    )
    assert stl10_target == {
        "registered": False,
        "median_seconds_at_most": None,
        "p90_seconds_at_most": None,
        "observed_stable_median_seconds": 2.7,
        "observed_stable_p90_seconds": 2.9,
        "passed": None,
        "reason": "dataset_timing_target_not_registered",
    }
    cifar_target = _salda_timing_target_payload(
        dataset_protocol=CIFAR100_INSTANTANEOUS_GA_PROTOCOL,
        complete_timing_workload=True,
        stable_wall={"median": 11.9, "p90": 12.4, "mean": 12.0, "count": 9},
    )
    assert cifar_target["registered"] is True
    assert cifar_target["passed"] is True


def test_cifar100_registered_runtime_and_data_protocol_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the existing CIFAR-100 resolved values while adding STL-10."""

    protocol = CIFAR100_INSTANTANEOUS_GA_PROTOCOL
    assert {
        "dataset": protocol.dataset,
        "num_classes": protocol.num_classes,
        "train_examples": protocol.train_examples,
        "validation_examples": protocol.validation_examples,
        "validation_class_counts": list(protocol.validation_class_counts),
        "endpoint_examples": protocol.endpoint_examples,
        "global_batch_size": protocol.global_batch_size,
        "steps_per_epoch": protocol.steps_per_epoch,
        "validation_batches_per_epoch": protocol.validation_batches_per_epoch,
        "endpoint_batches": protocol.endpoint_batches,
        "validation_split": protocol.validation_split,
        "validation_batch_size": protocol.validation_batch_size,
        "validation_reanchor_interval": protocol.validation_reanchor_interval,
        "weight_decay": protocol.weight_decay,
        "mixup_alpha": protocol.mixup_alpha,
        "allow_batch_aggregate": protocol.allow_batch_aggregate,
        "bounded_timing_epochs": protocol.bounded_timing_epochs,
        "timing_median_seconds_at_most": (
            protocol.timing_median_seconds_at_most
        ),
        "timing_p90_seconds_at_most": protocol.timing_p90_seconds_at_most,
    } == {
        "dataset": "cifar100",
        "num_classes": 100,
        "train_examples": 50_000,
        "validation_examples": 5_000,
        "validation_class_counts": [50] * 100,
        "endpoint_examples": 5_000,
        "global_batch_size": 128,
        "steps_per_epoch": 390,
        "validation_batches_per_epoch": 40,
        "endpoint_batches": 40,
        "validation_split": 0.5,
        "validation_batch_size": 500,
        "validation_reanchor_interval": 50,
        "weight_decay": 0.0001,
        "mixup_alpha": 0.2,
        "allow_batch_aggregate": True,
        "bounded_timing_epochs": (10,),
        "timing_median_seconds_at_most": 12.0,
        "timing_p90_seconds_at_most": 12.5,
    }
    args = _salda_args(monkeypatch, "--salda_ga_mode", "score_only")
    runtime = _salda_runtime_config_payload(
        args,
        method_name="mixup",
        dataset_protocol=protocol,
    )
    data_protocol = _salda_data_protocol_payload(
        method_name="mixup",
        validation_fingerprint="f" * 64,
        dataset_protocol=protocol,
    )
    assert runtime["validation"] == {
        "source": "test",
        "split": 0.5,
        "examples": 5_000,
        "epoch_batches": 40,
        "direction_refresh_optimizer_steps": 1,
    }
    assert data_protocol["train_examples"] == 50_000
    assert data_protocol["steps_per_epoch"] == 390
    assert data_protocol["validation_examples"] == 5_000


def test_completion_schema_keeps_direction_binding_stl_only() -> None:
    """Do not silently alter CIFAR completion with the STL prerequisite field."""

    from allthemix.cli.train import SALDA_COMPLETION_CORE_FIELDS

    cifar_payload = {
        field: None for field in SALDA_COMPLETION_CORE_FIELDS
    }
    cifar_payload["dataset"] = "cifar100"
    _validate_salda_completion_schema(
        cifar_payload,
        dataset_protocol=CIFAR100_INSTANTANEOUS_GA_PROTOCOL,
    )
    with pytest.raises(RuntimeError, match="must not gain"):
        _validate_salda_completion_schema(
            {**cifar_payload, "direction_prerequisite": {}},
            dataset_protocol=CIFAR100_INSTANTANEOUS_GA_PROTOCOL,
        )
    stl_payload = {field: None for field in SALDA_COMPLETION_CORE_FIELDS}
    stl_payload.update({"dataset": "stl10", "direction_prerequisite": {}})
    _validate_salda_completion_schema(
        stl_payload,
        dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    )


def test_stl10_direction_prerequisite_binds_file_payload_and_vdev(
    tmp_path: Path,
) -> None:
    """Reject modified, cross-commit, or cross-Vdev direction evidence."""

    pool_sha256 = "a" * 64
    commit = "b" * 40
    payload = {
        "status": "SUCCESS",
        "dataset": "stl10",
        "git_commit": commit,
        "backend": "tpu",
        "device_count": 4,
        "validation_examples": 4_000,
        "validation_class_counts": [400] * 10,
        "validation_pool_sha256": pool_sha256,
        "local_validation_examples": 1_000,
        "validation_direction_mode": "full",
        "validation_direction_layout": "single_complete_batch",
        "parameter_scope": "classifier_head",
        "direction_leaf_shapes": [[4, 10], [4, 512, 10]],
        "finite": True,
        "distributed": True,
        "sync_batch_stats": True,
        "main_table_eligible": False,
        "wandb": {
            "enabled": True,
            "mode": "online",
            "run_id": "run-id",
            "url": "https://wandb.example/run-id",
            "finish_completed": True,
        },
    }
    payload["payload_sha256"] = _canonical_salda_json_sha256(payload)
    path = tmp_path / "direction.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    binding = _validate_stl10_direction_prerequisite(
        path,
        expected_artifact_file_sha256=file_sha256,
        expected_commit=commit,
        expected_validation_pool_sha256=pool_sha256,
    )
    assert binding["artifact_file_sha256"] == file_sha256
    assert binding["payload_sha256"] == payload["payload_sha256"]

    with pytest.raises(ValueError, match="file SHA mismatch"):
        _validate_stl10_direction_prerequisite(
            path,
            expected_artifact_file_sha256="c" * 64,
            expected_commit=commit,
            expected_validation_pool_sha256=pool_sha256,
        )
    with pytest.raises(ValueError, match="validation_pool_sha256"):
        _validate_stl10_direction_prerequisite(
            path,
            expected_artifact_file_sha256=file_sha256,
            expected_commit=commit,
            expected_validation_pool_sha256="d" * 64,
        )
    tampered = dict(payload)
    tampered["finite"] = False
    path.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
    tampered_file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="payload_sha256"):
        _validate_stl10_direction_prerequisite(
            path,
            expected_artifact_file_sha256=tampered_file_sha256,
            expected_commit=commit,
            expected_validation_pool_sha256=pool_sha256,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_batches", 38),
        ("vdev_batches", 31),
        ("vdev_examples", 3_999),
    ],
)
def test_pre_endpoint_failure_never_calls_builder(field: str, value: int) -> None:
    """A missing train/Vdev unit fails before the sealed builder is invoked."""

    calls = 0

    def builder(**_kwargs):
        nonlocal calls
        calls += 1
        return object()

    row = {
        "epoch": 1,
        "train_batches": 39,
        "vdev_batches": 32,
        "vdev_examples": 4_000,
    }
    row[field] = value
    with pytest.raises(RuntimeError, match="pre-endpoint workload"):
        closure = _validate_salda_pre_endpoint_workload(
            stop_epoch=1,
            epoch_records=[row],
            train_batches_per_epoch=39,
            initial_optimizer_step=0,
            terminal_optimizer_step=int(row["train_batches"]),
            endpoint_builder_calls=0,
            endpoint_evaluations=0,
            dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
        )
        _build_salda_endpoint_after_closure(
            builder=builder,
            pre_endpoint_closure=closure,
            best_checkpoint_closure={"passed": True},
            builder_kwargs={},
        )
    assert calls == 0


def test_pre_endpoint_closure_checks_real_terminal_optimizer_step() -> None:
    """Baseline, noop, and GA share the state-step update proof."""

    record = {
        "epoch": 1,
        "train_batches": 1,
        "vdev_batches": 32,
        "vdev_examples": 4_000,
    }
    closure = _validate_salda_pre_endpoint_workload(
        stop_epoch=1,
        epoch_records=[record],
        train_batches_per_epoch=1,
        initial_optimizer_step=7,
        terminal_optimizer_step=8,
        endpoint_builder_calls=0,
        endpoint_evaluations=0,
        dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    )
    assert closure["terminal_optimizer_step"] == 8
    with pytest.raises(RuntimeError, match="terminal optimizer step mismatch"):
        _validate_salda_pre_endpoint_workload(
            stop_epoch=1,
            epoch_records=[record],
            train_batches_per_epoch=1,
            initial_optimizer_step=7,
            terminal_optimizer_step=7,
            endpoint_builder_calls=0,
            endpoint_evaluations=0,
            dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
        )


def test_stl10_twenty_epoch_pre_endpoint_workload_is_registered() -> None:
    """Use the shared bounded-timing registry for the 20-epoch closure."""

    records = [
        {
            "epoch": epoch,
            "train_batches": 39,
            "vdev_batches": 32,
            "vdev_examples": 4_000,
        }
        for epoch in range(1, 21)
    ]
    closure = _validate_salda_pre_endpoint_workload(
        stop_epoch=20,
        epoch_records=records,
        train_batches_per_epoch=39,
        initial_optimizer_step=0,
        terminal_optimizer_step=780,
        endpoint_builder_calls=0,
        endpoint_evaluations=0,
        dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    )
    assert closure == {
        "passed": True,
        "registered": True,
        "completed_epochs": 20,
        "train_batches_per_epoch": 39,
        "train_updates": 780,
        "vdev_evaluations": 20,
        "vdev_batches": 640,
        "vdev_examples": 80_000,
        "initial_optimizer_step": 0,
        "terminal_optimizer_step": 780,
        "endpoint_builder_calls_before_closure": 0,
        "endpoint_evaluations_before_closure": 0,
    }


def test_main_orders_workload_checkpoint_and_endpoint_access() -> None:
    """Keep closure and checkpoint hashing before the single endpoint read."""

    from allthemix.cli import train

    source = inspect.getsource(train.main)
    workload_position = source.index(
        "salda_pre_endpoint_closure = _validate_salda_pre_endpoint_workload("
    )
    checkpoint_hash_position = source.index(
        "best_checkpoint_directory_sha256 = _salda_directory_sha256(",
        workload_position,
    )
    restore_position = source.index(
        "salda_restored_best = restore_checkpoint(",
        checkpoint_hash_position,
    )
    endpoint_position = source.index(
        "final_test_ds = _build_salda_endpoint_after_closure(",
        restore_position,
    )
    evaluation_position = source.index(
        "endpoint_evaluations += 1",
        endpoint_position,
    )
    assert (
        workload_position
        < checkpoint_hash_position
        < restore_position
        < endpoint_position
        < evaluation_position
    )


def test_best_checkpoint_closure_binds_path_hash_epoch_and_step() -> None:
    """Bind the restored best checkpoint identity before endpoint construction."""

    closure = _validate_salda_best_checkpoint_pre_endpoint(
        pre_endpoint_closure={
            "passed": True,
            "initial_optimizer_step": 0,
            "train_batches_per_epoch": 39,
        },
        best_epoch=7,
        best_checkpoint_optimizer_step=273,
        best_checkpoint_path="C:/checkpoints/best",
        best_checkpoint_directory_sha256="a" * 64,
    )
    assert closure == {
        "passed": True,
        "best_epoch": 7,
        "best_checkpoint_optimizer_step": 273,
        "expected_best_checkpoint_optimizer_step": 273,
        "best_checkpoint_path": "C:/checkpoints/best",
        "best_checkpoint_directory_sha256": "a" * 64,
    }
    with pytest.raises(RuntimeError, match="optimizer step mismatch"):
        _validate_salda_best_checkpoint_pre_endpoint(
            pre_endpoint_closure={
                "passed": True,
                "initial_optimizer_step": 0,
                "train_batches_per_epoch": 39,
            },
            best_epoch=7,
            best_checkpoint_optimizer_step=272,
            best_checkpoint_path="C:/checkpoints/best",
            best_checkpoint_directory_sha256="a" * 64,
        )


def test_checkpoint_hash_rejects_missing_or_empty_directory(tmp_path: Path) -> None:
    """Prevent endpoint construction without a material saved checkpoint."""

    with pytest.raises(ValueError, match="contains no files"):
        _salda_directory_sha256(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="contains no files"):
        _salda_directory_sha256(empty)


def test_stl10_rejects_batch_aggregate_before_data_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not reuse the separately authorized CIFAR-100 approximation arm."""

    with pytest.raises(ValueError, match="not registered for stl10"):
        _stl10_salda_args(
            monkeypatch,
            "--salda_ga_mode",
            "score_only",
            "--salda_ga_validation_direction_mode",
            "batch_aggregate",
        )


@pytest.mark.parametrize(
    "flag",
    ["--resume_checkpoint", "--pretrained_checkpoint"],
)
def test_stl10_requires_fresh_initialization(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
) -> None:
    """Keep the bound direction and optimizer-step origin on the same state."""

    with pytest.raises(ValueError, match="fresh initialization"):
        _stl10_salda_args(
            monkeypatch,
            "--salda_ga_mode",
            "score_only",
            flag,
            "outputs/checkpoint",
        )


def test_stl10_timing_and_formal_workloads_close_observed_counts() -> None:
    """Require the exact 39-update and 32-batch STL-10 workload."""

    timing = _validate_salda_completion_workload(
        stop_epoch=10,
        final_test_enabled=False,
        completed_epochs=10,
        steps_per_epoch=39,
        train_updates=390,
        vdev_evaluations=10,
        vdev_batches=320,
        endpoint_builder_calls=0,
        endpoint_evaluations=0,
        vtest_batches=0,
        vtest_examples=0,
        vtest_result=None,
        dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    )
    assert timing["passed"] is True
    timing20 = _validate_salda_completion_workload(
        stop_epoch=20,
        final_test_enabled=False,
        completed_epochs=20,
        steps_per_epoch=39,
        train_updates=780,
        vdev_evaluations=20,
        vdev_batches=640,
        endpoint_builder_calls=0,
        endpoint_evaluations=0,
        vtest_batches=0,
        vtest_examples=0,
        vtest_result=None,
        dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    )
    assert timing20 == {
        "workload": "bounded_epoch_timing",
        "required": True,
        "passed": True,
        "expected": {
            "completed_epochs": 20,
            "steps_per_epoch": 39,
            "train_updates": 780,
            "vdev_evaluations": 20,
            "vdev_batches": 640,
            "endpoint_builder_calls": 0,
            "endpoint_evaluations": 0,
            "vtest_batches": 0,
            "vtest_examples": 0,
            "has_vtest_result": False,
        },
        "observed": {
            "completed_epochs": 20,
            "steps_per_epoch": 39,
            "train_updates": 780,
            "vdev_evaluations": 20,
            "vdev_batches": 640,
            "endpoint_builder_calls": 0,
            "endpoint_evaluations": 0,
            "vtest_batches": 0,
            "vtest_examples": 0,
            "has_vtest_result": False,
        },
    }
    assert _salda_timing_workload_closure_payload(
        stop_epoch=20,
        workload_closure=timing20,
        observed_epoch_rows=20,
        observed_train_updates=780,
        observed_vdev_batches=640,
        observed_vtest_batches=0,
        dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    ) == {
        "registered_workload": "bounded_epoch_timing",
        "required": True,
        "passed": True,
        "is_complete_ten_epoch_timing_run": False,
        "expected_epoch_rows": 20,
        "observed_epoch_rows": 20,
        "expected_train_updates": 780,
        "observed_train_updates": 780,
        "expected_vdev_batches": 640,
        "observed_vdev_batches": 640,
        "observed_vtest_batches": 0,
    }
    origin_impulse30 = _validate_salda_completion_workload(
        stop_epoch=30,
        final_test_enabled=True,
        completed_epochs=30,
        steps_per_epoch=39,
        train_updates=1_170,
        vdev_evaluations=30,
        vdev_batches=960,
        endpoint_builder_calls=1,
        endpoint_evaluations=1,
        vtest_batches=32,
        vtest_examples=4_000,
        vtest_result=_finite_vtest_result(),
        dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    )
    assert origin_impulse30["workload"] == "bounded_epoch_diagnostic"
    assert origin_impulse30["required"] is True
    assert origin_impulse30["passed"] is True
    assert origin_impulse30["expected"] == origin_impulse30["observed"]
    assert origin_impulse30["expected"]["train_updates"] == 1_170
    assert origin_impulse30["expected"]["vdev_batches"] == 960
    assert origin_impulse30["expected"]["endpoint_evaluations"] == 1
    assert origin_impulse30["expected"]["vtest_batches"] == 32
    assert origin_impulse30["expected"]["vtest_examples"] == 4_000
    origin_timing30 = _validate_salda_completion_workload(
        stop_epoch=30,
        final_test_enabled=False,
        completed_epochs=30,
        steps_per_epoch=39,
        train_updates=1_170,
        vdev_evaluations=30,
        vdev_batches=960,
        endpoint_builder_calls=0,
        endpoint_evaluations=0,
        vtest_batches=0,
        vtest_examples=0,
        vtest_result=None,
        dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    )
    assert origin_timing30["workload"] == "bounded_epoch_timing"
    assert origin_timing30["expected"] == origin_timing30["observed"]
    with pytest.raises(RuntimeError, match="bounded_epoch_diagnostic workload closure"):
        _validate_salda_completion_workload(
            stop_epoch=30,
            final_test_enabled=True,
            completed_epochs=30,
            steps_per_epoch=1,
            train_updates=30,
            vdev_evaluations=30,
            vdev_batches=960,
            endpoint_builder_calls=1,
            endpoint_evaluations=1,
            vtest_batches=32,
            vtest_examples=4_000,
            vtest_result=_finite_vtest_result(),
            dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
        )
    unregistered = _validate_salda_completion_workload(
        stop_epoch=2,
        final_test_enabled=False,
        completed_epochs=2,
        steps_per_epoch=39,
        train_updates=78,
        vdev_evaluations=2,
        vdev_batches=64,
        endpoint_builder_calls=0,
        endpoint_evaluations=0,
        vtest_batches=0,
        vtest_examples=0,
        vtest_result=None,
        dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    )
    assert unregistered["workload"] == "unregistered_short_run"
    assert unregistered["required"] is False
    cifar20 = _validate_salda_completion_workload(
        stop_epoch=20,
        final_test_enabled=False,
        completed_epochs=20,
        steps_per_epoch=390,
        train_updates=7_800,
        vdev_evaluations=20,
        vdev_batches=800,
        endpoint_builder_calls=0,
        endpoint_evaluations=0,
        vtest_batches=0,
        vtest_examples=0,
        vtest_result=None,
        dataset_protocol=CIFAR100_INSTANTANEOUS_GA_PROTOCOL,
    )
    assert cifar20["workload"] == "unregistered_short_run"
    assert cifar20["required"] is False
    formal = _validate_salda_completion_workload(
        stop_epoch=-1,
        final_test_enabled=True,
        completed_epochs=200,
        steps_per_epoch=39,
        train_updates=7_800,
        vdev_evaluations=200,
        vdev_batches=6_400,
        endpoint_builder_calls=1,
        endpoint_evaluations=1,
        vtest_batches=32,
        vtest_examples=4_000,
        vtest_result=_finite_vtest_result(),
        dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    )
    assert formal["passed"] is True
    with pytest.raises(RuntimeError, match="complete_training workload closure"):
        _validate_salda_completion_workload(
            stop_epoch=-1,
            final_test_enabled=True,
            completed_epochs=200,
            steps_per_epoch=39,
            train_updates=7_800,
            vdev_evaluations=200,
            vdev_batches=6_399,
            endpoint_builder_calls=1,
            endpoint_evaluations=1,
            vtest_batches=32,
            vtest_examples=4_000,
            vtest_result=_finite_vtest_result(),
            dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("endpoint_builder_calls", 0),
        ("endpoint_builder_calls", 2),
        ("endpoint_evaluations", 0),
        ("endpoint_evaluations", 2),
        ("vtest_batches", 31),
        ("vtest_examples", 3_999),
    ],
)
def test_stl10_bounded_endpoint_rejects_incomplete_counts(
    field: str,
    value: int,
) -> None:
    """Require exactly one complete 4000-example endpoint evaluation."""

    values = {
        "endpoint_builder_calls": 1,
        "endpoint_evaluations": 1,
        "vtest_batches": 32,
        "vtest_examples": 4_000,
    }
    values[field] = value
    with pytest.raises(RuntimeError, match="bounded_epoch_diagnostic workload"):
        _validate_salda_completion_workload(
            stop_epoch=30,
            final_test_enabled=True,
            completed_epochs=30,
            steps_per_epoch=39,
            train_updates=1_170,
            vdev_evaluations=30,
            vdev_batches=960,
            endpoint_builder_calls=values["endpoint_builder_calls"],
            endpoint_evaluations=values["endpoint_evaluations"],
            vtest_batches=values["vtest_batches"],
            vtest_examples=values["vtest_examples"],
            vtest_result=_finite_vtest_result(),
            dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
        )


@pytest.mark.parametrize(
    "vtest_result",
    [
        None,
        {**_finite_vtest_result(), "loss": float("nan")},
        {**_finite_vtest_result(), "top1_error": float("inf")},
        {"top1_error": 0.5},
    ],
)
def test_stl10_bounded_endpoint_rejects_invalid_metrics(
    vtest_result: dict[str, float] | None,
) -> None:
    """Reject missing, non-finite, or structurally incomplete Vtest metrics."""

    with pytest.raises(RuntimeError, match="exactly five finite metrics"):
        _validate_salda_completion_workload(
            stop_epoch=30,
            final_test_enabled=True,
            completed_epochs=30,
            steps_per_epoch=39,
            train_updates=1_170,
            vdev_evaluations=30,
            vdev_batches=960,
            endpoint_builder_calls=1,
            endpoint_evaluations=1,
            vtest_batches=32,
            vtest_examples=4_000,
            vtest_result=vtest_result,
            dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
        )


def test_endpoint_disabled_unregistered_smoke_rejects_any_vtest_access() -> None:
    """Keep even unregistered smoke runs sealed away from Vtest."""

    with pytest.raises(RuntimeError, match="endpoint-disabled"):
        _validate_salda_completion_workload(
            stop_epoch=1,
            final_test_enabled=False,
            completed_epochs=1,
            steps_per_epoch=1,
            train_updates=1,
            vdev_evaluations=1,
            vdev_batches=32,
            endpoint_builder_calls=1,
            endpoint_evaluations=1,
            vtest_batches=32,
            vtest_examples=4_000,
            vtest_result=_finite_vtest_result(),
            dataset_protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
        )


def test_checkout_commit_validation_uses_exact_git_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def clean_checkout(command, *args, **kwargs):
        if command[1:3] == ["rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        if command[1:3] == ["status", "--porcelain"]:
            return ""
        raise AssertionError(command)

    monkeypatch.setattr(
        "allthemix.cli.train.subprocess.check_output",
        clean_checkout,
    )
    assert _validate_salda_checkout("a" * 40) == "a" * 40
    with pytest.raises(ValueError, match="does not match"):
        _validate_salda_checkout("b" * 40)


def test_timing_summary_uses_epoch_one_and_epochs_two_through_ten() -> None:
    from allthemix.cli.train import SALDA_GA_COMPONENT_NAMES

    rows = [
        {
            "epoch": epoch,
            "component_timing_seconds": {
                name: float(epoch) for name in SALDA_GA_COMPONENT_NAMES
            },
        }
        for epoch in range(1, 12)
    ]
    summary = _summarize_salda_epoch_timing(rows)
    assert summary["compile_epoch_1_seconds"] == 1.0
    assert summary["stable_epoch_range"] == list(range(2, 11))
    assert summary["components"]["end_to_end_wall"]["median"] == 6.0


def test_timing_summary_uses_configured_epochs_two_through_twenty() -> None:
    """The STL-10 timing config reports all 19 post-compile epochs."""

    from allthemix.cli.train import SALDA_GA_COMPONENT_NAMES

    rows = [
        {
            "epoch": epoch,
            "component_timing_seconds": {
                name: float(epoch) for name in SALDA_GA_COMPONENT_NAMES
            },
        }
        for epoch in range(1, 21)
    ]
    summary = _summarize_salda_epoch_timing(rows, stable_epoch_end=20)
    assert summary["compile_epoch_1_seconds"] == 1.0
    assert summary["stable_epoch_range"] == list(range(2, 21))
    assert summary["components"]["end_to_end_wall"] == {
        "mean": 11.0,
        "median": 11.0,
        "p90": pytest.approx(18.2),
        "count": 19,
    }


def test_noop_epoch_metrics_are_complete_host_scalars_and_route_without_blanks(
    tmp_path: Path,
) -> None:
    raw_metrics: dict[str, object] = {}
    _inject_salda_noop_epoch_metrics(raw_metrics, policy_mode="noop")
    assert list(raw_metrics) == SALDA_GA_METRIC_NAMES
    assert raw_metrics == SALDA_GA_NOOP_METRICS
    assert all(type(value) is float for value in raw_metrics.values())
    namespaced = _namespace_extra_metrics(raw_metrics)
    assert _wandb_extra_metrics(namespaced) == {
        f"salda/{key.removeprefix('salda_')}": value
        for key, value in raw_metrics.items()
    }

    output = tmp_path / "noop.csv"
    write_csv_header(output, extra_metric_names=SALDA_GA_METRIC_NAMES)
    append_epoch_result(
        output_path=output,
        epoch=1,
        train_loss=1.0,
        train_accuracy=0.5,
        eval_loss=1.1,
        eval_top1_accuracy=0.4,
        eval_top5_accuracy=0.8,
        eval_top1_error=0.6,
        eval_top5_error=0.2,
        best_top1_error=0.6,
        best_epoch=1,
        epoch_time=2.0,
        extra_metrics=namespaced,
        extra_metric_names=SALDA_GA_METRIC_NAMES,
    )
    with output.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert all(row[name] != "" for name in SALDA_GA_METRIC_NAMES)


def test_noop_metric_injection_rejects_collisions() -> None:
    for name in SALDA_GA_METRIC_NAMES:
        with pytest.raises(RuntimeError, match="unexpected device metrics"):
            _inject_salda_noop_epoch_metrics({name: 0.0}, policy_mode="noop")


def test_inactive_score_phase_injects_metrics_only_on_the_host() -> None:
    raw_metrics: dict[str, object] = {}
    _inject_salda_noop_epoch_metrics(
        raw_metrics,
        policy_mode="soft_label",
        score_phase_active=False,
    )
    assert raw_metrics == SALDA_GA_NOOP_METRICS

    active_metrics: dict[str, object] = {}
    _inject_salda_noop_epoch_metrics(
        active_metrics,
        policy_mode="soft_label",
        score_phase_active=True,
    )
    assert active_metrics == {}


def test_action_summary_closes_mean_metrics_to_integer_counts() -> None:
    row = {
        "extra_metrics": {
            "salda_eligible_fraction": 0.5,
            "salda_applied_fraction": 0.25,
            "salda_batch_action_coverage": 0.75,
            "salda_fallback_fraction": 0.25,
            "salda_gain_abstention_fraction": 0.25,
            "salda_margin_abstention_fraction": 0.125,
            "salda_budget_excluded_fraction": 0.125,
            "salda_invalid_fraction": 0.0,
            "salda_dose_mean": 0.0025,
            "salda_weight_relative_ess": 0.95,
        }
    }
    summary = _summarize_salda_actions(
        [row],
        steps_per_epoch=4,
        global_batch_size=8,
        strategy_active=True,
    )
    assert summary["scored_batches"] == 4
    assert summary["scored_rows"] == 32
    assert summary["eligible_rows"] == 16
    assert summary["applied_rows"] == 8
    assert summary["batches_with_actions"] == 3


def test_origin_impulse_summary_counts_only_the_4992_scored_rows() -> None:
    records = []
    for epoch in range(1, 31):
        metrics = dict(SALDA_GA_NOOP_METRICS)
        if epoch == 21:
            metrics.update(
                {
                    "salda_scored_fraction": 1.0,
                    "salda_eligible_fraction": 0.75,
                    "salda_applied_fraction": 0.75,
                    "salda_batch_action_coverage": 1.0,
                    "salda_dose_mean": 0.075,
                    "salda_weight_relative_ess": 1.0,
                }
            )
        records.append({"extra_metrics": metrics})

    summary = _summarize_salda_actions(
        records,
        steps_per_epoch=39,
        global_batch_size=128,
        strategy_active=True,
    )
    assert summary["scored_batches"] == 39
    assert summary["scored_rows"] == 4_992
    assert summary["eligible_rows"] == 3_744
    assert summary["applied_rows"] == 3_744
    assert summary["batches_with_actions"] == 39
    assert summary["row_coverage"] == pytest.approx(0.75)


def test_timing_and_formal_workloads_fail_closed() -> None:
    timing = _validate_salda_completion_workload(
        stop_epoch=10,
        final_test_enabled=False,
        completed_epochs=10,
        steps_per_epoch=390,
        train_updates=3_900,
        vdev_evaluations=10,
        vdev_batches=400,
        endpoint_builder_calls=0,
        endpoint_evaluations=0,
        vtest_batches=0,
        vtest_examples=0,
        vtest_result=None,
    )
    assert timing == {
        "workload": "ten_epoch_timing",
        "required": True,
        "passed": True,
        "expected": {
            "completed_epochs": 10,
            "steps_per_epoch": 390,
            "train_updates": 3_900,
            "vdev_evaluations": 10,
            "vdev_batches": 400,
            "endpoint_builder_calls": 0,
            "endpoint_evaluations": 0,
            "vtest_batches": 0,
            "vtest_examples": 0,
            "has_vtest_result": False,
        },
        "observed": {
            "completed_epochs": 10,
            "steps_per_epoch": 390,
            "train_updates": 3_900,
            "vdev_evaluations": 10,
            "vdev_batches": 400,
            "endpoint_builder_calls": 0,
            "endpoint_evaluations": 0,
            "vtest_batches": 0,
            "vtest_examples": 0,
            "has_vtest_result": False,
        },
    }
    timing_closure = _salda_timing_workload_closure_payload(
        stop_epoch=10,
        workload_closure=timing,
        observed_epoch_rows=10,
        observed_train_updates=3_900,
        observed_vdev_batches=400,
        observed_vtest_batches=0,
        dataset_protocol=CIFAR100_INSTANTANEOUS_GA_PROTOCOL,
    )
    assert timing_closure == {
        "registered_workload": "ten_epoch_timing",
        "required": True,
        "passed": True,
        "is_complete_ten_epoch_timing_run": True,
        "expected_epoch_rows": 10,
        "observed_epoch_rows": 10,
        "expected_train_updates": 3_900,
        "observed_train_updates": 3_900,
        "expected_vdev_batches": 400,
        "observed_vdev_batches": 400,
        "observed_vtest_batches": 0,
    }
    formal = _validate_salda_completion_workload(
        stop_epoch=-1,
        final_test_enabled=True,
        completed_epochs=200,
        steps_per_epoch=390,
        train_updates=78_000,
        vdev_evaluations=200,
        vdev_batches=8_000,
        endpoint_builder_calls=1,
        endpoint_evaluations=1,
        vtest_batches=40,
        vtest_examples=5_000,
        vtest_result=_finite_vtest_result(),
    )
    assert formal["workload"] == "complete_training"
    with pytest.raises(RuntimeError, match="ten_epoch_timing workload closure"):
        _validate_salda_completion_workload(
            stop_epoch=10,
            final_test_enabled=False,
            completed_epochs=10,
            steps_per_epoch=390,
            train_updates=3_899,
            vdev_evaluations=10,
            vdev_batches=400,
            endpoint_builder_calls=0,
            endpoint_evaluations=0,
            vtest_batches=0,
            vtest_examples=0,
            vtest_result=None,
        )


def test_checkpoint_selection_uses_strict_vdev_error() -> None:
    best_error = float("inf")
    selected_epoch = 0
    for epoch, error in ((1, 0.40), (2, 0.35), (3, 0.35)):
        if should_replace_best_vdev_top1_error(error, best_error):
            best_error = error
            selected_epoch = epoch
    assert selected_epoch == 2
    assert STRICT_VDEV_TOP1_ERROR_RULE == (
        "strictly_lower_full_Vdev_top1_error_first_epoch_wins_ties"
    )
