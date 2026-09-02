"""Regression tests for the registered STL-10 instantaneous-GA entry point."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from allthemix.config import load_yaml_config
from salutary_da.protocol import (
    STL10_INSTANTANEOUS_GA_PROTOCOL,
    build_data_protocol,
    build_runtime_config,
    build_run_protocol_data_fields,
    build_training_recipe,
    canonical_protocol_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY_ROOT / "configs/stl10/preact_resnet18/salda_ga.yaml"
SMOKE_CONFIG = (
    REPOSITORY_ROOT / "configs/stl10/preact_resnet18/salda_ga_smoke.yaml"
)
TIMING20_CONFIG = (
    REPOSITORY_ROOT / "configs/stl10/preact_resnet18/salda_ga_timing20.yaml"
)
TIMING10_CONFIG = (
    REPOSITORY_ROOT / "configs/stl10/preact_resnet18/salda_ga_timing10.yaml"
)
LAUNCHER = REPOSITORY_ROOT / "scripts/experiment_run/run_stl10_salda_ga.sh"
DIRECTION_PROBE = (
    REPOSITORY_ROOT / "scripts/experiment_run/probe_stl10_salda_direction.py"
)
DIRECTION_LAUNCHER = (
    REPOSITORY_ROOT
    / "scripts/experiment_run/run_stl10_salda_direction_smoke.sh"
)


def _launcher_python_blocks() -> list[str]:
    """Return every embedded Python consumer from the config-only launcher."""

    source = LAUNCHER.read_text(encoding="utf-8")
    return [
        block.split("\nPY\n", maxsplit=1)[0]
        for block in source.split("<<'PY'\n")[1:]
    ]


def _canonical_sha256(value: dict[str, object]) -> str:
    """Hash one JSON mapping with the trainer artifact convention."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_expanded_config(tmp_path: Path, commit: str) -> Path:
    """Execute the launcher's exact config expander into a temporary run."""

    destination = tmp_path / "resolved_run_config.yaml"
    process = subprocess.run(
        [
            sys.executable,
            "-",
            str(TIMING20_CONFIG),
            str(destination),
            "/exact/data",
            str(tmp_path),
            "exact-run-name",
            commit,
        ],
        input=_launcher_python_blocks()[0],
        text=True,
        capture_output=True,
        check=False,
        cwd=REPOSITORY_ROOT,
    )
    assert process.returncode == 0, process.stderr
    return destination


def _write_direction_artifact(
    tmp_path: Path,
    *,
    commit: str,
    validation_pool_sha256: str,
) -> tuple[Path, dict[str, object], str]:
    """Write an exact self-hashed direction prerequisite."""

    payload: dict[str, object] = {
        "status": "SUCCESS",
        "dataset": "stl10",
        "git_commit": commit,
        "backend": "tpu",
        "device_count": 4,
        "validation_examples": 4_000,
        "validation_class_counts": [400] * 10,
        "validation_pool_sha256": validation_pool_sha256,
        "local_validation_examples": 1_000,
        "validation_direction_mode": "full",
        "validation_direction_layout": "single_complete_batch",
        "parameter_scope": "classifier_head",
        "direction_leaf_shapes": [[4, 10], [4, 512, 10]],
        "finite": True,
        "distributed": True,
        "sync_batch_stats": True,
        "main_table_eligible": False,
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    path = tmp_path / "direction.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path, payload, hashlib.sha256(path.read_bytes()).hexdigest()


def _write_smoke_artifacts(
    tmp_path: Path,
    *,
    commit: str,
    direction: dict[str, object],
    direction_path: Path,
    direction_file_sha256: str,
) -> tuple[Path, str]:
    """Write a linked one-update protocol and completion fixture."""

    timing_config = load_yaml_config(TIMING20_CONFIG)
    runtime_config = build_runtime_config(
        timing_config,
        method_name="mixup",
        protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    )
    training_recipe = build_training_recipe(runtime_config)
    data_protocol = build_data_protocol(
        method_name="mixup",
        validation_fingerprint=str(direction["validation_pool_sha256"]),
        protocol=STL10_INSTANTANEOUS_GA_PROTOCOL,
    )
    runtime_sha256 = canonical_protocol_sha256(runtime_config)
    training_sha256 = canonical_protocol_sha256(training_recipe)
    data_sha256 = canonical_protocol_sha256(data_protocol)
    direction_binding = {
        "artifact": str(direction_path),
        "artifact_file_sha256": direction_file_sha256,
        "payload_sha256": direction["payload_sha256"],
        "validation_pool_sha256": direction["validation_pool_sha256"],
        "git_commit": commit,
        "validation_direction_mode": "full",
        "validation_direction_layout": "single_complete_batch",
        "parameter_scope": "classifier_head",
        "distributed": True,
        "sync_batch_stats": True,
    }
    protocol = {
        **build_run_protocol_data_fields(data_protocol),
        "direction_prerequisite": direction_binding,
        "runtime_config": runtime_config,
        "runtime_config_sha256": runtime_sha256,
        "training_recipe": training_recipe,
        "training_recipe_sha256": training_sha256,
        "data_protocol_sha256": data_sha256,
    }
    assert "val_source" not in protocol
    assert "validation_split" not in protocol
    protocol_path = tmp_path / "smoke_protocol.json"
    protocol_path.write_text(json.dumps(protocol) + "\n", encoding="utf-8")
    smoke: dict[str, object] = {
        "status": "SUCCESS",
        "dataset": "stl10",
        "git_commit": commit,
        "completed_epochs": 1,
        "optimizer_horizon_epochs": 200,
        "policy_mode": "score_only",
        "seed": 0,
        "resolved_data_seed": 0,
        "method": "mixup",
        "parameter_scope": "classifier_head",
        "validation_direction_mode": "full",
        "validation_examples_per_gradient_evaluation": 4_000,
        "validation_pool_sha256": direction["validation_pool_sha256"],
        "train_updates": 1,
        "vdev_evaluations": 1,
        "vdev_batches": 32,
        "vdev_example_visits_for_epoch_readout": 4_000,
        "vtest_loaded": False,
        "vtest_batches": 0,
        "vtest_examples": 0,
        "endpoint_builder_calls": 0,
        "endpoint_evaluations": 0,
        "initial_optimizer_step": 0,
        "terminal_optimizer_step": 1,
        "direction_prerequisite": direction_binding,
        "protocol_artifact": str(protocol_path),
        "protocol_artifact_sha256": hashlib.sha256(
            protocol_path.read_bytes()
        ).hexdigest(),
        "runtime_config_sha256": runtime_sha256,
        "training_recipe_sha256": training_sha256,
        "data_protocol_sha256": data_sha256,
        "execution": {
            "action_enabled": True,
            "train_steps": 1,
            "direction_refreshes": 1,
            "validation_gradient_evaluations": 1,
            "validation_exact_reanchors": 0,
            "direction_validation_example_visits": 4_000,
            "validation_pool_examples": 4_000,
        },
        "action_summary": {
            "scored_batches": 1,
            "scored_rows": 128,
            "applied_rows": 0,
            "batches_with_actions": 0,
            "fallback_batches": 0,
            "invalid_score_rows": 0,
        },
        "best_vdev_checkpoint": None,
        "wandb": {
            "enabled": True,
            "mode": "online",
            "run_id": "fixture-run",
            "url": "https://wandb.example/fixture-run",
            "finish_completed": True,
        },
    }
    smoke["completion_sha256"] = _canonical_sha256(smoke)
    smoke_path = tmp_path / "smoke_completion.json"
    smoke_path.write_text(json.dumps(smoke) + "\n", encoding="utf-8")
    return smoke_path, hashlib.sha256(smoke_path.read_bytes()).hexdigest()


def _write_smoke_postflight_fixture(
    tmp_path: Path,
    commit: str,
) -> tuple[Path, Path]:
    """Write a real-shape one-update completion and its expanded smoke config."""

    config = load_yaml_config(SMOKE_CONFIG)
    config.update(
        {
            "output_dir": str(tmp_path),
            "output_name": "metrics.csv",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "salda_ga_git_commit": commit,
        }
    )
    config_path = tmp_path / "smoke_resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )
    (tmp_path / "source_config.yaml").write_text("smoke\n", encoding="utf-8")
    (tmp_path / "runtime_environment.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "metrics.csv").write_text(
        "epoch,salda_train_updates,salda_vdev_batches,salda_vtest_batches\n"
        "1,1,32,0\n",
        encoding="utf-8",
    )
    protocol_path = tmp_path / "protocol.json"
    protocol = {
        "runtime_config_sha256": "a" * 64,
        "training_recipe_sha256": "b" * 64,
        "data_protocol_sha256": "c" * 64,
    }
    protocol_path.write_text(json.dumps(protocol) + "\n", encoding="utf-8")
    wall = 7.0
    completion: dict[str, object] = {
        "status": "SUCCESS",
        "dataset": "stl10",
        "git_commit": commit,
        "completed_epochs": 1,
        "optimizer_horizon_epochs": 200,
        "policy_mode": "score_only",
        "seed": 0,
        "resolved_data_seed": 0,
        "method": "mixup",
        "parameter_scope": "classifier_head",
        "validation_direction_mode": "full",
        "validation_examples_per_gradient_evaluation": 4_000,
        "train_updates": 1,
        "vdev_evaluations": 1,
        "vdev_batches": 32,
        "vdev_example_visits_for_epoch_readout": 4_000,
        "vtest_loaded": False,
        "vtest_batches": 0,
        "vtest_examples": 0,
        "vtest_result": None,
        "endpoint_builder_calls": 0,
        "endpoint_evaluations": 0,
        "initial_optimizer_step": 0,
        "terminal_optimizer_step": 1,
        "best_vdev_checkpoint": None,
        "protocol_artifact": str(protocol_path),
        "protocol_artifact_sha256": hashlib.sha256(
            protocol_path.read_bytes()
        ).hexdigest(),
        "resolved_config_sha256": "e" * 64,
        **protocol,
        "validation_pool_sha256": "d" * 64,
        "workload_closure": {
            "workload": "unregistered_short_run",
            "required": False,
            "passed": True,
        },
        "pre_endpoint_workload_closure": {
            "passed": True,
            "registered": False,
            "completed_epochs": 1,
            "train_batches_per_epoch": 1,
            "train_updates": 1,
            "vdev_evaluations": 1,
            "vdev_batches": 32,
            "vdev_examples": 4_000,
            "initial_optimizer_step": 0,
            "terminal_optimizer_step": 1,
            "endpoint_builder_calls_before_closure": 0,
            "endpoint_evaluations_before_closure": 0,
        },
        "execution": {
            "action_enabled": True,
            "parameter_scope": "classifier_head",
            "validation_direction_mode": "full",
            "validation_pool_examples": 4_000,
            "validation_examples_per_gradient_evaluation": 4_000,
            "train_steps": 1,
            "direction_refreshes": 1,
            "validation_gradient_evaluations": 1,
            "validation_exact_reanchors": 0,
            "validation_anchor_drift_comparisons": 0,
            "direction_validation_example_visits": 4_000,
        },
        "action_summary": {
            "scored_batches": 1,
            "scored_rows": 128,
            "applied_rows": 0,
            "batches_with_actions": 0,
            "fallback_batches": 0,
            "invalid_score_rows": 0,
            "mean_dose_over_all_scored_rows": 0.0,
        },
        "epoch_timing_summary": {
            "compile_epoch_1_seconds": wall,
            "stable_epoch_range": [],
            "components": {
                "end_to_end_wall": {
                    "mean": None,
                    "median": None,
                    "p90": None,
                    "count": 0,
                }
            },
        },
        "epochs": [
            {
                "epoch": 1,
                "component_timing_seconds": {"end_to_end_wall": wall},
            }
        ],
        "timing_target": {
            "registered": False,
            "passed": None,
            "reason": "dataset_timing_target_not_registered",
        },
        "wandb": {
            "enabled": True,
            "mode": "online",
            "run_id": "fixture-run",
            "url": "https://wandb.example/fixture-run",
            "finish_completed": True,
        },
    }
    completion["completion_sha256"] = _canonical_sha256(completion)
    completion_path = tmp_path / "completion.json"
    completion_path.write_text(json.dumps(completion) + "\n", encoding="utf-8")
    return completion_path, config_path


def test_stl10_salda_config_uses_the_stl10_mixup_contract() -> None:
    """Lock the dataset-specific optimizer, MixUp, split, and topology values."""

    config = load_yaml_config(CONFIG)
    assert config["dataset"] == "stl10"
    assert config["method"] == "mixup"
    assert config["batch_size"] == 128
    assert config["epochs"] == 200
    assert config["validation_split"] == 0.5
    assert config["val_source"] == "test"
    assert config["weight_decay"] == 0.0005
    assert config["mixup_alpha"] == 1.0
    assert config["distributed"] is True
    assert config["sync_batch_stats"] is True
    assert config["wandb"] is True
    assert config["salda_ga_validation_direction_mode"] == "full"
    assert config["salda_ga_validation_batch_size"] == 400


def test_stl10_salda_smoke_and_timing_are_fully_configured() -> None:
    """Lock all scientific and workload differences in tracked YAML files."""

    smoke = load_yaml_config(SMOKE_CONFIG)
    timing10 = load_yaml_config(TIMING10_CONFIG)
    timing = load_yaml_config(TIMING20_CONFIG)
    shared = {
        "dataset": "stl10",
        "model": "preact_resnet18",
        "method": "mixup",
        "batch_size": 128,
        "epochs": 200,
        "seed": 0,
        "data_seed": -1,
        "resnet_stem_type": "cifar",
        "preact_stem_bn_relu": False,
        "preact_pytorch_default_init": False,
        "validation_split": 0.5,
        "val_source": "test",
        "eval_on_test_each_epoch": False,
        "val_select_split_fraction": 0.0,
        "max_eval_steps": -1,
        "final_test": False,
        "save_checkpoint": False,
        "save_best_only": False,
        "distributed": True,
        "sync_batch_stats": True,
        "cross_device_shuffle": False,
        "basic_aug": True,
        "aug_recipe": "basic",
        "shuffle_buffer_size": 10_000,
        "deterministic_data": True,
        "strict_determinism": False,
        "train_subset_fraction": 1.0,
        "debug_train_source": "none",
        "early_stop_enabled": False,
        "save_csv": True,
        "log_time": True,
        "wandb": True,
        "wandb_project": "allthemix-probes",
        "wandb_mode": "online",
        "learning_rate": 0.1,
        "momentum": 0.9,
        "nesterov": False,
        "weight_decay": 0.0005,
        "lr_schedule": "cosine",
        "min_learning_rate": 0.0,
        "warmup_epochs": 0,
        "lr_decay_epochs": [100, 150],
        "lr_decay_rate": 0.1,
        "mixup_alpha": 1.0,
        "salda_ga_mode": "score_only",
        "salda_ga_parameter_scope": "classifier_head",
        "salda_ga_validation_direction_mode": "full",
        "salda_ga_validation_batch_size": 400,
        "salda_ga_validation_reanchor_interval": 50,
        "salda_ga_maximum_rows": 128,
        "salda_ga_soft_label_dose": 0.01,
        "salda_ga_max_weight_deviation": 0.05,
        "salda_ga_weight_temperature": 1.0,
        "salda_ga_minimum_relative_ess": 0.9,
        "salda_ga_minimum_gain": 0.0,
        "salda_ga_minimum_label_margin": 0.0,
        "salda_ga_minimum_relative_label_margin": 0.0,
        "salda_ga_fallback_enabled": False,
        "salda_ga_fallback_soft_label_dose": 0.01,
    }
    for key, expected in shared.items():
        assert smoke[key] == expected
        assert timing10[key] == expected
        assert timing[key] == expected
    assert {
        key: timing[key]
        for key in (
            "salda_ga_stop_epoch",
            "max_train_steps",
            "salda_ga_audit_mode",
            "salda_ga_profile_components",
        )
    } == {
        "salda_ga_stop_epoch": 20,
        "max_train_steps": -1,
        "salda_ga_audit_mode": False,
        "salda_ga_profile_components": False,
    }
    assert {
        key: timing10[key]
        for key in (
            "salda_ga_stop_epoch",
            "max_train_steps",
            "salda_ga_audit_mode",
            "salda_ga_profile_components",
        )
    } == {
        "salda_ga_stop_epoch": 10,
        "max_train_steps": -1,
        "salda_ga_audit_mode": False,
        "salda_ga_profile_components": False,
    }
    assert {
        key: smoke[key]
        for key in (
            "salda_ga_stop_epoch",
            "max_train_steps",
            "salda_ga_audit_mode",
            "salda_ga_profile_components",
        )
    } == {
        "salda_ga_stop_epoch": 1,
        "max_train_steps": 1,
        "salda_ga_audit_mode": True,
        "salda_ga_profile_components": True,
    }


def test_stl10_salda_launcher_is_config_only_and_exact_full_vdev() -> None:
    """Keep one config-only entry point on the exact complete-Vdev direction."""

    source = LAUNCHER.read_text(encoding="utf-8")
    assert "python -u -m allthemix.cli.train" in source
    command_line = next(
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("python -u -m allthemix.cli.train")
    )
    assert command_line == (
        'python -u -m allthemix.cli.train --config "$RESOLVED_CONFIG" \\'
    )
    assert "--salda_ga_mode" not in source
    assert "--salda_ga_stop_epoch" not in source
    assert "LEGACY_STAGE" in source
    assert "LEGACY_ARM" in source
    assert "salda_ga_timing10.yaml" in source
    assert "salda_ga_timing20.yaml" in source
    assert "ALLTHEMIX_SALDA_CONFIG_ROUTE_ONLY" in source
    assert "load_yaml_config(Path(source))" in source
    assert 'yaml.safe_dump(config, sort_keys=True)' in source
    assert '"salda_ga_validation_direction_mode": "full"' in source
    assert '"salda_ga_validation_batch_size": 400' in source
    assert "batch_aggregate" not in source
    assert "validation_chunk" not in source
    assert "trainer_exit_code.txt" in source
    assert "tee_exit_code.txt" in source
    assert "wrapper_exit_code.txt" in source
    assert "set +e" in source
    assert "DIRECTION_SMOKE_ARTIFACT" in source
    assert "DIRECTION_SMOKE_ARTIFACT_SHA256" in source
    assert "ALLTHEMIX_SALDA_DIRECTION_ARTIFACT" in source
    assert "ALLTHEMIX_SALDA_DIRECTION_ARTIFACT_SHA256" in source
    assert '"validation_pool_sha256"' in source
    assert '"validation_direction_mode": "full"' in source
    assert '"validation_direction_layout": "single_complete_batch"' in source
    assert '"parameter_scope": "classifier_head"' in source
    assert '"direction_leaf_shapes": [[4, 10], [4, 512, 10]]' in source
    assert '"distributed": True' in source
    assert '"sync_batch_stats": True' in source
    assert "payload_sha256" in source
    assert "file_sha(direction_path) != direction_sha" in source
    assert '"vdev_batches": vdev_batches' in source
    assert '"vdev_example_visits_for_epoch_readout": readout_visits' in source
    assert '"endpoint_builder_calls": 0' in source
    assert '"endpoint_evaluations": 0' in source
    assert '"completed_epochs": epochs' in source
    assert '"dataset": "stl10"' in source
    assert 'wandb.get("mode") == "online"' in source
    assert '"wandb_mode": "online"' in source
    assert '"ten_epoch_timing" if epochs == 10 else "bounded_epoch_timing"' in source
    assert 'list(range(2, epochs + 1))' in source
    assert 'np.quantile(stable, 0.9)' in source
    assert '"direction_example_visits": direction_visits' in source
    assert '"scored_rows": scored_rows' in source
    assert '"wrapper_wall_seconds"' in source
    assert "runtime_environment.json" in source
    assert len(_launcher_python_blocks()) == 5
    assert "source, destination, data_dir" in _launcher_python_blocks()[0]
    assert "direction_path, direction_sha, smoke_path" in _launcher_python_blocks()[2]
    assert "jax_compilation_cache_dir" in _launcher_python_blocks()[3]
    assert "stable = wall[1:]" in _launcher_python_blocks()[4]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
@pytest.mark.parametrize(
    ("arguments", "expected_config"),
    [
        (("smoke", "last_score"), SMOKE_CONFIG),
        (("timing", "last_score"), TIMING10_CONFIG),
        ((), TIMING20_CONFIG),
    ],
)
def test_launcher_routes_legacy_and_default_invocations_to_tracked_configs(
    arguments: tuple[str, ...],
    expected_config: Path,
) -> None:
    """Execute the compatibility router without opening a TPU workload."""

    environment = dict(os.environ)
    environment.pop("SOURCE_CONFIG", None)
    environment["ALLTHEMIX_SALDA_CONFIG_ROUTE_ONLY"] = "true"
    process = subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        text=True,
        capture_output=True,
        check=False,
        cwd=REPOSITORY_ROOT,
        env=environment,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == expected_config.relative_to(
        REPOSITORY_ROOT
    ).as_posix()


def test_config_expander_writes_a_flat_config_with_only_operational_injections(
    tmp_path: Path,
) -> None:
    """Execute the tracked-to-run-local config transformation exactly."""

    commit = "a" * 40
    destination = _write_expanded_config(tmp_path, commit)
    raw = destination.read_text(encoding="utf-8")
    resolved = load_yaml_config(destination)
    assert "base:" not in raw
    assert resolved == {
        **load_yaml_config(TIMING20_CONFIG),
        "data_dir": "/exact/data",
        "output_dir": str(tmp_path),
        "output_name": "metrics.csv",
        "run_name": "exact-run-name",
        "wandb_run_name": "exact-run-name",
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "salda_ga_git_commit": commit,
    }


def test_timing_prerequisite_gate_accepts_exact_chain_and_rejects_smoke_sha(
    tmp_path: Path,
) -> None:
    """Execute the exact direction-plus-smoke binding used before 20e."""

    commit = "a" * 40
    pool_sha256 = "b" * 64
    config_path = _write_expanded_config(tmp_path, commit)
    direction_path, direction, direction_sha256 = _write_direction_artifact(
        tmp_path,
        commit=commit,
        validation_pool_sha256=pool_sha256,
    )
    smoke_path, smoke_sha256 = _write_smoke_artifacts(
        tmp_path,
        commit=commit,
        direction=direction,
        direction_path=direction_path,
        direction_file_sha256=direction_sha256,
    )
    command = [
        sys.executable,
        "-",
        str(direction_path),
        direction_sha256,
        str(smoke_path),
        smoke_sha256,
        str(config_path),
        commit,
    ]
    accepted = subprocess.run(
        command,
        input=_launcher_python_blocks()[2],
        text=True,
        capture_output=True,
        check=False,
        cwd=REPOSITORY_ROOT,
    )
    assert accepted.returncode == 0, accepted.stderr
    rejected = subprocess.run(
        [*command[:5], "e" * 64, *command[6:]],
        input=_launcher_python_blocks()[2],
        text=True,
        capture_output=True,
        check=False,
        cwd=REPOSITORY_ROOT,
    )
    assert rejected.returncode != 0
    assert "smoke prerequisite file SHA mismatch" in rejected.stderr


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    [
        ("learning_rate", 0.2, "protocol.runtime_config_sha256"),
        ("mixup_alpha", 0.3, "protocol.training_recipe_sha256"),
        ("preact_stem_bn_relu", True, "protocol.training_recipe"),
        ("method", "baseline", "protocol.data_protocol.method"),
    ],
)
def test_timing_prerequisite_gate_rejects_current_config_semantic_drift(
    tmp_path: Path,
    field: str,
    replacement: object,
    expected_error: str,
) -> None:
    """Fail closed when current optimizer, augmentation, training, or data drift."""

    commit = "a" * 40
    config_path = _write_expanded_config(tmp_path, commit)
    direction_path, direction, direction_sha256 = _write_direction_artifact(
        tmp_path,
        commit=commit,
        validation_pool_sha256="b" * 64,
    )
    smoke_path, smoke_sha256 = _write_smoke_artifacts(
        tmp_path,
        commit=commit,
        direction=direction,
        direction_path=direction_path,
        direction_file_sha256=direction_sha256,
    )
    config = load_yaml_config(config_path)
    config[field] = replacement
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )
    process = subprocess.run(
        [
            sys.executable,
            "-",
            str(direction_path),
            direction_sha256,
            str(smoke_path),
            smoke_sha256,
            str(config_path),
            commit,
        ],
        input=_launcher_python_blocks()[2],
        text=True,
        capture_output=True,
        check=False,
        cwd=REPOSITORY_ROOT,
    )
    assert process.returncode != 0
    assert expected_error in process.stderr


def test_smoke_postflight_accepts_exactly_one_configured_update(
    tmp_path: Path,
) -> None:
    """Execute postflight with the smoke config's one-update bound."""

    completion_path, config_path = _write_smoke_postflight_fixture(
        tmp_path,
        "a" * 40,
    )
    output = tmp_path / "postflight.json"
    process = subprocess.run(
        [
            sys.executable,
            "-",
            str(completion_path),
            str(config_path),
            "e" * 64,
            "",
            "",
            "0.0",
            str(output),
        ],
        input=_launcher_python_blocks()[4],
        text=True,
        capture_output=True,
        check=False,
        cwd=REPOSITORY_ROOT,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["completed_epochs"] == 1
    assert result["train_updates"] == 1
    assert result["direction_evaluations"] == 1
    assert result["direction_example_visits"] == 4_000
    assert result["scored_rows"] == 128


def test_stl10_direction_probe_is_exact_sealed_and_online() -> None:
    """Require a direction-only TPU smoke before the one-step train smoke."""

    source = DIRECTION_PROBE.read_text(encoding="utf-8")
    assert "validation_split=0.5" in source
    assert "(4_000, 96, 96, 3)" in source
    assert "np.full((10,), 400)" in source
    assert "prepare_validation_batch" in source
    assert "num_devices=4" in source
    assert "project=args.wandb_project" in source
    assert "mode=args.wandb_mode" in source
    assert '"main_table_eligible": False' in source
    assert 'payload["wandb"]["mode"] == "online"' in source
    assert 'payload["payload_sha256"]' in source
    assert '"validation_direction_mode": "full"' in source
    assert '"validation_direction_layout": "single_complete_batch"' in source
    assert '"parameter_scope": "classifier_head"' in source

    launcher = DIRECTION_LAUNCHER.read_text(encoding="utf-8")
    assert "flock -n" in launcher
    assert "python -u -m scripts.experiment_run.probe_stl10_salda_direction" in launcher
    assert "set +e" in launcher
    assert "pipeline_exit.txt" in launcher
    assert "probe_exit_code" in launcher
    assert "tee_exit_code" in launcher
    assert '"validation_examples": 4_000' in launcher
    assert '"validation_class_counts": [400] * 10' in launcher
    assert '"main_table_eligible": False' in launcher
    assert "payload_sha256" in launcher
    assert 'wandb.get("mode") == "online"' in launcher
    assert '"${WANDB_MODE:-online}" != "online"' in launcher
