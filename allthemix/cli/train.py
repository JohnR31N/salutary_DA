from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from absl import logging as absl_logging

absl_logging.set_verbosity(absl_logging.ERROR)

from allthemix.utils.backend_environment import validate_backend_environment

validate_backend_environment(
    "jax",
    expected_prefix=os.environ.get("VIRTUAL_ENV"),
)

import jax
import numpy as np

from allthemix.cli.args import parse_args
from allthemix.cli.formal_run_protocol import (
    prepare_formal_run,
    validate_pre_endpoint_workload,
    write_json_atomic,
)
from allthemix.cli.train_setup import (
    ALIA_METHOD_NAMES,
    DIFFUSEMIX_METHOD_NAMES,
    build_final_test_dataset,
    build_training_datasets,
    plan_offline_manifests,
    prepare_run_outputs,
    restore_initial_state,
)
from allthemix.competitors.metaaugment import (
    META_AUGMENT_METRIC_NAMES,
    create_metaaugment_context,
)
from allthemix.data.preprocessors.selector import get_metadata
from allthemix.data.splits import (
    resolve_training_validation_split,
)
from allthemix.data.utils.cardinality import (
    resolve_train_example_count,
)
from allthemix.methods.selector import get_mixer
from allthemix.methods.utils.validation import normalize_method_name
from allthemix.networks.builder import build_model, get_feature_hook_count
from allthemix.training.engine.parallel.parallel_loop import (
    parallel_evaluate,
    parallel_train_one_epoch,
)
from allthemix.training.engine.single.loop import evaluate, train_one_epoch
from allthemix.training.engine.single.train import create_train_state
from allthemix.training.utils.early_stop import (
    EarlyStopConfig,
    EarlyStopState,
    update_early_stop,
)
from allthemix.training.utils.lr_scheduler import build_lr_schedule
from allthemix.utils.checkpoint import (
    restore_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
)
from allthemix.utils.experiment_logger import (
    append_epoch_result,
    format_epoch_message,
    format_final_test_message,
    write_final_test_result,
)
from allthemix.utils.parallel import (
    create_device_rngs,
    replicate_state,
    unreplicate_state,
)
from allthemix.utils.reproducibility import resolve_data_seed, seed_everything
from allthemix.utils.timer import Timer
from salutary_da.checkpoint_selection import (
    STRICT_VDEV_TOP1_ERROR_RULE,
    should_replace_best_vdev_top1_error,
)
from salutary_da.protocol import (
    CIFAR100_INSTANTANEOUS_GA_PROTOCOL,
    InstantaneousGADatasetProtocol,
    build_data_protocol,
    build_run_protocol_data_fields,
    build_runtime_config,
    build_training_recipe,
    build_validation_direction_config,
    get_instantaneous_ga_dataset_protocol,
    resolve_registered_timing_epochs,
)

SUMIX_DEBUG_METRIC_NAMES = [
    "sumix_lam_a_mean",
    "sumix_lam_a_min",
    "sumix_lam_a_max",
    "sumix_lam_b_mean",
    "sumix_area_lam_mean",
    "sumix_classification_loss",
    "sumix_regularization_loss",
    "sumix_uncertainty_original_mean",
    "sumix_uncertainty_mixed_mean",
    "sumix_semantic_scale",
    "sumix_ina_feature_mean",
    "sumix_ina_feature_max",
    "sumix_inb_feature_mean",
    "sumix_inb_feature_max",
]

CATCHUPMIX_METHOD_NAMES = (
    "catchupmix",
    "catchup_mix",
    "catch_up_mix",
)


MIX_DEBUG_METRIC_NAMES = [
    "mix_lam_mean",
    "mix_lam_min",
    "mix_lam_max",
    "mix_lam_std",
    "mix_changed_ratio",
    "mix_applied_lam_mean",
    "mix_applied_changed_ratio",
    "mix_apply_rate",
    "mix_same_label_rate",
    "mix_applied_same_label_rate",
    "mix_identity_pair_rate",
    "mix_applied_identity_pair_rate",
    "mix_image_mean",
    "mix_image_std",
]

SALDA_GA_METRIC_NAMES = [
    "salda_scored_fraction",
    "salda_scores_valid",
    "salda_eligible_fraction",
    "salda_applied_fraction",
    "salda_batch_action_coverage",
    "salda_fallback_fraction",
    "salda_gain_abstention_fraction",
    "salda_margin_abstention_fraction",
    "salda_budget_excluded_fraction",
    "salda_invalid_fraction",
    "salda_score_mean",
    "salda_score_min",
    "salda_score_max",
    "salda_label_margin_mean",
    "salda_dose_mean",
    "salda_weight_mean",
    "salda_weight_score_mean",
    "salda_weight_score_min",
    "salda_weight_score_max",
    "salda_weight_min",
    "salda_weight_max",
    "salda_weight_relative_ess",
]

SALDA_GA_NOOP_METRICS = {
    name: float(
        name
        in {
            "salda_scores_valid",
            "salda_weight_mean",
            "salda_weight_min",
            "salda_weight_max",
            "salda_weight_relative_ess",
        }
    )
    for name in SALDA_GA_METRIC_NAMES
}

SALDA_GA_COMPONENT_NAMES = (
    "data",
    "augmentation_mix",
    "vdev_direction",
    "jvp",
    "policy",
    "update",
    "metric_sync",
    "vdev_eval",
    "checkpoint_copy",
    "hash_audit",
    "wandb",
    "end_to_end_wall",
)

SALDA_GA_PROTOCOL_CSV_NAMES = (
    "salda_git_commit",
    "salda_config_sha256",
    "salda_runtime_config_sha256",
    "salda_data_protocol_sha256",
    "salda_vdev_role",
    "salda_vtest_role",
    "salda_train_updates",
    "salda_vdev_batches",
    "salda_vtest_batches",
)

SALDA_COMPLETION_CORE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "dataset",
        "git_commit",
        "resolved_config_sha256",
        "runtime_config_sha256",
        "training_recipe_sha256",
        "data_protocol_sha256",
        "protocol_artifact",
        "protocol_artifact_sha256",
        "completed_epochs",
        "optimizer_horizon_epochs",
        "policy_mode",
        "seed",
        "resolved_data_seed",
        "data_seed_policy",
        "method",
        "training_data",
        "parameter_scope",
        "validation_direction_mode",
        "validation_examples_per_gradient_evaluation",
        "validation_direction_cycle_length",
        "validation_reanchor_interval",
        "validation_direction_main_table_eligible",
        "validation_pool_sha256",
        "validation_batch_schedule_sha256",
        "train_updates",
        "vdev_evaluations",
        "vdev_batches",
        "vdev_example_visits_for_epoch_readout",
        "vtest_loaded",
        "vtest_batches",
        "vtest_result",
        "vdev_role",
        "vtest_role",
        "endpoint_builder_calls",
        "endpoint_evaluations",
        "vtest_examples",
        "best_vdev_epoch",
        "best_vdev_top1_error",
        "checkpoint_selection_rule",
        "best_vdev_checkpoint",
        "best_checkpoint_optimizer_step",
        "initial_optimizer_step",
        "terminal_optimizer_step",
        "pre_endpoint_workload_closure",
        "best_checkpoint_pre_endpoint_closure",
        "endpoint_built_after_best_checkpoint_restore",
        "wandb",
        "origin_mixup_contract",
        "execution",
        "action_summary",
        "component_timing",
        "epoch_timing_summary",
        "timing_profile_synchronizes_component_boundaries",
        "component_timing_measurement_mode",
        "update_timing_includes_standard_base_method",
        "workload_closure",
        "timing_workload_closure",
        "timing_target",
        "epochs",
    }
)


def _validate_salda_checkout(declared_commit: str) -> str:
    """Require the active checkout to match the declared clean commit."""

    repository = Path(__file__).resolve().parents[2]
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()
    if observed != declared_commit:
        raise ValueError(
            "SalDA declared commit does not match the checkout: "
            f"declared={declared_commit}, observed={observed}"
        )
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        text=True,
    ).strip()
    if changed:
        raise RuntimeError("SalDA requires a clean checkout:\n" + changed)
    return observed


def _canonical_salda_json_sha256(value: dict[str, object]) -> str:
    """Hash one JSON object with the canonical SalDA serialization."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_salda_completion_schema(
    payload: dict[str, object],
    *,
    dataset_protocol: InstantaneousGADatasetProtocol,
) -> None:
    """Preserve the shared completion schema and STL-only evidence boundary."""

    missing = sorted(SALDA_COMPLETION_CORE_FIELDS.difference(payload))
    if missing:
        raise RuntimeError("SalDA completion fields are missing: " + ", ".join(missing))
    if payload.get("dataset") != dataset_protocol.dataset:
        raise RuntimeError("SalDA completion dataset does not match its protocol")
    direction_prerequisite = payload.get("direction_prerequisite")
    if dataset_protocol.dataset == "stl10":
        if not isinstance(direction_prerequisite, dict):
            raise RuntimeError("STL-10 completion has no direction prerequisite")
    elif "direction_prerequisite" in payload:
        raise RuntimeError(
            "non-STL SalDA completion must not gain an STL direction prerequisite"
        )


def _validate_stl10_direction_prerequisite(
    artifact_path: str | Path,
    *,
    expected_artifact_file_sha256: str,
    expected_commit: str,
    expected_validation_pool_sha256: str,
) -> dict[str, object]:
    """Bind an STL-10 trainer to one immutable exact-full direction smoke."""

    path = Path(artifact_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"STL-10 direction artifact does not exist: {path}")
    observed_artifact_file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_artifact_file_sha256 != expected_artifact_file_sha256:
        raise ValueError(
            "STL-10 direction artifact file SHA mismatch: "
            f"observed={observed_artifact_file_sha256}, "
            f"expected={expected_artifact_file_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("STL-10 direction artifact must be a JSON object")
    declared_payload_sha256 = payload.get("payload_sha256")
    unhashed_payload = dict(payload)
    unhashed_payload.pop("payload_sha256", None)
    observed_payload_sha256 = _canonical_salda_json_sha256(unhashed_payload)
    required = {
        "status": "SUCCESS",
        "dataset": "stl10",
        "git_commit": expected_commit,
        "backend": "tpu",
        "device_count": 4,
        "validation_examples": 4_000,
        "validation_class_counts": [400] * 10,
        "validation_pool_sha256": expected_validation_pool_sha256,
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
    mismatches = [
        key for key, expected in required.items() if payload.get(key) != expected
    ]
    if declared_payload_sha256 != observed_payload_sha256:
        mismatches.append("payload_sha256")
    wandb = payload.get("wandb")
    if not isinstance(wandb, dict) or not (
        wandb.get("enabled")
        and wandb.get("mode") == "online"
        and wandb.get("run_id")
        and wandb.get("url")
        and wandb.get("finish_completed")
    ):
        mismatches.append("wandb")
    if mismatches:
        raise ValueError(
            "STL-10 direction prerequisite mismatch: " + ", ".join(mismatches)
        )
    return {
        "artifact": str(path),
        "artifact_file_sha256": observed_artifact_file_sha256,
        "payload_sha256": observed_payload_sha256,
        "validation_pool_sha256": expected_validation_pool_sha256,
        "git_commit": expected_commit,
        "validation_direction_mode": "full",
        "validation_direction_layout": "single_complete_batch",
        "parameter_scope": "classifier_head",
        "distributed": True,
        "sync_batch_stats": True,
    }


def _materialize_salda_validation(
    eval_ds,
    *,
    num_classes: int,
    expected_validation_examples: int = 5_000,
    expected_examples_per_class: int = 50,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Materialize and fingerprint one exact registered Vdev stream."""

    image_batches = []
    label_batches = []
    for batch in eval_ds:
        if isinstance(batch, dict):
            images = batch["images"]
            labels = batch["labels"]
        else:
            images, labels = batch[:2]
        image_batches.append(np.asarray(images, dtype=np.float32))
        label_batches.append(np.asarray(labels, dtype=np.int32))
    if not image_batches:
        raise ValueError("SalDA Vdev pipeline produced no examples")
    images = np.concatenate(image_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    if images.shape[0] != expected_validation_examples or labels.shape != (
        expected_validation_examples,
    ):
        raise ValueError(
            "SalDA requires exactly "
            f"{expected_validation_examples} Vdev examples; "
            f"received images={images.shape[0]}, labels={labels.shape}"
        )
    class_counts = np.bincount(labels, minlength=num_classes)
    if class_counts.shape != (num_classes,) or not np.array_equal(
        class_counts,
        np.full(
            (num_classes,),
            expected_examples_per_class,
            dtype=class_counts.dtype,
        ),
    ):
        raise ValueError(
            "SalDA Vdev must contain exactly "
            f"{expected_examples_per_class} examples per class"
        )
    digest = hashlib.sha256()
    for value in (images, labels):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return images, labels, digest.hexdigest()


def _write_salda_json(output_dir: str, run_name: str, suffix: str, value) -> Path:
    """Write one deterministic SalDA runtime artifact."""

    path = Path(output_dir).expanduser().resolve() / f"{run_name}_{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _salda_directory_sha256(path: Path) -> str:
    """Hash one checkpoint directory including relative file names."""

    resolved = path.expanduser().resolve()
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"SalDA checkpoint directory contains no files: {resolved}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(resolved).as_posix().encode("utf-8"))
        with file.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def _summarize_salda_epoch_timing(
    records: list[dict],
    *,
    stable_epoch_end: int = 10,
) -> dict[str, object]:
    """Summarize compile epoch and the configured stable timing window."""

    if stable_epoch_end < 2:
        raise ValueError("stable_epoch_end must be at least 2")

    if not records:
        return {
            "epoch_count": 0,
            "compile_epoch_1_seconds": None,
            "stable_epoch_range": [],
            "components": {},
        }
    stable = records[1:stable_epoch_end]
    components = {}
    for name in SALDA_GA_COMPONENT_NAMES:
        values = np.asarray(
            [row["component_timing_seconds"][name] for row in stable],
            dtype=np.float64,
        )
        components[name] = (
            {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.9)),
                "count": int(values.size),
            }
            if values.size
            else {"mean": None, "median": None, "p90": None, "count": 0}
        )
    return {
        "epoch_count": len(records),
        "compile_epoch_1_seconds": records[0]["component_timing_seconds"][
            "end_to_end_wall"
        ],
        "stable_epoch_range": [row["epoch"] for row in stable],
        "components": components,
    }


def _salda_timing_target_payload(
    *,
    dataset_protocol: InstantaneousGADatasetProtocol,
    complete_timing_workload: bool,
    stable_wall: dict[str, float | int | None],
) -> dict[str, object]:
    """Evaluate only a timing target registered for the active dataset."""

    median_target = dataset_protocol.timing_median_seconds_at_most
    p90_target = dataset_protocol.timing_p90_seconds_at_most
    registered = bool(median_target is not None and p90_target is not None)
    return {
        "registered": registered,
        "median_seconds_at_most": median_target,
        "p90_seconds_at_most": p90_target,
        "observed_stable_median_seconds": stable_wall["median"],
        "observed_stable_p90_seconds": stable_wall["p90"],
        "passed": (
            bool(
                complete_timing_workload
                and stable_wall["median"] is not None
                and stable_wall["p90"] is not None
                and stable_wall["median"] <= median_target
                and stable_wall["p90"] <= p90_target
            )
            if registered
            else None
        ),
        "reason": None if registered else "dataset_timing_target_not_registered",
    }


def _salda_timing_workload_closure_payload(
    *,
    stop_epoch: int,
    workload_closure: dict[str, object],
    observed_epoch_rows: int,
    observed_train_updates: int,
    observed_vdev_batches: int,
    observed_vtest_batches: int,
    dataset_protocol: InstantaneousGADatasetProtocol,
) -> dict[str, object]:
    """Build the legacy-compatible closure for a registered timing run."""

    registered_timing_epochs = resolve_registered_timing_epochs(
        stop_epoch=stop_epoch,
        protocol=dataset_protocol,
    )
    expected_timing_epochs = registered_timing_epochs or 10
    return {
        "registered_workload": workload_closure["workload"],
        "required": workload_closure["required"],
        "passed": workload_closure["passed"],
        "is_complete_ten_epoch_timing_run": bool(
            workload_closure["workload"] == "ten_epoch_timing"
            and workload_closure["passed"]
        ),
        "expected_epoch_rows": expected_timing_epochs,
        "observed_epoch_rows": observed_epoch_rows,
        "expected_train_updates": (
            expected_timing_epochs * dataset_protocol.steps_per_epoch
        ),
        "observed_train_updates": observed_train_updates,
        "expected_vdev_batches": (
            expected_timing_epochs
            * dataset_protocol.validation_batches_per_epoch
        ),
        "observed_vdev_batches": observed_vdev_batches,
        "observed_vtest_batches": observed_vtest_batches,
    }


def _validate_salda_completion_workload(
    *,
    stop_epoch: int,
    final_test_enabled: bool,
    completed_epochs: int,
    steps_per_epoch: int,
    train_updates: int,
    vdev_evaluations: int,
    vdev_batches: int,
    endpoint_builder_calls: int,
    vtest_batches: int,
    vtest_result: dict[str, float] | None,
    endpoint_evaluations: int = 0,
    vtest_examples: int = 0,
    dataset_protocol: InstantaneousGADatasetProtocol = (
        CIFAR100_INSTANTANEOUS_GA_PROTOCOL
    ),
) -> dict[str, object]:
    """Fail closed on registered timing, endpoint, and formal workloads."""

    workload = None
    expected = None
    timing_epochs = resolve_registered_timing_epochs(
        stop_epoch=stop_epoch,
        protocol=dataset_protocol,
    )
    if final_test_enabled:
        if stop_epoch == -1:
            workload = "complete_training"
            expected_epochs = 200
        elif dataset_protocol.dataset == "stl10" and timing_epochs == 30:
            workload = "bounded_epoch_diagnostic"
            expected_epochs = timing_epochs
        else:
            raise ValueError(
                "SalDA final Vtest requires complete training or the registered "
                "STL-10 30-epoch endpoint workload"
            )
        expected_result_keys = {
            "loss",
            "top1_accuracy",
            "top5_accuracy",
            "top1_error",
            "top5_error",
        }
        if (
            not isinstance(vtest_result, dict)
            or set(vtest_result) != expected_result_keys
            or not all(
                not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value)
                for value in vtest_result.values()
            )
        ):
            raise RuntimeError(
                "SalDA final Vtest result must contain exactly five finite metrics"
            )
        expected = {
            "completed_epochs": expected_epochs,
            "steps_per_epoch": dataset_protocol.steps_per_epoch,
            "train_updates": expected_epochs * dataset_protocol.steps_per_epoch,
            "vdev_evaluations": expected_epochs,
            "vdev_batches": (
                expected_epochs * dataset_protocol.validation_batches_per_epoch
            ),
            "endpoint_builder_calls": 1,
            "endpoint_evaluations": 1,
            "vtest_batches": dataset_protocol.endpoint_batches,
            "vtest_examples": dataset_protocol.endpoint_examples,
            "has_vtest_result": True,
        }
    elif stop_epoch == -1:
        raise ValueError("complete SalDA training requires final_test_enabled=True")
    elif timing_epochs is not None:
        workload = (
            "ten_epoch_timing"
            if timing_epochs == 10
            else "bounded_epoch_timing"
        )
        expected = {
            "completed_epochs": timing_epochs,
            "steps_per_epoch": dataset_protocol.steps_per_epoch,
            "train_updates": timing_epochs * dataset_protocol.steps_per_epoch,
            "vdev_evaluations": timing_epochs,
            "vdev_batches": (
                timing_epochs * dataset_protocol.validation_batches_per_epoch
            ),
            "endpoint_builder_calls": 0,
            "endpoint_evaluations": 0,
            "vtest_batches": 0,
            "vtest_examples": 0,
            "has_vtest_result": False,
        }

    if not final_test_enabled and (
        endpoint_builder_calls != 0
        or endpoint_evaluations != 0
        or vtest_batches != 0
        or vtest_examples != 0
        or vtest_result is not None
    ):
        raise RuntimeError("SalDA endpoint-disabled workload accessed Vtest")

    observed = {
        "completed_epochs": completed_epochs,
        "steps_per_epoch": steps_per_epoch,
        "train_updates": train_updates,
        "vdev_evaluations": vdev_evaluations,
        "vdev_batches": vdev_batches,
        "endpoint_builder_calls": endpoint_builder_calls,
        "endpoint_evaluations": endpoint_evaluations,
        "vtest_batches": vtest_batches,
        "vtest_examples": vtest_examples,
        "has_vtest_result": vtest_result is not None,
    }
    if expected is not None:
        mismatches = [
            name
            for name, expected_value in expected.items()
            if observed[name] != expected_value
        ]
        if mismatches:
            raise RuntimeError(
                f"SalDA {workload} workload closure failed: "
                + ", ".join(
                    f"{name}={observed[name]!r} expected {expected[name]!r}"
                    for name in mismatches
                )
            )
    return {
        "workload": workload or "unregistered_short_run",
        "required": expected is not None,
        "passed": expected is None or observed == expected,
        "expected": expected,
        "observed": observed,
    }


# #### SALDA PRE-ENDPOINT WORKLOAD CLOSURE: START ####
def _validate_salda_pre_endpoint_workload(
    *,
    stop_epoch: int,
    epoch_records: list[dict[str, object]],
    train_batches_per_epoch: int,
    initial_optimizer_step: int,
    terminal_optimizer_step: int,
    endpoint_builder_calls: int,
    endpoint_evaluations: int,
    dataset_protocol: InstantaneousGADatasetProtocol,
) -> dict[str, object]:
    """Close every training and Vdev count before the sealed endpoint exists."""

    if not epoch_records:
        raise RuntimeError("SalDA pre-endpoint workload has no completed epoch")
    if train_batches_per_epoch <= 0:
        raise ValueError("SalDA train batches per epoch must be positive")
    if endpoint_builder_calls != 0 or endpoint_evaluations != 0:
        raise RuntimeError(
            "SalDA sealed endpoint was accessed before workload closure: "
            f"builder_calls={endpoint_builder_calls}, "
            f"evaluations={endpoint_evaluations}"
        )
    expected_epochs = list(range(1, len(epoch_records) + 1))
    observed_epochs = [int(row["epoch"]) for row in epoch_records]
    if observed_epochs != expected_epochs:
        raise RuntimeError(
            "SalDA pre-endpoint epoch sequence changed: "
            f"observed={observed_epochs}, expected={expected_epochs}"
        )
    for row in epoch_records:
        epoch = int(row["epoch"])
        train_batches = int(row["train_batches"])
        vdev_batches = int(row["vdev_batches"])
        vdev_examples = int(row["vdev_examples"])
        mismatches = []
        if train_batches != train_batches_per_epoch:
            mismatches.append(
                f"train_batches={train_batches} expected {train_batches_per_epoch}"
            )
        if vdev_batches != dataset_protocol.validation_batches_per_epoch:
            mismatches.append(
                "vdev_batches="
                f"{vdev_batches} expected "
                f"{dataset_protocol.validation_batches_per_epoch}"
            )
        if vdev_examples != dataset_protocol.validation_examples:
            mismatches.append(
                "vdev_examples="
                f"{vdev_examples} expected {dataset_protocol.validation_examples}"
            )
        if mismatches:
            raise RuntimeError(
                f"SalDA pre-endpoint workload changed at epoch {epoch}: "
                + ", ".join(mismatches)
            )
    train_updates = sum(int(row["train_batches"]) for row in epoch_records)
    expected_terminal_optimizer_step = initial_optimizer_step + train_updates
    if terminal_optimizer_step != expected_terminal_optimizer_step:
        raise RuntimeError(
            "SalDA terminal optimizer step mismatch: "
            f"observed={terminal_optimizer_step}, "
            f"expected={expected_terminal_optimizer_step}"
        )
    registered_epochs = (
        200
        if stop_epoch == -1
        else resolve_registered_timing_epochs(
            stop_epoch=stop_epoch,
            protocol=dataset_protocol,
        )
    )
    if registered_epochs is not None:
        expected_updates = registered_epochs * dataset_protocol.steps_per_epoch
        if (
            len(epoch_records) != registered_epochs
            or train_batches_per_epoch != dataset_protocol.steps_per_epoch
            or train_updates != expected_updates
        ):
            raise RuntimeError(
                "SalDA registered pre-endpoint workload closure failed: "
                f"epochs={len(epoch_records)} expected {registered_epochs}, "
                f"train_batches_per_epoch={train_batches_per_epoch} expected "
                f"{dataset_protocol.steps_per_epoch}, updates={train_updates} "
                f"expected {expected_updates}"
            )
    return {
        "passed": True,
        "registered": registered_epochs is not None,
        "completed_epochs": len(epoch_records),
        "train_batches_per_epoch": train_batches_per_epoch,
        "train_updates": train_updates,
        "vdev_evaluations": len(epoch_records),
        "vdev_batches": sum(int(row["vdev_batches"]) for row in epoch_records),
        "vdev_examples": sum(int(row["vdev_examples"]) for row in epoch_records),
        "initial_optimizer_step": initial_optimizer_step,
        "terminal_optimizer_step": terminal_optimizer_step,
        "endpoint_builder_calls_before_closure": endpoint_builder_calls,
        "endpoint_evaluations_before_closure": endpoint_evaluations,
    }
# #### SALDA PRE-ENDPOINT WORKLOAD CLOSURE: END ####


# #### SALDA BEST-VDEV CHECKPOINT GATE: START ####
def _validate_salda_best_checkpoint_pre_endpoint(
    *,
    pre_endpoint_closure: dict[str, object],
    best_epoch: int,
    best_checkpoint_optimizer_step: int,
    best_checkpoint_path: str,
    best_checkpoint_directory_sha256: str,
) -> dict[str, int | str | bool]:
    """Verify the selected checkpoint step before opening the sealed endpoint."""

    if pre_endpoint_closure.get("passed") is not True:
        raise RuntimeError("SalDA pre-endpoint workload closure did not pass")
    if best_epoch <= 0:
        raise RuntimeError("SalDA best checkpoint epoch is missing")
    expected = int(pre_endpoint_closure["initial_optimizer_step"]) + best_epoch * int(
        pre_endpoint_closure["train_batches_per_epoch"]
    )
    if best_checkpoint_optimizer_step != expected:
        raise RuntimeError(
            "SalDA best checkpoint optimizer step mismatch: "
            f"observed={best_checkpoint_optimizer_step}, expected={expected}"
        )
    if not best_checkpoint_path:
        raise RuntimeError("SalDA best checkpoint path is missing")
    if len(best_checkpoint_directory_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in best_checkpoint_directory_sha256
    ):
        raise RuntimeError("SalDA best checkpoint directory SHA-256 is invalid")
    return {
        "passed": True,
        "best_epoch": best_epoch,
        "best_checkpoint_optimizer_step": best_checkpoint_optimizer_step,
        "expected_best_checkpoint_optimizer_step": expected,
        "best_checkpoint_path": best_checkpoint_path,
        "best_checkpoint_directory_sha256": best_checkpoint_directory_sha256,
    }
# #### SALDA BEST-VDEV CHECKPOINT GATE: END ####


# #### SALDA SEALED VTEST DATASET GATE: START ####
def _build_salda_endpoint_after_closure(
    *,
    builder,
    pre_endpoint_closure: dict[str, object],
    best_checkpoint_closure: dict[str, object],
    builder_kwargs: dict[str, object],
):
    """Invoke the sealed endpoint builder only after both pre-access gates pass."""

    if pre_endpoint_closure.get("passed") is not True:
        raise RuntimeError("SalDA endpoint requires passed workload closure")
    if best_checkpoint_closure.get("passed") is not True:
        raise RuntimeError("SalDA endpoint requires passed best-checkpoint closure")
    return builder(**builder_kwargs)
# #### SALDA SEALED VTEST DATASET GATE: END ####


def _salda_origin_mixup_contract(method_name: str) -> dict[str, object]:
    """Record the exact base-recipe contract shared by parity runners."""

    if method_name not in {"baseline", "mixup"}:
        raise ValueError("SalDA recipe contract requires baseline or mixup")
    return {
        "origin": ("unmixed_one_hot_target" if method_name == "baseline" else None),
        "mixup_pairing": (
            "device_local_permutation_scalar_lambda_per_device"
            if method_name == "mixup"
            else None
        ),
        "dropout_rng": "one_key_per_device_per_update",
        "sync_batch_stats": True,
    }


def _summarize_salda_actions(
    records: list[dict],
    *,
    steps_per_epoch: int,
    global_batch_size: int,
    strategy_active: bool,
) -> dict[str, object]:
    """Convert epoch-mean device metrics into closed integer action counts."""

    total_batches = len(records) * steps_per_epoch
    total_rows = total_batches * global_batch_size
    if not strategy_active or total_rows == 0:
        return {
            "scored_batches": 0,
            "scored_rows": 0,
            "eligible_rows": 0,
            "applied_rows": 0,
            "batches_with_actions": 0,
            "fallback_batches": 0,
            "gain_threshold_abstention_rows": 0,
            "margin_threshold_abstention_rows": 0,
            "budget_excluded_rows": 0,
            "invalid_score_rows": 0,
            "row_coverage": 0.0,
            "batch_coverage": 0.0,
            "mean_dose_over_all_scored_rows": 0.0,
            "mean_relative_ess": 1.0,
        }

    def weighted_total(metric: str, scale: int) -> float:
        return float(
            sum(
                float(
                    row["extra_metrics"].get(
                        metric,
                        1.0 if metric == "salda_scored_fraction" else 0.0,
                    )
                )
                * scale
                for row in records
            )
        )

    scored_batches = round(
        weighted_total("salda_scored_fraction", steps_per_epoch)
    )
    scored_rows = scored_batches * global_batch_size
    eligible_rows = round(
        weighted_total(
            "salda_eligible_fraction", steps_per_epoch * global_batch_size
        )
    )
    applied_rows = round(
        weighted_total(
            "salda_applied_fraction", steps_per_epoch * global_batch_size
        )
    )
    batches_with_actions = round(
        weighted_total("salda_batch_action_coverage", steps_per_epoch)
    )
    fallback_batches = round(
        weighted_total("salda_fallback_fraction", steps_per_epoch)
    )
    reason_counts = {
        "gain_threshold_abstention_rows": "salda_gain_abstention_fraction",
        "margin_threshold_abstention_rows": "salda_margin_abstention_fraction",
        "budget_excluded_rows": "salda_budget_excluded_fraction",
        "invalid_score_rows": "salda_invalid_fraction",
    }
    result = {
        name: round(weighted_total(metric, steps_per_epoch * global_batch_size))
        for name, metric in reason_counts.items()
    }
    result.update(
        {
            "scored_batches": scored_batches,
            "scored_rows": scored_rows,
            "eligible_rows": eligible_rows,
            "applied_rows": applied_rows,
            "batches_with_actions": batches_with_actions,
            "fallback_batches": fallback_batches,
            "row_coverage": (
                applied_rows / scored_rows if scored_rows else 0.0
            ),
            "batch_coverage": (
                batches_with_actions / scored_batches if scored_batches else 0.0
            ),
            "mean_dose_over_all_scored_rows": weighted_total(
                "salda_dose_mean",
                steps_per_epoch,
            )
            / scored_batches
            if scored_batches
            else 0.0,
            "mean_relative_ess": weighted_total(
                "salda_weight_relative_ess",
                steps_per_epoch,
            )
            / scored_batches
            if scored_batches
            else 1.0,
        }
    )
    return result


def _compute_steps_per_epoch(
    train_examples: int,
    batch_size: int,
) -> int:
    """Return update steps and reject datasets that yield no full batch."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive. Got {batch_size}.")

    steps_per_epoch = train_examples // batch_size
    if steps_per_epoch < 1:
        raise ValueError(
            "Training dataset yields no full batch with drop_remainder=true: "
            f"train_examples={train_examples}, batch_size={batch_size}."
        )

    return steps_per_epoch


def _namespace_extra_metrics(
    metrics: dict[str, object],
) -> dict[str, object]:
    """Preserve namespaced metrics and prefix legacy SUMix metric names."""
    prefixed_metrics = {}

    for key, value in metrics.items():
        if key.startswith(("mix_", "sumix_", "metaaugment_", "salda_")):
            prefixed_metrics[key] = value
        else:
            prefixed_metrics[f"sumix_{key}"] = value

    return prefixed_metrics


def _inject_salda_noop_epoch_metrics(
    metrics: dict[str, object],
    *,
    policy_mode: str,
    score_phase_active: bool = True,
) -> None:
    """Inject inactive metrics once per epoch without device-side chains."""

    if policy_mode != "noop" and score_phase_active:
        return
    expected_keys = set(SALDA_GA_METRIC_NAMES)
    collisions = expected_keys.intersection(metrics)
    if collisions:
        raise RuntimeError(
            "SalDA no-op produced unexpected device metrics: "
            + ", ".join(sorted(collisions))
        )
    metrics.update(SALDA_GA_NOOP_METRICS)
    observed = {
        name: metrics[name] for name in SALDA_GA_METRIC_NAMES if name in metrics
    }
    if observed != SALDA_GA_NOOP_METRICS or set(observed) != expected_keys:
        raise RuntimeError("SalDA no-op epoch metric schema is incomplete")


def _wandb_extra_metrics(
    metrics: dict[str, object],
) -> dict[str, float]:
    """Route strategy metrics into readable W&B namespaces."""
    values = {}

    for key, value in metrics.items():
        if not isinstance(value, (bool, int, float, np.number)):
            continue
        if key.startswith("metaaugment_"):
            name = key.removeprefix(
                "metaaugment_",
            )
            values[f"metaaugment/{name}"] = value
        elif key.startswith("salda_"):
            name = key.removeprefix("salda_")
            values[f"salda/{name}"] = value
        else:
            values[f"debug/{key}"] = value

    return values


def _validate_method_model_compatibility(
    args,
    method_name: str,
) -> None:
    """Reject method/model combinations that cannot be wired correctly."""
    if method_name not in CATCHUPMIX_METHOD_NAMES:
        return

    feature_hook_count = get_feature_hook_count(
        args.model,
    )

    if args.catchupmix_num_layers > feature_hook_count:
        raise ValueError(
            "CatchUpMix layer count is incompatible with the selected model. "
            f"Model '{args.model}' exposes {feature_hook_count} feature-hook "
            f"layers, but catchupmix_num_layers={args.catchupmix_num_layers}. "
            f"Set catchupmix_num_layers <= {feature_hook_count} for this model.",
        )


def _wandb_config_from_args(
    args,
) -> dict[str, object]:
    """Build a compact scalar config for experiment tracking."""
    config = {}

    for key, value in vars(
        args,
    ).items():
        if isinstance(value, (str, int, float, bool, list)):
            config[key] = value

    return config


def _salda_validation_direction_config(
    args,
    *,
    dataset_protocol: InstantaneousGADatasetProtocol = (
        CIFAR100_INSTANTANEOUS_GA_PROTOCOL
    ),
) -> dict[str, object]:
    """Resolve the full-pool or balanced mini-batch Vdev direction layout."""

    return build_validation_direction_config(
        vars(args),
        protocol=dataset_protocol,
    )


def _expected_salda_direction_counts(
    *,
    mode: str,
    updates: int,
    validation_batch_size: int,
    reanchor_interval: int,
    initial_optimizer_step: int = 0,
    validation_pool_examples: int = 5_000,
) -> tuple[int, int]:
    """Return gradient-evaluation and exact-reanchor counts."""

    if updates < 0 or initial_optimizer_step < 0:
        raise ValueError("updates and initial optimizer step must be non-negative")
    if mode == "full":
        return updates, 0
    if mode != "batch_aggregate":
        raise ValueError("validation direction mode must be registered")
    if updates == 0:
        return 0, 0
    if (
        validation_batch_size <= 0
        or validation_pool_examples % validation_batch_size
        or reanchor_interval <= 0
    ):
        raise ValueError(
            "batch size must be positive and divide the Vdev pool; "
            "reanchor interval must be positive"
        )
    cycle_length = validation_pool_examples // validation_batch_size
    if reanchor_interval % cycle_length:
        raise ValueError("reanchor interval must contain complete cycles")
    final_step = initial_optimizer_step + updates - 1
    later_reanchors = (
        final_step // reanchor_interval - initial_optimizer_step // reanchor_interval
    )
    reanchors = 1 + later_reanchors
    gradient_evaluations = reanchors * cycle_length + updates - reanchors
    return gradient_evaluations, reanchors


def _expected_salda_direction_example_visits(
    *,
    mode: str,
    updates: int,
    validation_batch_size: int,
    reanchor_interval: int,
    initial_optimizer_step: int = 0,
    validation_pool_examples: int = 5_000,
) -> int:
    """Return exact Vdev row visits for full or batch-aggregate scoring."""

    _, reanchors = _expected_salda_direction_counts(
        mode=mode,
        updates=updates,
        validation_batch_size=validation_batch_size,
        reanchor_interval=reanchor_interval,
        initial_optimizer_step=initial_optimizer_step,
        validation_pool_examples=validation_pool_examples,
    )
    if mode == "full":
        return updates * validation_pool_examples
    return (
        reanchors * validation_pool_examples
        + (updates - reanchors) * validation_batch_size
    )


def _salda_direction_workload_by_epoch(
    *,
    mode: str,
    epochs: int,
    updates_per_epoch: int,
    direction_active: bool,
    score_start_optimizer_step: int = 0,
    score_stop_optimizer_step: int | None = None,
    validation_batch_size: int,
    reanchor_interval: int,
    validation_pool_examples: int = 5_000,
) -> list[dict[str, int]]:
    """Resolve exact phase-aware direction work for consecutive epochs."""

    if epochs < 0 or updates_per_epoch < 0:
        raise ValueError("epochs and updates per epoch must be non-negative")
    if mode not in {"full", "batch_aggregate"}:
        raise ValueError("validation direction mode must be registered")
    if mode == "batch_aggregate" and (
        validation_batch_size <= 0
        or validation_pool_examples % validation_batch_size
        or reanchor_interval <= 0
    ):
        raise ValueError("batch aggregate workload configuration is invalid")
    rows = []
    for epoch_index in range(epochs):
        initial_step = epoch_index * updates_per_epoch
        final_step = initial_step + updates_per_epoch
        direction_initial_step = max(initial_step, score_start_optimizer_step)
        direction_final_step = (
            final_step
            if score_stop_optimizer_step is None
            else min(final_step, score_stop_optimizer_step)
        )
        updates = (
            max(0, direction_final_step - direction_initial_step)
            if direction_active
            else 0
        )
        if mode == "full":
            evaluations = updates
            reanchors = 0
            visits = updates * validation_pool_examples
        elif mode == "batch_aggregate":
            reanchors = sum(
                step % reanchor_interval == 0
                for step in range(
                    direction_initial_step,
                    direction_initial_step + updates,
                )
            )
            cycle_length = validation_pool_examples // validation_batch_size
            evaluations = reanchors * cycle_length + updates - reanchors
            visits = (
                reanchors * validation_pool_examples
                + (updates - reanchors) * validation_batch_size
            )
        rows.append(
            {
                "epoch": epoch_index + 1,
                "initial_optimizer_step": initial_step,
                "direction_refreshes": updates,
                "validation_gradient_evaluations": evaluations,
                "validation_exact_reanchors": reanchors,
                "direction_validation_example_visits": visits,
            }
        )
    return rows


def _salda_runtime_config_payload(
    args,
    *,
    method_name: str,
    dataset_protocol: InstantaneousGADatasetProtocol | None = None,
) -> dict[str, object]:
    """Return the non-circular scientific configuration used by instantaneous GA."""

    if dataset_protocol is None:
        dataset_protocol = get_instantaneous_ga_dataset_protocol(args.dataset)
    return build_runtime_config(
        vars(args),
        method_name=method_name,
        protocol=dataset_protocol,
    )


def _salda_training_recipe_from_runtime_config(
    runtime_config: dict[str, object],
) -> dict[str, object]:
    """Remove only GA-policy fields from a sealed SalDA training config."""

    return build_training_recipe(runtime_config)


def _salda_data_protocol_payload(
    *,
    method_name: str,
    validation_fingerprint: str,
    dataset_protocol: InstantaneousGADatasetProtocol = (
        CIFAR100_INSTANTANEOUS_GA_PROTOCOL
    ),
) -> dict[str, object]:
    """Build one registered train/Vdev protocol used by instantaneous GA."""

    return build_data_protocol(
        method_name=method_name,
        validation_fingerprint=validation_fingerprint,
        protocol=dataset_protocol,
    )


def _salda_data_protocol_from_run_protocol(
    protocol: dict[str, object],
) -> dict[str, object]:
    """Rebuild and validate a run's data seal from its primitive fields."""

    if not isinstance(protocol, dict):
        raise TypeError("SalDA run protocol is invalid")
    runtime_config = protocol.get("runtime_config")
    _salda_training_recipe_from_runtime_config(runtime_config)
    if hashlib.sha256(
        json.dumps(
            runtime_config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest() != protocol.get("runtime_config_sha256"):
        raise ValueError("SalDA run runtime config seal changed")
    method_name = protocol.get("method")
    validation_fingerprint = protocol.get("validation_fingerprint")
    dataset_protocol = get_instantaneous_ga_dataset_protocol(protocol.get("dataset"))
    expected = _salda_data_protocol_payload(
        method_name=method_name,
        validation_fingerprint=validation_fingerprint,
        dataset_protocol=dataset_protocol,
    )
    primitive_fields = build_run_protocol_data_fields(expected)
    mismatches = [
        key for key, value in primitive_fields.items() if protocol.get(key) != value
    ]
    validation = runtime_config["validation"]
    runtime_expected = {
        "base_method": expected["method"],
        "training_data": expected["training_data"],
        "global_batch_size": expected["global_batch_size"],
    }
    mismatches.extend(
        f"runtime_config.{key}"
        for key, value in runtime_expected.items()
        if runtime_config.get(key) != value
    )
    validation_expected = {
        "source": expected["val_source"],
        "split": expected["validation_split"],
        "examples": expected["validation_examples"],
        "epoch_batches": dataset_protocol.validation_batches_per_epoch,
    }
    mismatches.extend(
        f"runtime_config.validation.{key}"
        for key, value in validation_expected.items()
        if validation.get(key) != value
    )
    if mismatches:
        raise ValueError(
            "SalDA run data protocol fields changed: " + ", ".join(mismatches)
        )
    expected_sha256 = hashlib.sha256(
        json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if protocol.get("data_protocol_sha256") != expected_sha256:
        raise ValueError("SalDA run data protocol SHA does not match its fields")
    return expected


def main() -> None:
    """Run the command-line entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    args = parse_args()
    args.data_seed = resolve_data_seed(
        experiment_seed=args.seed,
        data_seed=args.data_seed,
    )
    seed_everything(
        seed=args.seed,
        strict_determinism=args.strict_determinism,
    )
    print(
        "Reproducibility | "
        f"seed={args.seed} | data_seed={args.data_seed} | "
        f"deterministic_data={args.deterministic_data} | "
        f"strict_determinism={args.strict_determinism}"
    )
    source_validation_split = resolve_training_validation_split(
        validation_split=args.validation_split,
        val_source=args.val_source,
    )
    if args.val_source == "test":
        print(
            "Evaluation protocol | train=full official train | "
            f"checkpoint_validation={args.validation_split:.6g} of official "
            "eval | final_eval=sealed official-eval complement"
        )
    else:
        print(
            "Evaluation protocol | "
            f"train/validation=official train split "
            f"({1.0 - args.validation_split:.6g}/"
            f"{args.validation_split:.6g}) | final_eval=full official eval"
        )

    method_name = normalize_method_name(
        args.method,
    )
    formal_run_context = prepare_formal_run(
        args,
        method_name=method_name,
        local_device_count=jax.local_device_count(),
    )
    formal_run_active = formal_run_context is not None
    salda_ga_active = args.salda_ga_mode != "off"
    salda_strategy_active = args.salda_ga_mode not in {"off", "baseline"}
    salda_dataset_protocol = (
        get_instantaneous_ga_dataset_protocol(args.dataset) if salda_ga_active else None
    )
    if salda_ga_active:
        _validate_salda_checkout(args.salda_ga_git_commit)
    _validate_method_model_compatibility(
        args=args,
        method_name=method_name,
    )

    if method_name == "ifaugnet":
        from allthemix.competitors.ifaugnet.runner import run_ifaugnet

        run_ifaugnet(
            args,
        )

        return

    debug_sumix_metrics = args.debug_sumix_metrics and method_name == "cutmix_sumix"
    sumix_debug_metric_names = SUMIX_DEBUG_METRIC_NAMES if debug_sumix_metrics else []
    debug_mix_metrics = args.debug_mix_metrics and method_name not in {
        "baseline",
        "metaaugment",
        *DIFFUSEMIX_METHOD_NAMES,
        *ALIA_METHOD_NAMES,
    }
    extra_metric_names = [
        *(MIX_DEBUG_METRIC_NAMES if debug_mix_metrics else []),
        *sumix_debug_metric_names,
        *(META_AUGMENT_METRIC_NAMES if method_name == "metaaugment" else []),
        *(SALDA_GA_METRIC_NAMES if salda_strategy_active else []),
        *(
            [f"salda_time_{name}_seconds" for name in SALDA_GA_COMPONENT_NAMES]
            if salda_ga_active
            else []
        ),
        *(SALDA_GA_PROTOCOL_CSV_NAMES if salda_ga_active else []),
    ]
    precomputed_saliency_methods = (
        "saliencymix",
        "guidedmixup",
    )

    if method_name in precomputed_saliency_methods and args.basic_aug:
        raise ValueError(
            "For SaliencyMix/precomputed GuidedMixup, set basic_aug: false "
            "and use sal_aug_recipe for paired image/saliency augmentation.",
        )

    if method_name not in precomputed_saliency_methods and args.sal_basic_aug:
        raise ValueError(
            "sal_basic_aug should only be used with SaliencyMix/precomputed "
            "GuidedMixup.",
        )

    if (
        method_name not in precomputed_saliency_methods
        and args.sal_aug_recipe != "none"
    ):
        raise ValueError(
            "sal_aug_recipe should only be used with SaliencyMix/precomputed "
            "GuidedMixup.",
        )

    run_name, wandb_run, output_path, checkpoint_path = prepare_run_outputs(
        args=args,
        extra_metric_names=extra_metric_names,
        wandb_config=_wandb_config_from_args(
            args,
        ),
    )

    metadata = get_metadata(args.dataset)

    offline = plan_offline_manifests(
        args=args,
        method_name=method_name,
        metadata=metadata,
        source_validation_split=source_validation_split,
    )

    (
        train_ds,
        eval_ds,
        final_test_ds,
        meta_validation_ds,
    ) = build_training_datasets(
        args=args,
        method_name=method_name,
        precomputed_saliency_methods=precomputed_saliency_methods,
        offline=offline,
        # #### SALDA VTEST PRELOAD EXCLUSION: START ####
        include_final_test=(
            args.final_test and not salda_ga_active and args.val_source != "test"
        ),
        # #### SALDA VTEST PRELOAD EXCLUSION: END ####
    )

    salda_validation_images = None
    salda_validation_labels = None
    salda_validation_fingerprint = None
    salda_direction_prerequisite = None
    if salda_ga_active:
        (
            salda_validation_images,
            salda_validation_labels,
            salda_validation_fingerprint,
        ) = _materialize_salda_validation(
            eval_ds,
            num_classes=metadata.num_classes,
            expected_validation_examples=(salda_dataset_protocol.validation_examples),
            expected_examples_per_class=(
                salda_dataset_protocol.validation_examples_per_class
            ),
        )
        if salda_dataset_protocol.dataset == "stl10":
            direction_artifact = os.environ.get(
                "ALLTHEMIX_SALDA_DIRECTION_ARTIFACT",
                "",
            )
            direction_artifact_file_sha256 = os.environ.get(
                "ALLTHEMIX_SALDA_DIRECTION_ARTIFACT_SHA256",
                "",
            )
            if not direction_artifact or not direction_artifact_file_sha256:
                raise RuntimeError(
                    "STL-10 SalDA requires exact direction artifact path and SHA"
                )
            salda_direction_prerequisite = (
                _validate_stl10_direction_prerequisite(
                    direction_artifact,
                    expected_artifact_file_sha256=(
                        direction_artifact_file_sha256
                    ),
                    expected_commit=args.salda_ga_git_commit,
                    expected_validation_pool_sha256=(
                        salda_validation_fingerprint
                    ),
                )
            )

    model = build_model(
        name=args.model,
        num_classes=metadata.num_classes,
        resnet_stem_type=args.resnet_stem_type,
        preact_stem_bn_relu=args.preact_stem_bn_relu,
        preact_pytorch_default_init=args.preact_pytorch_default_init,
    )

    mixer_fn = get_mixer(
        # Validation-aware strategies own their augmented task batch. The
        # identity mixer keeps the shared loop signature uniform but is never
        # invoked by the strategy path.
        name=(
            "baseline"
            if (
                method_name == "metaaugment"
                or method_name in DIFFUSEMIX_METHOD_NAMES
                or method_name in ALIA_METHOD_NAMES
            )
            else args.method
        ),
        num_classes=metadata.num_classes,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        cutmix_no_repeat=args.cutmix_no_repeat,
        cutmix_variant=args.cutmix_variant,
        cutmix_per_sample_lam=args.cutmix_per_sample_lam,
        cutmix_min_lam=args.cutmix_min_lam,
        fmix_alpha=args.fmix_alpha,
        fmix_decay=args.fmix_decay,
        fmix_prob=args.fmix_prob,
        fmix_per_sample=args.fmix_per_sample,
        fmix_no_repeat=args.fmix_no_repeat,
        saliencymix_alpha=args.saliencymix_alpha,
        saliencymix_prob=args.saliencymix_prob,
        saliencymix_per_sample=args.saliencymix_per_sample,
        resizemix_scope_min=args.resizemix_scope_min,
        resizemix_scope_max=args.resizemix_scope_max,
        resizemix_prob=args.resizemix_prob,
        resizemix_per_sample=args.resizemix_per_sample,
        guidedmixup_alpha=args.guidedmixup_alpha,
        guidedmixup_prob=args.guidedmixup_prob,
        guidedmixup_blur_kernel=args.guidedmixup_blur_kernel,
        guidedmixup_condition=args.guidedmixup_condition,
        catchupmix_alpha=args.catchupmix_alpha,
        catchupmix_cutmix_alpha=args.catchupmix_cutmix_alpha,
        catchupmix_num_layers=args.catchupmix_num_layers,
        catchupmix_no_repeat=args.catchupmix_no_repeat,
    )

    train_examples = resolve_train_example_count(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        metadata=metadata,
        validation_split=source_validation_split,
        train_subset_fraction=args.train_subset_fraction,
    )

    if args.debug_train_source != "none":
        full_source_examples = resolve_train_example_count(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            metadata=metadata,
            validation_split=0.0,
        )
        if args.debug_train_source == "val_only":
            train_examples = full_source_examples - train_examples
        else:
            train_examples = full_source_examples
        print(
            "DEBUG LEAKAGE ARM | "
            f"debug_train_source={args.debug_train_source} | "
            f"train pipeline consumes {train_examples} examples "
            "(includes held-out validation data) | eval pipeline "
            "unchanged | diagnostics only, never a table row"
        )

    if offline.example_count is not None and offline.manifest_mode != "sample":
        train_examples = offline.example_count
        if offline.manifest_mode == "append":
            if offline.original_example_count is None:
                raise RuntimeError("Offline append cardinality was not initialized.")
            train_examples += offline.original_example_count

    steps_per_epoch = _compute_steps_per_epoch(
        train_examples=train_examples,
        batch_size=args.batch_size,
    )
    if salda_ga_active and (
        train_examples != salda_dataset_protocol.train_examples
        or steps_per_epoch != salda_dataset_protocol.steps_per_epoch
    ):
        raise ValueError(
            f"{salda_dataset_protocol.dataset} SalDA requires "
            f"{salda_dataset_protocol.train_examples} training examples and "
            f"{salda_dataset_protocol.steps_per_epoch} full batches per epoch; "
            f"received {train_examples} and {steps_per_epoch}"
        )
    logged_steps_per_epoch = steps_per_epoch

    if args.max_train_steps > 0:
        logged_steps_per_epoch = min(
            steps_per_epoch,
            args.max_train_steps,
        )

    lr_schedule = build_lr_schedule(
        schedule_name=args.lr_schedule,
        base_learning_rate=args.learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        decay_epochs=args.lr_decay_epochs,
        decay_rate=args.lr_decay_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_epochs=args.warmup_epochs,
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(
        rng=init_rng,
        model=model,
        learning_rate=lr_schedule,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        input_shape=(
            args.batch_size,
            metadata.image_size,
            metadata.image_size,
            metadata.channels,
        ),
    )

    salda_strategy = None
    salda_protocol_path = None
    if salda_strategy_active:
        if jax.local_device_count() != 4:
            raise ValueError("SalDA requires exactly four local devices")
        if salda_validation_images is None or salda_validation_labels is None:
            raise RuntimeError("SalDA Vdev was not materialized")
        from salutary_da.gradient_alignment_strategy import (
            GradientAlignmentBatchStrategy,
        )
        from salutary_da.policies.per_row_continuous import (
            PerRowContinuousPolicyConfig,
        )

        policy_mode = {
            "noop": "score_only",
            "shuffled_soft_label": "soft_label",
            "shuffled_reweight": "reweight",
        }.get(args.salda_ga_mode, args.salda_ga_mode)
        salda_strategy = GradientAlignmentBatchStrategy(
            apply_fn=state.apply_fn,
            template_params=state.params,
            mixer_fn=mixer_fn,
            num_classes=metadata.num_classes,
            validation_images=salda_validation_images,
            validation_labels=salda_validation_labels,
            validation_direction_mode=(args.salda_ga_validation_direction_mode),
            validation_batch_size=args.salda_ga_validation_batch_size,
            validation_batch_seed=args.seed,
            validation_reanchor_interval=(args.salda_ga_validation_reanchor_interval),
            initial_optimizer_step=int(jax.device_get(state.step)),
            learning_rate_fn=lr_schedule,
            policy=PerRowContinuousPolicyConfig(
                mode=policy_mode,
                maximum_rows=args.salda_ga_maximum_rows,
                soft_label_dose=args.salda_ga_soft_label_dose,
                maximum_weight_deviation=args.salda_ga_max_weight_deviation,
                weight_temperature=args.salda_ga_weight_temperature,
                minimum_relative_ess=args.salda_ga_minimum_relative_ess,
                minimum_gain=args.salda_ga_minimum_gain,
                minimum_label_margin=args.salda_ga_minimum_label_margin,
                minimum_relative_label_margin=(
                    args.salda_ga_minimum_relative_label_margin
                ),
                fallback_enabled=args.salda_ga_fallback_enabled,
                fallback_soft_label_dose=(args.salda_ga_fallback_soft_label_dose),
            ),
            parameter_scope=args.salda_ga_parameter_scope,
            sync_batch_stats=True,
            action_enabled=args.salda_ga_mode != "noop",
            expected_validation_examples=(salda_dataset_protocol.validation_examples),
            audit_mode=args.salda_ga_audit_mode,
            profile_components=args.salda_ga_profile_components,
            base_method=method_name,
            shuffled_control=args.salda_ga_mode.startswith("shuffled_"),
            control_seed=args.seed,
            score_start_optimizer_step=(
                args.salda_ga_score_start_epoch * steps_per_epoch
            ),
            score_stop_optimizer_step=(
                None
                if args.salda_ga_score_stop_epoch == -1
                else args.salda_ga_score_stop_epoch * steps_per_epoch
            ),
            action_start_optimizer_step=(
                args.salda_ga_action_start_epoch * steps_per_epoch
            ),
            action_stop_optimizer_step=(
                None
                if args.salda_ga_action_stop_epoch == -1
                else args.salda_ga_action_stop_epoch * steps_per_epoch
            ),
        )
    salda_protocol_payload = None
    salda_config_sha256 = None
    salda_runtime_config_sha256 = None
    salda_training_recipe_sha256 = None
    salda_data_protocol_sha256 = None
    if salda_ga_active:
        if jax.local_device_count() != 4:
            raise ValueError("SalDA requires exactly four local devices")
        if salda_validation_fingerprint is None:
            raise RuntimeError("SalDA Vdev fingerprint was not created")
        resolved_config = _wandb_config_from_args(args)
        resolved_config_sha256 = hashlib.sha256(
            json.dumps(
                resolved_config,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        salda_config_sha256 = resolved_config_sha256
        salda_runtime_config = _salda_runtime_config_payload(
            args,
            method_name=method_name,
            dataset_protocol=salda_dataset_protocol,
        )
        salda_validation_direction = _salda_validation_direction_config(
            args,
            dataset_protocol=salda_dataset_protocol,
        )
        salda_runtime_config_sha256 = hashlib.sha256(
            json.dumps(
                salda_runtime_config,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        salda_training_recipe = _salda_training_recipe_from_runtime_config(
            salda_runtime_config
        )
        salda_training_recipe_sha256 = hashlib.sha256(
            json.dumps(
                salda_training_recipe,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        data_protocol = _salda_data_protocol_payload(
            method_name=method_name,
            validation_fingerprint=salda_validation_fingerprint,
            dataset_protocol=salda_dataset_protocol,
        )
        salda_data_protocol_sha256 = hashlib.sha256(
            json.dumps(
                data_protocol,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        validation_strategy_summary = (
            salda_strategy.execution_summary() if salda_strategy is not None else None
        )
        validation_schedule_sha256 = (
            validation_strategy_summary["validation_batch_schedule_sha256"]
            if validation_strategy_summary is not None
            else None
        )
        expected_epoch_directions = (
            0
            if args.salda_ga_mode in {"baseline", "noop"}
            else salda_dataset_protocol.steps_per_epoch
        )
        score_start_optimizer_step = (
            args.salda_ga_score_start_epoch * salda_dataset_protocol.steps_per_epoch
        )
        score_stop_optimizer_step = (
            None
            if args.salda_ga_score_stop_epoch == -1
            else args.salda_ga_score_stop_epoch
            * salda_dataset_protocol.steps_per_epoch
        )
        first_ten_epoch_direction_workload = _salda_direction_workload_by_epoch(
            mode=args.salda_ga_validation_direction_mode,
            epochs=10,
            updates_per_epoch=salda_dataset_protocol.steps_per_epoch,
            direction_active=expected_epoch_directions > 0,
            score_start_optimizer_step=score_start_optimizer_step,
            score_stop_optimizer_step=score_stop_optimizer_step,
            validation_batch_size=args.salda_ga_validation_batch_size,
            reanchor_interval=args.salda_ga_validation_reanchor_interval,
            validation_pool_examples=(salda_dataset_protocol.validation_examples),
        )
        direction_workload_is_phase_dependent = bool(
            expected_epoch_directions > 0
            and (
                args.salda_ga_validation_direction_mode == "batch_aggregate"
                or score_start_optimizer_step > 0
                or score_stop_optimizer_step is not None
            )
        )
        first_epoch_direction_workload = first_ten_epoch_direction_workload[0]
        constant_epoch_direction_workload = (
            None
            if direction_workload_is_phase_dependent
            else first_epoch_direction_workload
        )
        salda_protocol_payload = {
            "schema_version": 1,
            "dataset": salda_dataset_protocol.dataset,
            "model": "preact_resnet18",
            "seed": args.seed,
            "resolved_data_seed": args.data_seed,
            "data_seed_policy": "resolved_from_training_seed",
            "method": method_name,
            "training_data": (
                "original_images" if method_name == "baseline" else "online_mixup"
            ),
            "git_commit": args.salda_ga_git_commit,
            "resolved_config_sha256": resolved_config_sha256,
            "runtime_config": salda_runtime_config,
            "runtime_config_sha256": salda_runtime_config_sha256,
            "training_recipe": salda_training_recipe,
            "training_recipe_sha256": salda_training_recipe_sha256,
            "data_protocol_sha256": salda_data_protocol_sha256,
            "validation_fingerprint": salda_validation_fingerprint,
            "validation_pool_sha256": salda_validation_fingerprint,
            **(
                {"direction_prerequisite": salda_direction_prerequisite}
                if salda_dataset_protocol.dataset == "stl10"
                else {}
            ),
            "validation_batch_schedule_sha256": (validation_schedule_sha256),
            "validation_batch_seed": (
                validation_strategy_summary["validation_batch_seed"]
                if validation_strategy_summary is not None
                else None
            ),
            "validation_initial_optimizer_step": (
                validation_strategy_summary["validation_initial_optimizer_step"]
                if validation_strategy_summary is not None
                else None
            ),
            "train_examples": train_examples,
            "validation_examples": salda_dataset_protocol.validation_examples,
            "validation_class_counts": list(
                salda_dataset_protocol.validation_class_counts
            ),
            "vdev_role": "training_direction_and_checkpoint_selection",
            "vtest_role": "sealed",
            "vtest_loaded": False,
            "endpoint_builder_calls": 0,
            "steps_per_epoch": steps_per_epoch,
            "direction_refresh_optimizer_steps": 1,
            "validation_direction_mode": salda_validation_direction[
                "validation_direction_mode"
            ],
            "validation_examples_per_gradient_evaluation": (
                salda_validation_direction[
                    "validation_examples_per_gradient_evaluation"
                ]
            ),
            "validation_reanchor_interval": salda_validation_direction[
                "validation_reanchor_interval"
            ],
            "validation_direction_cycle_length": salda_validation_direction[
                "validation_direction_cycle_length"
            ],
            "validation_direction_main_table_eligible": (
                salda_validation_direction["validation_direction_main_table_eligible"]
            ),
            "directions_per_complete_epoch": (
                None
                if (
                    score_start_optimizer_step > 0
                    or score_stop_optimizer_step is not None
                )
                else expected_epoch_directions
            ),
            "direction_validation_example_visits_per_complete_epoch": (
                None
                if constant_epoch_direction_workload is None
                else constant_epoch_direction_workload[
                    "direction_validation_example_visits"
                ]
            ),
            "validation_gradient_evaluations_per_complete_epoch": (
                None
                if constant_epoch_direction_workload is None
                else constant_epoch_direction_workload[
                    "validation_gradient_evaluations"
                ]
            ),
            "validation_exact_reanchors_per_complete_epoch": (
                None
                if constant_epoch_direction_workload is None
                else constant_epoch_direction_workload["validation_exact_reanchors"]
            ),
            "direction_workload_is_optimizer_phase_dependent": (
                direction_workload_is_phase_dependent
            ),
            "direction_workload_first_ten_epochs": (first_ten_epoch_direction_workload),
            "direction_validation_example_visits_are_runtime_counted": True,
            "parameter_scope": args.salda_ga_parameter_scope,
            "policy_mode": args.salda_ga_mode,
            "audit_mode": args.salda_ga_audit_mode,
            "component_profile": args.salda_ga_profile_components,
            "distributed": True,
            "sync_batch_stats": True,
            "global_batch_size": salda_dataset_protocol.global_batch_size,
            "input_pipeline_implementation": "tensorflow_data_pipeline",
        }
        _salda_data_protocol_from_run_protocol(salda_protocol_payload)
        salda_protocol_path = _write_salda_json(
            args.output_dir,
            run_name,
            "salda_protocol",
            salda_protocol_payload,
        )
        print(f"SalDA protocol artifact: {salda_protocol_path}")

    metaaugment_context = None
    if method_name == "metaaugment":
        if meta_validation_ds is None:
            raise RuntimeError("MetaAugment validation stream was not initialized.")
        rng, policy_rng = jax.random.split(
            rng,
        )
        metaaugment_context = create_metaaugment_context(
            rng=policy_rng,
            task_state=state,
            meta_dataset=meta_validation_ds,
            input_shape=(
                args.batch_size,
                metadata.image_size,
                metadata.image_size,
                metadata.channels,
            ),
            dataset=args.dataset,
            num_classes=metadata.num_classes,
            policy_learning_rate=args.metaaugment_policy_learning_rate,
            policy_momentum=args.metaaugment_policy_momentum,
            policy_weight_decay=args.metaaugment_policy_weight_decay,
            policy_nesterov=args.metaaugment_policy_nesterov,
            inner_learning_rate=args.metaaugment_inner_learning_rate,
            learn_inner_learning_rate=(args.metaaugment_learn_inner_learning_rate),
            epsilon=args.metaaugment_epsilon,
            num_transforms_per_sample=(args.metaaugment_num_transforms_per_sample),
            cutout_size=args.metaaugment_cutout_size,
            sampler_update_epochs=args.metaaugment_sampler_update_epochs,
            sampler_history_epochs=args.metaaugment_sampler_history_epochs,
            translate_const=args.metaaugment_translate_const,
            tiny_imagenet_normalization=args.tiny_imagenet_normalization,
            distributed=args.distributed,
            sync_batch_stats=args.sync_batch_stats,
        )
        print(
            "Using integrated MetaAugment policy with the shared "
            "AllTheMix model, data pipeline, and epoch loop"
        )

    state = restore_initial_state(
        args=args,
        state=state,
        metaaugment_context=metaaugment_context,
    )

    salda_initial_optimizer_step = None
    if salda_ga_active:
        salda_initial_optimizer_step = int(
            np.asarray(jax.device_get(state.step)).item()
        )

    if args.distributed:
        print(f"Using distributed training with {jax.local_device_count()} devices")
        if args.sync_batch_stats:
            print("Using synchronized BatchNorm statistics across devices")

        if method_name == "cutmix_sumix":
            print(
                "Warning: distributed cutmix_sumix mixes samples within each "
                "device shard. Official SUMix DDP shuffles across devices, so "
                "single-device runs are the closest paper-aligned setting here."
            )

        cross_device_supported_methods = {
            "mixup",
            "cutmix",
            "saliencymix",
            "fmix",
            "resizemix",
        }

        if (
            args.cross_device_shuffle
            and method_name not in cross_device_supported_methods
        ):
            print(
                "Warning: cross_device_shuffle is implemented for MixUp, "
                "CutMix, SaliencyMix, FMix, and ResizeMix only and will not "
                "affect this method."
            )

        state = replicate_state(state)
        rngs = create_device_rngs(rng)
        if metaaugment_context is not None:
            metaaugment_context.replicate_method_state()

    else:
        rngs = None

    best_top1_error = float("inf")
    best_epoch = -1
    best_eval_state = None
    eval_name = "test" if args.eval_on_test_each_epoch else "val"
    early_stop_config = EarlyStopConfig(
        enabled=args.early_stop_enabled,
        start_epoch=args.early_stop_start_epoch,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
    )
    early_stop_state = EarlyStopState()
    run_epochs = (
        args.epochs
        if not salda_ga_active or args.salda_ga_stop_epoch == -1
        else args.salda_ga_stop_epoch
    )
    salda_epoch_records = []
    formal_epoch_records = []

    for epoch in range(run_epochs):
        epoch_wall_started = time.perf_counter()
        epoch_component_times = {name: 0.0 for name in SALDA_GA_COMPONENT_NAMES}
        timer = Timer()
        epoch_time = None
        component_before = (
            salda_strategy.timing_totals()
            if salda_strategy is not None and args.salda_ga_profile_components
            else {}
        )

        if args.log_time:
            timer.start()

        if args.distributed:
            train_started = time.perf_counter()
            (
                state,
                rngs,
                train_loss,
                train_accuracy,
                raw_extra_metrics,
                train_batch_count,
            ) = parallel_train_one_epoch(
                state=state,
                rngs=rngs,
                train_ds=train_ds,
                mixer_fn=mixer_fn,
                method=args.method,
                num_classes=metadata.num_classes,
                sumix_gamma=args.sumix_gamma,
                sumix_semantic_scale=args.sumix_semantic_scale,
                max_train_steps=args.max_train_steps,
                return_sumix_metrics=debug_sumix_metrics,
                cross_device_shuffle=args.cross_device_shuffle,
                cross_device_no_repeat=(
                    args.fmix_no_repeat
                    if method_name == "fmix"
                    else args.cutmix_no_repeat
                ),
                sync_batch_stats=args.sync_batch_stats,
                return_mix_metrics=debug_mix_metrics,
                validation_aware_strategy=metaaugment_context,
                batch_training_strategy=salda_strategy,
                component_timing=(epoch_component_times if salda_ga_active else None),
                return_batch_count=True,
            )
            train_wall_seconds = time.perf_counter() - train_started
            if (
                salda_ga_active
                and args.salda_ga_mode in {"baseline", "noop"}
                and not args.salda_ga_profile_components
            ):
                epoch_component_times["update"] = max(
                    0.0,
                    train_wall_seconds
                    - epoch_component_times["data"]
                    - epoch_component_times["metric_sync"],
                )
            score_phase_active = bool(
                salda_strategy is None
                or (
                    epoch >= args.salda_ga_score_start_epoch
                    and (
                        args.salda_ga_score_stop_epoch == -1
                        or epoch < args.salda_ga_score_stop_epoch
                    )
                )
            )
            _inject_salda_noop_epoch_metrics(
                raw_extra_metrics,
                policy_mode=args.salda_ga_mode,
                score_phase_active=score_phase_active,
            )

            vdev_eval_started = time.perf_counter()
            (
                eval_loss,
                eval_top1_accuracy,
                eval_top5_accuracy,
                eval_top1_error,
                eval_top5_error,
                vdev_batch_count,
                vdev_example_count,
            ) = parallel_evaluate(
                state=state,
                test_ds=eval_ds,
                num_classes=metadata.num_classes,
                max_eval_steps=args.max_eval_steps,
                return_counts=True,
            )
            if salda_ga_active:
                epoch_component_times["vdev_eval"] = (
                    time.perf_counter() - vdev_eval_started
                )

        else:
            (
                state,
                rng,
                train_loss,
                train_accuracy,
                raw_extra_metrics,
                train_batch_count,
            ) = train_one_epoch(
                state=state,
                rng=rng,
                train_ds=train_ds,
                mixer_fn=mixer_fn,
                method=args.method,
                num_classes=metadata.num_classes,
                sumix_gamma=args.sumix_gamma,
                sumix_semantic_scale=args.sumix_semantic_scale,
                max_train_steps=args.max_train_steps,
                return_sumix_metrics=debug_sumix_metrics,
                return_mix_metrics=debug_mix_metrics,
                validation_aware_strategy=metaaugment_context,
                return_batch_count=True,
            )

            (
                eval_loss,
                eval_top1_accuracy,
                eval_top5_accuracy,
                eval_top1_error,
                eval_top5_error,
                vdev_batch_count,
                vdev_example_count,
            ) = evaluate(
                state=state,
                test_ds=eval_ds,
                num_classes=metadata.num_classes,
                max_eval_steps=args.max_eval_steps,
                return_counts=True,
            )

        if args.log_time:
            epoch_time = timer.stop()

        if salda_strategy is not None and args.salda_ga_profile_components:
            component_after = salda_strategy.timing_totals()
            for name in SALDA_GA_COMPONENT_NAMES:
                epoch_component_times[name] += component_after.get(
                    name, 0.0
                ) - component_before.get(name, 0.0)

        if salda_ga_active:
            raw_extra_metrics.update(
                {
                    **{
                        f"salda_time_{name}_seconds": epoch_component_times[name]
                        for name in SALDA_GA_COMPONENT_NAMES
                    },
                    "salda_git_commit": args.salda_ga_git_commit,
                    "salda_config_sha256": salda_config_sha256,
                    "salda_runtime_config_sha256": salda_runtime_config_sha256,
                    "salda_data_protocol_sha256": salda_data_protocol_sha256,
                    "salda_vdev_role": ("training_direction_and_checkpoint_selection"),
                    "salda_vtest_role": "sealed",
                    "salda_train_updates": logged_steps_per_epoch,
                    "salda_vdev_batches": vdev_batch_count,
                    "salda_vtest_batches": 0,
                }
            )

        extra_metrics = _namespace_extra_metrics(
            raw_extra_metrics,
        )

        checkpoint_copy_started = time.perf_counter()
        if should_replace_best_vdev_top1_error(
            eval_top1_error,
            best_top1_error,
        ):
            best_top1_error = eval_top1_error
            best_epoch = epoch + 1
            if (
                args.final_test
                and not salda_ga_active
                and args.final_test_checkpoint == "best"
            ):
                best_eval_state = state

            if (
                args.save_checkpoint
                and args.save_best_only
                and checkpoint_path is not None
            ):
                save_state = unreplicate_state(state) if args.distributed else state

                if metaaugment_context is not None:
                    save_state = metaaugment_context.checkpoint_state(
                        task_state=save_state,
                    )

                save_best_checkpoint(
                    state=save_state,
                    checkpoint_dir=checkpoint_path,
                )

        if (
            args.save_checkpoint
            and not args.save_best_only
            and checkpoint_path is not None
        ):
            save_state = unreplicate_state(state) if args.distributed else state

            if metaaugment_context is not None:
                save_state = metaaugment_context.checkpoint_state(
                    task_state=save_state,
                )

            save_checkpoint(
                state=save_state,
                checkpoint_dir=checkpoint_path,
                epoch=epoch + 1,
            )
        if salda_ga_active:
            epoch_component_times["checkpoint_copy"] = (
                time.perf_counter() - checkpoint_copy_started
            )
            extra_metrics["salda_time_checkpoint_copy_seconds"] = epoch_component_times[
                "checkpoint_copy"
            ]

        current_learning_rate = float(
            lr_schedule(
                max(
                    0,
                    (epoch + 1) * logged_steps_per_epoch - 1,
                )
            )
        )

        probe_metrics = {}


        wandb_started = time.perf_counter()
        wandb_run.log_epoch(
            epoch=epoch + 1,
            metrics={
                **probe_metrics,
                "train/learning_rate": current_learning_rate,
                "train/loss": train_loss,
                "train/accuracy": train_accuracy,
                f"{eval_name}/loss": eval_loss,
                f"{eval_name}/top1_accuracy": eval_top1_accuracy,
                f"{eval_name}/top5_accuracy": eval_top5_accuracy,
                f"{eval_name}/top1_error": eval_top1_error,
                f"{eval_name}/top5_error": eval_top5_error,
                f"{eval_name}/best_top1_error": best_top1_error,
                "best_epoch": best_epoch,
                **_wandb_extra_metrics(
                    extra_metrics,
                ),
                **(
                    {
                        "time/epoch_seconds": epoch_time,
                    }
                    if epoch_time is not None
                    else {}
                ),
            },
        )

        if salda_ga_active:
            epoch_component_times["wandb"] = time.perf_counter() - wandb_started
            epoch_component_times["end_to_end_wall"] = (
                time.perf_counter() - epoch_wall_started
            )
            extra_metrics["salda_time_wandb_seconds"] = epoch_component_times["wandb"]
            extra_metrics["salda_time_end_to_end_wall_seconds"] = epoch_component_times[
                "end_to_end_wall"
            ]
            epoch_time = epoch_component_times["end_to_end_wall"]

        if args.save_csv and output_path is not None:
            append_epoch_result(
                output_path=output_path,
                epoch=epoch + 1,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                eval_loss=eval_loss,
                eval_top1_accuracy=eval_top1_accuracy,
                eval_top5_accuracy=eval_top5_accuracy,
                eval_top1_error=eval_top1_error,
                eval_top5_error=eval_top5_error,
                best_top1_error=best_top1_error,
                best_epoch=best_epoch,
                epoch_time=epoch_time,
                extra_metrics=extra_metrics,
                extra_metric_names=extra_metric_names,
            )

        message = format_epoch_message(
            epoch=epoch + 1,
            total_epochs=run_epochs,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            eval_loss=eval_loss,
            eval_top1_accuracy=eval_top1_accuracy,
            eval_top5_accuracy=eval_top5_accuracy,
            eval_top1_error=eval_top1_error,
            eval_top5_error=eval_top5_error,
            best_top1_error=best_top1_error,
            epoch_time=epoch_time,
            extra_metrics=extra_metrics,
            eval_name=eval_name,
        )
        print(message)

        if formal_run_active:
            formal_epoch_records.append(
                {
                    "epoch": epoch + 1,
                    "train_batches": int(train_batch_count),
                    "vdev_batches": int(vdev_batch_count),
                    "vdev_examples": int(vdev_example_count),
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "vdev_loss": eval_loss,
                    "vdev_top1_accuracy": eval_top1_accuracy,
                    "vdev_top5_accuracy": eval_top5_accuracy,
                    "vdev_top1_error": eval_top1_error,
                    "vdev_top5_error": eval_top5_error,
                }
            )

        if salda_ga_active:
            salda_epoch_records.append(
                {
                    "epoch": epoch + 1,
                    "epoch_seconds": epoch_time,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "vdev_loss": eval_loss,
                    "vdev_top1_error": eval_top1_error,
                    "vdev_top5_error": eval_top5_error,
                    "train_batches": int(train_batch_count),
                    "vdev_batches": int(vdev_batch_count),
                    "vdev_examples": int(vdev_example_count),
                    "extra_metrics": extra_metrics,
                    "component_timing_seconds": dict(epoch_component_times),
                }
            )

        early_stop_state = update_early_stop(
            state=early_stop_state,
            config=early_stop_config,
            epoch=epoch + 1,
            metric=eval_top1_error,
            metric_name=f"{eval_name} top-1 error",
        )

        if early_stop_state.should_stop:
            print(early_stop_state.reason)
            break

    if best_epoch > 0:
        print(
            f"Best {eval_name} top-1 error: {best_top1_error * 100:.2f}% "
            f"at epoch {best_epoch}"
        )

    endpoint_builder_calls = 0
    endpoint_evaluations = 0
    endpoint_batches = 0
    endpoint_examples = 0
    final_test_result = None

    formal_pre_endpoint_closure = None
    formal_restored_best = None
    formal_best_checkpoint_step = None
    if formal_run_context is not None:
        if checkpoint_path is None:
            raise RuntimeError("formal run requires a checkpoint directory")
        terminal_state = unreplicate_state(state) if args.distributed else state
        terminal_optimizer_step = int(
            np.asarray(jax.device_get(terminal_state.step)).item()
        )
        formal_pre_endpoint_closure = validate_pre_endpoint_workload(
            formal_run_context,
            epoch_records=formal_epoch_records,
            terminal_optimizer_step=terminal_optimizer_step,
            best_epoch=best_epoch,
            best_top1_error=best_top1_error,
            checkpoint_path=checkpoint_path,
        )
        formal_restored_best = restore_checkpoint(
            terminal_state,
            str((checkpoint_path / "best").resolve()),
        )
        formal_best_checkpoint_step = int(
            np.asarray(jax.device_get(formal_restored_best.step)).item()
        )
        expected_best_checkpoint_step = (
            best_epoch
            * formal_run_context.protocol["expected_workload"][
                "train_batches_per_epoch"
            ]
        )
        if formal_best_checkpoint_step != expected_best_checkpoint_step:
            raise RuntimeError(
                "best checkpoint optimizer step mismatch: "
                f"observed={formal_best_checkpoint_step}, "
                f"expected={expected_best_checkpoint_step}"
            )

    salda_pre_endpoint_closure = None
    salda_best_checkpoint_closure = None
    salda_terminal_optimizer_step = None
    if salda_ga_active:
        if salda_initial_optimizer_step is None:
            raise RuntimeError("SalDA initial optimizer step was not recorded")
        salda_terminal_state = (
            unreplicate_state(state) if args.distributed else state
        )
        salda_terminal_optimizer_step = int(
            np.asarray(jax.device_get(salda_terminal_state.step)).item()
        )
        salda_pre_endpoint_closure = _validate_salda_pre_endpoint_workload(
            stop_epoch=args.salda_ga_stop_epoch,
            epoch_records=salda_epoch_records,
            train_batches_per_epoch=logged_steps_per_epoch,
            initial_optimizer_step=salda_initial_optimizer_step,
            terminal_optimizer_step=salda_terminal_optimizer_step,
            endpoint_builder_calls=endpoint_builder_calls,
            endpoint_evaluations=endpoint_evaluations,
            dataset_protocol=salda_dataset_protocol,
        )

    # #### SALDA BEST-VDEV RESTORE: START ####
    # Hash and restore the strictly Vdev-selected checkpoint before any Vtest
    # dataset exists; the checkpoint step is then checked against its epoch.
    salda_restored_best = None
    salda_best_checkpoint_step = None
    salda_best_vdev_checkpoint = None
    if (
        salda_ga_active
        and args.final_test
        and args.final_test_checkpoint == "best"
        and formal_run_context is None
    ):
        if checkpoint_path is None:
            raise RuntimeError("SalDA final Vtest requires a saved best checkpoint")
        best_checkpoint_path = (checkpoint_path / "best").resolve()
        best_checkpoint_directory_sha256 = _salda_directory_sha256(
            best_checkpoint_path
        )
        host_template = unreplicate_state(state) if args.distributed else state
        salda_restored_best = restore_checkpoint(
            host_template,
            str(best_checkpoint_path),
        )
        salda_best_checkpoint_step = int(
            np.asarray(jax.device_get(salda_restored_best.step)).item()
        )
        salda_best_checkpoint_closure = (
            _validate_salda_best_checkpoint_pre_endpoint(
                pre_endpoint_closure=salda_pre_endpoint_closure,
                best_epoch=best_epoch,
                best_checkpoint_optimizer_step=salda_best_checkpoint_step,
                best_checkpoint_path=str(best_checkpoint_path),
                best_checkpoint_directory_sha256=(
                    best_checkpoint_directory_sha256
                ),
            )
        )
        salda_best_vdev_checkpoint = {
            "path": str(best_checkpoint_path),
            "epoch": best_epoch,
            "top1_error": best_top1_error,
            "selection_rule": STRICT_VDEV_TOP1_ERROR_RULE,
            "directory_sha256": best_checkpoint_directory_sha256,
        }
    # #### SALDA BEST-VDEV RESTORE: END ####

    if args.final_test and final_test_ds is None:
        if salda_ga_active and (
            salda_pre_endpoint_closure is None
            or salda_pre_endpoint_closure.get("passed") is not True
            or salda_best_checkpoint_closure is None
            or salda_best_checkpoint_closure.get("passed") is not True
        ):
            raise RuntimeError(
                "SalDA endpoint cannot be built before workload and best-checkpoint "
                "closure"
            )
        if method_name in precomputed_saliency_methods and salda_ga_active:
            raise RuntimeError("SalDA supports only baseline or MixUp training data")
        if salda_ga_active:
            # #### SALDA SEALED VTEST DATASET OPEN: START ####
            final_test_ds = _build_salda_endpoint_after_closure(
                builder=build_final_test_dataset,
                pre_endpoint_closure=salda_pre_endpoint_closure,
                best_checkpoint_closure=salda_best_checkpoint_closure,
                builder_kwargs={
                    "args": args,
                    "method_name": method_name,
                    "precomputed_saliency_methods": precomputed_saliency_methods,
                },
            )
            # #### SALDA SEALED VTEST DATASET OPEN: END ####
        else:
            final_test_ds = build_final_test_dataset(
                args=args,
                method_name=method_name,
                precomputed_saliency_methods=precomputed_saliency_methods,
            )
        endpoint_builder_calls += 1

    if args.final_test and final_test_ds is not None:
        final_test_state = state
        if args.final_test_checkpoint == "best":
            if formal_run_context is not None:
                if formal_restored_best is None:
                    raise RuntimeError("formal best checkpoint was not restored")
                final_test_state = (
                    replicate_state(formal_restored_best)
                    if args.distributed
                    else formal_restored_best
                )
            elif salda_ga_active:
                if salda_restored_best is None:
                    raise RuntimeError("SalDA best checkpoint was not restored")
                final_test_state = (
                    replicate_state(salda_restored_best)
                    if args.distributed
                    else salda_restored_best
                )
            else:
                if best_eval_state is None:
                    raise RuntimeError(
                        "final_test_checkpoint='best' requires at least one "
                        "completed evaluation epoch.",
                    )
                final_test_state = best_eval_state
            print(f"Final test checkpoint: best {eval_name} epoch {best_epoch}")

        # #### SHARED FINAL-TEST EVALUATION: START ####
        if args.distributed:
            (
                test_loss,
                test_top1_accuracy,
                test_top5_accuracy,
                test_top1_error,
                test_top5_error,
                endpoint_batches,
                endpoint_examples,
            ) = parallel_evaluate(
                state=final_test_state,
                test_ds=final_test_ds,
                num_classes=metadata.num_classes,
                max_eval_steps=args.max_eval_steps,
                return_counts=True,
            )

        else:
            (
                test_loss,
                test_top1_accuracy,
                test_top5_accuracy,
                test_top1_error,
                test_top5_error,
                endpoint_batches,
                endpoint_examples,
            ) = evaluate(
                state=final_test_state,
                test_ds=final_test_ds,
                num_classes=metadata.num_classes,
                max_eval_steps=args.max_eval_steps,
                return_counts=True,
            )
        endpoint_evaluations += 1
        # #### SHARED FINAL-TEST EVALUATION: END ####

        print(
            format_final_test_message(
                test_loss=test_loss,
                test_top1_accuracy=test_top1_accuracy,
                test_top5_accuracy=test_top5_accuracy,
                test_top1_error=test_top1_error,
                test_top5_error=test_top5_error,
            )
        )
        final_test_result = {
            "loss": test_loss,
            "top1_accuracy": test_top1_accuracy,
            "top5_accuracy": test_top5_accuracy,
            "top1_error": test_top1_error,
            "top5_error": test_top5_error,
        }

        wandb_run.log_final_test(
            {
                "loss": test_loss,
                "top1_accuracy": test_top1_accuracy,
                "top5_accuracy": test_top5_accuracy,
                "top1_error": test_top1_error,
                "top5_error": test_top5_error,
            },
        )

        if args.save_csv and output_path is not None:
            write_final_test_result(
                output_path=output_path,
                test_loss=test_loss,
                test_top1_accuracy=test_top1_accuracy,
                test_top5_accuracy=test_top5_accuracy,
                test_top1_error=test_top1_error,
                test_top5_error=test_top5_error,
            )

    salda_vtest_builder_calls = endpoint_builder_calls
    salda_vtest_batches = endpoint_batches
    salda_final_test_result = final_test_result if salda_ga_active else None

    if metaaugment_context is not None and output_path is not None:
        metaaugment_context.save_sampler_probs(
            output_path.with_name(f"{output_path.stem}_sampler_probs.npy")
        )

    wandb_run.finish()
    salda_wandb_closure = wandb_run.closure_metadata() if salda_ga_active else None
    if salda_ga_active and (
        not salda_wandb_closure["enabled"]
        or salda_wandb_closure["mode"] != "online"
        or not salda_wandb_closure["run_id"]
        or not salda_wandb_closure["url"]
        or not salda_wandb_closure["finish_completed"]
    ):
        raise RuntimeError(
            "SalDA W&B closure is incomplete: "
            + json.dumps(salda_wandb_closure, sort_keys=True)
        )

    if formal_run_context is not None:
        expected_workload = formal_run_context.protocol["expected_workload"]
        endpoint_observed = {
            "endpoint_builder_calls": endpoint_builder_calls,
            "endpoint_evaluations": endpoint_evaluations,
            "endpoint_batches": int(endpoint_batches),
            "endpoint_examples": int(endpoint_examples),
        }
        for name, observed in endpoint_observed.items():
            expected = expected_workload[name]
            if observed != expected:
                raise RuntimeError(
                    f"formal endpoint closure failed for {name}: "
                    f"observed={observed}, expected={expected}"
                )
        if final_test_result is None or not all(
            np.isfinite(float(value)) for value in final_test_result.values()
        ):
            raise RuntimeError("formal endpoint metrics are missing or non-finite")
        wandb_closure = wandb_run.closure_metadata()
        if (
            not wandb_closure["enabled"]
            or wandb_closure["mode"] != "online"
            or not wandb_closure["run_id"]
            or not wandb_closure["url"]
            or not wandb_closure["finish_completed"]
        ):
            raise RuntimeError(
                "formal W&B closure is incomplete: "
                + json.dumps(wandb_closure, sort_keys=True)
            )
        if formal_pre_endpoint_closure is None:
            raise RuntimeError("formal pre-endpoint closure is missing")
        if output_path is None or checkpoint_path is None:
            raise RuntimeError("formal output or checkpoint path is missing")
        final_csv_path = output_path.with_name(
            f"{output_path.stem}_final_test{output_path.suffix}"
        ).resolve()
        completion_payload = {
            "schema_version": 1,
            "status": "SUCCESS",
            "protocol_id": formal_run_context.protocol["protocol_id"],
            "protocol_artifact": str(formal_run_context.protocol_path),
            "protocol_artifact_sha256": formal_run_context.protocol_sha256,
            "resolved_config": str(formal_run_context.resolved_config_path),
            "resolved_config_sha256": (formal_run_context.resolved_config_sha256),
            "inputs": formal_run_context.protocol["inputs"],
            "split": formal_run_context.protocol["split"],
            "workload": formal_pre_endpoint_closure,
            "terminal_optimizer_step": formal_pre_endpoint_closure["optimizer_steps"],
            "best_checkpoint_optimizer_step": formal_best_checkpoint_step,
            "endpoint": {
                **endpoint_observed,
                "result": final_test_result,
                "role": "sealed_official_test_complement",
                "built_after_training_and_checkpoint_closure": True,
            },
            "selection": {
                "rule": STRICT_VDEV_TOP1_ERROR_RULE,
                "best_epoch": best_epoch,
                "best_top1_error": best_top1_error,
                "selected_checkpoint": formal_pre_endpoint_closure[
                    "selected_checkpoint"
                ],
                "selected_checkpoint_sha256": formal_pre_endpoint_closure[
                    "selected_checkpoint_sha256"
                ],
            },
            "artifacts": {
                "metrics_csv": str(output_path.resolve()),
                "final_test_csv": str(final_csv_path),
            },
            "wandb": wandb_closure,
            "epochs": formal_epoch_records,
        }
        completion_payload["payload_sha256"] = hashlib.sha256(
            json.dumps(
                completion_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        ).hexdigest()
        completion_path = write_json_atomic(
            formal_run_context.protocol["artifacts"]["completion"],
            completion_payload,
        )
        print(f"Formal run completion artifact: {completion_path}")

    if salda_ga_active:
        if salda_pre_endpoint_closure is None:
            raise RuntimeError("SalDA pre-endpoint workload closure is missing")
        expected_updates = int(salda_pre_endpoint_closure["train_updates"])
        observed_vdev_batches = int(salda_pre_endpoint_closure["vdev_batches"])
        observed_vdev_examples = int(salda_pre_endpoint_closure["vdev_examples"])
        registered_timing_epochs = resolve_registered_timing_epochs(
            stop_epoch=args.salda_ga_stop_epoch,
            protocol=salda_dataset_protocol,
        )
        timing_epoch_end = registered_timing_epochs or 10
        timing_summary = _summarize_salda_epoch_timing(
            salda_epoch_records,
            stable_epoch_end=timing_epoch_end,
        )
        stable_wall = timing_summary["components"]["end_to_end_wall"]
        execution = (
            salda_strategy.execution_summary()
            if salda_strategy is not None
            else {
                "action_enabled": False,
                "parameter_scope": "none",
                "base_method": method_name,
                "shuffled_control": False,
                "validation_direction_mode": salda_validation_direction[
                    "validation_direction_mode"
                ],
                "validation_pool_examples": (
                    salda_dataset_protocol.validation_examples
                ),
                "validation_examples_per_gradient_evaluation": (
                    salda_validation_direction[
                        "validation_examples_per_gradient_evaluation"
                    ]
                ),
                "validation_direction_cycle_length": (
                    salda_validation_direction["validation_direction_cycle_length"]
                ),
                "validation_reanchor_interval": salda_validation_direction[
                    "validation_reanchor_interval"
                ],
                "validation_batch_seed": None,
                "validation_initial_optimizer_step": None,
                "validation_batch_schedule_sha256": None,
                "train_steps": expected_updates,
                "scored_steps": 0,
                "action_active_steps": 0,
                "score_start_optimizer_step": None,
                "score_stop_optimizer_step": None,
                "action_start_optimizer_step": None,
                "action_stop_optimizer_step": None,
                "direction_refreshes": 0,
                "validation_gradient_evaluations": 0,
                "validation_exact_reanchors": 0,
                "validation_anchor_drift_comparisons": 0,
                "validation_anchor_stale_to_exact_cosine_mean": None,
                "validation_anchor_stale_to_exact_cosine_min": None,
                "validation_anchor_stale_to_exact_relative_l2_mean": None,
                "validation_anchor_stale_to_exact_relative_l2_max": None,
                "direction_validation_example_visits": 0,
            }
        )
        if execution["train_steps"] != expected_updates:
            raise RuntimeError(
                "SalDA update closure failed: "
                f"observed {execution['train_steps']}, expected {expected_updates}"
            )
        score_start_step = args.salda_ga_score_start_epoch * steps_per_epoch
        score_stop_step = (
            salda_terminal_optimizer_step
            if args.salda_ga_score_stop_epoch == -1
            else args.salda_ga_score_stop_epoch * steps_per_epoch
        )
        expected_directions = (
            0
            if args.salda_ga_mode in {"baseline", "noop"}
            else max(
                0,
                min(salda_terminal_optimizer_step, score_stop_step)
                - max(salda_initial_optimizer_step, score_start_step),
            )
        )
        if execution.get("scored_steps", expected_directions) != expected_directions:
            raise RuntimeError(
                "SalDA scored-step closure failed: "
                f"observed {execution.get('scored_steps')}, "
                f"expected {expected_directions}"
            )
        action_start_step = args.salda_ga_action_start_epoch * steps_per_epoch
        action_stop_step = (
            salda_terminal_optimizer_step
            if args.salda_ga_action_stop_epoch == -1
            else args.salda_ga_action_stop_epoch * steps_per_epoch
        )
        expected_action_active_steps = (
            max(
                0,
                min(
                    salda_terminal_optimizer_step,
                    action_stop_step,
                    score_stop_step,
                )
                - max(salda_initial_optimizer_step, action_start_step),
            )
            if args.salda_ga_mode
            in {
                "soft_label",
                "reweight",
                "shuffled_soft_label",
                "shuffled_reweight",
            }
            else 0
        )
        if (
            execution.get("action_active_steps", expected_action_active_steps)
            != expected_action_active_steps
        ):
            raise RuntimeError(
                "SalDA action-phase closure failed: "
                f"observed {execution.get('action_active_steps')}, "
                f"expected {expected_action_active_steps}"
            )
        if execution["direction_refreshes"] != expected_directions:
            raise RuntimeError(
                "SalDA direction closure failed: "
                f"observed {execution['direction_refreshes']}, "
                f"expected {expected_directions}"
            )
        expected_direction_visits = _expected_salda_direction_example_visits(
            mode=args.salda_ga_validation_direction_mode,
            updates=expected_directions,
            validation_batch_size=args.salda_ga_validation_batch_size,
            reanchor_interval=args.salda_ga_validation_reanchor_interval,
            initial_optimizer_step=max(
                salda_initial_optimizer_step,
                score_start_step,
            ),
            validation_pool_examples=(salda_dataset_protocol.validation_examples),
        )
        (
            expected_gradient_evaluations,
            expected_exact_reanchors,
        ) = _expected_salda_direction_counts(
            mode=args.salda_ga_validation_direction_mode,
            updates=expected_directions,
            validation_batch_size=args.salda_ga_validation_batch_size,
            reanchor_interval=args.salda_ga_validation_reanchor_interval,
            initial_optimizer_step=max(
                salda_initial_optimizer_step,
                score_start_step,
            ),
            validation_pool_examples=(salda_dataset_protocol.validation_examples),
        )
        if (
            execution["validation_gradient_evaluations"]
            != expected_gradient_evaluations
            or execution["validation_exact_reanchors"] != expected_exact_reanchors
            or execution["validation_anchor_drift_comparisons"]
            != max(expected_exact_reanchors - 1, 0)
        ):
            raise RuntimeError(
                "SalDA validation-gradient count closure failed: "
                f"evaluations={execution['validation_gradient_evaluations']} "
                f"expected={expected_gradient_evaluations}, "
                f"reanchors={execution['validation_exact_reanchors']} "
                f"expected_reanchors={expected_exact_reanchors}, "
                f"drift_comparisons="
                f"{execution['validation_anchor_drift_comparisons']} "
                f"expected_drift_comparisons="
                f"{max(expected_exact_reanchors - 1, 0)}"
            )
        if (
            execution["direction_validation_example_visits"]
            != expected_direction_visits
        ):
            raise RuntimeError(
                "SalDA validation-direction visit closure failed: "
                f"observed "
                f"{execution['direction_validation_example_visits']}, "
                f"expected {expected_direction_visits}"
            )
        workload_closure = _validate_salda_completion_workload(
            stop_epoch=args.salda_ga_stop_epoch,
            final_test_enabled=args.final_test,
            completed_epochs=len(salda_epoch_records),
            steps_per_epoch=logged_steps_per_epoch,
            train_updates=expected_updates,
            vdev_evaluations=len(salda_epoch_records),
            vdev_batches=observed_vdev_batches,
            endpoint_builder_calls=salda_vtest_builder_calls,
            endpoint_evaluations=endpoint_evaluations,
            vtest_batches=salda_vtest_batches,
            vtest_examples=endpoint_examples,
            vtest_result=salda_final_test_result,
            dataset_protocol=salda_dataset_protocol,
        )
        complete_timing_workload = bool(
            registered_timing_epochs is not None
            and not args.final_test
            and workload_closure["passed"]
        )
        best_vdev_checkpoint = None
        if salda_best_vdev_checkpoint is not None:
            best_vdev_checkpoint = salda_best_vdev_checkpoint
        elif (
            args.save_checkpoint
            and args.save_best_only
            and checkpoint_path is not None
        ):
            best_checkpoint_path = (checkpoint_path / "best").resolve()
            if not best_checkpoint_path.is_dir():
                raise RuntimeError(
                    f"best-Vdev checkpoint is missing: {best_checkpoint_path}"
                )
            best_vdev_checkpoint = {
                "path": str(best_checkpoint_path),
                "epoch": best_epoch,
                "top1_error": best_top1_error,
                "selection_rule": STRICT_VDEV_TOP1_ERROR_RULE,
                "directory_sha256": _salda_directory_sha256(best_checkpoint_path),
            }
        if args.final_test and best_vdev_checkpoint is None:
            raise RuntimeError(
                "SalDA final Vtest requires a hashed best-Vdev checkpoint"
            )
        completion_payload = {
            "schema_version": 1,
            "status": "SUCCESS",
            "dataset": salda_dataset_protocol.dataset,
            "git_commit": args.salda_ga_git_commit,
            "resolved_config_sha256": salda_config_sha256,
            "runtime_config_sha256": salda_runtime_config_sha256,
            "training_recipe_sha256": salda_training_recipe_sha256,
            "data_protocol_sha256": salda_data_protocol_sha256,
            "protocol_artifact": str(salda_protocol_path),
            "protocol_artifact_sha256": hashlib.sha256(
                salda_protocol_path.read_bytes()
            ).hexdigest(),
            "completed_epochs": len(salda_epoch_records),
            "optimizer_horizon_epochs": args.epochs,
            "policy_mode": args.salda_ga_mode,
            "seed": args.seed,
            "resolved_data_seed": args.data_seed,
            "data_seed_policy": "resolved_from_training_seed",
            "method": method_name,
            "training_data": (
                "original_images" if method_name == "baseline" else "online_mixup"
            ),
            "parameter_scope": args.salda_ga_parameter_scope,
            "validation_direction_mode": salda_validation_direction[
                "validation_direction_mode"
            ],
            "validation_examples_per_gradient_evaluation": (
                salda_validation_direction[
                    "validation_examples_per_gradient_evaluation"
                ]
            ),
            "validation_direction_cycle_length": (
                salda_validation_direction["validation_direction_cycle_length"]
            ),
            "validation_reanchor_interval": salda_validation_direction[
                "validation_reanchor_interval"
            ],
            "validation_direction_main_table_eligible": (
                salda_validation_direction["validation_direction_main_table_eligible"]
            ),
            "validation_pool_sha256": salda_validation_fingerprint,
            "validation_batch_schedule_sha256": execution[
                "validation_batch_schedule_sha256"
            ],
            "train_updates": expected_updates,
            "vdev_evaluations": len(salda_epoch_records),
            "vdev_batches": observed_vdev_batches,
            "vdev_example_visits_for_epoch_readout": observed_vdev_examples,
            "vtest_loaded": bool(salda_vtest_builder_calls),
            "vtest_batches": salda_vtest_batches,
            "vtest_result": salda_final_test_result,
            "vdev_role": "training_direction_and_checkpoint_selection",
            "vtest_role": "sealed",
            "endpoint_builder_calls": salda_vtest_builder_calls,
            "endpoint_evaluations": endpoint_evaluations,
            "vtest_examples": endpoint_examples,
            "best_vdev_epoch": best_epoch,
            "best_vdev_top1_error": best_top1_error,
            "checkpoint_selection_rule": STRICT_VDEV_TOP1_ERROR_RULE,
            "best_vdev_checkpoint": best_vdev_checkpoint,
            "best_checkpoint_optimizer_step": salda_best_checkpoint_step,
            "initial_optimizer_step": salda_initial_optimizer_step,
            "terminal_optimizer_step": salda_terminal_optimizer_step,
            "pre_endpoint_workload_closure": salda_pre_endpoint_closure,
            "best_checkpoint_pre_endpoint_closure": (
                salda_best_checkpoint_closure
            ),
            **(
                {"direction_prerequisite": salda_direction_prerequisite}
                if salda_dataset_protocol.dataset == "stl10"
                else {}
            ),
            "endpoint_built_after_best_checkpoint_restore": (
                True if args.final_test and salda_restored_best is not None else None
            ),
            "wandb": salda_wandb_closure,
            "origin_mixup_contract": _salda_origin_mixup_contract(method_name),
            "execution": execution,
            "action_summary": _summarize_salda_actions(
                salda_epoch_records,
                steps_per_epoch=logged_steps_per_epoch,
                global_batch_size=args.batch_size,
                strategy_active=args.salda_ga_mode not in {"baseline", "noop"},
            ),
            "component_timing": (
                salda_strategy.timing_summary() if salda_strategy is not None else {}
            ),
            "epoch_timing_summary": timing_summary,
            "timing_profile_synchronizes_component_boundaries": bool(
                args.salda_ga_profile_components
            ),
            "component_timing_measurement_mode": (
                "synchronized_component_profile"
                if args.salda_ga_profile_components
                else "raw_end_to_end_without_internal_boundary_sync"
            ),
            "update_timing_includes_standard_base_method": bool(
                args.salda_ga_mode in {"baseline", "noop"}
                and not args.salda_ga_profile_components
            ),
            "workload_closure": workload_closure,
            "timing_workload_closure": (
                _salda_timing_workload_closure_payload(
                    stop_epoch=args.salda_ga_stop_epoch,
                    workload_closure=workload_closure,
                    observed_epoch_rows=len(salda_epoch_records),
                    observed_train_updates=expected_updates,
                    observed_vdev_batches=observed_vdev_batches,
                    observed_vtest_batches=salda_vtest_batches,
                    dataset_protocol=salda_dataset_protocol,
                )
                if not args.final_test
                else None
            ),
            "timing_target": (
                _salda_timing_target_payload(
                    dataset_protocol=salda_dataset_protocol,
                    complete_timing_workload=complete_timing_workload,
                    stable_wall=stable_wall,
                )
                if not args.final_test
                else {
                    "registered": False,
                    "median_seconds_at_most": None,
                    "p90_seconds_at_most": None,
                    "observed_stable_median_seconds": stable_wall["median"],
                    "observed_stable_p90_seconds": stable_wall["p90"],
                    "passed": None,
                    "reason": "endpoint_workload_not_timing",
                }
            ),
            "epochs": salda_epoch_records,
        }
        _validate_salda_completion_schema(
            completion_payload,
            dataset_protocol=salda_dataset_protocol,
        )
        completion_payload["completion_sha256"] = hashlib.sha256(
            json.dumps(
                completion_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        completion_path = _write_salda_json(
            args.output_dir,
            run_name,
            "salda_training_complete",
            completion_payload,
        )
        print(f"SalDA completion artifact: {completion_path}")


if __name__ == "__main__":
    main()
