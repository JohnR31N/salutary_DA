"""Registered dataset contracts for instantaneous gradient alignment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class InstantaneousGADatasetProtocol:
    """Immutable train, validation, and endpoint counts for one dataset."""

    dataset: str
    num_classes: int
    train_examples: int
    validation_examples: int
    validation_examples_per_class: int
    endpoint_examples: int
    global_batch_size: int
    steps_per_epoch: int
    validation_batches_per_epoch: int
    endpoint_batches: int
    validation_split: float
    validation_batch_size: int
    validation_reanchor_interval: int
    weight_decay: float
    mixup_alpha: float
    allow_batch_aggregate: bool
    bounded_timing_epochs: tuple[int, ...]
    timing_median_seconds_at_most: float | None
    timing_p90_seconds_at_most: float | None

    @property
    def validation_class_counts(self) -> tuple[int, ...]:
        """Return the exact balanced Vdev class-count vector."""

        return (self.validation_examples_per_class,) * self.num_classes


CIFAR100_INSTANTANEOUS_GA_PROTOCOL = InstantaneousGADatasetProtocol(
    dataset="cifar100",
    num_classes=100,
    train_examples=50_000,
    validation_examples=5_000,
    validation_examples_per_class=50,
    endpoint_examples=5_000,
    global_batch_size=128,
    steps_per_epoch=390,
    validation_batches_per_epoch=40,
    endpoint_batches=40,
    validation_split=0.5,
    validation_batch_size=500,
    validation_reanchor_interval=50,
    weight_decay=0.0001,
    mixup_alpha=0.2,
    allow_batch_aggregate=True,
    bounded_timing_epochs=(10,),
    timing_median_seconds_at_most=12.0,
    timing_p90_seconds_at_most=12.5,
)

STL10_INSTANTANEOUS_GA_PROTOCOL = InstantaneousGADatasetProtocol(
    dataset="stl10",
    num_classes=10,
    train_examples=5_000,
    validation_examples=4_000,
    validation_examples_per_class=400,
    endpoint_examples=4_000,
    global_batch_size=128,
    steps_per_epoch=39,
    validation_batches_per_epoch=32,
    endpoint_batches=32,
    validation_split=0.5,
    validation_batch_size=400,
    validation_reanchor_interval=50,
    weight_decay=0.0005,
    mixup_alpha=1.0,
    allow_batch_aggregate=False,
    bounded_timing_epochs=(10, 20, 30),
    timing_median_seconds_at_most=None,
    timing_p90_seconds_at_most=None,
)

INSTANTANEOUS_GA_DATASET_PROTOCOLS = {
    protocol.dataset: protocol
    for protocol in (
        CIFAR100_INSTANTANEOUS_GA_PROTOCOL,
        STL10_INSTANTANEOUS_GA_PROTOCOL,
    )
}


def resolve_registered_timing_epochs(
    *,
    stop_epoch: int,
    protocol: InstantaneousGADatasetProtocol,
) -> int | None:
    """Return the explicitly registered bounded timing length, if any."""

    return stop_epoch if stop_epoch in protocol.bounded_timing_epochs else None


def canonical_protocol_sha256(value: Mapping[str, object]) -> str:
    """Hash one protocol mapping with the trainer JSON convention."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_validation_direction_config(
    config: Mapping[str, object],
    *,
    protocol: InstantaneousGADatasetProtocol,
) -> dict[str, object]:
    """Build the registered validation-direction layout from exact keys."""

    mode = config["salda_ga_validation_direction_mode"]
    batch_size = config["salda_ga_validation_batch_size"]
    if mode not in {"full", "batch_aggregate"}:
        raise ValueError(
            "SalDA validation direction mode must be full or batch_aggregate"
        )
    if batch_size != protocol.validation_batch_size:
        raise ValueError(
            "SalDA validation batch size must be exactly "
            f"{protocol.validation_batch_size} for {protocol.dataset}"
        )
    if mode == "batch_aggregate" and not protocol.allow_batch_aggregate:
        raise ValueError(
            f"batch-aggregate SalDA is not registered for {protocol.dataset}"
        )
    validation_examples = protocol.validation_examples
    examples_per_gradient = (
        validation_examples if mode == "full" else batch_size
    )
    return {
        "validation_direction_mode": mode,
        "validation_direction_layout": (
            "single_complete_batch"
            if mode == "full"
            else "class_balanced_cached_gradient_mean"
        ),
        "validation_direction_pool_examples": validation_examples,
        "validation_examples_per_gradient_evaluation": examples_per_gradient,
        "validation_examples_per_device": examples_per_gradient // 4,
        "validation_direction_cycle_length": (
            validation_examples // examples_per_gradient
        ),
        "validation_direction_schedule": (
            "constant_full_pool"
            if mode == "full"
            else "cyclic_component_replacement_with_exact_reanchor"
        ),
        "validation_direction_main_table_eligible": mode == "full",
        "validation_batch_size": None if mode == "full" else batch_size,
        "validation_batch_seed_policy": (
            None if mode == "full" else "training_seed"
        ),
        "validation_reanchor_interval": (
            None
            if mode == "full"
            else config["salda_ga_validation_reanchor_interval"]
        ),
        "validation_reanchor_examples": (
            None if mode == "full" else validation_examples
        ),
    }


def build_runtime_config(
    config: Mapping[str, object],
    *,
    method_name: str,
    protocol: InstantaneousGADatasetProtocol,
) -> dict[str, object]:
    """Build the exact non-circular instantaneous-GA runtime payload."""

    if method_name not in {"baseline", "mixup"}:
        raise ValueError("SalDA runtime config requires baseline or mixup")
    policy_mode = {
        "shuffled_soft_label": "soft_label",
        "shuffled_reweight": "reweight",
    }.get(config["salda_ga_mode"], config["salda_ga_mode"])
    return {
        "schema_version": 1,
        "dataset": config["dataset"],
        "model": config["model"],
        "model_configuration": {
            "resnet_stem_type": config["resnet_stem_type"],
            "preact_stem_bn_relu": config["preact_stem_bn_relu"],
            "preact_pytorch_default_init": (
                config["preact_pytorch_default_init"]
            ),
        },
        "base_method": method_name,
        "training_data": (
            "original_images" if method_name == "baseline" else "online_mixup"
        ),
        "registered_training_seeds": [0, 1, 2],
        "data_seed_policy": "resolved_from_training_seed",
        "global_batch_size": config["batch_size"],
        "optimizer_horizon_epochs": config["epochs"],
        "optimizer": {
            "learning_rate": config["learning_rate"],
            "momentum": config["momentum"],
            "nesterov": config["nesterov"],
            "weight_decay": config["weight_decay"],
            "lr_schedule": config["lr_schedule"],
            "minimum_learning_rate": config["min_learning_rate"],
            "warmup_epochs": config["warmup_epochs"],
            "decay_epochs": list(config["lr_decay_epochs"]),
            "decay_rate": config["lr_decay_rate"],
        },
        "augmentation": {
            "recipe": config["aug_recipe"],
            "deterministic_data": config["deterministic_data"],
            "strict_determinism": config["strict_determinism"],
            "shuffle_buffer_size": config["shuffle_buffer_size"],
            "cross_device_shuffle": config["cross_device_shuffle"],
            "mixup_alpha": (
                config["mixup_alpha"] if method_name == "mixup" else None
            ),
        },
        "distributed": config["distributed"],
        "sync_batch_stats": config["sync_batch_stats"],
        "validation": {
            "source": config["val_source"],
            "split": config["validation_split"],
            "examples": protocol.validation_examples,
            "epoch_batches": protocol.validation_batches_per_epoch,
            "direction_refresh_optimizer_steps": 1,
        },
        "gradient_alignment": {
            "parameter_scope": config["salda_ga_parameter_scope"],
            **build_validation_direction_config(config, protocol=protocol),
            "policy_mode": policy_mode,
            "score_start_epoch": config.get("salda_ga_score_start_epoch", 0),
            "score_stop_epoch": config.get("salda_ga_score_stop_epoch", -1),
            "action_start_epoch": config.get("salda_ga_action_start_epoch", 0),
            "action_stop_epoch": config.get("salda_ga_action_stop_epoch", -1),
            "maximum_rows": config["salda_ga_maximum_rows"],
            "soft_label_dose": config["salda_ga_soft_label_dose"],
            "maximum_weight_deviation": config[
                "salda_ga_max_weight_deviation"
            ],
            "weight_temperature": config["salda_ga_weight_temperature"],
            "minimum_relative_ess": config["salda_ga_minimum_relative_ess"],
            "minimum_gain": config["salda_ga_minimum_gain"],
            "minimum_label_margin": config["salda_ga_minimum_label_margin"],
            "minimum_relative_label_margin": config[
                "salda_ga_minimum_relative_label_margin"
            ],
            "fallback_enabled": config["salda_ga_fallback_enabled"],
            "fallback_soft_label_dose": config[
                "salda_ga_fallback_soft_label_dose"
            ],
        },
    }


def build_training_recipe(
    runtime_config: Mapping[str, object],
) -> dict[str, object]:
    """Remove only GA-policy fields from a sealed runtime payload."""

    required = {
        "schema_version",
        "dataset",
        "model",
        "model_configuration",
        "base_method",
        "training_data",
        "registered_training_seeds",
        "data_seed_policy",
        "global_batch_size",
        "optimizer_horizon_epochs",
        "optimizer",
        "augmentation",
        "distributed",
        "sync_batch_stats",
        "validation",
        "gradient_alignment",
    }
    if not isinstance(runtime_config, Mapping) or set(runtime_config) != required:
        raise ValueError("SalDA runtime config fields changed")
    validation = runtime_config["validation"]
    if not isinstance(validation, Mapping) or set(validation) != {
        "source",
        "split",
        "examples",
        "epoch_batches",
        "direction_refresh_optimizer_steps",
    }:
        raise ValueError("SalDA validation config fields changed")
    recipe = {
        key: runtime_config[key]
        for key in (
            "schema_version",
            "dataset",
            "model",
            "model_configuration",
            "base_method",
            "training_data",
            "registered_training_seeds",
            "data_seed_policy",
            "global_batch_size",
            "optimizer_horizon_epochs",
            "optimizer",
            "augmentation",
            "distributed",
            "sync_batch_stats",
        )
    }
    recipe["validation"] = {
        key: validation[key]
        for key in ("source", "split", "examples", "epoch_batches")
    }
    return json.loads(json.dumps(recipe, sort_keys=True, allow_nan=False))


def build_data_protocol(
    *,
    method_name: str,
    validation_fingerprint: str,
    protocol: InstantaneousGADatasetProtocol,
) -> dict[str, object]:
    """Build the registered train and validation data payload."""

    if method_name not in {"baseline", "mixup"}:
        raise ValueError("SalDA data protocol requires baseline or mixup")
    if (
        not isinstance(validation_fingerprint, str)
        or len(validation_fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in validation_fingerprint
        )
    ):
        raise ValueError("SalDA validation fingerprint is invalid")
    return {
        "dataset": protocol.dataset,
        "method": method_name,
        "training_data": (
            "original_images" if method_name == "baseline" else "online_mixup"
        ),
        "train_examples": protocol.train_examples,
        "validation_examples": protocol.validation_examples,
        "validation_class_counts": list(protocol.validation_class_counts),
        "validation_fingerprint": validation_fingerprint,
        "global_batch_size": protocol.global_batch_size,
        "steps_per_epoch": protocol.steps_per_epoch,
        "val_source": "test",
        "validation_split": protocol.validation_split,
        "vtest_role": "sealed",
    }


def build_run_protocol_data_fields(
    data_protocol: Mapping[str, object],
) -> dict[str, object]:
    """Select data fields serialized at the run-protocol top level."""

    required = {
        "dataset",
        "method",
        "training_data",
        "train_examples",
        "validation_examples",
        "validation_class_counts",
        "validation_fingerprint",
        "global_batch_size",
        "steps_per_epoch",
        "val_source",
        "validation_split",
        "vtest_role",
    }
    if not isinstance(data_protocol, Mapping) or set(data_protocol) != required:
        raise ValueError("SalDA data protocol fields changed")
    return {
        key: data_protocol[key]
        for key in (
            "dataset",
            "method",
            "training_data",
            "train_examples",
            "validation_examples",
            "validation_class_counts",
            "validation_fingerprint",
            "global_batch_size",
            "steps_per_epoch",
            "vtest_role",
        )
    }


def get_instantaneous_ga_dataset_protocol(
    dataset: str,
) -> InstantaneousGADatasetProtocol:
    """Return the exact registered GA protocol for ``dataset``."""

    if not isinstance(dataset, str):
        raise TypeError("instantaneous-GA dataset must be a string")
    try:
        return INSTANTANEOUS_GA_DATASET_PROTOCOLS[dataset]
    except KeyError as error:
        registered = ", ".join(INSTANTANEOUS_GA_DATASET_PROTOCOLS)
        raise ValueError(
            "instantaneous GA is registered only for: " + registered
        ) from error


__all__ = [
    "CIFAR100_INSTANTANEOUS_GA_PROTOCOL",
    "INSTANTANEOUS_GA_DATASET_PROTOCOLS",
    "STL10_INSTANTANEOUS_GA_PROTOCOL",
    "InstantaneousGADatasetProtocol",
    "build_data_protocol",
    "build_run_protocol_data_fields",
    "build_runtime_config",
    "build_training_recipe",
    "build_validation_direction_config",
    "canonical_protocol_sha256",
    "get_instantaneous_ga_dataset_protocol",
    "resolve_registered_timing_epochs",
]
