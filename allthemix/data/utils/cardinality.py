"""Resolve exact training cardinality for static and local datasets."""

from __future__ import annotations

from typing import Any

from allthemix.data.datasets.loader import get_runtime_train_class_counts
from allthemix.data.splits import (
    count_class_stratified_split_examples,
    count_train_subset_examples_per_class,
    validate_train_subset_fraction,
)


def resolve_train_class_counts(
    dataset_name: str,
    data_dir: str,
    metadata: Any,
) -> tuple[int, ...] | None:
    """Return runtime or metadata class counts in stable label order."""
    runtime_class_counts = get_runtime_train_class_counts(
        name=dataset_name,
        data_dir=data_dir,
    )
    if runtime_class_counts is not None:
        return runtime_class_counts

    metadata_class_counts = getattr(
        metadata,
        "train_class_counts",
        None,
    )
    if metadata_class_counts is None:
        return None

    return tuple(
        int(class_count)
        for class_count in metadata_class_counts
    )


def resolve_train_example_count(
    dataset_name: str,
    data_dir: str,
    metadata: Any,
    validation_split: float,
    train_subset_fraction: float = 1.0,
) -> int:
    """Return the exact number of examples consumed by one training epoch."""
    validate_train_subset_fraction(
        train_subset_fraction=train_subset_fraction,
    )
    class_counts = resolve_train_class_counts(
        dataset_name=dataset_name,
        data_dir=data_dir,
        metadata=metadata,
    )
    if class_counts is not None:
        # The subset is taken first; the validation split applies within it.
        class_counts = count_train_subset_examples_per_class(
            class_counts=class_counts,
            train_subset_fraction=train_subset_fraction,
        )

        return count_class_stratified_split_examples(
            class_counts=class_counts,
            validation_split=validation_split,
            keep_validation=False,
        )

    if train_subset_fraction < 1.0:
        raise ValueError(
            "train_subset_fraction < 1.0 requires exact per-class train "
            f"counts, which are unavailable for dataset {dataset_name!r}."
        )

    num_train_examples = int(
        metadata.num_train_examples,
    )
    if validation_split == 0.0:
        return num_train_examples

    return int(
        round(
            num_train_examples * (1.0 - validation_split),
        )
    )


def resolve_source_train_example_count(
    dataset_name: str,
    data_dir: str,
    metadata: Any,
) -> int:
    """Return the full source-training cardinality before validation split."""
    return resolve_train_example_count(
        dataset_name=dataset_name,
        data_dir=data_dir,
        metadata=metadata,
        validation_split=0.0,
    )


__all__ = [
    "resolve_source_train_example_count",
    "resolve_train_class_counts",
    "resolve_train_example_count",
]
