from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf

_SPLIT_BUCKET_COUNT = 10_000
_SPLIT_MULTIPLIER = 1_103_515_245
_SPLIT_OFFSET = 12_345
_EXACT_SPLIT_TOLERANCE = 1e-6
VALIDATION_SOURCES = ("train", "test")


def validate_validation_split(
    validation_split: float,
) -> None:
    """Validate a train/validation split fraction."""
    if validation_split < 0.0 or validation_split >= 1.0:
        raise ValueError(
            "validation_split must be in [0, 1). "
            f"Got {validation_split}."
        )


def validate_validation_source(
    val_source: str,
) -> None:
    """Validate the source used to construct the checkpoint-validation set."""
    if val_source not in VALIDATION_SOURCES:
        raise ValueError(
            "val_source must be one of "
            f"{VALIDATION_SOURCES}. Got {val_source!r}."
        )


def resolve_training_validation_split(
    validation_split: float,
    val_source: str,
) -> float:
    """Return the fraction removed from the official training split.

    ``validation_split`` always describes the selected validation source.
    When validation comes from the official evaluation split, no examples
    are removed from the official training split.
    """
    validate_validation_split(
        validation_split=validation_split,
    )
    validate_validation_source(
        val_source=val_source,
    )

    return 0.0 if val_source == "test" else validation_split


def _pack_element(
    element: tuple,
):
    """Return a single dataset element from possibly expanded tf.data args."""
    if len(
        element,
    ) == 1:
        return element[0]

    return element


def _label_from_element(
    element,
):
    """Return the class label from a supported raw dataset element."""
    if isinstance(
        element,
        dict,
    ) and "label" in element:
        return element["label"]

    if isinstance(
        element,
        (
            tuple,
            list,
        ),
    ):
        for value in element:
            label = _label_from_element(
                value,
            )

            if label is not None:
                return label

    return None


def _has_label(
    element_spec,
) -> bool:
    """Return whether a dataset element spec contains a label field."""
    return _label_from_element(
        element_spec,
    ) is not None


def _is_reciprocal_split(
    validation_split: float,
) -> tuple[bool, int]:
    """Return whether the split is close to 1 / k for integer k."""
    period = int(
        round(
            1.0 / validation_split,
        )
    )

    if period <= 1:
        return False, period

    is_reciprocal = abs(
        validation_split
        - 1.0 / period
    ) <= _EXACT_SPLIT_TOLERANCE

    return is_reciprocal, period


def validate_train_subset_fraction(
    train_subset_fraction: float,
) -> None:
    """Validate a deterministic stratified train-subset fraction."""
    if train_subset_fraction <= 0.0 or train_subset_fraction > 1.0:
        raise ValueError(
            "train_subset_fraction must be in (0, 1]. "
            f"Got {train_subset_fraction}."
        )

    if train_subset_fraction < 1.0:
        threshold = int(
            round(
                train_subset_fraction * _SPLIT_BUCKET_COUNT,
            )
        )

        # Keep the counting API aligned with the dataset filter, which
        # rejects thresholds that round to an empty or full bucket set.
        if threshold <= 0 or threshold >= _SPLIT_BUCKET_COUNT:
            raise ValueError(
                "train_subset_fraction is too close to 0 or 1 for "
                f"deterministic bucketing: {train_subset_fraction}."
            )


def _count_kept_class_examples(
    class_count: int,
    fraction: float,
) -> int:
    """Count class examples kept by the deterministic keep-fraction filter."""
    is_reciprocal, period = _is_reciprocal_split(
        fraction,
    )

    if is_reciprocal:
        # Index zero is kept, followed by every ``period``-th item.
        return (
            class_count + period - 1
        ) // period

    threshold = int(
        round(
            fraction * _SPLIT_BUCKET_COUNT,
        )
    )

    return sum(
        (
            class_index * _SPLIT_MULTIPLIER + _SPLIT_OFFSET
        )
        % _SPLIT_BUCKET_COUNT
        < threshold
        for class_index in range(
            class_count,
        )
    )


def count_train_subset_examples_per_class(
    class_counts: Sequence[int],
    train_subset_fraction: float,
) -> tuple[int, ...]:
    """Count per-class examples kept by the deterministic train subset."""
    validate_train_subset_fraction(
        train_subset_fraction=train_subset_fraction,
    )

    if train_subset_fraction >= 1.0:
        return tuple(
            int(class_count)
            for class_count in class_counts
        )

    return tuple(
        _count_kept_class_examples(
            class_count=int(class_count),
            fraction=train_subset_fraction,
        )
        for class_count in class_counts
    )


def subset_train_dataset(
    dataset: tf.data.Dataset,
    train_subset_fraction: float,
) -> tf.data.Dataset:
    """Keep a deterministic class-stratified fraction of a training stream.

    The subset is selected BEFORE any validation split, so a later
    validation_split is taken from within the reduced labeled budget. The
    kept membership reuses the split bucketing and is therefore fixed
    across experiment seeds.
    """
    validate_train_subset_fraction(
        train_subset_fraction=train_subset_fraction,
    )

    if train_subset_fraction >= 1.0:
        return dataset

    return split_train_validation_dataset(
        dataset=dataset,
        validation_split=train_subset_fraction,
        keep_validation=True,
    )


def count_class_stratified_split_examples(
    class_counts: Sequence[int],
    validation_split: float,
    keep_validation: bool = False,
) -> int:
    """Count one side of the deterministic class-stratified split exactly."""
    validate_validation_split(
        validation_split=validation_split,
    )
    normalized_counts = []
    for class_count in class_counts:
        if isinstance(
            class_count,
            bool,
        ) or not isinstance(
            class_count,
            int,
        ):
            raise TypeError(
                "class_counts must contain integers. "
                f"Got {class_count!r}."
            )
        if class_count < 0:
            raise ValueError(
                "class_counts must be nonnegative. "
                f"Got {class_count}."
            )
        normalized_counts.append(
            class_count,
        )

    total_examples = sum(
        normalized_counts,
    )
    if validation_split == 0.0:
        return 0 if keep_validation else total_examples

    threshold = int(
        round(
            validation_split * _SPLIT_BUCKET_COUNT,
        )
    )
    if threshold <= 0 or threshold >= _SPLIT_BUCKET_COUNT:
        raise ValueError(
            "validation_split is too close to 0 or 1 for deterministic "
            f"bucketing: {validation_split}."
        )

    is_reciprocal, period = _is_reciprocal_split(
        validation_split,
    )
    validation_examples = 0
    for class_count in normalized_counts:
        if is_reciprocal:
            # Index zero is held out, followed by every ``period``-th item.
            validation_examples += (
                class_count + period - 1
            ) // period
            continue

        validation_examples += sum(
            (
                class_index * _SPLIT_MULTIPLIER + _SPLIT_OFFSET
            )
            % _SPLIT_BUCKET_COUNT
            < threshold
            for class_index in range(
                class_count,
            )
        )

    if keep_validation:
        return validation_examples

    return total_examples - validation_examples


def count_dataset_examples_by_class(
    dataset: tf.data.Dataset,
    num_classes: int,
) -> tuple[int, ...]:
    """Count labels in a finite dataset when metadata has no histogram."""
    if num_classes < 1:
        raise ValueError(
            f"num_classes must be positive. Got {num_classes}."
        )

    class_counts = [
        0
        for _ in range(
            num_classes,
        )
    ]

    def extract_label(
        *element,
    ) -> tf.Tensor:
        packed = _pack_element(
            element,
        )
        label = _label_from_element(
            packed,
        )
        if label is None:
            raise ValueError(
                "Cannot count class examples because the dataset has no "
                "'label' field."
            )

        return tf.cast(
            label,
            tf.int64,
        )

    label_ds = dataset.map(
        extract_label,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    for label in label_ds.as_numpy_iterator():
        label_value = int(
            label,
        )
        if label_value < 0 or label_value >= num_classes:
            raise ValueError(
                "Dataset label is outside the expected class range: "
                f"label={label_value}, num_classes={num_classes}."
            )
        class_counts[label_value] += 1

    return tuple(
        class_counts,
    )


def _keep_by_index(
    index: tf.Tensor,
    validation_split: float,
    keep_validation: bool,
) -> tf.Tensor:
    """Return whether a deterministic index belongs to train or validation."""
    is_reciprocal, period = _is_reciprocal_split(
        validation_split,
    )

    if is_reciprocal:
        is_validation = tf.equal(
            tf.math.floormod(
                index,
                tf.cast(
                    period,
                    index.dtype,
                ),
            ),
            tf.zeros(
                (),
                dtype=index.dtype,
            ),
        )

    else:
        threshold = int(
            round(
                validation_split * _SPLIT_BUCKET_COUNT,
            )
        )

        bucket = tf.math.floormod(
            index * _SPLIT_MULTIPLIER + _SPLIT_OFFSET,
            _SPLIT_BUCKET_COUNT,
        )

        is_validation = bucket < threshold

    return is_validation if keep_validation else tf.logical_not(is_validation)


def _split_train_validation_by_global_index(
    dataset: tf.data.Dataset,
    validation_split: float,
    keep_validation: bool,
) -> tf.data.Dataset:
    """Split a dataset deterministically by global example index."""
    return (
        dataset.enumerate()
        .filter(
            lambda index, example: _keep_by_index(
                index=index,
                validation_split=validation_split,
                keep_validation=keep_validation,
            )
        )
        .map(
            lambda _index, example: example,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    )


def _split_train_validation_by_label(
    dataset: tf.data.Dataset,
    validation_split: float,
    keep_validation: bool,
) -> tf.data.Dataset:
    """Split a labeled dataset independently within each class."""

    def key_func(
        *element,
    ) -> tf.Tensor:
        packed = _pack_element(
            element,
        )
        label = _label_from_element(
            packed,
        )

        return tf.cast(
            label,
            tf.int64,
        )

    def reduce_func(
        _label: tf.Tensor,
        class_dataset: tf.data.Dataset,
    ) -> tf.data.Dataset:
        return (
            class_dataset.enumerate()
            .filter(
                lambda class_index, example: _keep_by_index(
                    index=class_index,
                    validation_split=validation_split,
                    keep_validation=keep_validation,
                )
            )
            .map(
                lambda _class_index, example: example,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        )

    return dataset.group_by_window(
        key_func=key_func,
        reduce_func=reduce_func,
        window_size=1_000_000_000,
    )


def split_train_validation_dataset(
    dataset: tf.data.Dataset,
    validation_split: float,
    keep_validation: bool,
) -> tf.data.Dataset:
    """Split train data into deterministic class-stratified train/validation."""
    validate_validation_split(
        validation_split=validation_split,
    )

    if validation_split == 0.0:
        return dataset

    threshold = int(
        round(
            validation_split * _SPLIT_BUCKET_COUNT,
        )
    )

    if threshold <= 0 or threshold >= _SPLIT_BUCKET_COUNT:
        raise ValueError(
            "validation_split is too close to 0 or 1 for deterministic "
            f"bucketing: {validation_split}."
        )

    if _has_label(
        dataset.element_spec,
    ):
        return _split_train_validation_by_label(
            dataset=dataset,
            validation_split=validation_split,
            keep_validation=keep_validation,
        )

    return _split_train_validation_by_global_index(
        dataset=dataset,
        validation_split=validation_split,
        keep_validation=keep_validation,
    )
