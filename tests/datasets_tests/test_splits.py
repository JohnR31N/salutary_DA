from __future__ import annotations

import tensorflow as tf

import pytest

from allthemix.data.splits import (
    count_class_stratified_split_examples,
    count_dataset_examples_by_class,
    count_train_subset_examples_per_class,
    split_train_validation_dataset,
    subset_train_dataset,
)


def _labeled_id_dataset(
    class_counts: tuple[int, ...],
) -> tf.data.Dataset:
    """Build a labeled dataset whose ids identify every example uniquely."""
    labels = tf.concat(
        [
            tf.fill(
                (count,),
                tf.cast(
                    label,
                    tf.int64,
                ),
            )
            for label, count in enumerate(
                class_counts,
            )
        ],
        axis=0,
    )

    return tf.data.Dataset.from_tensor_slices(
        {
            "image": tf.range(
                sum(
                    class_counts,
                ),
                dtype=tf.int64,
            ),
            "label": labels,
        }
    )


def test_exact_class_split_count_handles_uneven_classes() -> None:
    """Verify counting matches the reciprocal per-class index rule."""
    class_counts = (
        25,
        11,
        1,
    )

    assert count_class_stratified_split_examples(
        class_counts=class_counts,
        validation_split=0.1,
        keep_validation=True,
    ) == 6
    assert count_class_stratified_split_examples(
        class_counts=class_counts,
        validation_split=0.1,
        keep_validation=False,
    ) == 31


def test_exact_class_split_count_matches_tensorflow_hash_split() -> None:
    """Verify non-reciprocal Python counting mirrors the tf.data filter."""
    class_counts = (
        7,
        11,
        19,
    )
    labels = tf.concat(
        [
            tf.fill(
                (count,),
                tf.cast(
                    label,
                    tf.int64,
                ),
            )
            for label, count in enumerate(
                class_counts,
            )
        ],
        axis=0,
    )
    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "image": tf.zeros(
                (
                    sum(
                        class_counts,
                    ),
                    1,
                ),
                dtype=tf.float32,
            ),
            "label": labels,
        }
    )
    validation_ds = split_train_validation_dataset(
        dataset=dataset,
        validation_split=0.17,
        keep_validation=True,
    )

    assert count_class_stratified_split_examples(
        class_counts=class_counts,
        validation_split=0.17,
        keep_validation=True,
    ) == sum(
        1
        for _example in validation_ds
    )
    assert count_dataset_examples_by_class(
        dataset=dataset,
        num_classes=3,
    ) == class_counts


def test_split_train_validation_dataset_is_disjoint() -> None:
    """Verify that deterministic validation splitting keeps disjoint examples."""
    dataset = tf.data.Dataset.range(1_000)

    train_ds = split_train_validation_dataset(
        dataset=dataset,
        validation_split=0.1,
        keep_validation=False,
    )

    validation_ds = split_train_validation_dataset(
        dataset=dataset,
        validation_split=0.1,
        keep_validation=True,
    )

    train_values = set(train_ds.as_numpy_iterator())
    validation_values = set(validation_ds.as_numpy_iterator())

    assert train_values.isdisjoint(validation_values)
    assert train_values | validation_values == set(range(1_000))
    assert 80 <= len(validation_values) <= 120


def test_split_train_validation_dataset_is_class_stratified() -> None:
    """Verify that labeled datasets are split independently per class."""
    num_classes = 5
    examples_per_class = 20
    labels = tf.repeat(
        tf.range(
            num_classes,
            dtype=tf.int64,
        ),
        examples_per_class,
    )
    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "image": tf.zeros(
                (
                    num_classes * examples_per_class,
                    1,
                ),
                dtype=tf.float32,
            ),
            "label": labels,
        }
    )

    validation_ds = split_train_validation_dataset(
        dataset=dataset,
        validation_split=0.1,
        keep_validation=True,
    )

    validation_labels = [
        int(
            example["label"].numpy(),
        )
        for example in validation_ds
    ]

    assert len(validation_labels) == num_classes * 2
    assert {
        label: validation_labels.count(
            label,
        )
        for label in range(
            num_classes,
        )
    } == {
        label: 2
        for label in range(
            num_classes,
        )
    }


def test_split_train_validation_dataset_keeps_zipped_aux_alignment() -> None:
    """Verify that stratified splitting preserves zipped saliency alignment."""
    labels = tf.repeat(
        tf.range(
            3,
            dtype=tf.int64,
        ),
        10,
    )
    ids = tf.range(
        labels.shape[0],
        dtype=tf.int64,
    )
    image_ds = tf.data.Dataset.from_tensor_slices(
        {
            "image": ids,
            "label": labels,
        }
    )
    saliency_ds = tf.data.Dataset.from_tensor_slices(
        ids + 1_000,
    )
    dataset = tf.data.Dataset.zip(
        (
            image_ds,
            saliency_ds,
        )
    )

    validation_ds = split_train_validation_dataset(
        dataset=dataset,
        validation_split=0.1,
        keep_validation=True,
    )

    for example, saliency in validation_ds:
        assert int(
            saliency.numpy(),
        ) == int(
            example["image"].numpy(),
        ) + 1_000


def test_train_subset_is_deterministic_stratified_and_nested() -> None:
    """Verify the train subset is stable, per-class exact, and a subset."""
    class_counts = (
        40,
        37,
        23,
    )
    dataset = _labeled_id_dataset(
        class_counts,
    )
    expected_per_class = count_train_subset_examples_per_class(
        class_counts=class_counts,
        train_subset_fraction=0.25,
    )

    def subset_ids() -> set[int]:
        return {
            int(
                example["image"].numpy(),
            )
            for example in subset_train_dataset(
                dataset=dataset,
                train_subset_fraction=0.25,
            )
        }

    first_ids = subset_ids()
    second_ids = subset_ids()
    full_ids = {
        int(
            example["image"].numpy(),
        )
        for example in dataset
    }
    per_class_observed = [
        0
        for _ in class_counts
    ]

    for example in subset_train_dataset(
        dataset=dataset,
        train_subset_fraction=0.25,
    ):
        per_class_observed[
            int(
                example["label"].numpy(),
            )
        ] += 1

    assert first_ids == second_ids
    assert first_ids <= full_ids
    assert tuple(per_class_observed) == expected_per_class
    assert sum(expected_per_class) == len(first_ids)


def test_train_subset_then_validation_split_partitions_subset() -> None:
    """Verify the validation split partitions the reduced labeled budget."""
    class_counts = (
        30,
        50,
        20,
    )
    dataset = _labeled_id_dataset(
        class_counts,
    )
    subset_ds = subset_train_dataset(
        dataset=dataset,
        train_subset_fraction=0.2,
    )
    subset_ids = {
        int(
            example["image"].numpy(),
        )
        for example in subset_ds
    }
    train_ids = {
        int(
            example["image"].numpy(),
        )
        for example in split_train_validation_dataset(
            dataset=subset_ds,
            validation_split=0.1,
            keep_validation=False,
        )
    }
    validation_ids = {
        int(
            example["image"].numpy(),
        )
        for example in split_train_validation_dataset(
            dataset=subset_ds,
            validation_split=0.1,
            keep_validation=True,
        )
    }
    subset_per_class = count_train_subset_examples_per_class(
        class_counts=class_counts,
        train_subset_fraction=0.2,
    )

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == subset_ids
    assert len(train_ids) == count_class_stratified_split_examples(
        class_counts=subset_per_class,
        validation_split=0.1,
        keep_validation=False,
    )
    assert len(validation_ids) == count_class_stratified_split_examples(
        class_counts=subset_per_class,
        validation_split=0.1,
        keep_validation=True,
    )


def test_train_subset_fraction_validates_range_and_keeps_identity() -> None:
    """Verify fraction bounds and the exact identity pass-through at 1.0."""
    dataset = _labeled_id_dataset(
        (
            10,
            10,
        )
    )

    with pytest.raises(ValueError):
        subset_train_dataset(
            dataset=dataset,
            train_subset_fraction=0.0,
        )
    with pytest.raises(ValueError):
        subset_train_dataset(
            dataset=dataset,
            train_subset_fraction=1.5,
        )

    identity_ds = subset_train_dataset(
        dataset=dataset,
        train_subset_fraction=1.0,
    )

    assert identity_ds is dataset
    assert count_train_subset_examples_per_class(
        class_counts=(
            10,
            10,
        ),
        train_subset_fraction=1.0,
    ) == (
        10,
        10,
    )
