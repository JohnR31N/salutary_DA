"""Source-aligned stochastic replacement for offline augmentation artifacts."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf


def attach_source_indices(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Attach the stable raw-dataset index used by offline generators."""

    def attach(index, example):
        result = dict(example)
        result["_source_index"] = tf.cast(index, tf.int64)

        return result

    return dataset.enumerate().map(
        attach,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )


def build_source_replacer(
    source_indices: Sequence[int],
    generated_paths: Sequence[Sequence[str]],
    source_labels: Sequence[int],
    probability: float,
):
    """Build a map function that samples only edits of the current source."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "Offline replacement probability must be in [0, 1]. "
            f"Got {probability}."
        )
    if not source_indices:
        raise ValueError("Offline replacement catalog is empty.")
    if not (
        len(source_indices) == len(generated_paths) == len(source_labels)
    ):
        raise ValueError("Offline replacement catalog columns disagree.")
    if any(not paths for paths in generated_paths):
        raise ValueError("Every replacement source needs at least one image.")

    starts = []
    lengths = []
    flat_paths = []
    for paths in generated_paths:
        starts.append(len(flat_paths))
        lengths.append(len(paths))
        flat_paths.extend(paths)
    index_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(source_indices, dtype=tf.int64),
            values=tf.range(len(source_indices), dtype=tf.int64),
        ),
        default_value=tf.constant(-1, dtype=tf.int64),
    )
    starts_tensor = tf.constant(starts, dtype=tf.int64)
    lengths_tensor = tf.constant(lengths, dtype=tf.int64)
    paths_tensor = tf.constant(flat_paths, dtype=tf.string)
    labels_tensor = tf.constant(source_labels, dtype=tf.int64)
    probability_tensor = tf.constant(probability, dtype=tf.float32)

    def replace(example: dict[str, tf.Tensor], seed: tf.Tensor | None = None):
        """Stochastically replace one raw image while preserving its label."""
        if "_source_index" not in example:
            raise ValueError(
                "Source replacement requires attach_source_indices first."
            )
        source_index = tf.cast(example["_source_index"], tf.int64)
        row = index_table.lookup(source_index)
        has_replacement = row >= 0
        safe_row = tf.maximum(row, 0)
        length = tf.gather(lengths_tensor, safe_row)
        if seed is None:
            apply_value = tf.random.uniform((), dtype=tf.float32)
            path_offset = tf.random.uniform(
                (),
                minval=0,
                maxval=length,
                dtype=tf.int64,
            )
        else:
            seeds = tf.random.experimental.stateless_split(
                tf.cast(seed, tf.int64),
                num=2,
            )
            apply_value = tf.random.stateless_uniform(
                (),
                seed=seeds[0],
                dtype=tf.float32,
            )
            path_offset = tf.random.stateless_uniform(
                (),
                seed=seeds[1],
                minval=0,
                maxval=length,
                dtype=tf.int64,
            )
        should_replace = tf.logical_and(
            has_replacement,
            apply_value < probability_tensor,
        )

        def decode_replacement():
            expected_label = tf.gather(labels_tensor, safe_row)
            label = tf.cast(example["label"], tf.int64)
            assertion = tf.debugging.assert_equal(
                expected_label,
                label,
                message=(
                    "Offline generated label does not match its original "
                    "source label."
                ),
            )
            with tf.control_dependencies([assertion]):
                path = tf.gather(
                    paths_tensor,
                    tf.gather(starts_tensor, safe_row) + path_offset,
                )
                image = tf.io.decode_image(
                    tf.io.read_file(path),
                    channels=3,
                    expand_animations=False,
                )
                image.set_shape([None, None, 3])

                return image

        original_image = example["image"]
        original_image.set_shape([None, None, 3])
        image = tf.cond(
            should_replace,
            decode_replacement,
            lambda: original_image,
        )

        return {
            "image": image,
            "label": tf.cast(example["label"], tf.int64),
        }

    return replace
