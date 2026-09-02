from __future__ import annotations

import tensorflow as tf


def attach_random_seed_stream(
    dataset: tf.data.Dataset,
    seed: int,
) -> tf.data.Dataset:
    """Attach a reproducible seed that is rerandomized for each iteration."""
    seed_dataset = tf.data.Dataset.random(
        seed=seed,
        rerandomize_each_iteration=True,
    )

    return tf.data.Dataset.zip(
        (
            dataset,
            seed_dataset,
        )
    )


def make_stateless_seed(
    base_seed: int,
    random_value: tf.Tensor,
) -> tf.Tensor:
    """Build a two-element TensorFlow stateless RNG seed."""
    return tf.stack(
        (
            tf.cast(
                base_seed,
                tf.int64,
            ),
            tf.cast(
                random_value,
                tf.int64,
            ),
        )
    )


def apply_dataset_determinism(
    dataset: tf.data.Dataset,
    deterministic: bool,
) -> tf.data.Dataset:
    """Set an explicit ordering policy for an entire input pipeline."""
    options = tf.data.Options()
    options.deterministic = deterministic

    return dataset.with_options(
        options,
    )
