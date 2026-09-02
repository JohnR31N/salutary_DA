from __future__ import annotations

import random

import numpy as np
import tensorflow as tf

from allthemix.data.pipeline import build_train_pipeline
from allthemix.data.preprocessors.selector import get_preprocessor
from allthemix.data.preprocessors.simple_sal_aug import (
    apply_sal_augmentation_recipe,
)
from allthemix.utils.reproducibility import resolve_data_seed, seed_everything


def _synthetic_cifar_source() -> tf.data.Dataset:
    """Return identifiable images whose crops and ordering can be compared."""
    images = tf.reshape(
        tf.range(
            12 * 32 * 32 * 3,
            dtype=tf.int32,
        ),
        (
            12,
            32,
            32,
            3,
        ),
    )
    images = tf.cast(
        tf.math.floormod(
            images,
            251,
        ),
        tf.uint8,
    )

    return tf.data.Dataset.from_tensor_slices(
        {
            "image": images,
            "label": tf.range(
                12,
                dtype=tf.int64,
            ),
        }
    )


def _collect_epoch(
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one finite dataset iteration as an epoch snapshot."""
    images = []
    labels = []
    for batch_images, batch_labels in dataset.as_numpy_iterator():
        images.append(
            batch_images,
        )
        labels.append(
            batch_labels,
        )

    return (
        np.concatenate(
            images,
            axis=0,
        ),
        np.concatenate(
            labels,
            axis=0,
        ),
    )


def _build_synthetic_pipeline(
    monkeypatch,
    seed: int,
) -> tf.data.Dataset:
    """Build the real CIFAR train pipeline over a local synthetic source."""
    import allthemix.data.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "load_train_dataset",
        lambda **_kwargs: _synthetic_cifar_source(),
    )

    return build_train_pipeline(
        name="cifar10",
        data_dir="unused",
        batch_size=4,
        shuffle_buffer_size=12,
        drop_remainder=False,
        use_basic_augmentation=True,
        augmentation_recipe="basic",
        seed=seed,
        deterministic_data=True,
    )


def test_same_data_seed_reproduces_each_epoch(
    monkeypatch,
) -> None:
    """Independent pipelines must emit the same epoch-indexed data stream."""
    first = _build_synthetic_pipeline(
        monkeypatch=monkeypatch,
        seed=17,
    )
    second = _build_synthetic_pipeline(
        monkeypatch=monkeypatch,
        seed=17,
    )

    first_epoch_1 = _collect_epoch(
        first,
    )
    first_epoch_2 = _collect_epoch(
        first,
    )
    second_epoch_1 = _collect_epoch(
        second,
    )
    second_epoch_2 = _collect_epoch(
        second,
    )

    for first_snapshot, second_snapshot in (
        (
            first_epoch_1,
            second_epoch_1,
        ),
        (
            first_epoch_2,
            second_epoch_2,
        ),
    ):
        np.testing.assert_array_equal(
            first_snapshot[0],
            second_snapshot[0],
        )
        np.testing.assert_array_equal(
            first_snapshot[1],
            second_snapshot[1],
        )

    assert not np.array_equal(
        first_epoch_1[1],
        first_epoch_2[1],
    )


def test_different_data_seeds_change_training_stream(
    monkeypatch,
) -> None:
    """Changing data_seed must change at least ordering or augmentation."""
    first = _collect_epoch(
        _build_synthetic_pipeline(
            monkeypatch=monkeypatch,
            seed=17,
        )
    )
    second = _collect_epoch(
        _build_synthetic_pipeline(
            monkeypatch=monkeypatch,
            seed=18,
        )
    )

    assert not (
        np.array_equal(
            first[0],
            second[0],
        )
        and np.array_equal(
            first[1],
            second[1],
        )
    )


def test_stateless_augmentation_ignores_global_rng_progress() -> None:
    """An explicit example seed must fully determine its augmentation."""
    preprocessor = get_preprocessor(
        "cifar10",
    )
    example = next(
        iter(
            _synthetic_cifar_source(),
        )
    )
    seed = tf.constant(
        [
            3,
            11,
        ],
        dtype=tf.int64,
    )

    first, _ = preprocessor(
        example,
        True,
        "basic",
        seed,
    )
    _ = tf.random.uniform(
        shape=(
            100,
        ),
    )
    second, _ = preprocessor(
        example,
        True,
        "basic",
        seed,
    )

    np.testing.assert_array_equal(
        first.numpy(),
        second.numpy(),
    )


def test_stateless_saliency_augmentation_keeps_pair_reproducible() -> None:
    """Paired image/saliency geometry must share one reproducible seed."""
    saliency_map = tf.reshape(
        tf.linspace(
            0.0,
            1.0,
            32 * 32,
        ),
        (
            32,
            32,
        ),
    )
    image = tf.repeat(
        saliency_map[:, :, None],
        repeats=3,
        axis=2,
    )
    seed = tf.constant(
        [
            5,
            19,
        ],
        dtype=tf.int64,
    )

    first_image, first_map = apply_sal_augmentation_recipe(
        image=image,
        saliency_map=saliency_map,
        image_size=32,
        use_sal_basic_augmentation=True,
        saliency_augmentation_recipe="basic",
        seed=seed,
    )
    _ = tf.random.uniform(
        shape=(
            100,
        ),
    )
    second_image, second_map = apply_sal_augmentation_recipe(
        image=image,
        saliency_map=saliency_map,
        image_size=32,
        use_sal_basic_augmentation=True,
        saliency_augmentation_recipe="basic",
        seed=seed,
    )

    np.testing.assert_array_equal(
        first_image.numpy(),
        second_image.numpy(),
    )
    np.testing.assert_array_equal(
        first_map.numpy(),
        second_map.numpy(),
    )
    np.testing.assert_allclose(
        first_image[:, :, 0].numpy(),
        first_map.numpy(),
        rtol=0.0,
        atol=0.0,
    )


def test_seed_everything_resets_host_random_streams() -> None:
    """The experiment seed must reset Python, NumPy, and TensorFlow RNGs."""
    seed_everything(
        23,
    )
    first = (
        random.random(),
        float(
            np.random.uniform(),
        ),
        float(
            tf.random.uniform(
                shape=(),
            ).numpy()
        ),
    )

    seed_everything(
        23,
    )
    second = (
        random.random(),
        float(
            np.random.uniform(),
        ),
        float(
            tf.random.uniform(
                shape=(),
            ).numpy()
        ),
    )

    assert first == second


def test_data_seed_defaults_to_experiment_seed() -> None:
    """The common case should need only one user-facing seed value."""
    assert resolve_data_seed(
        experiment_seed=7,
        data_seed=-1,
    ) == 7
    assert resolve_data_seed(
        experiment_seed=7,
        data_seed=29,
    ) == 29
