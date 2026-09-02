from __future__ import annotations

import tensorflow as tf

from allthemix.data.preprocessors import simple_sal_aug
from allthemix.data.preprocessors.simple_sal_aug import apply_sal_augmentation_recipe


def test_saliency_imagenet_aug_resizes_image_and_saliency_pair() -> None:
    """Verify paired ImageNet aug returns fixed-size image and saliency tensors."""
    tf.random.set_seed(
        0,
    )

    image = tf.ones(
        (
            260,
            320,
            3,
        ),
        dtype=tf.float32,
    )
    saliency_map = tf.ones(
        (
            260,
            320,
        ),
        dtype=tf.float32,
    )

    image, saliency_map = apply_sal_augmentation_recipe(
        image=image,
        saliency_map=saliency_map,
        image_size=224,
        use_sal_basic_augmentation=False,
        saliency_augmentation_recipe="imagenet",
    )

    assert image.shape == (
        224,
        224,
        3,
    )
    assert saliency_map.shape == (
        224,
        224,
    )
    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                image,
            )
        )
    )
    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                saliency_map,
            )
        )
    )


def test_saliency_tiny_openmixup_aug_resizes_image_and_saliency_pair() -> None:
    """Verify paired Tiny OpenMixup aug returns fixed-size tensors."""
    tf.random.set_seed(
        0,
    )

    image = tf.ones(
        (
            64,
            64,
            3,
        ),
        dtype=tf.float32,
    )
    saliency_map = tf.ones(
        (
            64,
            64,
        ),
        dtype=tf.float32,
    )

    image, saliency_map = apply_sal_augmentation_recipe(
        image=image,
        saliency_map=saliency_map,
        image_size=64,
        use_sal_basic_augmentation=False,
        saliency_augmentation_recipe="tiny_openmixup",
    )

    assert image.shape == (
        64,
        64,
        3,
    )
    assert saliency_map.shape == (
        64,
        64,
    )


def test_saliency_cub_aug_resizes_image_and_saliency_pair(
    monkeypatch,
) -> None:
    """Verify the CUB recipe keeps image and saliency shapes aligned."""
    tf.random.set_seed(
        0,
    )

    coordinate_map = tf.reshape(
        tf.linspace(
            0.0,
            1.0,
            280 * 320,
        ),
        (
            280,
            320,
        ),
    )
    image = tf.repeat(
        coordinate_map[:, :, None],
        repeats=3,
        axis=2,
    )
    saliency_map = coordinate_map

    monkeypatch.setattr(
        simple_sal_aug,
        "apply_cub_color_jitter",
        lambda value: value,
    )

    image, saliency_map = apply_sal_augmentation_recipe(
        image=image,
        saliency_map=saliency_map,
        image_size=224,
        use_sal_basic_augmentation=False,
        saliency_augmentation_recipe="cub",
    )

    assert image.shape == (
        224,
        224,
        3,
    )
    assert saliency_map.shape == (
        224,
        224,
    )
    assert bool(
        tf.reduce_all(
            tf.abs(
                image[:, :, 0]
                - saliency_map
            )
            < 1e-6
        )
    )


def test_saliency_fine_grained_aug_keeps_pair_aligned(
    monkeypatch,
) -> None:
    """Verify fine-grained image and saliency geometry stays aligned."""
    tf.random.set_seed(
        0,
    )
    coordinate_map = tf.reshape(
        tf.linspace(
            0.0,
            1.0,
            224 * 224,
        ),
        (
            224,
            224,
        ),
    )
    image = tf.repeat(
        coordinate_map[:, :, None],
        repeats=3,
        axis=2,
    )

    monkeypatch.setattr(
        simple_sal_aug,
        "apply_fine_grained_color_jitter",
        lambda value: value,
    )
    image, saliency_map = apply_sal_augmentation_recipe(
        image=image,
        saliency_map=coordinate_map,
        image_size=224,
        use_sal_basic_augmentation=False,
        saliency_augmentation_recipe="fine_grained",
    )

    assert image.shape == (
        224,
        224,
        3,
    )
    assert saliency_map.shape == (
        224,
        224,
    )
    assert bool(
        tf.reduce_all(
            tf.abs(
                image[:, :, 0]
                - saliency_map
            )
            < 1e-6
        )
    )


def test_legacy_sal_basic_aug_selects_basic_pair_recipe() -> None:
    """Verify the legacy boolean still runs the paired basic augmentation."""
    tf.random.set_seed(
        0,
    )

    image = tf.ones(
        (
            32,
            32,
            3,
        ),
        dtype=tf.float32,
    )
    saliency_map = tf.ones(
        (
            32,
            32,
        ),
        dtype=tf.float32,
    )

    image, saliency_map = apply_sal_augmentation_recipe(
        image=image,
        saliency_map=saliency_map,
        image_size=32,
        use_sal_basic_augmentation=True,
        saliency_augmentation_recipe=None,
    )

    assert image.shape == (
        32,
        32,
        3,
    )
    assert saliency_map.shape == (
        32,
        32,
    )
