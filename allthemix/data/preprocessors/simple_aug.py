from __future__ import annotations

import tensorflow as tf


def random_crop_with_padding(
    image: tf.Tensor,
    padding: int = 4,
) -> tf.Tensor:
    """Apply CIFAR-style random crop after reflected padding."""
    image = tf.pad(
        image,
        paddings=[
            [padding, padding],
            [padding, padding],
            [0, 0],
        ],
        mode="REFLECT",
    )

    image = tf.image.random_crop(
        image,
        size=[32, 32, 3],
    )

    return image


def random_horizontal_flip(
    image: tf.Tensor,
) -> tf.Tensor:
    """Randomly flip an image left-to-right."""
    return tf.image.random_flip_left_right(image)


def apply_basic_aug(
    image: tf.Tensor,
) -> tf.Tensor:
    """Apply the standard random crop and horizontal flip augmentation."""
    image = random_crop_with_padding(image)
    image = random_horizontal_flip(image)

    return image
