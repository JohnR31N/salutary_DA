from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from allthemix.data.datasets.tiny_imagenet import TINY_IMAGENET_NAME
from allthemix.data.preprocessors.augmentation import apply_augmentation_recipe

TINY_IMAGENET_NORMALIZATION_MODES = (
    "imagenet",
    "none",
)


@dataclass(frozen=True)
class TinyImageNetMetadata:
    name: str
    num_classes: int
    image_size: int
    channels: int
    num_train_examples: int
    num_test_examples: int
    train_class_counts: tuple[int, ...] | None = None


def get_metadata(
    name: str = TINY_IMAGENET_NAME,
) -> TinyImageNetMetadata:
    """Return Tiny-ImageNet image and class metadata."""
    del name

    return TinyImageNetMetadata(
        name=TINY_IMAGENET_NAME,
        num_classes=200,
        image_size=64,
        channels=3,
        num_train_examples=100_000,
        num_test_examples=10_000,
        train_class_counts=(
            500,
        )
        * 200,
    )


def get_normalization_stats() -> tuple[tf.Tensor, tf.Tensor]:
    # ImageNet normalization, commonly used for Tiny ImageNet / ImageNet-like data.
    """Get normalization stats."""
    mean = tf.constant(
        [0.485, 0.456, 0.406],
        dtype=tf.float32,
    )

    std = tf.constant(
        [0.229, 0.224, 0.225],
        dtype=tf.float32,
    )

    return mean, std


def validate_normalization_mode(
    normalization: str,
) -> str:
    """Validate and normalize the Tiny ImageNet normalization mode."""
    normalization = normalization.lower()

    if normalization not in TINY_IMAGENET_NORMALIZATION_MODES:
        raise ValueError(
            "Unsupported Tiny ImageNet normalization mode: "
            f"{normalization}. Expected one of "
            f"{TINY_IMAGENET_NORMALIZATION_MODES}."
        )

    return normalization


def normalize_float_image(
    image: tf.Tensor,
    normalization: str = "imagenet",
) -> tf.Tensor:
    """Normalize a float Tiny ImageNet image with ImageNet statistics."""
    normalization = validate_normalization_mode(
        normalization,
    )

    if normalization == "none":
        return image

    mean, std = get_normalization_stats()

    image = (image - mean) / std  # Standardize each channel to ImageNet statistics.

    return image


def random_crop_with_padding(
    image: tf.Tensor,
    image_size: int = 64,
    padding: int = 4,
) -> tf.Tensor:
    """Apply Tiny-ImageNet-style random crop after reflected padding."""
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
        size=[
            image_size,
            image_size,
            3,
        ],
    )

    return image


def random_horizontal_flip(
    image: tf.Tensor,
) -> tf.Tensor:
    """Randomly flip a Tiny ImageNet image left-to-right."""
    return tf.image.random_flip_left_right(
        image,
    )


def apply_basic_aug(
    image: tf.Tensor,
) -> tf.Tensor:
    """Apply standard Tiny ImageNet crop and horizontal flip augmentation."""
    image = random_crop_with_padding(
        image=image,
        image_size=64,
        padding=4,
    )

    image = random_horizontal_flip(
        image,
    )

    return image


def preprocess_example(
    example: dict[str, tf.Tensor],
    use_basic_augmentation: bool = False,
    augmentation_recipe: str | None = None,
    normalization: str = "imagenet",
    seed: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Preprocess example."""
    image = example["image"]
    label = example["label"]

    image = tf.cast(  # Convert uint8 pixels to [0, 1].
        image,
        tf.float32,
    ) / 255.0

    # Tiny ImageNet images are 64x64, but resizing here makes the pipeline
    # robust even if a decoded image has an unexpected shape.
    image = tf.image.resize(
        image,
        size=[
            64,
            64,
        ],
        method="bilinear",
    )

    image = tf.ensure_shape(
        image,
        [
            64,
            64,
            3,
        ],
    )

    image = apply_augmentation_recipe(
        image=image,
        image_size=64,
        use_basic_augmentation=use_basic_augmentation,
        augmentation_recipe=augmentation_recipe,
        seed=seed,
    )

    image = normalize_float_image(
        image,
        normalization=normalization,
    )

    label = tf.cast(
        label,
        tf.int64,
    )

    return image, label
