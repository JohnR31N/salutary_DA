from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from allthemix.data.preprocessors.augmentation import apply_augmentation_recipe


@dataclass(frozen=True)
class CifarMetadata:
    name: str
    num_classes: int
    image_size: int
    channels: int
    num_train_examples: int
    num_test_examples: int
    train_class_counts: tuple[int, ...] | None = None


def get_metadata(name: str) -> CifarMetadata:
    """Return CIFAR-10/100 image and class metadata."""
    dataset_name = name.lower()

    if dataset_name == "cifar10":
        return CifarMetadata(
            name="cifar10",
            num_classes=10,
            image_size=32,
            channels=3,
            num_train_examples=50_000,
            num_test_examples=10_000,
            train_class_counts=(
                5_000,
            )
            * 10,
        )

    if dataset_name == "cifar100":
        return CifarMetadata(
            name="cifar100",
            num_classes=100,
            image_size=32,
            channels=3,
            num_train_examples=50_000,
            num_test_examples=10_000,
            train_class_counts=(
                500,
            )
            * 100,
        )

    raise ValueError(f"Unsupported CIFAR dataset: {name}")


def get_normalization_stats(
    name: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Get normalization stats."""
    dataset_name = name.lower()

    if dataset_name == "cifar10":
        mean = tf.constant(
            [0.4914, 0.4822, 0.4465],
            dtype=tf.float32,
        )

        std = tf.constant(
            [0.2470, 0.2435, 0.2616],
            dtype=tf.float32,
        )

        return mean, std

    if dataset_name == "cifar100":
        mean = tf.constant(
            [0.5071, 0.4867, 0.4408],
            dtype=tf.float32,
        )

        std = tf.constant(
            [0.2675, 0.2565, 0.2761],
            dtype=tf.float32,
        )

        return mean, std

    raise ValueError(f"Unsupported CIFAR dataset: {name}")


def normalize_float_image(
    image: tf.Tensor,
    name: str,
) -> tf.Tensor:
    """Normalize a float CIFAR image with dataset statistics."""
    mean, std = get_normalization_stats(
        name,
    )

    image = (image - mean) / std  # Standardize each channel to dataset statistics.

    return image


def preprocess_example(
    example: dict[str, tf.Tensor],
    use_basic_augmentation: bool = False,
    name: str = "cifar10",
    augmentation_recipe: str | None = None,
    seed: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Preprocess example."""
    image = example["image"]
    label = example["label"]

    image = tf.cast(image, tf.float32) / 255.0  # Convert uint8 pixels to [0, 1].

    image = apply_augmentation_recipe(
        image=image,
        image_size=32,
        use_basic_augmentation=use_basic_augmentation,
        augmentation_recipe=augmentation_recipe,
        seed=seed,
    )

    image = normalize_float_image(
        image,
        name=name,
    )

    return image, label
