from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf

from allthemix.data.datasets.registry import CIFAR10, CIFAR100, canonicalize
from allthemix.data.datasets.tiny_imagenet import is_tiny_imagenet_name
from allthemix.data.preprocessors import cifar, tfds_image, tiny_imagenet

PreprocessFn = Callable[
    [dict[str, tf.Tensor], bool, str | None, tf.Tensor | None],
    tuple[tf.Tensor, tf.Tensor],
]


def get_preprocessor(
    name: str,
    tiny_imagenet_normalization: str = "imagenet",
) -> PreprocessFn:
    """Get preprocessor."""
    dataset_name = name.lower()

    if dataset_name in {CIFAR10, CIFAR100}:
        def preprocess(
            example: dict[str, tf.Tensor],
            use_basic_augmentation: bool = False,
            augmentation_recipe: str | None = None,
            seed: tf.Tensor | None = None,
        ) -> tuple[tf.Tensor, tf.Tensor]:
            """Preprocess one dataset example."""
            return cifar.preprocess_example(
                example=example,
                use_basic_augmentation=use_basic_augmentation,
                name=dataset_name,
                augmentation_recipe=augmentation_recipe,
                seed=seed,
            )

        return preprocess

    if is_tiny_imagenet_name(dataset_name):
        def preprocess(
            example: dict[str, tf.Tensor],
            use_basic_augmentation: bool = False,
            augmentation_recipe: str | None = None,
            seed: tf.Tensor | None = None,
        ) -> tuple[tf.Tensor, tf.Tensor]:
            """Preprocess one dataset example."""
            return tiny_imagenet.preprocess_example(
                example=example,
                use_basic_augmentation=use_basic_augmentation,
                augmentation_recipe=augmentation_recipe,
                normalization=tiny_imagenet_normalization,
                seed=seed,
            )

        return preprocess

    if tfds_image.is_supported_tfds_image_dataset(dataset_name):
        def preprocess(
            example: dict[str, tf.Tensor],
            use_basic_augmentation: bool = False,
            augmentation_recipe: str | None = None,
            seed: tf.Tensor | None = None,
        ) -> tuple[tf.Tensor, tf.Tensor]:
            """Preprocess one dataset example."""
            return tfds_image.preprocess_example(
                example=example,
                use_basic_augmentation=use_basic_augmentation,
                name=dataset_name,
                augmentation_recipe=augmentation_recipe,
                seed=seed,
            )

        return preprocess

    raise ValueError(
        f"Unsupported dataset preprocessor: {name}"
    )


def get_metadata(name: str):
    """Resolve dataset metadata via the canonical dataset registry."""
    dataset_name = canonicalize(name)

    if dataset_name in {CIFAR10, CIFAR100}:
        return cifar.get_metadata(name)

    if is_tiny_imagenet_name(dataset_name):
        return tiny_imagenet.get_metadata(name)

    if tfds_image.is_supported_tfds_image_dataset(dataset_name):
        return tfds_image.get_metadata(name)

    raise ValueError(
        f"Unsupported dataset metadata: {name}"
    )
