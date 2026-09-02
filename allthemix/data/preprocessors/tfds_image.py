from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from allthemix.data.datasets.imagenet100 import (
    CMC_IMAGENET100_NUM_CLASSES,
    CMC_IMAGENET100_TRAIN_EXAMPLES,
    CMC_IMAGENET100_VAL_EXAMPLES,
)
from allthemix.data.preprocessors.augmentation import (
    apply_augmentation_recipe,
    resolve_augmentation_recipe,
)

IMAGENET_EVAL_RESIZE_SIZE = 256


@dataclass(frozen=True)
class TfdsImageMetadata:
    name: str
    num_classes: int
    image_size: int
    channels: int
    num_train_examples: int
    num_test_examples: int
    train_class_counts: tuple[int, ...] | None = None


SUPPORTED_TFDS_IMAGE_DATASETS = {
    "caltech_birds2011",
    "cars196",
    "imagenet100",
    "svhn_cropped",
    "stl10",
    "oxford_iiit_pet",
}


def is_supported_tfds_image_dataset(
    name: str,
) -> bool:
    """Return whether a dataset is handled by the generic TFDS image preprocessor."""
    return name.lower() in SUPPORTED_TFDS_IMAGE_DATASETS


def get_metadata(
    name: str,
) -> TfdsImageMetadata:
    """Return metadata for supported TFDS image-classification datasets."""
    dataset_name = name.lower()

    if dataset_name == "svhn_cropped":
        return TfdsImageMetadata(
            name="svhn_cropped",
            num_classes=10,
            image_size=32,
            channels=3,
            num_train_examples=73_257,
            num_test_examples=26_032,
        )

    if dataset_name == "stl10":
        return TfdsImageMetadata(
            name="stl10",
            num_classes=10,
            image_size=96,
            channels=3,
            num_train_examples=5_000,
            num_test_examples=8_000,
            train_class_counts=(
                500,
            )
            * 10,
        )

    if dataset_name == "oxford_iiit_pet":
        return TfdsImageMetadata(
            name="oxford_iiit_pet",
            num_classes=37,
            image_size=224,
            channels=3,
            num_train_examples=3_680,
            num_test_examples=3_669,
        )

    if dataset_name == "cars196":
        return TfdsImageMetadata(
            name="cars196",
            num_classes=196,
            image_size=224,
            channels=3,
            num_train_examples=8_144,
            num_test_examples=8_041,
        )

    if dataset_name == "imagenet100":
        return TfdsImageMetadata(
            name="imagenet100",
            num_classes=CMC_IMAGENET100_NUM_CLASSES,
            image_size=224,
            channels=3,
            num_train_examples=CMC_IMAGENET100_TRAIN_EXAMPLES,
            num_test_examples=CMC_IMAGENET100_VAL_EXAMPLES,
        )

    if dataset_name == "caltech_birds2011":
        return TfdsImageMetadata(
            name="caltech_birds2011",
            num_classes=200,
            image_size=224,
            channels=3,
            num_train_examples=5_994,
            num_test_examples=5_794,
        )

    raise ValueError(f"Unsupported TFDS image dataset: {name}")


def get_normalization_stats(
    name: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return channel normalization statistics for supported TFDS datasets."""
    dataset_name = name.lower()

    if dataset_name == "svhn_cropped":
        mean = tf.constant(
            [0.4377, 0.4438, 0.4728],
            dtype=tf.float32,
        )
        std = tf.constant(
            [0.1980, 0.2010, 0.1970],
            dtype=tf.float32,
        )
        return mean, std

    if dataset_name in {
        "stl10",
        "oxford_iiit_pet",
        "cars196",
        "imagenet100",
        "caltech_birds2011",
    }:
        mean = tf.constant(
            [0.485, 0.456, 0.406],
            dtype=tf.float32,
        )
        std = tf.constant(
            [0.229, 0.224, 0.225],
            dtype=tf.float32,
        )
        return mean, std

    raise ValueError(f"Unsupported TFDS image dataset: {name}")


def normalize_float_image(
    image: tf.Tensor,
    name: str,
) -> tf.Tensor:
    """Normalize a float image with dataset-specific channel statistics."""
    mean, std = get_normalization_stats(
        name,
    )

    image = (image - mean) / std  # Standardize channels before model input.

    return image


def resize_shorter_side_and_center_crop(
    image: tf.Tensor,
    image_size: int,
    resize_size: int = IMAGENET_EVAL_RESIZE_SIZE,
) -> tf.Tensor:
    """Resize the shorter side and take a centered square evaluation crop."""
    image_shape = tf.shape(
        image,
    )
    image_height = image_shape[0]
    image_width = image_shape[1]
    shorter_side = tf.cast(
        tf.minimum(
            image_height,
            image_width,
        ),
        tf.float32,
    )
    resize_scale = tf.cast(
        resize_size,
        tf.float32,
    ) / shorter_side
    resized_height = tf.cast(
        tf.round(
            tf.cast(
                image_height,
                tf.float32,
            ) * resize_scale
        ),
        tf.int32,
    )
    resized_width = tf.cast(
        tf.round(
            tf.cast(
                image_width,
                tf.float32,
            ) * resize_scale
        ),
        tf.int32,
    )
    resized_height = tf.maximum(
        resized_height,
        image_size,
    )
    resized_width = tf.maximum(
        resized_width,
        image_size,
    )

    image = tf.image.resize(
        image,
        size=[
            resized_height,
            resized_width,
        ],
        method="bilinear",
        antialias=True,
    )
    image = tf.image.resize_with_crop_or_pad(
        image,
        target_height=image_size,
        target_width=image_size,
    )

    return tf.ensure_shape(
        image,
        [
            image_size,
            image_size,
            3,
        ],
    )


def resize_to_square(
    image: tf.Tensor,
    image_size: int,
) -> tf.Tensor:
    """Resize an image directly to the dataset's fixed square input size."""
    return tf.image.resize(
        image,
        size=[
            image_size,
            image_size,
        ],
        method="bilinear",
    )


def random_crop_with_padding(
    image: tf.Tensor,
    image_size: int,
    padding: int = 4,
) -> tf.Tensor:
    """Apply reflected-padding random crop at the dataset image size."""
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


def apply_basic_aug(
    image: tf.Tensor,
    image_size: int,
) -> tf.Tensor:
    """Apply crop and horizontal flip augmentation at the dataset image size."""
    image = random_crop_with_padding(
        image=image,
        image_size=image_size,
    )

    image = tf.image.random_flip_left_right(
        image,
    )

    return image


def preprocess_example(
    example: dict[str, tf.Tensor],
    use_basic_augmentation: bool = False,
    name: str = "svhn_cropped",
    augmentation_recipe: str | None = None,
    seed: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Preprocess one supported TFDS image-classification example."""
    metadata = get_metadata(
        name,
    )

    image = example["image"]
    label = example["label"]

    image = tf.cast(
        image,
        tf.float32,
    ) / 255.0  # Convert uint8 pixels to [0, 1].

    recipe = resolve_augmentation_recipe(
        use_basic_augmentation=use_basic_augmentation,
        augmentation_recipe=augmentation_recipe,
    )
    dataset_name = name.lower()

    uses_source_aspect_train_crop = (
        dataset_name == "imagenet100"
        and recipe == "imagenet"
    ) or (
        dataset_name == "cars196"
        and recipe == "fine_grained"
    )
    uses_center_crop_evaluation = dataset_name in {
        "imagenet100",
        "cars196",
    } and recipe == "none"

    if uses_source_aspect_train_crop:
        # Sample train crops before any square resize distorts the source image.
        augmentation_kwargs = {}
        if seed is not None:
            augmentation_kwargs["seed"] = seed
        image = apply_augmentation_recipe(
            image=image,
            image_size=metadata.image_size,
            use_basic_augmentation=use_basic_augmentation,
            augmentation_recipe=recipe,
            **augmentation_kwargs,
        )
    elif uses_center_crop_evaluation:
        image = resize_shorter_side_and_center_crop(
            image=image,
            image_size=metadata.image_size,
        )
    else:
        image = resize_to_square(
            image=image,
            image_size=metadata.image_size,
        )
        augmentation_kwargs = {}
        if seed is not None:
            augmentation_kwargs["seed"] = seed
        image = apply_augmentation_recipe(
            image=image,
            image_size=metadata.image_size,
            use_basic_augmentation=use_basic_augmentation,
            augmentation_recipe=recipe,
            **augmentation_kwargs,
        )

    image = tf.ensure_shape(
        image,
        [
            metadata.image_size,
            metadata.image_size,
            metadata.channels,
        ],
    )

    image = normalize_float_image(
        image=image,
        name=name,
    )

    label = tf.cast(
        label,
        tf.int64,
    )

    return image, label
