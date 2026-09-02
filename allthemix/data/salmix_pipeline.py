from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from allthemix.data.datasets.cars196 import is_cars196_name
from allthemix.data.datasets.imagenet100 import is_imagenet100_name
from allthemix.data.datasets.loader import load_test_dataset, load_train_dataset
from allthemix.data.datasets.tiny_imagenet import is_tiny_imagenet_name
from allthemix.data.preprocessors import cifar, tfds_image, tiny_imagenet
from allthemix.data.preprocessors.augmentation import resolve_augmentation_recipe
from allthemix.data.preprocessors.selector import get_metadata, get_preprocessor
from allthemix.data.preprocessors.simple_sal_aug import apply_sal_augmentation_recipe
from allthemix.data.saliency import load_train_saliency_maps
from allthemix.data.splits import (
    resolve_training_validation_split,
    split_train_validation_dataset,
    validate_validation_source,
)
from allthemix.data.utils.cardinality import resolve_source_train_example_count
from allthemix.data.utils.random import (
    apply_dataset_determinism,
    attach_random_seed_stream,
    make_stateless_seed,
)


def _canonical_dataset_name(
    name: str,
) -> str:
    """Normalize dataset names to lowercase without accepting aliases."""
    return name.lower()


def _load_train_dataset_without_file_shuffle(
    name: str,
    data_dir: str,
) -> tf.data.Dataset:
    """
    Load the training dataset without file-level shuffling.

    This is important because precomputed saliency maps are stored in the
    original training order. If the image dataset is shuffled before it is
    zipped with saliency maps, images and saliency maps become misaligned.

    Custom local loaders are already deterministic. For TFDS datasets, use
    shuffle_files=False to preserve saliency-map pairing order.
    """
    if is_tiny_imagenet_name(
        name,
    ) or is_cars196_name(
        name,
    ) or is_imagenet100_name(
        name,
    ):
        return load_train_dataset(
            name=name,
            data_dir=data_dir,
        )

    dataset = tfds.load(
        name=name,
        split="train",
        data_dir=data_dir,
        shuffle_files=False,
        download=True,
    )

    return dataset


def _resize_image(
    image: tf.Tensor,
    image_size: int,
) -> tf.Tensor:
    """Resize an image and enforce static image shape."""
    image = tf.image.resize(
        image,
        size=[
            image_size,
            image_size,
        ],
        method="bilinear",
    )

    image = tf.ensure_shape(
        image,
        [
            image_size,
            image_size,
            3,
        ],
    )

    return image


def _resize_saliency_map(
    saliency_map: tf.Tensor,
    image_size: int,
) -> tf.Tensor:
    """Resize a saliency map and return it as a single-channel 2D tensor."""
    if saliency_map.shape.rank == 2:
        saliency_map = saliency_map[:, :, None]

    saliency_map = tf.image.resize(
        saliency_map,
        size=[
            image_size,
            image_size,
        ],
        method="bilinear",
    )

    saliency_map = tf.ensure_shape(
        saliency_map,
        [
            image_size,
            image_size,
            1,
        ],
    )

    saliency_map = saliency_map[:, :, 0]

    return saliency_map


def _uses_source_aspect_train_crop(
    dataset_name: str,
    recipe: str,
) -> bool:
    """Match the standard pipeline's source-aspect train-crop datasets."""
    return (
        is_imagenet100_name(
            dataset_name,
        )
        and recipe == "imagenet"
    ) or (
        is_cars196_name(
            dataset_name,
        )
        and recipe == "fine_grained"
    )


def _resize_saliency_map_to_image(
    saliency_map: tf.Tensor,
    image: tf.Tensor,
) -> tf.Tensor:
    """Resize a saliency map onto the source image pixel grid."""
    if saliency_map.shape.rank == 2:
        saliency_map = saliency_map[:, :, None]

    saliency_map = tf.image.resize(
        saliency_map,
        size=tf.shape(
            image,
        )[0:2],
        method="bilinear",
    )

    return saliency_map[:, :, 0]


def _build_saliency_map_dataset(
    saliency_maps: np.ndarray,
) -> tf.data.Dataset:
    """Read cached maps by index without materializing the full NPY in RAM."""
    map_shape = tuple(
        int(dimension)
        for dimension in saliency_maps.shape[1:]
    )
    output_dtype = tf.as_dtype(
        saliency_maps.dtype,
    )

    def read_numpy_map(
        index: np.ndarray,
    ) -> np.ndarray:
        """Copy one memory-mapped saliency entry into a TensorFlow buffer."""
        scalar_index = int(
            np.asarray(
                index,
            ).item()
        )
        return np.array(
            saliency_maps[scalar_index],
            copy=True,
        )

    def read_map(
        index: tf.Tensor,
    ) -> tf.Tensor:
        """Load one indexed map and restore its static spatial shape."""
        saliency_map = tf.numpy_function(
            read_numpy_map,
            [
                index,
            ],
            output_dtype,
        )
        return tf.ensure_shape(
            saliency_map,
            map_shape,
        )

    return tf.data.Dataset.range(
        int(saliency_maps.shape[0]),
    ).map(
        read_map,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )


def _normalize_image(
    image: tf.Tensor,
    dataset_name: str,
    tiny_imagenet_normalization: str = "imagenet",
) -> tf.Tensor:
    """Normalize an image with dataset-specific channel statistics."""
    if is_tiny_imagenet_name(
        dataset_name,
    ):
        return tiny_imagenet.normalize_float_image(
            image=image,
            normalization=tiny_imagenet_normalization,
        )

    if tfds_image.is_supported_tfds_image_dataset(
        dataset_name,
    ):
        return tfds_image.normalize_float_image(
            image=image,
            name=dataset_name,
        )

    return cifar.normalize_float_image(
        image=image,
        name=dataset_name,
    )


def _preprocess_salmix_train_example(
    example: dict[str, tf.Tensor],
    saliency_map: tf.Tensor,
    dataset_name: str,
    use_sal_basic_augmentation: bool,
    image_size: int,
    tiny_imagenet_normalization: str = "imagenet",
    saliency_augmentation_recipe: str | None = None,
    seed: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Preprocess one SaliencyMix / GuidedMixup training example.

    Input:
        example:
            Dataset example containing image and label.

        saliency_map:
            Precomputed saliency map aligned with example["image"].

        dataset_name:
            Dataset name, used for dataset-specific normalization.

        use_sal_basic_augmentation:
            Legacy switch for paired random crop / flip.

        image_size:
            Fixed model input size for the dataset.

        saliency_augmentation_recipe:
            Optional named paired augmentation recipe. Supports none, basic,
            and imagenet.

    Output:
        image:
            Normalized image.

        label:
            Integer class label.

        saliency_map:
            Saliency map after the same random crop / flip as image.
    """
    image = example["image"]
    label = example["label"]

    image = tf.cast(  # Convert uint8 pixels to [0, 1].
        image,
        tf.float32,
    ) / 255.0

    saliency_map_is_integer = saliency_map.dtype.is_integer
    saliency_map = tf.cast(
        saliency_map,
        tf.float32,
    )
    if saliency_map_is_integer:
        saliency_map = saliency_map / 255.0

    recipe = resolve_augmentation_recipe(
        use_basic_augmentation=use_sal_basic_augmentation,
        augmentation_recipe=saliency_augmentation_recipe,
    )

    if _uses_source_aspect_train_crop(
        dataset_name=dataset_name,
        recipe=recipe,
    ):
        # Keep the source aspect ratio so the paired random resized crop is
        # sampled on original geometry, matching the standard train pipeline
        # instead of squashing to a square first.
        saliency_map = _resize_saliency_map_to_image(
            saliency_map=saliency_map,
            image=image,
        )
    else:
        image = _resize_image(
            image=image,
            image_size=image_size,
        )

        saliency_map = _resize_saliency_map(
            saliency_map=saliency_map,
            image_size=image_size,
        )

    image, saliency_map = apply_sal_augmentation_recipe(
        image=image,
        saliency_map=saliency_map,
        image_size=image_size,
        use_sal_basic_augmentation=use_sal_basic_augmentation,
        saliency_augmentation_recipe=saliency_augmentation_recipe,
        seed=seed,
    )

    image = tf.ensure_shape(
        image,
        [
            image_size,
            image_size,
            3,
        ],
    )

    saliency_map = tf.ensure_shape(
        saliency_map,
        [
            image_size,
            image_size,
        ],
    )

    image = _normalize_image(
        image=image,
        dataset_name=dataset_name,
        tiny_imagenet_normalization=tiny_imagenet_normalization,
    )

    label = tf.cast(
        label,
        tf.int64,
    )

    return image, label, saliency_map


def build_salmix_train_pipeline(
    name: str,
    data_dir: str,
    batch_size: int,
    shuffle_buffer_size: int,
    drop_remainder: bool = True,
    use_sal_basic_augmentation: bool = True,
    saliency_dir: str = "./data",
    validation_split: float = 0.0,
    tiny_imagenet_normalization: str = "imagenet",
    saliency_augmentation_recipe: str | None = None,
    seed: int = 0,
    deterministic_data: bool = True,
    val_source: str = "train",
) -> tf.data.Dataset:
    """
    Build the SaliencyMix / GuidedMixup training pipeline.

    This pipeline returns:

        images, labels, saliency_maps

    Then engine/single/loop.py or engine/parallel/parallel_loop.py should
    convert the third element into:

        aux_info["saliency_maps"]
    """
    validate_validation_source(
        val_source=val_source,
    )
    validation_split = resolve_training_validation_split(
        validation_split=validation_split,
        val_source=val_source,
    )
    dataset_name = _canonical_dataset_name(
        name,
    )

    metadata = get_metadata(
        name,
    )

    train_ds = _load_train_dataset_without_file_shuffle(
        name=name,
        data_dir=data_dir,
    )

    saliency_maps = load_train_saliency_maps(
        dataset_name=dataset_name,
        saliency_dir=saliency_dir,
    )

    expected_saliency_maps = resolve_source_train_example_count(
        dataset_name=dataset_name,
        data_dir=data_dir,
        metadata=metadata,
    )
    if saliency_maps.shape[0] != expected_saliency_maps:
        raise ValueError(
            "Number of saliency maps does not match the source train split: "
            f"got {saliency_maps.shape[0]}, "
            f"expected {expected_saliency_maps}."
        )

    saliency_ds = _build_saliency_map_dataset(
        saliency_maps=saliency_maps,
    )

    train_ds = tf.data.Dataset.zip(
        (
            train_ds,
            saliency_ds,
        )
    )

    train_ds = split_train_validation_dataset(
        dataset=train_ds,
        validation_split=validation_split,
        keep_validation=False,
    )

    train_ds = train_ds.shuffle(
        buffer_size=shuffle_buffer_size,
        seed=seed if deterministic_data else None,
        reshuffle_each_iteration=True,
    )

    if deterministic_data:
        train_ds = attach_random_seed_stream(
            dataset=train_ds,
            seed=seed,
        )
        train_ds = train_ds.map(
            lambda example_and_map, random_value: (
                _preprocess_salmix_train_example(
                    example=example_and_map[0],
                    saliency_map=example_and_map[1],
                    dataset_name=dataset_name,
                    use_sal_basic_augmentation=use_sal_basic_augmentation,
                    image_size=metadata.image_size,
                    tiny_imagenet_normalization=(
                        tiny_imagenet_normalization
                    ),
                    saliency_augmentation_recipe=(
                        saliency_augmentation_recipe
                    ),
                    seed=make_stateless_seed(
                        base_seed=seed,
                        random_value=random_value,
                    ),
                )
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
    else:
        train_ds = train_ds.map(
            lambda example, saliency_map: _preprocess_salmix_train_example(
                example=example,
                saliency_map=saliency_map,
                dataset_name=dataset_name,
                use_sal_basic_augmentation=use_sal_basic_augmentation,
                image_size=metadata.image_size,
                tiny_imagenet_normalization=tiny_imagenet_normalization,
                saliency_augmentation_recipe=saliency_augmentation_recipe,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = train_ds.batch(
        batch_size=batch_size,
        drop_remainder=drop_remainder,
    )

    train_ds = train_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE,
    )

    return apply_dataset_determinism(
        dataset=train_ds,
        deterministic=deterministic_data,
    )


def build_salmix_validation_pipeline(
    name: str,
    data_dir: str,
    batch_size: int,
    validation_split: float,
    tiny_imagenet_normalization: str = "imagenet",
    val_source: str = "train",
) -> tf.data.Dataset:
    """Build validation from train (legacy) or the official eval source."""
    validate_validation_source(
        val_source=val_source,
    )
    preprocess_example = get_preprocessor(
        name,
        tiny_imagenet_normalization=tiny_imagenet_normalization,
    )

    if val_source == "test":
        validation_ds = load_test_dataset(
            name=name,
            data_dir=data_dir,
        )
    else:
        validation_ds = _load_train_dataset_without_file_shuffle(
            name=name,
            data_dir=data_dir,
        )

    validation_ds = split_train_validation_dataset(
        dataset=validation_ds,
        validation_split=validation_split,
        keep_validation=True,
    )

    validation_ds = validation_ds.map(
        lambda example: preprocess_example(
            example,
            False,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    validation_ds = validation_ds.batch(
        batch_size=batch_size,
        drop_remainder=False,
    )

    validation_ds = validation_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE,
    )

    return apply_dataset_determinism(
        dataset=validation_ds,
        deterministic=True,
    )


def build_salmix_test_pipeline(
    name: str,
    data_dir: str,
    batch_size: int,
    tiny_imagenet_normalization: str = "imagenet",
    val_source: str = "train",
    validation_split: float = 0.0,
) -> tf.data.Dataset:
    """
    Build normal test / validation pipeline.

    Evaluation does not need saliency maps.
    """
    validate_validation_source(
        val_source=val_source,
    )
    preprocess_example = get_preprocessor(
        name,
        tiny_imagenet_normalization=tiny_imagenet_normalization,
    )

    test_ds = load_test_dataset(
        name=name,
        data_dir=data_dir,
    )
    if val_source == "test" and validation_split > 0.0:
        test_ds = split_train_validation_dataset(
            dataset=test_ds,
            validation_split=validation_split,
            keep_validation=False,
        )

    test_ds = test_ds.map(
        preprocess_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    test_ds = test_ds.batch(
        batch_size=batch_size,
        drop_remainder=False,
    )

    test_ds = test_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE,
    )

    return apply_dataset_determinism(
        dataset=test_ds,
        deterministic=True,
    )


def build_salmix_dataset_pipeline(
    name: str,
    data_dir: str,
    batch_size: int,
    shuffle_buffer_size: int,
    drop_remainder: bool = True,
    use_sal_basic_augmentation: bool = True,
    saliency_dir: str = "./data",
    validation_split: float = 0.0,
    eval_on_test: bool = True,
    tiny_imagenet_normalization: str = "imagenet",
    saliency_augmentation_recipe: str | None = None,
    seed: int = 0,
    deterministic_data: bool = True,
    val_source: str = "train",
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build salmix dataset pipeline."""
    train_ds = build_salmix_train_pipeline(
        name=name,
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        drop_remainder=drop_remainder,
        use_sal_basic_augmentation=use_sal_basic_augmentation,
        saliency_dir=saliency_dir,
        validation_split=validation_split,
        tiny_imagenet_normalization=tiny_imagenet_normalization,
        saliency_augmentation_recipe=saliency_augmentation_recipe,
        seed=seed,
        deterministic_data=deterministic_data,
        val_source=val_source,
    )

    if validation_split > 0.0 and not eval_on_test:
        test_ds = build_salmix_validation_pipeline(
            name=name,
            data_dir=data_dir,
            batch_size=batch_size,
            validation_split=validation_split,
            tiny_imagenet_normalization=tiny_imagenet_normalization,
            val_source=val_source,
        )

    else:
        test_ds = build_salmix_test_pipeline(
            name=name,
            data_dir=data_dir,
            batch_size=batch_size,
            tiny_imagenet_normalization=tiny_imagenet_normalization,
            val_source=val_source,
            validation_split=validation_split,
        )

    return train_ds, test_ds
