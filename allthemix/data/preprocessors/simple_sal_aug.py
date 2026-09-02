from __future__ import annotations

import tensorflow as tf

from allthemix.data.preprocessors.augmentation import (
    CUB_RANDOM_RESIZED_CROP_RATIO,
    CUB_RANDOM_RESIZED_CROP_SCALE,
    FINE_GRAINED_RANDOM_RESIZED_CROP_RATIO,
    FINE_GRAINED_RANDOM_RESIZED_CROP_SCALE,
    apply_color_jitter,
    apply_cub_color_jitter,
    apply_fine_grained_color_jitter,
    resolve_augmentation_recipe,
    sample_random_resized_crop_box,
)

RandomSeed = tf.Tensor | None


def _split_seed(
    seed: RandomSeed,
    count: int,
) -> list[RandomSeed]:
    """Split an optional stateless seed into independent operation seeds."""
    if seed is None:
        return [
            None,
        ] * count

    return list(
        tf.unstack(
            tf.random.experimental.stateless_split(
                seed=tf.convert_to_tensor(
                    seed,
                ),
                num=count,
            ),
            num=count,
        )
    )


def _ensure_saliency_channel(
    saliency_map: tf.Tensor,
) -> tf.Tensor:
    """Ensure a saliency map has an explicit channel dimension."""
    if saliency_map.shape.rank == 2:
        saliency_map = saliency_map[:, :, None]

    return saliency_map


def _remove_saliency_channel(
    saliency_map: tf.Tensor,
) -> tf.Tensor:
    """Remove a singleton saliency-map channel dimension."""
    if saliency_map.shape.rank != 3:
        return saliency_map

    last_dimension = saliency_map.shape[-1]

    if last_dimension == 1:
        return saliency_map[:, :, 0]

    if last_dimension is None:
        # Graph tracing loses the channel size after dynamic-size tf.slice;
        # the maps are single-channel by construction, so squeeze checks it.
        return tf.squeeze(
            saliency_map,
            axis=-1,
        )

    return saliency_map


def random_crop_with_padding_pair(
    image: tf.Tensor,
    saliency_map: tf.Tensor,
    padding: int = 4,
    image_size: int = 32,
    seed: RandomSeed = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply the same reflected-pad random crop to image and saliency map."""
    saliency_map = _ensure_saliency_channel(
        saliency_map,
    )

    image = tf.pad(
        image,
        paddings=[
            [padding, padding],
            [padding, padding],
            [0, 0],
        ],
        mode="REFLECT",
    )

    saliency_map = tf.pad(
        saliency_map,
        paddings=[
            [padding, padding],
            [padding, padding],
            [0, 0],
        ],
        mode="REFLECT",
    )

    padded_size = image_size + 2 * padding  # Size after symmetric reflected padding.

    y_seed, x_seed = _split_seed(
        seed=seed,
        count=2,
    )
    if seed is None:
        offset_y = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=padded_size - image_size + 1,
            dtype=tf.int32,
        )
        offset_x = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=padded_size - image_size + 1,
            dtype=tf.int32,
        )
    else:
        offset_y = tf.random.stateless_uniform(
            shape=[],
            seed=y_seed,
            minval=0,
            maxval=padded_size - image_size + 1,
            dtype=tf.int32,
        )
        offset_x = tf.random.stateless_uniform(
            shape=[],
            seed=x_seed,
            minval=0,
            maxval=padded_size - image_size + 1,
            dtype=tf.int32,
        )

    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_y,
        offset_width=offset_x,
        target_height=image_size,
        target_width=image_size,
    )

    saliency_map = tf.image.crop_to_bounding_box(
        saliency_map,
        offset_height=offset_y,
        offset_width=offset_x,
        target_height=image_size,
        target_width=image_size,
    )

    saliency_map = _remove_saliency_channel(
        saliency_map,
    )

    return image, saliency_map


def random_horizontal_flip_pair(
    image: tf.Tensor,
    saliency_map: tf.Tensor,
    seed: RandomSeed = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply the same random horizontal flip to image and saliency map."""
    if seed is None:
        random_value = tf.random.uniform(
            shape=[],
            minval=0.0,
            maxval=1.0,
            dtype=tf.float32,
        )
    else:
        random_value = tf.random.stateless_uniform(
            shape=[],
            seed=seed,
            minval=0.0,
            maxval=1.0,
            dtype=tf.float32,
        )

    do_flip = random_value < 0.5

    image = tf.cond(
        do_flip,
        lambda: tf.image.flip_left_right(image),
        lambda: image,
    )

    saliency_map = tf.cond(
        do_flip,
        lambda: tf.reverse(
            saliency_map,
            axis=[1],
        ),
        lambda: saliency_map,
    )

    return image, saliency_map


def random_resized_crop_pair(
    image: tf.Tensor,
    saliency_map: tf.Tensor,
    image_size: int,
    image_resize_method: str = "bilinear",
    scale: tuple[float, float] = (
        0.08,
        1.0,
    ),
    ratio: tuple[float, float] = (
        0.75,
        1.3333333333333333,
    ),
    seed: RandomSeed = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply the same ImageNet-style random resized crop to image and saliency."""
    saliency_map = _ensure_saliency_channel(
        saliency_map,
    )

    offset_y, offset_x, crop_height, crop_width = sample_random_resized_crop_box(
        image_shape=tf.shape(
            image,
        ),
        scale=scale,
        ratio=ratio,
        seed=seed,
    )

    cropped_image = tf.slice(  # Crop image with the sampled box.
        image,
        [
            offset_y,
            offset_x,
            0,
        ],
        [
            crop_height,
            crop_width,
            -1,
        ],
    )

    cropped_saliency = tf.slice(  # Crop saliency with the same sampled box.
        saliency_map,
        [
            offset_y,
            offset_x,
            0,
        ],
        [
            crop_height,
            crop_width,
            -1,
        ],
    )

    image = tf.image.resize(
        cropped_image,
        size=[
            image_size,
            image_size,
        ],
        method=image_resize_method,
    )

    saliency_map = tf.image.resize(
        cropped_saliency,
        size=[
            image_size,
            image_size,
        ],
        method="bilinear",
    )

    saliency_map = _remove_saliency_channel(
        saliency_map,
    )

    return image, saliency_map


def apply_sal_basic_aug(
    image: tf.Tensor,
    saliency_map: tf.Tensor,
    image_size: int = 32,
    seed: RandomSeed = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply paired crop and flip augmentation for saliency-based methods."""
    crop_seed, flip_seed = _split_seed(
        seed=seed,
        count=2,
    )
    image, saliency_map = random_crop_with_padding_pair(
        image=image,
        saliency_map=saliency_map,
        padding=4,
        image_size=image_size,
        seed=crop_seed,
    )

    image, saliency_map = random_horizontal_flip_pair(
        image=image,
        saliency_map=saliency_map,
        seed=flip_seed,
    )

    return image, saliency_map


def apply_sal_augmentation_recipe(
    image: tf.Tensor,
    saliency_map: tf.Tensor,
    image_size: int,
    use_sal_basic_augmentation: bool,
    saliency_augmentation_recipe: str | None = None,
    seed: RandomSeed = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply a named paired augmentation recipe for saliency-based methods."""
    recipe = resolve_augmentation_recipe(
        use_basic_augmentation=use_sal_basic_augmentation,
        augmentation_recipe=saliency_augmentation_recipe,
    )

    if recipe == "none":
        return image, saliency_map

    crop_seed, flip_seed, jitter_seed = _split_seed(
        seed=seed,
        count=3,
    )

    if recipe == "basic":
        return apply_sal_basic_aug(
            image=image,
            saliency_map=saliency_map,
            image_size=image_size,
            seed=seed,
        )

    if recipe in {
        "hflip",
        "horizontal_flip",
        "tiny_official",
    }:
        return random_horizontal_flip_pair(
            image=image,
            saliency_map=saliency_map,
            seed=flip_seed,
        )

    if recipe == "tiny_openmixup":
        image, saliency_map = random_resized_crop_pair(
            image=image,
            saliency_map=saliency_map,
            image_size=image_size,
            image_resize_method="bicubic",
            seed=crop_seed,
        )
        image = tf.clip_by_value(
            image,
            0.0,
            1.0,
        )

        return random_horizontal_flip_pair(
            image=image,
            saliency_map=saliency_map,
            seed=flip_seed,
        )

    if recipe == "cub":
        image, saliency_map = random_resized_crop_pair(
            image=image,
            saliency_map=saliency_map,
            image_size=image_size,
            scale=CUB_RANDOM_RESIZED_CROP_SCALE,
            ratio=CUB_RANDOM_RESIZED_CROP_RATIO,
            seed=crop_seed,
        )
        image, saliency_map = random_horizontal_flip_pair(
            image=image,
            saliency_map=saliency_map,
            seed=flip_seed,
        )
        if jitter_seed is None:
            image = apply_cub_color_jitter(
                image,
            )
        else:
            image = apply_cub_color_jitter(
                image,
                seed=jitter_seed,
            )

        return image, saliency_map

    if recipe == "fine_grained":
        image, saliency_map = random_resized_crop_pair(
            image=image,
            saliency_map=saliency_map,
            image_size=image_size,
            scale=FINE_GRAINED_RANDOM_RESIZED_CROP_SCALE,
            ratio=FINE_GRAINED_RANDOM_RESIZED_CROP_RATIO,
            seed=crop_seed,
        )
        image, saliency_map = random_horizontal_flip_pair(
            image=image,
            saliency_map=saliency_map,
            seed=flip_seed,
        )
        if jitter_seed is None:
            image = apply_fine_grained_color_jitter(
                image,
            )
        else:
            image = apply_fine_grained_color_jitter(
                image,
                seed=jitter_seed,
            )

        return image, saliency_map

    image, saliency_map = random_resized_crop_pair(
        image=image,
        saliency_map=saliency_map,
        image_size=image_size,
        seed=crop_seed,
    )
    image, saliency_map = random_horizontal_flip_pair(
        image=image,
        saliency_map=saliency_map,
        seed=flip_seed,
    )
    image = apply_color_jitter(  # Jitter image appearance without moving saliency.
        image,
        seed=jitter_seed,
    )

    return image, saliency_map
