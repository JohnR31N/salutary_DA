from __future__ import annotations

import tensorflow as tf

RandomSeed = tf.Tensor | None


FINE_GRAINED_RANDOM_RESIZED_CROP_SCALE = (
    0.6,
    1.0,
)
FINE_GRAINED_RANDOM_RESIZED_CROP_RATIO = (
    0.75,
    1.3333333333333333,
)

# Keep the public CUB constants available for existing callers.
CUB_RANDOM_RESIZED_CROP_SCALE = FINE_GRAINED_RANDOM_RESIZED_CROP_SCALE
CUB_RANDOM_RESIZED_CROP_RATIO = FINE_GRAINED_RANDOM_RESIZED_CROP_RATIO


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


def resolve_augmentation_recipe(
    use_basic_augmentation: bool,
    augmentation_recipe: str | None = None,
) -> str:
    """Resolve legacy basic_aug and the newer named augmentation recipe."""
    if augmentation_recipe:
        recipe = augmentation_recipe.lower()
    else:
        recipe = "basic" if use_basic_augmentation else "none"

    if recipe not in {
        "none",
        "basic",
        "hflip",
        "horizontal_flip",
        "cub",
        "fine_grained",
        "imagenet",
        "tiny_official",
        "tiny_openmixup",
    }:
        raise ValueError(f"Unsupported aug_recipe: {augmentation_recipe}")

    return recipe


def random_crop_with_padding(
    image: tf.Tensor,
    image_size: int,
    padding: int = 4,
    seed: RandomSeed = None,
) -> tf.Tensor:
    """Apply reflected-padding random crop at the requested image size."""
    image = tf.pad(
        image,
        paddings=[
            [padding, padding],
            [padding, padding],
            [0, 0],
        ],
        mode="REFLECT",
    )

    size = [
        image_size,
        image_size,
        3,
    ]

    if seed is None:
        return tf.image.random_crop(
            image,
            size=size,
        )

    return tf.image.stateless_random_crop(
        image,
        size=size,
        seed=seed,
    )


def random_resized_crop(
    image: tf.Tensor,
    image_size: int,
    method: str = "bilinear",
    scale: tuple[float, float] = (
        0.08,
        1.0,
    ),
    ratio: tuple[float, float] = (
        0.75,
        1.3333333333333333,
    ),
    seed: RandomSeed = None,
) -> tf.Tensor:
    """Apply a torchvision-style random resized crop."""
    offset_y, offset_x, crop_height, crop_width = sample_random_resized_crop_box(
        image_shape=tf.shape(
            image,
        ),
        scale=scale,
        ratio=ratio,
        seed=seed,
    )

    cropped = tf.slice(
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

    return tf.image.resize(
        cropped,
        size=[
            image_size,
            image_size,
        ],
        method=method,
    )


def sample_random_resized_crop_box(
    image_shape: tf.Tensor,
    scale: tuple[float, float] = (
        0.08,
        1.0,
    ),
    ratio: tuple[float, float] = (
        0.75,
        1.3333333333333333,
    ),
    max_attempts: int = 10,
    seed: RandomSeed = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Sample a crop box with torchvision RandomResizedCrop semantics."""
    image_height = image_shape[0]
    image_width = image_shape[1]

    image_area = tf.cast(
        image_height * image_width,
        tf.float32,
    )

    target_seed, ratio_seed, offset_y_seed, offset_x_seed = _split_seed(
        seed=seed,
        count=4,
    )
    uniform = (
        tf.random.uniform
        if seed is None
        else tf.random.stateless_uniform
    )
    target_kwargs = {}
    ratio_kwargs = {}
    if target_seed is not None:
        target_kwargs["seed"] = target_seed
        ratio_kwargs["seed"] = ratio_seed

    target_area = uniform(
        shape=[
            max_attempts,
        ],
        minval=scale[0],
        maxval=scale[1],
        dtype=tf.float32,
        **target_kwargs,
    ) * image_area

    log_ratio = uniform(
        shape=[
            max_attempts,
        ],
        minval=tf.math.log(
            tf.constant(
                ratio[0],
                dtype=tf.float32,
            )
        ),
        maxval=tf.math.log(
            tf.constant(
                ratio[1],
                dtype=tf.float32,
            )
        ),
        dtype=tf.float32,
        **ratio_kwargs,
    )

    aspect_ratio = tf.exp(
        log_ratio,
    )

    crop_widths = tf.cast(
        tf.round(
            tf.sqrt(
                target_area * aspect_ratio,
            )
        ),
        tf.int32,
    )

    crop_heights = tf.cast(
        tf.round(
            tf.sqrt(
                target_area / aspect_ratio,
            )
        ),
        tf.int32,
    )

    valid = (
        (crop_widths > 0)
        & (crop_widths <= image_width)
        & (crop_heights > 0)
        & (crop_heights <= image_height)
    )

    valid_index = tf.argmax(
        tf.cast(
            valid,
            tf.int32,
        ),
        output_type=tf.int32,
    )

    def use_random_box() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return the first valid random crop box."""
        crop_height = crop_heights[
            valid_index
        ]
        crop_width = crop_widths[
            valid_index
        ]

        max_offset_y = image_height - crop_height + 1
        max_offset_x = image_width - crop_width + 1

        offset_y_kwargs = {}
        offset_x_kwargs = {}
        if offset_y_seed is not None:
            offset_y_kwargs["seed"] = offset_y_seed
            offset_x_kwargs["seed"] = offset_x_seed

        offset_y = uniform(
            shape=[],
            minval=0,
            maxval=max_offset_y,
            dtype=tf.int32,
            **offset_y_kwargs,
        )
        offset_x = uniform(
            shape=[],
            minval=0,
            maxval=max_offset_x,
            dtype=tf.int32,
            **offset_x_kwargs,
        )

        return offset_y, offset_x, crop_height, crop_width

    def use_fallback_box() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return torchvision's centered fallback crop box."""
        in_ratio = tf.cast(
            image_width,
            tf.float32,
        ) / tf.cast(
            image_height,
            tf.float32,
        )

        min_ratio = tf.constant(
            ratio[0],
            dtype=tf.float32,
        )
        max_ratio = tf.constant(
            ratio[1],
            dtype=tf.float32,
        )

        def too_narrow() -> tuple[tf.Tensor, tf.Tensor]:
            crop_width = image_width
            crop_height = tf.cast(
                tf.round(
                    tf.cast(
                        crop_width,
                        tf.float32,
                    )
                    / min_ratio
                ),
                tf.int32,
            )

            return crop_height, crop_width

        def too_wide_or_full() -> tuple[tf.Tensor, tf.Tensor]:
            def too_wide() -> tuple[tf.Tensor, tf.Tensor]:
                crop_height = image_height
                crop_width = tf.cast(
                    tf.round(
                        tf.cast(
                            crop_height,
                            tf.float32,
                        )
                        * max_ratio
                    ),
                    tf.int32,
                )

                return crop_height, crop_width

            def full_image() -> tuple[tf.Tensor, tf.Tensor]:
                return image_height, image_width

            return tf.cond(
                in_ratio > max_ratio,
                too_wide,
                full_image,
            )

        crop_height, crop_width = tf.cond(
            in_ratio < min_ratio,
            too_narrow,
            too_wide_or_full,
        )

        crop_height = tf.minimum(
            crop_height,
            image_height,
        )
        crop_width = tf.minimum(
            crop_width,
            image_width,
        )

        offset_y = (image_height - crop_height) // 2
        offset_x = (image_width - crop_width) // 2

        return offset_y, offset_x, crop_height, crop_width

    return tf.cond(
        tf.reduce_any(
            valid,
        ),
        use_random_box,
        use_fallback_box,
    )


def apply_color_jitter(
    image: tf.Tensor,
    seed: RandomSeed = None,
) -> tf.Tensor:
    """Apply ImageNet-style color jitter to a float image in [0, 1]."""
    brightness_seed, contrast_seed, saturation_seed, hue_seed = _split_seed(
        seed=seed,
        count=4,
    )

    if seed is None:
        image = tf.image.random_brightness(
            image,
            max_delta=0.4,
        )
        image = tf.image.random_contrast(
            image,
            lower=0.6,
            upper=1.4,
        )
        image = tf.image.random_saturation(
            image,
            lower=0.6,
            upper=1.4,
        )
        image = tf.image.random_hue(
            image,
            max_delta=0.1,
        )
    else:
        image = tf.image.stateless_random_brightness(
            image,
            max_delta=0.4,
            seed=brightness_seed,
        )
        image = tf.image.stateless_random_contrast(
            image,
            lower=0.6,
            upper=1.4,
            seed=contrast_seed,
        )
        image = tf.image.stateless_random_saturation(
            image,
            lower=0.6,
            upper=1.4,
            seed=saturation_seed,
        )
        image = tf.image.stateless_random_hue(
            image,
            max_delta=0.1,
            seed=hue_seed,
        )

    return tf.clip_by_value(
        image,
        0.0,
        1.0,
    )


def apply_fine_grained_color_jitter(
    image: tf.Tensor,
    seed: RandomSeed = None,
) -> tf.Tensor:
    """Apply mild color jitter while preserving fine-grained visual cues."""
    brightness_seed, contrast_seed, saturation_seed, hue_seed = _split_seed(
        seed=seed,
        count=4,
    )

    if seed is None:
        image = tf.image.random_brightness(
            image,
            max_delta=0.1,
        )
        image = tf.image.random_contrast(
            image,
            lower=0.9,
            upper=1.1,
        )
        image = tf.image.random_saturation(
            image,
            lower=0.9,
            upper=1.1,
        )
        image = tf.image.random_hue(
            image,
            max_delta=0.02,
        )
    else:
        image = tf.image.stateless_random_brightness(
            image,
            max_delta=0.1,
            seed=brightness_seed,
        )
        image = tf.image.stateless_random_contrast(
            image,
            lower=0.9,
            upper=1.1,
            seed=contrast_seed,
        )
        image = tf.image.stateless_random_saturation(
            image,
            lower=0.9,
            upper=1.1,
            seed=saturation_seed,
        )
        image = tf.image.stateless_random_hue(
            image,
            max_delta=0.02,
            seed=hue_seed,
        )

    return tf.clip_by_value(
        image,
        0.0,
        1.0,
    )


def apply_cub_color_jitter(
    image: tf.Tensor,
    seed: RandomSeed = None,
) -> tf.Tensor:
    """Apply the legacy CUB alias of the fine-grained color jitter."""
    return apply_fine_grained_color_jitter(
        image,
        seed=seed,
    )


def apply_augmentation_recipe(
    image: tf.Tensor,
    image_size: int,
    use_basic_augmentation: bool,
    augmentation_recipe: str | None = None,
    seed: RandomSeed = None,
) -> tf.Tensor:
    """Apply a named train-time augmentation recipe."""
    recipe = resolve_augmentation_recipe(
        use_basic_augmentation=use_basic_augmentation,
        augmentation_recipe=augmentation_recipe,
    )

    if recipe == "none":
        return image

    crop_seed, flip_seed, jitter_seed = _split_seed(
        seed=seed,
        count=3,
    )

    if recipe == "basic":
        image = random_crop_with_padding(
            image=image,
            image_size=image_size,
            seed=crop_seed,
        )
        if flip_seed is None:
            return tf.image.random_flip_left_right(
                image,
            )
        return tf.image.stateless_random_flip_left_right(
            image,
            seed=flip_seed,
        )

    if recipe in {
        "hflip",
        "horizontal_flip",
        "tiny_official",
    }:
        if flip_seed is None:
            return tf.image.random_flip_left_right(
                image,
            )
        return tf.image.stateless_random_flip_left_right(
            image,
            seed=flip_seed,
        )

    if recipe == "tiny_openmixup":
        image = random_resized_crop(
            image=image,
            image_size=image_size,
            method="bicubic",
            seed=crop_seed,
        )
        image = tf.clip_by_value(
            image,
            0.0,
            1.0,
        )

        if flip_seed is None:
            return tf.image.random_flip_left_right(
                image,
            )
        return tf.image.stateless_random_flip_left_right(
            image,
            seed=flip_seed,
        )

    if recipe in {
        "cub",
        "fine_grained",
    }:
        image = random_resized_crop(
            image=image,
            image_size=image_size,
            scale=FINE_GRAINED_RANDOM_RESIZED_CROP_SCALE,
            ratio=FINE_GRAINED_RANDOM_RESIZED_CROP_RATIO,
            seed=crop_seed,
        )
        if flip_seed is None:
            image = tf.image.random_flip_left_right(
                image,
            )
        else:
            image = tf.image.stateless_random_flip_left_right(
                image,
                seed=flip_seed,
            )

        return apply_fine_grained_color_jitter(
            image,
            seed=jitter_seed,
        )

    image = random_resized_crop(
        image=image,
        image_size=image_size,
        seed=crop_seed,
    )
    if flip_seed is None:
        image = tf.image.random_flip_left_right(
            image,
        )
    else:
        image = tf.image.stateless_random_flip_left_right(
            image,
            seed=flip_seed,
        )
    image = apply_color_jitter(
        image,
        seed=jitter_seed,
    )

    return image
