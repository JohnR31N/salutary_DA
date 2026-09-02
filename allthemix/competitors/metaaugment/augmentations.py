from __future__ import annotations

# Adapted from the Apache-2.0 MetaAugment implementation for the shared
# AllTheMix data/model/training stack; base preprocessing intentionally lives
# outside this module.
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, random

OP_NAMES: tuple[str, ...] = (
    "AutoContrast",
    "Equalize",
    "Rotate",
    "Posterize",
    "Solarize",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "Identity",
)
NUM_OPS = len(
    OP_NAMES,
)
NO_MAGNITUDE_OPS = jnp.asarray(
    [
        0,
        1,
        13,
    ],
    dtype=jnp.int32,
)


def initial_sampler_probs() -> jnp.ndarray:
    """Return a uniform distribution over ordered operation pairs."""
    pair_count = NUM_OPS * NUM_OPS

    return jnp.full(
        (
            NUM_OPS,
            NUM_OPS,
        ),
        1.0 / float(
            pair_count,
        ),
        dtype=jnp.float32,
    )


def _uses_magnitude(
    op_id: jnp.ndarray,
) -> jnp.ndarray:
    """Return whether each operation consumes a sampled magnitude."""
    return ~jnp.any(
        op_id[..., None] == NO_MAGNITUDE_OPS,
        axis=-1,
    )


def transformation_embedding(
    op1: jnp.ndarray,
    op2: jnp.ndarray,
    magnitude1: jnp.ndarray,
    magnitude2: jnp.ndarray,
) -> jnp.ndarray:
    """Encode an ordered operation pair with the paper's 28-D representation."""
    batch_size = op1.shape[0]
    rows = jnp.arange(
        batch_size,
    )
    value1 = jnp.where(
        _uses_magnitude(
            op1,
        ),
        magnitude1 + 1.0,
        11.0,
    )
    value2 = jnp.where(
        _uses_magnitude(
            op2,
        ),
        magnitude2 + 1.0,
        11.0,
    )
    embedding = jnp.zeros(
        (
            batch_size,
            NUM_OPS * 2,
        ),
        dtype=jnp.float32,
    )
    embedding = embedding.at[
        rows,
        op1 * 2,
    ].set(
        value1,
    )
    embedding = embedding.at[
        rows,
        op2 * 2 + 1,
    ].set(
        value2,
    )

    return embedding


def sample_transformations(
    key: jax.Array,
    sampler_probs: jnp.ndarray,
    batch_size: int,
    num_transforms_per_sample: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample ordered operation pairs and continuous magnitudes."""
    total = batch_size * num_transforms_per_sample
    key_ids, key_mag1, key_mag2 = random.split(
        key,
        3,
    )
    logits = jnp.log(
        jnp.reshape(
            sampler_probs,
            (-1,),
        )
        + 1.0e-8
    )
    pair_ids = random.categorical(
        key_ids,
        logits,
        shape=(
            total,
        ),
    )
    op1 = pair_ids // NUM_OPS
    op2 = pair_ids % NUM_OPS
    magnitude1 = random.uniform(
        key_mag1,
        (
            total,
        ),
        minval=0.0,
        maxval=10.0,
    )
    magnitude2 = random.uniform(
        key_mag2,
        (
            total,
        ),
        minval=0.0,
        maxval=10.0,
    )

    return op1, op2, magnitude1, magnitude2, pair_ids


def _random_signed_magnitude(
    magnitude: jnp.ndarray,
    key: jax.Array,
    max_value: float,
) -> jnp.ndarray:
    """Scale a level in [0, 10] and sample its direction."""
    sign = jnp.where(
        random.bernoulli(
            key,
        ),
        1.0,
        -1.0,
    )

    return sign * magnitude / 10.0 * max_value


def _clip(
    image: jnp.ndarray,
) -> jnp.ndarray:
    """Clip an image to the policy operation range."""
    return jnp.clip(
        image,
        0.0,
        1.0,
    )


def _rgb_to_luma(
    image: jnp.ndarray,
) -> jnp.ndarray:
    """Convert RGB values to one luma channel."""
    coefficients = jnp.asarray(
        [
            0.299,
            0.587,
            0.114,
        ],
        dtype=image.dtype,
    )

    return jnp.sum(
        image * coefficients,
        axis=-1,
        keepdims=True,
    )


def _blend(
    image1: jnp.ndarray,
    image2: jnp.ndarray,
    factor: jnp.ndarray,
) -> jnp.ndarray:
    """Linearly blend two images and clip the result."""
    return _clip(
        image1 + factor * (image2 - image1),
    )


def _auto_contrast(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Stretch every channel to its available intensity range."""
    del magnitude, key
    low = jnp.min(
        image,
        axis=(
            0,
            1,
        ),
        keepdims=True,
    )
    high = jnp.max(
        image,
        axis=(
            0,
            1,
        ),
        keepdims=True,
    )
    scale = jnp.where(
        high > low,
        1.0 / (high - low),
        1.0,
    )

    return _clip(
        (image - low) * scale,
    )


def _equalize_channel(
    channel: jnp.ndarray,
) -> jnp.ndarray:
    """Histogram-equalize one floating-point image channel."""
    values = jnp.clip(
        jnp.rint(
            channel * 255.0,
        ),
        0,
        255,
    ).astype(
        jnp.int32,
    )
    histogram = jnp.bincount(
        values.reshape(
            (-1,),
        ),
        length=256,
    )
    cumulative = jnp.cumsum(
        histogram,
    )
    nonzero = histogram > 0
    cumulative_min = jnp.min(
        jnp.where(
            nonzero,
            cumulative,
            cumulative[-1],
        )
    )
    denominator = jnp.maximum(
        cumulative[-1] - cumulative_min,
        1,
    )
    lookup = jnp.clip(
        jnp.rint(
            (cumulative - cumulative_min) * 255.0 / denominator,
        ),
        0,
        255,
    )

    return lookup[
        values
    ].astype(
        jnp.float32,
    ) / 255.0


def _equalize(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Histogram-equalize every image channel."""
    del magnitude, key

    return _clip(
        jax.vmap(
            _equalize_channel,
            in_axes=2,
            out_axes=2,
        )(
            image,
        )
    )


def _posterize(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Reduce the number of significant bits per channel."""
    del key
    bits = jnp.rint(
        8.0 - magnitude / 10.0 * 4.0,
    )
    shift = 8.0 - jnp.clip(
        bits,
        4.0,
        8.0,
    )
    scale = jnp.power(
        2.0,
        shift,
    )

    return jnp.floor(
        image * 255.0 / scale,
    ) * scale / 255.0


def _solarize(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Invert pixels above a magnitude-dependent threshold."""
    del key
    threshold = 1.0 - magnitude / 10.0

    return jnp.where(
        image < threshold,
        image,
        1.0 - image,
    )


def _color(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Adjust color saturation around a grayscale image."""
    factor = 1.0 + _random_signed_magnitude(
        magnitude,
        key,
        0.9,
    )
    gray = jnp.broadcast_to(
        _rgb_to_luma(
            image,
        ),
        image.shape,
    )

    return _blend(
        gray,
        image,
        factor,
    )


def _contrast(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Adjust contrast around the mean luma value."""
    factor = 1.0 + _random_signed_magnitude(
        magnitude,
        key,
        0.9,
    )
    mean = jnp.mean(
        _rgb_to_luma(
            image,
        ),
        axis=(
            0,
            1,
        ),
        keepdims=True,
    )

    return _clip(
        (image - mean) * factor + mean,
    )


def _brightness(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Scale image brightness."""
    factor = 1.0 + _random_signed_magnitude(
        magnitude,
        key,
        0.9,
    )

    return _clip(
        image * factor,
    )


def _blur(
    image: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the small smoothing kernel used by Sharpness."""
    padded = jnp.pad(
        image,
        (
            (
                1,
                1,
            ),
            (
                1,
                1,
            ),
            (
                0,
                0,
            ),
        ),
        mode="edge",
    )
    total = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1] * 5.0
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )

    return total / 13.0


def _sharpness(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Blend the image against a blurred reference."""
    factor = 1.0 + _random_signed_magnitude(
        magnitude,
        key,
        0.9,
    )

    return _blend(
        _blur(
            image,
        ),
        image,
        factor,
    )


def _affine(
    image: jnp.ndarray,
    matrix: jnp.ndarray,
    translate: jnp.ndarray,
) -> jnp.ndarray:
    """Sample an inverse affine warp with bilinear interpolation."""
    height, width = image.shape[:2]
    ys, xs = jnp.meshgrid(
        jnp.arange(
            height,
        ),
        jnp.arange(
            width,
        ),
        indexing="ij",
    )
    center = jnp.asarray(
        [
            (width - 1) / 2.0,
            (height - 1) / 2.0,
        ],
        dtype=jnp.float32,
    )
    coordinates = jnp.stack(
        [
            xs.astype(
                jnp.float32,
            ),
            ys.astype(
                jnp.float32,
            ),
        ],
        axis=-1,
    ) - center
    source = coordinates @ matrix.T + center + translate
    source_x = source[..., 0]
    source_y = source[..., 1]
    x0 = jnp.floor(
        source_x,
    ).astype(
        jnp.int32,
    )
    y0 = jnp.floor(
        source_y,
    ).astype(
        jnp.int32,
    )
    x1 = x0 + 1
    y1 = y0 + 1

    def gather(
        y: jnp.ndarray,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Gather pixels while filling coordinates outside the image."""
        in_bounds = (
            (x >= 0)
            & (x < width)
            & (y >= 0)
            & (y < height)
        )
        clipped_x = jnp.clip(
            x,
            0,
            width - 1,
        )
        clipped_y = jnp.clip(
            y,
            0,
            height - 1,
        )
        pixel = image[
            clipped_y,
            clipped_x,
        ]

        return jnp.where(
            in_bounds[..., None],
            pixel,
            0.5,
        )

    weight_a = (x1.astype(jnp.float32) - source_x) * (
        y1.astype(jnp.float32) - source_y
    )
    weight_b = (x1.astype(jnp.float32) - source_x) * (
        source_y - y0.astype(jnp.float32)
    )
    weight_c = (source_x - x0.astype(jnp.float32)) * (
        y1.astype(jnp.float32) - source_y
    )
    weight_d = (source_x - x0.astype(jnp.float32)) * (
        source_y - y0.astype(jnp.float32)
    )

    return _clip(
        gather(
            y0,
            x0,
        )
        * weight_a[..., None]
        + gather(
            y1,
            x0,
        )
        * weight_b[..., None]
        + gather(
            y0,
            x1,
        )
        * weight_c[..., None]
        + gather(
            y1,
            x1,
        )
        * weight_d[..., None]
    )


def _rotate(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Rotate an image by at most 30 degrees."""
    angle = -jnp.deg2rad(
        _random_signed_magnitude(
            magnitude,
            key,
            30.0,
        )
    )
    cosine = jnp.cos(
        angle,
    )
    sine = jnp.sin(
        angle,
    )
    matrix = jnp.stack(
        [
            jnp.stack(
                [
                    cosine,
                    -sine,
                ]
            ),
            jnp.stack(
                [
                    sine,
                    cosine,
                ]
            ),
        ]
    ).astype(
        jnp.float32,
    )

    return _affine(
        image,
        matrix,
        jnp.zeros(
            (2,),
            dtype=jnp.float32,
        ),
    )


def _shear_x(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Shear an image along its x axis."""
    shear = _random_signed_magnitude(
        magnitude,
        key,
        0.3,
    )
    matrix = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.asarray(
                        1.0,
                        dtype=jnp.float32,
                    ),
                    -shear,
                ]
            ),
            jnp.stack(
                [
                    jnp.asarray(
                        0.0,
                        dtype=jnp.float32,
                    ),
                    jnp.asarray(
                        1.0,
                        dtype=jnp.float32,
                    ),
                ]
            ),
        ]
    )

    return _affine(
        image,
        matrix,
        jnp.zeros(
            (2,),
            dtype=jnp.float32,
        ),
    )


def _shear_y(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Shear an image along its y axis."""
    shear = _random_signed_magnitude(
        magnitude,
        key,
        0.3,
    )
    matrix = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.asarray(
                        1.0,
                        dtype=jnp.float32,
                    ),
                    jnp.asarray(
                        0.0,
                        dtype=jnp.float32,
                    ),
                ]
            ),
            jnp.stack(
                [
                    -shear,
                    jnp.asarray(
                        1.0,
                        dtype=jnp.float32,
                    ),
                ]
            ),
        ]
    )

    return _affine(
        image,
        matrix,
        jnp.zeros(
            (2,),
            dtype=jnp.float32,
        ),
    )


def _translate_x(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
    translate_const: float,
) -> jnp.ndarray:
    """Translate an image horizontally."""
    shift = -_random_signed_magnitude(
        magnitude,
        key,
        translate_const,
    )

    return _affine(
        image,
        jnp.eye(
            2,
            dtype=jnp.float32,
        ),
        jnp.stack(
            [
                shift,
                jnp.asarray(
                    0.0,
                    dtype=jnp.float32,
                ),
            ]
        ),
    )


def _translate_y(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
    translate_const: float,
) -> jnp.ndarray:
    """Translate an image vertically."""
    shift = -_random_signed_magnitude(
        magnitude,
        key,
        translate_const,
    )

    return _affine(
        image,
        jnp.eye(
            2,
            dtype=jnp.float32,
        ),
        jnp.stack(
            [
                jnp.asarray(
                    0.0,
                    dtype=jnp.float32,
                ),
                shift,
            ]
        ),
    )


def _identity(
    image: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Return an image unchanged."""
    del magnitude, key

    return image


def apply_op(
    image: jnp.ndarray,
    op_id: jnp.ndarray,
    magnitude: jnp.ndarray,
    key: jax.Array,
    translate_const: float,
) -> jnp.ndarray:
    """Dispatch one sampled policy operation."""
    branches = (
        _auto_contrast,
        _equalize,
        _rotate,
        _posterize,
        _solarize,
        _color,
        _contrast,
        _brightness,
        _sharpness,
        _shear_x,
        _shear_y,
        partial(
            _translate_x,
            translate_const=translate_const,
        ),
        partial(
            _translate_y,
            translate_const=translate_const,
        ),
        _identity,
    )

    return lax.switch(
        op_id,
        branches,
        image,
        magnitude,
        key,
    )


def cutout(
    images: jnp.ndarray,
    key: jax.Array,
    size: int,
) -> jnp.ndarray:
    """Fill a sampled square in every image with middle gray."""
    if size <= 0:
        return images

    keys = random.split(
        key,
        images.shape[0],
    )
    height, width = images.shape[1:3]
    half = size // 2
    yy, xx = jnp.meshgrid(
        jnp.arange(
            height,
        ),
        jnp.arange(
            width,
        ),
        indexing="ij",
    )

    def apply(
        image: jnp.ndarray,
        image_key: jax.Array,
    ) -> jnp.ndarray:
        """Apply one independently located Cutout square."""
        y_key, x_key = random.split(
            image_key,
        )
        y = random.randint(
            y_key,
            (),
            0,
            height,
        )
        x = random.randint(
            x_key,
            (),
            0,
            width,
        )
        y_start = y - half
        x_start = x - half
        y_end = y_start + size
        x_end = x_start + size
        keep = (
            (yy < y_start)
            | (yy >= y_end)
            | (xx < x_start)
            | (xx >= x_end)
        )

        return jnp.where(
            keep[..., None],
            image,
            0.5,
        )

    return jax.vmap(
        apply,
    )(
        images,
        keys,
    )


def apply_metaaugment(
    images: jnp.ndarray,
    labels: jnp.ndarray,
    key: jax.Array,
    sampler_probs: jnp.ndarray,
    *,
    num_transforms_per_sample: int,
    cutout_size: int,
    translate_const: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply sampled policy pairs to already base-augmented [0, 1] images."""
    key_sample, key_op1, key_op2, key_cutout = random.split(
        key,
        4,
    )
    batch_size = images.shape[0]
    op1, op2, magnitude1, magnitude2, pair_ids = sample_transformations(
        key=key_sample,
        sampler_probs=sampler_probs,
        batch_size=batch_size,
        num_transforms_per_sample=num_transforms_per_sample,
    )
    repeated_images = jnp.repeat(
        images,
        num_transforms_per_sample,
        axis=0,
    )
    repeated_labels = jnp.repeat(
        labels,
        num_transforms_per_sample,
        axis=0,
    )
    keys1 = random.split(
        key_op1,
        repeated_images.shape[0],
    )
    keys2 = random.split(
        key_op2,
        repeated_images.shape[0],
    )
    augmented = jax.vmap(
        apply_op,
        in_axes=(
            0,
            0,
            0,
            0,
            None,
        ),
    )(
        repeated_images,
        op1,
        magnitude1,
        keys1,
        translate_const,
    )
    augmented = jax.vmap(
        apply_op,
        in_axes=(
            0,
            0,
            0,
            0,
            None,
        ),
    )(
        augmented,
        op2,
        magnitude2,
        keys2,
        translate_const,
    )
    augmented = cutout(
        images=augmented,
        key=key_cutout,
        size=cutout_size,
    )
    embedding = transformation_embedding(
        op1=op1,
        op2=op2,
        magnitude1=magnitude1,
        magnitude2=magnitude2,
    )

    return augmented, repeated_labels, embedding, pair_ids
