from __future__ import annotations

import jax
import jax.numpy as jnp

from allthemix.methods.output import MixOutput
from allthemix.methods.utils.validation import (
    validate_labels_match_images,
    validate_nhwc_images,
    validate_num_classes,
    validate_probability,
    validate_scope_range,
)


def _resize_source_to_box_nearest(
    source_images: jnp.ndarray,
    x1: jnp.ndarray,
    y1: jnp.ndarray,
    x2: jnp.ndarray,
    y2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Resize each source image into a sampled paste box with nearest pixels."""
    batch_size = source_images.shape[0]
    image_height = source_images.shape[1]
    image_width = source_images.shape[2]

    x1 = jnp.broadcast_to(
        jnp.asarray(x1, dtype=jnp.int32),
        (
            batch_size,
        ),
    )
    y1 = jnp.broadcast_to(
        jnp.asarray(y1, dtype=jnp.int32),
        (
            batch_size,
        ),
    )
    x2 = jnp.broadcast_to(
        jnp.asarray(x2, dtype=jnp.int32),
        (
            batch_size,
        ),
    )
    y2 = jnp.broadcast_to(
        jnp.asarray(y2, dtype=jnp.int32),
        (
            batch_size,
        ),
    )

    y_positions = jnp.arange(
        image_height,
    )

    x_positions = jnp.arange(
        image_width,
    )

    grid_y = y_positions[:, None]
    grid_x = x_positions[None, :]

    def resize_one(
        source_image: jnp.ndarray,
        sample_x1: jnp.ndarray,
        sample_y1: jnp.ndarray,
        sample_x2: jnp.ndarray,
        sample_y2: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Resize one source image into one target box."""
        box_width = jnp.maximum(  # Keep the sampled box at least one pixel wide.
            sample_x2 - sample_x1,
            1,
        )

        box_height = jnp.maximum(  # Keep the sampled box at least one pixel tall.
            sample_y2 - sample_y1,
            1,
        )

        inside_y = jnp.logical_and(
            grid_y >= sample_y1,
            grid_y < sample_y2,
        )

        inside_x = jnp.logical_and(
            grid_x >= sample_x1,
            grid_x < sample_x2,
        )

        box_mask = jnp.logical_and(  # Mark the target pixels covered by the paste box.
            inside_y,
            inside_x,
        )

        relative_y = grid_y - sample_y1  # Target y coordinate relative to the box origin.
        relative_x = grid_x - sample_x1  # Target x coordinate relative to the box origin.

        source_y = jnp.floor(  # Project box y coordinates back into source image space.
            relative_y.astype(jnp.float32)
            * image_height
            / box_height.astype(jnp.float32)
        ).astype(jnp.int32)

        source_x = jnp.floor(  # Project box x coordinates back into source image space.
            relative_x.astype(jnp.float32)
            * image_width
            / box_width.astype(jnp.float32)
        ).astype(jnp.int32)

        source_y = jnp.clip(
            source_y,
            0,
            image_height - 1,
        )

        source_x = jnp.clip(
            source_x,
            0,
            image_width - 1,
        )

        resized_source_full = source_image[
            source_y,
            source_x,
            :,
        ]

        return resized_source_full, box_mask[:, :, None]

    return jax.vmap(
        resize_one,
    )(
        source_images,
        x1,
        y1,
        x2,
        y2,
    )


def resizemix(
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    scope_min: float = 0.1,
    scope_max: float = 0.8,
    prob: float = 1.0,
    per_sample: bool = False,
    paired_images: jnp.ndarray | None = None,
    paired_labels: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply ResizeMix by shrinking a paired image into a paste box."""
    validate_num_classes(
        num_classes,
    )
    validate_scope_range(
        min_name="resizemix_scope_min",
        min_value=scope_min,
        max_name="resizemix_scope_max",
        max_value=scope_max,
    )
    validate_probability(
        name="resizemix_prob",
        value=prob,
    )
    validate_nhwc_images(
        images=images,
        method_name="ResizeMix",
    )
    validate_labels_match_images(
        labels=labels,
        images=images,
        method_name="ResizeMix",
    )

    if (paired_images is None) != (paired_labels is None):
        raise ValueError(
            "ResizeMix paired_images and paired_labels must be provided together.",
        )

    if paired_images is not None:
        validate_nhwc_images(
            images=paired_images,
            method_name="ResizeMix paired",
        )
        validate_labels_match_images(
            labels=paired_labels,
            images=paired_images,
            method_name="ResizeMix paired",
        )

        if paired_images.shape != images.shape:
            raise ValueError(
                "ResizeMix paired_images must have the same shape as images. "
                f"Got {paired_images.shape} and {images.shape}.",
            )

        if paired_labels.shape != labels.shape:
            raise ValueError(
                "ResizeMix paired_labels must have the same shape as labels. "
                f"Got {paired_labels.shape} and {labels.shape}.",
            )

    batch_size = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]

    rng_tao, rng_perm, rng_cx, rng_cy, rng_prob = jax.random.split(
        rng,
        5,
    )

    tao_shape = (
        (batch_size,)
        if per_sample
        else ()
    )

    tao = jax.random.uniform(
        rng_tao,
        shape=tao_shape,
        minval=scope_min,
        maxval=scope_max,
    )

    do_resizemix = jax.random.bernoulli(
        rng_prob,
        p=prob,
        shape=(),
    )

    if paired_images is None:
        permutation = jax.random.permutation(
            rng_perm,
            batch_size,
        )

        source_images = images[permutation]
        source_labels = labels[permutation]

    else:
        del rng_perm
        permutation = jnp.arange(
            batch_size,
            dtype=jnp.int32,
        )
        source_images = paired_images
        source_labels = paired_labels

    cut_width = jnp.maximum(  # Sample paste width from tao and clamp to one pixel.
        (image_width * tao).astype(jnp.int32),
        1,
    )

    cut_height = jnp.maximum(  # Sample paste height from tao and clamp to one pixel.
        (image_height * tao).astype(jnp.int32),
        1,
    )

    center_x = jax.random.randint(
        rng_cx,
        shape=tao_shape,
        minval=0,
        maxval=image_width,
    )

    center_y = jax.random.randint(
        rng_cy,
        shape=tao_shape,
        minval=0,
        maxval=image_height,
    )

    x1 = jnp.clip(  # Clip the sampled box against the image boundary.
        center_x - cut_width // 2,
        0,
        image_width,
    )

    x2 = jnp.clip(  # Keep the clipped right edge inside the image.
        center_x + cut_width // 2,
        0,
        image_width,
    )

    y1 = jnp.clip(  # Clip the sampled box against the image boundary.
        center_y - cut_height // 2,
        0,
        image_height,
    )

    y2 = jnp.clip(  # Keep the clipped bottom edge inside the image.
        center_y + cut_height // 2,
        0,
        image_height,
    )

    x2 = jnp.minimum(  # Keep at least one pasted column after clipping.
        jnp.maximum(
            x2,
            x1 + 1,
        ),
        image_width,
    )

    y2 = jnp.minimum(  # Keep at least one pasted row after clipping.
        jnp.maximum(
            y2,
            y1 + 1,
        ),
        image_height,
    )

    resized_source_full, box_mask = _resize_source_to_box_nearest(
        source_images=source_images,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )

    resizemixed_images = jnp.where(  # Paste resized source pixels into the target box.
        box_mask,
        resized_source_full,
        images,
    )

    pasted_area = (x2 - x1) * (y2 - y1)  # Measure actual clipped paste area.

    adjusted_lam = 1.0 - (  # Convert pasted area into retained-target lambda.
        pasted_area.astype(jnp.float32)
        / float(image_height * image_width)
    )

    mixed_images = jnp.where(
        do_resizemix,
        resizemixed_images,
        images,
    )

    labels_a = labels

    labels_b = jnp.where(
        do_resizemix,
        source_labels,
        labels,
    )

    lam = jnp.where(
        do_resizemix,
        adjusted_lam,
        1.0,
    )

    return MixOutput(
        images=mixed_images,
        labels_a=labels_a,
        labels_b=labels_b,
        lam=lam,
        perm=permutation,
    )
