from __future__ import annotations

import jax
import jax.numpy as jnp

from allthemix.methods.output import MixOutput
from allthemix.methods.utils.validation import (
    validate_labels_match_images,
    validate_nhwc_images,
    validate_no_repeat_batch_size,
    validate_num_classes,
    validate_positive,
    validate_probability,
)

CUTMIX_VARIANTS = (
    "standard",
    "area_adjusted",
    "torchbearer",
    "torchbearer_area",
    "torchbearer_inside",
)


def normalize_cutmix_variant(
    variant: str,
) -> str:
    """Normalize CutMix variant names."""
    variant = variant.lower().replace(
        "-",
        "_",
    ).replace(
        " ",
        "_",
    )

    if variant in {
        "default",
        "area",
        "adjusted",
        "standard",
    }:
        return "standard"

    if variant in {
        "area_adjusted",
        "actual_area",
    }:
        return "standard"

    if variant in {
        "torchbearer",
        "fmix_repo",
        "torchbearer_sampled",
        "fmix_repo_sampled",
    }:
        return "torchbearer"

    if variant in {
        "torchbearer_area",
        "torchbearer_actual_area",
        "fmix_repo_area",
        "clipped_area",
    }:
        return "torchbearer_area"

    if variant in {
        "torchbearer_inside",
        "inside",
        "inside_box",
        "shift_inside",
        "shifted_inside",
    }:
        return "torchbearer_inside"

    raise ValueError(
        "cutmix_variant must be one of "
        f"{CUTMIX_VARIANTS}. Got {variant}.",
    )


def _no_repeat_permutation(
    rng: jax.Array,
    batch_size: int,
) -> jnp.ndarray:
    """Create a permutation without fixed points when possible."""
    if batch_size <= 1:
        return jnp.arange(
            batch_size,
        )

    order = jax.random.permutation(
        rng,
        batch_size,
    )

    shifted_order = jnp.roll(  # Shift the cycle so each ordered index maps elsewhere.
        order,
        1,
    )

    return jnp.arange(
        batch_size,
    ).at[
        order
    ].set(
        shifted_order,
    )


def _build_batch_box_mask(
    image_height: int,
    image_width: int,
    x1: jnp.ndarray,
    y1: jnp.ndarray,
    x2: jnp.ndarray,
    y2: jnp.ndarray,
) -> jnp.ndarray:
    """Build one shared rectangle mask for the full batch."""
    x_positions = jnp.arange(image_width)
    y_positions = jnp.arange(image_height)

    x_mask = jnp.logical_and(
        x_positions >= x1,
        x_positions < x2,
    )

    y_mask = jnp.logical_and(
        y_positions >= y1,
        y_positions < y2,
    )

    box_mask = jnp.logical_and(
        y_mask[:, None],
        x_mask[None, :],
    )

    return box_mask[None, :, :, None]


def _build_per_sample_box_mask(
    image_height: int,
    image_width: int,
    x1: jnp.ndarray,
    y1: jnp.ndarray,
    x2: jnp.ndarray,
    y2: jnp.ndarray,
) -> jnp.ndarray:
    """Build one rectangle mask per sample."""
    y_positions = jnp.arange(
        image_height,
    )[None, :, None]
    x_positions = jnp.arange(
        image_width,
    )[None, None, :]

    x_mask = jnp.logical_and(
        x_positions >= x1[:, None, None],
        x_positions < x2[:, None, None],
    )
    y_mask = jnp.logical_and(
        y_positions >= y1[:, None, None],
        y_positions < y2[:, None, None],
    )

    return jnp.logical_and(
        y_mask,
        x_mask,
    )[:, :, :, None]


def cutmix(
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    alpha: float = 1.0,
    prob: float = 1.0,
    no_repeat: bool = False,
    paired_images: jnp.ndarray | None = None,
    paired_labels: jnp.ndarray | None = None,
    paired_perm: jnp.ndarray | None = None,
    variant: str = "standard",
    per_sample_lam: bool = False,
    min_lam: float = 0.0,
) -> MixOutput:
    """Apply CutMix by pasting a random paired image rectangle."""
    variant = normalize_cutmix_variant(
        variant,
    )
    is_torchbearer_variant = variant in {
        "torchbearer",
        "torchbearer_area",
        "torchbearer_inside",
    }
    if per_sample_lam and not is_torchbearer_variant:
        raise ValueError(
            "cutmix_per_sample_lam currently requires "
            "a torchbearer-style cutmix_variant.",
        )
    validate_num_classes(
        num_classes,
    )
    validate_positive(
        name="cutmix_alpha",
        value=alpha,
    )
    validate_probability(
        name="cutmix_prob",
        value=prob,
    )
    validate_probability(
        name="cutmix_min_lam",
        value=min_lam,
    )
    validate_nhwc_images(
        images=images,
        method_name="CutMix",
    )
    validate_labels_match_images(
        labels=labels,
        images=images,
        method_name="CutMix",
    )

    if (paired_images is None) != (paired_labels is None):
        raise ValueError(
            "CutMix paired_images and paired_labels must be provided together.",
        )

    if paired_images is not None:
        validate_nhwc_images(
            images=paired_images,
            method_name="CutMix paired",
        )
        validate_labels_match_images(
            labels=paired_labels,
            images=paired_images,
            method_name="CutMix paired",
        )

        if paired_images.shape != images.shape:
            raise ValueError(
                "CutMix paired_images must have the same shape as images. "
                f"Got {paired_images.shape} and {images.shape}.",
            )

        if paired_labels.shape != labels.shape:
            raise ValueError(
                "CutMix paired_labels must have the same shape as labels. "
                f"Got {paired_labels.shape} and {labels.shape}.",
            )

    batch_size = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]

    if no_repeat and paired_images is None:
        validate_no_repeat_batch_size(
            batch_size=batch_size,
            method_name="CutMix",
        )

    rng_lam, rng_perm, rng_cx, rng_cy, rng_prob = jax.random.split(rng, 5)

    lam_shape = (
        (batch_size,)
        if per_sample_lam
        else ()
    )

    lam = jax.random.beta(  # Sample retained-area ratio from Beta(alpha, alpha).
        rng_lam,
        alpha,
        alpha,
        shape=lam_shape,
    )

    lam = jnp.maximum(  # Keep source-A evidence above the requested floor.
        lam,
        jnp.asarray(
            min_lam,
            dtype=images.dtype,
        ),
    )

    do_cutmix = jax.random.bernoulli(
        rng_prob,
        p=prob,
        shape=(),
    )

    if paired_images is None and no_repeat:
        permutation = _no_repeat_permutation(
            rng_perm,
            batch_size,
        )

    elif paired_images is None:
        permutation = jax.random.permutation(
            rng_perm,
            batch_size,
        )

    else:
        del rng_perm
        if paired_perm is None:
            paired_perm = jnp.arange(
                batch_size,
                dtype=jnp.int32,
            )
        permutation = paired_perm

    identity_permutation = jnp.arange(batch_size)

    if paired_images is None:
        shuffled_images = images[permutation]
        shuffled_labels = labels[permutation]

    else:
        shuffled_images = paired_images
        shuffled_labels = paired_labels

    cut_ratio = jnp.sqrt(1.0 - lam)  # Convert patch area ratio to side-length ratio.

    if is_torchbearer_variant:
        cut_width = jnp.round(  # Torchbearer rounds the sampled patch width.
            image_width * cut_ratio,
        ).astype(jnp.int32)
        cut_height = jnp.round(  # Torchbearer rounds the sampled patch height.
            image_height * cut_ratio,
        ).astype(jnp.int32)

        if min_lam > 0.0:
            max_cut_ratio = jnp.sqrt(
                1.0
                - jnp.asarray(
                    min_lam,
                    dtype=images.dtype,
                )
            )
            max_cut_width = jnp.maximum(
                jnp.floor(
                    image_width * max_cut_ratio,
                ).astype(jnp.int32),
                1,
            )
            max_cut_height = jnp.maximum(
                jnp.floor(
                    image_height * max_cut_ratio,
                ).astype(jnp.int32),
                1,
            )
            cut_width = jnp.minimum(
                cut_width,
                max_cut_width,
            )
            cut_height = jnp.minimum(
                cut_height,
                max_cut_height,
            )

    else:
        cut_width = (image_width * cut_ratio).astype(jnp.int32)  # Patch width in pixels.
        cut_height = (image_height * cut_ratio).astype(jnp.int32)  # Patch height in pixels.

    if is_torchbearer_variant:
        if not per_sample_lam:
            cut_width = jnp.full(
                (
                    batch_size,
                ),
                cut_width,
                dtype=jnp.int32,
            )
            cut_height = jnp.full(
                (
                    batch_size,
                ),
                cut_height,
                dtype=jnp.int32,
            )

        center_x = jax.random.randint(
            rng_cx,
            shape=(batch_size,),
            minval=0,
            maxval=image_width,
        )

        center_y = jax.random.randint(
            rng_cy,
            shape=(batch_size,),
            minval=0,
            maxval=image_height,
        )

    else:
        center_x = jax.random.randint(
            rng_cx,
            shape=(),
            minval=0,
            maxval=image_width,
        )

        center_y = jax.random.randint(
            rng_cy,
            shape=(),
            minval=0,
            maxval=image_height,
        )

    if variant == "torchbearer_inside":
        max_x1 = jnp.maximum(  # Last valid x origin that keeps the patch inside.
            image_width - cut_width,
            0,
        )
        max_y1 = jnp.maximum(  # Last valid y origin that keeps the patch inside.
            image_height - cut_height,
            0,
        )

        x1 = jnp.clip(  # Shift the sampled box into the image instead of clipping it.
            center_x - cut_width // 2,
            0,
            max_x1,
        )
        y1 = jnp.clip(  # Shift the sampled box into the image instead of clipping it.
            center_y - cut_height // 2,
            0,
            max_y1,
        )

        x2 = x1 + cut_width  # Preserve the sampled patch width.
        y2 = y1 + cut_height  # Preserve the sampled patch height.

    elif is_torchbearer_variant:
        x1 = jnp.clip(  # Clip the Torchbearer box against image bounds.
            center_x - cut_width // 2,
            0,
            image_width,
        )

        x2 = jnp.clip(  # Keep clipped right edge within the image.
            center_x + cut_width // 2,
            0,
            image_width,
        )

        y1 = jnp.clip(  # Clip the Torchbearer box against image bounds.
            center_y - cut_height // 2,
            0,
            image_height,
        )

        y2 = jnp.clip(  # Keep clipped bottom edge within the image.
            center_y + cut_height // 2,
            0,
            image_height,
        )

    else:
        x1 = jnp.clip(
            center_x - cut_width // 2,
            0,
            image_width,
        )

        x2 = jnp.clip(
            center_x + cut_width // 2,
            0,
            image_width,
        )

        y1 = jnp.clip(
            center_y - cut_height // 2,
            0,
            image_height,
        )

        y2 = jnp.clip(
            center_y + cut_height // 2,
            0,
            image_height,
        )

    if is_torchbearer_variant:
        box_mask = _build_per_sample_box_mask(
            image_height=image_height,
            image_width=image_width,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

    else:
        box_mask = _build_batch_box_mask(
            image_height=image_height,
            image_width=image_width,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

    cutmixed_images = jnp.where(  # Paste paired pixels inside the rectangle.
        box_mask,
        shuffled_images,
        images,
    )

    patch_area = (x2 - x1) * (y2 - y1)  # Measure clipped rectangle area.

    adjusted_lam = 1.0 - (  # Recompute lambda from the actual retained area.
        patch_area.astype(jnp.float32)
        / float(image_height * image_width)
    )

    if variant == "torchbearer" and not per_sample_lam and min_lam <= 0.0:
        adjusted_lam = lam

    mixed_images = jnp.where(
        do_cutmix,
        cutmixed_images,
        images,
    )

    lam = jnp.where(
        do_cutmix,
        adjusted_lam,
        1.0,
    )

    labels_a = labels

    labels_b = jnp.where(
        do_cutmix,
        shuffled_labels,
        labels,
    )

    perm = jnp.where(
        do_cutmix,
        permutation,
        identity_permutation,
    )

    return MixOutput(
        images=mixed_images,
        labels_a=labels_a,
        labels_b=labels_b,
        lam=lam,
        perm=perm,
    )
