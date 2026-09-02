from __future__ import annotations

import jax
import jax.numpy as jnp

from allthemix.methods.output import MixOutput
from allthemix.methods.utils.validation import (
    validate_labels_match_images,
    validate_nhwc_images,
    validate_num_classes,
    validate_positive,
)


def mixup(
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    alpha: float = 1.0,
    paired_images: jnp.ndarray | None = None,
    paired_labels: jnp.ndarray | None = None,
    paired_perm: jnp.ndarray | None = None,
) -> MixOutput:
    """Apply Mixup augmentation to one image batch."""
    validate_num_classes(
        num_classes,
    )
    validate_positive(
        name="mixup_alpha",
        value=alpha,
    )
    validate_nhwc_images(
        images=images,
        method_name="MixUp",
    )
    validate_labels_match_images(
        labels=labels,
        images=images,
        method_name="MixUp",
    )

    if (paired_images is None) != (paired_labels is None):
        raise ValueError(
            "MixUp paired_images and paired_labels must be provided together.",
        )

    if paired_images is not None:
        validate_nhwc_images(
            images=paired_images,
            method_name="MixUp paired",
        )
        validate_labels_match_images(
            labels=paired_labels,
            images=paired_images,
            method_name="MixUp paired",
        )

        if paired_images.shape != images.shape:
            raise ValueError(
                "MixUp paired_images must have the same shape as images. "
                f"Got {paired_images.shape} and {images.shape}.",
            )

        if paired_labels.shape != labels.shape:
            raise ValueError(
                "MixUp paired_labels must have the same shape as labels. "
                f"Got {paired_labels.shape} and {labels.shape}.",
            )

    batch_size = images.shape[0]

    rng_lam, rng_perm = jax.random.split(rng)

    lam = jax.random.beta(  # Sample interpolation strength from Beta(alpha, alpha).
        rng_lam,
        alpha,
        alpha,
        shape=(),
    )

    if paired_images is None:
        permutation = jax.random.permutation(
            rng_perm,
            batch_size,
        )

        shuffled_images = images[permutation]
        shuffled_labels = labels[permutation]

    else:
        del rng_perm
        if paired_perm is None:
            paired_perm = jnp.arange(
                images.shape[0],
                dtype=jnp.int32,
            )
        permutation = paired_perm
        shuffled_images = paired_images
        shuffled_labels = paired_labels

    mixed_images = lam * images + (1.0 - lam) * shuffled_images  # Blend paired images.

    labels_a = labels
    labels_b = shuffled_labels

    return MixOutput(
        images=mixed_images,
        labels_a=labels_a,
        labels_b=labels_b,
        lam=lam,
        perm=permutation,
    )
